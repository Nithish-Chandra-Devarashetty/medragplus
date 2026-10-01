"""End-to-end MedRAG+ query pipeline.

voice? -> ASR -> language detection -> translate to English -> emergency pre-check
-> history -> PubMedBERT + Qdrant retrieval -> BioMistral draft -> SafetyLayer
-> deliver / rewrite / escalate -> translate back -> TTS -> persist
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from backend.db.store import ConversationStore, iso
from backend.services.generation_service import GenerationService
from backend.services.language_service import LanguageService
from backend.services.retrieval_service import RetrievalService
from backend.services.safety_service import ESCALATE, SafetyLayer, SafetyResult
from backend.services.speech_service import SpeechService
from backend.safety.escalation import EscalationService
from backend.utils.timing import StageTimer

log = logging.getLogger(__name__)


class SessionNotFound(Exception):
    pass


@dataclass
class ChatRequest:
    user_id: str
    message: str | None = None
    audio: bytes | None = None
    session_id: str | None = None
    language: str | None = "auto"  # auto | en | hi | te
    tts: bool = False


class ChatPipeline:
    def __init__(self, store: ConversationStore, language: LanguageService, speech: SpeechService,
                 retrieval: RetrievalService, generation: GenerationService, safety: SafetyLayer,
                 escalation: EscalationService, history_turns: int = 3):
        self.store = store
        self.language = language
        self.speech = speech
        self.retrieval = retrieval
        self.generation = generation
        self.safety = safety
        self.escalation = escalation
        self.history_turns = history_turns

    def run(self, req: ChatRequest) -> dict:
        timer = StageTimer()
        preference = req.language if req.language in self.language.supported else None

        if req.session_id and not self.store.get_session(req.session_id, req.user_id):
            raise SessionNotFound(req.session_id)

        # 1. speech -> text
        transcript = None
        text = (req.message or "").strip()
        asr_language = None
        if req.audio is not None:
            # Whisper's own language ID often mistakes Telugu speech for Hindi, so on
            # "auto" a non-English profile language is a better guess than detection.
            hint = preference
            if hint is None:
                profile = (self.store.get_user(req.user_id) or {}).get("language")
                hint = profile if profile in self.language.supported and profile != "en" else None
            with timer.stage("asr"):
                transcript, asr_language = self.speech.transcribe(req.audio, hint)
            text = transcript
        if not text:
            raise ValueError("Empty message" if req.audio is None else "No speech was recognised in the recording")

        # 2. language detection + translation to English
        detected = self.language.detect_language(text, hint=asr_language or preference)
        response_language = self.language.response_language(detected, preference)
        with timer.stage("translate_in"):
            query_en = self.language.to_english(text, detected)

        # 3. emergency red flags - checked before anything slow happens
        with timer.stage("safety"):
            emergency = self.safety.precheck(query_en, text)

        session_id = req.session_id or self.store.create_session(req.user_id, text, response_language)
        history = self.store.recent_turns(session_id, self.history_turns)

        chunks, retrieval_query, draft = [], None, None
        if emergency.emergency:
            result = self.safety.escalate_emergency(emergency)
        else:
            # 4. retrieval
            chunks, retrieval_timings, retrieval_query = self.retrieval.retrieve(query_en, history)
            for name, ms in retrieval_timings.items():
                timer.add(name, ms)

            if not chunks:
                result = self.safety.escalate_no_context(self.safety.classify_triage(query_en, text))
            else:
                # 5. generation
                try:
                    with timer.stage("generation"):
                        generation, chunks = self.generation.generate(query_en, chunks, history)
                    draft = generation.text
                except Exception:
                    log.exception("Generation failed")
                    generation = None

                # 6. safety layer on the draft
                with timer.stage("safety"):
                    if generation is None:
                        result = self.safety.escalate_no_context(self.safety.classify_triage(query_en, text))
                        result.flags.append("generation_error")
                        result.reason = "low_confidence"
                        result.answer_en = self.escalation.message("low_confidence", "en")[0]
                    else:
                        result = self.safety.evaluate(query_en, text, draft, chunks, generation.mean_token_prob,
                                                      emergency)

        # 7. back to the user's language (escalations use fixed, pre-translated text)
        with timer.stage("translate_out"):
            answer = self._localise(result, response_language)

        # 8. text -> speech
        audio = None
        if req.tts and self.speech.tts is not None:
            try:
                with timer.stage("tts"):
                    audio = {"mime": "audio/mpeg",
                             "base64": base64.b64encode(self.speech.synthesize(answer, response_language)).decode()}
            except Exception:
                log.exception("TTS failed")

        timings = timer.finish()
        sources = [c.to_source() for c in chunks]
        message = {
            "session_id": session_id,
            "user_id": req.user_id,
            "input_type": "voice" if req.audio is not None else "text",
            "user_query": text,
            "transcript": transcript,
            "detected_language": detected,
            "response_language": response_language,
            "query_english": query_en,
            "retrieval_query": retrieval_query,
            "retrieved_chunks": sources,
            "draft_answer": draft,
            "answer": answer,
            "answer_english": result.answer_en,
            "safety": result.to_dict(),
            "timings_ms": timings,
        }
        message = self.store.add_message(message)
        message_id = message["_id"]
        if result.flags:
            self.store.add_safety_flag({"message_id": message_id, "session_id": session_id, "user_id": req.user_id,
                                        "flags": result.flags, "action": result.action,
                                        "confidence": result.confidence, "triage_level": result.triage_level,
                                        "emergency_rules": result.emergency_rules,
                                        "unsupported_claims": result.unsupported_claims})
        response = self.to_response(message)
        response["audio"] = audio
        return response

    def _localise(self, result: SafetyResult, language: str) -> str:
        if result.action == ESCALATE and result.reason:
            text, localised = self.escalation.message(result.reason, language)
            if localised:
                return text
        return self.language.from_english(result.answer_en, language)

    def to_response(self, message: dict) -> dict:
        safety = message["safety"]
        language = message["response_language"]
        return {
            "session_id": message["session_id"],
            "message_id": message["_id"],
            "query": {
                "original": message["user_query"],
                "transcript": message.get("transcript"),
                "detected_language": message["detected_language"],
                "english": message["query_english"],
            },
            "retrieval_query": message.get("retrieval_query"),
            "language": language,
            "answer": message["answer"],
            "answer_english": message["answer_english"],
            "triage_level": safety["triage_level"],
            "action": safety["action"],
            "escalated": safety["escalated"],
            "emergency": safety["emergency"],
            "confidence": safety["confidence"],
            "confidence_components": safety.get("confidence_components", {}),
            "hallucination": safety["hallucination"],
            "safety_flags": safety["flags"],
            "unsupported_claims": safety["unsupported_claims"],
            "emergency_rules": safety.get("emergency_rules", []),
            "sources": message["retrieved_chunks"],
            "audio": None,
            "timings_ms": message["timings_ms"],
            "disclaimer": self.escalation.disclaimer(language),
            "created_at": iso(message.get("created_at")),
        }
