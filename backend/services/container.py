"""Builds and wires all services from Settings. Any component can be overridden (tests)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.config.settings import Settings
from backend.db.store import ConversationStore, connect
from backend.models.biomistral import LLM, create_llm
from backend.rag.embeddings import Embedder, create_embedder
from backend.rag.qdrant_store import QdrantStore
from backend.rag.retriever import Retriever
from backend.safety.confidence import ConfidenceScorer
from backend.safety.emergency import EmergencyClassifier
from backend.safety.escalation import EscalationService
from backend.safety.hallucination import create_detector
from backend.safety.triage import RuleTriageClassifier
from backend.services.chat_service import ChatPipeline
from backend.services.generation_service import GenerationService
from backend.services.language_service import LanguageService, Translator, create_translator
from backend.services.retrieval_service import RetrievalService
from backend.services.safety_service import SafetyLayer
from backend.services.speech_service import SpeechService, create_asr, create_tts

log = logging.getLogger(__name__)

_UNSET = object()


@dataclass
class Services:
    settings: Settings
    store: ConversationStore
    embedder: Embedder
    vector_store: QdrantStore
    llm: LLM
    language: LanguageService
    speech: SpeechService
    retrieval: RetrievalService
    generation: GenerationService
    safety: SafetyLayer
    escalation: EscalationService
    pipeline: ChatPipeline

    def components(self) -> dict:
        return {
            "llm": self.llm.name,
            "embeddings": self.embedder.name,
            "vector_store": self.settings.qdrant_url or str(self.settings.qdrant_path or ":memory:"),
            "knowledge_chunks": self.vector_store.count(),
            "translation": self.language.translator.name,
            "asr": getattr(self.speech.asr, "name", None),
            "tts": getattr(self.speech.tts, "name", None),
            "hallucination_check": getattr(self.safety.detector, "name", None),
            "confidence_threshold": self.safety.confidence_threshold,
            "languages": self.language.supported,
        }

    def preload(self) -> None:
        """Load the heavy models up front instead of on the first request."""
        log.info("Preloading models…")
        self.embedder.embed_query("warm up")
        if hasattr(self.llm, "llm"):
            _ = self.llm.llm
        if self.language.translator.name == "nllb":
            self.language.translator.translate(["warm up"], "en", "hi")
        if hasattr(self.safety.detector, "_load"):
            self.safety.detector._load()
        if self.speech.asr is not None and hasattr(self.speech.asr, "model"):
            _ = self.speech.asr.model
        log.info("Models loaded")


def build_services(settings: Settings, *, db=None, embedder: Embedder | None = None,
                   vector_store: QdrantStore | None = None, llm: LLM | None = None,
                   translator: Translator | None = None, asr=_UNSET, tts=_UNSET, detector=None) -> Services:
    store = ConversationStore(db if db is not None else connect(settings))
    embedder = embedder or create_embedder(settings)
    vector_store = vector_store or QdrantStore.from_settings(settings)
    llm = llm or create_llm(settings)
    language = LanguageService(translator or create_translator(settings), settings.supported_languages)
    speech = SpeechService(create_asr(settings) if asr is _UNSET else asr,
                           create_tts(settings) if tts is _UNSET else tts)

    generation = GenerationService(llm, settings.llm_max_tokens, settings.llm_temperature, settings.llm_n_ctx,
                                   settings.llm_context_chars_per_chunk)
    retriever = Retriever(embedder, vector_store, settings.top_k, settings.min_retrieval_score)
    retrieval = RetrievalService(retriever, generation, settings.condense_mode)

    escalation = EscalationService.from_yaml(settings.messages_path)
    safety = SafetyLayer(
        emergency=EmergencyClassifier.from_yaml(settings.red_flags_path, settings.emergency_negation),
        triage=RuleTriageClassifier.from_yaml(settings.triage_rules_path),
        detector=detector or create_detector(settings),
        scorer=ConfidenceScorer(settings.confidence_weight_retrieval, settings.confidence_weight_generation,
                                settings.confidence_weight_grounding, settings.retrieval_score_floor,
                                settings.retrieval_score_ceiling),
        escalation=escalation,
        confidence_threshold=settings.confidence_threshold,
        max_unsupported_ratio_for_rewrite=settings.max_unsupported_ratio_for_rewrite,
    )
    pipeline = ChatPipeline(store, language, speech, retrieval, generation, safety, escalation,
                            settings.history_turns)
    return Services(settings, store, embedder, vector_store, llm, language, speech, retrieval, generation, safety,
                    escalation, pipeline)
