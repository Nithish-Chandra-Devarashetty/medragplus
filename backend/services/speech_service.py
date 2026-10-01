"""Speech in (Whisper ASR) and speech out (TTS)."""
from __future__ import annotations

import io
import threading
from typing import Protocol


class SpeechUnavailable(RuntimeError):
    pass


class UnclearSpeech(ValueError):
    """Low-confidence transcript; surfaced to the user as a 400 asking them to retry."""


# Whisper's initial_prompt biases decoding toward these words, e.g. "fever" rather
# than "food" for accented speech. Written as a patient would speak.
VOCABULARY_PROMPTS = {
    "en": "I have fever, cough, cold, headache, stomach pain, vomiting, diarrhoea, chest pain, breathing problem.",
    "hi": "मुझे बुखार, खांसी, सर्दी, सिरदर्द, पेट दर्द, उल्टी, दस्त, सीने में दर्द है।",
    "te": "నాకు జ్వరం, దగ్గు, జలుబు, తలనొప్పి, కడుపు నొప్పి, వాంతులు, విరేచనాలు, ఛాతీ నొప్పి ఉంది.",
}


class ASR(Protocol):
    name: str

    def transcribe(self, audio: bytes, language_hint: str | None) -> tuple[str, str | None]: ...


class TTS(Protocol):
    name: str

    def synthesize(self, text: str, language: str) -> bytes: ...


class FasterWhisperASR:
    """Multilingual Whisper via CTranslate2 (much faster than openai-whisper on CPU)."""

    name = "faster_whisper"

    def __init__(self, model_size: str = "large-v3-turbo", compute_type: str = "int8",
                 supported: list[str] | None = None, min_avg_logprob: float = -1.0,
                 min_language_probability: float = 0.5):
        self.model_size = model_size
        self.compute_type = compute_type
        self.supported = supported or ["en", "hi", "te"]
        self.min_avg_logprob = min_avg_logprob
        self.min_language_probability = min_language_probability
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_size, device="cpu", compute_type=self.compute_type)
        return self._model

    def detect_language(self, samples) -> tuple[str, float]:
        """Most likely *supported* language. Open detection over Whisper's ~100
        languages turns Indian-accented English into e.g. Icelandic."""
        _, _, probs = self.model.detect_language(samples, vad_filter=True)
        return max(((lang, p) for lang, p in probs if lang in self.supported), key=lambda x: x[1])

    def transcribe(self, audio: bytes, language_hint: str | None) -> tuple[str, str | None]:
        from faster_whisper.audio import decode_audio

        samples = decode_audio(io.BytesIO(audio))
        with self._lock:
            if language_hint in self.supported:
                language = language_hint
            else:
                language, probability = self.detect_language(samples)
                if probability < self.min_language_probability:
                    raise UnclearSpeech("I could not tell which language you spoke. Please choose your language "
                                        "in the selector and try again, or type your question.")
            segments, _ = self.model.transcribe(
                samples, language=language, beam_size=1, vad_filter=True,
                condition_on_previous_text=False,  # avoids repetition loops
                initial_prompt=VOCABULARY_PROMPTS.get(language),
            )
            segments = list(segments)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        # A garbled transcript must not reach the medical pipeline as if the user had said it.
        if segments and sum(s.avg_logprob for s in segments) / len(segments) < self.min_avg_logprob:
            raise UnclearSpeech("The recording could not be understood clearly. Please try again or type your question.")
        return text, language


class GTTSEngine:
    """Google Translate TTS (en/hi/te). Needs internet access."""

    name = "gtts"

    def synthesize(self, text: str, language: str) -> bytes:
        from gtts import gTTS

        buffer = io.BytesIO()
        gTTS(text=text, lang=language).write_to_fp(buffer)
        return buffer.getvalue()


class FakeASR:
    name = "fake"

    def __init__(self, transcript: str = "What are the common symptoms of flu?", language: str | None = "en"):
        self.transcript = transcript
        self.language = language

    def transcribe(self, audio: bytes, language_hint: str | None) -> tuple[str, str | None]:
        if not audio:
            raise ValueError("empty audio")
        return self.transcript, self.language or language_hint


class FakeTTS:
    name = "fake"

    def synthesize(self, text: str, language: str) -> bytes:
        return b"ID3" + f"{language}:{text}".encode("utf-8")


def create_asr(settings) -> ASR | None:
    return {
        "faster_whisper": lambda: FasterWhisperASR(settings.whisper_model, settings.whisper_compute_type,
                                                   settings.supported_languages, settings.asr_min_avg_logprob),
        "fake": FakeASR,
        "none": lambda: None,
    }[settings.asr_backend]()


def create_tts(settings) -> TTS | None:
    return {"gtts": GTTSEngine, "fake": FakeTTS, "none": lambda: None}[settings.tts_backend]()


class SpeechService:
    def __init__(self, asr: ASR | None, tts: TTS | None):
        self.asr = asr
        self.tts = tts

    def transcribe(self, audio: bytes, language_hint: str | None = None) -> tuple[str, str | None]:
        if self.asr is None:
            raise SpeechUnavailable("Speech recognition is not configured (ASR_BACKEND=none)")
        return self.asr.transcribe(audio, language_hint)

    def synthesize(self, text: str, language: str) -> bytes:
        if self.tts is None:
            raise SpeechUnavailable("Text-to-speech is not configured (TTS_BACKEND=none)")
        return self.tts.synthesize(text, language)
