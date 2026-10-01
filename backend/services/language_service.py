"""Language detection and NLLB translation (Hindi/Telugu <-> English).

The RAG pipeline and BioMistral operate in English, so non-English queries are
translated in, and answers are translated back out.
"""
from __future__ import annotations

import threading
from typing import Protocol

from backend.utils.text import split_sentences

NLLB_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu"}

_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "te": (0x0C00, 0x0C7F),  # Telugu
}


def detect_script_language(text: str) -> str | None:
    """Detect en/hi/te from the Unicode script (deterministic, no model needed).

    Romanised Hindi/Telugu ("mujhe bukhar hai") is Latin script and is reported
    as "en"; callers can combine this with the user's selected language.
    """
    counts = {"hi": 0, "te": 0, "en": 0}
    for ch in text:
        code = ord(ch)
        for lang, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= code <= hi:
                counts[lang] += 1
                break
        else:
            if ch.isascii() and ch.isalpha():
                counts["en"] += 1
    lang, count = max(counts.items(), key=lambda kv: kv[1])
    return lang if count else None


class Translator(Protocol):
    name: str

    def translate(self, texts: list[str], source: str, target: str) -> list[str]: ...


class NLLBTranslator:
    name = "nllb"

    def __init__(self, model_name: str, num_beams: int = 2, max_new_tokens: int = 256):
        self.model_name = model_name
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).eval()
            self._model.generation_config.max_length = None  # we pass max_new_tokens instead

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        import torch

        self._load()
        with self._lock:
            self._tokenizer.src_lang = NLLB_CODES[source]
            inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=400)
            with torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(NLLB_CODES[target]),
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                )
            return self._tokenizer.batch_decode(output, skip_special_tokens=True)


class IdentityTranslator:
    """No-op translator (English-only deployments)."""

    name = "identity"

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        return list(texts)


class FakeTranslator:
    """Phrase-book translator for tests: exact matches are translated, the rest is tagged."""

    name = "fake"

    def __init__(self, phrasebook: dict[tuple[str, str, str], str] | None = None):
        self.phrasebook = phrasebook or {}
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, texts: list[str], source: str, target: str) -> list[str]:
        out = []
        for text in texts:
            self.calls.append((source, target, text))
            out.append(self.phrasebook.get((source, target, text), text if target == "en" else f"<{target}> {text}"))
        return out


def create_translator(settings) -> Translator:
    if settings.translation_backend == "nllb":
        return NLLBTranslator(settings.nllb_model)
    if settings.translation_backend == "identity":
        return IdentityTranslator()
    if settings.translation_backend == "fake":
        return FakeTranslator()
    raise ValueError(f"Unknown TRANSLATION_BACKEND: {settings.translation_backend}")


class LanguageService:
    def __init__(self, translator: Translator, supported: list[str]):
        self.translator = translator
        self.supported = supported

    def detect_language(self, text: str, hint: str | None = None) -> str:
        script = detect_script_language(text)
        if script in self.supported and script != "en":
            return script  # Devanagari/Telugu script is unambiguous
        if script == "en":
            return "en"
        return hint if hint in self.supported else "en"

    def response_language(self, detected: str, preference: str | None) -> str:
        """Answer in the user's explicitly selected language, else in the detected one."""
        return preference if preference in self.supported else detected

    def to_english(self, text: str, source_language: str) -> str:
        if source_language == "en":
            return text
        return self._translate(text, source_language, "en")

    def from_english(self, text: str, target_language: str) -> str:
        if target_language == "en":
            return text
        return self._translate(text, "en", target_language)

    def _translate(self, text: str, source: str, target: str) -> str:
        # NLLB is sentence-level: translate line by line, sentence by sentence.
        lines_out = []
        for line in text.splitlines():
            if not line.strip():
                lines_out.append("")
                continue
            bullet = "- " if line.lstrip().startswith(("-", "*", "•")) else ""
            sentences = split_sentences(line)
            lines_out.append(bullet + " ".join(self.translator.translate(sentences, source, target)))
        return "\n".join(lines_out).strip()
