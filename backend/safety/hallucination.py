"""Hallucination check: is every claim in the draft answer supported by the retrieved context?

The answer is split into sentences. Advice/meta sentences ("please consult a
doctor") are not factual claims and are skipped. Every remaining claim is
scored against each retrieved chunk; a claim is supported if its best score
reaches the threshold.

Backends:
* `nli`     - an NLI cross-encoder; score = P(entailment | premise=chunk, hypothesis=claim)
* `lexical` - fraction of the claim's content words present in the chunk
              (no model; used in tests and as a fallback)
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

from backend.rag.qdrant_store import RetrievedChunk
from backend.utils.text import content_words, split_sentences

NON_CLAIM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(consult|see|visit|contact|talk to|speak (to|with)|check with)\b.{0,40}\b(doctor|physician|health ?care|provider|professional|gp|specialist|pharmacist)\b",
        r"\bseek (immediate |urgent |prompt |emergency )?(medical|emergency|professional) (attention|care|help|advice)\b",
        r"\b(go to|visit) (the )?(nearest )?(emergency|hospital|er\b|a&e)",
        r"\bcall (108|112|911|an ambulance|emergency services)\b",
        r"\b(i|we) (do not|don't|cannot|can't) have enough information\b",
        r"^(i'?m sorry|sorry|i hope)\b",
        r"\bthis (is|does) not (a )?(replace|substitute|diagnosis)",
    ]
]


def is_claim(sentence: str) -> bool:
    if len(content_words(sentence)) < 2 or sentence.rstrip().endswith("?"):  # questions assert nothing
        return False
    return not any(p.search(sentence) for p in NON_CLAIM_PATTERNS)


@dataclass
class SentenceSupport:
    sentence: str
    is_claim: bool
    score: float
    supported: bool
    evidence_id: str | None = None


@dataclass
class HallucinationResult:
    hallucination: bool
    support_ratio: float  # supported claims / claims (1.0 when there are no claims)
    unsupported_claims: list[str] = field(default_factory=list)
    sentences: list[SentenceSupport] = field(default_factory=list)

    def supported_text(self) -> str:
        return " ".join(s.sentence for s in self.sentences if s.supported)

    @property
    def claim_count(self) -> int:
        return sum(1 for s in self.sentences if s.is_claim)

    @property
    def supported_claim_count(self) -> int:
        return sum(1 for s in self.sentences if s.is_claim and s.supported)


class _BaseDetector:
    threshold: float

    def _scores(self, claims: list[str], chunks: list[RetrievedChunk]) -> list[tuple[float, str | None]]:
        raise NotImplementedError

    def check(self, answer: str, chunks: list[RetrievedChunk], query: str | None = None) -> HallucinationResult:
        """`query` (the patient's own message) also counts as evidence, so restating
        the patient's symptoms ("It sounds like your knee pain keeps returning") is
        not flagged. It cannot support medical facts the patient did not state."""
        sentences = split_sentences(answer)
        claims = [s for s in sentences if is_claim(s)]
        evidence = list(chunks)
        if chunks and query:
            evidence.append(RetrievedChunk("patient_message", f"The patient says: {query}", 1.0, {}))
        scored = dict(zip(claims, self._scores(claims, evidence))) if claims and chunks else {}
        results = []
        for sentence in sentences:
            if sentence not in claims:
                results.append(SentenceSupport(sentence, False, 1.0, True))
                continue
            score, evidence = scored.get(sentence, (0.0, None))
            results.append(SentenceSupport(sentence, True, round(score, 4), score >= self.threshold, evidence))
        unsupported = [s.sentence for s in results if not s.supported]
        ratio = (len(claims) - len(unsupported)) / len(claims) if claims else 1.0
        return HallucinationResult(bool(unsupported), round(ratio, 4), unsupported, results)


class LexicalSupportDetector(_BaseDetector):
    name = "lexical"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def _scores(self, claims, chunks):
        chunk_words = [(c.id, set(content_words(c.text))) for c in chunks]
        out = []
        for claim in claims:
            words = set(content_words(claim))
            best = max(((len(words & cw) / len(words), cid) for cid, cw in chunk_words), default=(0.0, None))
            out.append(best)
        return out


class NLISupportDetector(_BaseDetector):
    name = "nli"

    def __init__(self, model_name: str, threshold: float = 0.5, window: int = 2, batch_size: int = 32):
        self.model_name = model_name
        self.threshold = threshold
        self.window = window
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._entail_index = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
            labels = {v.lower(): int(k) for k, v in self._model.config.id2label.items()}
            self._entail_index = labels.get("entailment", labels.get("entail"))
            if self._entail_index is None:
                raise ValueError(f"{self.model_name} has no 'entailment' label: {labels}")

    def premises(self, chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
        """Short sliding windows of sentences per chunk (SummaC-style).

        Small NLI models lose the evidence when the premise is a long passage, so
        each claim is scored against every `window`-sentence span and the best
        span wins.
        """
        out = []
        for chunk in chunks:
            sentences = split_sentences(chunk.text)
            if len(sentences) <= self.window:
                out.append((chunk.id, chunk.text))
                continue
            for start in range(0, len(sentences) - self.window + 1):
                out.append((chunk.id, " ".join(sentences[start:start + self.window])))
        return out

    def _scores(self, claims, chunks):
        import torch

        self._load()
        premises = self.premises(chunks)
        pairs = [(premise, claim) for claim in claims for _, premise in premises]
        probs: list[float] = []
        with self._lock, torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                inputs = self._tokenizer(
                    [p for p, _ in batch], [h for _, h in batch],
                    return_tensors="pt", padding=True, truncation="only_first", max_length=512,
                )
                probs += torch.softmax(self._model(**inputs).logits, dim=-1)[:, self._entail_index].tolist()
        out = []
        for i in range(len(claims)):
            row = probs[i * len(premises):(i + 1) * len(premises)]
            j = max(range(len(row)), key=row.__getitem__)
            out.append((row[j], premises[j][0]))
        return out


def create_detector(settings):
    if settings.hallucination_backend == "nli":
        return NLISupportDetector(settings.nli_model, settings.entailment_threshold, settings.nli_window_sentences)
    if settings.hallucination_backend == "lexical":
        return LexicalSupportDetector(settings.lexical_support_threshold)
    raise ValueError(f"Unknown HALLUCINATION_BACKEND: {settings.hallucination_backend}")
