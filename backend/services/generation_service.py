"""Prompt construction and BioMistral generation (the 'answer' half of the CRC)."""
from __future__ import annotations

from backend.models.biomistral import LLM, Generation
from backend.rag.qdrant_store import RetrievedChunk

# Kept short on purpose: BioMistral follows brief instructions better, and every
# prompt token costs ~50 ms on a laptop CPU.
SYSTEM_PROMPT = """You are a healthcare triage assistant. Reply to the patient's latest message using only the medical context below, in 2-5 short, plain sentences. If they describe a symptom, explain its common causes, simple self-care and when to see a doctor, as far as the context covers it. Do not diagnose, do not invent facts or doses, and do not only ask questions. If the context does not cover it, say you do not have enough information. If the symptoms could be serious, advise seeing a doctor."""

CONDENSE_PROMPT = """[INST] Rewrite the patient's latest message as one standalone question that includes the relevant details from the earlier messages. Reply with the question only.

Earlier messages:
{history}

Latest message: {query} [/INST]"""

def format_history(history: list[dict]) -> str:
    """Earlier *patient* messages only. With greedy decoding BioMistral copies its
    own previous replies verbatim when they are in the prompt, so they are left
    out; the patient's messages carry the context that matters (onset, duration…)."""
    return "\n".join(f"Patient: {turn['user']}" for turn in history)


def truncate(text: str, max_chars: int | None) -> str:
    """Shorten to max_chars, cutting at a sentence boundary when possible."""
    if not max_chars or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > max_chars // 2 else cut.rsplit(" ", 1)[0] + "…"


def format_context(chunks: list[RetrievedChunk], max_chars: int | None = None) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = ": ".join(p for p in (chunk.metadata.get("source"), chunk.metadata.get("title")) if p)
        parts.append(f"[{i}] ({label}) {truncate(chunk.text, max_chars)}")
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: list[RetrievedChunk], history: list[dict], max_chars: int | None = None) -> str:
    return (
        f"[INST] {SYSTEM_PROMPT}\n\n"
        f"Medical context:\n{format_context(chunks, max_chars)}\n\n"
        f"Earlier patient messages:\n{format_history(history) or 'None'}\n\n"
        f"Patient's latest message: {query} [/INST] Answer:"
    )


class GenerationService:
    def __init__(self, llm: LLM, max_tokens: int = 256, temperature: float = 0.1, n_ctx: int = 4096,
                 context_chars_per_chunk: int | None = None):
        self.llm = llm
        self.context_chars = context_chars_per_chunk
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n_ctx = n_ctx

    def fit_prompt(self, query: str, chunks: list[RetrievedChunk], history: list[dict]) -> tuple[str, list[RetrievedChunk]]:
        """Drop the oldest history, then the lowest-ranked chunks, until the prompt fits the context window."""
        budget = self.n_ctx - self.max_tokens - 16
        chunks, history = list(chunks), list(history)
        prompt = build_prompt(query, chunks, history, self.context_chars)
        while self.llm.count_tokens(prompt) > budget and (history or len(chunks) > 1):
            if history:
                history.pop(0)
            else:
                chunks.pop()
            prompt = build_prompt(query, chunks, history, self.context_chars)
        return prompt, chunks

    def generate(self, query: str, chunks: list[RetrievedChunk], history: list[dict]) -> tuple[Generation, list[RetrievedChunk]]:
        prompt, used_chunks = self.fit_prompt(query, chunks, history)
        return self.llm.generate(prompt, self.max_tokens, self.temperature), used_chunks

    def condense(self, query: str, history: list[dict]) -> str:
        """LLM-based standalone-question rewrite (LangChain CRC 'condense question' step)."""
        if not history:
            return query
        prompt = CONDENSE_PROMPT.format(history="\n".join(f"Patient: {t['user']}" for t in history), query=query)
        text = self.llm.generate(prompt, 64, 0.0).text.strip().splitlines()
        return text[0] if text and text[0] else query
