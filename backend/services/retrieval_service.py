"""History-aware retrieval (the 'retrieval' half of the Conversational Retrieval Chain).

Follow-up messages such as "It started yesterday" carry no retrievable content
on their own, so the retrieval query is built from recent user turns as well.
`concat` mode does this without an extra LLM call; `llm` mode asks BioMistral to
rewrite the follow-up as a standalone question (LangChain's condense step),
which is more accurate on topic shifts but adds a full LLM round-trip.
"""
from __future__ import annotations

from backend.rag.qdrant_store import RetrievedChunk
from backend.rag.retriever import Retriever
from backend.services.generation_service import GenerationService


class RetrievalService:
    def __init__(self, retriever: Retriever, generation: GenerationService | None = None,
                 condense_mode: str = "concat", concat_turns: int = 2):
        self.retriever = retriever
        self.generation = generation
        self.condense_mode = condense_mode
        self.concat_turns = concat_turns

    def build_query(self, query: str, history: list[dict]) -> str:
        if not history:
            return query
        if self.condense_mode == "llm" and self.generation is not None:
            return self.generation.condense(query, history)
        previous = [turn["user"] for turn in history[-self.concat_turns:]]
        return " ".join(previous + [query])

    def retrieve(self, query: str, history: list[dict]) -> tuple[list[RetrievedChunk], dict[str, float], str]:
        retrieval_query = self.build_query(query, history)
        chunks, timings = self.retriever.retrieve(retrieval_query)
        if retrieval_query != query:
            # Also search with the latest message alone and keep the best-scoring
            # chunks of both: a follow-up ("it started yesterday") needs the earlier
            # turns, but a new symptom ("I also have a fever") must not be drowned
            # out by the previous topic.
            alone, alone_timings = self.retriever.retrieve(query)
            for name, ms in alone_timings.items():
                timings[name] = timings.get(name, 0.0) + ms
            best: dict[str, RetrievedChunk] = {}
            for chunk in chunks + alone:
                if chunk.id not in best or chunk.score > best[chunk.id].score:
                    best[chunk.id] = chunk
            chunks = sorted(best.values(), key=lambda c: c.score, reverse=True)[:self.retriever.top_k]
        return chunks, timings, retrieval_query
