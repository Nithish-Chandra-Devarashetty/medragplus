"""Semantic retrieval: query embedding + Qdrant top-k search."""
from __future__ import annotations

import time

from backend.rag.embeddings import Embedder
from backend.rag.qdrant_store import QdrantStore, RetrievedChunk


class Retriever:
    def __init__(self, embedder: Embedder, store: QdrantStore, top_k: int = 4, min_score: float = 0.0):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str, k: int | None = None) -> tuple[list[RetrievedChunk], dict[str, float]]:
        """Return chunks scoring >= min_score, plus embedding/retrieval timings in ms."""
        t0 = time.perf_counter()
        vector = self.embedder.embed_query(query)
        t1 = time.perf_counter()
        chunks = self.store.search(vector, k or self.top_k)
        t2 = time.perf_counter()
        timings = {"embedding": (t1 - t0) * 1000, "retrieval": (t2 - t1) * 1000}
        return [c for c in chunks if c.score >= self.min_score], timings
