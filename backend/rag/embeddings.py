"""Text embedding backends.

`PubMedBertEmbedder` is the production backend (768-d PubMedBERT sentence
embeddings, as in the base paper). `HashingEmbedder` is a deterministic,
dependency-free stand-in used by tests and for smoke-testing without models.
"""
from __future__ import annotations

import math
import zlib
from typing import Protocol, Sequence

from backend.utils.text import content_words


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class PubMedBertEmbedder:
    name = "pubmedbert"

    def __init__(self, model_name: str, dim: int = 768, batch_size: int = 32):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
            get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            actual = get_dim()
            if actual != self.dim:
                raise ValueError(f"{self.model_name} produces {actual}-d vectors but EMBEDDING_DIM={self.dim}")
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingEmbedder:
    """Bag-of-words feature hashing (unigrams + bigrams), L2-normalised."""

    name = "fake"

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        words = content_words(text)
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for feature in features:
            h = zlib.crc32(feature.encode("utf-8"))
            vector[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def create_embedder(settings) -> Embedder:
    if settings.embedding_backend == "pubmedbert":
        return PubMedBertEmbedder(settings.embedding_model, settings.embedding_dim, settings.embedding_batch_size)
    if settings.embedding_backend == "fake":
        return HashingEmbedder(settings.embedding_dim)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {settings.embedding_backend}")
