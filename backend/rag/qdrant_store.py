"""Qdrant vector store wrapper (cosine distance, HNSW index)."""
from __future__ import annotations

import atexit
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

_ID_NAMESPACE = uuid.UUID("6f1c5f0e-6c8e-4a43-9a52-3d1b5c0f7e21")


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    def to_source(self, snippet_chars: int = 240) -> dict:
        snippet = self.text if len(self.text) <= snippet_chars else self.text[:snippet_chars].rsplit(" ", 1)[0] + "…"
        return {
            "id": self.id,
            "source": self.metadata.get("source"),
            "title": self.metadata.get("title"),
            "url": self.metadata.get("url"),
            "section": self.metadata.get("section"),
            "page": self.metadata.get("page"),
            "score": round(self.score, 4),
            "snippet": snippet,
        }


def point_id(key: str) -> str:
    """Stable id so re-ingesting the same chunk overwrites instead of duplicating."""
    return str(uuid.uuid5(_ID_NAMESPACE, key))


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int, local: bool = True):
        self.client = client
        self.collection = collection
        self.dim = dim
        self._local = local

    @classmethod
    def from_settings(cls, settings) -> "QdrantStore":
        if settings.qdrant_url:
            return cls(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection, settings.embedding_dim,
                       local=False)
        if settings.qdrant_path:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(settings.qdrant_path))
            # Release the on-disk lock before interpreter teardown; closing it from
            # __del__ during shutdown fails on Windows with a noisy traceback.
            atexit.register(client.close)
        else:
            client = QdrantClient(location=":memory:")
        return cls(client, settings.qdrant_collection, settings.embedding_dim)

    @classmethod
    def in_memory(cls, collection: str = "test", dim: int = 768) -> "QdrantStore":
        return cls(QdrantClient(location=":memory:"), collection, dim)

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.exists():
            self.client.delete_collection(self.collection)
        if not self.exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
            )
            if not self._local:  # payload indexes only exist on a Qdrant server
                self.client.create_payload_index(self.collection, "source", qm.PayloadSchemaType.KEYWORD)

    def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], payloads: Sequence[dict]) -> None:
        points = [qm.PointStruct(id=i, vector=list(v), payload=p) for i, v, p in zip(ids, vectors, payloads)]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: Sequence[float], k: int, source: str | None = None) -> list[RetrievedChunk]:
        if not self.exists():
            return []
        query_filter = None
        if source:
            query_filter = qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))])
        result = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            limit=k,
            with_payload=True,
            query_filter=query_filter,
        )
        chunks = []
        for point in result.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            chunks.append(RetrievedChunk(id=str(point.id), text=text, score=float(point.score), metadata=payload))
        return chunks

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, exact=True).count
