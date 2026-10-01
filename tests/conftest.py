"""Shared fixtures: the full app wired with lightweight fake model backends.

Real code paths are exercised everywhere except the neural models: Qdrant runs
in-memory, MongoDB is mongomock, embeddings are feature-hashed, BioMistral is
an extractive fake, NLLB is a phrase-book, Whisper/TTS are stubs.
"""
from __future__ import annotations

import mongomock
import pytest

from backend.app import create_app
from backend.config.settings import Settings
from backend.models.biomistral import FakeLLM
from backend.rag.embeddings import HashingEmbedder
from backend.rag.ingestion import Document, chunk_documents, index_chunks
from backend.rag.qdrant_store import QdrantStore
from backend.services.container import build_services
from backend.services.language_service import FakeTranslator
from backend.services.speech_service import FakeASR, FakeTTS

CORPUS = [
    ("Influenza (Flu)", "Flu is a contagious respiratory illness caused by influenza viruses. Common symptoms of the flu "
     "include fever, cough, sore throat, runny nose, body aches, headache, chills and fatigue. Most people recover "
     "from the flu in less than two weeks with rest and fluids."),
    ("Type 2 Diabetes", "Treatment options for type 2 diabetes include lifestyle changes, oral medications such as "
     "metformin, and insulin therapy. Healthy eating and regular physical activity help control blood sugar."),
    ("Headache", "Tension headaches are the most common type of headache. A headache that started yesterday is often "
     "caused by stress, poor sleep, dehydration or eye strain. Rest, fluids and over-the-counter pain relievers "
     "usually help a tension headache."),
    ("Heart Health", "Lifestyle changes that improve cardiovascular health include regular exercise, a healthy diet, "
     "weight loss, and quitting smoking."),
    ("Common Cold", "The common cold is a viral infection of the nose and throat. Symptoms include sneezing, a stuffy "
     "nose and a mild sore throat."),
]

PHRASEBOOK = {
    ("hi", "en", "फ्लू के सामान्य लक्षण क्या हैं?"): "What are the common symptoms of the flu?",
    ("te", "en", "ఫ్లూ యొక్క సాధారణ లక్షణాలు ఏమిటి?"): "What are the common symptoms of the flu?",
    ("hi", "en", "मुझे सीने में दर्द है"): "I have chest pain",
}


def make_settings(**overrides) -> Settings:
    base = Settings().with_overrides(
        secret_key="test-secret-key-that-is-long-enough-for-hs256",
        mongo_uri="mongomock://",
        qdrant_url=None,
        qdrant_path=None,
        embedding_backend="fake",
        llm_backend="fake",
        translation_backend="fake",
        asr_backend="fake",
        tts_backend="fake",
        hallucination_backend="lexical",
        min_retrieval_score=0.05,
        retrieval_score_floor=0.0,
        retrieval_score_ceiling=0.35,
        confidence_threshold=0.6,
    )
    return base.with_overrides(**overrides)


def make_vector_store(embedder) -> QdrantStore:
    store = QdrantStore.in_memory("test", embedder.dim)
    store.ensure_collection()
    docs = [Document(text, {"source": "TestRef", "title": title, "url": f"https://example.org/{i}",
                            "document": "corpus", "section": None, "page": None})
            for i, (title, text) in enumerate(CORPUS)]
    index_chunks(chunk_documents(docs, 1000, 100), embedder, store)
    return store


@pytest.fixture
def settings():
    return make_settings()


@pytest.fixture
def services(settings):
    embedder = HashingEmbedder(settings.embedding_dim)
    return build_services(
        settings,
        db=mongomock.MongoClient()["test"],
        embedder=embedder,
        vector_store=make_vector_store(embedder),
        llm=FakeLLM(),
        translator=FakeTranslator(PHRASEBOOK),
        asr=FakeASR(),
        tts=FakeTTS(),
    )


@pytest.fixture
def app(settings, services):
    return create_app(settings, services)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def auth_headers(client):
    resp = client.post("/api/register", json={"email": "patient@example.com", "password": "password123",
                                              "name": "Test", "language": "en"})
    assert resp.status_code == 201, resp.get_json()
    return {"Authorization": f"Bearer {resp.get_json()['token']}"}
