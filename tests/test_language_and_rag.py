"""Language routing, chunking, ingestion loaders and retrieval."""
from pathlib import Path

from backend.rag.embeddings import HashingEmbedder
from backend.rag.ingestion import chunk_text, ingest_directory, load_documents
from backend.rag.qdrant_store import QdrantStore
from backend.rag.retriever import Retriever
from backend.services.language_service import FakeTranslator, LanguageService, detect_script_language
from backend.services.retrieval_service import RetrievalService


def test_script_detection():
    assert detect_script_language("What are flu symptoms?") == "en"
    assert detect_script_language("फ्लू के लक्षण क्या हैं?") == "hi"
    assert detect_script_language("ఫ్లూ లక్షణాలు ఏమిటి?") == "te"
    assert detect_script_language("123 ?") is None


def test_language_service_routing():
    translator = FakeTranslator({("hi", "en", "नमस्ते।"): "Hello."})
    svc = LanguageService(translator, ["en", "hi", "te"])
    assert svc.detect_language("नमस्ते", hint="te") == "hi"  # script beats hint
    assert svc.detect_language("hello", hint="hi") == "en"
    assert svc.response_language("en", "hi") == "hi"  # explicit preference wins
    assert svc.response_language("te", None) == "te"
    assert svc.to_english("hello", "en") == "hello" and not translator.calls  # English is never translated
    assert svc.to_english("नमस्ते।", "hi") == "Hello."
    assert svc.from_english("Rest well.", "te") == "<te> Rest well."


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is about medicine." for i in range(60))
    chunks = chunk_text(text, chunk_size=300, overlap=80)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].split(". ")[-1].rstrip(".") in chunks[1]  # overlap carries the last sentence


def test_loaders_and_ingestion(tmp_path: Path):
    (tmp_path / "medlineplus").mkdir()
    (tmp_path / "medlineplus" / "mplus_topics.xml").write_text(
        """<?xml version="1.0"?><health-topics>
        <health-topic title="Flu" url="https://medlineplus.gov/flu.html" language="English">
          <also-called>Influenza</also-called>
          <full-summary>&lt;p&gt;Flu symptoms include fever and cough.&lt;/p&gt;</full-summary>
          <group>Infections</group>
        </health-topic>
        <health-topic title="Gripe" url="https://medlineplus.gov/spanish/flu.html" language="Spanish">
          <full-summary>&lt;p&gt;La gripe.&lt;/p&gt;</full-summary>
        </health-topic></health-topics>""", encoding="utf-8")
    (tmp_path / "cdc").mkdir()
    (tmp_path / "cdc" / "cold.txt").write_text("Common cold\n\nColds cause sneezing and a stuffy nose.", encoding="utf-8")

    docs = list(load_documents(tmp_path))
    assert [d.metadata["source"] for d in docs] == ["CDC", "MedlinePlus"]
    flu = docs[1]
    assert flu.metadata["url"] == "https://medlineplus.gov/flu.html" and "Influenza" in flu.text
    assert "<p>" not in flu.text

    embedder = HashingEmbedder(64)
    store = QdrantStore.in_memory("t", 64)
    assert ingest_directory(tmp_path, embedder, store, chunk_size=500, overlap=50) == 2
    assert ingest_directory(tmp_path, embedder, store, chunk_size=500, overlap=50) == 2
    assert store.count() == 2  # re-ingestion is idempotent

    hits, timings = Retriever(embedder, store, top_k=1).retrieve("flu fever cough")
    assert hits[0].metadata["title"] == "Flu" and set(timings) == {"embedding", "retrieval"}


def test_history_aware_retrieval_query():
    svc = RetrievalService(retriever=None, condense_mode="concat")
    history = [{"user": "I have a headache.", "assistant": "..."}]
    assert svc.build_query("It started yesterday.", history) == "I have a headache. It started yesterday."
    assert svc.build_query("What is flu?", []) == "What is flu?"


def test_markdown_front_matter_and_sections(tmp_path: Path):
    (tmp_path / "who").mkdir()
    (tmp_path / "who" / "influenza.md").write_text(
        "---\ntitle: Influenza (seasonal)\nurl: https://www.who.int/flu\n---\n\n"
        "## Overview\n\nSeasonal influenza is an acute respiratory infection.\n\n"
        "## Signs and symptoms\n\nSymptoms include sudden fever, cough and sore throat.\n", encoding="utf-8")
    docs = list(load_documents(tmp_path))
    assert [d.metadata["section"] for d in docs] == ["Overview", "Signs and symptoms"]
    assert {d.metadata["url"] for d in docs} == {"https://www.who.int/flu"}
    assert all(d.metadata["source"] == "WHO" and d.metadata["title"] == "Influenza (seasonal)" for d in docs)
    assert list(load_documents(tmp_path, folders=["medlineplus"])) == []
