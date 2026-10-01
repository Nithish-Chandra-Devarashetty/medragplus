"""Offline knowledge-base pipeline: load -> clean -> chunk -> embed -> Qdrant.

Supported inputs under DOCUMENTS_DIR (the first sub-folder names the source):

    data/medical_documents/
        medlineplus/   MedlinePlus health-topic XML (mplus_topics_*.xml)
        who/           PDFs
        cdc/           .html / .txt / .md
        statpearls/    JATS .nxml articles
        <anything>/    .pdf .txt .md .html .nxml
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from backend.rag.embeddings import Embedder
from backend.rag.qdrant_store import QdrantStore, point_id
from backend.utils.text import normalize_whitespace, split_sentences, strip_html

log = logging.getLogger(__name__)

SOURCE_NAMES = {
    "medlineplus": "MedlinePlus",
    "who": "WHO",
    "cdc": "CDC",
    "statpearls": "StatPearls",
    "nhs": "NHS",
    "merck": "Merck Manual",
    "harrison": "Harrison's",
    "oxford": "Oxford Handbook",
    "gale": "Gale Encyclopedia",
}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict

    def embedding_text(self) -> str:
        heading = " - ".join(p for p in (self.metadata.get("title"), self.metadata.get("section")) if p)
        return f"{heading}\n{self.text}" if heading else self.text

    def payload(self) -> dict:
        return {"text": self.text, **self.metadata}


# --------------------------------------------------------------------------- loaders


def source_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    folder = rel.parts[0] if len(rel.parts) > 1 else path.stem
    return SOURCE_NAMES.get(folder.lower(), folder)


def load_medlineplus_xml(path: Path, source: str) -> Iterator[Document]:
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "health-topic":
            continue
        if elem.get("language", "English") == "English":
            summary = strip_html(elem.findtext("full-summary") or "")
            if summary:
                also_called = [e.text for e in elem.findall("also-called") if e.text]
                title = elem.get("title", "")
                header = f"{title} (also called: {', '.join(also_called)})" if also_called else title
                yield Document(
                    text=f"{header}\n\n{summary}",
                    metadata={
                        "source": source,
                        "title": title,
                        "url": elem.get("url"),
                        "document": path.name,
                        "section": ", ".join(g.text for g in elem.findall("group") if g.text) or None,
                        "page": None,
                    },
                )
        elem.clear()


def load_pdf(path: Path, source: str) -> Iterator[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = (reader.metadata.title if reader.metadata else None) or path.stem.replace("_", " ")
    for number, page in enumerate(reader.pages, start=1):
        text = normalize_whitespace(page.extract_text() or "")
        if len(text) > 50:
            yield Document(
                text=text,
                metadata={"source": source, "title": title, "url": None, "document": path.name,
                          "section": None, "page": number},
            )


_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_MD_HEADING = re.compile(r"^#{1,3} +(.+)$", re.M)


def load_text(path: Path, source: str) -> Iterator[Document]:
    """Plain text / HTML / Markdown. Markdown may carry `title:` / `url:` front matter
    and is split into one document per `##` section."""
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    meta = {}
    match = _FRONT_MATTER.match(raw)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
        raw = raw[match.end():]
    text = strip_html(raw) if path.suffix.lower() in {".html", ".htm"} else normalize_whitespace(raw)
    if not text:
        return
    title = meta.get("title") or text.splitlines()[0].lstrip("# ")[:120] or path.stem
    base = {"source": source, "title": title, "url": meta.get("url"), "document": path.name, "page": None}

    sections: list[tuple[str | None, str]] = []
    if path.suffix.lower() == ".md" and _MD_HEADING.search(text):
        positions = [(m.start(), m.end(), m.group(1).strip()) for m in _MD_HEADING.finditer(text)]
        if positions[0][0] > 0:
            sections.append((None, text[:positions[0][0]]))
        for i, (_, end, heading) in enumerate(positions):
            stop = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            sections.append((heading, text[end:stop]))
    else:
        sections.append((None, text))

    for heading, body in sections:
        body = body.strip()
        if len(body) > 30:
            yield Document(text=body, metadata={**base, "section": heading})


def load_jats(path: Path, source: str) -> Iterator[Document]:
    """Minimal JATS/NXML reader (StatPearls bulk download): one document per section."""
    root = ET.parse(path).getroot()
    title = " ".join("".join(root.find(".//article-title").itertext()).split()) if root.find(".//article-title") is not None else path.stem
    body = root.find(".//body")
    if body is None:
        return
    for sec in body.iter("sec"):
        sec_title_el = sec.find("title")
        sec_title = " ".join("".join(sec_title_el.itertext()).split()) if sec_title_el is not None else None
        paragraphs = [" ".join("".join(p.itertext()).split()) for p in sec.findall("p")]
        text = "\n\n".join(p for p in paragraphs if p)
        if text:
            yield Document(
                text=text,
                metadata={"source": source, "title": title, "url": None, "document": path.name,
                          "section": sec_title, "page": None},
            )


LOADERS: dict[str, Callable[[Path, str], Iterator[Document]]] = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
    ".html": load_text,
    ".htm": load_text,
    ".nxml": load_jats,
}


def load_documents(root: Path, folders: list[str] | None = None) -> Iterator[Document]:
    """Load every supported file under root (optionally only the given top-level folders)."""
    wanted = {f.lower() for f in folders or []}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if wanted and path.relative_to(root).parts[0].lower() not in wanted:
            continue
        source = source_name(path, root)
        suffix = path.suffix.lower()
        try:
            if suffix == ".xml":
                yield from load_medlineplus_xml(path, source)
            elif suffix in LOADERS:
                yield from LOADERS[suffix](path, source)
        except Exception as exc:  # one bad file must not abort a long ingestion
            log.warning("Skipping %s: %s", path, exc)


# --------------------------------------------------------------------------- chunking


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Greedy sentence packing with a sentence-aligned overlap between chunks."""
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        for sentence in split_sentences(paragraph):
            while len(sentence) > chunk_size:  # pathological long "sentences" (tables)
                sentences.append(sentence[:chunk_size])
                sentence = sentence[chunk_size:]
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if current and length + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(documents: Iterable[Document], chunk_size: int, overlap: int) -> Iterator[Chunk]:
    for doc in documents:
        for index, text in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            meta = {**doc.metadata, "chunk_index": index}
            key = f"{meta.get('source')}|{meta.get('document')}|{meta.get('url') or meta.get('title')}|{meta.get('section')}|{meta.get('page')}|{index}"
            yield Chunk(id=point_id(key), text=text, metadata=meta)


# --------------------------------------------------------------------------- indexing


def index_chunks(chunks: Iterable[Chunk], embedder: Embedder, store: QdrantStore, batch_size: int = 64,
                 progress: Callable[[int], None] | None = None) -> int:
    total = 0
    batch: list[Chunk] = []

    def flush() -> None:
        nonlocal total
        vectors = embedder.embed_documents([c.embedding_text() for c in batch])
        store.upsert([c.id for c in batch], vectors, [c.payload() for c in batch])
        total += len(batch)
        batch.clear()
        if progress:
            progress(total)

    for chunk in chunks:
        batch.append(chunk)
        if len(batch) >= batch_size:
            flush()
    if batch:
        flush()
    return total


def ingest_directory(root: Path, embedder: Embedder, store: QdrantStore, chunk_size: int = 1000,
                     overlap: int = 150, recreate: bool = False, limit: int | None = None,
                     progress: Callable[[int], None] | None = None, folders: list[str] | None = None) -> int:
    store.ensure_collection(recreate=recreate)
    chunks: Iterable[Chunk] = chunk_documents(load_documents(root, folders), chunk_size, overlap)
    if limit:
        chunks = (c for i, c in enumerate(chunks) if i < limit)
    return index_chunks(chunks, embedder, store, progress=progress)
