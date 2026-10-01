"""Build the Qdrant knowledge base from data/medical_documents.

    python scripts/ingest.py                   # incremental (idempotent upsert)
    python scripts/ingest.py --recreate        # drop and rebuild the collection
    python scripts/ingest.py --source who      # only one sub-folder

Stop the backend first when using embedded Qdrant (QDRANT_PATH): only one
process may open it at a time. With Docker Qdrant (QDRANT_URL) this is not needed.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import Settings  # noqa: E402
from backend.rag.embeddings import create_embedder  # noqa: E402
from backend.rag.ingestion import ingest_directory  # noqa: E402
from backend.rag.qdrant_store import QdrantStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recreate", action="store_true", help="delete the collection first")
    parser.add_argument("--source", help="only ingest this sub-folder of DOCUMENTS_DIR (e.g. medlineplus)")
    parser.add_argument("--limit", type=int, help="stop after N chunks (for quick tests)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = Settings.from_env()
    root = settings.documents_dir
    if args.source:
        if not (root / args.source).is_dir():
            sys.exit(f"No such folder: {root / args.source}")
    if not root.is_dir() or not any(root.iterdir()):
        sys.exit(f"No documents in {root}. Run: python scripts/download_data.py")

    embedder = create_embedder(settings)
    store = QdrantStore.from_settings(settings)
    print(f"Embeddings: {settings.embedding_model if embedder.name == 'pubmedbert' else embedder.name}")
    print(f"Qdrant:     {settings.qdrant_url or settings.qdrant_path} / {settings.qdrant_collection}")

    t0 = time.perf_counter()

    def progress(n: int) -> None:
        rate = n / max(time.perf_counter() - t0, 1e-6)
        print(f"\r  {n} chunks indexed ({rate:.1f}/s)", end="", flush=True)

    total = ingest_directory(root, embedder, store, settings.chunk_size, settings.chunk_overlap,
                             recreate=args.recreate, limit=args.limit, progress=progress,
                             folders=[args.source] if args.source else None)
    print(f"\nIndexed {total} chunks in {time.perf_counter() - t0:.0f}s. Collection now holds {store.count()} chunks.")


if __name__ == "__main__":
    main()
