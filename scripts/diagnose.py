"""Trace one or more queries through the real pipeline and print every safety signal.

    python scripts/diagnose.py "What are the common symptoms of the flu?" "I have had a cough for three weeks."
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mongomock  # noqa: E402

from backend.config.settings import Settings  # noqa: E402
from backend.services.chat_service import ChatRequest  # noqa: E402
from backend.services.container import build_services  # noqa: E402


def main() -> None:
    queries = sys.argv[1:] or ["What are the common symptoms of the flu?"]
    services = build_services(Settings.from_env().with_overrides(mongo_uri="mongomock://"),
                              db=mongomock.MongoClient()["diag"])
    user = services.store.create_user("diag@example.com", "x", "diag", "en")["_id"]
    for query in queries:
        out = services.pipeline.run(ChatRequest(user_id=user, message=query))
        stored = services.store.messages.find_one({"_id": out["message_id"]})
        safety = stored["safety"]
        print("=" * 100)
        print("QUERY:", query, "| english:", out["query"]["english"])
        print("SOURCES:")
        for s in out["sources"]:
            print(f"   {s['score']:.3f}  {s['source']}: {s['title']} / {s['section']}")
        print("DRAFT:", stored["draft_answer"])
        print("ACTION:", out["action"], "| triage:", out["triage_level"], "| flags:", out["safety_flags"])
        print("CONFIDENCE:", out["confidence"], out["confidence_components"])
        if stored["draft_answer"] and out["sources"]:
            check = services.safety.detector.check(stored["draft_answer"], services.retrieval.retrieve(
                out["query"]["english"], [])[0], out["query"]["english"])
            for s in check.sentences:
                print(f"   claim={s.is_claim!s:<5} score={s.score:.3f} supported={s.supported!s:<5} | {s.sentence}")
        print("TIMINGS:", out["timings_ms"])


if __name__ == "__main__":
    main()
