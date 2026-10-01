"""MedRAG+ evaluation suites. Results are written to eval/results/.

    python scripts/evaluate.py retrieval [--n 300]     Recall@K / MRR on MedQuAD (MedlinePlus subset)
    python scripts/evaluate.py safety                  emergency detection + triage, per language
                                                       (real NLLB translation, no LLM - fast)
    python scripts/evaluate.py e2e [--limit 20] [--medquad 10]
                                                       full pipeline incl. BioMistral: triage, escalation,
                                                       hallucination rate, BLEU/ROUGE, per-stage latency

Uses the models configured in .env. e2e uses an in-memory MongoDB, so it does
not touch real users' history.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import ROOT_DIR, Settings  # noqa: E402
from backend.evaluation.metrics import (  # noqa: E402
    bleu, classification_report, latency_summary, mrr, recall_at_k, rouge_1, rouge_l)
from backend.safety.triage import TRIAGE_LEVELS  # noqa: E402

CASES = ROOT_DIR / "eval" / "data" / "triage_cases.jsonl"
MEDQUAD = ROOT_DIR / "data" / "raw" / "medquad" / "4_MPlus_Health_Topics_QA"
RESULTS = ROOT_DIR / "eval" / "results"


def slug(url: str | None) -> str | None:
    return url.rstrip("/").rsplit("/", 1)[-1].lower() if url else None


def load_cases(limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def load_medquad(kb_slugs: set[str] | None = None) -> list[dict]:
    if not MEDQUAD.is_dir():
        sys.exit("MedQuAD not found. Run: python scripts/download_data.py --medquad")
    items = []
    for path in sorted(MEDQUAD.glob("*.xml")):
        doc = ET.parse(path).getroot()
        target = slug(doc.get("url"))
        if kb_slugs is not None and target not in kb_slugs:
            continue
        for pair in doc.iter("QAPair"):
            question = (pair.findtext("Question") or "").strip()
            answer = " ".join((pair.findtext("Answer") or "").split())
            if question and answer:
                items.append({"question": question, "answer": answer, "slug": target, "focus": doc.findtext("Focus")})
    return items


def kb_medlineplus_slugs(settings: Settings) -> set[str]:
    slugs = set()
    for xml in (settings.documents_dir / "medlineplus").glob("*.xml"):
        for _, elem in ET.iterparse(xml):
            if elem.tag == "health-topic":
                if elem.get("language", "English") == "English":
                    slugs.add(slug(elem.get("url")))
                elem.clear()
    return slugs


def save(name: str, result: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# ------------------------------------------------------------------------ retrieval

def run_retrieval(args, settings: Settings) -> dict:
    from backend.rag.embeddings import create_embedder
    from backend.rag.qdrant_store import QdrantStore

    kb = kb_medlineplus_slugs(settings)
    items = load_medquad(kb)
    random.Random(args.seed).shuffle(items)
    items = items[:args.n]
    embedder, store = create_embedder(settings), QdrantStore.from_settings(settings)
    k_max = max(args.k)
    ranks, rows, t_embed, t_search = [], [], [], []
    for item in items:
        t0 = time.perf_counter()
        vector = embedder.embed_query(item["question"])
        t1 = time.perf_counter()
        hits = store.search(vector, k_max)
        t2 = time.perf_counter()
        t_embed.append((t1 - t0) * 1000)
        t_search.append((t2 - t1) * 1000)
        rank = next((i for i, h in enumerate(hits, 1) if slug(h.metadata.get("url")) == item["slug"]), None)
        ranks.append(rank)
        rows.append({"question": item["question"], "target": item["slug"], "rank": rank,
                     "top": [slug(h.metadata.get("url")) or h.metadata.get("title") for h in hits[:3]]})
    result = {
        "suite": "retrieval", "dataset": "MedQuAD 4_MPlus_Health_Topics_QA", "n": len(items),
        "kb_chunks": store.count(),
        "recall_at_k": {f"@{k}": round(recall_at_k(ranks, k), 4) for k in sorted(args.k)},
        "mrr": round(mrr(ranks), 4),
        "latency": latency_summary([{"embedding": a, "retrieval": b} for a, b in zip(t_embed, t_search)]),
        "rows": rows,
    }
    print(f"Retrieval on {len(items)} MedQuAD questions ({result['kb_chunks']} chunks in KB)")
    for k, v in result["recall_at_k"].items():
        print(f"  Recall{k}: {v:.3f}")
    print(f"  MRR:      {result['mrr']:.3f}")
    return result


# ------------------------------------------------------------------------ safety (no LLM)

def by_language(rows: list[dict], key_true: str, key_pred: str, labels=None) -> dict:
    out = {"all": classification_report([r[key_true] for r in rows], [r[key_pred] for r in rows], labels)}
    for lang in sorted({r["language"] for r in rows}):
        sub = [r for r in rows if r["language"] == lang]
        out[lang] = classification_report([r[key_true] for r in sub], [r[key_pred] for r in sub], labels)
    return out


def run_safety(args, settings: Settings) -> dict:
    from backend.safety.emergency import EmergencyClassifier
    from backend.safety.triage import RuleTriageClassifier
    from backend.services.language_service import LanguageService, create_translator

    language = LanguageService(create_translator(settings), settings.supported_languages)
    emergency = EmergencyClassifier.from_yaml(settings.red_flags_path, settings.emergency_negation)
    triage = RuleTriageClassifier.from_yaml(settings.triage_rules_path)
    rows, timings = [], []
    for case in load_cases(args.limit):
        detected = language.detect_language(case["text"])
        t0 = time.perf_counter()
        english = language.to_english(case["text"], detected)
        t1 = time.perf_counter()
        em = emergency.classify(english, case["text"])
        level = "emergency" if em.emergency else triage.classify(english, case["text"])
        t2 = time.perf_counter()
        timings.append({"translate_in": (t1 - t0) * 1000, "safety": (t2 - t1) * 1000})
        rows.append({**case, "detected_language": detected, "translated": english,
                     "pred_emergency": em.emergency, "pred_rules": em.rule_ids, "pred_triage": level})
    result = {
        "suite": "safety", "n": len(rows),
        "language_detection_accuracy": round(sum(r["detected_language"] == r["language"] for r in rows) / len(rows), 4),
        "emergency": by_language(rows, "emergency", "pred_emergency", [True, False]),
        "triage": by_language(rows, "triage_level", "pred_triage", TRIAGE_LEVELS),
        "latency": latency_summary(timings),
        "errors": [{k: r[k] for k in ("id", "text", "translated", "triage_level", "pred_triage")}
                   for r in rows if r["triage_level"] != r["pred_triage"]],
        "rows": rows,
    }
    print(f"Safety rules on {len(rows)} labelled cases (translation: {settings.translation_backend})")
    for lang, rep in result["emergency"].items():
        cls = rep["per_class"]["True"]
        print(f"  [{lang:>3}] emergency P={cls['precision']:.2f} R={cls['recall']:.2f} F1={cls['f1']:.2f}   "
              f"triage acc={result['triage'][lang]['accuracy']:.2f} macro-F1={result['triage'][lang]['macro_f1']:.2f}")
    for err in result["errors"]:
        print(f"  miss {err['id']}: expected {err['triage_level']}, got {err['pred_triage']}  <- {err['translated']}")
    return result


# ------------------------------------------------------------------------ end-to-end

def run_e2e(args, settings: Settings) -> dict:
    import mongomock

    from backend.services.chat_service import ChatRequest
    from backend.services.container import build_services

    settings = settings.with_overrides(mongo_uri="mongomock://")
    services = build_services(settings, db=mongomock.MongoClient()["eval"])
    print("Loading models…", flush=True)
    services.preload()
    user_id = services.store.create_user("eval@example.com", "x", "eval", "en")["_id"]

    def ask(text: str) -> dict:
        return services.pipeline.run(ChatRequest(user_id=user_id, message=text, language="auto", tts=args.tts))

    rows = []
    cases = load_cases(args.limit)
    for i, case in enumerate(cases, 1):
        out = ask(case["text"])
        rows.append({**case, "pred_triage": out["triage_level"], "pred_emergency": out["emergency"],
                     "action": out["action"], "escalated": out["escalated"], "hallucination": out["hallucination"],
                     "confidence": out["confidence"], "flags": out["safety_flags"], "answer": out["answer"],
                     "answer_english": out["answer_english"], "timings_ms": out["timings_ms"]})
        print(f"  [{i}/{len(cases)}] {case['id']}: {out['action']:<8} {out['triage_level']:<15} "
              f"conf={out['confidence']:.2f} {out['timings_ms']['total'] / 1000:.1f}s", flush=True)

    qa_rows = []
    if args.medquad:
        items = load_medquad(kb_medlineplus_slugs(settings))
        random.Random(args.seed).shuffle(items)
        for i, item in enumerate(items[:args.medquad], 1):
            out = ask(item["question"])
            qa_rows.append({"question": item["question"], "reference": item["answer"][:1500],
                            "answer": out["answer_english"], "action": out["action"],
                            "hallucination": out["hallucination"], "confidence": out["confidence"],
                            "bleu": bleu(out["answer_english"], item["answer"]),
                            "rouge_1": rouge_1(out["answer_english"], item["answer"]),
                            "rouge_l": rouge_l(out["answer_english"], item["answer"]),
                            "timings_ms": out["timings_ms"]})
            print(f"  [QA {i}/{args.medquad}] {out['action']:<8} conf={out['confidence']:.2f} "
                  f"ROUGE-L={qa_rows[-1]['rouge_l']:.2f} {out['timings_ms']['total'] / 1000:.1f}s", flush=True)

    generated = [r for r in rows + qa_rows if "generation" in r["timings_ms"]]  # BioMistral actually ran
    non_emergency = [r for r in rows if not r["emergency"]]
    answered = [r for r in qa_rows if r["action"] != "escalate"]
    emergencies = [r for r in rows if r["emergency"]]
    result = {
        "suite": "e2e", "n_cases": len(rows), "n_qa": len(qa_rows),
        "config": {"llm": str(settings.llm_model_path.name), "confidence_threshold": settings.confidence_threshold,
                   "hallucination_backend": settings.hallucination_backend, "top_k": settings.top_k},
        "triage": by_language(rows, "triage_level", "pred_triage", TRIAGE_LEVELS) if rows else None,
        "emergency_escalation_recall": round(sum(r["escalated"] for r in emergencies) / len(emergencies), 4)
        if emergencies else None,
        "non_emergency_escalation_rate": round(sum(r["escalated"] for r in non_emergency) / len(non_emergency), 4)
        if non_emergency else None,
        "hallucination_rate": round(sum(bool(r["hallucination"]) for r in generated) / len(generated), 4)
        if generated else None,
        "actions": {a: sum(r["action"] == a for r in rows + qa_rows) for a in ("deliver", "rewrite", "escalate")},
        # BLEU/ROUGE only make sense for answers that were actually shown (not escalation text)
        "qa_quality": {"n_answered": len(answered),
                       **{m: round(sum(r[m] for r in answered) / len(answered), 4)
                          for m in ("bleu", "rouge_1", "rouge_l")}} if answered else None,
        "latency": latency_summary([r["timings_ms"] for r in rows + qa_rows]),
        "rows": rows, "qa_rows": qa_rows,
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "qa_rows", "triage")}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", choices=["retrieval", "safety", "e2e"])
    parser.add_argument("--n", type=int, default=300, help="retrieval: number of MedQuAD questions")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10], help="retrieval: K values")
    parser.add_argument("--limit", type=int, help="safety/e2e: only the first N triage cases")
    parser.add_argument("--medquad", type=int, default=0, help="e2e: also answer N MedQuAD questions (BLEU/ROUGE)")
    parser.add_argument("--tts", action="store_true", help="e2e: include TTS in the latency measurement")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = Settings.from_env()
    result = {"retrieval": run_retrieval, "safety": run_safety, "e2e": run_e2e}[args.suite](args, settings)
    print(f"\nSaved {save(args.suite, result).relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
