"""Evaluation metrics (dependency-free): Recall@K, MRR, classification report, BLEU, ROUGE, latency."""
from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# --- retrieval -------------------------------------------------------------------

def recall_at_k(ranks: list[int | None], k: int) -> float:
    """ranks: 1-based rank of the first relevant hit per query (None = not retrieved)."""
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks) if ranks else 0.0


# --- classification -----------------------------------------------------------------

def classification_report(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    labels = labels or sorted(set(y_true) | set(y_pred), key=str)
    per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                                 "support": tp + fn}
    present = [c for c in per_class.values() if c["support"]]
    return {
        "n": len(y_true),
        "accuracy": round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 4) if y_true else 0.0,
        "macro_f1": round(mean(c["f1"] for c in present), 4) if present else 0.0,
        "per_class": per_class,
    }


# --- generation quality --------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU-4 with add-one smoothing (Lin & Och 2004) and brevity penalty."""
    cand, ref = tokenize(candidate), tokenize(reference)
    if not cand or not ref:
        return 0.0
    log_precisions = []
    for n in range(1, max_n + 1):
        c, r = _ngrams(cand, n), _ngrams(ref, n)
        overlap = sum(min(count, r[g]) for g, count in c.items())
        total = max(len(cand) - n + 1, 0)
        log_precisions.append(math.log((overlap + 1) / (total + 1)) if n > 1 else
                              math.log(overlap / total) if overlap else math.log(1e-9))
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    return bp * math.exp(sum(log_precisions) / max_n)


def _f1(overlap: int, cand_len: int, ref_len: int) -> float:
    if not overlap:
        return 0.0
    p, r = overlap / cand_len, overlap / ref_len
    return 2 * p * r / (p + r)


def rouge_1(candidate: str, reference: str) -> float:
    cand, ref = Counter(tokenize(candidate)), Counter(tokenize(reference))
    return _f1(sum((cand & ref).values()), sum(cand.values()), sum(ref.values()))


def rouge_l(candidate: str, reference: str) -> float:
    a, b = tokenize(candidate), tokenize(reference)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return _f1(prev[-1], len(a), len(b))


# --- latency ----------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = (len(ordered) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def latency_summary(timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    stages = sorted({s for t in timings for s in t}, key=lambda s: (s == "total", s))
    out = {}
    for stage in stages:
        values = [t[stage] for t in timings if stage in t]
        out[stage] = {"n": len(values), "mean_ms": round(mean(values), 1), "p50_ms": round(percentile(values, 0.5), 1),
                      "p95_ms": round(percentile(values, 0.95), 1), "max_ms": round(max(values), 1)}
    return out
