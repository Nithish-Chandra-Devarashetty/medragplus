import pytest

from backend.evaluation.metrics import (
    bleu, classification_report, latency_summary, mrr, recall_at_k, rouge_1, rouge_l)


def test_retrieval_metrics():
    ranks = [1, 3, None, 2]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 0 + 0.5) / 4)


def test_classification_report():
    rep = classification_report(["a", "a", "b", "c"], ["a", "b", "b", "c"], ["a", "b", "c"])
    assert rep["accuracy"] == 0.75
    assert rep["per_class"]["a"] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(0.6667, abs=1e-4),
                                     "support": 2}
    assert rep["macro_f1"] == pytest.approx((0.6667 + 0.6667 + 1.0) / 3, abs=1e-3)


def test_text_overlap_metrics():
    ref = "fever cough sore throat and body aches"
    assert bleu(ref, ref) == pytest.approx(1.0)
    assert rouge_1(ref, ref) == pytest.approx(1.0) and rouge_l(ref, ref) == pytest.approx(1.0)
    assert rouge_1("fever and chills", ref) == pytest.approx(2 * (2 / 3) * (2 / 7) / (2 / 3 + 2 / 7))
    assert bleu("completely unrelated words here", ref) < 0.05
    assert rouge_l("", ref) == 0.0


def test_latency_summary():
    out = latency_summary([{"llm": 100, "total": 150}, {"llm": 300, "total": 350}])
    assert list(out) == ["llm", "total"]
    assert out["llm"]["mean_ms"] == 200 and out["llm"]["p50_ms"] == 200 and out["total"]["max_ms"] == 350
