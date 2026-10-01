"""Answer confidence score in [0, 1].

A weighted combination of three signals (weights are configurable and are
renormalised over the signals that are available):

* retrieval  - how well the best retrieved chunk matches the query (cosine),
               mapped linearly from [floor, ceiling] to [0, 1];
* generation - geometric-mean token probability reported by BioMistral;
* grounding  - fraction of the answer's claims supported by the retrieved
               context (from the hallucination check).

This is a heuristic score, not a calibrated probability. Fit the weights /
threshold on a labelled set (e.g. with temperature scaling) before relying on it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    score: float
    components: dict[str, float]


class ConfidenceScorer:
    def __init__(self, weight_retrieval: float = 0.3, weight_generation: float = 0.3, weight_grounding: float = 0.4,
                 retrieval_floor: float = 0.35, retrieval_ceiling: float = 0.85):
        self.weights = {"retrieval": weight_retrieval, "generation": weight_generation, "grounding": weight_grounding}
        self.floor = retrieval_floor
        self.ceiling = retrieval_ceiling

    def retrieval_component(self, scores: list[float]) -> float | None:
        if not scores:
            return None
        span = max(self.ceiling - self.floor, 1e-6)
        return min(1.0, max(0.0, (max(scores) - self.floor) / span))

    def score(self, retrieval_scores: list[float], mean_token_prob: float | None,
              grounding: float | None) -> ConfidenceResult:
        components = {
            "retrieval": self.retrieval_component(retrieval_scores),
            "generation": mean_token_prob,
            "grounding": grounding,
        }
        available = {k: v for k, v in components.items() if v is not None and self.weights[k] > 0}
        total_weight = sum(self.weights[k] for k in available)
        if not total_weight:
            return ConfidenceResult(0.0, {})
        value = sum(self.weights[k] * v for k, v in available.items()) / total_weight
        return ConfidenceResult(round(value, 4), {k: round(v, 4) for k, v in available.items()})
