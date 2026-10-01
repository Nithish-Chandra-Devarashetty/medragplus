"""Deterministic red-flag (emergency) symptom classifier.

Runs on every query *before* generation, so emergencies never wait for the LLM
and never depend on it. Rules live in config/red_flags.yaml so clinicians can
edit them without touching code; a trained classifier can later be combined
with this via `EmergencyClassifier.classify` returning the union of both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CLAUSE_BREAK = re.compile(r"[.,;:!?\n]|\bbut\b|\bhowever\b|\balthough\b", re.IGNORECASE)


@dataclass
class EmergencyRule:
    id: str
    label: str
    patterns: list[re.Pattern] = field(default_factory=list)
    all_of: list[list[re.Pattern]] = field(default_factory=list)
    message: str = "emergency"


@dataclass
class EmergencyResult:
    emergency: bool
    rule_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    message_key: str | None = None

    def to_dict(self) -> dict:
        return {"emergency": self.emergency, "rule_ids": self.rule_ids, "labels": self.labels,
                "message_key": self.message_key}


def _compile(patterns) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


class EmergencyClassifier:
    def __init__(self, rules: list[EmergencyRule], negation_terms: list[str] | None = None,
                 negation_window: int = 4, use_negation: bool = True):
        self.rules = rules
        self.negation_terms = {t.lower() for t in (negation_terms or [])}
        self.negation_window = negation_window
        self.use_negation = use_negation

    @classmethod
    def from_yaml(cls, path: Path, use_negation: bool = True) -> "EmergencyClassifier":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        rules = [
            EmergencyRule(
                id=r["id"],
                label=r.get("label", r["id"]),
                patterns=_compile(r.get("patterns", [])),
                all_of=[_compile(group) for group in r.get("all_of", [])],
                message=r.get("message", "emergency"),
            )
            for r in data["rules"]
        ]
        negation = data.get("negation", {})
        return cls(rules, negation.get("terms", []), negation.get("window_words", 4), use_negation)

    def _negated(self, text: str, start: int) -> bool:
        if not self.use_negation:
            return False
        clause = _CLAUSE_BREAK.split(text[:start])[-1]
        words = re.findall(r"[\w']+", clause.lower())[-self.negation_window:]
        return any(w in self.negation_terms for w in words)

    def _matches(self, pattern: re.Pattern, text: str) -> bool:
        return any(not self._negated(text, m.start()) for m in pattern.finditer(text))

    def _rule_fires(self, rule: EmergencyRule, text: str) -> bool:
        if any(self._matches(p, text) for p in rule.patterns):
            return True
        return bool(rule.all_of) and all(any(self._matches(p, text) for p in group) for group in rule.all_of)

    def classify(self, *texts: str | None) -> EmergencyResult:
        fired = [r for r in self.rules if any(t and self._rule_fires(r, t) for t in texts)]
        if not fired:
            return EmergencyResult(False)
        # A crisis-specific message (e.g. mental health) wins over the generic one.
        message = next((r.message for r in fired if r.message != "emergency"), "emergency")
        return EmergencyResult(True, [r.id for r in fired], [r.label for r in fired], message)
