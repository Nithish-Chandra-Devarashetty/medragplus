"""Triage level assignment: self_care < gp_appointment < urgent_care < emergency."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import yaml

TRIAGE_LEVELS = ["self_care", "gp_appointment", "urgent_care", "emergency"]


def max_level(*levels: str) -> str:
    return max(levels, key=TRIAGE_LEVELS.index)


class TriageClassifier(Protocol):
    def classify(self, *texts: str | None) -> str: ...


class RuleTriageClassifier:
    """Keyword/regex triage for non-emergency queries (emergency is decided by red flags)."""

    def __init__(self, levels: dict[str, list[str]]):
        self.levels = {
            level: [re.compile(p, re.IGNORECASE) for p in patterns] for level, patterns in levels.items()
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "RuleTriageClassifier":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["levels"])

    def classify(self, *texts: str | None) -> str:
        for level in reversed(TRIAGE_LEVELS):  # most severe first
            patterns = self.levels.get(level, [])
            if any(p.search(t) for p in patterns for t in texts if t):
                return level
        return "self_care"
