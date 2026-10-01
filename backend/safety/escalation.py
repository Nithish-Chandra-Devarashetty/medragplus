"""Doctor / emergency escalation messages (fixed, pre-translated text)."""
from __future__ import annotations

from pathlib import Path

import yaml


class EscalationService:
    def __init__(self, messages: dict):
        self.messages = messages

    @classmethod
    def from_yaml(cls, path: Path) -> "EscalationService":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def message(self, reason: str, language: str = "en") -> tuple[str, bool]:
        """Return (text, is_in_requested_language). Falls back to English if a translation is missing."""
        variants = self.messages["escalation"].get(reason) or self.messages["escalation"]["low_confidence"]
        if language in variants:
            return variants[language], True
        return variants["en"], language == "en"

    def disclaimer(self, language: str = "en") -> str:
        variants = self.messages["disclaimer"]
        return variants.get(language, variants["en"])

    @property
    def rewrite_note(self) -> str:
        return self.messages["rewrite_note"]
