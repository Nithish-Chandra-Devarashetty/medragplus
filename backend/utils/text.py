"""Small text helpers shared by ingestion and the safety layer."""
from __future__ import annotations

import html
import re

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+(?=[\"'(\[]?[A-Z0-9ऀ-ॿఀ-౿])")
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just may me might more most must my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them themselves
    then there these they this those through to too under until up very was we were what when where
    which while who whom why will with would you your yours yourself yourselves usually often
    include includes including such like many may can""".split()
)


def strip_html(text: str) -> str:
    text = re.sub(r"(?i)<\s*(br|/p|/li|/h\d|/div)\s*/?>", "\n", text)
    text = re.sub(r"(?i)<\s*li[^>]*>", "\n- ", text)
    return normalize_whitespace(html.unescape(_TAG.sub(" ", text)))


def normalize_whitespace(text: str) -> str:
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split into sentences; bullet lines are treated as separate sentences."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        sentences.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return sentences


def content_words(text: str) -> list[str]:
    """Lower-cased, crudely stemmed content words (for lexical overlap checks)."""
    words = _WORD.findall(text.lower())
    return [_stem(w) for w in words if w not in STOPWORDS and len(w) > 2]


def _stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "es", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word
