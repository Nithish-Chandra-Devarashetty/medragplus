"""Per-stage latency instrumentation for the chat pipeline."""
from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    """Collects wall-clock durations (ms) for named pipeline stages.

    Stages that run more than once (e.g. translation in and out use different
    names, but retries would share one) are accumulated.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - start) * 1000)

    def add(self, name: str, ms: float) -> None:
        self.timings[name] = round(self.timings.get(name, 0.0) + ms, 1)

    def finish(self) -> dict[str, float]:
        self.timings["total"] = round((time.perf_counter() - self._start) * 1000, 1)
        return dict(self.timings)
