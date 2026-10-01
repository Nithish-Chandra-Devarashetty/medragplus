"""BioMistral-7B inference through llama.cpp (quantised GGUF), plus a test double."""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


@dataclass
class Generation:
    text: str
    mean_token_prob: float | None  # geometric-mean probability of generated tokens
    completion_tokens: int
    prompt_tokens: int


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Generation: ...

    def count_tokens(self, text: str) -> int: ...


STOP_SEQUENCES = ["</s>", "[INST]", "\nPatient:", "\nPatient's latest message:"]


class LlamaCppLLM:
    name = "llamacpp"

    def __init__(self, model_path: Path, n_ctx: int = 4096, n_threads: int = 0, n_batch: int = 512):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or None
        self.n_batch = n_batch
        self._llm = None
        self._logprobs_supported = True
        self._lock = threading.Lock()  # a llama.cpp context is not thread-safe

    @property
    def llm(self):
        if self._llm is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"BioMistral GGUF not found at {self.model_path}. Run: python scripts/download_models.py --llm"
                )
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                logits_all=True,  # needed for token log-probabilities (confidence score)
                verbose=False,
            )
        return self._llm

    def count_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False))

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Generation:
        kwargs = dict(max_tokens=max_tokens, temperature=temperature, top_p=0.9, repeat_penalty=1.1,
                      stop=STOP_SEQUENCES)
        with self._lock:
            if self._logprobs_supported:
                try:
                    out = self.llm.create_completion(prompt, logprobs=1, **kwargs)
                except ValueError:  # build without logits_all support
                    self._logprobs_supported = False
                    out = self.llm.create_completion(prompt, **kwargs)
            else:
                out = self.llm.create_completion(prompt, **kwargs)
        choice = out["choices"][0]
        mean_prob = None
        logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
        logprobs = [lp for lp in logprobs if lp is not None]
        if logprobs:
            mean_prob = math.exp(sum(logprobs) / len(logprobs))
        usage = out.get("usage", {})
        return Generation(
            text=choice["text"].strip(),
            mean_token_prob=mean_prob,
            completion_tokens=usage.get("completion_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
        )


class FakeLLM:
    """Deterministic stand-in for tests and model-free smoke runs.

    By default it answers extractively with the first sentences of the first
    context passage in the prompt, so its answers are grounded. Pass `responder`
    to script arbitrary (e.g. hallucinated) answers.
    """

    name = "fake"

    def __init__(self, responder: Callable[[str], str] | None = None, mean_token_prob: float | None = 0.9):
        self.responder = responder
        self.mean_token_prob = mean_token_prob
        self.prompts: list[str] = []

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Generation:
        self.prompts.append(prompt)
        if self.responder:
            text = self.responder(prompt)
        else:
            match = re.search(r"\[1\] \(.*?\) (.+?)(?:\n\n\[2\]|\n\nEarlier patient messages:)", prompt, re.S)
            if match:
                sentences = re.split(r"(?<=[.!?])\s+", match.group(1).strip())
                text = " ".join(sentences[:2])
            else:
                text = "I do not have enough information to answer that."
        return Generation(text=text, mean_token_prob=self.mean_token_prob,
                          completion_tokens=self.count_tokens(text), prompt_tokens=self.count_tokens(prompt))


def create_llm(settings) -> LLM:
    if settings.llm_backend == "llamacpp":
        return LlamaCppLLM(settings.llm_model_path, settings.llm_n_ctx, settings.llm_n_threads, settings.llm_n_batch)
    if settings.llm_backend == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown LLM_BACKEND: {settings.llm_backend}")
