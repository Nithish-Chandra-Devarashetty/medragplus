"""Download all model weights used by MedRAG+.

    python scripts/download_models.py            # everything (~7 GB)
    python scripts/download_models.py --llm      # only BioMistral-7B GGUF

BioMistral goes to ./models (LLM_MODEL_PATH); the rest goes to the Hugging Face
cache (~/.cache/huggingface) and is loaded from there automatically.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import Settings  # noqa: E402

GGUF_REPO = "MaziyarPanahi/BioMistral-7B-GGUF"


def step(name: str):
    print(f"\n==> {name}", flush=True)
    return time.perf_counter()


def done(t0: float):
    print(f"    done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("llm", "embeddings", "nllb", "nli", "whisper"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    selected = {k for k, v in vars(args).items() if v} or {"llm", "embeddings", "nllb", "nli", "whisper"}
    s = Settings.from_env()

    if "llm" in selected:
        t0 = step(f"BioMistral-7B ({s.llm_model_path.name}) from {GGUF_REPO}")
        if s.llm_model_path.exists():
            print("    already present")
        else:
            from huggingface_hub import hf_hub_download

            s.llm_model_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(GGUF_REPO, s.llm_model_path.name, local_dir=str(s.llm_model_path.parent))
        done(t0)

    if "embeddings" in selected:
        t0 = step(f"PubMedBERT embeddings ({s.embedding_model})")
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(s.embedding_model, device="cpu")
        done(t0)

    if "nllb" in selected:
        t0 = step(f"NLLB translation ({s.nllb_model})")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        AutoTokenizer.from_pretrained(s.nllb_model)
        AutoModelForSeq2SeqLM.from_pretrained(s.nllb_model)
        done(t0)

    if "nli" in selected:
        t0 = step(f"NLI hallucination checker ({s.nli_model})")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        AutoTokenizer.from_pretrained(s.nli_model)
        AutoModelForSequenceClassification.from_pretrained(s.nli_model)
        done(t0)

    if "whisper" in selected:
        t0 = step(f"Whisper ASR ({s.whisper_model}, faster-whisper)")
        from faster_whisper import WhisperModel

        WhisperModel(s.whisper_model, device="cpu", compute_type=s.whisper_compute_type)
        done(t0)

    print("\nAll requested models are ready.")


if __name__ == "__main__":
    main()
