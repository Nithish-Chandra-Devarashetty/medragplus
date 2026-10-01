"""Environment-driven configuration.

Every model path, threshold and service address is read from the environment
(or a `.env` file at the project root) so nothing medical or deployment-specific
is hard-coded. See `.env.example` for the full list with documentation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent

load_dotenv(ROOT_DIR / ".env")


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _opt(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT_DIR / path


@dataclass(frozen=True)
class Settings:
    # --- server -----------------------------------------------------------
    secret_key: str = "change-me"
    jwt_expiry_hours: int = 24
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    max_audio_mb: int = 10

    # --- storage ----------------------------------------------------------
    mongo_uri: str = "mongodb://localhost:27017"  # "mongomock://" = in-memory
    mongo_db: str = "medrag_plus"

    # --- vector store -----------------------------------------------------
    qdrant_url: str | None = None  # e.g. http://localhost:6333 (Docker)
    qdrant_path: Path | None = None  # embedded on-disk mode when no URL
    qdrant_collection: str = "medical_knowledge"

    # --- embeddings -------------------------------------------------------
    embedding_backend: str = "pubmedbert"  # pubmedbert | fake
    embedding_model: str = "NeuML/pubmedbert-base-embeddings"
    embedding_dim: int = 768
    embedding_batch_size: int = 32

    # --- ingestion --------------------------------------------------------
    documents_dir: Path = ROOT_DIR / "data" / "medical_documents"
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- retrieval / conversation ----------------------------------------
    top_k: int = 3
    min_retrieval_score: float = 0.35
    history_turns: int = 3
    condense_mode: str = "concat"  # concat | llm

    # --- LLM --------------------------------------------------------------
    llm_backend: str = "llamacpp"  # llamacpp | fake
    llm_model_path: Path = ROOT_DIR / "models" / "BioMistral-7B.Q4_K_M.gguf"
    llm_n_ctx: int = 4096
    llm_n_threads: int = 0  # 0 = let llama.cpp decide
    llm_n_batch: int = 512
    llm_max_tokens: int = 200
    llm_context_chars_per_chunk: int = 700  # chunk text is truncated to this in the prompt
    llm_temperature: float = 0.0  # greedy decoding: reproducible answers

    # --- language ---------------------------------------------------------
    supported_languages: list[str] = field(default_factory=lambda: ["en", "hi", "te"])
    translation_backend: str = "nllb"  # nllb | identity | fake
    nllb_model: str = "facebook/nllb-200-distilled-600M"

    # --- speech -----------------------------------------------------------
    asr_backend: str = "faster_whisper"  # faster_whisper | fake | none
    whisper_model: str = "large-v3-turbo"  # "small" cannot transcribe Telugu
    asr_min_avg_logprob: float = -1.0  # below this the recording is rejected as unclear
    whisper_compute_type: str = "int8"
    tts_backend: str = "gtts"  # gtts | fake | none

    # --- safety -----------------------------------------------------------
    confidence_threshold: float = 0.70
    confidence_weight_retrieval: float = 0.3
    confidence_weight_generation: float = 0.3
    confidence_weight_grounding: float = 0.4
    retrieval_score_floor: float = 0.35  # cosine score mapped to confidence 0
    retrieval_score_ceiling: float = 0.85  # cosine score mapped to confidence 1
    hallucination_backend: str = "nli"  # nli | lexical
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    entailment_threshold: float = 0.5
    nli_window_sentences: int = 2  # premise = sliding window of N chunk sentences
    lexical_support_threshold: float = 0.5
    max_unsupported_ratio_for_rewrite: float = 0.34
    emergency_negation: bool = True
    red_flags_path: Path = CONFIG_DIR / "red_flags.yaml"
    triage_rules_path: Path = CONFIG_DIR / "triage_rules.yaml"
    messages_path: Path = CONFIG_DIR / "messages.yaml"

    # --- misc -------------------------------------------------------------
    preload_models: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            secret_key=_str("SECRET_KEY", d.secret_key),
            jwt_expiry_hours=_int("JWT_EXPIRY_HOURS", d.jwt_expiry_hours),
            cors_origins=_list("CORS_ORIGINS", ",".join(d.cors_origins)),
            max_audio_mb=_int("MAX_AUDIO_MB", d.max_audio_mb),
            mongo_uri=_str("MONGO_URI", d.mongo_uri),
            mongo_db=_str("MONGO_DB", d.mongo_db),
            qdrant_url=_opt("QDRANT_URL"),
            qdrant_path=_path("QDRANT_PATH", ROOT_DIR / "data" / "qdrant") if not _opt("QDRANT_URL") else None,
            qdrant_collection=_str("QDRANT_COLLECTION", d.qdrant_collection),
            embedding_backend=_str("EMBEDDING_BACKEND", d.embedding_backend),
            embedding_model=_str("EMBEDDING_MODEL", d.embedding_model),
            embedding_dim=_int("EMBEDDING_DIM", d.embedding_dim),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", d.embedding_batch_size),
            documents_dir=_path("DOCUMENTS_DIR", d.documents_dir),
            chunk_size=_int("CHUNK_SIZE", d.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", d.chunk_overlap),
            top_k=_int("TOP_K", d.top_k),
            min_retrieval_score=_float("MIN_RETRIEVAL_SCORE", d.min_retrieval_score),
            history_turns=_int("HISTORY_TURNS", d.history_turns),
            condense_mode=_str("CONDENSE_MODE", d.condense_mode),
            llm_backend=_str("LLM_BACKEND", d.llm_backend),
            llm_model_path=_path("LLM_MODEL_PATH", d.llm_model_path),
            llm_n_ctx=_int("LLM_N_CTX", d.llm_n_ctx),
            llm_n_threads=_int("LLM_N_THREADS", d.llm_n_threads),
            llm_n_batch=_int("LLM_N_BATCH", d.llm_n_batch),
            llm_max_tokens=_int("LLM_MAX_TOKENS", d.llm_max_tokens),
            llm_context_chars_per_chunk=_int("LLM_CONTEXT_CHARS_PER_CHUNK", d.llm_context_chars_per_chunk),
            llm_temperature=_float("LLM_TEMPERATURE", d.llm_temperature),
            supported_languages=_list("SUPPORTED_LANGUAGES", ",".join(d.supported_languages)),
            translation_backend=_str("TRANSLATION_BACKEND", d.translation_backend),
            nllb_model=_str("NLLB_MODEL", d.nllb_model),
            asr_backend=_str("ASR_BACKEND", d.asr_backend),
            whisper_model=_str("WHISPER_MODEL", d.whisper_model),
            whisper_compute_type=_str("WHISPER_COMPUTE_TYPE", d.whisper_compute_type),
            asr_min_avg_logprob=_float("ASR_MIN_AVG_LOGPROB", d.asr_min_avg_logprob),
            tts_backend=_str("TTS_BACKEND", d.tts_backend),
            confidence_threshold=_float("CONFIDENCE_THRESHOLD", d.confidence_threshold),
            confidence_weight_retrieval=_float("CONFIDENCE_WEIGHT_RETRIEVAL", d.confidence_weight_retrieval),
            confidence_weight_generation=_float("CONFIDENCE_WEIGHT_GENERATION", d.confidence_weight_generation),
            confidence_weight_grounding=_float("CONFIDENCE_WEIGHT_GROUNDING", d.confidence_weight_grounding),
            retrieval_score_floor=_float("RETRIEVAL_SCORE_FLOOR", d.retrieval_score_floor),
            retrieval_score_ceiling=_float("RETRIEVAL_SCORE_CEILING", d.retrieval_score_ceiling),
            hallucination_backend=_str("HALLUCINATION_BACKEND", d.hallucination_backend),
            nli_model=_str("NLI_MODEL", d.nli_model),
            entailment_threshold=_float("ENTAILMENT_THRESHOLD", d.entailment_threshold),
            nli_window_sentences=_int("NLI_WINDOW_SENTENCES", d.nli_window_sentences),
            lexical_support_threshold=_float("LEXICAL_SUPPORT_THRESHOLD", d.lexical_support_threshold),
            max_unsupported_ratio_for_rewrite=_float(
                "MAX_UNSUPPORTED_RATIO_FOR_REWRITE", d.max_unsupported_ratio_for_rewrite
            ),
            emergency_negation=_bool("EMERGENCY_NEGATION", d.emergency_negation),
            red_flags_path=_path("RED_FLAGS_PATH", d.red_flags_path),
            triage_rules_path=_path("TRIAGE_RULES_PATH", d.triage_rules_path),
            messages_path=_path("MESSAGES_PATH", d.messages_path),
            preload_models=_bool("PRELOAD_MODELS", d.preload_models),
        )

    def with_overrides(self, **overrides) -> "Settings":
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return replace(self, **overrides)
