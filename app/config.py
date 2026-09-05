"""Runtime configuration, loaded once from the environment.

Why a single frozen settings object rather than reading ``os.environ`` where
needed: the pipeline has ~20 knobs (model ids, timeouts, paths, safety limits)
that must be identical across the API process, the ingestion CLI, and the eval
scripts. Centralising them here means the Docker build, the Space secrets, and
``.env.example`` describe the same surface, and tests can override one field
without monkeypatching the environment.

Nothing in this module imports torch / chromadb / llama_cpp — importing
``app.config`` must stay cheap so CLI ``--help`` and unit tests are fast.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _corpus_version_from_pyproject() -> str:
    """Read ``[tool.ad_fontes].corpus_version`` so the value lives in one place."""
    pyproject = _REPO_ROOT / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data["tool"]["ad_fontes"]["corpus_version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return "unknown"


class Settings(BaseSettings):
    """All backend configuration. Field names map to ``AD_FONTES_*`` env vars.

    ``CORS_ORIGINS`` and ``HF_TOKEN`` keep their bare names to match the
    fons-iuris deployment and the hosting provider's conventions.
    """

    model_config = SettingsConfigDict(
        env_prefix="AD_FONTES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    # --- CORS / hosting ---------------------------------------------------
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:4173"],
        alias="CORS_ORIGINS",
    )
    port: int = 8000
    log_level: str = "INFO"

    # --- Hosted fallback ----------------------------------------------
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hosted_model: str = "Qwen/Qwen2.5-7B-Instruct"
    hosted_provider: str = "auto"

    # --- Generation -------------------------------------------------------
    default_model: str = "tuned"
    local_timeout_s: float = 25.0
    llm_ctx: int = 4096
    llm_threads: int | None = None
    llm_max_tokens: int = 512
    llm_temperature: float = 0.3
    # "json_object" (default) constrains to valid JSON with the schema in the
    # prompt; "schema" uses a JSON-schema grammar (crashes some llama-cpp builds);
    # "none" relies on prompt + parsing. See ARCHITECTURE.md.
    llm_grammar_mode: str = "json_object"
    base_gguf_repo: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    base_gguf_file: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    tuned_gguf_repo: str = "DEMONKINGKAI/ad-fontes-generator-1.5b-dpo-gguf"
    tuned_gguf_file: str = "ad-fontes-1.5b-dpo-q4_k_m.gguf"
    # A local GGUF path that wins over the repo download when set and present —
    # for testing the tuned path before the Hub upload, and as an offline option.
    tuned_gguf_path: str | None = None
    base_gguf_path: str | None = None
    model_cache_dir: Path = _REPO_ROOT / "data" / "models"

    # --- Retrieval / verification models ------------------------------
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    retrieval_top_k: int = 6

    # --- Scope gate (guardrails) --------------------------------------
    # Decline only when the best retrieved chunk is below this — i.e. the corpus
    # has nothing close. "Answerable-looking but unanswerable" questions are the
    # generator's job, not the gate's. Set from the Phase 2 eval.
    scope_top_score_threshold: float = 0.55
    scope_centroid_threshold: float = 0.60  # diagnostic only

    # When False the API lifespan skips loading the embedder / retriever / models
    # (tests inject fakes instead). Always True in Docker and on the Space.
    eager_model_load: bool = True

    # --- Paths ----------------------------------------------------------
    corpus_dir: Path = _REPO_ROOT / "data" / "corpus"
    index_dir: Path = _REPO_ROOT / "data" / "index"
    feedback_dir: Path = _REPO_ROOT / "data" / "feedback"

    # --- Public-endpoint safety --------------------------------------
    rate_limit_requests: int = 10
    rate_limit_window_s: int = 300
    max_question_chars: int = 400

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string (how env vars carry lists)."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def corpus_version(self) -> str:
        return _corpus_version_from_pyproject()

    @property
    def chroma_collection(self) -> str:
        return "portfolio_corpus"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Tests override via ``app.dependencies.settings_override``."""
    return Settings()
