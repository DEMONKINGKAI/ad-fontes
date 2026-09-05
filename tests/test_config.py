from __future__ import annotations

from app.config import Settings


def test_cors_origins_split_from_string(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://a.com, https://b.com ,")
    s = Settings()
    assert s.cors_origins == ["https://a.com", "https://b.com"]


def test_defaults_are_safe_without_env(monkeypatch):
    for var in ("CORS_ORIGINS", "HF_TOKEN", "AD_FONTES_DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.default_model == "tuned"
    assert s.max_question_chars == 400
    assert s.rate_limit_requests == 10
    assert s.hf_token is None


def test_corpus_version_from_pyproject():
    assert Settings(_env_file=None).corpus_version == "2026-09-05"
