"""Shared fixtures. Tests never touch the real ``data/feedback`` dir or the
process-wide settings singleton — each test gets a Settings pointed at a tmp dir.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.api.main import create_app
from app.config import Settings
from fastapi.testclient import TestClient

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        corpus_dir=_REPO_ROOT / "data" / "corpus",
        index_dir=tmp_path / "index",
        feedback_dir=tmp_path / "feedback",
        rate_limit_requests=3,
        rate_limit_window_s=60,
        max_question_chars=400,
        cors_origins=["http://localhost:5173"],
        hf_token=None,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c
