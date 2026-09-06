"""``app.feedback.record_feedback`` — local JSONL append plus the best-effort
HF Dataset mirror (Phase 6). The Space filesystem is ephemeral, so the mirror is
how feedback survives a rebuild; it must never break ``POST /api/feedback``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.api.schemas import FeedbackRequest
from app.config import Settings
from app.feedback import record_feedback

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _settings(tmp_path: Path, **over) -> Settings:
    return Settings(
        corpus_dir=_REPO_ROOT / "data" / "corpus",
        index_dir=tmp_path / "index",
        feedback_dir=tmp_path / "feedback",
        eager_model_load=False,
        **over,
    )


def _payload() -> FeedbackRequest:
    return FeedbackRequest(
        session_id="s1", question="What did Kai build at EffiGO?", answer_id="a-1", rating="up"
    )


def test_appends_locally_without_mirror(tmp_path: Path) -> None:
    s = _settings(tmp_path)  # no feedback_dataset / hf_token
    path = record_feedback(s, _payload())
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["answer_id"] == "a-1"
    assert row["rating"] == "up"
    assert "ip" not in row and "session_id" in row


def test_mirrors_to_dataset_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            calls.append({"token": token})

        def upload_file(self, **kw: object) -> None:
            calls.append(kw)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
    s = _settings(tmp_path, feedback_dataset="acme/ad-fontes-feedback", hf_token="hf_x")

    path = record_feedback(s, _payload())

    assert calls[0] == {"token": "hf_x"}
    up = calls[1]
    assert up["repo_id"] == "acme/ad-fontes-feedback"
    assert up["repo_type"] == "dataset"
    assert up["path_in_repo"] == path.name
    assert Path(up["path_or_fileobj"]) == path


def test_mirror_failure_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BoomApi:
        def __init__(self, token: str | None = None) -> None:
            pass

        def upload_file(self, **kw: object) -> None:
            raise RuntimeError("network down")

    monkeypatch.setattr("huggingface_hub.HfApi", BoomApi)
    s = _settings(tmp_path, feedback_dataset="acme/fb", hf_token="hf_x")

    path = record_feedback(s, _payload())  # must not raise
    assert path.exists()
