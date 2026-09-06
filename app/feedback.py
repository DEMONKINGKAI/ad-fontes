"""Append thumbs-up/down feedback to ``data/feedback/*.jsonl`` (brief §5) — each
row is a candidate signal for a later DPO round.

The Space filesystem is ephemeral (a rebuild or restart wipes it), so when
``AD_FONTES_FEEDBACK_DATASET`` + ``HF_TOKEN`` are set the row is also appended to
a private Hugging Face **Dataset** repo via ``huggingface_hub``. That push is
best-effort — a failure logs a warning and never breaks the request.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from app.api.schemas import FeedbackRequest
from app.config import Settings

log = logging.getLogger("ad_fontes.feedback")
_lock = Lock()


def _row(payload: FeedbackRequest) -> dict:
    return {
        "ts": time.time(),
        "session_id": payload.session_id,
        "answer_id": payload.answer_id,
        "question": payload.question,
        "rating": payload.rating,
        "note": payload.note,
    }


def record_feedback(settings: Settings, payload: FeedbackRequest) -> Path:
    settings.feedback_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    path = settings.feedback_dir / f"feedback-{day}.jsonl"
    line = json.dumps(_row(payload), ensure_ascii=False)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    _mirror_to_dataset(settings, path, day)
    return path


def _mirror_to_dataset(s: Settings, local_path: Path, day: str) -> None:
    """Upload the day's file to a HF Dataset repo (best effort)."""
    if not (s.feedback_dataset and s.hf_token):
        return
    try:
        from huggingface_hub import HfApi

        HfApi(token=s.hf_token).upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"feedback-{day}.jsonl",
            repo_id=s.feedback_dataset,
            repo_type="dataset",
            commit_message=f"feedback {day}",
        )
    except Exception as exc:
        log.warning("feedback dataset mirror failed (never breaks /api/feedback): %s", exc)
