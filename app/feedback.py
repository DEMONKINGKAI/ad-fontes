"""Append thumbs-up/down feedback to ``data/feedback/*.jsonl``.

This is the live preference-collection hook (brief §5): each row is a candidate
signal for a later DPO round. One file per UTC day, append-only, one JSON object
per line. No IPs, no PII beyond what the user typed in ``note``.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from app.api.schemas import FeedbackRequest

_lock = Lock()


def record_feedback(feedback_dir: Path, payload: FeedbackRequest) -> Path:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    path = feedback_dir / f"feedback-{day}.jsonl"
    row = {
        "ts": time.time(),
        "session_id": payload.session_id,
        "answer_id": payload.answer_id,
        "question": payload.question,
        "rating": payload.rating,
        "note": payload.note,
    }
    line = json.dumps(row, ensure_ascii=False)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path
