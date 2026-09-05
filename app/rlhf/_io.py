"""Small resumable-JSONL helpers shared by the RLHF scripts.

The Phase 3 pipeline is long (hundreds of LLM calls). Every stage appends one
JSON object per line and can be re-run: it skips ids already present. Nothing
here imports a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

RLHF_DIR = Path("data/rlhf")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            out.append(json.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(path: str | Path, key: str = "id") -> set[str]:
    return {r[key] for r in iter_jsonl(path) if key in r}
