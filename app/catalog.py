"""Read ``manifest.json`` + project files for ``GET /api/projects``.

Kept separate from the ingestion/chunking path because the widget's suggestion
chips only need a flat list (id, name, repo, one-liner, domain) and should work
even before the Chroma index is built.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from app.api.schemas import ProjectSummary

_ONE_LINER_RE = re.compile(r"^##\s+One-line summary\s*\n+(?P<body>.+?)(?:\n##|\n*\Z)", re.S | re.M)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<fm>.*?)\n---\s*\n", re.S)


def _clean_name(raw: str | None) -> str:
    return (raw or "").strip().strip('"').strip()


def _one_liner(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = _ONE_LINER_RE.search(text)
    if not m:
        return ""
    return " ".join(m.group("body").split())


@lru_cache(maxsize=8)
def load_projects(corpus_dir: Path) -> tuple[ProjectSummary, ...]:
    manifest_path = corpus_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[ProjectSummary] = []
    for doc in data.get("documents", []):
        if doc.get("doc_type") != "project":
            continue
        rel = doc["path"]
        repo = doc.get("repo")
        if repo and repo.startswith("null"):
            repo = None
        out.append(
            ProjectSummary(
                id=doc.get("id") or Path(rel).stem,
                name=_clean_name(doc.get("name")) or (doc.get("id") or Path(rel).stem),
                repo=repo,
                summary=_one_liner(corpus_dir / rel),
                domain=list(doc.get("domain") or []),
            )
        )
    out.sort(key=lambda p: p.id)
    return tuple(out)
