"""Parse corpus Markdown into citable chunks. (Full implementation: Phase 1.)

Chunking rules from the corpus README and brief §3:
  * Split each file on ``##`` headings; keep YAML frontmatter as chunk metadata.
  * Deterministic id: ``<file-stem>#<section-slug>``.
  * The embedded text is prefixed with a breadcrumb ``<name> › <section>``; the
    ``text`` field stays pure so citations quote exactly what the corpus says.
  * ``stack/tech-stack-map.md`` and ``faq/recruiter-faq.md`` have no ``##``
    project sections in the usual sense — the FAQ is chunked per ``**Q:**`` block,
    the stack map per ``#`` section.

This module must not import torch / chromadb — determinism tests run in CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True, frozen=True)
class Chunk:
    """One citable unit of the corpus."""

    chunk_id: str
    text: str
    """Pure citable text — no breadcrumb, no heading."""
    embed_text: str
    """What actually gets embedded: ``breadcrumb + '\\n' + text``."""
    title: str
    """Breadcrumb: ``<name> › <section>``."""
    section: str
    doc_type: str
    source_path: str
    project_id: str | None = None
    repo_url: str | None = None
    stack: tuple[str, ...] = ()
    domain: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


def slugify(text: str) -> str:
    """Lowercase, non-alphanumerics to single hyphens, trimmed. Deterministic."""
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")


def load_corpus(corpus_dir: Path) -> list[Chunk]:  # pragma: no cover - Phase 1
    """Walk ``corpus_dir`` and return all chunks in a stable order."""
    raise NotImplementedError("Implemented in Phase 1 (ingestion + retrieval).")


def chunk_markdown(path: Path, corpus_dir: Path) -> list[Chunk]:  # pragma: no cover - Phase 1
    """Chunk a single Markdown file."""
    raise NotImplementedError("Implemented in Phase 1 (ingestion + retrieval).")
