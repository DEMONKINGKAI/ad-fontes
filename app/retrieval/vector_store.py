"""Persistent Chroma collection wrapper. (Impl: Phase 1.)

In-process ``chromadb.PersistentClient`` under ``data/index/`` — no server, per
brief §2. This wrapper exists so the retriever depends on a 4-method interface
(``upsert``, ``query``, ``count``, ``reset``) that a fake can satisfy in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class QueryHit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str]


class VectorStore:  # pragma: no cover - Phase 1
    def __init__(self, index_dir: Path, collection: str) -> None:
        raise NotImplementedError("Implemented in Phase 1 (ingestion + retrieval).")

    def count(self) -> int: ...

    def query(self, embedding, top_k: int, where: dict | None = None) -> list[QueryHit]: ...
