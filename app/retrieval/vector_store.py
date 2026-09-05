"""Persistent Chroma collection wrapper.

In-process ``chromadb.PersistentClient`` under ``data/index/`` — no server (brief
§2). The retriever depends only on the ``VectorStore`` protocol
(``upsert`` / ``query`` / ``count`` / ``reset``), so a fake satisfies it in tests.

Chroma is configured for the ``cosine`` space; ``query`` converts Chroma's
distance back to a similarity in ``[0, 1]`` as ``1 - distance`` so scores compose
with the retriever's boosts.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class QueryHit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, str]


class VectorStore(Protocol):
    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None: ...

    def query(
        self, embedding: np.ndarray, top_k: int, where: dict | None = None
    ) -> list[QueryHit]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def all_embeddings(self) -> np.ndarray: ...


class ChromaVectorStore:
    def __init__(self, index_dir: Path, collection: str) -> None:
        import logging
        import os

        # chromadb 0.5.x ships a posthog integration that raises on every event
        # ("capture() takes 1 positional argument but 3 were given"). The flag
        # doesn't fully silence it in this version, so mute the logger too.
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
        import chromadb

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection
        self._client = chromadb.PersistentClient(
            path=str(index_dir),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            embeddings=[e.tolist() for e in np.asarray(embeddings, dtype=np.float32)],
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, embedding: np.ndarray, top_k: int, where: dict | None = None) -> list[QueryHit]:
        res = self._collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
            n_results=max(1, top_k),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[QueryHit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            hits.append(
                QueryHit(
                    chunk_id=cid,
                    text=doc or "",
                    score=round(1.0 - float(dist), 6),
                    metadata=dict(meta or {}),
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def all_embeddings(self) -> np.ndarray:
        got = self._collection.get(include=["embeddings"])
        emb = got.get("embeddings")
        if emb is None or len(emb) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(emb, dtype=np.float32)

    def reset(self) -> None:
        with contextlib.suppress(Exception):  # collection may not exist yet
            self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
