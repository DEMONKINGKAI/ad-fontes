"""Deterministic test doubles for the retrieval stack — no model downloads.

``HashingEmbedder`` is a bag-of-words hashed embedding: lexically similar strings
land near each other, which is enough to make retriever ranking/boost tests
meaningful without pulling in torch. ``InMemoryVectorStore`` implements the
``VectorStore`` protocol with brute-force cosine and a tiny ``where`` filter.
"""

from __future__ import annotations

import re
from zlib import crc32

import numpy as np
from app.retrieval.vector_store import QueryHit

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _h(text: str) -> int:
    return crc32(text.encode("utf-8"))


class HashingEmbedder:
    model_name = "fake-hashing-embedder"
    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        for tok in tokens:
            v[_h(tok) % self.dim] += 1.0
            if len(tok) > 4:  # a little sub-word signal
                v[_h(tok[:4]) % self.dim] += 0.5
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._emb: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._docs: dict[str, str] = {}
        self._meta: dict[str, dict[str, str]] = {}

    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        self._ids = list(ids)
        self._emb = np.asarray(embeddings, dtype=np.float32)
        self._docs = dict(zip(ids, documents, strict=True))
        self._meta = dict(zip(ids, metadatas, strict=True))

    def query(self, embedding, top_k, where=None) -> list[QueryHit]:
        q = np.asarray(embedding, dtype=np.float32)
        sims = self._emb @ q
        order = np.argsort(-sims)
        hits: list[QueryHit] = []
        for i in order:
            cid = self._ids[i]
            meta = self._meta[cid]
            if where and any(meta.get(k) != v for k, v in where.items()):
                continue
            hits.append(
                QueryHit(chunk_id=cid, text=self._docs[cid], score=float(sims[i]), metadata=meta)
            )
            if len(hits) >= top_k:
                break
        return hits

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self.__init__()


def build_fake_retriever(chunks, config=None):
    """Wire a Retriever over the fake stack with the corpus already indexed."""
    from app.retrieval.retriever import Retriever

    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    store.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embedder.embed_documents([c.embed_text for c in chunks]),
        documents=[c.text for c in chunks],
        metadatas=[c.to_chroma_metadata() for c in chunks],
    )
    return Retriever(embedder, store, chunks, config)


# --------------------------------------------------------------------------- #
# Generation / verification fakes
# --------------------------------------------------------------------------- #


class FakeGenerator:
    """Emits a scripted answer as prose deltas + sets ``.last`` like the real ones."""

    def __init__(self, kind, prose: str, claims: list[dict], *, timed_out=False, error=None):
        from app.generation.base import GeneratedAnswer
        from app.generation.schema import AnswerDraft

        self.kind = kind
        self._prose = prose
        self._draft = AnswerDraft(prose=prose, claims=claims)
        self._timed_out = timed_out
        self._error = error
        self.last = None
        self._GeneratedAnswer = GeneratedAnswer
        self.calls = 0

    async def astream(self, question, context_block, audience, *, deadline=None):
        from app.generation.base import GenerationDelta

        self.calls += 1
        for word in self._prose.split(" "):
            yield GenerationDelta(prose_delta=word + " ")
        import json

        raw = json.dumps(self._draft.model_dump())
        self.last = self._GeneratedAnswer(
            draft=self._draft,
            generator=self.kind,
            raw=raw,
            prose_streamed=self._prose,
            timed_out=self._timed_out,
            error=self._error,
        )
        yield GenerationDelta(done=True, answer=self.last)

    async def collect(self, question, context_block, audience, *, deadline=None):
        async for _ in self.astream(question, context_block, audience):
            pass
        return self.last


class FakeRetriever:
    """Returns a fixed chunk set (by id) as RetrievedChunk — for pipeline tests
    that shouldn't depend on embedding quality."""

    def __init__(self, chunks, want_ids, *, scores=None):
        from app.retrieval.retriever import RetrievedChunk

        by_id = {c.chunk_id: c for c in chunks}
        self._out = []
        for i, cid in enumerate(want_ids):
            c = by_id[cid]
            self._out.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    title=c.title,
                    section=c.section,
                    doc_type=c.doc_type,
                    source_path=c.source_path,
                    project_id=c.project_id,
                    repo_url=c.repo_url,
                    score=(scores or {}).get(cid, 0.85 - 0.02 * i),
                    base_score=(scores or {}).get(cid, 0.85 - 0.02 * i),
                )
            )
        self.embedder = HashingEmbedder()

    def retrieve(self, question, top_k=None, *, query_vector=None):
        return self._out[: top_k or len(self._out)]

    def corpus_centroid(self):
        return None

    def index_health(self):
        return {"indexed": len(self._out), "expected": len(self._out)}


class FakeNLI:
    """Returns a fixed score, or a per-claim-substring lookup."""

    def __init__(self, default=(0.9, 0.08, 0.02), rules=None):
        self.default = default
        self.rules = rules or {}

    def _score(self, claim):
        from app.verification.nli import NLIScore

        for needle, triple in self.rules.items():
            if needle in claim:
                return NLIScore(*triple)
        return NLIScore(*self.default)

    def score_claim(self, claim, premises):
        return self._score(claim)

    def score_prose(self, sentences, premises):
        return [self._score(s) for s in sentences]

    def score(self, premise, hypothesis):
        return self._score(hypothesis)
