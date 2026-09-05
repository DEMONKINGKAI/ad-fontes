"""Query -> top-k chunks, with metadata-aware boosting.

Behaviour (brief §4, Phase 1 notes):

  * dense top-k over the whole corpus (default k = 6), returning ``chunk_id``s;
  * **project boost** — if a query names a project (id or name), its chunks get a
    score bump;
  * **tech boost** — if a query names a technology that appears in a chunk's
    ``stack`` frontmatter, that chunk gets a bump;
  * **stack-map inclusion** — when a query names a (non-generic) technology, the
    ``tech-stack-map`` chunks are pulled into the candidate set and boosted, since
    they answer cross-project "where has Kai used X?" questions directly.

Boosting is additive on the cosine similarity and every weight is in
``RetrievalConfig`` so the Phase 1 eval can run with boosts on and off and the
difference is recorded in ARCHITECTURE.md (brief: don't optimise past what the
eval justifies).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.ingestion.loader import Chunk, load_corpus
from app.retrieval.embedder import Embedder, load_embedder
from app.retrieval.vector_store import ChromaVectorStore, VectorStore

# Technologies too broad to discriminate between projects — kept out of boosting.
_GENERIC_TECH = {
    "python",
    "numpy",
    "pandas",
    "scipy",
    "pytest",
    "uvicorn",
    "yaml config",
    "matplotlib",
    "seaborn",
    "requests",
    "tqdm",
    "axios",
}

# Category phrases that point at the stack map even though no concrete tech is
# named ("which projects use a vector database?", "what datasets has Kai used?").
_CONCEPT_TRIGGERS = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bvector (?:database|db|store|search)\b",
        r"\bdata\s?bases?\b",
        r"\bdatasets?\b",
        r"\btech(?:nology)? stack\b",
        r"\bframeworks?\b",
        r"\barchitectural? patterns?\b",
        r"\brecurring patterns?\b",
        r"\bwhich (?:of \w+ )?projects?\b.*\b(?:use[ds]?|using|have|has|involve|with|built)\b",
        r"\bwhere (?:has|did) \w+ use[ds]?\b",
        r"\bacross (?:all )?(?:his|kai'?s|the) projects?\b",
    )
)


_STOP = frozenset(
    "a an the of to in on for and or is are was were be been what which who how does do did "
    "kai kai's his he they with use used using have has had about tell me project projects "
    "work worked".split()
)


@dataclass(slots=True, frozen=True)
class RetrievalConfig:
    top_k: int = 6
    candidate_pool: int = 24
    project_boost: float = 0.15
    tech_boost: float = 0.08
    stackmap_boost: float = 0.10
    section_boost: float = 0.10
    """When the question's content words cover a chunk's ``##`` section title
    (e.g. "recommender systems experience" vs the section "Recommender systems"),
    nudge that chunk up. Lexical, but only over the *title* — the breadcrumb
    idea from fons-iuris, not the BM25 hybrid it reverted."""
    section_overlap_threshold: float = 0.6
    faq_penalty: float = 0.06
    """Answer-shaped FAQ chunks systematically out-score the operative skills /
    experience / stack chunks a question should really cite (the direct analog of
    fons-iuris's recital de-prioritisation). A small fixed penalty restores the
    hierarchy; an FAQ chunk with a genuine similarity lead still wins."""
    enable_boosts: bool = True


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    title: str
    section: str
    doc_type: str
    source_path: str
    project_id: str | None
    repo_url: str | None
    score: float
    base_score: float
    boosts: tuple[str, ...] = field(default_factory=tuple)


def _word_in(term: str, text: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text) is not None


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chunks: list[Chunk],
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.config = config or RetrievalConfig()
        self._by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self._stackmap_ids = tuple(c.chunk_id for c in chunks if c.doc_type == "stack_map")
        self._tech_terms, self._boost_terms = self._build_tech_vocab(chunks)
        self._project_terms = self._build_project_vocab(chunks)

    # -- vocab ---------------------------------------------------------------

    @staticmethod
    def _build_tech_vocab(chunks: list[Chunk]) -> tuple[frozenset[str], frozenset[str]]:
        terms: set[str] = set()
        for c in chunks:
            for raw in c.stack:
                # keep the head term of "Qwen3-30B-A3B via HF Inference Providers ..."
                term = re.split(r"\s+via\s+|\s*\(", raw.strip().lower())[0].strip()
                if len(term) >= 2:
                    terms.add(term)
        boost = frozenset(t for t in terms if t not in _GENERIC_TECH)
        return frozenset(terms), boost

    @staticmethod
    def _build_project_vocab(chunks: list[Chunk]) -> dict[str, str]:
        vocab: dict[str, str] = {}
        for c in chunks:
            if not c.project_id:
                continue
            pid = c.project_id
            vocab[pid.lower()] = pid
            vocab[pid.replace("-", " ").lower()] = pid
            vocab[pid.replace("-", "").lower()] = pid
            if c.name:
                head = re.split(r"\s*[—:(-]\s*", c.name.strip())[0].strip().lower()
                if len(head) >= 3:
                    vocab[head] = pid
        return vocab

    # -- query analysis --------------------------------------------------

    def techs_in_query(self, q_lower: str) -> set[str]:
        return {t for t in self._tech_terms if _word_in(t, q_lower)}

    def _boost_techs_in_query(self, q_lower: str) -> set[str]:
        return {t for t in self._boost_terms if _word_in(t, q_lower)}

    def projects_in_query(self, q_lower: str) -> set[str]:
        return {pid for surface, pid in self._project_terms.items() if _word_in(surface, q_lower)}

    @staticmethod
    def _has_concept_trigger(q_lower: str) -> bool:
        return any(rx.search(q_lower) for rx in _CONCEPT_TRIGGERS)

    @staticmethod
    def _content_words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}

    def _section_overlap(self, q_words: set[str], section: str) -> float:
        """Fraction of the section title's content words present in the question."""
        title_words = self._content_words(section)
        if not title_words:
            return 0.0
        return len(title_words & q_words) / len(title_words)

    # -- retrieval ------------------------------------------------------

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        query_vector=None,
    ) -> list[RetrievedChunk]:
        k = top_k or self.config.top_k
        q_lower = question.lower()
        q_vec = self.embedder.embed_query(question) if query_vector is None else query_vector

        candidates = {h.chunk_id: h for h in self.store.query(q_vec, self.config.candidate_pool)}

        boost_techs = self._boost_techs_in_query(q_lower)
        named_projects = self.projects_in_query(q_lower)
        wants_stackmap = bool(boost_techs) or self._has_concept_trigger(q_lower)
        q_words = self._content_words(question)

        if wants_stackmap and self._stackmap_ids:
            for h in self.store.query(
                q_vec, len(self._stackmap_ids) + 1, where={"doc_type": "stack_map"}
            ):
                candidates.setdefault(h.chunk_id, h)

        scored: list[RetrievedChunk] = []
        for hit in candidates.values():
            chunk = self._by_id.get(hit.chunk_id)
            if chunk is None:
                continue  # index is ahead of the loaded corpus; skip stale row
            boost = 0.0
            reasons: list[str] = []
            if self.config.enable_boosts:
                if named_projects and chunk.project_id in named_projects:
                    boost += self.config.project_boost
                    reasons.append("project")
                if boost_techs and {s.lower() for s in chunk.stack} & boost_techs:
                    boost += self.config.tech_boost
                    reasons.append("tech")
                if chunk.doc_type == "stack_map" and wants_stackmap:
                    boost += self.config.stackmap_boost
                    reasons.append("stackmap")
                if (
                    chunk.section
                    and self._section_overlap(q_words, chunk.section)
                    >= self.config.section_overlap_threshold
                ):
                    boost += self.config.section_boost
                    reasons.append("section")
                if chunk.doc_type == "faq":
                    boost -= self.config.faq_penalty
                    reasons.append("faq-penalty")
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    title=chunk.title,
                    section=chunk.section,
                    doc_type=chunk.doc_type,
                    source_path=chunk.source_path,
                    project_id=chunk.project_id,
                    repo_url=chunk.repo_url,
                    score=round(hit.score + boost, 6),
                    base_score=round(hit.score, 6),
                    boosts=tuple(reasons),
                )
            )

        scored.sort(key=lambda r: (-r.score, r.chunk_id))
        return scored[:k]

    def index_health(self) -> dict[str, int]:
        return {"indexed": self.store.count(), "expected": len(self._by_id)}

    def corpus_centroid(self):
        """Mean of all chunk embeddings — the scope gate's reference point."""
        import numpy as np

        emb = self.store.all_embeddings()
        if emb.size == 0:
            return None
        c = emb.mean(axis=0)
        n = np.linalg.norm(c)
        return c / n if n else c


def build_retriever(settings: Settings, *, embedder: Embedder | None = None) -> Retriever:
    chunks = load_corpus(Path(settings.corpus_dir))
    store = ChromaVectorStore(Path(settings.index_dir), settings.chroma_collection)
    embedder = embedder or load_embedder(settings.embed_model)
    return Retriever(embedder, store, chunks, RetrievalConfig(top_k=settings.retrieval_top_k))
