"""Glue: run the verification layers over a generated answer.

Produces the API ``Claim`` objects (per-claim: structural + NLI + numeric), and a
**prose-level** check — the recruiter reads the ``prose``, not ``claims[]``, so a
model that hallucinates in prose while extracting only safe claims must still be
caught. Kept out of the pure layer modules and out of the pipeline.
"""

from __future__ import annotations

import re

from app.api.schemas import Claim, ClaimLabel, ClaimVerification, SourceChunk
from app.generation.schema import AnswerDraft
from app.retrieval.retriever import RetrievedChunk
from app.verification.labels import fuse_label
from app.verification.nli import NLIVerifier
from app.verification.numeric import check_numbers
from app.verification.structural import check_citations

_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been being this that these those "
    "it its his her their kai kais he she they them with as by from into about over under "
    "has have had do does did not no yes also more most such like than then which who what "
    "where when how why can could would should may might will using used use built".split()
)
_PROSE_CLAIM_COVERAGE = 0.7  # prose sentence considered "mirrored by a claim" above this
_PROSE_ENTAIL_THRESHOLD = 0.5
_MAX_PROSE_SENTENCES = 10  # cap NLI work on a runaway answer


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _coverage(text: str, haystack: str) -> float:
    words = _content_words(text)
    if not words:
        return 1.0
    return len(words & _content_words(haystack)) / len(words)


def to_source_chunk(c: RetrievedChunk) -> SourceChunk:
    return SourceChunk(
        chunk_id=c.chunk_id,
        project_id=c.project_id,
        doc_type=c.doc_type,
        title=c.title,
        section=c.section,
        source_path=c.source_path,
        repo_url=c.repo_url,
        text=c.text,
        score=round(c.score, 4),
    )


def verify_answer(
    draft: AnswerDraft,
    retrieved: list[RetrievedChunk],
    nli: NLIVerifier | None,
) -> list[Claim]:
    by_id = {c.chunk_id: c for c in retrieved}
    retrieved_ids = set(by_id)
    claims: list[Claim] = []

    for cd in draft.claims:
        structural = check_citations(cd.cite, retrieved_ids)
        cited_chunks = [by_id[cid] for cid in structural.valid_cites]
        cited_texts = [c.text for c in cited_chunks]
        numeric = check_numbers(cd.text, cited_texts)

        nli_score = None
        if not structural.fabricated and cited_chunks and nli is not None:
            # breadcrumbed premise: a project chunk's pure text often omits its
            # subject ("A solo narrative RPG where…") -> NLI rates a claim that
            # names the project neutral. Citable text the user sees stays pure.
            premises = [f"{c.title}\n{c.text}" for c in cited_chunks]
            nli_score = nli.score_claim(cd.text, premises)

        label, numeric_flag = fuse_label(structural, nli_score, numeric)

        claims.append(
            Claim(
                text=cd.text,
                cite=list(structural.valid_cites) + list(structural.invalid_cites),
                verification=ClaimVerification(
                    label=label,
                    entailment=round(nli_score.entailment, 4) if nli_score else 0.0,
                    contradiction=round(nli_score.contradiction, 4) if nli_score else 0.0,
                    numeric_flag=numeric_flag,
                    numeric_detail=numeric.detail if numeric_flag else None,
                ),
                sources=[to_source_chunk(c) for c in cited_chunks],
            )
        )
    return claims


def verify_prose(
    prose: str,
    claims: list[Claim],
    retrieved: list[RetrievedChunk],
    nli: NLIVerifier | None,
) -> list[str]:
    """Return prose sentences the model asserted without backing them up.

    A sentence is cleared only if a **supported** claim mirrors it — a sentence
    mirrored solely by an ``unsupported`` / ``contradicted`` claim is *not*
    covered (the recruiter reads the prose, and a confident sentence behind a
    failing claim is exactly what must surface: e.g. "Kai chose pgmpy for
    pharmacausal" when the corpus says causal-learn). Everything else is checked
    by NLI against the cited chunks plus the top retrieved ones, in one batch.
    ``neg-k8s`` in the Phase 2 eval is the original motivating case.
    """
    sentences = [s.strip() for s in _SENT.split(prose.strip()) if len(s.strip()) > 15]
    if not sentences:
        return []

    supported_blob = " ".join(
        c.text for c in claims if c.verification.label == ClaimLabel.supported
    )
    pending = [s for s in sentences if _coverage(s, supported_blob) < _PROSE_CLAIM_COVERAGE]
    pending = pending[:_MAX_PROSE_SENTENCES]
    if not pending:
        return []
    if nli is None:
        return pending

    by_id = {r.chunk_id: r for r in retrieved}
    cited = [by_id[cid] for c in claims for cid in c.cite if cid in by_id]
    ranked = sorted(retrieved, key=lambda r: r.score or 0.0, reverse=True)
    prem_chunks: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in [*cited, *ranked]:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            prem_chunks.append(chunk)
    premises = [f"{c.title}\n{c.text}" for c in prem_chunks]

    scores = nli.score_prose(pending, premises)
    return [
        s for s, sc in zip(pending, scores, strict=True) if sc.entailment < _PROSE_ENTAIL_THRESHOLD
    ]
