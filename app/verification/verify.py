"""Glue: run the verification layers over a generated answer and produce the API
claim objects.

Kept separate from the individual layer modules (which stay pure and independently
testable) and from the pipeline (which shouldn't do per-claim mechanics).
"""

from __future__ import annotations

import re

from app.api.schemas import Claim, ClaimVerification, SourceChunk
from app.generation.schema import AnswerDraft
from app.retrieval.retriever import RetrievedChunk
from app.verification.labels import fuse_label
from app.verification.nli import NLIVerifier
from app.verification.numeric import check_numbers
from app.verification.structural import check_citations

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on at for and or is are was were be been being this that these those "
    "it its his her their kai kais he she they them with as by from into about over under "
    "has have had do does did not no yes also more most such like than then which who what "
    "where when how why can could would should may might will".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def lexical_coverage(claim_text: str, cited_texts: list[str]) -> float:
    """Fraction of the claim's content words that appear in the cited text.

    A high-precision backstop for NLI's poor recall on aggregate claims — see
    ``app.verification.labels``.
    """
    claim_words = _content_words(claim_text)
    if not claim_words:
        return 0.0
    haystack = _content_words(" ".join(cited_texts))
    return len(claim_words & haystack) / len(claim_words)


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
        coverage = 0.0
        if not structural.fabricated and cited_chunks:
            coverage = lexical_coverage(cd.text, cited_texts)
            if nli is not None:
                # Prepend the breadcrumb to each premise: a project chunk's pure
                # text often lacks its subject ("A solo narrative RPG where…"),
                # which NLI then rates neutral against a claim that names the
                # project. The citable text the user sees stays pure.
                premises = [f"{c.title}\n{c.text}" for c in cited_chunks]
                nli_score = nli.score_claim(cd.text, premises)

        label, numeric_flag, lexical_backstop = fuse_label(
            structural, nli_score, numeric, lexical_coverage=coverage
        )

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
                    lexical_backstop=lexical_backstop,
                ),
                sources=[to_source_chunk(c) for c in cited_chunks],
            )
        )
    return claims
