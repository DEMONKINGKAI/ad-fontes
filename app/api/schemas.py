"""Pydantic v2 models for the public API contract (brief §5).

These types are the frozen interface the Vercel widget depends on. Treat changes
here as breaking: the brief requires the §5 contract to be unchanged at the end,
or for changes to be called out explicitly. Every field carries a description so
the OpenAPI docs are self-explanatory.

The streamed SSE events (``token``, ``claims``, ``sources``, ``meta``, ``done``,
``error``) carry the same payloads as the corresponding fields of
``AskResponse``; ``/api/ask/sync`` returns the whole ``AskResponse`` at once.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Audience(str, Enum):
    """Who the answer is written for. ``auto`` lets the guardrail classifier pick."""

    recruiter = "recruiter"
    engineer = "engineer"
    auto = "auto"


class ModelChoice(str, Enum):
    """Which generator to use. ``tuned`` is the DPO model; default is set by config."""

    tuned = "tuned"
    base = "base"


class GeneratorKind(str, Enum):
    """Which generator actually produced the answer — surfaced so the UI can badge it.

    ``local-base`` / ``local-tuned`` are the on-Space GGUF models. ``hosted-fallback``
    means local generation exceeded the timeout and a hosted model was used
    instead; the brief forbids failing silently to a different model.
    """

    local_base = "local-base"
    local_tuned = "local-tuned"
    hosted_fallback = "hosted-fallback"


class ClaimLabel(str, Enum):
    """Verification verdict for a single claim (brief §4, mirrors fons-iuris).

    Deliberately four labels, not pass/fail — different failure modes need
    different UI treatment.
    """

    supported = "supported"
    unsupported = "unsupported"
    contradicted = "contradicted"
    fabricated_citation = "fabricated_citation"


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    """Body for ``POST /api/ask`` and ``POST /api/ask/sync``."""

    question: str = Field(
        ...,
        min_length=1,
        description="Recruiter's question about Kai's portfolio. Length is capped by "
        "the server (AD_FONTES_MAX_QUESTION_CHARS, default 400); over-long "
        "questions are rejected with 422.",
    )
    audience: Audience = Field(
        default=Audience.auto,
        description="Tone/technical-depth target. 'auto' classifies from the question.",
    )
    model: ModelChoice | None = Field(
        default=None,
        description="Generator to use. Omit to use the server default (config: 'tuned').",
    )
    session_id: str | None = Field(
        default=None,
        max_length=128,
        description="Opaque client-generated id, echoed back and stored with feedback "
        "so later DPO rounds can group a session's turns.",
    )


class FeedbackRequest(BaseModel):
    """Body for ``POST /api/feedback`` — the live preference-collection hook."""

    session_id: str = Field(..., max_length=128)
    question: str = Field(..., max_length=2000)
    answer_id: str = Field(..., description="The AskResponse.answer_id this rating applies to.")
    rating: Literal["up", "down"]
    note: str | None = Field(default=None, max_length=2000)


# --------------------------------------------------------------------------- #
# Response components
# --------------------------------------------------------------------------- #


class SourceChunk(BaseModel):
    """A retrieved corpus chunk that the answer may cite."""

    chunk_id: str = Field(..., description="Deterministic id: '<file-stem>#<section-slug>'.")
    project_id: str | None = Field(
        default=None, description="From chunk frontmatter, when present."
    )
    doc_type: str = Field(
        ..., description="project | profile | experience | skills | faq | stack_map"
    )
    title: str = Field(..., description="Human breadcrumb: '<name> › <section>'.")
    section: str = Field(..., description="The '##' heading this chunk came from.")
    source_path: str = Field(..., description="Path relative to the corpus dir.")
    repo_url: str | None = Field(default=None, description="GitHub repo for the project, if any.")
    text: str = Field(..., description="The pure citable text (no breadcrumb prefix).")
    score: float | None = Field(default=None, description="Retrieval similarity in [0, 1].")


class ClaimVerification(BaseModel):
    """NLI + numeric-guard result for one claim."""

    label: ClaimLabel
    entailment: float = Field(..., ge=0.0, le=1.0, description="NLI P(premise entails claim).")
    contradiction: float = Field(
        ..., ge=0.0, le=1.0, description="NLI P(premise contradicts claim)."
    )
    numeric_flag: bool = Field(
        default=False,
        description="True if a number/percentage/date in the claim does not appear "
        "verbatim (or as an exact unit conversion) in a cited chunk.",
    )
    numeric_detail: str | None = Field(
        default=None, description="Which token failed the numeric guard, when flagged."
    )


class Claim(BaseModel):
    """One atomic assertion the generator made, with its citations and verdict."""

    text: str
    cite: list[str] = Field(
        default_factory=list,
        description="chunk_ids this claim relies on. Every id must be in the retrieved "
        "set or the claim is labelled fabricated_citation.",
    )
    verification: ClaimVerification
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Resolved cited chunks (subset of the response 'sources').",
    )


class ResponseMeta(BaseModel):
    """Non-content metadata about how the answer was produced."""

    generator: GeneratorKind
    model_requested: ModelChoice
    audience_resolved: Audience
    latency_ms: int
    retrieval_ms: int
    generation_ms: int
    verification_ms: int
    retrieved_chunk_ids: list[str]
    corpus_version: str
    in_scope: bool = Field(..., description="False when the guardrail declined to answer.")


class AskResponse(BaseModel):
    """Full non-streaming answer (``/api/ask/sync``); SSE sends these fields as events."""

    answer_id: str = Field(..., description="Unique per answer; used by /api/feedback.")
    session_id: str | None = None
    prose: str = Field(..., description="The streamed human-readable answer.")
    claims: list[Claim] = Field(default_factory=list)
    unverified_prose: list[str] = Field(
        default_factory=list,
        description="Prose sentences that neither mirror a verified claim nor are "
        "entailed by a retrieved chunk — the model asserted them without support. "
        "A UI should visually flag these in the prose.",
    )
    sources: list[SourceChunk] = Field(default_factory=list)
    meta: ResponseMeta
    declined: bool = Field(
        default=False,
        description="True when the question was out of scope (salary, personal life, "
        "immigration, or anything absent from the corpus). 'prose' then holds a "
        "brief decline message and 'claims' is empty.",
    )


# --------------------------------------------------------------------------- #
# Other endpoints
# --------------------------------------------------------------------------- #


class ProjectSummary(BaseModel):
    """One row for ``GET /api/projects`` — powers the widget's suggestion chips."""

    id: str
    name: str
    repo: str | None = None
    summary: str = Field(..., description="One-line description from the project file.")
    domain: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    stored: bool
    answer_id: str


class ModelStatus(BaseModel):
    name: str
    loaded: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "starting"]
    corpus_version: str
    index_size: int = Field(..., description="Number of chunks in the Chroma collection.")
    models: list[ModelStatus]
    default_model: ModelChoice
    hosted_fallback_configured: bool


class ErrorPayload(BaseModel):
    """Shape of the SSE ``error`` event and of HTTP error bodies."""

    error: str
    detail: str | None = None
