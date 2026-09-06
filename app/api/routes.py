"""HTTP routes for the §5 contract.

Route bodies stay thin: validate, enforce the two public-endpoint safeguards
(length cap + per-IP rate limit) on the paid path, delegate to ``Pipeline``, and
serialise. Anything model-heavy is behind ``Pipeline`` and returns 503 until the
relevant phase lands.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.api import sse
from app.api.dependencies import AppState, get_pipeline, get_state
from app.api.rate_limit import client_key
from app.api.schemas import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelChoice,
    ModelStatus,
    ProjectSummary,
)
from app.catalog import load_projects
from app.feedback import record_feedback
from app.pipeline import Pipeline, PipelineNotReadyError

router = APIRouter(prefix="/api")


def _peer_ip(request: Request) -> str | None:
    """First hop of X-Forwarded-For (Spaces/most proxies), else the socket peer."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


def _enforce_public_limits(request: Request, state: AppState, question: str) -> None:
    if len(question) > state.settings.max_question_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"question exceeds {state.settings.max_question_chars} characters",
        )
    key = client_key(_peer_ip(request))
    allowed, remaining, retry_after = state.rate_limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(int(retry_after) + 1)},
        )
    request.state.rate_remaining = remaining


# Headers that keep an SSE stream unbuffered through a proxy (HF Spaces, nginx).
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/ask")
async def ask_stream(
    body: AskRequest,
    request: Request,
    state: AppState = Depends(get_state),
    pipeline: Pipeline = Depends(get_pipeline),
) -> StreamingResponse:
    _enforce_public_limits(request, state, body.question)

    async def event_source():
        try:
            async for event in pipeline.answer_stream(body):
                yield event.encode()
        except PipelineNotReadyError as exc:
            yield sse.error("pipeline_not_ready", str(exc)).encode()
        except Exception as exc:
            # surface as an SSE error event, never a 500 mid-stream
            yield sse.error("internal_error", type(exc).__name__).encode()

    return StreamingResponse(event_source(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/ask/sync", response_model=AskResponse)
async def ask_sync(
    body: AskRequest,
    request: Request,
    state: AppState = Depends(get_state),
    pipeline: Pipeline = Depends(get_pipeline),
) -> AskResponse:
    _enforce_public_limits(request, state, body.question)
    try:
        return await pipeline.answer_sync(body)
    except PipelineNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    body: FeedbackRequest,
    state: AppState = Depends(get_state),
) -> FeedbackResponse:
    record_feedback(state.settings, body)
    return FeedbackResponse(stored=True, answer_id=body.answer_id)


@router.get("/projects", response_model=list[ProjectSummary])
async def projects(state: AppState = Depends(get_state)) -> list[ProjectSummary]:
    try:
        return list(load_projects(state.settings.corpus_dir))
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"could not read project manifest: {exc}",
        ) from exc


@router.get("/health", response_model=HealthResponse)
async def health(state: AppState = Depends(get_state)) -> HealthResponse:
    components = state.components
    tuned_loaded = components.tuned_generator is not None
    models = [
        ModelStatus(name="retriever", loaded=components.retriever is not None),
        ModelStatus(name="embedder", loaded=components.embedder is not None),
        ModelStatus(name="generator:base", loaded=components.base_generator is not None),
        ModelStatus(
            name="generator:tuned",
            loaded=tuned_loaded,
            detail=(
                None
                if tuned_loaded
                else "optional: tuned GGUF not published yet; 'model=tuned' serves base"
            ),
        ),
        ModelStatus(name="nli", loaded=components.nli is not None),
    ]
    index_size = 0
    if components.retriever is not None and hasattr(components.retriever, "index_health"):
        try:
            index_size = int(components.retriever.index_health()["indexed"])
        except Exception:
            index_size = 0
    # `generator:tuned` is optional: it is absent until the DPO GGUF is published,
    # and `model:"tuned"` transparently serves base meanwhile. It does not gate "ok".
    required = [m for m in models if m.name != "generator:tuned"]
    if all(m.loaded for m in required):
        status_str = "ok"
    elif any(m.loaded for m in models):
        status_str = "degraded"
    else:
        status_str = "starting"
    return HealthResponse(
        status=status_str,
        corpus_version=state.settings.corpus_version,
        index_size=index_size,
        models=models,
        default_model=ModelChoice(state.settings.default_model),
        hosted_fallback_configured=bool(state.settings.hf_token),
    )


@router.get("/health/live")
async def liveness() -> JSONResponse:
    """Cheap liveness probe for the container platform — no state touched."""
    return JSONResponse({"status": "alive"})
