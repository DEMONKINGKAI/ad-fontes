"""FastAPI application factory.

The lifespan handler is where the heavy models load exactly once (brief §2:
"load once at startup"). In Phase 0 it only wires ``AppState``; Phases 1-2 add
the embedder, Chroma retriever, the two GGUF generators, and the NLI model to
``state.components`` here.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.dependencies import AppState
from app.api.routes import router
from app.config import Settings, get_settings

log = logging.getLogger("ad_fontes")


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    _configure_logging(settings)
    log.info("ad fontes %s starting; corpus_version=%s", __version__, settings.corpus_version)
    state = AppState.create(settings)
    app.state.app_state = state

    if settings.eager_model_load:
        from app.bootstrap import load_generation, load_retrieval

        try:
            load_retrieval(state.components, settings)
        except Exception:
            log.exception("retrieval failed to load; /api/ask will report degraded")
        try:
            load_generation(state.components, settings)
        except Exception:
            log.exception("generation failed to load; /api/ask will 503")

    try:
        yield
    finally:
        log.info("ad fontes shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="ad fontes — portfolio assistant API",
        version=__version__,
        summary="Grounded, per-claim-verified RAG over Kai Sharma's portfolio.",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": "ad-fontes", "version": __version__, "docs": "/docs"}

    return app


app = create_app()
