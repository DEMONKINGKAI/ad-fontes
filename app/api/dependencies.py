"""Shared application state and FastAPI dependencies.

Holds the process singletons — settings, the rate limiter, and the (Phase 2+)
loaded model components — and exposes them as dependency callables so routes stay
free of global lookups and tests can override them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Depends, Request

from app.api.rate_limit import RateLimiter
from app.config import Settings, get_settings
from app.pipeline import Pipeline, PipelineComponents


@dataclass
class AppState:
    settings: Settings
    rate_limiter: RateLimiter
    components: PipelineComponents = field(default_factory=PipelineComponents)

    @classmethod
    def create(cls, settings: Settings | None = None) -> AppState:
        s = settings or get_settings()
        return cls(
            settings=s,
            rate_limiter=RateLimiter(s.rate_limit_requests, s.rate_limit_window_s),
        )

    @property
    def pipeline(self) -> Pipeline:
        return Pipeline(self.settings, self.components)


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def get_settings_dep(state: AppState = Depends(get_state)) -> Settings:
    return state.settings


def get_pipeline(state: AppState = Depends(get_state)) -> Pipeline:
    return state.pipeline
