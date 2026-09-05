"""FastAPI layer: request/response schemas, routes, and public-endpoint safety.

This package is a thin adapter. It validates input, enforces the rate limit and
length cap, calls into ``app.retrieval`` / ``app.generation`` / ``app.verification``,
and serialises the result. No pipeline logic lives here.
"""
