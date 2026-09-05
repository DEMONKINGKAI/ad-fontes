"""ad fontes — a recruiter-facing RAG assistant over Kai Sharma's portfolio.

The package is split so that the expensive, model-loading parts (retrieval,
generation, verification) can be imported and tested in isolation, and so the
FastAPI layer stays a thin adapter over them. See ARCHITECTURE.md for the
pipeline and the reasoning behind each stage.
"""

__version__ = "0.1.0"
