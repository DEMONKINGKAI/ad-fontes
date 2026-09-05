"""Pre-retrieval guardrails: scope (in-corpus vs decline) and audience selection.

The length cap and rate limit live in the API layer; this package is about *what*
to answer and *how*. ``audience`` has a working heuristic now; ``scope``'s
embedding-similarity gate needs the embedder and lands in Phase 2, but its
denylist is active from Phase 0.
"""
