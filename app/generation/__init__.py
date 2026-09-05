"""Answer generation: prompts, the JSON-schema contract, and the two backends.

One interface (``Generator``), two implementations: ``local_llm`` (llama.cpp GGUF,
``base`` and ``tuned``) and ``hosted_llm`` (HF Inference Providers fallback).
Implemented in Phase 2; the hosted fallback is wired in Phase 2 and exercised
from Phase 5's comparison run.
"""
