"""Load the heavy components once — shared by the API lifespan and the eval /
RLHF scripts so they build the exact same pipeline.

Each loader is best-effort: a failure is logged and leaves that component ``None``
(``/api/health`` then reports ``degraded``; the eval fails loudly instead).
"""

from __future__ import annotations

import logging

from app.api.schemas import GeneratorKind
from app.config import Settings
from app.guardrails.scope import ScopeGate
from app.pipeline import PipelineComponents

log = logging.getLogger("ad_fontes.bootstrap")


def load_retrieval(components: PipelineComponents, settings: Settings) -> None:
    from app.retrieval.retriever import build_retriever

    retriever = build_retriever(settings)
    health = retriever.index_health()
    if health["indexed"] == 0:
        log.warning("Chroma index is empty; run: python -m app.ingestion.cli --rebuild")
    elif health["indexed"] != health["expected"]:
        log.warning("index/corpus mismatch %s — rebuild recommended", health)

    components.retriever = retriever
    components.embedder = retriever.embedder
    try:
        centroid = retriever.corpus_centroid()
    except Exception:
        centroid = None
    components.scope_gate = ScopeGate(
        centroid=centroid,
        top_score_threshold=settings.scope_top_score_threshold,
        centroid_threshold=settings.scope_centroid_threshold,
    )
    log.info("retriever ready: %s", health)


def load_generation(
    components: PipelineComponents, settings: Settings, *, load_tuned: bool = True
) -> None:
    from app.generation.hosted_llm import HostedGenerator
    from app.generation.local_llm import load_local_generator
    from app.verification.nli import NLIVerifier

    s = settings
    common = {
        "n_ctx": s.llm_ctx,
        "n_threads": s.llm_threads,
        "max_tokens": s.llm_max_tokens,
        "temperature": s.llm_temperature,
        "grammar_mode": s.llm_grammar_mode,
    }
    components.base_generator = load_local_generator(
        GeneratorKind.local_base, s.base_gguf_repo, s.base_gguf_file, **common
    )
    log.info("base generator ready: %s", s.base_gguf_file)

    if load_tuned:
        try:
            components.tuned_generator = load_local_generator(
                GeneratorKind.local_tuned, s.tuned_gguf_repo, s.tuned_gguf_file, **common
            )
            log.info("tuned generator ready: %s", s.tuned_gguf_file)
        except Exception:
            log.info("tuned generator not available yet; model=tuned will serve base")

    components.nli = NLIVerifier(s.nli_model)
    log.info("NLI verifier ready: %s", s.nli_model)

    if s.hf_token:
        components.hosted_generator = HostedGenerator(
            s.hosted_model, s.hf_token, provider=s.hosted_provider
        )
        log.info("hosted fallback ready: %s", s.hosted_model)
    else:
        log.info("no HF_TOKEN — hosted fallback disabled")


def load_all(settings: Settings, *, load_tuned: bool = True) -> PipelineComponents:
    components = PipelineComponents()
    load_retrieval(components, settings)
    load_generation(components, settings, load_tuned=load_tuned)
    return components
