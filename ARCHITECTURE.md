# Architecture

> Living document. Each phase adds a section with the decisions it made, the
> alternatives tried, and the numbers that settled them. No metric appears here
> unless it came from a run of `app/eval`.

## Goal

Answer recruiters' questions about Kai's portfolio **without embellishing**. The
interesting failure mode is not "can't answer" — it's "answers fluently, but
inflates." Every design choice below is in service of catching or preventing that.

## Pipeline

```
question
  │
  ├─ guardrails         length cap (400 chars) · per-IP rate limit (10 / 5 min)
  │                     scope: denylist + corpus-centroid similarity → decline if out of scope
  │                     audience: recruiter | engineer | auto→classified
  │
  ├─ retrieval          dense, ChromaDB (in-process, persisted to data/index/)
  │                     top-k = 6, returns chunk_ids
  │                     metadata filter/boost when the query names a project_id or a stack entry
  │                     always add tech-stack-map chunks when the query names a technology
  │
  ├─ generation         local GGUF (base | tuned), JSON-schema-constrained:
  │                     {"prose": str, "claims": [{"text": str, "cite": [chunk_id]}]}
  │                     prose streamed as tokens; claims parsed at the end
  │                     hosted fallback (HF Inference Providers) if local exceeds timeout
  │
  ├─ verification L1     structural — every cited chunk_id ∈ retrieved set, else fabricated_citation
  ├─ verification L2     NLI (DeBERTa-v3-base-mnli-fever-anli): premise = cited chunk,
  │                     hypothesis = claim → supported | unsupported | contradicted
  ├─ verification L3     numeric guard — every number/%/date in a claim must appear verbatim
  │                     (or as an exact unit conversion) in a cited chunk, else numeric_flag
  │
  └─ response           streamed prose + claims[] with {label, entailment, contradiction,
                        numeric_flag, sources[{path, section, repo_url}]} + meta{generator,
                        latency, retrieved_chunk_ids}
```

Two generators sit behind one interface: `base` (untuned GGUF) and `tuned` (DPO
GGUF). Request picks; default is `tuned` (config).

## Phase 0 decisions

| Decision | Choice | Why |
|---|---|---|
| Project name | **ad fontes** | "to the sources"; matches the `fons-iuris` naming and the repo dir. |
| Package layout | `app/{api,ingestion,retrieval,generation,verification,guardrails,eval,rlhf}` | Each stage independently testable; the API is a thin adapter over `app.pipeline`. |
| Config | one frozen `pydantic-settings` object (`app/config.py`) | Docker build, Space secrets, `.env.example`, and eval scripts describe the same ~20 knobs. |
| Corpus location | flattened `data/corpus/portfolio-corpus/*` → `data/corpus/*`, dropped the loose duplicate files | Matches brief §3; content untouched. |
| Corpus version | `[tool.ad_fontes].corpus_version` in `pyproject.toml`, surfaced by `/api/health` | Single source of truth; bump on content change. |
| Base / tunable model | **Qwen2.5-1.5B-Instruct** | Mature llama.cpp GGUF support, well-trodden TRL DPO + QLoRA recipes, fastest ≤2B option on 2 vCPU, and the model the brief names in Phase 4. HF Pro does not relax the CPU-only serving box, so a bigger tunable model isn't on the table; Pro *is* used for the hosted fallback + the Phase 3 judge. |
| Hosted fallback model | `Qwen/Qwen2.5-7B-Instruct` via HF Inference Providers (configurable) | Same family as the local model; Kai's HF Pro token covers it. Marked `generator: "hosted-fallback"`, never silent. |
| NLI timing | **final claims pass only**, not per-request | The DeBERTa cross-encoder on CPU is the latency-dominant step; scoring retrieved chunks pre-generation buys little for a ~60-chunk corpus. Revisit if Phase 2 retrieval eval shows a need. |
| Rate limiter | in-process sliding window, salted-hashed IP keys | No Redis (brief §2); goal is "stop casual abuse", same as fons-iuris. Full IPs never stored or logged. |
| Streaming | SSE via `sse-starlette`, events `token`/`sources`/`claims`/`meta`/`done`/`error` | The Vercel widget consumes a plain `fetch`/`EventSource` reader. |
| Python | 3.11 is the deployment + CI target; `requires-python` allows 3.12 for local dev | Local box here is 3.12; llama-cpp / training wheels are happiest on 3.11, which Docker and CI pin. |
| CI test deps | a light `requirements-test.txt` (no torch/chroma/llama-cpp) | Phase 0-1 unit tests deliberately don't import the heavy libs; the Docker build job covers the full install. |
| Docker | multi-stage, non-root runtime; model warm-up + index bake are stubbed in the builder with the activation commands visible | Phase 0 image boots and serves the metadata endpoints; Phases 1-2 turn on the bake. |

### Deviations from the brief (Phase 0)

- **Corpus path.** The brief says `data/corpus/`; the delivered corpus was at
  `data/corpus/portfolio-corpus/` with a partial set of byte-identical duplicates
  loose in `data/corpus/`. Flattened to match the brief (confirmed with Kai). No
  corpus content changed.
- **Model bake in the Docker build.** Deferred to Phases 1-2 (needs the ingestion
  loader and the generators/NLI to exist). The builder stage documents the exact
  steps and the runtime falls back to first-boot download until then.
- Nothing else. The §5 API contract is implemented as written.

## Open items carried forward

- Phase 1: confirm `nomic-embed-text-v1.5` vs. a smaller embedder on the CPU
  budget once the index exists and hit@k is measurable.
- Phase 2: tune `ENTAILMENT_THRESHOLD` / `CONTRADICTION_THRESHOLD`
  (`app/verification/labels.py`) against the eval; record the values here.
- Phase 3: the RLAIF preference labels come from this repo's own verifier + an
  LLM judge — a stated limitation to keep visible in the final write-up.

## Limitations (running list)

- The corpus is small (~60 chunks, one subject). Retrieval metrics will have wide
  run-to-run bands; report them as bands, not points (as fons-iuris does).
- RLAIF labels are self-generated. Judge–human agreement is measured on ~100
  hand-labelled pairs in Phase 3; if < ~80%, the rubric is fixed before training.
