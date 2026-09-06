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

## Phase 1 decisions — ingestion + retrieval

### Chunking (`app/ingestion/loader.py`)

| Decision | Choice | Why |
|---|---|---|
| Split point | `##` headings for project/profile/skills/experience; per-`**Q:**` block for the FAQ; per-`#` section for the stack map | Matches the corpus README. The FAQ and stack map don't use `##`; each FAQ Q&A and each stack-map section is already an independently citable unit. |
| Chunk id | `<file-stem>#<section-slug>`, slug lowercased, non-alnum→`-`, capped 80 chars, `-2` suffix on collision | Deterministic and human-readable — eval gold and citations both reference these. |
| Breadcrumb | `text` (citable) is pure; `embed_text` = `"<name> › <section>\n<text>"` | fons-iuris's finding: operative text alone has little lexical overlap with question phrasing. Verified here — see the boost ablation below. |
| H1 preamble | Prose between `# H1` and the first `##` becomes a chunk (`kai-profile#who-kai-is`); the bare `# H1` line is stripped | kai-profile's intro paragraph is real content and would otherwise be lost. |
| Corpus size | **94 chunks** (61 project, 12 FAQ, 8 skills, 6 profile, 4 experience, 3 stack-map) | The brief estimated ~60; the projects carry more `##` sections than expected. Still small enough that retrieval tuning must stay conservative. |

### Retrieval (`app/retrieval/`)

- **Embedder:** `nomic-ai/nomic-embed-text-v1.5` via `sentence-transformers`, `search_document:` / `search_query:` prefixes, L2-normalised, CPU. Kept behind the `Embedder` protocol so it is swappable; not re-evaluated against a smaller model yet because hit@k is already strong (below) and the brief says not to over-optimise a 94-chunk corpus.
- **Store:** in-process `chromadb.PersistentClient` under `data/index/`, cosine space, `1 - distance` similarity. Behind a 4-method `VectorStore` protocol; tests use an in-memory brute-force fake.
- **Retriever:** dense top-k over a candidate pool of 24, then additive boosts (all weights in `RetrievalConfig`, all eval-justified below):
  - `project_boost` (+0.15) — query names a `project_id` or project name;
  - `tech_boost` (+0.08) — query names a technology in the chunk's `stack` frontmatter (generic terms — python, numpy, pandas … — excluded);
  - `stackmap_boost` (+0.10) + forced candidate inclusion — query names a non-generic tech, or matches a concept trigger (`"vector database"`, `"which projects use …"`, `"datasets"`, …);
  - `section_boost` (+0.10) — ≥60% of a chunk's `##` title's content words appear in the query (the breadcrumb idea, title-only — *not* the BM25 hybrid fons-iuris reverted);
  - `faq_penalty` (−0.06) — FAQ chunks are answer-shaped and systematically out-scored the operative skills/experience/stack chunk a question should cite. This is the direct analog of fons-iuris's recital de-prioritisation.

### Retrieval eval (`python -m app.eval.run_eval --stage retrieval`)

76 questions (`app/eval/questions.jsonl`): 66 answerable across 8 categories + 10 negative controls (salary, Rust, Kubernetes, visa, …). Gold is 1–3 acceptable chunk ids per question, hand-picked against the built index. Scored at chunk and file granularity.

**Run: commit `e17fa62`, corpus `2026-09-05`, index 94 chunks, nomic-embed-text-v1.5, CPU. Deterministic across 3 repeats** (exact search at this scale — no ANN or MoE nondeterminism, unlike fons-iuris).

| k | chunk hit@k (boosts on) | file hit@k (boosts on) | chunk hit@k (boosts off) |
|---|---|---|---|
| 1 | 62.1% | 90.9% | 33.3% |
| 3 | 81.8% | 97.0% | 60.6% |
| 5 | 87.9% | 98.5% | 69.7% |
| 6 | **93.9%** | **100.0%** | 75.8% |
| 10 | 100.0% | 100.0% | 92.4% |

chunk MRR 0.741 (0.509 off) · file MRR 0.943 (0.726 off). **The boosts add +18 pts chunk hit@6 and +14 pts file hit@6** — they are kept, on by default.

Negative controls: mean top-1 similarity **0.69** vs **0.82** for answerable questions — a ~0.13 gap the Phase 2 scope gate can threshold on.

**Known misses (4, all file-level hits — the right file is retrieved, not the single best chunk):** `exp-axisray` and `exp-ocr` (proper-noun / concept terms the dense model under-weights), `skill-ways` ("engineering decisions" vs the section "Ways of working"), `cross-rag`. Documented rather than chased — the fix would be BM25-hybrid territory, which fons-iuris measured and reverted. `k=10` recovers all of them.

## Phase 2 decisions — generation + verification (base model)

### Generation

| Decision | Choice | Why |
|---|---|---|
| Local backend | `llama-cpp-python` + `Qwen2.5-1.5B-Instruct` Q4_K_M GGUF, loaded once, `n_ctx=4096` | Phase 0 choice. Load is ~1.5 s; ~7–12 s per recruiter answer on this dev box (2× that on a 2 vCPU Space). |
| Output constraint | **`response_format={"type":"json_object"}`** (valid-JSON grammar) + the `{prose, claims}` schema *in the prompt* + a one-shot example, then `parse_answer` + validate | **A full JSON-schema grammar crashes the current `llama-cpp-python` build** — `from_json_schema` and `response_format` with `schema` both raise `OSError: stack overflow` / `access violation` in the grammar sampler (Windows wheel 0.3.35). `json_object` mode is stable and the 1.5B model follows the demonstrated shape reliably. `grammar_mode` is configurable (`schema` / `none`) to retry on other builds. |
| Prose streaming | `ProseStreamer` — an incremental JSON-string parser that emits the growing `prose` value token-by-token, ignores `claims` | "Stream the prose, compute claims at the end" (brief §4) without a second model call. |
| Timeout / fallback | hard wall-clock `deadline` (`local_timeout_s`, default 25 s); on timeout/parse-fail/empty the pipeline calls the hosted model and marks `generator: "hosted-fallback"`. Streaming sends a `token` event with `replace: true` carrying the hosted answer. | llama.cpp chat completions have no clean mid-gen stop, so generation runs on a worker thread the async side abandons; `max_tokens` bounds a runaway. Never a silent swap (brief). |
| Hosted model | `Qwen/Qwen2.5-7B-Instruct` via HF Inference Providers, `json_object` format | Same family as local; ~2 s; Kai's HF Pro token. |
| Citation format | context passages as `[N] id: <chunk_id>\n<breadcrumb>\n<text>`; the model copies the `id:` value | An earlier `chunk_id="..."` format made the 1.5B model cite the literal string `chunk_id=`. |

### Verification (`app/verification/`)

| Layer | Implementation | Notes |
|---|---|---|
| 1 — structural | `check_citations`: every cited id ∈ retrieved set, else `fabricated_citation` (NLI skipped) | Deterministic, cheap, first. |
| 2 — NLI | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, once per answer. Premise = **breadcrumbed** cited chunk (`<name> › <section>\n<text>`), markdown stripped, ≤5 sentence-windows. Claim split into sentences; per sentence, entailment = best over {sentence, **frame-stripped** sentence} × windows; claim entailment = weakest sentence, contradiction = strongest (framed form only). | Three findings, all fixed in Phase 2.5: **(a)** `sentencepiece` is a hard dependency — without it the DeBERTa-v3 tokenizer silently returns all-neutral. **(b)** Breadcrumb premise is essential — a chunk's pure text often omits its subject ("A solo narrative RPG where…") → a claim that names the project scores *neutral* (0.003 → 0.997 with the breadcrumb). **(c)** DeBERTa-base rates a *framed* claim ("Kai's design philosophy emphasises X") **contradiction** even when X is verbatim in the premise; stripping the leading "Kai's … is/are/emphasises …" frame recovers it (0.06 → 0.998). |
| 3 — numeric guard | `check_numbers`: every number/%/date/byte-size/multiplier in a claim must be present (verbatim or same value) in a cited chunk, else `numeric_flag` (advisory, doesn't change the label) | Catches rounded-up metrics NLI is blind to (97.2 vs 98). |
| prose check | `verify_prose`: each prose *sentence* not mirrored by a claim (≥0.7 content-word overlap) is NLI'd against the retrieved chunks; below 0.5 entailment → `unverified_prose` | The recruiter reads `prose`, not `claims[]`. A model can hallucinate in prose and extract only safe claims (`neg-k8s`: "Kai has worked on Kubernetes" in prose, no matching claim). Surfaced on `AskResponse.unverified_prose` and as `prose_unverified_sentences_per_100` in the eval. |
| labels | `ENTAILMENT_THRESHOLD` 0.55 · `CONTRADICTION_THRESHOLD` 0.50 | fons-iuris defaults; contradiction also requires `contradiction ≥ entailment`, so a framed-true claim (high entailment via the stripped variant, moderate contradiction from the framed form) stays `supported`. |

### Scope gate

- **Denylist** (compensation / personal life / immigration / politics-health) — the reliable gate.
- **Retrieval-floor**: decline only when the top retrieved score < 0.55 (gibberish / different subject).
- **The centroid gate the brief suggested was measured and dropped.** The corpus is about one person, so answerable *and* unanswerable "Kai …" questions both sit ~0.59–0.62 from the centroid — no signal. And short factual questions ("Who is Kai?") retrieve at ~0.71, overlapping the negative-control band, so a 0.74 top-score threshold false-declined 25% of the answerable set in testing. Dropping the centroid AND-condition and lowering the floor to 0.55 gives 0% false declines. "Answerable-looking but unanswerable" ("Does Kai know Rust?") is deliberately **not** the gate's job — the generator must say "the corpus doesn't cover that"; the NLI layer flags it if the model rambles instead. This is exactly the base-model weakness the DPO stage targets.

### Phase 2.5 — verification hardening

The first full base eval surfaced four issues; fixed before Phase 3 because the preference-data pipeline depends on this signal:

1. **Prose vs. claims.** Verification checked `claims[]` only, so a model could hallucinate in prose and extract safe claims. Added `verify_prose` + `AskResponse.unverified_prose` + a `prose_unverified_sentences_per_100` eval metric.
2. **NLI frame-strip.** DeBERTa-base was marking framed-but-true claims `contradicted`. Added leading-frame stripping (§ NLI table row) — this was the single biggest quality lever.
3. **Lexical backstop removed.** It rescued 0 claims in the full run and risked false-positives on paraphrase-but-wrong claims; the frame-strip made it redundant.
4. **`self.last` race.** The generator stashed its result on a mutable attribute two overlapping requests could stomp; the terminal `GenerationDelta` now carries the `GeneratedAnswer`.

### Phase 2.6 — verification: prose coverage + latency

Manual testing of the deployed image (five recruiter-style questions) surfaced three things:

1. **`verify_prose` was label-blind.** A prose sentence was cleared if *any* claim lexically mirrored it — including an `unsupported` / `contradicted` one. So "Kai chose pgmpy for pharmacausal" (the corpus says causal-learn) showed **no prose flag** even though its claim was `unsupported`. Fixed: only a **`supported`** claim clears a sentence; everything else goes through NLI against the cited chunks + top-ranked retrieved. The pgmpy false-premise, "Kai deployed Threadfall to production", and the `neg-k8s` Kubernetes case now all surface in `unverified_prose`.
2. **`contradicted` fired on conflations.** The base model merges two source statements ("document pipelines" + "applications on Gemini/Titan" → "document pipelines on Gemini/Titan"); NLI then scored one window as partly contradicting. Added `CONTRADICTION_MARGIN = 0.12` — contradiction must *beat* entailment, not tie it. Conflations now land as `unsupported` ("not confirmed"), which is the honest reading.
3. **Verification latency was ~20 s on multi-sentence answers.** `verify_prose` ran NLI once per sentence, and the `tech-stack-map` chunk (a 2,500-char markdown table with no sentence punctuation) became one 512-token window that forced every batch to pad to max length. Fixes: `_windows` hard-splits any long punctuation-free premise; NLI max sequence length 512 → 384; a single batched forward for all prose (sentence, window) pairs; prose premises restricted to cited + top-ranked. **Measured: verification p50 ≈ 19 s → 1.3 s, p90 ≈ 3.7 s** (Phase 2 eval, 76 questions).

### Deviations from the brief (Phase 2)

- **`app/generation/base.py` and `app/bootstrap.py`** added to the file list — the two generator backends need one shared interface + `ProseStreamer`, and the API lifespan and the eval must build the *same* pipeline.
- **No JSON-schema grammar** — see the generation table. Same guarantee (well-formed `{prose, claims}`) reached via `json_object` + prompt + validation.
- **Centroid scope gate not used** — see above.
- **SSE `token` event gained an optional `replace: true`** (sent once if generation falls back mid-stream) and `meta` gained `warnings` / `fell_back` / `unverified_prose`. Backward-compatible additions to the §5 contract, called out here.
- `sentencepiece` added to runtime deps; `numpy` pin relaxed to `<3` (scipy 1.18 / torch 2.14 require numpy 2).

### Phase 2 eval — base-model baseline

`python -m app.eval.run_eval --stage generation --model base` · 76 questions · base `f89e135` + Phase 2.5/2.6 hardening · full report at [`app/eval/baselines/generation-base-phase2.md`](app/eval/baselines/generation-base-phase2.md).

| metric | before 2.5 | after 2.5 | **base, after 2.6** |
|---|---|---|---|
| **unsupported + fabricated claims / 100 answers** | 56.6 | 44.7 | **48.7** |
| **unverified prose sentences / 100 answers** | — | 35.5 | **68.4** (49% of answers have ≥1) |
| citation hit rate | 93.8% | 93.7% | 93.7% |
| supported rate (of all claims) | 42% | 51% | 48% |
| label distribution | 34 / 38 / 5 / 3 | 40 / 29 / 5 / 5 | supported 38 · unsupported 32 · fabricated 5 · contradicted 4 |
| contradicted / 100 · numeric violations / 100 | 3.9 / 3.9 | 6.6 / 3.9 | **5.3** / 3.9 |
| decline on the 10 negative controls · false-decline on 66 answerable | 40% · 0% | 40% · 0% | 40% · 0% |
| **verification** p50 / p90 | — | ~19 s / — | **1.3 s / 3.7 s** |
| latency p50 / p95 (dev box ≈ ½ a 2 vCPU Space) | 14 / 28 s | 16 / 33 s | 15 / 32 s |
| hosted-fallback rate | 7% | 7% | 7% |

**Why the two claim/prose metrics moved in 2.6** — not a regression, a correction:

- `unsupported+fab` 44.7 → 48.7: the `contradicted` margin moves conflation claims from `contradicted` (not in this metric) into `unsupported` (in it). `contradicted/100` fell 6.6 → 5.3 in step.
- `unverified prose` 35.5 → 68.4: the old number was **artificially low** — sentences behind `unsupported`/`contradicted` claims were silently cleared. The new number is the honest one for a base model that asserts more in prose than it extracts as claims. Roughly half the new flags are legitimate (`adv-thread-prod`, `neg-k8s`, `exp-axisray`, verb inflation, `det-fons-revert` "removed" vs corpus "added"); the rest are **DeBERTa-base false-positives on reworded-but-true prose** ("Kai's strongest area is *production-level language model (LLM) and retrieval-Augmented Generation (RAG) engineering*" vs corpus "production LLM/RAG engineering") — the same model limitation already documented for claims. It inflates the metric for both models; the base more, since it rewords more. The base↔tuned **delta** (Phase 5) is the signal, not the absolute.
- Phase 3 note: `judge.py`'s hard veto at `unverified_prose_count >= 2` should probably become a soft penalty given this noise — flagged for the Phase 3 run.

**Variance:** on repeated small subsets during development the headline moved ±10 (temperature 0.3 + subset composition). The full-set number is more stable; Phase 5 runs base and tuned ≥2× each with bootstrap CIs.

**Genuine failures the layer catches:** `exp-axisray` — the model attributed EffiGO's work to "Axisray"; `adv-thread-prod` — "Kai deployed Threadfall to production for real users" (corpus says `demo: null`); `exp-effigo`/`exp-intern` — correct facts cited to a non-retrieved chunk (`fabricated_citation`); `neg-k8s` — "Kai has worked on Kubernetes" (corpus says "serverless and containerised microservices", never k8s); `adv-led-team` — "responsible for the ML team" (verb inflation). The tuned model should (a) make atomic, checkable claims, (b) keep prose == claims, (c) decline the unanswerable negatives, (d) drop over-reaching verbs.

## Phase 3 decisions — RLAIF preference data

Pipeline (`app/rlhf/`, all stages resumable JSONL): `gen_questions` → `gen_candidates` (+`perturb`) → `judge` → `build_pairs` → `data/rlhf/pairs.jsonl` (TRL DPO format). `report.py` summarises; `hand_label.py` exports a blind sample for Kai and scores judge–human agreement.

| Decision | Choice | Why |
|---|---|---|
| Question set | templated: `section_template × persona × project` + profile/experience/skills/cross/adversarial/negative buckets. **~378** unique (templating ceiling, not 600); deterministic 20% holdout by id-hash | Covers every project/section; personas (recruiter / HR screener / ML lead / skeptical CTO) vary tone. `--paraphrase` (hosted) could push toward 600 — not built, the set is diverse enough for DPO. |
| Candidates per question | `base-t0.3`, `base-t0.9`, `hosted`, + 1–2 `perturb:<type>` of the most-faithful real candidate | Two local temps give a faithful/verbose spread; the hosted 7B is the usual "good" answer; perturbations are known-bad by construction and **labelled** so the judge's detection rate is measurable. |
| Perturbations (`perturb.py`) | inflate_number, upgrade_verb (contributed→led, prototype→production), invent_demo_url, add_unsupported_tech, drop_limitation, first_person — deterministic, one edit each | Exactly the failure modes the project targets (brief §1). |
| Judge | **verification labels + numeric guard + LLM-judge rubric** (hosted, JSON: faithfulness / humility / audience_fit / concision / third_person_voice, 1–5). Weighted scalar + **hard veto** on any contradicted/fabricated claim, ≥2 unverified-prose sentences, judge-faithfulness ≤ 2, or first-person voice. | The verifier and the judge catch *different* failures (below) — using both is the point. |
| Pair construction | per question: highest `combined = judge_scalar − verification_penalties` **non-perturbed, non-vetoed** candidate is `chosen`, paired with every candidate it beats by `--margin` (0.12). | A perturbed candidate can only ever be `rejected`. |
| Length control | after building pairs, down-sample the majority sign of `len(rejected) − len(chosen)` toward the minority (dropping smallest-margin pairs first) and report the residual z. | Brief: length must not predict preference. Pilot: z 0.50 → 0.30. |

### Pilot (20 questions, `app/rlhf/pilot/phase3-report.md`) — validates the pipeline

**Perturbation detection — 23/24 caught by *either* layer:**

| perturbation | verification caught | judge caught |
|---|---|---|
| invent_demo_url | 9/9 | 9/9 |
| add_unsupported_tech | 6/6 | 6/6 |
| drop_limitation | 0/5 | **5/5** (verifier is blind to omission; judge isn't) |
| first_person | 0/2 | **2/2** (verifier is blind to voice; judge isn't) |
| inflate_number | 1/2 | 0/2 → 1 miss; judge prompt strengthened afterward to check numbers digit-for-digit |

Judge scalar cleanly separates sources: hosted **0.93**, base **0.78**, perturb **0.59**. This is the two-signal design working — the LLM judge covers omission and voice, which NLI/numeric can't see.

**Full run** is Kai's to launch (~5–6 h compute, resumable): `gen_candidates --holdout all` → `judge` → `build_pairs`, then `hand_label export -n 100` for the agreement check. If judge–human agreement < 80%, revise `_JUDGE_SYSTEM` before Phase 4.

## Phase 4 decisions — DPO training + GGUF export

| Decision | Choice | Why |
|---|---|---|
| Trainer | TRL `DPOTrainer` (0.13), `DPOConfig(beta=0.1, loss_type="sigmoid")`, 1–2 epochs, cosine LR 5e-6 | Brief §6. `ref_model=None` + `peft_config` → the reference is the adapter-disabled base (no second model in VRAM). |
| PEFT | QLoRA: 4-bit nf4 + double-quant, LoRA `r=16` on all attention + MLP projections, `paged_adamw_8bit`, gradient checkpointing | Fits Qwen2.5-1.5B training in ~10–12 GB → free T4. |
| Resumability | checkpoints every 25 steps to `MyDrive/ad-fontes/dpo-checkpoints/` (`save_total_limit=3`); the train cell auto-detects the newest `checkpoint-*` and passes it to `trainer.train(resume_from_checkpoint=…)` | Colab sessions die (brief §2). Re-running the cell continues. |
| Sanity check | notebook cell 5 computes `trainer.compute_ref_log_probs` on a batch and asserts the reference log-probs are finite and negative | Brief §6 — catches a dtype/padding bug before wasting a T4 session. |
| DPO target completion | the full `{"prose": …, "claims": [...]}` JSON string (what the model emits), from `build_pairs._completion` | The tuned model must keep the served output format. |
| Export | merge adapter into the **fp16** base (not the 4-bit model), `convert_hf_to_gguf.py` → `llama-quantize Q4_K_M` → HF Hub. Full recipe in [`app/rlhf/export_gguf.md`](app/rlhf/export_gguf.md). | Merging into a bnb-4bit model degrades quality silently. |
| Serving | no API change needed — `AD_FONTES_TUNED_GGUF_REPO`/`_FILE` already point at the target repo. `AD_FONTES_TUNED_GGUF_PATH` added for a local GGUF (test before upload / offline). `model: "tuned"` serves it once loaded; `meta.model_requested` reports honestly. | |

The exact TRL 0.13 call sequence is pinned by `tests/test_generation_backends.py::test_dpo_training_code_path` (3-step DPO on a tiny model) so an API change fails a test, not a Colab session. `train_dpo.ipynb` itself runs only on Colab (bitsandbytes needs CUDA).

### Deviations from the brief (Phase 4)

- No W&B by default — `report_to=[]` writes a plain `dpo-log.csv` to Drive; the notebook has a one-line note to enable W&B.
- `bitsandbytes` is **not** in any `requirements*.txt` (CUDA-only) — it's `%pip install`ed in the notebook.

## Phase 5 — base vs. tuned comparison (built; awaits the tuned GGUF)

`python -m app.rlhf.compare --a base --b tuned` runs both arms through the **full
pipeline** (`Pipeline.answer_sync(..., force_generator=…)` bypasses the base/tuned
selection and disables the hosted fallback so each arm's own output is measured).

| Piece | What |
|---|---|
| question set | RLHF **holdout** questions + [`app/eval/adversarial.jsonl`](app/eval/adversarial.jsonl) (**31** overclaim-bait: fake deployment/usage, verb/metric/scope inflation, unanswerable-but-plausible, false-premise, comparison) + the negative controls from `questions.jsonl` |
| per-answer metrics | unsupported+fabricated / 100, contradicted / 100, numeric violations / 100, unverified-prose / 100, supported rate, decline-on-unanswerable, false-decline, prose length, latency p50/p95 |
| **bootstrap 95% CI** | on `metric(B) − metric(A)`, resampling the *paired* question index N times (default 2000); a `✓` marks deltas whose CI excludes 0 in the good direction |
| **LLM judge win rate** | blind A/B per question ("more faithful and humble; a supported short answer beats a longer one that reaches"), order randomised, reported as B-wins / decided |
| plot | grouped bar chart (matplotlib → PNG) of the four /100 metrics |

Before the tuned GGUF exists, `--a base --b hosted` is a valid stand-in that exercises every path. `--limit N` for a quick run. Adversarial-set integrity (every question has an `expected` note, gold ids resolve, false-premise questions target real corpus contradictions like the pgmpy/causal-learn and Qwen3/Qwen2.5 caveats) is pinned by `tests/test_compare.py`.

### Deviations from the brief (Phase 5)

- `compare.py` lives in `app/rlhf/` (brief §6 layout) but is the Phase 5 harness; `app.eval.run_eval --stage compare` still stubs out (points at it).
- `answer_row` / summariser factored out of `app/eval/generation.py` and shared.

## Phase 6 — deployment (Hugging Face Space, Docker SDK)

Target: a free HF Space, 2 vCPU / 16 GB, CPU-only, ephemeral disk, sleeps when idle.

| Piece | What |
|---|---|
| image | the repo `Dockerfile`, multi-stage. **builder** installs deps (CPU-only `torch` from the PyTorch CPU index — a generic `torch` pulls ~2 GB of unused CUDA wheels; prebuilt `llama-cpp-python` CPU wheel via `--extra-index-url`), bakes the corpus + Chroma index + embedder + NLI + base GGUF (+ tuned GGUF when its repo exists), via `python -m scripts.download_models`. **runtime** is CPU-only, non-root uid 1000, models load once at startup. **~10.5 GB on disk / 4.4 GB content** to push (was ~18 GB before the torch-CPU pin). |
| private source repos | the tuned GGUF download step reads `HF_TOKEN` as a **build secret** (`RUN --mount=type=secret,id=HF_TOKEN`) — no token baked into the image. Missing/failed tuned download is a logged no-op; `model:"tuned"` then serves `base`. |
| Space card | [`deploy/README.md`](deploy/README.md) — `sdk: docker`, `app_port: 7860`. `scripts/prepare_space.sh <user>/<space>` stages it as the root `README.md` on a `space` branch and adds the `space` remote. |
| runtime env | dropped `HF_HUB_OFFLINE=1` (a rebuild re-checks etags; offline was brittle when a model 404s), added `HF_HUB_DISABLE_TELEMETRY=1`, `HF_HOME=/app/data/models/hf`, `AD_FONTES_PORT=7860`. `HEALTHCHECK` on `/api/health/live` with `--start-period=150s`. `CMD` is `uvicorn --workers 1` (the rate limiter and the llama.cpp lock are per-process). |
| health semantics | `/api/health` reaches `ok` on the **serving-critical** set (retriever, embedder, `generator:base`, NLI). `generator:tuned` is optional — absent until the DPO GGUF is published, and `model:"tuned"` transparently serves base — so it is reported (with a `detail` note) but does **not** hold the status at `degraded`. |
| feedback persistence | the Space FS is wiped on rebuild/restart, so `record_feedback(settings, payload)` also mirrors the day's JSONL to a private HF **Dataset** repo (`AD_FONTES_FEEDBACK_DATASET` + `HF_TOKEN`) via `HfApi.upload_file`. Best-effort: a failure logs a warning and never breaks `POST /api/feedback`. Covered by `tests/test_feedback.py` (mocked `HfApi`). |
| smoke tests | `scripts/smoke_test.sh <url> [--allow-cold]` hits every §5 endpoint incl. a real SSE parse (events reach `done`, `meta.generator` present) and the 422 length cap, exits non-zero on failure. `scripts/smoke_sse.mjs` does the same SSE consume from Node (Vercel-widget parity, no deps). |
| verify | `docker compose up --build` then `scripts/smoke_test.sh http://localhost:8000 --allow-cold`. CI `docker` job builds the image (buildx). |

**Validated locally** (Docker Desktop / Windows, WSL2): `docker build --target runtime` succeeds (torch `2.14.0+cpu`, no CUDA wheels; base GGUF baked, tuned GGUF a logged no-op via `RepositoryNotFoundError`). The container cold-starts to `status: ok` in **~20 s**, and `scripts/smoke_test.sh` passes 7/7 — `/api/ask` (SSE: `sources → token → claims → meta → done`, `generator: local-base`), `/api/ask/sync`, `/api/feedback`, the 422 cap. Generation was ~12 s/answer here; a 2 vCPU Space will be slower, which is what the hosted fallback is for.

### Deviations from the brief (Phase 6)

- The §5 contract is unchanged. `record_feedback`'s **internal** signature changed from `(feedback_dir, payload)` to `(settings, payload)` so the dataset-mirror config threads through without a global settings read (matters for test overrides); no HTTP-visible change.
- `/api/health` `ok` no longer requires `generator:tuned` (see health semantics above) — a behaviour change to a diagnostic endpoint, not the §5 answer contract.

## Open items carried forward

- Phase 3 full run + judge–human agreement check (Kai; scripts + pilot in place).
- Phase 4 training run on Colab (Kai; notebook + export recipe + code-path test
  in place). Record real numbers in `export_gguf.md`.
- Phase 5 real run once the tuned GGUF exists; then re-tune
  `ENTAILMENT_THRESHOLD` etc. if the label distribution warrants it and write the
  README results table + the base-vs-tuned figure.

## Limitations (running list)

- The corpus is small (94 chunks, one subject). Retrieval at this scale is
  deterministic (exact search), so unlike fons-iuris the Phase 1 numbers are
  points, not bands — but they are also easy to over-fit, so boost weights were
  only added where the ablation showed a clear gain.
- RLAIF labels are self-generated. Judge–human agreement is measured on ~100
  hand-labelled pairs in Phase 3; if < ~80%, the rubric is fixed before training.
- **NLI recall on aggregate/summary claims is weak** (DeBERTa-v3-base). The
  frame-strip (Phase 2.5) fixed the framed-claim class; some genuinely-supported
  broad claims still land as `unsupported`, and the Phase 2.6 prose check adds a
  second surface for the same weakness — a reworded-but-true prose sentence
  ("production-level language model (LLM) and retrieval-Augmented Generation (RAG)
  engineering" vs corpus "production LLM/RAG engineering") is flagged. This
  inflates both the "unsupported / 100" and "unverified-prose / 100" metrics for
  *both* models, the base more (it rewords more), which is itself a faithfulness
  signal. Treat `unsupported` / flagged-prose as "not confirmed", not "false"
  (fons-iuris's four-label rationale); the base↔tuned delta is the real signal.
  A better NLI model would help but is out of the CPU-serving budget.
- Local generation on a 2 vCPU Space is slow (~15–40 s); the hosted fallback
  carries the long tail. The `generator` field always says which model answered.
  A consequence: on the deployed Space many answers are `hosted-fallback`, which
  weakens the base↔tuned toggle as a *live* demo — the honest comparison is the
  Phase 5 `compare` run, not clicking the toggle on a cold Space. Raising
  `AD_FONTES_LOCAL_TIMEOUT_S` trades latency for more genuine local answers.
- Feedback rows only persist if `AD_FONTES_FEEDBACK_DATASET` is set; otherwise
  they live in the container's ephemeral disk until the next restart.
