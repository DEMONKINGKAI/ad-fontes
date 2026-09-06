# ad fontes

*ad fontes* — "to the sources." A recruiter-facing assistant that answers questions about
Archit ("Kai") Sharma's portfolio, and — the actual point of the project — **does not lie
about it**: no invented deployments, no rounded-up metrics, no "led" where the source says
"contributed."

Two halves:

1. **Grounded RAG with per-claim verification.** Retrieval over a curated corpus, generation
   constrained to a JSON schema where every claim cites a retrieved chunk, then three
   verification layers — structural (was the cited chunk actually retrieved?), NLI entailment
   (does the cited text support the claim?), and a numeric guard (does every number/date
   appear verbatim in a cited chunk?). The two-layer structural + NLI approach is carried
   over from [`fons-iuris`](https://github.com/DEMONKINGKAI/fons-iuris).
2. **An RLHF/DPO stage** that tunes a small generator toward *faithful-and-humble* over
   *fluent-and-inflated*, using preference pairs whose labels come from that same
   verification layer plus an LLM judge.

The demo lets a recruiter toggle between the `base` and `tuned` models and watch the base
one embellish the CV. Headline metric: **unsupported or fabricated claims per 100 answers,
base vs. tuned.**

This repo is the **backend only** — a CORS-enabled, streaming HTTP API. The chat widget
(Vite + React on Vercel) is separate.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Skeleton: layout, config, API contract, safety, Docker, CI | ✅ done |
| 1 | Ingestion + retrieval + retrieval eval | ✅ done |
| 2 | Generation + verification (base model) | ✅ done |
| 3 | Preference-data pipeline (RLAIF) | ✅ pipeline built + pilot; full run pending |
| 4 | DPO training (Colab) + GGUF export | ✅ notebook + export recipe + code-path test; training run pending |
| 5 | Evaluation: base vs. tuned, the story | ✅ harness + adversarial set built; awaits the tuned GGUF |
| 6 | Deployment (HF Space) | ✅ image built + container smoke-tested locally (cold-start `ok` ~20 s, 4.4 GB to push); Space card + feedback mirror in place; Space push pending |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline and the reasoning behind each choice.

---

## Run it locally

### With Docker (matches the deployment)

```bash
cp .env.example .env          # set HF_TOKEN if you want the hosted fallback
docker compose up --build     # API on http://localhost:8000
```

- `GET  http://localhost:8000/docs` — OpenAPI docs
- `GET  http://localhost:8000/api/health` — models loaded, corpus version, index size
- `GET  http://localhost:8000/api/projects` — project list for the widget's chips

> **Status note:** the full pipeline is live — `/api/ask` (SSE) and `/api/ask/sync`
> answer with grounded, per-claim-verified responses from the **base** model.
> `model: "tuned"` currently falls back to `base` (the DPO model lands in Phase 4;
> `meta.model_requested` reports what actually ran). `/api/health` shows which
> models loaded.

The first boot downloads the embedding model (~0.5 GB), the base GGUF (~1 GB) and the
NLI model (~0.4 GB), and builds the index. To warm everything ahead of time:

```bash
python -m app.ingestion.cli --rebuild               # index -> data/index/
python -m scripts.download_models --embed --nli --base-gguf
```

Set `HF_TOKEN` (HF Pro recommended) to enable the hosted fallback for slow local
generations — without it, a local timeout surfaces as an error rather than a
different model.

### Without Docker (dev loop)

```bash
python -m venv .venv && . .venv/Scripts/activate   # 3.11 is the deployment target; 3.12 works for dev
pip install -r requirements-dev.txt
pip install --no-deps -e .
uvicorn app.api.main:app --reload --port 8000
```

### Tests, lint

```bash
pip install -r requirements-test.txt      # light deps; the full set builds in Docker
pytest -q
ruff check app tests && ruff format --check app tests
pre-commit install                        # optional: run the hooks on every commit
```

### Rebuild the index (Phase 1+)

```bash
python -m app.ingestion.cli --rebuild
```

### Reproduce the evaluation

```bash
python -m app.eval.run_eval --stage retrieval                  # Phase 1
python -m app.eval.run_eval --stage generation --model base    # Phase 2 baseline
python -m app.eval.run_eval --stage compare                    # Phase 5: base vs tuned
```

### Preference data + DPO (Phases 3–4)

```bash
python -m app.rlhf.gen_questions --target 600
python -m app.rlhf.gen_candidates --holdout all     # resumable; ~4 h local
python -m app.rlhf.judge                            # resumable; hosted judge
python -m app.rlhf.build_pairs                      # -> data/rlhf/pairs.jsonl (TRL DPO)
python -m app.rlhf.report                           # perturbation detection, length bias
python -m app.rlhf.hand_label export -n 100         # label, then: hand_label score
```

Then upload `data/rlhf/pairs.jsonl` to Drive and run
[`app/rlhf/train_dpo.ipynb`](app/rlhf/train_dpo.ipynb) on a Colab T4 (QLoRA, resumable),
followed by [`app/rlhf/export_gguf.md`](app/rlhf/export_gguf.md) to merge → GGUF → HF Hub.
The API then serves it as `model: "tuned"`.

### Deploy to a Hugging Face Space

The repo `Dockerfile` is the Space image (Docker SDK). Full walkthrough in
[deploy/DEPLOY.md](deploy/DEPLOY.md); in short:

```bash
scripts/prepare_space.sh <hf-user>/ad-fontes    # stages deploy/README.md, adds the 'space' remote
git push space space:main
# set Space secrets: HF_TOKEN, CORS_ORIGINS, AD_FONTES_FEEDBACK_DATASET
scripts/smoke_test.sh https://<hf-user>-ad-fontes.hf.space --allow-cold
```

Cold start is ~40–70 s (models load once). The Space FS is ephemeral, so
`/api/feedback` rows are mirrored to a private HF Dataset when
`AD_FONTES_FEEDBACK_DATASET` is set. `scripts/smoke_sse.mjs` is a zero-dependency
Node SSE check for Vercel-widget parity.

---

## API contract

`POST /api/ask` (SSE), `POST /api/ask/sync`, `POST /api/feedback`, `GET /api/projects`,
`GET /api/health`. Full request/response models in [`app/api/schemas.py`](app/api/schemas.py)
and the live OpenAPI at `/docs`. The contract is frozen — changes are called out in
ARCHITECTURE.md.

---

## Constraints (why the design looks the way it does)

- **Training:** Colab / Kaggle free T4 only. ≤ ~2B params, QLoRA 4-bit, resumable checkpoints.
- **Serving:** free Hugging Face Space, CPU-only, 2 vCPU / 16 GB. Tuned model served as GGUF
  Q4_K_M via `llama-cpp-python`. Base + tuned (~1 GB each) + NLI (~0.7 GB) + embedder
  (~0.5 GB) load once at startup.
- **Fallback:** if local generation exceeds a timeout, fall back to a hosted model via HF
  Inference Providers and mark the response `generator: "hosted-fallback"`. Never a silent
  switch.
- **No paid services, no databases, no RAG frameworks** — plain Python around `chromadb`,
  `llama-cpp-python`, `transformers`, `sentence-transformers`, `trl`, `peft`, `fastapi`.

---

## Results

**Retrieval (Phase 1)** — 76-question eval, 94-chunk corpus, `nomic-embed-text-v1.5`, CPU,
deterministic across runs:

| | hit@6 | hit@10 | MRR |
|---|---|---|---|
| chunk-level | 93.9% | 100% | 0.74 |
| file-level | 100% | 100% | 0.94 |

Metadata boosts (project / tech / stack-map / section-title / FAQ de-prioritisation) add
**+18 pts** chunk hit@6 over plain dense retrieval. Negative-control questions (salary, Rust,
…) sit ~0.13 lower in top-1 similarity than answerable ones. Full method and the ablation are
in [ARCHITECTURE.md](ARCHITECTURE.md#phase-1-decisions--ingestion--retrieval); reproduce with
`python -m app.eval.run_eval --stage retrieval`.

**Generation faithfulness — base model baseline (Phase 2)** — same 76 questions through the
full pipeline (retrieve → base GGUF → structural + NLI + numeric verification), commit
`e17fa62`, full report in
[`app/eval/baselines/generation-base-phase2.md`](app/eval/baselines/generation-base-phase2.md):

| metric | base (Qwen2.5-1.5B) |
|---|---|
| **unsupported + fabricated claims / 100 answers** | **44.7** |
| **unverified prose sentences / 100 answers** | **35.5** |
| citation hit rate · supported rate | 93.7% · 51% |
| supported / unsupported / fabricated / contradicted (79 claims) | 40 / 29 / 5 / 5 |
| decline on 10 negative controls · false-decline on 66 answerable | 40% · 0% |
| latency p50 / p95 (dev box) | 16 s / 33 s · 7% hosted-fallback |

Genuine failures the layer catches: attributing EffiGO's work to Axisray (*contradicted*),
"deployed Threadfall to production for real users" (*unsupported* — corpus says no demo),
correct facts cited to a non-retrieved chunk (*fabricated_citation*), "Kai has worked on
Kubernetes" appearing only in the prose (*unverified_prose*). See ARCHITECTURE.md for the
verification layers and the Phase 2.5 hardening that moved the headline from 56.6.

**Preference data (Phase 3)** — pipeline built and pilot-validated (20 questions,
[`app/rlhf/pilot/phase3-report.md`](app/rlhf/pilot/phase3-report.md)). The verification layer
and the LLM judge catch *different* deliberate degradations — **23/24** perturbed candidates
were caught by one or the other; the judge covers omission (`drop_limitation` 5/5) and voice
(`first_person` 2/2) that NLI/numeric can't see. Judge scalar separates sources cleanly:
hosted 0.93 / base 0.78 / perturbed 0.59. The full ~378-question run (resumable) and the
judge–human agreement check are Kai's to launch.

**Base vs. tuned (Phase 5)** — `python -m app.rlhf.compare --a base --b tuned` runs both
through the full pipeline over the RLHF holdout + a **31-question overclaiming-bait set**
([`app/eval/adversarial.jsonl`](app/eval/adversarial.jsonl): fake deployments, verb/metric/scope
inflation, false premises like "why did Kai use pgmpy in pharmacausal?" when the corpus says
causal-learn), reports every /100 metric with **bootstrap 95% CIs on the delta**, an **LLM
judge win rate**, and a base-vs-tuned bar chart. The harness, adversarial set, and its
integrity tests are in place; the headline table and figure land when the tuned GGUF exists.
No number in this README isn't from a run of `app/eval` or `app/rlhf`.
