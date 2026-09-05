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
| 1 | Ingestion + retrieval + retrieval eval | ⬜ |
| 2 | Generation + verification (base model) | ⬜ |
| 3 | Preference-data pipeline (RLAIF) | ⬜ |
| 4 | DPO training (Colab) + GGUF export | ⬜ |
| 5 | Evaluation: base vs. tuned, the story | ⬜ |
| 6 | Deployment (HF Space) | ⬜ |

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

> **Phase 0 note:** `/api/ask` and `/api/ask/sync` return a `pipeline_not_ready` error /
> `503` until Phase 2. Retrieval-backed answers arrive in Phase 2; retrieval itself in
> Phase 1. `/api/health`, `/api/projects`, `/api/feedback` work now.

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

### Reproduce the evaluation (Phase 2 / 5)

```bash
python -m app.eval.run_eval --stage compare      # one command, base vs. tuned
```

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

_Populated from real eval runs in Phase 2 (baseline) and Phase 5 (base vs. tuned). No number
appears here that isn't from a run of `app/eval`._
