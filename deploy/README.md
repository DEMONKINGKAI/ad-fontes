---
title: ad fontes — portfolio assistant
emoji: 📚
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Grounded, per-claim-verified RAG over Kai Sharma's portfolio
---

# ad fontes

Backend for a recruiter-facing assistant that answers questions about Archit
("Kai") Sharma's portfolio — and is built not to embellish it (no invented
deployments, rounded-up metrics, or "led" where the source says "contributed").

- Grounded RAG + structural / NLI / numeric per-claim verification.
- A DPO-tuned 1.5B generator served as GGUF; toggle `base` ↔ `tuned` to watch the
  base model overclaim. Local generation falls back to a hosted model past a
  timeout (marked `generator: "hosted-fallback"`).

**This Space is the API only.** The chat widget is a separate Vercel app.

| Endpoint | |
|---|---|
| `POST /api/ask` | SSE stream: `token` / `sources` / `claims` / `meta` / `done` |
| `POST /api/ask/sync` | same, one JSON response |
| `POST /api/feedback` | 👍/👎 on an answer |
| `GET /api/projects` | project list for the widget's chips |
| `GET /api/health` | models loaded, corpus version, index size |
| `GET /docs` | OpenAPI |

**Cold start**: the embedder, NLI model and GGUF(s) load once at startup (~20 s
on a dev box; longer on a 2 vCPU Space). `GET /api/health/live` answers
immediately; `GET /api/health` reports `starting → degraded → ok`.

Source, architecture, and evaluation: <https://github.com/DEMONKINGKAI/ad-fontes>
