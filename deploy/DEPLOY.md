# Deploying to a Hugging Face Space (Docker SDK)

The Space runs the `runtime` stage of the repo `Dockerfile`. Corpus, Chroma
index, embedder, NLI model and the base GGUF are baked at build time; the tuned
GGUF is pulled if its repo exists.

## 1. One-time: make the tuned GGUF repo readable at build

Either make `DEMONKINGKAI/ad-fontes-generator-1.5b-dpo-gguf` **public**, or pass
`HF_TOKEN` as a **build secret** in the Space settings (the Dockerfile already
declares `--mount=type=secret,id=HF_TOKEN`). Until the tuned GGUF exists the
build step is a harmless no-op and `model: "tuned"` requests serve `base`.

## 2. Create the Space

On huggingface.co → New Space → **Docker** (blank) → name it `ad-fontes`.
Overwrite the generated `README.md` with [`deploy/README.md`](README.md) (it has
the `sdk: docker` / `app_port: 7860` front-matter the Space needs).

## 3. Secrets (Space → Settings → Variables and secrets)

| name | value | required |
|---|---|---|
| `HF_TOKEN` | a HF token (Pro recommended) | for the hosted fallback + judge |
| `CORS_ORIGINS` | `https://<your-portfolio>.vercel.app` (comma-separated for more) | yes |
| `AD_FONTES_DEFAULT_MODEL` | `base` until the tuned GGUF is uploaded, then `tuned` | optional |
| `AD_FONTES_FEEDBACK_DATASET` | `DEMONKINGKAI/ad-fontes-feedback` (a private Dataset repo) | optional — the Space FS is ephemeral, so feedback is otherwise lost |
| `AD_FONTES_LOCAL_TIMEOUT_S` | `35`–`60` — a 2 vCPU Space is slower than a laptop | optional |

## 4. Push

```bash
scripts/prepare_space.sh <hf-username>/ad-fontes    # sets up the 'space' remote + branch
git push space space:main
```

or manually: `git remote add space https://huggingface.co/spaces/<user>/ad-fontes`
then push a tree whose root `README.md` is `deploy/README.md`.

The build takes **~15–25 min** (torch, the prebuilt `llama-cpp-python` CPU wheel,
model bakes). Watch the build logs in the Space.

## 5. Verify

```bash
scripts/smoke_test.sh https://<user>-ad-fontes.hf.space
```

`GET /api/health` should reach `"status": "ok"` once the models finish loading
(a minute or so after the container starts).

## 6. Point the widget at it

In the Vercel project: `VITE_API_BASE_URL=https://<user>-ad-fontes.hf.space`.
The widget consumes `POST /api/ask` as SSE — exactly what `scripts/smoke_test.sh`
and `scripts/smoke_sse.mjs` exercise.

## Notes

- **Single worker** — the in-memory rate limiter and the llama.cpp lock are
  per-process (`--workers 1` in the Dockerfile `CMD`). Don't scale horizontally
  without moving those to a shared store.
- **Ephemeral disk** — a rebuild or a Space restart wipes `data/feedback/` and
  any runtime state. Set `AD_FONTES_FEEDBACK_DATASET` to keep feedback.
- **Free-tier sleep** — the Space sleeps after inactivity; the next request pays
  the cold start again.
- Local parity: `docker compose up --build` runs the same image on `:8000`.
