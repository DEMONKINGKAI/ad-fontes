# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ad fontes — serving image. Multi-stage:
#   builder  : install deps into a venv, bake the corpus + Chroma index +
#              embedding model into the image
#   runtime  : CPU-only, non-root, everything loaded once at startup
# Target host: a free Hugging Face Space (Docker SDK), 2 vCPU / 16 GB, cold starts.
# ---------------------------------------------------------------------------

# ----------------------------- builder -------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/build/hfcache \
    AD_FONTES_CORPUS_DIR=/build/data/corpus \
    AD_FONTES_INDEX_DIR=/build/data/index

# llama-cpp-python builds from source: needs a C/C++ toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"
WORKDIR /build

# Dependency layers first for cache reuse.
COPY requirements.txt ./
# CPU-only torch in its own layer (~190 MB, not the ~2.8 GB CUDA build with its
# nvidia-* wheels a CPU image never loads). Install it explicitly from the PyTorch
# CPU index so `torch` in requirements.txt is already satisfied and pip never
# reaches for the PyPI CUDA wheel. Keep the range in sync with requirements.txt.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
# The rest. The llama-cpp extra index has prebuilt CPU wheels (no source compile);
# requirements.txt also carries the PyTorch CPU index as a backstop.
RUN pip install -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-deps .

# Corpus + index baked into the image (brief §2: no live dependency, warm at build).
COPY data/corpus ./data/corpus
COPY scripts ./scripts
RUN python -m app.ingestion.cli --rebuild && python -m app.ingestion.cli --stats

# Bake every model: embedder (also warmed by --rebuild), NLI, base GGUF, and the
# tuned GGUF once it exists (a no-op warning until then). `HF_TOKEN` as a build
# secret only if a source repo is private — the tuned GGUF repo can be public.
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || true) \
    python -m scripts.download_models --embed --nli --base-gguf --tuned-gguf

# ----------------------------- runtime ------------------------------------
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    AD_FONTES_PORT=7860 \
    AD_FONTES_INDEX_DIR=/app/data/index \
    AD_FONTES_FEEDBACK_DIR=/app/data/feedback

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/data/corpus ./data/corpus
COPY --from=builder /build/data/index ./data/index
COPY --from=builder /build/hfcache ./data/models/hf
COPY app ./app
COPY pyproject.toml README.md ./

RUN mkdir -p /app/data/feedback \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 7860

# Models are baked, so startup needs no downloads — just a fast cached-etag check
# per model (set HF_HUB_OFFLINE=1 as a Space secret to skip even that). Cold start
# on a 2 vCPU Space is ~40-70 s (embedder + NLI + one/two GGUFs load once).
# /api/health/live answers immediately; /api/health reports starting -> degraded -> ok.
HEALTHCHECK --interval=30s --timeout=5s --start-period=150s --retries=4 \
    CMD curl -fsS "http://localhost:${AD_FONTES_PORT}/api/health/live" || exit 1

# Single worker: the rate limiter and the llama.cpp lock are per-process.
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${AD_FONTES_PORT} --workers 1"]
