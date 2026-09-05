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

# Dependency layer first for cache reuse.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-deps .

# Corpus + index baked into the image (brief §2: no live dependency, warm at build).
COPY data/corpus ./data/corpus
COPY scripts ./scripts
RUN python -m app.ingestion.cli --rebuild && python -m app.ingestion.cli --stats

# Warm the NLI model + the base GGUF into the image. The tuned GGUF is pulled too
# once it exists (Phase 4); until then that step is a no-op warning.
RUN python -m scripts.download_models --nli --base-gguf --tuned-gguf

# ----------------------------- runtime ------------------------------------
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/models/hf \
    HF_HUB_OFFLINE=1 \
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

# HF_HUB_OFFLINE=1 keeps startup from reaching the network for the baked model.
# Unset it (compose / Space secret) if you point at a model that isn't baked.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://localhost:${AD_FONTES_PORT}/api/health/live" || exit 1

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${AD_FONTES_PORT}"]
