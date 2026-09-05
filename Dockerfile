# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ad fontes — serving image. Multi-stage:
#   builder  : install deps into a venv, (Phase 1-2) bake corpus index + models
#   runtime  : CPU-only, non-root, models loaded once at startup
# Target host: a free Hugging Face Space (Docker SDK), 2 vCPU / 16 GB, cold starts.
# ---------------------------------------------------------------------------

# ----------------------------- builder -------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# llama-cpp-python builds from source: needs a C/C++ toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV"

# Dependency layer first for cache reuse.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-deps .

# Corpus is baked into the image (brief §2: no live corpus dependency).
COPY data/corpus ./data/corpus

# --- Phase 1-2: warm models + build the Chroma index at build time ---------
# Enabled once ingestion (Phase 1) and the generators/NLI (Phase 2) land. Kept
# here so the build contract is visible now.
#   ARG HF_TOKEN
#   RUN python -m scripts.download_models --embed --nli --base-gguf --tuned-gguf
#   RUN python -m app.ingestion.cli --rebuild
# Until then the index builds on first boot and models download at startup.

# ----------------------------- runtime ------------------------------------
FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/models/hf \
    AD_FONTES_PORT=7860 \
    AD_FONTES_INDEX_DIR=/app/data/index \
    AD_FONTES_FEEDBACK_DIR=/app/data/feedback

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/data/corpus ./data/corpus
COPY app ./app
COPY pyproject.toml README.md ./

RUN mkdir -p /app/data/index /app/data/feedback /app/data/models \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${AD_FONTES_PORT}/api/health/live" || exit 1

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${AD_FONTES_PORT}"]
