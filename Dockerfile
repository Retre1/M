# =============================================================================
# ApexFX Quantum — Production Dockerfile
# Multi-stage build: Python 3.11 + CUDA 12 (optional GPU)
# =============================================================================

# ---------- Stage 1: Base with system deps ----------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Stage 2: Dependencies ----------
FROM base AS deps

COPY pyproject.toml ./
# Install core deps first (cached layer)
RUN pip install --no-cache-dir -e ".[dashboard,nlp]" 2>/dev/null || \
    pip install --no-cache-dir \
    torch>=2.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[dashboard,nlp]"

# ---------- Stage 3: Application ----------
FROM deps AS app

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY Makefile ./

# Create runtime directories
RUN mkdir -p data/raw data/processed data/synthetic \
    models/checkpoints models/best \
    logs

# Non-root user for security
RUN groupadd -r apexfx && useradd -r -g apexfx -d /app apexfx \
    && chown -R apexfx:apexfx /app
USER apexfx

# Health check — verifies trading loop state file is fresh (updated within 5 min)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "\
import json, time; \
from pathlib import Path; \
p = Path('data/portfolio_state.json'); \
assert p.exists(), 'No state file'; \
age = time.time() - p.stat().st_mtime; \
assert age < 300, f'State file stale ({age:.0f}s)'; \
print('ok')" || exit 1

EXPOSE 8050

# Default: run live trading
CMD ["python3", "scripts/live_trade.py"]

# =============================================================================
# GPU variant — build with: docker build --target gpu -t apexfx:gpu .
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir torch>=2.0 \
    && pip install --no-cache-dir -e ".[dashboard,nlp]"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY Makefile ./

RUN mkdir -p data/raw data/processed models/checkpoints models/best logs

RUN groupadd -r apexfx && useradd -r -g apexfx -d /app apexfx \
    && chown -R apexfx:apexfx /app
USER apexfx

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD python3 -c "\
import json, time; \
from pathlib import Path; \
p = Path('data/portfolio_state.json'); \
assert p.exists(), 'No state file'; \
age = time.time() - p.stat().st_mtime; \
assert age < 300, f'State file stale ({age:.0f}s)'; \
print('ok')" || exit 1

EXPOSE 8050
CMD ["python3", "scripts/live_trade.py"]
