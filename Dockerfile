# =============================================================================
# ApexFX Quantum — Multi-purpose Docker Image
#
# Targets:
#   default (app)  — GPU training & hyperopt & backtest
#   live           — Live trading with health checks
#
# Build:
#   docker build -t apexfx .
#   docker compose up train          # full training pipeline
#   docker compose up hyperopt       # hyperparameter search
#   docker compose up backtest       # backtest on real data
# =============================================================================

# ---------- Stage 1: GPU base with CUDA 12.4 + Python 3.11 ----------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ---------- Stage 2: Dependencies (cached layer) ----------
FROM base AS deps

# PyTorch with CUDA — largest dep, cached separately
RUN pip install --no-cache-dir \
    torch>=2.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Project deps — install from pyproject.toml requirements directly
RUN pip install --no-cache-dir \
    "gymnasium>=0.29" \
    "stable-baselines3>=2.0" \
    "pandas>=2.0" \
    "numpy>=1.24" \
    "scipy>=1.11" \
    "scikit-learn>=1.3" \
    "PyWavelets>=1.4" \
    "pyarrow>=14.0" \
    "pydantic>=2.0" \
    "pyyaml>=6.0" \
    "structlog>=23.0" \
    "requests>=2.31" \
    "beautifulsoup4>=4.12" \
    "sb3-contrib>=2.0" \
    "yfinance>=0.2" \
    "rich>=13.0" \
    "optuna>=3.4" \
    "tensorboard>=2.15" \
    "pytest>=7.4"

# ---------- Stage 3: Application ----------
FROM deps AS app

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Install package in editable mode (src/ now available)
RUN pip install --no-cache-dir -e .

# Create runtime directories
RUN mkdir -p data/raw/bars data/cache \
    models/pretrained models/checkpoints models/best \
    logs

# Non-root user
RUN groupadd -r apexfx && useradd -r -g apexfx -d /app apexfx \
    && chown -R apexfx:apexfx /app
USER apexfx

# Persistent volumes
VOLUME ["/app/data", "/app/models", "/app/logs"]

# TensorBoard port
EXPOSE 6006

# Default: show help
CMD ["python", "-c", "print('Use docker compose: up train | up hyperopt | up backtest')"]

# ---------- Stage 4: Live trading (optional target) ----------
FROM app AS live

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD python3 -c "\
import json, time; \
from pathlib import Path; \
p = Path('data/portfolio_state.json'); \
assert p.exists(), 'No state file'; \
age = time.time() - p.stat().st_mtime; \
assert age < 300, f'State file stale ({age:.0f}s)'; \
print('ok')" || exit 1

CMD ["python", "scripts/live_trade.py"]
