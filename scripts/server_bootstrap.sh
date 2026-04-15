#!/usr/bin/env bash
# Bootstrap a fresh Linux/CUDA server for ApexFX v2 training.
#
# Expects: Python 3.11+ and CUDA drivers already installed.
# Creates venv, installs deps (with GPU torch), runs env check.
#
# Usage (from the repo root on the server):
#   bash scripts/server_bootstrap.sh
#   bash scripts/server_bootstrap.sh cu121          # specify CUDA version

set -euo pipefail

CUDA_SUFFIX="${1:-cu121}"   # cu118 | cu121 | cu124 | cpu
PY="${PYTHON:-python3.11}"

echo "── ApexFX server bootstrap ──"
echo "  python     : $PY"
echo "  cuda wheels: $CUDA_SUFFIX"
echo

# 1) venv
if [ ! -d ".venv" ]; then
  echo "[1/4] Creating .venv"
  "$PY" -m venv .venv
else
  echo "[1/4] .venv already exists"
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 2) Torch with CUDA
echo "[2/4] Installing torch ($CUDA_SUFFIX)"
if [ "$CUDA_SUFFIX" = "cpu" ]; then
  pip install torch torchvision
else
  pip install --index-url "https://download.pytorch.org/whl/$CUDA_SUFFIX" \
    torch torchvision
fi

# 3) Project + training extras
echo "[3/4] Installing apexfx + deps"
pip install -e ".[dev]"

# 4) Pre-flight
echo "[4/4] Env check"
python scripts/env_check.py || {
  echo "Env check failed — inspect output above"
  exit 1
}

echo
echo "Bootstrap complete. Next:"
echo "  bash scripts/run_server.sh                 # full pipeline"
echo "  MODE=smoke bash scripts/run_server.sh      # smoke only"
