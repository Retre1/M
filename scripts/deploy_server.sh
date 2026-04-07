#!/bin/bash
# =============================================================================
# ApexFX Quantum — GPU Server Deploy Script
#
# Quick setup on a fresh GPU server (Ubuntu + NVIDIA drivers)
#
# Usage:
#   # On your Mac — push to git, then on server:
#   git clone <your-repo> apexfx && cd apexfx
#   chmod +x scripts/deploy_server.sh
#   ./scripts/deploy_server.sh
#
# Supported servers:
#   - Vast.ai (cheapest GPUs, RTX 4090 ~$0.30/hr)
#   - RunPod   (RTX 4090 ~$0.40/hr, A100 ~$1.50/hr)
#   - Lambda   (A100 $1.10/hr, H100 $2.50/hr)
#   - AWS p4d  (A100), GCP a2 (A100)
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  ApexFX Quantum — GPU Server Setup"
echo "=============================================="

# ── 1. Check prerequisites ──
echo ""
echo "[1/6] Checking prerequisites..."

if ! command -v docker &>/dev/null; then
    echo "  Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  Docker installed. You may need to re-login for group changes."
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "  ERROR: NVIDIA drivers not found. Install them first:"
    echo "    sudo apt install nvidia-driver-550"
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "  Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "  NVIDIA Container Toolkit installed."
fi

echo "  GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'

# ── 2. Build Docker image ──
echo ""
echo "[2/6] Building Docker image (first time takes ~10 min)..."
docker compose build train

# ── 3. Download market data ──
echo ""
echo "[3/6] Downloading market data..."
docker compose run --rm download-data

# ── 4. Pretrain TFT encoder ──
echo ""
echo "[4/6] Pre-training TFT encoder on real data..."
docker compose run --rm pretrain

# ── 5. Run hyperparameter optimization ──
echo ""
echo "[5/6] Running hyperparameter optimization (40 trials)..."
docker compose run --rm hyperopt

# ── 6. Full curriculum training ──
echo ""
echo "[6/6] Starting full curriculum RL training (5M timesteps)..."
echo "  TensorBoard: docker compose up -d tensorboard"
echo "  Monitor:     docker compose logs -f train"
echo ""
docker compose up train

echo ""
echo "=============================================="
echo "  Training complete!"
echo "  Models saved in: ./models/"
echo "  Logs in: ./logs/"
echo ""
echo "  Next steps:"
echo "    docker compose up backtest    # Run backtest"
echo "    docker compose up dashboard   # View results"
echo "=============================================="
