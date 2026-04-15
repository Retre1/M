#!/usr/bin/env bash
# One-shot server launcher for ApexFX Curriculum v2 training.
#
# Sequence:
#   1) Pre-flight env check (CUDA, deps, data)
#   2) Cache features for selected symbol (skip if already cached)
#   3) Run smoke test (~2 stages × 2k steps) to verify pipeline
#   4) Launch full training under tmux/nohup
#
# Usage:
#   bash scripts/run_server.sh                          # EURUSD H1 full
#   SYMBOL=GBPUSD bash scripts/run_server.sh
#   MODE=smoke bash scripts/run_server.sh               # smoke only
#   MODE=cache bash scripts/run_server.sh               # cache features only

set -euo pipefail

SYMBOL="${SYMBOL:-EURUSD}"
TIMEFRAME="${TIMEFRAME:-H1}"
MODE="${MODE:-full}"               # full | smoke | cache | check
PYTHON="${PYTHON:-.venv/bin/python}"
TB_DIR="${TB_DIR:-runs/v2_${SYMBOL}_${TIMEFRAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-models/v2_checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"
TS=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOG_DIR" "$TB_DIR" "$CHECKPOINT_DIR"

echo "── ApexFX v2 server launcher ─────────────────────────────"
echo "  symbol     : $SYMBOL $TIMEFRAME"
echo "  mode       : $MODE"
echo "  python     : $PYTHON"
echo "  tb log dir : $TB_DIR"
echo "  checkpoints: $CHECKPOINT_DIR"
echo "──────────────────────────────────────────────────────────"

# 1) Pre-flight
echo "[1/4] Env check"
$PYTHON scripts/env_check.py --symbol "$SYMBOL" --timeframe "$TIMEFRAME"

if [ "$MODE" = "check" ]; then
  exit 0
fi

# 2) Cache features (idempotent)
CACHE_PATH="data/cache/features/$SYMBOL/$TIMEFRAME/features.parquet"
if [ ! -f "$CACHE_PATH" ]; then
  echo "[2/4] Caching features → $CACHE_PATH"
  $PYTHON scripts/cache_features_v2.py --symbol "$SYMBOL" --timeframe "$TIMEFRAME"
else
  echo "[2/4] Features cache exists, skipping"
fi

if [ "$MODE" = "cache" ]; then
  exit 0
fi

# 3) Smoke
echo "[3/4] Smoke test"
$PYTHON scripts/train_v2.py --smoke --symbol "$SYMBOL" --timeframe "$TIMEFRAME" \
  2>&1 | tee "$LOG_DIR/smoke_${TS}.log"

if [ "$MODE" = "smoke" ]; then
  exit 0
fi

# 4) Full training (under nohup so it survives SSH disconnects)
echo "[4/4] Launching full training (nohup, PID stored in $LOG_DIR/train.pid)"
LOG_FILE="$LOG_DIR/train_${SYMBOL}_${TS}.log"
nohup $PYTHON scripts/train_v2.py \
  --symbol "$SYMBOL" --timeframe "$TIMEFRAME" \
  --tb-log-dir "$TB_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  > "$LOG_FILE" 2>&1 &
echo $! > "$LOG_DIR/train.pid"
echo
echo "Training PID: $(cat $LOG_DIR/train.pid)"
echo "Log file    : $LOG_FILE"
echo "TensorBoard : tensorboard --logdir $TB_DIR --bind_all"
echo "Tail log    : tail -f $LOG_FILE"
