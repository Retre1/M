# Run #1 — Baseline (TradingReward)

**Date:** April 2026
**Server:** 212.41.28.164, RTX 4090 24GB
**Config:** `gpu1.yaml` + `training.yaml` (original)

## Setup

- Reward: `TradingReward` — 10 компонент (Sharpe, drawdown, position cost, etc.)
- Algorithm: TQC (Truncated Quantile Critics)
- Curriculum: Stage 1 `real_warmup` only (500K steps target)
- Features: 15 selected from 99 via `feature_selector.json`
- MTF: D1(20 lookback) + H1(60) + M5(30)

## Results

| Metric | Value |
|--------|-------|
| Total return | -1.22% |
| Win rate | 38.55% |
| Profit factor | 0.67 |
| Total trades | 22 / 3550 bars |
| Max drawdown | ~2% |

## Diagnosis

**Проблема:** Agent выучил "ничего не делать" — 22 trades на 3550 bars = 1 trade / 161 bars.

**Root cause:** 10-компонентный reward создавал contradictory gradients:
- Sharpe penalty толкал к hold
- Position cost penalty толкал к exit
- Drawdown penalty толкал к reduce size
- Net effect: optimal policy = flat

## Артефакты

- Checkpoints: `/Users/abobik/Desktop/ApexFX_Export/apexfx_export/checkpoints/`
- Logs: `/Users/abobik/Desktop/ApexFX_Export/apexfx_export/logs/`

## Action Taken

→ Заменили TradingReward на [[decisions/reward-v4-profit-focused|ProfitFocusedReward v4]] (4 компонента)

#run #baseline #failed
