# Decision: TQC over PPO

**Date:** April 2026
**Status:** Active

## Context

Выбор RL-алгоритма для торгового бота.

## Decision

TQC (Truncated Quantile Critics) из sb3-contrib.

## Rationale

| Factor | PPO | TQC |
|--------|-----|-----|
| Sample efficiency | Low (on-policy) | High (off-policy, replay buffer) |
| Continuous actions | Ok | Excellent (designed for it) |
| Overestimation | N/A | Fixed via truncated quantiles |
| Distributional RL | No | Yes (risk-aware by design) |
| Multi-agent | Needs tricks | Natural with shared replay |

TQC = SAC + distributional critics, отсекает верхние квантили → меньше overestimation → лучше для финансов где overconfidence = потеря денег.

## Trade-offs

- Более сложный debug (critic loss != actor loss)
- Нужен больший buffer_size для off-policy (400K)
- Медленнее per-step (quantile regression)

#decision #algorithm #active
