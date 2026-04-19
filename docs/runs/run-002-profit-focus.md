# Run #2 — ProfitFocusedReward v4

**Date:** April 2026
**Server:** 212.41.28.164, RTX 4090 24GB
**Config:** `gpu1.yaml` v2 + ProfitFocusedReward

## Changes from Run #1

- Reward: 10-component → 4-component (realized PnL dominant)
- `realized_pnl_weight`: 5000
- `unrealized_delta_weight`: 500
- `trade_cost`: 0.05
- `inactivity_penalty`: 0.001
- `loss_asymmetry`: 1.2
- `ent_coef`: 0.02 → 0.05
- `learning_starts`: 5K → 15K
- `batch_size`: 256 → 384
- `buffer_size`: 200K → 400K
- `gradient_steps`: 1 → 2
- D1 lookback: 20→15, H1: 60→40

## Progress

- **0–100K:** Learning starts, reward climbing from -80 → -65
- **100K–200K:** Reward -65 → -63.9, slow improvement
- **200K–300K:** **Plateau** at -63.9 ± 0.5

```
ep_rew_mean: -63.9, std: 3.0
critic_loss: falling (confident)
actor_loss: rising (stuck)
trades/episode: ~550 (overtrading)
```

## Diagnosis

**Проблема:** Local minimum — agent overtrades (550 trades/ep) without edge.

**Причина:**
- `trade_cost = 0.05` слишком дешёвый → нет penalty за churning
- `inactivity_penalty = 0.001` слишком жёсткий → толкает в random trades
- `loss_asymmetry = 1.2` слабый → не учит избегать убытков

**Gradient penalty warning:** inplace operation в adversarial.py (non-blocking)

## Visualization

- `artifacts/trade_viz_step200k_stoch.html` — stochastic rollout, 401 trades
- Deterministic rollout: 0 trades (mean action in dead zone)

## Planned Retune for Run #3

See [[decisions/reward-retune-v3]]

#run #profit-focus #plateau #local-minimum
