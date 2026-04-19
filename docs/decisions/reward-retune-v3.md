# Decision: Reward Retune for Run #3

**Date:** April 2026
**Status:** Planned

## Problem

Run #2 plateau at -63.9 — agent overtrades (550 trades/ep) without profitable edge.

## Proposed Changes

| Parameter | v4 (Run #2) | v4-retuned (Run #3) | Why |
|-----------|------------|---------------------|-----|
| `inactivity_penalty` | 0.001 | 0.0002 | Снижаем давление к торговле — пусть ждёт хорошие сетапы |
| `trade_cost` | 0.05 | 0.20 | 4x дороже → каждый trade должен оправдывать себя |
| `loss_asymmetry` | 1.2 | 1.6 | Убыток в 1.6x больнее прибыли → учит cut losses |
| `realized_pnl_weight` | 5000 | 10000 | Усиливаем доминирующий сигнал |

## Expected Effect

- Trades/episode: 550 → 50-100 (меньше, но осмысленнее)
- Agent должен научиться ждать edge перед входом
- Losses отрезаются быстрее из-за asymmetry 1.6

## Risk

- Слишком высокий trade_cost → agent может вернуться к "do nothing"
- Мониторить n_trades в первые 50K steps

## Validation

- Если trades < 10/ep после 100K → снизить trade_cost до 0.10
- Если trades > 200/ep → поднять trade_cost до 0.30

#decision #reward #planned
