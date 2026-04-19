# Decision: ProfitFocusedReward v4

**Date:** April 2026
**Status:** Implemented, needs retune

## Context

Run #1 с TradingReward (10 компонент) провалился — agent ушёл в "do nothing" из-за противоречивых градиентов.

## Decision

Заменить 10-компонентный reward на 4-компонентный с доминирующим realized PnL:

```python
reward = realized_pnl_weight * pnl        # 5000 — dominating
       + unrealized_delta_weight * delta   # 500
       - trade_cost * did_trade            # 0.05
       - inactivity_penalty * is_flat      # 0.001
```

`loss_asymmetry = 1.2` — штраф за убыток чуть выше бонуса за прибыль.

## Rationale

- Один доминирующий сигнал (PnL) → чистый градиент
- Inactivity penalty → agent не может "сидеть"
- Trade cost → penalty за churning
- Loss asymmetry → обучение risk aversion

## Result

Частично успешно — agent стал торговать (550 trades/ep vs 22), но застрял в local min (-63.9 plateau). Нужна [[reward-retune-v3|ретюна]].

#decision #reward
