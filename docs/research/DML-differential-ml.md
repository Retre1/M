# DML — Differential Machine Learning

**Paper:** "Differential Machine Learning: 0DTE pricing with stochastic volatility and jumps"

## Core Idea

Нейронная сеть учит одновременно:
- **f(x)** — значение функции (цена опциона / value function)
- **df/dx** — производные (Greeks / sensitivities)

Дифференциальная регуляризация:
```
L = MSE(f(x), y) + lambda * MSE(df/dx, dy)
```

## Зачем для Forex Trading Bot

1. **World model** — предсказывает не только цену, но и чувствительности
2. **Risk management** — встроенные Greeks для position sizing
3. **Sample efficiency** — differential регуляризация даёт 2-5x лучшую обобщаемость

## Merton Jump-Diffusion

```
dS/S = (mu - lambda*k)dt + sigma*dW + J*dN
```

- J ~ LogNormal(jump_mean, jump_std) — размер прыжка
- N ~ Poisson(lambda) — процесс прыжков
- Моделирует внезапные движения (news, interventions)

## Адаптация для ApexFX

- DML Network как альтернативный world model (замена текущего ensemble)
- Вход: observation (market features + position state)
- Выход: predicted return + predicted volatility
- Derivatives: sensitivity к каждому входному feature

## Реализация

- Файл: `src/apexfx/models/dml_network.py`
- Классы: `DMLNetwork`, `MertonJumpDiffusion`
- Phase 4 в roadmap

## Open Questions

- [ ] Как интегрировать с TQC? Как auxiliary loss или отдельная голова?
- [ ] Training schedule — DML до RL или jointly?
- [ ] Нужен ли differential loss для critic network?

#research #dml #world-model #greeks
