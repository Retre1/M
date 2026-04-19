# FSD — Functional Skewness Dispersion

**Paper:** "Functional Skewness Dispersion and the Cross-Section of Stock Returns"

## Core Idea

Измеряем дисперсию скоса (skewness) по cross-section фичей за скользящее окно. Высокая дисперсия = переход между режимами / стресс. Низкая = стабильный рынок.

## Формулы

```
skew_i(t) = E[(r_i - mu_i)^3] / sigma_i^3   (rolling window W)
FSD(t) = std(skew_1(t), ..., skew_N(t))       (cross-sectional)
```

## Адаптация для Forex

- Вместо акций — cross-section фичей одной пары (RSI, MACD, ATR, etc.)
- Или cross-section нескольких пар (EURUSD, GBPUSD, USDJPY)
- Window = 252 bars (1 год H1 ≈ 252 торговых дня × ~bars/day)

## Режимы

| FSD quantile | Regime | Trading implication |
|-------------|--------|-------------------|
| < 30th pct | RISK_ON | Trend-following profitable |
| 30-70th pct | NEUTRAL | Mixed strategies |
| > 70th pct | RISK_OFF | Mean-reversion / reduce size |

## Как используется

- 4 фичи в obs space: `[dispersion_value, onehot_0, onehot_1, onehot_2]`
- Reward v5 модулирует веса по режиму (risk_off → heavier loss penalty)
- Gating network может учитывать regime при выборе агента

## Реализация

- Файл: `src/apexfx/data/fsd_regime.py`
- Класс: `FSDRegimeDetector`
- Config: `configs/gpu1.yaml → fsd.enabled, fsd.window, fsd.n_quantiles`

## Open Questions

- [ ] Оптимальный window для H1 forex? 252 может быть слишком много
- [ ] Cross-section фичей vs cross-section пар — что даёт лучший сигнал?
- [ ] Нужен ли adaptive threshold вместо фиксированного 30/70?

#research #fsd #regime-detection
