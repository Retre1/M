# SBBTS — Schrodinger-Bass Synthetic Time Series

**Paper:** "Schrodinger-Bass model for synthetic financial time series"
**Reference impl:** https://github.com/alexouadi/SBBTS

## Core Idea

Генерация синтетических финансовых временных рядов, которые:
1. Сохраняют статистические свойства реальных данных (fat tails, volatility clustering, skewness)
2. Позволяют augment training data для RL
3. Создают stress-сценарии, которых нет в истории

## Модель

Комбинация трёх компонент:

### 1. Schrodinger Wave Equation (цена)
```
dψ/dt = -iHψ
```
Двухямный потенциал V(x) = -depth*(x^2-1)^2 моделирует два режима (bull/bear).

### 2. Bass Diffusion (drift modulation)
```
F(t) = (1 - exp(-(p+q)*t)) / (1 + (q/p)*exp(-(p+q)*t))
```
Моделирует "adoption" тренда — как быстро рынок переходит в новый режим.

### 3. Stochastic Volatility (Heston-like)
```
dv = κ(θ - v)dt + ξ√v dW_v
```
Волатильность кластеризуется, что критично для forex.

## Параметры для Forex

| Param | EURUSD | GBPUSD | USDJPY |
|-------|--------|--------|--------|
| dt | 1/6240 (H1) | 1/6240 | 1/6240 |
| bass_p | 0.03 | 0.04 | 0.02 |
| bass_q | 0.38 | 0.35 | 0.40 |
| vol_of_vol | 0.3 | 0.4 | 0.25 |
| mean_rev | 2.0 | 2.5 | 1.5 |

## Validation

Синтетика валидна если:
- [ ] KL divergence маргиналов < 0.05
- [ ] Autocorrelation returns ≈ 0 (no serial correlation)
- [ ] Autocorrelation |returns| > 0 (volatility clustering preserved)
- [ ] Kurtosis в пределах ±20% от реальных данных
- [ ] Hurst exponent в пределах ±0.1

## Как используется

- `sbbts.ratio = 0.3` — 30% синтетики подмешивается в training data
- Curriculum stage `real_augmented` заменяется на `sbbts_augmented`
- Разные β для разных стадий (мягче → жёстче)

## Реализация

- Файл: `src/apexfx/data/sbbts_generator.py`
- Класс: `SBBTSGenerator`
- Конфиг: `configs/gpu1.yaml → sbbts.*`

#research #synthetic-data #augmentation
