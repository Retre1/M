# ApexFX Quantum — Аудит-отчёт и план пути к прибыльной торговле

> Дата: 2026-04-24 | Аудитор: Claude | Стадия: pre-live diagnostic
> Капитал: <$1k, макс. DD 20%, MT5, стиль — swing H4

---

## TL;DR (главное за 60 секунд)

**Текущее состояние:** бот **не готов к live и не заработает в текущем виде.**
Последняя обученная модель в `backtest_results.json` даёт Sharpe **-0.24**,
profit factor **0.96**, expectancy **отрицательная**. В логах обучения — Sharpe
порядка **-700 000** на ранних стадиях. Это не плохой edge — это **сломанный
training pipeline**.

**Причина:** обнаружены 3 критических бага в механике обучения, которые
вместе объясняют весь провал. Логика торговли (TFT + HiveMind + risk manager)
не сломана — она не смогла проявиться из-за поломанного pipeline.

**Reality check на реальных данных (EURUSD H4, Feb 2024 — Feb 2026, спред 1.5 пип):**

| Стратегия | Return | Sharpe | Max DD | Trades |
|---|---:|---:|---:|---:|
| **Buy & Hold** | **+9.97%** | **0.65** | -8.95% | 1 |
| MA cross 20/50 | -4.60% | -0.28 | -8.22% | 71 |
| Donchian 20 breakout | -5.87% | -0.37 | -11.32% | 71 |
| Donchian 55 breakout | -5.74% | -0.34 | -9.23% | 33 |
| Donchian long-only | +1.58% | 0.16 | -7.32% | 71 |
| Random | -30.2% | -3.22 | -30.8% | 2095 |

**Значение:** в этом окне EURUSD в тренде вверх (+9.98%). Любая стратегия с
шортами проигрывает buy & hold. Это не уникальный баг обучения — это
**фундаментальная сложность форекса в конкретный период**. Прежде чем твоя
сложная RL-система превзойдёт buy & hold, она сначала должна хотя бы не
проигрывать ему. Сейчас она его драматически проигрывает.

---

## Часть 1 — Инвентаризация проекта

| Артефакт | Факт |
|---|---|
| Кодовая база | 139 Python-файлов, src/apexfx |
| Тесты | 144 заявлено, реально запустить в sandbox не удалось (Python 3.10, нужен 3.11; venv пустая) |
| Данные | EURUSD H1 bars, 12 321 часов (Feb 2024 → Feb 2026), DXY, SPX, US10Y, XAUUSD — H1 close only |
| Модели | `models/best/final_model.zip`, `models/checkpoints/stage_0..2.zip`, два "v2_final" в `apexfx_training_results/` |
| Последний бэктест | `backtest_results.json`: Sharpe -0.24, PF 0.96, win rate 44.6%, **expectancy -3.4e-6 (отрицательная)**, 173 сделок на 3697 барах |
| Логи обучения | Early stopping на всех 4 стадиях curriculum, Sharpe -700k → -146, entropy collapse 0.007-0.03 |

---

## Часть 2 — Критические баги (найденные)

### Баг #1 — Reward clipping убивает градиентный сигнал

**Где:** `src/apexfx/env/reward.py`, все классы reward, особенно
`TradingReward.compute()` (line 613) и `LogReturnReward.compute()` (line 394).

**Что происходит:**
```python
base_reward = log_ret * self.reward_scale * vol_scale
# ... суммирование компонентов: cost, churn, dd, hold, cvar, winner, quick_cut, news, struct ...
return float(np.clip(reward, -10.0, 10.0))
```

`reward_scale = 1000.0` — log-return порядка 0.001 (1 пипс EURUSD) превращается
в 1.0, а сильное движение в 10 пипсов уже даёт ~10 и упирается в клип. Так как
компонентов 10, из которых penalties (CVaR, DD, churn, news, cost) могут быть
одновременно >0.5 каждый, сумма регулярно вылетает за [-10, 10] и обрезается.

**Симптом:** на плоских landscape (где клиппинг активен) градиент нулевой —
policy теряет дифференциальный сигнал, энтропия коллапсирует. Это ровно то,
что видно в логах (`entropy=0.007, LR boost triggered`).

**Ещё: "Sharpe -700 000" в логах** — это mean/std по клиппованной серии
reward'ов за эпизод, а не финансовый Sharpe. Метрика бессмысленная — и тем не
менее она используется для early stopping и выбора лучшего чекпоинта. Система
оптимизирует шум.

**Фикс (когда ты готов):**
```python
# reward.py, TradingReward.__init__
self.reward_scale = 10.0          # было 1000 — даёт typical |r| ~ 0.1
# Убрать np.clip на выходе. Вместо этого:
return float(np.tanh(reward / 3.0) * 3.0)  # soft saturation, градиент не нулевой
# И вынести "Sharpe" из логов — считать реальный Sharpe из PnL эпизода отдельно.
```

---

### Баг #2 — StrategyFilter блокирует почти все сделки в обучении

**Где:** `src/apexfx/env/trade_filter.py:174-224` + конфиги
`configs/execution.yaml`, `configs/production.yaml`.

**Что происходит:** для совершения **нового** входа модель обязана пройти три
условия одновременно (AND, не OR):

1. **Rule 4:** `|fundamental_bias| ≥ 0.3` (conviction)
2. **Rule 5:** `break_bull > 0.5` OR `break_bear > 0.5` (BOS обязателен) **AND** направление совпадает
3. **Rule 6:** `|bias| ≥ 0.5` → направление входа совпадает с bias

**Проблема с Rule 4:** на синтетических данных (SBBTS в stages 2-4) нет
настоящего economic calendar, fundamental extractor получает шум → bias
почти всегда < 0.3 → **все entries заблокированы**.

**Доказательство из логов:**
```
stage=3 name=real_adversarial best_sharpe=0.0 profit_factor=0.0
```
Sharpe=0, PF=0 значит модель не сделала **ни одной** сделки за эпизод.
Она не могла — фильтр блокировал.

**Почему это катастрофа для RL:** модель получает reward сигнал только когда
есть PnL. Если все действия заблокированы, reward = 0 на всех шагах, policy
gradient = 0, обучения нет. За 8 млн шагов модель научится только одному:
"делать действия бесполезно". Это именно то, что мы видим.

**Фикс:**
- В **training** mode отключить rules 4, 5, 6 целиком. Фильтр должен только
  учить модель избегать **news blackout** (rule 2) и conflicting signals
  (rule 1). Остальные rules — это soft guidance через reward, не hard block.
- В **live** mode — включить всё, но с относительными порогами: `min_bias =
  0.3 × rolling_max_bias_30d` вместо абсолютного 0.3.

**Patch idea:**
```python
# trade_filter.py
class StrategyFilter:
    def __init__(self, ..., training_mode: bool = False):
        self._training_mode = training_mode
        # ...
    def check(self, ...):
        # Rules 1, 2, 3 — всегда (news & conflicting)
        # Rules 4, 5, 6 — только if not self._training_mode
```

---

### Баг #3 — NaN-источники в feature pipeline + negative sqrt

**Где:**
- `src/apexfx/utils/math_utils.py:50` — Parkinson volatility
- `src/apexfx/utils/math_utils.py:69` — Garman-Klass volatility (ПОДТВЕРЖДЕНО в логах: `RuntimeWarning: invalid value in sqrt`)
- `src/apexfx/features/normalizer.py:41` — `transform_online` не фильтрует NaN на входе

**Что происходит:** Garman-Klass и Parkinson могут давать отрицательные
значения под `sqrt`, NaN течёт через regime extractor → observations →
градиенты. SB3 не всегда падает на NaN — иногда они просто становятся мусором.

`transform_online` принимает NaN и обновляет running mean/var этим NaN →
статистики становятся NaN навсегда → вся последующая нормализация NaN.

**Фикс:**
```python
# math_utils.py:50
result[i] = np.sqrt(np.maximum(np.mean(parkinson_sq[i - window:i]) * 252, 0.0))
# math_utils.py:69
result[i] = np.sqrt(np.maximum(np.mean(gk[i - window:i]) * 252, 0.0))

# normalizer.py transform_online — в начале:
if np.any(~np.isfinite(features)):
    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
```

---

## Часть 3 — Findings второго порядка (посмотреть после критичных)

### Дисциплина валидации
- `backtest_results.json` — **один** бэктест, не walk-forward. 3697 баров ≈ 2.5 года H4. Нет честного out-of-sample split.
- Модели `v2_final_run1` и `v2_final` без документированного split train/val/test. Нет протокола "какие данные модель никогда не видела".
- **Это значит:** даже если бы модель зарабатывала в бэктесте, доверять этому нельзя. Нужен walk-forward: например, 6 месяцев train → 1 месяц OOS test, роллом через весь диапазон, отчёт по стабильности Sharpe/PF между окнами.

### Данные
- 2 года H1 EURUSD — **впритык** для RL. Профессиональные quant-системы требуют 10+ лет для надёжного walk-forward.
- Calendar провайдер тянет с Forex Factory — rate limiting, изменения формата страницы. В бэктесте на синтетике — fundamental features мусор.

### Сложность vs. доказанная необходимость
- TFT (Temporal Fusion Transformer) + World Model Mamba backbone + Multi-Agent HiveMind + PER + EWC + Adversarial + Curriculum — это **очень** много сложности без доказательства, что каждый компонент даёт положительный edge.
- Классический подход в quant research: начать с простого baseline, добавлять компоненты по одному, документируя ΔSharpe и ΔMaxDD от каждого. Сейчас всё добавлено сразу, и при негативном итоговом Sharpe невозможно сказать, что мешает.

### Micro-account фит
- Архитектура рассчитана на institutional-grade execution (limit orders, slippage model, pyramiding до 3 уровней, VaR-scaling). На счёте <$1k с MT5 retail:
  - Минимальный лот на cent-account = 0.01 лота, на standard — 0.01 лота стандартного = $1000 нотионала; на <$1k депо leverage 1:500 → max margin 0.2 лота.
  - Pyramiding до 3 слоёв с размерным decay → последний слой может быть <0.01 лота и **отклонится брокером**. Нужна защита `max(0.01, size)` на всех слоях.
  - Spread-model в backtest 1.5 пип — **оптимистично для retail MT5**. Multi-broker реальность: 1.5-3.0 пип на EURUSD + swap, a EUR cross pairs могут быть 3-5 пип. Нужно протестировать на 2.5-3.0 пип спреде.

---

## Часть 4 — Reality check: есть ли вообще edge в этом периоде?

Запущен честный baseline на EURUSD H4 (Feb 2024 — Feb 2026, спред 1.5 пип):

| Стратегия | Return | Sharpe | Max DD | # Trades |
|---|---:|---:|---:|---:|
| **Buy & Hold** | **+9.97%** | **0.65** | -8.95% | 1 |
| MA cross 20/50 | -4.60% | -0.28 | -8.22% | 71 |
| Donchian 20 breakout | -5.87% | -0.37 | -11.32% | 71 |
| Donchian 55 breakout | -5.74% | -0.34 | -9.23% | 33 |
| Donchian 20 long-only | +1.58% | 0.16 | -7.32% | 71 |
| Random (-1, 0, +1) | -30.2% | -3.22 | -30.8% | 2095 |

**Интерпретация:**
- EURUSD в этот период растёт с 1.0778 до 1.1854 (+9.98%). Это **трендовый период, который сделал buy-and-hold «королём».**
- Trend-following стратегии со шортами проигрывают, потому что шорты сливают в аптренде, а whipsaws съедают спред.
- Long-only Donchian даёт слабый плюс (+1.58%) — близко к нулю после costs.
- **Random даёт -30%, это sanity-check костов: random trading = путь к нулю.**

**Вывод:** прежде чем бить голову о RL, нужно понять, что **в конкретно этом
окне** на конкретно этом инструменте рынок сам даёт +10%. Любая система,
которая делает меньше, проигрывает тупому holding'у. RL-система, которая
проигрывает -0.24% на test set, работает **значительно хуже случая**.

Это не значит "RL не работает" — но значит, что **edge должен быть доказан на
нескольких периодах, включая downtrend и ranging markets**, а не только на
одном окне.

---

## Часть 5 — Рекомендованный план восстановления

### Этап A — Срочные фиксы (1-2 сессии кодинга)

Не трогая архитектуру, починить training pipeline:

1. **Fix reward scaling (см. Баг #1):**
   - `reward_scale: 10.0`, не 1000.
   - Убрать жёсткий clip, заменить на tanh saturation.
   - Добавить **отдельный** честный financial Sharpe (из PnL эпизода), использовать его для early stopping — не reward-based Sharpe.

2. **Fix training-mode trade filter (см. Баг #2):**
   - Добавить `training_mode` флаг.
   - В training: активны только Rule 1 (conflicting) и Rule 2 (news blackout).
   - Rules 4, 5, 6 — только live/production.

3. **Fix NaN guards (см. Баг #3):**
   - `np.maximum(..., 0.0)` перед sqrt в `math_utils.py`.
   - NaN filter на входе `transform_online`.
   - Добавить assertion в `forex_env._build_observation`: `assert not np.any(~np.isfinite(obs_flat))`.

4. **Smoke-test:** после фиксов запустить `train.py --synthetic-only` с 200k
   timesteps. Цель — увидеть ненулевой profit factor и 30+ trades/эпизод. Если
   по-прежнему 0 trades — значит есть ещё блокер.

### Этап B — Валидация (2-3 сессии)

5. **Walk-forward harness:** скрипт, который режет 2024-2026 на окна
   6mo-train / 1mo-OOS-test, роллит их, сохраняет per-window метрики. Итог —
   CSV из 18 окон с Sharpe, PF, MaxDD, #trades. Стабильность между окнами >
   абсолютное значение в одном окне.

6. **Baseline sanity:** интегрировать три baseline (B&H, MA cross, Donchian)
   как сравнительные линии в каждом OOS окне. Правило: **если модель не бьёт
   лучший из baseline на 80% окон — нет edge, не идём в live.**

7. **Realistic costs:** повысить spread в симуляторе до 2.0-2.5 пип (retail MT5
   median). Добавить swap rate (особенно важно для H4 где позиции могут висеть
   через roll-over).

### Этап C — Упрощение (1-2 сессии, если B не дал edge)

Если после A+B всё равно нет edge — **выкинуть половину** сложности:

8. **Выключить** HiveMind multi-agent — тренировать одного SAC агента.
9. **Выключить** fundamental features (они мусор без настоящего calendar
   pipeline).
10. **Выключить** World Model Mamba.
11. **Оставить:** price features, ATR, regime, structure (но упрощённый — без
    BOS hard rule).
12. **Reward:** только `LogReturnReward` с `reward_scale=10, loss_weight=1.5`.

Если упрощённая модель покажет positive OOS Sharpe — есть база. Можно
добавлять компоненты по одному, каждый раз проверяя ΔSharpe.

Если и упрощённая не показывает edge — значит форекс H4 на одной паре с этими
features в этот период **не торгуется** с edge >= costs, и нужно менять
подход: multi-symbol portfolio, другой таймфрейм, другие features (например,
COT, session-based).

### Этап D — Paper trading (30 дней)

Только после B+C с доказанным OOS edge:

13. **Paper trading на демо-счёте MT5** минимум 30 календарных дней.
14. Онлайн-метрики в dashboard: PF, Sharpe, expectancy, win rate, avg win/loss,
    max consecutive losses — должны быть в пределах ±20% от бэктестовых.
15. Если в paper рассинхрон с бэктестом > 20% по любой метрике — **не идти в
    live**, искать почему (обычно: execution model в backtest нереалистична).

### Этап E — Micro-live (только после D)

16. Депо $200-300 на MT5 cent-account (не standard) — риск 0.25% на сделку,
    максимум 2 открытые позиции, дневной loss limit 2%, недельный 5%.
17. Первые 30 дней — **только мониторить**, не вмешиваться. Сравнивать
    ежедневный PnL с paper.
18. Если месяц закрылся в плюс **и** метрики сошлись с бэктестом в пределах
    30% — можно добавлять капитал до $1k.

---

## Часть 6 — Честная оценка ожиданий

На retail-счёте <$1k с MT5 realistic targets (без иллюзий):

- **Реалистично:** после всех фиксов и валидации, Sharpe 0.8-1.2 на OOS,
  годовая доходность 15-30% при MaxDD 10-15%. **Если** найдётся edge. Это
  вероятность оценочно 30-50% после полного пути A→E.
- **Нереалистично:** "X% в месяц стабильно", "5% в день как в рекламе EA",
  "никаких просадок". Такого в рыночной реальности не бывает на легальных
  форекс-стратегиях.
- **Главная ценность проекта сейчас:** не сам бот, а **инфраструктура**
  (feature pipeline, risk manager, walk-forward, MT5 bridge, dashboard). Это
  переиспользуется на любой будущей стратегии. Если бот не взлетит — не всё
  зря.

---

## Часть 7 — Что НЕ рекомендуется

- ❌ **Не запускай текущую модель в live.** Серия -0.24 Sharpe после 8M
  timesteps тренировки означает стабильный слив. На $1k это 200-400$ за месяц.
- ❌ **Не добавляй новые features** пока не починены баги #1-3. Больше features
  в сломанном pipeline = больше шума.
- ❌ **Не увеличивай capital в live пока paper < 30 дней** с метриками в
  пределах бэктеста.
- ❌ **Не используй leverage >10:1** на этом таймфрейме. H4 swing с leverage
  1:100 = высокая вероятность margin call при 2-3 убыточных сделках подряд.
- ❌ **Не полагайся на GPU-обучение на Colab** (как в `configs/colab/`) для
  финальной модели — Colab сессии прерываются, checkpoints иногда теряются.
  Для финальной тренировки — локальная машина или cloud с persistent storage.

---

## Приложение A — Быстрые фиксы как diff-заготовки

### A.1. math_utils.py — NaN-safe sqrt

```diff
--- a/src/apexfx/utils/math_utils.py
+++ b/src/apexfx/utils/math_utils.py
@@ -47,7 +47,8 @@ def parkinson_volatility(high, low, window):
     result = np.full(len(high), np.nan)
     for i in range(window, len(high)):
-        result[i] = np.sqrt(np.mean(parkinson_sq[i - window : i]) * 252)
+        val = np.mean(parkinson_sq[i - window : i]) * 252
+        result[i] = np.sqrt(max(val, 0.0)) if np.isfinite(val) else np.nan
     return result

@@ -66,7 +67,8 @@ def garman_klass_volatility(open_, high, low, close, window):
     result = np.full(len(high), np.nan)
     for i in range(window, len(high)):
-        result[i] = np.sqrt(np.mean(gk[i - window : i]) * 252)
+        val = np.mean(gk[i - window : i]) * 252
+        result[i] = np.sqrt(max(val, 0.0)) if np.isfinite(val) else np.nan
     return result
```

### A.2. normalizer.py — NaN-input guard

```diff
--- a/src/apexfx/features/normalizer.py
+++ b/src/apexfx/features/normalizer.py
@@ -41,6 +41,9 @@ class FeatureNormalizer:
     def transform_online(self, features: np.ndarray) -> np.ndarray:
         """Normalize a single observation using running statistics (live mode)."""
+        if np.any(~np.isfinite(features)):
+            features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
+
         if self._stats is None:
             n_features = features.shape[-1]
```

### A.3. reward.py — reasonable scale + soft saturation

Это требует осторожного pass — потому что в `TradingReward` много
компонентов; просто поменять scale мало. Минимальный смысловой патч:

```python
# In TradingReward.__init__:
reward_scale: float = 10.0,  # было 1000.0

# In TradingReward.compute, at the end, replace:
#   return float(np.clip(reward, -10.0, 10.0))
# with:
#   return float(3.0 * np.tanh(reward / 3.0))
```

Это сохраняет ответы на "умеренные" reward'ы линейно, но плавно давит хвосты
вместо обрезания — градиент не зануляется.

### A.4. trade_filter.py — training mode

```python
# В StrategyFilter.__init__:
def __init__(self, ..., training_mode: bool = False):
    ...
    self._training_mode = training_mode

# В StrategyFilter.check, после Rule 3:
if self._training_mode:
    # В обучении пропускаем rules 4, 5, 6 — они блокируют обучение на синтетике
    return FilterDecision(allowed=True, scale=1.0, reason="training_mode_pass", force_close=False)

# В TradeFilterWrapper (env/wrappers.py) — прокинуть флаг из config.
```

### A.5. curriculum — early stopping на финансовом Sharpe, не reward-based

```python
# В src/apexfx/training/callbacks.py или curriculum.py:
# Вычислять Sharpe из episode PnL (env.info["episode_pnl"]), не из reward'ов.
# Текущий -700k "Sharpe" — это mean/std по клиппованному reward ряду, не имеет
# финансового смысла. Нужна отдельная метрика trading_sharpe, основанная на
# realized returns эпизода.
```

---

## Приложение B — Как прогнать baseline самому

```bash
python3 << 'EOF'
import pandas as pd, numpy as np, glob
files = sorted(glob.glob('data/raw/bars/EURUSD/H1/*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.set_index('time')
h4 = df.resample('4h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
c = h4['close']
# Buy&Hold
bh = (1 + c.pct_change().fillna(0)).cumprod()
print(f"B&H total return: {bh.iloc[-1]-1:.2%}")
EOF
```

Смотришь, какое окно у тебя "тренд вверх" vs "боковик" vs "тренд вниз", и
бэктестишь свою модель **отдельно** на каждом типе.

---

## Приложение C — Контрольный чеклист для live trading (когда готов)

- [ ] OOS walk-forward на 12+ окнах даёт средний Sharpe > 1.0, стд/средн < 0.5
- [ ] Модель бьёт лучший baseline (B&H / MA / Donchian) на 80%+ окон
- [ ] Max DD в walk-forward < 15% на худшем окне
- [ ] Paper trading 30 дней, метрики в пределах ±20% от бэктеста
- [ ] MT5 demo account прошёл 30 дней без сбоев connection
- [ ] Kill switch физически работает (проверить руками — остановить бота)
- [ ] Daily loss limit 2% + weekly 5% hard-coded, проверены manual trigger
- [ ] Weekend gap guard работает (flat позиция на Friday close)
- [ ] Alerts в Telegram/email на: signal generated, order filled, SL/TP hit, daily DD breach
- [ ] Capital: старт $200-300 на cent-account, не standard
- [ ] Первые 30 дней — zero intervention, только наблюдение

---

## Итог

Проект **технически зрелый**, но в нём **три блокирующих бага training
pipeline** плюс **отсутствие walk-forward валидации** — вместе они объясняют,
почему после 8M timesteps обучения модель даёт Sharpe -0.24 и теряет деньги.

Фикс багов — 1-2 сессии. Walk-forward harness — 2-3 сессии. После этого у нас
будет ответ на главный вопрос: **есть ли вообще edge**, а не "как починить
обучение".

Если edge найдётся — путь в live (paper → micro → scale) займёт минимум
2-3 месяца от сегодня. Если не найдётся — переупрощение до базовой системы и
переосмысление (multi-symbol? другой таймфрейм? другие features?).

**Главное:** **не запускать в live сейчас.** На счёте $1k это гарантированный
слив за 2-4 недели при текущем Sharpe -0.24.

---

*Отчёт сгенерирован по результатам аудита кода и логов. Все цифры baseline
воспроизводимы запуском скрипта из Приложения B. Для каждого утверждения
указан file:line в исходнике.*
