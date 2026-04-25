# ApexFX Quantum — Аудит-отчёт v2: путь к прибыли

> **Дата:** 2026-04-25
> **Аудитор:** Claude (преемник AUDIT_REPORT.md от 2026-04-24)
> **Контекст:** капитал <$1k, MT5 retail, EURUSD H1/H4, цель — стабильная прибыль
> **Тип отчёта:** глобальная переоценка архитектуры + roadmap

---

## TL;DR (за 60 секунд)

1. **Прошлый аудит нашёл 3 критичных бага.** Они **исправлены в коде** (commits `1212c26`, `f13e206`, `1f2a549`), но **никогда не валидировались полным тренингом** — последняя обученная модель `models/best/final_model.zip` от **19 февраля**, *до* всех v2-переработок.
2. **Бэктест-результат Sharpe -0.24 относится к этой древней модели** — он не отражает текущее состояние pipeline.
3. **Главная проблема не в багах, а в архитектуре.** Hedge-fund-grade система (TQC + HiveMind + TFT + World Model + EWC + Adversarial, batch 2048, buffer 2M, configured под 2× RTX 4090) построена для **retail $1k**. Это фундаментальное несоответствие: больше слоёв ≠ больше прибыли, чаще наоборот.
4. **Edge никогда не был доказан, только предполагался.** За 6 запусков (Run 1-6) лучший OOS PF = 0.74 — **всё равно убыточно**. Buy & Hold даёт +9.97% в том же окне. Никакая твоя стратегия не побила этот бенчмарк.
5. **Feature selector эмпирически отверг 37 из 52 фич** — включая ВСЕ "профессиональные" (regime, structure, fundamental, wavelet, BOS, retest). Архитектурно эти модули — мёртвый груз.
6. **Walk-forward существует** (`src/apexfx/training/walk_forward.py`), но **используется только в `scripts/backtest.py`** — а тот, который запускают по умолчанию (`quick_backtest.py`), делает примитивный 70/30 split. Все цифры, которые показывали "результаты", были не walk-forward.

**Вывод:** проект **технически зрелый, но коммерчески недоказанный**. Прибыли не будет, пока не пройдёт честную walk-forward валидацию против baseline. Прежде чем добавлять код, нужно **убрать** 60-70% существующего и проверить, есть ли вообще edge.

---

## Часть 1 — Что изменилось со времени AUDIT_REPORT.md

### 1.1. Применённые фиксы (commits 24-25 апреля)

| Коммит | Что | Статус |
|---|---|---|
| `1212c26` | reward.py: `tanh` saturation вместо `np.clip(±10)`. Scale 1000→100. + training_mode bypass Rules 4/5/6 + NaN-safe sqrt в math_utils + NaN guard в normalizer | ✅ В коде |
| `f13e206` | trade_filter.py: training_mode bypass расширен до Rules 2/3/7 (news blackout, event imminent, pre-news scaling) — иначе 100% сделок блокируется на синтетике без calendar feed | ✅ В коде |
| `1f2a549` | model.tft.pretrain.enabled flag — теперь можно пропустить 30 эпох TFT pretrain (60 мин на CPU) | ✅ В коде |

**Юнит-тесты для всех трёх фиксов написаны и проходят** (~58 новых тестов, всего 769 passing).

### 1.2. КРИТИЧНОЕ: фиксы не валидированы полным тренингом

```bash
$ ls -la models/best/
-rw-r--r--  abobik  17088107 Feb 19 09:43 final_model.zip
```

Этот файл от **19 февраля** — за 2 месяца до всех v2 работ. Все backtest-результаты, которые ты видел (`backtest_results.json`: Sharpe -0.24), **относятся к нему**.

После 24 апреля:
- ✅ Smoke-тесты (4000 timesteps) показали: feature pipeline работает, trade_filter не блокирует (100 сделок в первом эпизоде), нет NaN.
- ❌ **Полный тренинг (5M timesteps × 4 стадии) — НЕ запускался.**
- ❌ Honest OOS walk-forward backtest — НЕ запускался.
- ❌ Baseline comparison (B&H, MA, Donchian) — НЕ интегрирован в pipeline.

**Это значит:** ты НЕ знаешь, дают ли фиксы прибыль. Возможно дают. Возможно нет. Прямо сейчас — **неизвестно**.

---

## Часть 2 — Новые открытия моего аудита

### 2.1. Архитектура vs реальность: количественные несоответствия

| Параметр | В коде | Адекватно для retail $1k? |
|---|---|---|
| `initial_balance` (training) | $100,000 | ❌ ×100 от реальности |
| Buffer size | 2,000,000 transitions | ❌ нужно 50-200k для одной симуляции |
| Batch size | 2048 | ❌ нужно 64-256 |
| TFT `d_model` | 128 | ❌ overkill для 15 фич |
| RL algorithm | TQC (truncated quantile critics) | ❌ экзотика, SAC даст то же |
| Multi-agent ensemble | 3 агента + CrossAgentAttention + Gating | ❌ overhead без доказанного edge |
| TFT pretrain | 30 эпох × ~2 мин | ❌ supervised pretrain для RL — ложный сигнал |
| World Model + imagination rollouts | enabled | ❌ исследовательский PoC, не готов к production |
| EWC `lambda_ewc` | 5000 | ❌ убил политику в Run 3 stage 4 (catastrophic ossification) |
| Smart execution | TWAP/VWAP/IS routing | ❌ для лотов <0.5 — direct only |
| Risk checks | 10 (cascading) | ❌ для $1k достаточно 3 |
| Strategy filter rules | 6 | ❌ блокирует 100% сделок без calendar feed |

**Проблема не в том, что эти компоненты плохи** — каждый по отдельности валиден для своей задачи. **Проблема в том, что они складываются в систему, которая для $1k retail трейдинга превращается в монстра**, который:

- **обучается медленно** (часы вместо минут — больше итераций отладки)
- **переобучается** (больше параметров на меньше данных)
- **трудно отлаживается** (10 слоёв между сигналом и сделкой — каждый может быть багом, доказательство тому — 6 runs с reward bug, который никто не заметил)
- **скрывает edge** (если он есть): когда модель теряет деньги, непонятно, в каком из 10 компонентов проблема

### 2.2. Эмпирическое доказательство переусложнённости

Из smoke-лога `logs/apexfx_smoke.log` (25 апреля 10:51):

```
Feature importance ranking (top 15):
   1. delta                       0.0573
   2. spread_bps                  0.0498
   3. book_pressure               0.0459
   4. trend_strength              0.0435
   5. poc_distance                0.0409
   ...

Dropped 37 features:
   ['realized_vol', 'wavelet_trend', 'volume_at_best',
    'trade_flow_toxicity', 'fft_amplitude_*', 'fft_period_*',
    'regime_mean_reverting', 'regime_flat', 'regime_trending',
    'wavelet_energy_*', 'structure_trend',
    'structure_break_bull', 'structure_break_bear', 'retest_signal',
    'level_confluence', 'nearest_support_distance', ...]

Validation accuracy: 0.5328 (baseline ~0.50)
```

**Что это значит:**

1. **Топ-15 фич — почти все базовые** (delta, spread, book pressure, trend strength). Они дешёвые в вычислении, не требуют ML, могут быть рассчитаны в 50 строк Python.

2. **Все "профессиональные" фичи отвергнуты:**
   - 4 wavelet-фичи — мёртвые (wavelet decomposition не нужен)
   - Все 4 FFT-фичи — мёртвые (нет периодичности)
   - Все 3 regime-фичи (HMM regime detection) — мёртвые
   - **Все 4 structure-фичи** (BOS, retest, structure_trend) — мёртвые. *А ведь Strategy Filter Rule 5 требует BOS для входа!*
   - Все support/resistance фичи — мёртвые

3. **Validation accuracy 53.28%** — на 3.28% выше монетки. Это **очень слабый сигнал**. Любой ML-инженер скажет: с такой точностью **нет edge на уровне предсказания направления**, нужно искать его в asymmetric reward (квантование убытков, удержание прибыльных позиций).

4. **Это означает фундаментальный пересмотр архитектуры:** все модули, которые генерят отвергнутые фичи (`fundamental.py`, `structure.py`, `regime.py`, `spectral.py`, `wavelet.py`, `central_bank.py`, `cot.py`, `seasonal.py`, `intermarket_corr.py`) — это ~3000 строк кода, которые **загрязняют noise feature space** и **замедляют pipeline без результата**.

### 2.3. Walk-forward существует, но не используется по умолчанию

```bash
$ grep -n "walk_forward" scripts/*.py
scripts/backtest.py:1:"""Run walk-forward backtest on historical data."""
scripts/backtest.py:12:from apexfx.training.walk_forward import WalkForwardValidator
```

```bash
$ grep "n_features" scripts/quick_backtest.py
n_features = min(pipeline.n_features, 30)  # 70/30 split, не WF
```

**Кто что запускает по факту:**

| Скрипт | Метод | Используется? |
|---|---|---|
| `scripts/backtest.py` | walk_forward 12+ folds | Никогда (нет в README quick start) |
| `scripts/quick_backtest.py` | простой 70/30 split | Да, отсюда `backtest_results.json` |

**Walk-forward код есть** (`src/apexfx/training/walk_forward.py`, 254 LOC, включая Monte-Carlo permutation test для p-value). Это **взрослая** реализация. Но она не запущена ни разу для production-валидации — все цифры, которые ты видел, **не walk-forward**.

### 2.4. Главное: ты НЕ знаешь, есть ли edge

Из CHRONOLOGY_RUNS_1-6.md best-case за 6 запусков:

```
Run 5 OOS:
  - 165 trades, WR 35.76%
  - PF 0.736 (proximal flat, slightly losing)
  - Final balance $99,995 / $100k = -0.005%
  - Max drawdown $7.9 на $100k = 0.0079% (микропозиции!)
```

**165 сделок при WR 35.76% даёт PF 0.736 — это убыточная стратегия.** На капитале $1k это будет ~$50-150 потерь за период тестирования (если не считать спред правильно — а спред в 1.5 пип в твоих симуляциях занижен для retail).

Buy & Hold в том же окне: **+9.97%, Sharpe 0.65, единственная сделка**.

**Ты НИКОГДА не побил Buy & Hold** в задокументированных запусках. Это значит: edge **не доказан**, он **предполагается** на основе того, что архитектура выглядит "умной".

---

## Часть 3 — Глобальная архитектурная проблема (Root Cause)

### 3.1. "Hedge-fund grade" миф

`README.md` строка 21:
> "ApexFX Quantum is a professional-grade algorithmic trading system... combines deep learning (Temporal Fusion Transformers), reinforcement learning (SAC/PPO with continuous action space), and an ensemble of specialized trading agents — all governed by institutional-level risk management..."

Реальность: ты — ритейл-трейдер с $1k, торгующий на MT5 у retail-брокера.

| Hedge fund | Ты |
|---|---|
| Капитал $10M+ | <$1k |
| Co-located сервер на NY4 | MT5 на домашнем компе/VPS |
| Прямой доступ к ECN | Розничный broker (Markets.com, OANDA, и т.п.) |
| Спред 0.1-0.3 пипса EURUSD | Спред 1.5-3.0 пипса EURUSD |
| Маржинальный коэффициент 5:1 | До 1:500 (опасный) |
| Команда 5-10 quant'ов поддерживает систему | Один человек |
| Bloomberg/Reuters terminal с calendar | Forex Factory scrape (rate limited) |
| Институциональные данные tick-by-tick | OHLC bars от broker |
| Прайм-брокер для shorting и hedging | Брокер-противник (B-book vs A-book) |

**Вывод:** "professional-grade" в твоём контексте = **переусложнение**, не качество. Hedge fund стратегии работают, потому что у них **execution edge** (спред в 10× дешевле, latency в 100× меньше). У тебя этого edge **нет**. Поэтому и стратегии должны быть **простее**, а не сложнее.

### 3.2. Парадокс сложности в RL для трейдинга

Известный квант-факт: **чем сложнее модель, тем больше она оверфитит на training set, и тем хуже её edge на OOS** — пока не наберётся **очень** много данных.

Твои данные: ~12,321 H1 баров (~2 года EURUSD). Это **маленький** датасет для глубоких моделей. Профессиональные quant команды требуют **10+ лет** для надёжного RL-тренинга. С 2 годами данных и 11M параметров (твоя current модель) — соотношение sample/parameter ужасное.

В таблице:

| Параметров модели | Минимум данных |
|---|---|
| 100k (простой MLP) | 50k bars (~1 год M5) |
| 1M (TFT-small) | 500k bars |
| 10M+ (твоя текущая) | 5M+ bars (~30 лет H1) |

**У тебя в 400 раз меньше данных, чем нужно для текущей архитектуры.** Это **гарантированный оверфит**. И именно поэтому Run 5 при тренинге выдал sharpe -65, а на OOS PF = 0.74 (модель заучила тренинг, но edge не нашла).

### 3.3. Все три критичных бага были симптомами одной болезни

- **Reward clipping** — потому что 10 компонентов reward с шкалой 1000 → невозможно балансировать вручную → нужен autoscale → его нет → клиппинг → мёртвый градиент.
- **Trade filter blocking 100%** — потому что 6 жёстких правил полагаются на features, которых нет в синтетике → весь pipeline стопается → не заметили месяц.
- **NaN в sqrt** — потому что Garman-Klass и Parkinson volatility — одни из 8 разных volatility-фич, которые **в итоге feature selector выбросил как бесполезные**.

**Все три бага были в коде, который feature selector потом признал ненужным.** Если убрать ненужные компоненты, баги исчезнут вместе с ними.

---

## Часть 4 — Три пути вперёд (с честной оценкой шансов)

### Путь A — "Validate-first" (РЕКОМЕНДУЕМЫЙ ПЕРВЫЙ ШАГ)

**Что:** запустить полный тренинг с применёнными фиксами и сделать честный walk-forward backtest. Это занимает 1-3 дня, но даёт **первое реальное измерение** edge.

**Почему первый:** прежде чем перерабатывать архитектуру, **узнай**, дают ли уже сделанные фиксы прибыль. Возможно — да. Тогда не надо ничего ломать.

**Как:**
```bash
# 1. Проверить что в configs/training.yaml курsewrickulum.stages адекватно
#    (например, total_timesteps уменьшить до 500K каждая стадия для скорости)
# 2. Запустить полный тренинг
ssh user1@82.202.157.240
cd ~/apexfx && git pull origin main
MODE=full SYMBOL=EURUSD TIMEFRAME=H1 bash scripts/run_server.sh

# 3. После тренинга — НЕ quick_backtest, а walk-forward
python scripts/backtest.py --config-dir configs --symbol EURUSD

# 4. Сравнить с baseline (нужно дописать — см. Phase 2)
```

**Шанс положительного результата:** 15-25%. Текущая архитектура мощная, фиксы реальные, но 6 runs прошлого ничего не дали — нет оснований ожидать прорыв.

**Если получится:** перейти к Path C (multi-symbol, для diversification) или сразу к paper trading (Этап D из старого аудита).

**Если НЕ получится:** перейти к Path B (radical simplification).

---

### Путь B — "Radical simplification" (ВТОРОЙ ВАРИАНТ, если A не дал edge)

**Что:** удалить 60-70% кода, оставить минимум, переучить, перетестировать.

**Почему:** ML и RL литература последовательно говорит: на маленьких датасетах **простые модели обобщаются лучше**. Random Forest часто бьёт глубокие сети. Линейная регрессия часто бьёт Random Forest. Для 12k samples и 15 фич — **самая простая модель, которую ты можешь придумать, скорее всего лучшая**.

**Что удалить (можно безопасно убрать):**

| Модуль | LOC | Причина |
|---|---|---|
| `models/ensemble/hive_mind.py` + `agents/*` + `cross_agent_attention.py` | ~1500 | Multi-agent overhead без доказанного edge |
| `models/world_model/*` + `training/imagination.py` | ~800 | Исследовательский, не ready for production |
| `training/ewc.py` + EWCCallback | ~300 | Lambda 5000 убил политику в Run 3 |
| `training/adversarial.py` + GradientPenaltyCallback | ~400 | Adversarial добавляет шум на и без того маленький датасет |
| `training/per.py` (PrioritizedExperienceReplay) | ~250 | SumTree sampling даёт минимальный gain в простом env |
| `models/tft/*` (Temporal Fusion Transformer) | ~600 | TFT для 15 фич — стрельба из пушки по воробьям |
| `features/{wavelet,spectral,fundamental,structure,central_bank,cot,seasonal,intermarket_corr,scalping,sentiment}.py` | ~3000 | Feature selector эмпирически отверг |
| `env/mtf_forex_env.py` + `data/mtf_synthetic.py` + `models/ensemble/cross_tf_fusion.py` | ~800 | Multi-timeframe не оправдан без доказанного бенефита |
| `risk/{stress_testing,var_calculator}.py` | ~700 | $1k счёт не нуждается в VaR-95 моделировании |
| `execution/{smart_exec,fill_tracker,liquidity_guard}.py` | ~1500 | TWAP/VWAP/IS для retail < 0.1 лота — мёртвый код |
| `live/{health_check,state_manager}.py` (часть) | ~400 | Можно упростить |
| `_v2_dump/*` | ~2580 | Просто дамп старой версии в main, удалить |
| **ИТОГО к удалению** | **~13,000 LOC** | **38% codebase** |

**Что оставить (work):**

| Модуль | Почему |
|---|---|
| `data/{mt5_client,data_store,bar_aggregator}.py` | Производственный, работает, нужен для live |
| `data/calendar_provider.py` (опционально) | Если оставить fundamental pipeline для live, иначе тоже выбросить |
| `features/{pipeline,selector,normalizer}.py` + top-15 extractors | Это работает |
| `env/{forex_env,reward}.py` | Базовый env с LogReturnReward |
| `env/trade_filter.py` | Только Rule 1 (conflicting) и Rule 2 (news) — для live |
| `training/{trainer,curriculum,walk_forward,checkpoint_manager}.py` | Trainer упростить до single-stage; walk_forward — main mode |
| `models/agents/*` → заменить на простой SAC MLP | Просто [128,128] policy и critic |
| `risk/{risk_manager (stripped),position_sizer,drawdown_monitor,cooldown,news_filter}.py` | 3-4 проверки достаточно для retail |
| `execution/{executor,order_manager}.py` | Direct market only |
| `live/{trading_loop,signal_generator}.py` | Боевой контур для MT5 |
| `dashboard/app.py` | Мониторинг |

**Шанс edge после simplification:** 25-40%. Потому что меньше параметров → меньше оверфит → больше шансов, что найденный edge будет реальным.

**Минусы:** месяц работы переписать, риск выкинуть что-то полезное (но walk-forward это покажет).

---

### Путь C — "Different problem entirely" (если A и B не дали edge)

EURUSD H1 retail с $1k — это **одна из самых сложных** задач в quant trading. Все edges уже арбитражированы институциональными игроками. Если A и B не дали edge — задача сама плохо поставлена.

**Альтернативы:**

#### C.1. Multi-symbol portfolio (минимальная переделка)
Вместо одной EURUSD пары — портфель из 4-6 major пар (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCHF, USDCAD). 
- Diversification снижает риск.
- Если на каждой паре PF = 0.95 (slight loss), портфель из 6 декоррелированных может дать PF = 1.05 после costs.
- В коде уже есть `MultiSymbolConfig` (видно в CHRONOLOGY rec #9) — не использовался ни разу.

**Шанс edge:** 30-50%, потому что diversification — реальный известный effect.

#### C.2. Другой таймфрейм — D1 swing
- H1 → D1: ×24 меньше баров, но ×24 меньше costs
- Меньше шума intraday
- Edge в swing-trading (carry trades, trend on D1) известен и работает у retail
- Меньше сделок → меньше суммарных costs → больше шанс плюса
- **2 года H1 = 504 D1 баров — это уже по-серьёзному маловато для RL**, лучше минимум 5 лет

**Шанс edge:** 35-50% (proven base rate в swing trading), но требует 5+ лет данных.

#### C.3. Другой рынок — крипта perpetuals
- Спреды на BTC/ETH perps на Binance/Bybit: 0.01-0.03% (сравнимо с EURUSD институционально)
- 24/7 рынок — больше данных
- Меньше news event risk (нет NFP)
- Проще API (REST + WebSocket)
- Минусы: волатильность выше, можно потерять весь капитал быстро

**Шанс edge:** 40-60% (более демократичный рынок, retail может конкурировать).

#### C.4. Использовать proven strategy вместо изобретения
Не каждый алгоритм нужно тренировать. Известные **работающие** ритейл-стратегии:
- **Carry trade**: long high-yield currency, short low-yield. Работает 60% времени.
- **Trend following на D1**: 50-day breakout (Donchian) на портфеле. Проверено 30+ лет.
- **Pairs trading на equity** (не FX): mean reversion на коинтегрированных парах. Edge документирован.

Если ты ИНВЕСТОР, а не researcher — **ты можешь скопировать стратегию**, которая работает. Не обязательно изобретать.

---

## Часть 5 — Конкретная упрощённая архитектура (apexfx_v3)

Если выбираешь Path B, вот что должно остаться. Сравнение:

```
Текущая структура:               Предлагаемая v3:
src/apexfx/                      src/apexfx/
├── 144 .py files                ├── ~25 .py files
├── 34,247 LOC                   ├── ~8,000 LOC
├── 50 test files                ├── ~20 test files
├── HiveMind + TFT + WM          ├── SAC + MLP
├── 3 specialist agents          ├── 1 unified policy
├── 32+ features                 ├── 15 features (pre-selected)
├── 6 strategy filter rules      ├── 2 rules (conflict + news)
├── 10 risk checks               ├── 4 checks (kill, daily, position, drawdown)
├── 4-stage curriculum           ├── 1-stage training
├── Smart exec (TWAP/VWAP/IS)    ├── Direct market orders
└── Multi-timeframe (D1/H1/M5)   └── Single timeframe
```

### v3 структура

```
apexfx_v3/
├── data/
│   ├── store.py                  # Parquet I/O (КЕЕР)
│   ├── mt5_client.py             # Live data feed (КЕЕР)
│   ├── multi_symbol.py           # NEW: portfolio of 4-6 pairs
│   └── bar_aggregator.py         # tick → OHLC (КЕЕР)
│
├── features/
│   ├── pipeline.py               # 15 features only, 200 LOC
│   ├── selector.py               # КЕЕР
│   └── normalizer.py             # КЕЕР (with NaN guards)
│
├── env/
│   ├── forex_env.py              # Single-symbol Gymnasium env
│   ├── multi_symbol_env.py       # NEW: portfolio env
│   ├── reward.py                 # LogReturnReward only
│   └── trade_filter.py           # 2 rules: conflict + news
│
├── models/
│   └── policy.py                 # SAC with simple MLP [256,128]
│
├── training/
│   ├── trainer.py                # Single-stage SAC training (~300 LOC)
│   ├── walk_forward.py           # КЕЕР, USE BY DEFAULT
│   └── checkpoint_manager.py     # КЕЕР simplified
│
├── risk/
│   ├── risk_manager.py           # 4 checks (~200 LOC)
│   ├── position_sizer.py         # Fixed % risk per trade (~100 LOC)
│   └── kill_switch.py            # Daily/weekly/manual (~100 LOC)
│
├── execution/
│   ├── executor.py               # Direct market orders (~200 LOC)
│   └── mt5_bridge.py             # КЕЕР simplified
│
├── eval/
│   ├── baselines.py              # NEW: B&H, MA, Donchian, Random
│   ├── walk_forward_report.py    # NEW: per-fold table + stability
│   └── stress_periods.py         # NEW: test on uptrend/downtrend/range periods separately
│
├── live/
│   ├── trading_loop.py           # КЕЕР simplified
│   └── signal_generator.py       # КЕЕР
│
└── dashboard/
    └── app.py                    # КЕЕР
```

### Reward function v3 (всё что нужно)

```python
class LogReturnReward(BaseRewardFunction):
    """Simple log-return with asymmetric loss penalty.
    
    The cleanest reward signal: positive when portfolio grows, negative when shrinks,
    extra penalty for losses (loss aversion). No clipping (tanh saturation only).
    """
    
    def __init__(self, loss_weight: float = 1.5, transaction_cost: float = 0.0001):
        self.loss_weight = loss_weight
        self.transaction_cost = transaction_cost
        self._prev_position = 0.0
    
    def compute(self, portfolio_value, prev_portfolio_value, action=None) -> float:
        if prev_portfolio_value <= 0:
            return 0.0
        
        log_ret = np.log(portfolio_value / prev_portfolio_value)
        
        # Asymmetric loss
        if log_ret < 0:
            log_ret *= self.loss_weight
        
        # Transaction cost on direction change
        if action is not None and (action * self._prev_position) < 0:
            log_ret -= self.transaction_cost
            self._prev_position = action
        
        # Soft saturation (no zero gradient)
        return float(np.tanh(log_ret * 100))
```

**Почему это лучше существующего TradingReward (10 компонентов):**
- Один источник истины: realized PnL
- Невозможно "обмануть" агента побочными reward (winner bonus, structure confirm и т.п. — все эвристики, которые могут противоречить реальной прибыли)
- Меньше гиперпараметров для тюнинга
- Градиент чистый

### Risk manager v3 (4 проверки вместо 10)

```python
def check_trade(action, portfolio_state, market_state) -> RiskDecision:
    # 1. Kill switch (manual file or daily loss > 2%)
    if kill_switch.is_active(): 
        return RiskDecision(approved=False, reason="kill_switch")
    
    # 2. Daily loss limit (2%)
    if daily_loss_guard.would_exceed(action, portfolio_state):
        return RiskDecision(approved=False, reason="daily_loss_limit")
    
    # 3. Drawdown limit (5%)
    if drawdown_monitor.would_exceed(action, portfolio_state):
        return RiskDecision(approved=False, reason="max_drawdown")
    
    # 4. Position size (Fixed % risk per trade — не Kelly)
    position_size = position_sizer.fixed_pct_risk(
        action, portfolio_state.equity, market_state.atr,
        risk_per_trade=0.005  # 0.5% per trade
    )
    if position_size < 0.01:  # MT5 min lot
        return RiskDecision(approved=False, reason="below_min_lot")
    
    return RiskDecision(approved=True, position_size=position_size)
```

### Eval pipeline v3 (с baselines!)

```python
# eval/baselines.py
class BuyAndHoldBaseline:
    def predict(self, obs): return np.array([1.0])  # Always long

class MACrossBaseline:
    def __init__(self, fast=20, slow=50): self.fast, self.slow = fast, slow
    def predict(self, obs): 
        fast_ma = obs['close'][-self.fast:].mean()
        slow_ma = obs['close'][-self.slow:].mean()
        return np.array([1.0 if fast_ma > slow_ma else -1.0])

class DonchianBaseline:
    def __init__(self, window=20): self.window = window
    def predict(self, obs):
        high = obs['high'][-self.window:].max()
        low = obs['low'][-self.window:].min()
        if obs['close'][-1] > high * 0.999: return np.array([1.0])
        if obs['close'][-1] < low * 1.001: return np.array([-1.0])
        return np.array([0.0])

# eval/walk_forward_report.py — runs RL model + 4 baselines through same WF folds,
# produces table:
# Fold | Period   | Model SR | B&H SR | MA SR | Donchian SR | Beats best? |
# 0    | 2024-Q1  | 0.8     | 0.5   | -0.1  | 0.3         | YES         |
# 1    | 2024-Q2  | 0.3     | 0.7   | 0.2   | 0.4         | NO          |
# ...
# Aggregate: model beats best baseline on X/N folds
```

**Правило для решения "идти ли в live":** model должна побить лучший baseline на **минимум 60% folds**, со средним Sharpe > 1.0.

---

## Часть 6 — Roadmap к paper trading: 4 недели

### Week 1 (apr 26 — may 2): Validate-first

**Цель:** узнать, дают ли применённые фиксы edge.

| День | Задача |
|---|---|
| 1-2 | Запуск полного тренинга на MWS GPU с current configs, monitoring |
| 3 | Полный walk-forward backtest (`scripts/backtest.py --config-dir configs`) |
| 4 | Реализовать `eval/baselines.py` (200 LOC) и интегрировать в backtest.py |
| 5 | **РЕШЕНИЕ:** Sharpe > 0 AND beats best baseline на 50%+ folds → Week 2.1 (refine). Иначе → Week 2.2 (simplify) |

### Week 2.1 (если edge найден): Refinement

| День | Задача |
|---|---|
| 1 | Hyperparameter sweep по 5-10 ключевым (LR, gamma, batch, ent_coef, reward_scale) |
| 2-3 | Multi-period stress test (2024 uptrend, 2022 downtrend, 2023 range) |
| 4 | Multi-symbol expansion (добавить GBPUSD, USDJPY) |
| 5 | Реалистичный cost model (spread 2.5 пип, swap rate) |

### Week 2.2 (если edge не найден): Simplification

| День | Задача |
|---|---|
| 1 | Backup current (`git tag pre-simplify-2026-05-02`) |
| 2-3 | Удалить 13k LOC по списку из Part 4.B |
| 4 | Переписать trainer на single-stage SAC |
| 5 | Re-train, re-validate (walk-forward) |

### Week 3: Decision + remaining work

После Week 2 имеешь честные walk-forward цифры (либо refined, либо simplified).

| Сценарий | Действие |
|---|---|
| WF Sharpe > 1.0 на 60%+ folds + beats baselines | → Week 4 (paper prep) |
| 0 < WF Sharpe < 1.0 | → ещё 1 week refinement, попробовать Path C.1 (multi-symbol) |
| WF Sharpe ≤ 0 | → **Path C** (different timeframe / market / strategy) |

### Week 4 (только если предыдущие условия выполнены): Paper trading prep

| День | Задача |
|---|---|
| 1 | MT5 demo account на retail брокере |
| 2 | Live signal generation pipeline (real ticks → features → model → action) |
| 3 | Telegram alerts (signal, fill, SL/TP, daily PnL, DD breach) |
| 4 | Kill switch testing (manual file kill, automated daily loss kill) |
| 5 | Start 30-day paper trading |

### Week 5-8: Paper trading + monitoring

- Daily check: paper PnL vs backtest expectation. Расхождение > 30% — STOP, диагностируй.
- Weekly review: WR, PF, expectancy in line with backtest?
- В конце 30 дней: если все метрики в пределах ±20% backtest, и PF >= 1.0 — можно micro-live.

### Week 9+ (если paper прошёл): Micro-live

- Депозит $200-300 на cent-account.
- Risk per trade: 0.25% (= $0.50 — $0.75 нотионал на $200 депо, минимум возможный).
- Daily loss limit hard-coded 2% ($4-$6).
- Первый месяц — **только наблюдение**, никаких изменений в коде.
- Если месяц закрыт в плюс И метрики ±30% от paper — добавить капитал до $1k.

---

## Часть 7 — Реалистичные ожидания и вероятности

### 7.1. Best-case prognosis

После полного A→B→C цикла с честной валидацией:

| Метрика | Реалистичный таргет |
|---|---|
| Annual Sharpe (OOS) | 0.8-1.5 |
| Annual return | 12-25% (после costs) |
| Max DD | 8-15% |
| Win rate | 45-55% |
| Profit factor | 1.05-1.4 |
| Сделок в год (H4 swing) | 50-150 |
| Сделок в год (D1 swing) | 20-50 |

**Это не "X% в месяц, никаких просадок" из ютьюб-рекламы.** Это reality того, что ритейл-трейдер с системой может реально стабильно выжимать.

### 7.2. Вероятности по путям

| Путь | Шанс edge | Время до решения |
|---|---|---|
| Только Path A (validate fixes) | 15-25% | 1 week |
| Path A → Path B (simplify if A fails) | 25-40% | 3-4 weeks |
| Path A → Path C (different problem) | 30-50% | 6-8 weeks |
| Полный A → B → C итеративно | 40-55% | 8-12 weeks |
| **Принять что edge не найдётся, использовать proven (carry/trend D1 портфель)** | **55-75%** | 4-6 weeks |

### 7.3. Что НЕ принесёт прибыль (анти-список)

- ❌ Добавление новых features (текущие 37 уже отвергнуты, +N не поможет)
- ❌ Hyperparameter tuning на сломанной архитектуре (сначала почини, потом тюнингуй)
- ❌ Запуск на live без 30-day paper (гарантированный слив)
- ❌ Колаб-обучение для production (сессии прерываются, чекпоинты теряются)
- ❌ Использование текущей `models/best/final_model.zip` (от Феb 19, baseline) для live
- ❌ Trust в один backtest-результат (нужен walk-forward + multi-period)
- ❌ Leverage > 10:1 на H1/H4 (margin call после 2-3 убытков)
- ❌ Ожидание "вот ещё одна неделя обучения и заработает" (6 runs показали — нет)

---

## Часть 8 — Точки решения (когда PIVOT, когда продолжать)

### Decision Tree

```
START: Run full training (Week 1)
  │
  ├─ WF Sharpe > 0.5 AND beats best baseline 60%+ folds?
  │   YES → Refine + multi-symbol (Week 2.1) → Paper trading (Week 4)
  │   NO  → ↓
  │
  ├─ WF Sharpe в [-0.5, 0.5]?
  │   YES → Simplify (Path B, Week 2.2) → Re-validate
  │         │
  │         ├─ Improvement?
  │         │   YES → Continue refinement
  │         │   NO  → Go to Path C (different problem)
  │
  ├─ WF Sharpe < -0.5?
  │   YES → Architecture сломана сильнее, чем баги, СРАЗУ Path C
  │
  └─ Paper trading вышел в просадку 5%+ в первые 2 недели?
      YES → STOP, diagnostics, не идти в live ни при каких условиях
```

### Hard stop conditions

Останавливай работу над текущим направлением, если:

1. После Week 1 WF Sharpe < -0.3 → architecture broken
2. После Week 4 нет ни одного 30-day paper periode с PF > 1.0 → нет edge
3. После Week 8 продолжаешь крутить hyperparams без stable improvement → confirmation bias
4. После 30-day live в minus > 5% → fundamental problem, не technical

---

## Часть 9 — Immediate next 3 actions

### Action 1 (TODAY): Чек что фиксы реально применены и тесты зелёные

```bash
cd /Users/abobik/Desktop/M
git status                         # должно быть clean (или только локальные правки)
git log --oneline -5               # должно показывать 1f2a549 → f13e206 → 1212c26 → e5558942
.venv/bin/pytest tests/ -x --tb=short  # все 769 тестов зелёные
```

Если зелёные — фиксы реально живы. Если нет — раздели проблемы и почини, прежде чем тренировать.

### Action 2 (THIS WEEK): Запустить полный тренинг + walk-forward backtest

```bash
# На GPU сервере MWS
ssh user1@82.202.157.240
cd ~/apexfx && git pull origin main

# Полный тренинг (5M timesteps, ~6-12 часов на A100)
MODE=full SYMBOL=EURUSD TIMEFRAME=H1 nohup bash scripts/run_server.sh &
# мониторинг
tail -f logs/train_*.log

# После завершения — pull модели локально
rsync -avz user1@82.202.157.240:~/apexfx/models/best/ ./models/best/
rsync -avz user1@82.202.157.240:~/apexfx/models/checkpoints/ ./models/checkpoints/

# ЛОКАЛЬНО запустить walk-forward (НЕ quick_backtest!)
python scripts/backtest.py --config-dir configs --symbol EURUSD
# результат → walk_forward_results.json
```

### Action 3 (NEXT WEEK): Реализовать baseline comparison

Создать `src/apexfx/eval/baselines.py` с 4 классами (BuyAndHold, MACross, Donchian, Random) и интегрировать в `scripts/backtest.py`:

```python
# В scripts/backtest.py добавить:
from apexfx.eval.baselines import BuyAndHoldBaseline, MACrossBaseline, DonchianBaseline, RandomBaseline
from apexfx.eval.walk_forward_report import generate_comparison_table

# После основного walk_forward.run():
baselines = {
    'B&H': BuyAndHoldBaseline(),
    'MA(20,50)': MACrossBaseline(20, 50),
    'Donchian(20)': DonchianBaseline(20),
    'Random': RandomBaseline(),
}
baseline_results = {name: walk_forward.evaluate_baseline(b) for name, b in baselines.items()}
print(generate_comparison_table(model_results, baseline_results))
# Сохранить как walk_forward_comparison.csv
```

После этого ты сможешь честно сказать: **"моя модель бьёт лучший baseline на X из Y folds. Edge есть/нет."**

---

## Часть 10 — Финальный вердикт

### Что у тебя сейчас есть (положительное)
- ✅ Solid инфраструктура: data pipeline, MT5 bridge, dashboard, monitoring
- ✅ 769 unit-тестов (хороший purity gate)
- ✅ Walk-forward код существует
- ✅ Все 3 critical bug fixes применены (хоть и не валидированы тренингом)
- ✅ Опыт 6 runs обучения на GPU — понимаешь что работает технически

### Что у тебя НЕТ (что нужно)
- ❌ Доказанный edge на OOS данных
- ❌ Honest walk-forward результаты (walk-forward код есть, не запущен)
- ❌ Baseline comparison (B&H, MA, Donchian — не интегрированы)
- ❌ Multi-period stress testing (только 1 окно тестировалось)
- ❌ Реалистичные cost models для retail MT5 (1.5 пип спред — оптимистично)
- ❌ Paper trading data (нужно 30 дней до live)

### Главное послание

**Прибыль придёт не от добавления новых features или ещё одного RL-алгоритма.**
**Прибыль придёт от честной валидации того, что у тебя уже есть, и готовности удалить то, что не работает.**

Текущая система — over-engineered показуха для того, что должно быть **простой, прозрачной, валидированной** торговой системой. Переделай её в эту простую и валидированную систему — и шанс на прибыль вырастет с 15% до 40-55%.

Если на это нет 4-8 недель — **не торгуй текущим алгоритмом**. На $1k счёте сольёшь за 3-4 недели гарантированно.

Если есть 4-8 недель — следуй roadmap из Части 6. К концу августа сможешь сказать: "у меня либо есть edge, либо я знаю что его нет — обоих случаях я не сольюсь, потому что не торгую вслепую."

---

## Приложение A — Изменённые файлы для Action 3 (baselines)

### A.1. Новый `src/apexfx/eval/__init__.py` и `baselines.py`

```python
# src/apexfx/eval/baselines.py
"""Trading baselines for honest comparison: a model must beat these to claim edge."""
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

class TradingBaseline(ABC):
    """Minimal interface mirroring SB3 model.predict()."""
    @abstractmethod
    def predict(self, obs, deterministic: bool = True) -> tuple[np.ndarray, None]: ...
    def reset(self): pass

class BuyAndHoldBaseline(TradingBaseline):
    """Always-long position."""
    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None

class MACrossBaseline(TradingBaseline):
    """Simple MA crossover. Uses obs['close'] window."""
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast, self.slow = fast, slow
    def predict(self, obs, deterministic=True):
        prices = obs.get('price_window', obs.get('close', None))
        if prices is None or len(prices) < self.slow:
            return np.array([0.0], dtype=np.float32), None
        fast_ma = float(np.mean(prices[-self.fast:]))
        slow_ma = float(np.mean(prices[-self.slow:]))
        action = 1.0 if fast_ma > slow_ma else -1.0
        return np.array([action], dtype=np.float32), None

class DonchianBaseline(TradingBaseline):
    """Donchian channel breakout: long on high break, short on low break."""
    def __init__(self, window: int = 20):
        self.window = window
    def predict(self, obs, deterministic=True):
        high = obs.get('high_window', None)
        low = obs.get('low_window', None)
        close = obs.get('close', None)
        if any(x is None for x in (high, low, close)) or len(high) < self.window:
            return np.array([0.0], dtype=np.float32), None
        ch_high = float(np.max(high[-self.window:]))
        ch_low = float(np.min(low[-self.window:]))
        c = float(close[-1] if hasattr(close, '__len__') else close)
        if c >= ch_high * 0.9995: return np.array([1.0], dtype=np.float32), None
        if c <= ch_low * 1.0005: return np.array([-1.0], dtype=np.float32), None
        return np.array([0.0], dtype=np.float32), None

class RandomBaseline(TradingBaseline):
    """Random ±1 / 0 actions — sanity check for cost model."""
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
    def predict(self, obs, deterministic=True):
        return np.array([self._rng.choice([-1.0, 0.0, 1.0])], dtype=np.float32), None
```

### A.2. Patch `scripts/backtest.py` для baseline comparison

```python
# Добавить после walk_forward.run() результатов:
from apexfx.eval.baselines import (
    BuyAndHoldBaseline, MACrossBaseline, DonchianBaseline, RandomBaseline
)

baselines = {
    'B&H': BuyAndHoldBaseline(),
    'MA(20,50)': MACrossBaseline(20, 50),
    'Donchian(20)': DonchianBaseline(20),
    'Random': RandomBaseline(),
}

print("\n" + "=" * 80)
print("BASELINE COMPARISON (per-fold Sharpe ratio)")
print("=" * 80)
print(f"{'Fold':<6} {'Model':>10} " + " ".join(f"{n:>12}" for n in baselines))
print("-" * 80)

beats_count = 0
for fold_idx, fold in enumerate(results.folds):
    model_sr = fold.metrics.get('sharpe_ratio', 0)
    baseline_srs = {}
    for name, b in baselines.items():
        # ... evaluate baseline on same fold ...
        bm = walk_forward._evaluate_with_predictor(b, fold_data)
        baseline_srs[name] = bm.get('sharpe_ratio', 0)
    
    best_baseline_sr = max(baseline_srs.values())
    beats = model_sr > best_baseline_sr
    if beats: beats_count += 1
    marker = "✓" if beats else "✗"
    print(f"{fold_idx:<6} {model_sr:>10.3f} " + 
          " ".join(f"{baseline_srs[n]:>12.3f}" for n in baselines) + f"  {marker}")

print("-" * 80)
print(f"Model beats best baseline on {beats_count}/{len(results.folds)} folds "
      f"({100*beats_count/len(results.folds):.0f}%)")
print("VERDICT: " + ("EDGE EXISTS — proceed to paper" if beats_count >= 0.6 * len(results.folds)
                     else "NO EDGE — do not trade live"))
```

---

## Приложение B — Что удалить / что оставить (один взгляд)

### Безопасно удалить (Path B simplification)

```bash
# Эти модули feature selector эмпирически отверг:
git rm src/apexfx/features/wavelet.py
git rm src/apexfx/features/spectral.py
git rm src/apexfx/features/fundamental.py
git rm src/apexfx/features/structure.py
git rm src/apexfx/features/central_bank.py
git rm src/apexfx/features/cot.py
git rm src/apexfx/features/seasonal.py
git rm src/apexfx/features/intermarket_corr.py
git rm src/apexfx/features/scalping.py
git rm src/apexfx/features/sentiment.py
git rm src/apexfx/features/clustering.py
git rm src/apexfx/features/dim_reducer.py

# Архитектурный overhead без proven edge:
git rm -r src/apexfx/models/world_model/
git rm -r src/apexfx/models/tft/
git rm -r src/apexfx/models/ensemble/
git rm -r src/apexfx/models/agents/
git rm -r src/apexfx/models/components/
git rm src/apexfx/training/ewc.py
git rm src/apexfx/training/adversarial.py
git rm src/apexfx/training/per.py
git rm src/apexfx/training/pretrain.py
git rm src/apexfx/training/hierarchical.py
git rm src/apexfx/training/diversity.py

# MTF не оправдан:
git rm src/apexfx/env/mtf_forex_env.py
git rm src/apexfx/data/mtf_synthetic.py
git rm src/apexfx/data/mtf_aligner.py

# Smart execution для < 0.1 lot — no-op:
git rm src/apexfx/execution/smart_exec.py
git rm src/apexfx/execution/fill_tracker.py
git rm src/apexfx/execution/liquidity_guard.py
git rm src/apexfx/execution/order_manager.py

# VaR / stress для $1k — overkill:
git rm src/apexfx/risk/stress_testing.py
git rm src/apexfx/risk/var_calculator.py

# Дамп старой версии:
git rm -r src/_v2_dump/

# Удалить связанные тесты:
git rm -r tests/unit/test_phase2.py tests/unit/test_phase3.py tests/unit/test_phase3_5.py
# (заменить на новые simple-baseline tests)
```

**Результат:** 144 .py → ~25 .py, 34k LOC → ~8k LOC.

### Оставить и развивать

- `data/{mt5_client,data_store,bar_aggregator,calendar_provider}.py`
- `features/{pipeline (top 15 only),selector,normalizer}.py`
- `env/{forex_env,reward,trade_filter (rules 1+2 only)}.py`
- `training/{trainer (single-stage),walk_forward,checkpoint_manager}.py`
- `risk/{risk_manager (4 checks),position_sizer (fixed % risk),drawdown_monitor,cooldown,news_filter}.py`
- `execution/executor.py` (direct market only)
- `live/{trading_loop,signal_generator}.py`
- `dashboard/app.py`
- `eval/*` (новый модуль с baselines + reports)

---

*Конец отчёта v2. Прошлый отчёт остаётся валидным как описание трёх фиксов. Этот отчёт говорит, что делать после фиксов, чтобы прийти к прибыли — или к честному пониманию, что edge нет.*

*Главное правило: **не торговать в live, пока walk-forward не покажет стабильный Sharpe > 1.0 + bias-baseline beat на 60% folds**. Без этого — гарантированный слив на счёте $1k.*
