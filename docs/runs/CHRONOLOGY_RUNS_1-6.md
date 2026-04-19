# ApexFX Curriculum v2 — Полная хронология запусков 1–6

**Период:** 2026-04-18 00:53 → 2026-04-19 02:57 (≈26 часов)
**Платформа:** MWS GPU VM `82.202.157.240` (A100-PCIE-40GB, 24 vCPU, 236 GiB RAM)
**Бранч:** `v2.0-quantum-hybrid`
**Симвoл/таймфрейм:** EURUSD H1, 12354 баров (~3 года)
**Архитектура:** TQC (Truncated Quantile Critics) + WorldModelHybrid (Mamba) + GatingV2

---

## Сводная таблица всех запусков

| # | Дата старта | Seed | n_envs | Total steps | Дошли до | Best Sharpe | Best PF | Финал | Итог |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 18.04 00:53 | 42 | 1 (dummy) | 5M | stage 3 step 930K | -147 | 0.0 | OOS PF=0.93 | Завис в локальном минимуме (PF=0 во всех stage) |
| 2 | 18.04 10:37 | 42 | 16 (subproc) | 8M | stage 1 step 800K | -594k | 0.0 | прерван | Early-stop сработал на PF=0 (AND-логика) |
| 3 | 18.04 11:38 | 42 | 16 (subproc) | 8M | **завершён 4 stage** | -147 | 0.0 | **OOS: 0 trades** | EWC+adversarial убил политику в финальной стадии |
| 4 | 18.04 14:17 | 7 | 16 (subproc) | 5M | crashed step ~140K | 0.0 | 0.0 | crash | ConnectionResetError у subproc workers; reward bug — все компоненты =0 |
| 5 | 18.04 18:06 | 7 | 16 (subproc) | 5M | **завершён 4 stage** | **-58** | 0.0 | OOS: 165 trades, WR 35.76%, PF 0.74 | Прорыв (reward fix) но позиции <1% капитала |
| 6 | 19.04 00:15 | 7 | 16 (subproc) | 4.5M | stage 2 step 2.85M (~63%) | **-52** | 0.0 | прерван VM off | Лучший Sharpe из всех; **VM выкл по подписке** |

---

## RUN 1 — Baseline (seed 42, n_envs=1)

**Лог:** `logs/train_EURUSD_20260418_005103.log` (68 MB, в основном CUBLAS warnings)
**Чекпоинты:** `models/v2_checkpoints_run1/`
**Запущен:** 18.04.2026 00:53:33
**Завершён:** 18.04.2026 ~03:05 (по timestamp checkpoint latest)

### Конфигурация
- **Seed:** 42 (deterministic)
- **n_envs:** 1 (DummyVecEnv) — однопоточный, не использовал A100 толком
- **Total timesteps:** 5,000,000 (per-stage: 500K / 2M / 1.5M / 1M)
- **Buffer:** 1M, batch=256, learning_starts=10K
- **LR:** 3e-4 базовый, AdaptiveLR (0.1 boost, 0.7 reduce)
- **Reward:** RARAReward_v5 с **багом** (см. ниже)
- **EWC:** lambda=5000

### Что произошло
- Stage 0 (real_warmup, 500K шагов) — старт sharpe=-1.96M (катастрофически плохо)
- Постепенный прогресс: -1.96M → -685k за 50K шагов
- Лог обрывается после 50K (CUBLAS warnings залили файл — реальные info-строки потеряны)
- **Финал по checkpoint:** stage 3, step 930K, sharpe=-146.69, **PF=0.0**
- ep_rew_mean=-2646, n_episodes=81, gating_entropy=0.031 (коллапс гейтинга)

### Проблемы
1. **Single-env throughput:** A100 простаивал, обучение медленное
2. **PF=0 во всех stage** — агент не открывал прибыльных сделок
3. **Энтропия гейтинга 0.031** — модель использовала 1-2 экспертa из ансамбля
4. **Ep_rew = -2646** = интегрированная inactivity penalty за 2000-step flat episode → **агент вообще не торгует**

### Корневая причина (выявлена позже в Run 4 анализе)
`forex_env.step()` НЕ вызывал `set_step_context()` для `RARAReward_v5` (отсутствовал в isinstance chain). Все position-aware компоненты (realized_pnl×10000, unrealized_delta, trade_cost, jump, fsd, gamma, structure) **оставались 0 за весь эпизод**. Жил только inactivity ramp.

→ **Run 1 был "пустышкой": награда была одномерной (только штраф за бездействие)**

---

## RUN 2 — Параллелизация (seed 42, n_envs=16)

**Лог:** `logs/train_EURUSD_20260418_103757.log` (14 KB)
**Чекпоинты:** `models/v2_checkpoints_run2/`
**Запущен:** 18.04.2026 10:37:59
**Прерван:** 18.04.2026 11:35:07 (early-stop)

### Что изменилось vs Run 1
- **n_envs: 1 → 16** (SubprocVecEnv) — параллельный сбор опыта на 16 копиях env
- **Total timesteps: 5M → 8M** (per-stage scaled: 800K / 3.2M / 2.4M / 1.6M)
- Прочее идентично Run 1 (тот же reward bug)

### Что произошло
- Stage 0: старт sharpe=-720k → -604k за 230K шагов
- **10:58:38: Early stopping на step 380K** (раньше плана 800K)
- Stage 0 итог: sharpe=-658k, PF=0, ep_rew=-2814, entropy=0.1
- Перешли в Stage 1 (real_full, +30% SBBTS синтетика)
- Stage 1: sharpe не улучшался, на step 800K — PF=0
- **11:35:07: Early stopping на step 800K** (вместо плана 3.2M)
- Stage 1 итог: sharpe=**-827968** (упал!), PF=0
- Run прерван перед Stage 2

### Проблемы
1. **Early-stop срабатывает на PF=0** — `MultiMetricEarlyStopConfig` использует AND-логику: если PF тривиально 0 → сразу stale
2. **Sharpe деградировал** в Stage 1 (-658k → -828k) при warm-start с худшего чекпоинта
3. **Reward bug всё ещё активен** — те же -2813 ep_rew = чистая inactivity

---

## RUN 3 — Полный 4-стадийный прогон (seed 42, n_envs=16)

**Лог:** `logs/train_EURUSD_20260418_113833.log` (27 KB)
**Чекпоинты:** `models/v2_checkpoints/` (default dir, перезапис)
**Запущен:** 18.04.2026 11:38:35
**Завершён:** 18.04.2026 13:24:49 (полные 4 стадии за ~1ч 46мин)

### Что изменилось vs Run 2
- Похоже отключён ранний останов (или ослаблен) — иначе бы прервался
- Целились: 8M шагов

### Что произошло
| Stage | Sharpe начало | Sharpe конец | PF | ep_rew | Entropy | Длительность |
|---|---|---|---|---|---|---|
| 0 (warmup) | -1.7M | **-701k** | 0.0 | -2813 | 0.0036 | ~27 мин |
| 1 (real_full) | -701k | **-594k** | 0.0 | -2813 | 0.0093 | ~32 мин |
| 2 (augmented) | -594k | **-120k** | 0.0 | -2813 | 0.0148 | ~30 мин |
| 3 (adversarial) | -120k | **-147** | 0.0 | -2646 | 0.0311 | ~47 мин |

### Финальный результат
- Все 4 стадии завершены номинально
- Sharpe прыгнул с -120k до -147 в Stage 3 (в 800 раз!)
- **Но PF=0 во всех 4 стадиях**
- ep_rew улучшился только в Stage 3: -2813 → -2646 (значит начал чуть-чуть торговать)
- Gating entropy 0.0036–0.031 — **сильный коллапс**, модель использует 1 эксперта

### OOS бэктест (artifact: `backtest_results.json`)
- 173 сделки, WR 44.55%, **PF=0.957** (около безубытка)
- Final balance: $99,822 / $100k = **-0.18%**
- Max drawdown 0.48%
- Sharpe ratio: -0.241

### Выводы Run 3
- Reward bug всё ещё есть, но gain в Stage 3 показал что агент *что-то* выучил
- Однако итоговая модель — флэт-политика с 0 trade signal
- **Все ранние решения (LR boost, EWC, adversarial) дали результат = 0**

---

## RUN 4 — Reward fix + early-stop tuning (seed 7, n_envs=16) — CRASHED

**Лог:** `logs/train_EURUSD_20260418_141703_run4.log` (24 KB)
**Чекпоинты:** `models/v2_checkpoints_run4/`
**Запущен:** 18.04.2026 14:17:05
**Прерван:** 18.04.2026 ~14:37 (ConnectionResetError у workers)

### Коммит изменений: `f5b0107` "fix: reward_v5 and early-stop tuning for Run 4"
- **Seed: 42 → 7** (сломать deterministic lock Run 1/3)
- `loss_asymmetry: 1.6 → 1.0` (меньше асимметрии)
- `cvar_weight: 0.3 → 0.0` (выкл CVaR penalty)
- `early_stop.patience: 15 → 60` (4× терпения)
- `early_stop.min_delta_pf: 0.01 → -1.0` (фактически выкл PF в early-stop)

### Анализ (из commit message Run 4)
> Run 1-3 (PF=0 all stages, Sharpe plateau -701k → -147) showed CVaR penalty + asymmetric loss dominate reward, making any exploration net-negative by Bellman. Agent converges to flat policy. Multi-metric early stop then kills stages at 18-27% of budget because PF=0 trivially triggers stall on the AND-logic.

### Что произошло
- Stage 0 запустился, дошёл до step ~140K
- LR бустился: 3e-4 → 4.5e-4 → 6.75e-4 → 1e-3 за первые 140K шагов (низкая энтропия)
- **ConnectionResetError**: pickle pipe умер у subproc workers (massive write spam)
- Run 4 крашнулся
- Latest checkpoint: stage 0, step 450K, sharpe=0.0, PF=0, ep_rew=-1758
- **ep_rew = -1758** именно соответствует чистой inactivity penalty (отгадка!) — Reward bug **подтверждён**

### Открытие после Run 4 (см. коммит `e856a7c`)
> Root cause for Run 1/3/4 "always flat" failure: forex_env.step() only dispatched per-step context (set_trade_info/set_position_info) to QuantumZScoreReward / HoldAwareReward / ProfitFocusedReward / TradingReward. **RARAReward_v5 inherits from BaseRewardFunction directly and was missing from the isinstance chain**, so its internal position/realized_pnl state was never updated.
>
> Integrated inactivity over a 2000-step flat episode = -1758.027, which matches Run 4 ep_rew_mean exactly (Run 1 = -1758 × 1.6 = -2813 with old loss_asymmetry).

→ **3 запуска коту под хвост из-за одного отсутствующего isinstance branch**

---

## RUN 5 — REWARD FIXED (seed 7, n_envs=16) — ПЕРВЫЙ РАБОЧИЙ

**Лог:** `logs/train_EURUSD_20260418_180631_run5.log` (33 KB)
**Чекпоинты:** `models/v2_checkpoints_run5/`
**Запущен:** 18.04.2026 18:06:35
**Завершён:** 18.04.2026 21:58:07 (полные 4 стадии за ~3ч 52мин)

### Коммит `e856a7c` "fix: unblock RARAReward_v5"
- `forex_env.py`: добавлена `isinstance(RARAReward_v5)` ветка → вызывает `set_step_context(position, unrealized_pnl, realized_pnl, price, time_in_position, fsd_regime, structure_aligned)` каждый step
- `reward_v5.py`:
  - `realized_pnl_weight: 10000 → 2000` (избегаем reward clip saturation)
  - `reward_clip: 30 → 50` (шире окно)
  - `inactivity_weight: 0.0002 → 0.00005` (мягче штраф, 1758 → 440 на эпизод)
- `trainer_v2.py`: `ent_coef: auto → 0.1` (фикс) — авто-тюнер коллапсил с 0.66 до 0.0018
- 20/20 reward_v5 тестов прошли, 34/34 env+curriculum

### Что произошло (правильные числа!)
| Stage | Steps | Sharpe нач | Sharpe кон | PF | ep_rew | Entropy | Длительность |
|---|---|---|---|---|---|---|---|
| 0 (warmup) | 500K | -190 | **-142** | 0.0 | -82 | 0.1 | ~23 мин |
| 1 (real_full) | 2M | -142 | **-119** | 0.0 | -83 | 0.1 | ~1ч 30мин |
| 2 (augmented) | 1.5M | -119 | **-119** | 0.0 | -99 | 0.1 | ~1ч 10мин |
| 3 (adversarial) | 1M | -119 | **-65** ★ | 0.0 | **-249** | 0.1 | ~49 мин |

★ **best_sharpe в Stage 3: -58.04 на step 1.23M**

### OOS бэктест (artifact: `artifacts/run5/oos_report.json`)
- **best_sharpe checkpoint:**
  - 165 trades opened, **165 closed**
  - Trade WR: **35.76%**
  - Trade PF: **0.736**
  - Gross profit $13.30 / Gross loss $18.07
  - Avg win $0.226 / Avg loss $-0.171
  - Final balance $99,995 / $100k = **-0.005%**
  - **Max drawdown $7.9 на $100k = 0.0079%** (микропозиции!)
- **v2_final checkpoint: 0 trades** на 165 OOS барах ← катастрофа

### Выводы Run 5 (POSTMORTEM в коммите `dbc5bd6`)
1. **`v2_final` мёртв:** EWC + adversarial stage 4 закостенил политику → 0 trades
2. **`best_sharpe` живой**, но позиции в долях процента капитала ($10 max DD на $100k → notional ~$50)
3. **PF=0 в логах vs PF=0.74 в OOS** — расхождение объясняется тем, что внутренний PF в callback считает только закрытые сделки в окне последних эпизодов, OOS считает все 165 за период
4. `realized_pnl_weight=2000` слишком слабый — агент видит реализованный PnL как шум на фоне unrealized_delta
5. **MC-dropout uncertainty** душил позиции (`min_position_scale=0.1` → max 10% nominal)

→ **Run 5 = первый запуск где агент реально торгует, но микро-позициями**

---

## RUN 6 — Catastrophic forgetting + micro-position fix (seed 7, n_envs=16) — ПРЕРВАН

**Лог:** `logs/train_EURUSD_20260419_001555_run6.log` (24 KB, до 02:21)
**Чекпоинты:** `models/v2_checkpoints_run6/`
**Запущен:** 19.04.2026 00:15:57
**Прерван:** 19.04.2026 ~02:57 (последний checkpoint), VM выкл по окончанию подписки

### Коммит `dbc5bd6` "feat: Run 6 — fix catastrophic forgetting + micro-position problem"
- `reward_v5.py`: `realized_pnl_weight: 2000 → **4000**` (×2 сигнал от закрытых сделок)
- `configs/training.yaml`:
  - `stage 3 (real_adversarial) total_timesteps: 1M → **500K**` (меньше adversarial pressure)
  - `stage 3 noise_std: 0.005 → **0.003**` (мягче adversarial шум)
  - `stage 3 price_shift_std: 0.003 → 0.002`
  - `ewc.lambda_ewc: 5000 → **2000**` (меньше rigid EWC)
- `configs/model.yaml`:
  - `uncertainty.uncertainty_weight: 0.5 → **0.2**` (меньше throttle от MC-dropout)
  - `uncertainty.min_position_scale: 0.1 → **0.3**` (минимум 30% nominal вместо 10%)
- `scripts/eval_run5_oos.py` — новый OOS eval скрипт (195 строк)

### Коммит `a7b611e` "fix: Run 6 — port stage 4 adversarial softening to CurriculumV2Config"
**Критическое открытие:** trainer_v2 использовал **hardcoded defaults** в `src/apexfx/training/config.py` и **игнорировал yaml**! Run 6 первоначально стартовал с `stage 3 = 1M шагов, noise=0.01` (старые значения).

Run 6 был **убит** (PID 2886 + 4 orphan workers), config.py отредактирован вручную:
```python
StageConfig(
    name="real_adversarial",
    total_timesteps=500_000,  # was 1_000_000
    noise_std=0.003,          # was 0.01
    ...)
```
Перезапущен 00:15:57 — корректно.

### Что произошло (Run 6 actual run)
| Stage | Steps | Sharpe нач | Sharpe кон | PF | ep_rew | Entropy | Длит |
|---|---|---|---|---|---|---|---|
| 0 (warmup) | 500K | -182 | **-118** | 0.0 | -96 | 0.1 | ~23 мин |
| 1 (real_full) | 2M | -118 | **-100** | 0.0 | -96 | 0.1 | ~1ч 30мин |
| 2 (augmented) | 1.5M (план) | -100 | **прерван на step 2.85M** | 0.0 | -125 | 0.1 | прерван |

### Ключевые события Run 6
- **00:21:08** Stage 0 step 40K: первые best (sharpe=-182, reward=-88) — лучший старт из всех runs!
- **00:38:49** Stage 0 завершён: sharpe=-118, PF=0, **ep_rew=-96 vs Run 5: -82** (немного хуже)
- **01:50:20** Stage 1 step 1.86M: sharpe=-83.10 (новый best)
- **02:08:51** Stage 1 завершён: sharpe=**-99.85**, PF=0 (Run 5: -118.6 — лучше чем Run 5!)
- **02:20:28** Stage 2 step 1.9M: **sharpe=-52.27** ★ — лучший Sharpe из всех 6 запусков
- **02:57:51** latest checkpoint: stage 2, step 2.85M, sharpe=-75, ep_rew=-125, **entropy=0.1**
- **02:21+:** лог обрывается (последняя запись), но checkpoint обновлялся до 02:57

### Прогресс на момент остановки
- Дошли до **2.85M из 4.5M шагов = 63%**
- Stage 2 (real_augmented) **активен**, не завершён
- Stage 3 (real_adversarial) **не запущен**
- VM выключилась около 03:00 (подписка истекла)

### Чекпоинты Run 6 (сохранены локально)
| Checkpoint | Step | Stage | Sharpe | PF | ep_rew | Entropy | Время |
|---|---|---|---|---|---|---|---|
| best_entropy | 40K | 0 | -182 | 0.0 | -88 | 0.1 | 00:21 |
| best_reward | 40K | 0 | -182 | 0.0 | -88 | 0.1 | 00:21 |
| **best_sharpe** | 1.9M | 2 | **-52.27** | 0.0 | — | 0.1 | 02:20 |
| latest | 2.85M | 2 | -75.44 | 0.0 | -125 | 0.1 | 02:57 |

### Выводы Run 6 (промежуточные, обучение не закончено)
1. **Лучший Sharpe из всех runs (-52.27)** на середине Stage 2 — позитивная динамика
2. **Profit Factor всё ещё 0.0** — двойной realized_pnl_weight НЕ решил проблему
3. **Entropy=0.1 (=fixed ent_coef)** — гейтинг **по-прежнему коллапсирован**, fix `ent_coef: auto → 0.02→0.1` дал floor но не разнообразие
4. ep_rew негативный (-96…-125) — общий профиль реварда отрицательный
5. **Stage 3 не дошли** — главный adversarial fix (noise softening) не успел проявиться

### Открытые вопросы Run 6
- Был бы PF>0 если бы дошли до stage 3? (в Run 5 PF был 0 во всех stage внутренне, но 0.74 в OOS)
- Решил бы `min_position_scale=0.3` проблему микро-позиций? (не успели проверить)

---

## ОБЩАЯ ХРОНОЛОГИЯ ИСПРАВЛЕНИЙ

```
Run 1 (seed=42, n_envs=1)
  └─ Reward bug: RARAReward_v5 не получает context → only inactivity penalty
  └─ Result: PF=0 во всех 4 stages, ep_rew=-2646
  
Run 2 (seed=42, n_envs=16)  
  └─ Fix: Параллелизация → A100 утилизирована
  └─ Bug сохраняется + early-stop ругается на PF=0 (AND-logic)
  └─ Result: Прерван в stage 1, sharpe деградировал -658k → -828k
  
Run 3 (seed=42, n_envs=16)
  └─ Только параллельная инфра, bug не тронут
  └─ Result: 4 стадии завершены, sharpe -147, PF=0, OOS PF=0.957 (174 trades, флэт)
  
Run 4 (seed=7, n_envs=16) — CRASH
  └─ Fix: loss_asymmetry 1.6→1.0, cvar_weight 0.3→0.0, early_stop_patience 15→60
  └─ Bug всё равно активен → ep_rew=-1758 (точно inactivity floor)
  └─ Result: ConnectionResetError у subproc, ep_rew подтверждает bug
  └─ ПОСЛЕ Run 4: найден root cause (отсутствие isinstance branch)
  
Run 5 (seed=7, n_envs=16) — ПЕРВЫЙ РАБОЧИЙ
  └─ Fix: forex_env.step() добавлен isinstance(RARAReward_v5)
  └─ Fix: realized_pnl_weight 10000→2000, ent_coef auto→0.1
  └─ Result: 4 стадии, sharpe -65, PF=0, OOS: 165 trades, WR 36%, PF 0.74
  └─ ПРОБЛЕМА: микро-позиции (0.0079% max DD), v2_final = 0 trades
  
Run 6 (seed=7, n_envs=16) — ПРЕРВАН VM
  └─ Fix: realized_pnl_weight 2000→4000 (x2 сигнал)
  └─ Fix: stage 3 timesteps 1M→500K (меньше catastrophic forgetting)
  └─ Fix: ewc_lambda 5000→2000 (меньше rigid policy)
  └─ Fix: uncertainty.min_position_scale 0.1→0.3 (минимум 30% notional)
  └─ Fix: ВТОРОЙ проход — обнаружено что yaml не читается trainer_v2,
          порт изменений в src/apexfx/training/config.py defaults
  └─ Result: 63% обучения, sharpe -52 (лучший!), PF=0, прерван VM
```

---

## КЛЮЧЕВЫЕ МЕТРИКИ ПО ЗАПУСКАМ

### Best Sharpe эволюция
```
Run 1: -147       (4 stage завершены, но bug)
Run 2: -827k      (прерван stage 1, bug)
Run 3: -147       (4 stage завершены, bug)
Run 4: 0.0        (crash на step 140K)
Run 5: -58.04     (4 stage завершены, FIX!)  ← первый осмысленный
Run 6: -52.27     (63% выполнено, прерван)   ← лучший на момент остановки
```

### Profit Factor (внутренняя метрика в callback)
- **Все 6 запусков: PF=0.0 во всех stage** (проблема сохраняется)

### OOS PF (бэктест на отложенных данных)
- Run 3: PF 0.957 (174 trades, near-flat)
- Run 5 best_sharpe: **PF 0.736** (165 trades, micro positions)
- Run 5 v2_final: **0 trades** (политика ossified)
- Run 6: **не оценено** (требуется eval с локальных чекпоинтов)

### ep_rew_mean эволюция
- Run 1-3: -2646…-2813 (reward bug, inactivity-dominated)
- Run 4: -1758 (bug, новый ent_coef, but inactivity)
- Run 5: -65…-249 (FIX, варьируется по стадиям)
- Run 6: -96…-125 (стабильнее чем Run 5 в первых стадиях)

### Gating Entropy
- Run 1-3: 0.003–0.031 (массовый коллапс гейтинга)
- Run 4: 0.5694 (на старте, до краша)
- Run 5-6: 0.1 (зафиксирован floor через ent_coef=0.1, но floor а не diversity)

---

## ОТКРЫТЫЕ ПРОБЛЕМЫ К RUN 7

### 1. Profit Factor = 0 во всех внутренних метриках
- В callback PF считается за окно эпизодов, всегда 0
- Возможно проблема в том как `n_episodes` агрегируется с n_envs=16
- В OOS PF растёт до 0.74-0.93 → есть какая-то *локальная прибыль*

### 2. Микро-позиции
- В Run 5 max drawdown $7.9 на $100k капитале (0.008%)
- Notional каждой сделки ~$50 — модель боится лоссов
- Run 6 повысил `min_position_scale` 0.1→0.3, но не успели проверить

### 3. Catastrophic forgetting в stage 3
- Run 3 final: 0 trades на OOS (политика умерла после adversarial)
- Run 5 best_sharpe (stage 3 раннее) лучше чем v2_final
- Run 6 уменьшил adversarial stage 1M→500K — не успели проверить эффект

### 4. Gating entropy floor != diversity
- ent_coef=0.1 даёт *среднюю* энтропию 0.1, но 1-2 эксперта доминируют
- Анти-collapse loss есть в gating_v2, но weight=1.0 видимо мало
- Реальная diversity не отслеживается отдельно

### 5. Yaml configs игнорируются
- **Найдено в Run 6**: `trainer_v2.py → CurriculumV2Config()` использует hardcoded defaults
- yaml файлы (`configs/training.yaml`) **не парсятся**!
- Все параметры стадий, EWC, adversarial живут в `src/apexfx/training/config.py`
- Любые изменения требуют патча code, а не config

### 6. EWC consolidation = no-op
- В trainer_v2 `_consolidate_ewc()` это **пустая заглушка**
- При этом log пишет "EWC consolidation stage=N" — впечатление что работает
- EWC lambda меняли (5000→2000) — но **функция не делает ничего**

---

## РЕКОМЕНДАЦИИ ДЛЯ RUN 7

### Приоритет HIGH
1. **Подключить yaml configs к trainer_v2** или удалить hardcoded defaults
2. **Удалить EWC заглушки** или реализовать настоящий Fisher-information consolidation
3. **Прогнать OOS eval Run 6 best_sharpe** на локальных данных — проверить торгует ли с лучшим sharpe
4. **Старт с best_sharpe Run 6** в качестве warm_start (skip stage 0-1, начать с stage 2)

### Приоритет MEDIUM
5. **Reward shaping**: попробовать `realized_pnl_weight=8000` (×2 vs Run 6) если micro-positions сохранились
6. **Position sizing**: экспонента вместо linear `position_scale = exp(certainty - 1)` для агрессивных позиций при confidence
7. **Gating diversity bonus**: добавить explicit reward за энтропию выборки экспертов >0.3

### Приоритет LOW
8. **Curriculum redesign**: возможно убрать adversarial stage целиком (bias-variance: уменьшает test perf)
9. **Multi-symbol training**: уже реализовано в `MultiSymbolConfig`, не использовалось — добавить GBPUSD/USDJPY
10. **MC-Dropout**: проверить что uncertainty estimator вообще на train path (был отдельным wrapper)

---

## ИНФРАСТРУКТУРА И ИНЦИДЕНТЫ

### Технические инциденты
- **Run 4 crash**: `ConnectionResetError` у SubprocVecEnv workers — pickle pipe overload
- **Run 6 wrong config**: yaml не читался, требовался re-launch с hardcoded edit
- **VM SSH timeouts**: 3 раза за сессию — banner exchange timeout (MWS egress quirk, лечится IP rotation)
- **Orphan workers**: после SIGTERM main PID, 16 SubprocVecEnv workers выживали → требуется `pkill -KILL -f train_v2.py`

### Backup стратегия (применена в Run 6)
1. **Watchdog на VM**: `/home/user1/apexfx/run6_watchdog.sh` (PID 27914) — мониторит train PID, при завершении делает tarball
2. **Local incremental rsync**: периодический `rsync -avz` с VM в `/Users/abobik/Desktop/M/vm_snapshot_20260419_024501/`
3. **Защита от auto-shutdown**: оба механизма независимы

### Финальный бэкап
- `vm_snapshot_20260419_024501/` (1.3 GB) содержит:
  - `models/` (1.2 GB) — все 6 runs чекпоинтов + v2_final
  - `logs/` (65 MB) — все training логи
  - `runs/` (664 KB) — TensorBoard event files
  - `data_cache/` (6 MB) — features parquet
  - `backtest_results.json` (Run 3 OOS)
  - `artifacts/run5/oos_report.json` (Run 5 OOS)

---

## ВРЕМЕННА́Я ЛИНИЯ (timeline)

```
2026-04-18
─ 00:53  Run 1 старт (seed=42, n_envs=1, reward bug active)
─ 03:05  Run 1 финал — sharpe=-147, PF=0, 4 stages
─ 10:38  Run 2 старт (n_envs=16)
─ 11:35  Run 2 прерван early-stop (stage 1)
─ 11:38  Run 3 старт
─ 13:25  Run 3 финал — sharpe=-147, PF=0, 4 stages, OOS PF=0.957
─ 14:17  Run 4 старт (seed=7, reward tuning)
─ 14:37  Run 4 CRASH (ConnectionResetError)
─ 14:??  Reward bug найден (commit e856a7c)
─ 18:07  Run 5 старт (REWARD FIXED)
─ 21:58  Run 5 финал — sharpe=-58, PF=0, OOS: 165 trades WR 36% PF 0.74
─ 22:53  Run 6 commit (dbc5bd6 micro-position + ewc fixes)
─ 23:10+ Run 6 старт (первая попытка — yaml игнорируется)

2026-04-19
─ 00:00  Run 6 убит, config.py edited (commit a7b611e), relaunch
─ 00:16  Run 6 старт (правильно)
─ 02:08  Run 6 stage 1 завершён (sharpe=-100)
─ 02:20  Run 6 best_sharpe achieved: -52.27 (stage 2 step 1.9M) ★
─ 02:21  Лог последняя запись
─ 02:45  Local snapshot rsync
─ 02:57  latest checkpoint обновлён (step 2.85M)
─ ~03:00 VM ВЫКЛЮЧЕНА (subscription expired)
```

---

**Документ составлен:** 2026-04-19 после остановки VM
**Источники:** training логи, checkpoint metadata.json, git history (8 коммитов 18-19 апреля), backtest_results.json, Run 5 OOS report
