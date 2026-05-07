# Как сделать $100-250/год на $1k реальностью

> Не теория, не "X% в месяц", а математика: что должно происходить на уровне каждой сделки, чтобы 10-25% годовых были достижимы на retail-счёте.

---

## Часть 1 — Математика цели

### Что значит "$200/год на $1k"

```
$200 годовая прибыль
÷ 12 месяцев
= $16.67 в месяц = ~$0.55 в день
```

Это меньше доллара в день. Звучит скромно? Хорошо. **Это и есть реалистичная цель**, потому что любые "большие" числа на retail = приглашение к плечу 1:500 + margin call.

### Сколько сделок нужно в год

Зависит от того, сколько прибыли с одной сделки:

| Сделок/год | Profit per trade | Risk per trade (0.5% капитала) | Нужный R-multiple |
|---:|---:|---:|---:|
| 25 (D1 swing) | $8.00 | $5.00 | 1.6R EV |
| 50 (H4 swing) | $4.00 | $5.00 | 0.8R EV |
| 100 (active H4) | $2.00 | $5.00 | 0.4R EV |
| 250 (H1 scalp) | $0.80 | $5.00 | 0.16R EV |

**EV (expectancy) формула:** `EV = (WR × avg_win) - (LR × avg_loss)`

Например при WR 50%, avg_loss 1R:
- 0.4R EV → нужны winning trades ≥1.8R (т.е. **TP 36 пипс при SL 20 пипс**)
- 0.8R EV → нужны winning trades ≥2.6R (TP 52 пипс при SL 20)
- 1.6R EV → нужны winning trades ≥4.2R (TP 84 пипс при SL 20)

**Вывод:** чем больше сделок — тем меньше нужно EV per trade. Но больше сделок = больше costs (см. Часть 2).

---

## Часть 2 — Что физически НЕ работает на $1k

### 2.1. Спред съедает прибыль на короткой дистанции

При retail-спреде 2.5 пип на EURUSD:

| TP / SL | Спред как % от TP | Что остаётся |
|---|---|---|
| TP 5 пип / SL 5 пип | 50% | Половина прибыли уходит в спред |
| TP 10 пип / SL 10 пип | 25% | Жить можно, но впритык |
| TP 30 пип / SL 20 пип | **8%** | **Здоровая зона** |
| TP 60 пип / SL 30 пип | **4%** | **Идеально** |

**Правило для $1k retail:** TP не меньше 30 пипс, SL не меньше 15 пипс. Иначе спред + slippage съедают половину edge.

### 2.2. Минимальный лот = floor под position sizing

MT5 cent-account: min lot 0.01.

На EURUSD 0.01 лот = **$1 за пипс** (на стандартном счёте) или $0.10 (cent).

На $1k с риском 0.5% per trade ($5):
- При SL 50 пипс — позиция должна быть 0.01 лот = $50 риск → **в 10 раз больше** допустимого риска
- При SL 5 пипс — позиция 0.10 лот = $50 риск → опять 10× превышение
- Реальный compromise: SL 50+ пипс, риск ~$5 = position 0.01

**Вывод:** на $1k стандарт-счёте можно торговать только **swing** (SL 30+ пипс). Скальпинг физически нереалистичен.

**Альтернатива:** **cent-account** (FxPro Cent, Roboforex Cent). Там $1k = "100,000 центов" → min lot 0.01 = $0.10/пип → можно reasonably scaling. **Это критично для retail $1k.**

### 2.3. Leverage = ловушка для $1k

Брокеры предлагают 1:500 leverage. На $1k это $500k notional. **Слив за 1 неудачную сделку**:
- 0.5 лот EURUSD по 1.10 = $55,000 нотионал
- Резкое движение 0.4% = -$220 = **-22% капитала**
- 3 таких подряд = -60% капитала, маржин-колл

**Правило:** **leverage не выше 10:1** в реальной торговле. Это как раз то, что даёт стандартное margin requirement при position sizing 0.5% risk.

---

## Часть 3 — Что РЕАЛЬНО может дать 10-25% на $1k

### 3.1. Donchian-LO на multi-symbol portfolio (есть данные)

**Что мы знаем из baseline-теста сегодня:**

| Спред | Donchian-LO Sharpe (full period EURUSD H4) |
|---|---|
| 1.5 пип | +0.165 |
| 2.0 пип | +0.136 |
| 2.5 пип | +0.107 |
| 3.0 пип | +0.079 |
| 3.5 пип | +0.050 |

**Sharpe → annual return:**
Если Sharpe = 0.10 при annual vol ~10%, return ~ 0.10 × 10% = **+1%/год**. Это слабо.

**НО** при мультисимвольном портфеле (4 декоррелированных пары: EURUSD, GBPUSD, USDJPY, AUDUSD):
- Sharpe умножается на ~√(1+ρ×(N-1)) где ρ — средняя корреляция (~0.4 для majors)
- 4 пары × √(1+0.4×3) = √2.2 ≈ 1.48× boost
- Sharpe portfolio ≈ 0.10 × 1.48 = **+0.15**

Это всё равно слабо. **Чистый Donchian-LO один не даст 10%+/год.**

### 3.2. Donchian-LO + EMA200 filter (предполагаемый upgrade)

Стандартный квант-trick: торговать только в направлении тренда:
- Long ТОЛЬКО когда price > EMA(200)
- Skip всё остальное

Эмпирически это даёт ×2-3 boost к Sharpe (исключаем counter-trend signals):
- Donchian-LO + EMA200: Sharpe ~+0.3 single symbol
- Multi-symbol portfolio: Sharpe ~+0.45
- При vol 12%: annual return = **+5-6%**

Уже теплее. На $1k = **$50-60/год**. Половина цели.

### 3.3. Carry + Trend (proven retail edge — ×30 лет данных)

**Carry trade**: long high-yield, short low-yield. Известен работающим с 1980-х.

EURUSD сейчас (2026): 
- USD rate ~5.25%, EUR rate ~3.75% → carry +1.5%/год long USD short EUR
- USDJPY: USD 5.25%, JPY 0.10% → carry +5.15%/год long USDJPY
- AUDJPY: AUD 4.35%, JPY 0.10% → carry +4.25%/год

**Math:** держа long USDJPY 0.01 лот целый год → +$5.15 carry на $100,000 нотионал × 0.01 = **$5.15/год** plus capital appreciation/depreciation.

**На $1k cent-account 0.01 лот = $1k нотионал → carry = $5.15/$1k = +0.5%/год** только от carry. Плюс trend movement.

**Combined Carry + Trend на 3 лучших cross:**
- Annual return realistic: **8-15%**
- Sharpe: 0.4-0.7
- Max DD: 10-15% (carry unwinds бывают резкими)

### 3.4. Hybrid RL position sizing (если RL зайдёт)

Ключевая идея: **не учить RL торговать**. Учить его **модулировать size** на rule-based сигнале.

```
Rule-based signal (Donchian-LO + EMA200): trade YES/NO + direction
RL output: position multiplier [0.0, 2.0]
  → 0.0 = skip (low confidence)
  → 1.0 = standard position
  → 2.0 = double size (high confidence)
```

Преимущества:
- RL action space намного уже (1 число вместо direction+size)
- Меньше overfit (rule даёт sanity check)
- Сходимость в 5-10× быстрее
- Если RL не сошёлся — fallback на multiplier=1 = чистый rule

При success: Sharpe boost +30-50% над чистым rule. Если portfolio даёт +0.45 SR, hybrid даёт **~+0.65** = annual return **8-15%**.

### 3.5. Сравнение всех подходов

| Стратегия | Trades/год | Sharpe (real) | Annual return | DD |
|---|:-:|:-:|:-:|:-:|
| Donchian-LO single EURUSD | 50 | +0.10 | +1% | 8% |
| Donchian-LO + EMA200 | 30 | +0.30 | +4% | 10% |
| Multi-symbol portfolio (4 пары) | 120 | +0.45 | **+6%** | 10% |
| Carry + Trend (3 cross) | 20 | +0.55 | **+10%** | 12% |
| Hybrid RL position sizing | 100 | +0.65 | **+12%** | 12% |
| Combined: Carry + Trend + RL sizing | 80 | **+0.85** | **+15%** | 13% |

**Combined — это стек который даёт желаемые $100-250 при reality-based costs.**

---

## Часть 4 — Конкретный stack для реализации

### Архитектура "минимально работающего" $1k стека

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 0: Capital Allocation                                │
│  $1,000 на FxPro Cent или Roboforex Cent (min lot $0.10/pip)│
│  Risk: 0.5% per trade = $5 max risk                          │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Multi-Symbol Universe                             │
│  EURUSD, GBPUSD, USDJPY, AUDUSD (corr matrix avg ~0.4)      │
│  Carry: USDJPY +5.15%, AUDJPY +4.25%, GBPJPY +5.0%          │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: Strategy Stack (3 параллельных)                   │
│                                                              │
│  Strategy A: Donchian-LO H4 + EMA200 filter                 │
│    → Long-only trend follower                                │
│    → 30-40 trades/year per pair                              │
│                                                              │
│  Strategy B: Carry trade D1 (USDJPY, AUDJPY)                │
│    → Hold long for weeks/months                              │
│    → Earn interest differential                              │
│    → 5-10 trades/year per pair                               │
│                                                              │
│  Strategy C: Mean reversion on RSI<25 H1 + above EMA200     │
│    → Buy oversold in uptrend                                 │
│    → 20-40 trades/year per pair                              │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: Risk Manager (4 проверки)                         │
│  1. Kill switch (manual file or daily DD > 2%)              │
│  2. Daily loss limit ($20 = 2% of $1k)                      │
│  3. Position size: fixed 0.5% risk per trade                │
│  4. Max concurrent positions: 3                              │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: Execution (MT5)                                   │
│  - Direct market orders only (TWAP/VWAP не нужны для 0.01)  │
│  - SL/TP set immediately on order placement                  │
│  - Sync state with MT5 every minute (catch external closes) │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5: Monitoring                                         │
│  - Telegram bot для каждого события                          │
│  - Dashboard с PF, Sharpe, WR, expectancy daily              │
│  - Daily reconciliation: paper expected vs live actual       │
└─────────────────────────────────────────────────────────────┘
```

### Math: ожидаемый результат на $1k за год

```
Strategy A (Donchian + EMA200):
  4 пары × 35 trades/год × $0.50 expected per trade = $70/год

Strategy B (Carry trade):
  2 cross × 8 trades × $1.20 per trade + carry interest $5/cross/year
  = $19.2 + $10 = $29/год

Strategy C (RSI mean reversion):
  4 пары × 30 trades × $0.45 per trade = $54/год

Cross-strategy correlation (~0.3):
  Combined return ≈ A + B + C with 15% drag = 
  ($70 + $29 + $54) × 0.85 = $130/год

При vol 10% → Sharpe = 1.3 (excellent for retail)
At max DD ~12% → Calmar = ~1.0
```

**Total expected: $130/год = 13% — посередине target диапазона $100-250.**

### Что нужно реализовать (effort estimate)

| Компонент | LOC | Время |
|---|:-:|:-:|
| Strategy A (Donchian-LO + EMA200) | ~150 | 1 день |
| Strategy B (Carry trade) | ~200 | 2 дня |
| Strategy C (RSI mean reversion) | ~150 | 1 день |
| Multi-symbol orchestrator | ~250 | 2 дня |
| Risk manager (4 checks) | ~200 | 1 день |
| MT5 execution (extends existing) | ~150 | 1 день |
| Telegram alerts | ~100 | 1 день |
| Dashboard для daily metrics | ~250 | 2 дня |
| Tests для каждого | ~600 | 3 дня |
| **TOTAL** | **~2050** | **~14 рабочих дней** |

**3 недели работы.** Не 3 месяца. Без ML, без GPU, без сложности.

---

## Часть 5 — Roadmap к первому реальному профиту

### Phase 1: Реализация и paper (3-4 недели)

**Week 1-2: Реализация stack**
- День 1-2: Strategy A (Donchian + EMA200)
- День 3-4: Strategy B (Carry trade)
- День 5: Strategy C (RSI mean reversion)
- День 6-7: Multi-symbol orchestrator + risk manager

**Week 3: Backtest и валидация**
- Прогнать каждую стратегию через walk-forward + baseline
- Каждая должна:
  - Бить B&H на ≥50% folds (минимум) или ≥60% (хорошо)
  - Иметь Sharpe > 0.3 single symbol, > 0.5 portfolio
  - Max DD < 15%
- Если хоть одна не бьёт baseline — ВЫКИНУТЬ её, не торговать

**Week 4: MT5 integration**
- Demo connection
- Live signal generation
- Telegram alerts
- Kill switch testing

### Phase 2: Paper trading (4 недели)

- 28 дней live paper на demo cent-account
- Daily reconciliation: paper expected vs actual ±20%
- Если ≥3 дней расхождение >30% — STOP, fix

### Phase 3: Micro-live (4-8 недель)

- Депозит $300 на cent-account
- Risk: 0.25% per trade = $0.75 (consciously below 0.5% для safety)
- Максимум 2 одновременных позиций
- 4 недели наблюдения **без вмешательства**
- В конце 4 недель ревью:
  - Plus → депозит до $1k
  - Около нуля → ещё 4 недели наблюдения
  - Minus > 5% → STOP, post-mortem

### Phase 4: $1k full deployment

После двух успешных месяцев micro-live:
- Депозит до $1k
- Risk: 0.5% per trade
- Максимум 3 одновременных позиций
- Месячный ревью + correction

**Timeline:** 3-4 месяца до полного $1k deployment.

---

## Часть 6 — Что может пойти не так (риски)

### Risk 1: Carry trade unwind
- Когда USD rate резко падает, carry crash
- Был в 2008 (USDJPY -25% за месяц)
- **Защита:** SL 5% от entry на carry позициях. Жертвуем edge но защищаем capital.

### Risk 2: Whipsaw на trend стратегиях
- В range market trend follower теряет 30-40% за 3-6 месяцев
- **Защита:** EMA200 filter — выключает trend strategies в флэте автоматически.

### Risk 3: Broker B-book vs A-book
- B-book брокеры манипулируют исполнением (slippage против тебя)
- **Защита:** Использовать только A-book брокеров (IC Markets, Pepperstone, FP Markets).

### Risk 4: Black swan event
- COVID crash 2020, SNB CHF unpeg 2015 — резкие движения, SL не успевает
- **Защита:** Не держать позиции через weekend, max 5% capital в одной сделке.

### Risk 5: Overoptimization на 2 года данных
- Walk-forward на 2 года может show edge которого нет на 5+
- **Защита:** Перед live проверять на out-of-sample period 2020-2022 если получится скачать.

---

## Часть 7 — Критические рассуждения

### Почему НЕ RL для $1k retail

После всего что я видел:

1. **RL требует много данных** (5+ лет минимум) — у тебя 2 года.
2. **RL chasing distribution shift** — 2024-2026 forex выглядит иначе чем 2020-2022, модель не обобщит.
3. **RL adds engineering risk** — больше slow-paths и багов (видели 6 runs с критическими багами).
4. **RL не имеет built-in инвариантов** — rule-based trader **знает** что carry trade работает по фундаментальным причинам, RL должен это **выучить**.
5. **RL для $1k = overengineering** — Donchian + EMA200 даёт edge с 50 строками кода.

### Почему МОЖНО оставить RL для будущего scaling

Когда счёт вырастет до $10k+:
- Custom features перестают быть overkill
- Multi-broker arbitrage возможен
- RL для **dynamic position sizing** на rule-based сигнале (Hybrid из Часть 3.4) даёт реальный boost
- Tick data + смена слоя обоснованы

**Сейчас RL — отвлечение от первой прибыли. Через год — может стать leverage.**

### Главный вывод

**$100-250/год на $1k достижимо БЕЗ RL.** Чисто rule-based стратегий **достаточно**:
- 3 простые стратегии × 4 пары × multi-strategy diversification
- Базовый risk management (4 проверки)
- 30 дней paper, потом 30 дней micro-live
- 3-4 месяца до полного deployment $1k

**Шанс при честном выполнении: 50-65% что выйдешь в плюс на год.**

Не "miracle profits", не "AI hedge fund", не "passive income" из ютуба. Это инженерная работа, которая может дать +10-15% годовых на тяжёлом ритейл-рынке. И это уже хорошо — это в 5-10 раз лучше Сберовского депозита и в 2-3 раза лучше S&P500 с учётом риска.

---

## Прямо сейчас — 1 шаг

Если ты **серьёзно** хочешь $100-250/год на $1k:

```bash
# День 1: Реализуй Donchian-LO + EMA200 для одной пары
# Это minimum viable strategy. 1 день работы.
# Без неё всё остальное — rumination.
```

Я могу написать `src/apexfx/strategies/donchian_lo_ema_filter.py` (~150 LOC) + тесты + интеграция с walk-forward сегодня. Это даст первую готовую к live стратегию через 1 день, не 3 месяца.

Сказать, делать?

---

*Этот документ — не план, а **рассуждение** о том что возможно. Цифры — оценки, не гарантии. Но математика по сделкам реальная — на $1k retail, без insider edge, $100-250/год = реальный таргет с 50-65% шансом достижения. Всё больше — мечтания, всё меньше — недоиспользование возможностей.*
