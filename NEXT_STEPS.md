# ApexFX — План действий: что делать дальше

> Сгенерировано: 2026-04-26 (после AUDIT_REPORT_v2.md)
> Формат: каждый шаг — команда → ожидаемый результат → критерий → следующий шаг
> Принцип: **никогда не торговать в live, пока walk-forward + baseline не подтвердят edge**

---

## 🟢 СЕГОДНЯ (1-2 часа)

### Шаг 1. Закоммитить новую инфраструктуру валидации
```bash
cd /Users/abobik/Desktop/M
git add AUDIT_REPORT_v2.md NEXT_STEPS.md \
        src/apexfx/eval/ tests/unit/test_eval/ \
        scripts/compare_baselines.py
git commit -m "feat: honest baseline validation infrastructure + audit v2

- AUDIT_REPORT_v2.md: global architecture review + 4-week roadmap
- NEXT_STEPS.md: concrete prioritized action plan
- src/apexfx/eval/: 4 baselines (B&H/MA/Donchian/Random) + WF report
- 42 unit tests covering no-look-ahead, cost realism, Sharpe sanity
- scripts/compare_baselines.py: CLI works standalone (no model needed)"
```
**Зачем:** зафиксировать рабочее состояние перед изменениями.
**Критерий:** `git log -1 --stat` показывает 7+ новых файлов, ~2500+ insertions.

---

### Шаг 2. Запустить baseline-сравнение на 3 разных таймфреймах
```bash
.venv/bin/python scripts/compare_baselines.py --timeframe H1 --detail > reports/baselines_H1.txt
.venv/bin/python scripts/compare_baselines.py --timeframe H4 --detail > reports/baselines_H4.txt
.venv/bin/python scripts/compare_baselines.py --timeframe D1 --detail > reports/baselines_D1.txt
mkdir -p reports && ls reports/
```
**Зачем:** найти таймфрейм, на котором baselines дают самый высокий avg Sharpe — это потенциально лучший рынок для системы.
**Критерий:** в каждом отчёте видно, какой baseline побеждает в каждом окне.
**Решение:** таймфрейм с самым высоким avg-best-Sharpe и самой низкой volatility-of-Sharpe — твой основной кандидат для тренинга.

---

### Шаг 3. Проверить чувствительность к спреду (retail-реальность)
```bash
for spread in 1.5 2.0 2.5 3.0 3.5; do
  echo "=== spread=$spread pips ===" >> reports/spread_sensitivity.txt
  .venv/bin/python scripts/compare_baselines.py --timeframe H4 --no-folds \
    --spread-pips $spread 2>&1 | grep -E "Sharpe|Return|VERDICT" \
    >> reports/spread_sensitivity.txt
done
cat reports/spread_sensitivity.txt
```
**Зачем:** реальный retail-спред 2.5-3.0 пипса. Если B&H выживает только на 1.5 — твоя модель тем более не выживет на 3.0.
**Критерий:** сколько baselines остаются с положительным Sharpe при spread=3.0?
**Решение:** если все baselines уходят в минус при 3.0 → этот рынок/таймфрейм нерентабелен retail вне зависимости от модели.

---

## 🟡 ЭТА НЕДЕЛЯ (5-7 дней): Validate-first

### Шаг 4. Подготовить GPU-сервер
```bash
# Проверить, что MWS VM ещё доступна (подписка может быть истекшей)
ssh -o ConnectTimeout=10 -i ~/.ssh/apexfx_mts user1@82.202.157.240 'echo OK'
```
**Если OK:** перейти к Шагу 5.
**Если не OK:** заказать новую VM (cloud.ru / vast.ai / runpod.io ≈ $0.5-1/час за A100), либо использовать локальный CPU (медленнее в 50-100 раз).

---

### Шаг 5. Деплой кода на сервер
```bash
ssh -i ~/.ssh/apexfx_mts user1@82.202.157.240 << 'EOF'
cd ~/apexfx 2>/dev/null || git clone https://github.com/Retre1/M.git ~/apexfx
cd ~/apexfx
git pull origin main
git log --oneline -5  # должен показать 1f2a549 (TFT pretrain skip) на верху
ls -la .venv 2>/dev/null || PYTHON=python3.11 bash scripts/server_bootstrap.sh cu121
EOF
```
**Критерий:** последний commit `1f2a549` доступен, `.venv` создана.

---

### Шаг 6. Smoke-тест на сервере (15-20 мин)
```bash
ssh -i ~/.ssh/apexfx_mts user1@82.202.157.240 'cd ~/apexfx && MODE=smoke bash scripts/run_server.sh'
```
**Критерий:** последняя строка лога `tail -20 logs/train_*.log` показывает feature_pipeline_complete + Stage 0 завершён без NaN.
**Если падает:** диагностируй до полного тренинга — баг в дереве зависимостей дешевле найти на 2k шагах, чем на 5M.

---

### Шаг 7. Полный тренинг (6-12 часов на A100)
```bash
ssh -i ~/.ssh/apexfx_mts user1@82.202.157.240 'cd ~/apexfx && \
  MODE=full SYMBOL=EURUSD TIMEFRAME=H1 nohup bash scripts/run_server.sh > /dev/null 2>&1 &'

# мониторинг (можно повесить локально на ScheduleWakeup)
ssh -i ~/.ssh/apexfx_mts user1@82.202.157.240 'tail -f ~/apexfx/logs/train_*.log'
```
**Критерий:** все 4 стадии завершены, лог содержит "Training complete".
**Чекпоинты для забора:** `models/best/final_model.zip` + `models/checkpoints/stage_*.zip`.

---

### Шаг 8. Забрать модели локально
```bash
rsync -avz -e "ssh -i ~/.ssh/apexfx_mts" \
  user1@82.202.157.240:~/apexfx/models/ ./models/
ls -la models/best/  # final_model.zip должен быть с сегодняшней датой
```
**Критерий:** дата файла `final_model.zip` — сегодняшняя.
⚠️ **Сохрани старый `models/best/final_model.zip` (Feb 19) под другим именем перед перезаписью** — на случай регрессии.

---

### Шаг 9. Walk-forward backtest (НЕ quick_backtest)
```bash
.venv/bin/python scripts/backtest.py --config-dir configs --symbol EURUSD \
  > reports/wf_results_$(date +%Y%m%d).log 2>&1
cat reports/wf_results_*.log | grep -E "fold|sharpe|profit_factor|aggregate"
```
**Критерий:** в логе есть `n_folds >= 6`, для каждого fold — свой Sharpe.
**Если падает:** скорее всего walk_forward.py использует устаревший интерфейс trainer — это надо чинить отдельной сессией.

---

## 🟠 СЛЕДУЮЩАЯ НЕДЕЛЯ: Точка решения

### Шаг 10. Подключить baseline-сравнение к backtest.py
Цель — после walk-forward сразу видеть таблицу "model vs baselines".

Открой `scripts/backtest.py`, найди место после `validator.run()`, добавь:
```python
from apexfx.eval.baselines import (
    BuyAndHoldBaseline, MACrossBaseline, DonchianBaseline, RandomBaseline
)
from apexfx.eval.walk_forward_report import compare_to_baselines, format_comparison_table

baselines = [
    BuyAndHoldBaseline(),
    MACrossBaseline(20, 50),
    DonchianBaseline(20),
    DonchianBaseline(20, long_only=True),
    RandomBaseline(seed=42),
]

# Извлечь model Sharpes per fold + price slices per fold
fold_data = [(f"fold-{i}", data.iloc[f.test_start:f.test_end]) for i, f in enumerate(results.folds)]
model_sharpes = [f.metrics.get("sharpe_ratio", 0.0) for f in results.folds]

rows = compare_to_baselines(fold_data, model_sharpes, baselines, annualisation_periods=252*24)
print(format_comparison_table(rows, edge_threshold_pct=60.0))
```
**Критерий:** в выводе `backtest.py` появляется таблица + VERDICT.

---

### Шаг 11. ВЕРДИКТ — что показывает walk-forward + baseline?

| Сценарий | Решение | Дальше |
|---|---|---|
| **A. Avg WF Sharpe > 1.0 AND beats best baseline на ≥60% folds** | EDGE найден | → Шаг 12 (refinement, multi-symbol) |
| **B. WF Sharpe в [0, 1.0] OR beats baseline на 40-60% folds** | Слабый edge | → Шаг 13 (hyperparameter sweep) |
| **C. WF Sharpe в [-0.5, 0]** | Близко к нулю, переобучение | → Шаг 14 (RADICAL SIMPLIFICATION, удалить 13k LOC) |
| **D. WF Sharpe < -0.5** | Архитектура сломана сильнее, чем баги | → Шаг 15 (Path C — другой рынок/таймфрейм) |

**Это решение определяет следующие 3-4 недели работы. Принимай его на основе цифр, не интуиции.**

---

## 🔵 СЦЕНАРИЙ A — Refinement (если edge найден)

### Шаг 12.1. Multi-symbol portfolio
Добавь GBPUSD, USDJPY, AUDUSD к тренингу.
```bash
# Скачать данные
.venv/bin/python scripts/download_mt5_data.py --symbol GBPUSD --timeframe H1
.venv/bin/python scripts/download_mt5_data.py --symbol USDJPY --timeframe H1
.venv/bin/python scripts/download_mt5_data.py --symbol AUDUSD --timeframe H1
# Конфиг symbols.yaml уже поддерживает массив — добавь туда новые пары
# Перезапусти тренинг с MultiSymbolConfig
```
**Критерий:** WF Sharpe на portfolio выше, чем на single-symbol → diversification работает.

### Шаг 12.2. Реалистичный cost model в env
В `configs/symbols.yaml` подними `transaction_cost_pips: 1.0 → 2.5` (retail median).
Перезапусти WF backtest.
**Критерий:** edge сохраняется при реалистичных costs. Если нет — это не edge, это иллюзия.

### Шаг 12.3. Stress periods
```bash
# Тестируй на 3 разных периодах отдельно: trend up, trend down, range
.venv/bin/python scripts/compare_baselines.py --timeframe H4 --no-folds  # full
# Затем модифицируй backtest для slice'ов 2024-Q1, 2024-Q4, 2025-Q3 и т.п.
```

---

## 🟣 СЦЕНАРИЙ B — Hyperparameter sweep

### Шаг 13.1. Сделай grid 10-20 экспериментов
Ключевые параметры (в `configs/training.yaml` + `configs/model.yaml`):
- `rl.learning_rate`: [1e-5, 3e-5, 1e-4, 3e-4]
- `rl.gamma`: [0.99, 0.995, 0.999]
- `rl.batch_size`: [256, 512, 1024]
- `reward.reward_scale`: [10, 50, 100]
- `ent_coef`: [auto, 0.05, 0.1, 0.2]

```bash
# Используй существующий hyperopt_sac.py
.venv/bin/python scripts/hyperopt_sac.py --n-trials 20 --timeout 86400
# результат → optuna.db, best_params.json
```
**Критерий:** найдена комбинация, дающая avg WF Sharpe выше базового на 30%+. Если нет за 20 trials — нет дешёвого тюнинга, идти на Сценарий C.

---

## 🔴 СЦЕНАРИЙ C — Radical Simplification (если edge не найден)

### Шаг 14.1. Бэкап текущего состояния
```bash
git tag pre-simplify-$(date +%Y%m%d)
git push origin --tags
```

### Шаг 14.2. Удалить мёртвые модули (13k LOC)
По списку из `AUDIT_REPORT_v2.md` Приложение B:
```bash
# Features, отвергнутые feature selector:
git rm src/apexfx/features/{wavelet,spectral,fundamental,structure,central_bank,cot,seasonal,intermarket_corr,scalping,sentiment,clustering,dim_reducer}.py
# Архитектурный overhead:
git rm -r src/apexfx/models/{world_model,tft,ensemble,agents,components}/
git rm src/apexfx/training/{ewc,adversarial,per,pretrain,hierarchical,diversity}.py
# MTF не оправдан:
git rm src/apexfx/env/mtf_forex_env.py src/apexfx/data/{mtf_synthetic,mtf_aligner}.py
# Smart execution для < 0.1 lot:
git rm src/apexfx/execution/{smart_exec,fill_tracker,liquidity_guard,order_manager}.py
# VaR/stress overkill:
git rm src/apexfx/risk/{stress_testing,var_calculator}.py
# Дамп старой версии:
git rm -r src/_v2_dump/
# Связанные тесты:
git rm -r tests/unit/test_phase{2,3,3_5}.py
git commit -m "refactor: remove 13k LOC of unproven complexity (Path B simplification)"
```

### Шаг 14.3. Переписать trainer в простой single-stage
Создай `src/apexfx/training/simple_trainer.py` (~200 LOC):
- Один SAC агент (НЕ HiveMind, НЕ TQC)
- MLP policy [256, 128]
- LogReturnReward only
- Walk-forward по умолчанию
- Без EWC, adversarial, world model, PER

### Шаг 14.4. Re-train + re-validate
```bash
# Локально или на CPU — с упрощённой моделью даже без GPU будет ~2-4 часа
.venv/bin/python scripts/train.py --config-dir configs/simple
.venv/bin/python scripts/backtest.py --config-dir configs/simple
```
**Критерий:** после simplification WF Sharpe не хуже, чем до (часто — лучше, потому что меньше overfit).

---

## ⚫ СЦЕНАРИЙ D — Path C (другой рынок/таймфрейм)

### Шаг 15.1. Multi-symbol forex portfolio (минимальная переделка)
См. Шаг 12.1 — но это может быть и primary, не только refinement.

### Шаг 15.2. Crypto perpetuals
```bash
# Скачать BTC/ETH данные с Binance
pip install python-binance
# Скрипт-адаптер (нужно написать) — конвертация tick данных в OHLC parquet формат твоего store
.venv/bin/python scripts/download_binance.py --symbol BTCUSDT --interval 1h --start 2020-01-01
.venv/bin/python scripts/compare_baselines.py --symbol BTCUSDT --spread-pips 0.0  # crypto: bps not pips
```
**Критерий:** на BTC Sharpe baselines значимо выше, чем на EURUSD → это лучший рынок для retail.

### Шаг 15.3. Daily timeframe + carry trade
- D1 даёт ×24 меньше costs
- Carry trade (long high-yield, short low-yield) — proven retail edge
- Реализуй простой carry-screener вместо RL

---

## 🟢 ЧЕРЕЗ 4-6 НЕДЕЛЬ: Paper Trading

### Шаг 16. MT5 demo account
1. Открыть demo на retail-брокере (FxPro, OANDA, IC Markets)
2. Подключить через `MetaTrader5` Python пакет
3. Проверить `data/mt5_client.py` коннект — должен принимать ticks без ошибок

### Шаг 17. Live signal generation pipeline
```bash
.venv/bin/python scripts/live_trade.py --symbol EURUSD --paper
```
**Критерий:** каждый час (или per H4 close) генерируется signal, видно в `logs/live_*.log`.

### Шаг 18. Telegram alerts
Создай бота через @BotFather, добавь `TELEGRAM_BOT_TOKEN` в `.env`.
Хук в `live/trading_loop.py`: на каждое событие (signal_generated, order_filled, sl_hit, tp_hit, daily_dd_breach) — POST в Telegram.
**Критерий:** видишь все events на телефоне в реальном времени.

### Шаг 19. Kill switch testing
```bash
# Создать файл .kill_switch — должен остановить торговлю в течение 1 минуты
touch .kill_switch
# Проверить логи: bot должен закрыть все позиции и остановиться
```

---

## 🟢 ЧЕРЕЗ 8-12 НЕДЕЛЬ: Micro-live (только если paper прошёл)

### Шаг 20. Cent account на $200-300
**Условия запуска:**
- 30 дней paper trading закрылось в плюс
- Метрики (PF, Sharpe, WR, expectancy) в пределах ±20% от backtest
- Kill switch протестирован вручную
- Telegram alerts работают

**Параметры:**
- Risk per trade: 0.25% (= $0.50-$0.75 на $200)
- Daily loss limit: 2% ($4-6) — hard-coded
- Weekly loss limit: 5% ($10-15) — auto-pause торговли до понедельника
- Максимум открытых позиций: 2

### Шаг 21. Первые 30 дней — ZERO INTERVENTION
- Никаких изменений в коде
- Никаких manual override сделок
- Только мониторинг + ежедневный лог paper-vs-live drift

### Шаг 22. Ревью через месяц
| Результат месяца | Решение |
|---|---|
| Плюс + метрики ±30% от backtest | Депозит до $1000, продолжать |
| Около нуля + метрики в пределах | Дополнительный месяц наблюдения |
| Минус > 5% | STOP. Откат к paper, разбор почему backtest не сошёлся |
| Минус > 10% | STOP. Архитектурный пересмотр (вернуться на Шаг 11) |

---

## 📋 Чек-листы готовности на ключевых точках

### Готов к full training?
- [ ] `git log -1 --oneline` показывает свежий commit
- [ ] `.venv/bin/pytest tests/unit/test_eval/ tests/unit/test_env/ tests/unit/test_features/ tests/unit/test_training/ -q` зелёный
- [ ] GPU сервер доступен (`ssh ... 'nvidia-smi'` работает)
- [ ] Smoke-тест прошёл на сервере без NaN

### Готов к paper trading?
- [ ] Walk-forward Sharpe > 1.0 на ≥60% folds
- [ ] Модель бьёт лучший baseline на ≥60% folds
- [ ] Тестирована на 3 типах рынка (uptrend/downtrend/range) отдельно
- [ ] Cost model реалистичный (≥2.5 пипса спред в симуляции)
- [ ] Walk-forward voluneers Sharpe std/mean < 0.7 (стабильность)

### Готов к micro-live?
- [ ] 30 дней paper — PF > 1.0
- [ ] Все метрики paper в пределах ±20% от backtest
- [ ] Kill switch проверен вручную
- [ ] Telegram alerts работают
- [ ] MT5 connection проверен — нет потерь сигналов
- [ ] Daily loss limit срабатывает на тестовом triggering

### Готов к scaling (>$1k)?
- [ ] 30 дней micro-live — закрыто в плюс
- [ ] Все метрики live в пределах ±30% от paper
- [ ] Нет manual interventions за месяц
- [ ] Нет downtime бота > 1 часа за месяц

---

## ⚠️ Hard stop conditions (когда ОСТАНОВИТЬСЯ)

Прерви работу над текущим направлением, если:
1. **После Шага 11**: WF Sharpe < -0.3 → architecture broken, иди на Сценарий C/D
2. **После Шага 17**: paper trading в просадке 5%+ в первые 2 недели → не идти в live
3. **После Шага 20**: live в минусе > 5% за первый месяц → STOP, не добавлять капитал
4. **После 8 недель работы**: ни один сценарий не дал WF Sharpe > 1.0 → проект research-only, не пытайся это превратить в torговлю
5. **Психологический стоп**: если эмоционально не можешь выдержать DD > 5% — не торгуй вообще, это не для тебя

---

## 💡 Что делать ПАРАЛЛЕЛЬНО (пока тренинг идёт)

Тренинг занимает 6-12 часов. Не сиди — работай:

1. **Сделай code review** существующих модулей: `risk/risk_manager.py`, `live/trading_loop.py`, `execution/executor.py` — там может быть скрытые баги ещё.
2. **Подготовь paper trading инфру** заранее — Telegram bot, demo account, мониторинг dashboard.
3. **Изучи 1-2 paper по retail trading** — например, "Fact, Fiction, and Momentum Investing" (Asness 2014). Эмпирические базы — твой друг.
4. **Прочитай `docs/research/`** — там 6 research-папок, которые ты уже сохранил, но возможно не углубился.
5. **Подготовь backup plan** — если apexfx не взлетит, что? Carry trade на portfolio? Crypto DCA bot? Имей запасной вариант.

---

## 🎯 Главное правило

**Каждый шаг должен заканчиваться измерением.** Если ты сделал что-то и не знаешь, помогло это или нет — ты потратил время впустую. Edge доказывается цифрами, а не интуицией.

**Без honest validation — нет live trading.** Точка.

---

*Этот план рассчитан на 8-12 недель упорной работы до micro-live. Если нет такого времени или мотивации — лучше не торговать вообще, сэкономишь $1k.*

*При обновлении прогресса — отмечай шаги выполненными в этом файле, чтобы не терять контекст между сессиями.*
