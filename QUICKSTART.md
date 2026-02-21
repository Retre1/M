# ApexFX Quantum — Полное руководство по запуску

## 📋 Требования

| Компонент | Требование |
|-----------|-----------|
| Python | 3.11+ |
| ОС для обучения | macOS / Linux / Windows |
| ОС для live-торговли | Windows (MT5 API) |
| RAM | 8 GB минимум, 16 GB рекомендуется |
| GPU | Опционально (CUDA / Apple MPS) |
| MetaTrader 5 | Для live-торговли и сбора данных |

---

## 🚀 Шаг 1: Установка

```bash
cd /Users/abobik/Desktop/M

# Создать виртуальное окружение
python3.11 -m venv .venv
source .venv/bin/activate

# Установить проект со ВСЕМИ зависимостями
pip install -e ".[all]"

# Или только core (без MT5 и Dashboard):
# pip install -e .
```

## 🧪 Шаг 2: Проверить установку

```bash
# Запустить тесты
make test

# Или напрямую:
pytest tests/ -v --tb=short
```

---

## 📊 Шаг 3: Данные

### Вариант A: Без MetaTrader (синтетические данные) — РЕКОМЕНДУЕТСЯ для старта

Вам НЕ нужны реальные данные для первого запуска!
Система сама сгенерирует синтетические данные через Curriculum Learning.

```bash
# Запустить обучение только на синтетике:
python scripts/train.py --synthetic-only
```

### Вариант B: С реальными данными из MetaTrader 5

**Требуется Windows** (MT5 Python API работает только на Windows).

1. Установите MetaTrader 5 от любого брокера
2. Откройте демо-счёт (бесплатно)
3. Настройте `.env` файл:

```bash
cp .env.example .env
# Отредактируйте .env:
```

```env
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
MT5_LOGIN=12345678         # Ваш номер счёта
MT5_PASSWORD=your_password  # Ваш пароль
MT5_SERVER=YourBroker-Demo  # Сервер брокера
```

4. Собрать исторические данные:

```bash
python scripts/collect_data.py --symbol EURUSD --days 750 --timeframes M1 M5 H1 D1
```

Это скачает ~750 дней истории и сохранит в `data/raw/` в формате Parquet.

### Вариант C: Свои CSV/Parquet данные

Если у вас есть данные в CSV, конвертируйте в нужный формат:

```python
import pandas as pd
df = pd.read_csv("your_data.csv")
# Нужные колонки: time, open, high, low, close, volume
df["time"] = pd.to_datetime(df["time"], utc=True)
df.to_parquet("data/raw/bars/EURUSD/H1/2023-01-01.parquet")
```

---

## 🧠 Шаг 4: Обучение модели

### Быстрый старт (синтетика, ~30 мин)

```bash
python scripts/train.py --synthetic-only --symbol EURUSD
```

### Полное обучение (с реальными данными, ~2-8 часов)

```bash
python scripts/train.py --symbol EURUSD --timeframe H1
```

### Что происходит при обучении:

```
Stage 1 (500K шагов): Чистая синтетика — бот учит базовые паттерны
Stage 2 (1M шагов):   Шум + реальные данные — устойчивость к шуму
Stage 3 (2M шагов):   Полная реальность + чёрные лебеди — боевой режим
```

Модель сохраняется в:
- `models/checkpoints/` — промежуточные чекпоинты
- `models/best/final_model.zip` — финальная модель

### Мониторинг обучения

```bash
# TensorBoard (в отдельном терминале):
tensorboard --logdir=logs/
```

---

## 📈 Шаг 5: Бэктест

```bash
python scripts/backtest.py --symbol EURUSD --timeframe H1 --output results.json
```

Результат — Walk-Forward валидация с метриками:
- Sharpe Ratio, Sortino, Calmar
- Max Drawdown, Win Rate, Profit Factor
- По каждому фолду отдельно

---

## 🔴 Шаг 6: Live Trading (только Windows + MT5)

```bash
# Убедитесь что .env настроен и MT5 запущен
python scripts/live_trade.py --symbol EURUSD --model-path models/best/final_model
```

**⚠️ ВАЖНО: Начинайте ТОЛЬКО на демо-счёте!**

Бот автоматически:
- Подключится к MT5
- Начнёт собирать тики
- На каждой новой H1-свече: features → модель → risk check → execution
- При отключении MT5 — автоматический реконнект
- При Ctrl+C — закроет позиции и сохранит состояние

---

## 📊 Шаг 7: Dashboard (мониторинг)

```bash
python scripts/launch_dashboard.py
# Откройте http://127.0.0.1:8050
```

4 страницы:
- **Overview**: Equity curve, P&L, текущая позиция, история сделок
- **Signals**: Ценовой график, действия агентов, gating weights
- **Risk**: VaR, drawdown, cooldown статус, качество исполнения
- **Training**: Прогресс обучения, walk-forward результаты

---

## ⚙️ Конфигурация

Все настройки в `configs/`:

| Файл | Что настраивает |
|------|----------------|
| `base.yaml` | Seed, device (cuda/mps/cpu), пути, логирование |
| `symbols.yaml` | Торговые пары, pip value, сессии |
| `data.yaml` | Таймфреймы, размеры буферов, хранилище |
| `model.yaml` | TFT (d_model, heads), агенты, RL (SAC/PPO, lr, gamma) |
| `training.yaml` | Curriculum стадии, walk-forward окна, checkpointing |
| `risk.yaml` | VaR лимит (2%), max drawdown (5%), cooldown, Kelly |
| `execution.yaml` | Тип ордеров, spread лимиты, retry, сессии |
| `dashboard.yaml` | Хост, порт, тема |

### Ключевые параметры для тюнинга:

```yaml
# risk.yaml — самые важные:
daily_var_limit: 0.02    # Не рисковать > 2% в день
max_drawdown_pct: 0.05   # Стоп при 5% просадке

# model.yaml:
rl:
  algorithm: SAC          # SAC или PPO
  learning_rate: 0.0003   # Скорость обучения

tft:
  d_model: 64             # Размер модели (32/64/128)
  n_heads: 4              # Число attention heads
```

---

## 🔧 Типичные команды

```bash
make install      # Установить зависимости
make train        # Обучить модель
make backtest     # Бэктест
make live         # Live торговля
make dashboard    # Запустить дашборд
make test         # Тесты
make lint         # Линтер + типы
make clean        # Очистить кэши
```

---

## ❓ FAQ

**Q: Нужен ли GPU?**
A: Нет, но ускорит обучение в 3-5x. Поддерживается CUDA и Apple MPS (M1/M2/M3).

**Q: Можно ли без MetaTrader?**
A: Да! Обучение и бэктест работают полностью без MT5 (на синтетике или своих данных).

**Q: Сколько данных нужно?**
A: Минимум 6 месяцев H1 баров (4380 свечей). Рекомендуется 2+ года.

**Q: Какой брокер?**
A: Любой с MT5 и хорошими спредами (ICMarkets, Pepperstone, RoboForex демо).
