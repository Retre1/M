# Live Demo Quickstart — Запуск на демо-счёте за 15 минут

> Уже есть MT5 demo? Этот гайд показывает как подключить бота к нему **сейчас**.
> Не нужны webhooks, не нужен TradingView, не нужен публичный сервер.

---

## Что у тебя должно быть

- ✅ MT5 demo account (логин/пароль/сервер от брокера)
- ✅ MT5 terminal установлен (Windows) и залогинен в demo
- ✅ Python 3.11 (НЕ 3.12 — `MetaTrader5` пакет не поддерживает)

---

## Шаг 1 — Установка (5 минут)

На той же машине где работает MT5 terminal:

```powershell
cd C:\path\to\apexfx
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install MetaTrader5
```

Проверь что пакет загрузился:

```powershell
python -c "import MetaTrader5 as mt5; print(mt5.__version__)"
# Должно вывести версию (5.0.45+)
```

---

## Шаг 2 — Включи AutoTrading в MT5

1. Запусти MT5 terminal
2. Залогинься в demo (если не залогинен)
3. **Tools → Options → Expert Advisors:**
   - ✅ Allow algorithmic trading
   - ✅ Allow DLL imports
4. На главной панели MT5 нажми **Algo Trading** (зелёная кнопка) — должна стать зелёной/яркой

⚠️ **Без AutoTrading Python скрипт не сможет торговать.**  Это самая частая причина "почему не работает".

---

## Шаг 3 — Заполни свои credentials (2 минуты)

Открой `examples/my_demo_bot.py` в любом редакторе.  Найди эти строки:

```python
mt5=Mt5LoginConfig(
    login=12345678,                    # <-- замени на свой
    password="YOUR-DEMO-PASSWORD",     # <-- замени на свой
    server="MetaQuotes-Demo",          # <-- замени на свой
),
```

### Где взять эти значения

**Login** — номер счёта.  Видно в:
- Welcome email от брокера
- В MT5 terminal: правый верхний угол (под Profile) или Navigator → Accounts

**Password** — *trading* password (НЕ investor password).  Из welcome email.
Если потерял — реset через личный кабинет брокера.

**Server** — имя сервера брокера.  В MT5:
- Tools → Options → Server tab — поле "Server"
- Или: File → Login to Trade Account — dropdown показывает все известные сервера

Примеры серверов:
- `MetaQuotes-Demo` — generic MQ demo
- `RoboForex-DemoPro` — RoboForex demo
- `FxPro-MT5` — FxPro
- `Tickmill-Demo` — Tickmill
- `Exness-MT5Trial` — Exness demo

---

## Шаг 4 — (Опционально) Telegram alerts

Если хочешь видеть каждый трейд на телефоне:

1. В Telegram найди **@BotFather** → команда `/newbot`
2. Дай имя боту, сохрани token (формат `1234567890:AAH...`)
3. Напиши боту любое сообщение (это создаст чат)
4. В браузере: `https://api.telegram.org/bot<ТВОЙ_ТОКЕН>/getUpdates`
5. В JSON найди `result[0].message.chat.id` (число типа `123456789`)

Заполни в `my_demo_bot.py`:

```python
telegram=TelegramSettings(
    bot_token="1234567890:AAH...",
    chat_id="123456789",
),
```

---

## Шаг 5 — Запуск

```powershell
cd C:\path\to\apexfx
.venv\Scripts\activate
python examples\my_demo_bot.py
```

Должен увидеть:

```
2026-05-11 12:00:00 [INFO] apexfx.aggressive.live.run_bot — ApexFX MT5 bot starting
2026-05-11 12:00:00 [INFO] apexfx.aggressive.live.connection — MT5 connected attempt=1
2026-05-11 12:00:00 [INFO] apexfx.aggressive.strategies.turtle_runner — TurtleRunner starting
```

В Telegram (если настроил):

```
🚀 ApexFX bot started
Account: 12345678@MetaQuotes-Demo
Symbols: EURUSD, GBPUSD, USDJPY, AUDUSD
Timeframe: H4
Risk/unit: 1.50%
```

**Бот теперь работает.** Каждые 60 секунд он проверяет новые H4 бары на каждом из 4 символов. Когда происходит Donchian breakout — открывает позицию.  На H4 это **~3-10 сделок в неделю** на портфеле.

---

## Шаг 6 — Остановка

Два способа:

**A) Ctrl-C в терминале** — мгновенно

**B) Создай файл `.kill_switch`** — graceful stop, бот закроет цикл и выйдет:

```powershell
# Windows
New-Item .kill_switch -ItemType File

# или просто Notepad → save as .kill_switch в той же папке
```

После остановки → удали файл чтобы можно было перезапустить:

```powershell
Remove-Item .kill_switch
```

---

## Часто задаваемые вопросы

### Бот не открывает позиции — это нормально?

Скорее всего да.  H4 Donchian breakout случается **в среднем 1-3 раза в неделю на пару**.  На 4 парах = 4-12 раз в неделю.

Проверь:
1. В Telegram приходит "bot started" — связь работает
2. В логах нет ошибок типа `ERROR` или `Traceback`
3. В MT5 terminal: View → Toolbox → "Expert Advisors" tab — там история API calls

### Бот сделал сделку — где её посмотреть?

В MT5 terminal:
- **Trade** tab — открытые позиции (с комментарием "apexfx-turtle")
- **History** tab → переключи на нужный период → закрытые сделки

В Telegram:
- Сразу после открытия — `📥 Entry EURUSD` сообщение
- При закрытии — `📤 Exit EURUSD` с PnL

### Я хочу более консервативные настройки

В `my_demo_bot.py`:

```python
risk_per_unit_pct=0.005,  # 0.5% вместо 1.5%
max_units=2,              # 2 unit max вместо 4
daily_loss_pct=0.04,      # Стоп при -4% в день
```

### Я хочу торговать конкретные символы

Узнай точные имена символов в MT5 terminal → **Market Watch** (Ctrl+M).
Брокеры добавляют суффиксы: `EURUSDp`, `EURUSD.`, `EURUSDcent` и т.д.

```python
symbols=["EURUSDcent", "GBPUSDcent"],  # точные имена от твоего брокера
```

### Бот делает странные сделки — как отлаживать?

Включи DEBUG логи:

```python
log_level="DEBUG",
```

Тогда увидишь каждое решение стратегии — почему она зашла или не зашла.

### MT5 disconnected — что делать?

Ничего.  Бот auto-reconnect через 30 секунд, потом еще через 60, 120, 240 — до 5 минут.  Если 3 подряд reconnect не удались — приходит Telegram alert "❤️‍🩹 Health Check Failed".

### Можно запускать на Mac/Linux?

`MetaTrader5` Python package **только Windows**.  Варианты:
1. **Windows VPS** (RoboForex Free VPS если деп $300+, или AEZA ~$5/мес)
2. **Parallels/Boot Camp** на Mac — работает, но требует Windows лицензии
3. **Wine на Linux** — MT5 terminal работает, но Python API нестабилен

Самый дешёвый production setup: AEZA Windows VPS за $5/мес.

### Бот будет торговать когда я выключу комп?

Только если он запущен на VPS.  На локальной машине — отключился ноут, остановился бот.  Для 24/7 нужен VPS.

Чтобы бот сам стартовал при reboot VPS — оформи как Windows service через NSSM (`nssm install ApexfxTurtle ...`).  Детали в `MT5_SETUP.md`.

---

## Чек-лист первого запуска

- [ ] Python 3.11 установлен
- [ ] `pip install MetaTrader5` отработал без ошибок
- [ ] MT5 terminal запущен и залогинен в demo
- [ ] AutoTrading (зелёная кнопка) включен в MT5
- [ ] Tools → Options → Expert Advisors → ✅ Allow algorithmic trading
- [ ] Заполнены login/password/server в `my_demo_bot.py`
- [ ] Точные имена символов от твоего брокера в `symbols=[...]`
- [ ] (Optional) Telegram bot_token и chat_id
- [ ] `python examples\my_demo_bot.py` → видишь "MT5 connected"
- [ ] (Optional) В Telegram пришло "🚀 ApexFX bot started"

Если все галки — бот работает на demo.  Дай ему 1-2 недели чтобы накопить статистику, потом смотри PF / win rate / Sharpe.

---

## Что дальше после первой недели

1. **Day 1-7:** просто наблюдай.  Не вмешивайся даже если хочется.
2. **Day 8:** открой MT5 → History → посмотри статистику.  Сколько сделок?  WR?  Total profit?
3. **Day 14:** если PF > 1.0 и без system errors — продолжай ещё 2 недели
4. **Day 30:** делай решение про переход на live (см. `MT5_SETUP.md` шаги 6+)

**До 30 дней paper trading НЕ переключайся на real account.**  Чтобы не было соблазна — закрой real account в MT5 и оставь только demo.

---

## Помощь / debug

Если что-то не работает — собери эту информацию:

```powershell
# Лог последних 100 строк
Get-Content turtle.log -Tail 100

# Версия Python и MetaTrader5
python --version
python -c "import MetaTrader5; print(MetaTrader5.__version__)"

# Проверка подключения
python -c "from apexfx.aggressive import BotConfig, Mt5LoginConfig, run_bot; run_bot(BotConfig(mt5=Mt5LoginConfig(login=ТВОЙ_ЛОГИН, password='ТВОЙ_ПАРОЛЬ', server='ТВОЙ_СЕРВЕР'), symbols=['EURUSD']), once=True)"
```

Последняя команда сделает один цикл и выйдет — если она проходит без ошибок, всё ок.
