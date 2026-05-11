# MT5 Donchian Turtle — setup гайд

> Полный путь от регистрации брокера до запуска бота в Windows VPS
> Время setup: 3-4 часа (1 час кода + 2-3 часа KYC у брокера)

---

## Архитектура

```
Windows VPS ($5-10/мес — RoboForex VPS, AEZA Windows, AWS Workspaces)
  │
  ├── MetaTrader 5 terminal (логин: твой broker account)
  │     ↑
  │     │ shared memory (MetaTrader5 Python package)
  │     ↓
  └── Python скрипт (run_mt5_turtle.py):
        ├── Каждые 60 сек: проверка новых баров через mt5.copy_rates_from_pos()
        ├── На закрытии H4 бара: вычисляет Donchian/EMA/ATR в numpy
        ├── Принимает решение: ENTER/PYRAMID/EXIT/HOLD
        ├── Через mt5.order_send() — market orders с SL
        ├── Risk: kill_switch (файл) + circuit_breaker (daily/weekly/monthly DD)
        └── Telegram alerts на каждое событие
```

**Преимущества MT5 vs OKX/TradingView:**
- Нет нужды в TradingView Pro+ ($30/мес сэкономлено)
- Нет публичного webhook endpoint (не нужен домен, SSL, HMAC)
- Стратегия в Python — debug и backtest проще
- Прямое подключение к broker через MT5 terminal
- Forex retail-friendly (cent accounts, депозит rubles через карту)

**Недостатки vs OKX:**
- MetaTrader5 Python package — **только Windows**
- Forex volatility ниже crypto → потенциал меньше
- Spread retail forex 1.5-3 пип vs crypto perp 0.01-0.03%

---

## Часть 1 — Выбор брокера (россияне в 2026)

| Брокер | Регулирование | Russian KYC | Min Deposit | Cent account | Recommend |
|---|---|:-:|:-:|:-:|:-:|
| **RoboForex** | IFSC Belize | ✅ | $10 | ✅ | ⭐⭐⭐⭐⭐ |
| **Tickmill** | Seychelles | ✅ | $100 | ❌ | ⭐⭐⭐⭐ |
| **FxPro** | Bahamas | ✅ | $100 | ❌ | ⭐⭐⭐ |
| **Alpari** | Mauritius | ✅ | $5 | ✅ | ⭐⭐⭐ |
| **Exness** | Seychelles | ✅ | $10 | ✅ | ⭐⭐⭐⭐ |
| **AMarkets** | SVG | ✅ | $100 | ❌ | ⭐⭐⭐ |

**Рекомендация: RoboForex Cent** — для $1k:

- ✅ Регистрация россиян + KYC за 1-2 дня
- ✅ Депозит rubles через Тинькофф/Сбер карту напрямую (без P2P)
- ✅ **Cent account**: $1k = 100,000 центов → можно торговать 0.01 cent lot = $0.10/пип (vs $1/пип на стандарте)
- ✅ MT5 platform поддерживается
- ✅ Бесплатный VPS если депозит > $300 (Roboforex VPS — Windows Server)
- ✅ Spreads ECN: 0.4-0.7 пип EURUSD (vs 1.5-2.0 retail)

### Регистрация RoboForex (~30 минут + 1-2 дня wait)

```
1. roboforex.com → Open Account
2. Account type: ProCent (= cent account на $1k)
3. KYC: загрузи скан паспорта + selfie
4. Wait 1-2 рабочих дня
5. Депозит rubles: Personal Cabinet → Deposit
   - Visa/MasterCard РФ карта работает
   - SBP через Тинькофф
   - YooMoney, QIWI
   - $1000 = ~100,000 RUB (по текущему курсу)
6. Скачай MetaTrader 5 для Windows (можно для Mac/Linux но нам нужен Windows для Python API)
7. Логин в MT5:
   - Сервер: RoboForex-ECN (или ProCent для cent-account)
   - Логин: твой account number
   - Пароль: из welcome email
```

---

## Часть 2 — Windows VPS setup

### Опция A: Бесплатный RoboForex VPS (если депозит > $300)

Если выбрал RoboForex и депозит $300+:

```
1. Personal Cabinet → Free VPS
2. Заказать VPS (Windows Server 2019)
3. Получишь RDP credentials по почте через 1-2 часа
4. Подключайся через Windows Remote Desktop или Microsoft Remote Desktop на Mac
```

Бесплатно, специально оптимизирован под MT5, низкая latency к broker'у.

### Опция B: AEZA (Russian provider, рубли)

```
1. aeza.net → Windows VPS → Минимальный тариф ($5-7/мес)
2. Windows Server 2019/2022
3. 2 vCPU, 2GB RAM, 30GB SSD — достаточно
4. Получишь RDP по email
```

### Опция C: AWS Workspaces / VK Cloud / Selectel

- AWS Workspaces $7-10/мес
- VK Cloud Windows VPS $5-8/мес
- Selectel Windows VPS $6-10/мес

---

## Часть 3 — Setup на Windows VPS

После RDP login на VPS:

### 3.1. Установка Python + MT5 terminal

```powershell
# Python 3.11 (важно — не 3.12, MetaTrader5 пакет требует 3.10-3.11)
# Скачай с python.org → "Add to PATH" при установке

# Проверь
python --version  # Должно быть 3.11.x

# MetaTrader 5 terminal — уже должен быть установлен от broker'а
# Если нет: скачай с MT5 сайта (https://www.metatrader5.com)
```

### 3.2. Клонирование проекта

```powershell
# Git for Windows: git-scm.com
git clone https://github.com/Retre1/M.git C:\apexfx
cd C:\apexfx

# Виртуальное окружение
python -m venv .venv
.venv\Scripts\activate

# Зависимости
pip install -e .
pip install MetaTrader5  # Windows-only пакет
```

### 3.3. Логин MT5 terminal

```
1. Запусти MetaTrader 5 terminal
2. File → Login to Trade Account
3. Введи:
   - Server: RoboForex-ECN (или твой broker server)
   - Login: твой account number
   - Password: твой пароль
4. Должна появиться зелёная "Connection" в правом нижнем углу
5. Включи AutoTrading button (зелёная кнопка вверху или Ctrl+E)
   — без этого Python API не сможет торговать
```

### 3.4. Tools → Options → Expert Advisors

```
✅ Allow algorithmic trading
✅ Allow DLL imports
✅ Disable algorithmic trading via external Python API: ОТКЛЮЧИТЬ (т.е. оставить разрешённым)
```

### 3.5. Telegram bot

```
1. Открой Telegram на телефоне → @BotFather → /newbot
2. Сохрани токен
3. Напиши боту любое сообщение
4. Открой:
   https://api.telegram.org/bot<ТВОЙ_ТОКЕН>/getUpdates
5. Найди result[0].message.chat.id
```

### 3.6. Environment variables

В Windows PowerShell от Administrator:

```powershell
# Permanent env vars (выживут reboot)
[Environment]::SetEnvironmentVariable("APEXFX_MT5_LOGIN", "12345678", "Machine")
[Environment]::SetEnvironmentVariable("APEXFX_MT5_PASSWORD", "your-mt5-password", "Machine")
[Environment]::SetEnvironmentVariable("APEXFX_MT5_SERVER", "RoboForex-ECN", "Machine")
[Environment]::SetEnvironmentVariable("APEXFX_TELEGRAM_TOKEN", "1234:AAH...", "Machine")
[Environment]::SetEnvironmentVariable("APEXFX_TELEGRAM_CHAT_ID", "123456789", "Machine")

# Strategy tuning (опционально — defaults работают)
[Environment]::SetEnvironmentVariable("APEXFX_RISK_PER_UNIT", "0.015", "Machine")
[Environment]::SetEnvironmentVariable("APEXFX_BREAKER_DAILY_PCT", "0.08", "Machine")
```

Перезапусти PowerShell чтобы переменные стали активными.

### 3.7. Smoke test

```powershell
cd C:\apexfx
.venv\Scripts\activate

# Single-cycle test (не loop)
python scripts\run_mt5_turtle.py --once --symbols EURUSD GBPUSD

# Ожидаемый вывод:
# [INFO] MT5 connected login=12345 server=RoboForex-ECN currency=USD balance=1000
# [INFO] TurtleRunner starting symbols=['EURUSD','GBPUSD'] timeframe=H4
# [INFO] Strategy decide → HOLD (no signal)
# (никаких ордеров — это ожидаемо без breakout)
```

Если работает — отлично. Если ошибка — см. Troubleshooting ниже.

---

## Часть 4 — Long-running setup (24/7)

### NSSM (Non-Sucking Service Manager) — лучшее для Windows

```powershell
# Скачай nssm.cc → распакуй nssm.exe в C:\Windows\System32

# Создать сервис
nssm install ApexfxTurtle "C:\apexfx\.venv\Scripts\python.exe"
nssm set ApexfxTurtle AppParameters "C:\apexfx\scripts\run_mt5_turtle.py --symbols EURUSD GBPUSD USDJPY AUDUSD"
nssm set ApexfxTurtle AppDirectory "C:\apexfx"
nssm set ApexfxTurtle AppStdout "C:\apexfx\logs\turtle-stdout.log"
nssm set ApexfxTurtle AppStderr "C:\apexfx\logs\turtle-stderr.log"
nssm set ApexfxTurtle AppRotateFiles 1  # rotate logs
nssm set ApexfxTurtle AppRotateBytes 10485760  # 10MB
nssm set ApexfxTurtle Start SERVICE_AUTO_START

# Запустить
nssm start ApexfxTurtle

# Проверить статус
nssm status ApexfxTurtle  # SERVICE_RUNNING

# Логи
Get-Content C:\apexfx\logs\turtle-stdout.log -Tail 50 -Wait
```

NSSM auto-restart при crash, запуск при reboot VPS, прозрачные логи.

---

## Часть 5 — 30 дней Paper Trading

**ОБЯЗАТЕЛЬНО** перед live.  В MT5 это просто demo account:

```
1. В MT5 terminal: File → Open an Account → demo
2. Если уже залогинен в demo — перезайди в другой demo для чистоты
3. Депозит фейковый $1000 (выбирается при регистрации demo)
4. Перезапусти бот: nssm restart ApexfxTurtle
```

### Что мониторить ежедневно

```powershell
# Логи
Get-Content C:\apexfx\logs\turtle-stdout.log -Tail 100

# Telegram — должны приходить:
#   - Daily summary в полночь UTC
#   - Entry/Exit/Pyramid alerts по мере появления сигналов
#   - Kill switch alert если activated

# MT5 terminal — Trade tab:
#   - Видишь position'ы с magic = 770125
#   - Видишь свои SL установлены на ордерах
```

### Чеклист после 30 дней

- [ ] Total trades > 20 (если меньше — низкая активность стратегии, увеличь количество символов)
- [ ] Profit factor > 1.0
- [ ] Max drawdown < 30% от начального баланса
- [ ] Никаких system crashes — бот работал без сбоев
- [ ] Все Telegram alerts доходят
- [ ] Kill switch работает (вручную: `New-Item C:\apexfx\.kill_switch -ItemType File` → бот останавливается)

---

## Часть 6 — Переход на Live

После успешного paper:

```
1. В MT5 terminal: File → Login to Trade Account → real account credentials
2. Перезапусти бот:
   nssm stop ApexfxTurtle
   # Обнови env vars если account number изменился
   nssm start ApexfxTurtle
3. Депозит на real account: $200-300 для micro-live
4. Risk per unit: уменьши до 0.5% на первый месяц
   [Environment]::SetEnvironmentVariable("APEXFX_RISK_PER_UNIT", "0.005", "Machine")
   nssm restart ApexfxTurtle
5. ZERO override mode — только наблюдение 4 недели
6. После прибыльного месяца — деп до $1000, risk обратно 1.5%
```

---

## Стоимость per month

| Item | Цена |
|---|---:|
| RoboForex VPS | $0 (если деп > $300) или $5-10 |
| Альтернативный Windows VPS (AEZA) | $5-7 |
| MT5 terminal | $0 (от broker'а) |
| Python + MetaTrader5 package | $0 |
| Telegram bot | $0 |
| Broker spread costs ~100 trades/мес | $5-15 (EURUSD ECN 0.5 пип × 0.01 lot × 100) |
| **Total** | **~$10-25/мес** |

**Дешевле чем OKX/TradingView setup на $15/мес.**

---

## Symbol mapping per broker

Разные брокеры именуют символы по-разному:

| Broker | EURUSD | GBPUSD | USDJPY | AUDUSD |
|---|---|---|---|---|
| RoboForex (Pro/ECN) | EURUSD | GBPUSD | USDJPY | AUDUSD |
| RoboForex (ProCent) | EURUSDcent | GBPUSDcent | USDJPYcent | AUDUSDcent |
| Tickmill | EURUSD | GBPUSD | USDJPY | AUDUSD |
| FxPro (FXPro UK) | EURUSD | GBPUSD | USDJPY | AUDUSD |
| IC Markets cTrader | EURUSD. | GBPUSD. | USDJPY. | AUDUSD. |
| Exness | EURUSDm | GBPUSDm | USDJPYm | AUDUSDm |

Проверь в MT5 terminal → Market Watch (Ctrl+M) и используй точное имя в `--symbols` параметре.

---

## Troubleshooting

### `MetaTrader5` import error
- Windows only — на Mac/Linux работает через WSL2 + Python в Windows
- Python 3.10-3.11 (не 3.12)
- `pip install --upgrade MetaTrader5`

### `mt5.initialize() returned False`
- MT5 terminal не запущен → запусти, дождись подключения (зелёный значок)
- AutoTrading не включен → Ctrl+E или нажми зелёную кнопку Algo Trading
- Tools → Options → Expert Advisors → Allow algorithmic trading

### `position not found` после place_order
- Проверь magic number = 770125 (default) — без него бот не "видит" свои позиции
- В MT5 Tools → Options → Trade → "One-click trading" может конфликтовать, отключи

### Orders rejected `retcode=10027` (Disabled by client side)
- AutoTrading отключен в самом MT5 (большая зелёная кнопка)
- Или в Tools → Options → Expert Advisors сняли галку Allow algorithmic trading

### `retcode=10021` (Off-quote)
- Спред расширился слишком сильно (новости, low liquidity)
- Бот retry-ит через minute — не проблема если редко

### Symbol not visible
- В MT5 → Market Watch (Ctrl+M) — найди символ → правый клик → Show
- Или: `--symbols` параметр содержит exact symbol name от broker (см. таблицу выше)

### Daily limit triggered too easily
- Default daily 8% при cent account = $80 на $1000 = крупная просадка
- Если cent account с центами реальными ($1000 = 100,000 центов нотионально):
  ```
  [Environment]::SetEnvironmentVariable("APEXFX_BREAKER_DAILY_PCT", "0.04", "Machine")
  # 4% daily лимит более reasonable для cent account
  ```

---

## Чеклист готовности

Перед запуском live trading:

- [ ] RoboForex real account открыт, депозит $200+
- [ ] Windows VPS работает 24/7 (RoboForex Free VPS или AEZA)
- [ ] MT5 terminal залогинен, AutoTrading включен
- [ ] Python 3.11 + MetaTrader5 package установлены
- [ ] Все env vars установлены (LOGIN, PASSWORD, SERVER, TELEGRAM_*)
- [ ] Telegram bot получает alerts (тест: `python -c "from apexfx.aggressive.alerts.telegram import *; TelegramNotifier(TelegramConfig.from_env()).send('test')"`)
- [ ] Smoke test `--once` прошёл успешно
- [ ] NSSM service создан и в статусе SERVICE_RUNNING
- [ ] 30 дней demo paper без сбоев и с PF > 1.0
- [ ] Kill switch проверен вручную (создание/удаление файла)
- [ ] Risk per unit на первый месяц установлен в 0.005 (0.5%, не 1.5%)
- [ ] Max units = 2 (вместо 4 default) на первый месяц
- [ ] Magic number unique (770125 если только один бот; другой если несколько)

---

*Setup похож на OKX, но проще — нет webhook, нет HMAC, нет публичного домена.  Trade-off — forex retail вместо crypto perp, ниже потенциальная доходность но меньше волатильность.*
