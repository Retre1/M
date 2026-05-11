"""Packager — собирает минимальный zip-архив для Windows-машины с MT5.

Зачем
-----
Полный репозиторий содержит torch, stable-baselines3, тестовые данные,
OKX/TradingView пути — всё это **не нужно** для запуска MT5 бота.
Этот скрипт создаёт чистый deployment bundle (~50KB вместо ~2GB) с:

  * Только MT5-нужным кодом (apexfx/aggressive/{config,exchanges/base,
    exchanges/mt5_client,strategies,risk,alerts,live} + utils/logging)
  * Минимальный requirements.txt (numpy, requests, structlog, MetaTrader5)
  * Windows-friendly install.bat и run.bat
  * Шаблон конфига my_bot.py
  * README.txt с пошаговой инструкцией

Использование
-------------
На Mac/Linux (где сейчас лежит проект)::

    python scripts/package_for_mt5.py

Появится файл ``apexfx_mt5_deploy.zip`` рядом с этим скриптом.

Передай этот zip на Windows-машину любым способом — email, Telegram,
USB-флешка, Google Drive, OneDrive.  На Windows: распакуй zip, открой
README.txt, следуй инструкциям.

Output
------
``apexfx_mt5_deploy.zip`` — готов к копированию.  Размер ~30-50 KB.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from textwrap import dedent

# Root of the repository (relative to this script)
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

# Files to include from the source tree.  Keep this list explicit — if a
# new MT5-needed module is added, append here.  We do NOT use glob because
# that picks up webhook/, okx_client.py, tradingview/ which we want to skip.
FILES_TO_BUNDLE = [
    # Top-level package
    "src/apexfx/__init__.py",

    # Utils
    "src/apexfx/utils/__init__.py",
    "src/apexfx/utils/logging.py",

    # Aggressive — public API
    "src/apexfx/aggressive/__init__.py",
    "src/apexfx/aggressive/config.py",

    # Aggressive — exchanges (NO okx_client.py — MT5 only)
    "src/apexfx/aggressive/exchanges/__init__.py",
    "src/apexfx/aggressive/exchanges/base.py",
    "src/apexfx/aggressive/exchanges/mt5_client.py",

    # Aggressive — strategies (NO Pine Script)
    "src/apexfx/aggressive/strategies/__init__.py",
    "src/apexfx/aggressive/strategies/donchian_turtle.py",
    "src/apexfx/aggressive/strategies/turtle_runner.py",

    # Aggressive — risk
    "src/apexfx/aggressive/risk/__init__.py",
    "src/apexfx/aggressive/risk/kill_switch.py",
    "src/apexfx/aggressive/risk/circuit_breaker.py",
    "src/apexfx/aggressive/risk/position_sizer.py",

    # Aggressive — alerts
    "src/apexfx/aggressive/alerts/__init__.py",
    "src/apexfx/aggressive/alerts/telegram.py",

    # Aggressive — live runner
    "src/apexfx/aggressive/live/__init__.py",
    "src/apexfx/aggressive/live/connection.py",
    "src/apexfx/aggressive/live/run_bot.py",
]


# ---------------------------------------------------------------------------
# Bundled files (generated, not copied from src/)
# ---------------------------------------------------------------------------


REQUIREMENTS_TXT = """\
# ApexFX MT5 bot — minimal dependencies
# Run on Windows: pip install -r requirements.txt
#
# MetaTrader5 is Windows-only (no Mac/Linux wheels). Python 3.10 or 3.11 only.

numpy>=1.24,<3.0
requests>=2.31
structlog>=23.0
MetaTrader5>=5.0.45
"""


MY_BOT_PY = '''\
"""Твой trading bot — отредактируй credentials и запусти.

ВАЖНО: замени значения login/password/server на СВОИ от demo-счёта.
Где взять:
  * Login    — номер счёта (8-9 цифр), в MT5 правый-верхний угол
  * Password — trading password из welcome-email брокера
  * Server   — Tools → Options → Server tab в MT5, например "MetaQuotes-Demo"
"""

from apexfx.aggressive import (
    BotConfig,
    Mt5LoginConfig,
    TelegramSettings,
    run_bot,
)


config = BotConfig(
    # ==== MT5 credentials (замени!) ===========================================
    mt5=Mt5LoginConfig(
        login=12345678,                     # СВОЙ номер счёта
        password="YOUR-DEMO-PASSWORD",      # СВОЙ trading password
        server="MetaQuotes-Demo",           # СВОЙ broker server
    ),

    # ==== Что торгуем =========================================================
    # Используй ТОЧНЫЕ имена символов от твоего брокера (см. Market Watch в MT5)
    symbols=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
    timeframe="H4",

    # ==== Risk (для demo можно держать дефолты; для live уменьши до 0.005) ===
    risk_per_unit_pct=0.015,    # 1.5% per pyramid unit
    max_units=4,                # макс 4 unit в одну сторону
    daily_loss_pct=0.08,        # дневной лимит -8% → kill switch

    # ==== Telegram (опционально — оставь пустым чтобы выключить) =============
    telegram=TelegramSettings(
        bot_token="",           # "1234567890:AAH..." от @BotFather
        chat_id="",             # "123456789" из getUpdates
    ),
)


if __name__ == "__main__":
    print("Starting:", config.summary())
    print("Stop: Ctrl-C или создай файл .kill_switch")
    run_bot(config)
'''


INSTALL_BAT = """\
@echo off
REM ApexFX MT5 bot — Windows installer

echo === Step 1/3: Checking Python ===
python --version 2>nul || (
    echo ERROR: Python not found. Install Python 3.11 from python.org first.
    pause
    exit /b 1
)

echo.
echo === Step 2/3: Creating virtual environment ===
if not exist .venv (
    python -m venv .venv
)
call .venv\\Scripts\\activate.bat

echo.
echo === Step 3/3: Installing dependencies ===
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo === Install complete ===
echo.
echo Next steps:
echo   1. Edit my_bot.py — set login, password, server
echo   2. Make sure MT5 terminal is running and logged in to demo
echo   3. Click "Algo Trading" button in MT5 (green)
echo   4. Run the bot: run.bat
echo.
pause
"""


RUN_BAT = """\
@echo off
REM ApexFX MT5 bot — launcher

if not exist .venv\\Scripts\\activate.bat (
    echo ERROR: Virtual env missing. Run install.bat first.
    pause
    exit /b 1
)

call .venv\\Scripts\\activate.bat

echo Starting ApexFX bot...
echo Stop: Ctrl-C or create .kill_switch file in this folder
echo.
python my_bot.py
pause
"""


README_TXT = """\
==============================================================
  ApexFX MT5 Bot — Windows Deployment
==============================================================

Что это
-------
Готовый к запуску trading bot для MetaTrader 5.  Не нужно
дополнительной разработки — только заполнить свои credentials
и запустить.

Что нужно
---------
1) Python 3.11 (НЕ 3.12 — пакет MetaTrader5 поддерживает 3.10-3.11)
   Скачать: https://www.python.org/downloads/release/python-3119/
   ВАЖНО: при установке поставь галку "Add Python to PATH"

2) MetaTrader 5 terminal установлен и залогинен в demo-счёт
   Скачать: https://www.metatrader5.com/en/download

3) В MT5 включи AutoTrading:
   - Tools → Options → Expert Advisors
   - [X] Allow algorithmic trading
   - [X] Allow DLL imports
   - На главной панели нажми "Algo Trading" (зелёная кнопка)

Установка (один раз)
--------------------
1) Распакуй этот zip в любую папку, например C:\\apexfx
2) Двойной клик на install.bat
3) Подожди 1-2 минуты пока установятся зависимости
4) Закрой окно когда увидишь "Install complete"

Настройка (один раз)
--------------------
1) Открой my_bot.py в любом редакторе (Notepad подойдёт,
   но лучше VS Code или PyCharm — они подсветят синтаксис)

2) Найди блок mt5=Mt5LoginConfig(...) и замени:
     login=12345678              ->  свой номер счёта
     password="YOUR-DEMO-PASSWORD" ->  свой trading password
     server="MetaQuotes-Demo"    ->  свой broker server

Где взять эти значения:
   * Login    — в MT5 правый-верхний угол или welcome-email
   * Password — trading password (НЕ investor) из welcome-email
   * Server   — в MT5: Tools -> Options -> Server tab
                Примеры: MetaQuotes-Demo, RoboForex-DemoPro,
                FxPro-MT5, Tickmill-Demo

3) (Опционально) Telegram уведомления:
   - Создай бота: в Telegram найди @BotFather, /newbot
   - Сохрани token (формат 1234567890:AAH...)
   - Напиши боту любое сообщение
   - Открой в браузере:
     https://api.telegram.org/bot<TOKEN>/getUpdates
   - Найди chat.id (число типа 123456789)
   - Вставь оба значения в TelegramSettings(...) в my_bot.py

4) (Опционально) Поменяй symbols на свои:
   - В MT5 → Market Watch (Ctrl+M) посмотри точные имена
   - Некоторые брокеры добавляют суффиксы: EURUSDp, EURUSD., EURUSDcent

Запуск
------
1) Убедись что MT5 terminal запущен и подключён к broker (зелёный
   значок в правом нижнем углу MT5)
2) Двойной клик на run.bat
3) В окне должно появиться:
     Starting ApexFX MT5 bot starting ...
     MT5 connected attempt=1
     TurtleRunner starting symbols=[...] timeframe=H4
4) Если настроен Telegram — придёт сообщение "ApexFX bot started"

Бот теперь работает. Каждые 60 секунд проверяет новые H4 бары
и торгует при breakout.

Остановка
---------
Способ 1 (мгновенно): в окне с ботом нажми Ctrl-C
Способ 2 (graceful): создай пустой файл с именем .kill_switch
                     в той же папке.  Бот закроет текущий цикл
                     и выйдет.

Поведение
---------
* Bot читает H4-бары через MT5 Python API (нет webhook, нет TradingView)
* На каждом breakout открывает позицию через mt5.order_send()
* Каждая позиция получает SL = -2*ATR от entry
* Pyramid: добавляет unit каждые +0.5*ATR прибыли (макс 4 unit)
* При daily DD >8% или 3 подряд rejection — kill switch активируется
* Все события приходят в Telegram (если включён)

Что НЕ делает
-------------
* НЕ закрывает позиции вручную — только по правилам стратегии
* НЕ торгует на новостях (нет news-filter)
* НЕ adapts параметры — стратегия фиксированная
* НЕ работает если MT5 terminal закрыт или вышел из broker

Troubleshooting
---------------
* "Could not connect to MT5"
  → Запусти MT5 terminal, залогинься, нажми Algo Trading

* "Login failed"
  → Проверь login/password/server в my_bot.py
  → password — это TRADING password, не investor

* "ImportError: MetaTrader5"
  → Запусти install.bat снова
  → Проверь Python 3.11 (НЕ 3.12)

* "Position not found after order"
  → magic number конфликт; в my_bot.py добавь magic_number=999999

* Бот не открывает сделки 1-2 дня
  → Нормально на H4: breakout случается 1-3 раз/неделю на пару
  → Включи log_level="DEBUG" в config чтобы видеть каждое решение

Файлы в этой папке
------------------
my_bot.py           — ТВОЯ конфигурация (редактируй)
install.bat         — установщик (запусти один раз)
run.bat             — запуск бота (двойной клик чтобы стартовать)
requirements.txt    — список Python-пакетов
README.txt          — этот файл
apexfx/             — исходный код бота (НЕ редактируй)

После первого запуска появятся:
.venv/              — Python virtual environment
.kill_switch        — создавай чтобы остановить бот
.breaker_state.json — состояние risk engine
*.log               — логи

==============================================================
"""


# ---------------------------------------------------------------------------
# Main packaging logic
# ---------------------------------------------------------------------------


def main() -> int:
    out_path = REPO_ROOT / "apexfx_mt5_deploy.zip"
    if out_path.exists():
        out_path.unlink()

    # Sanity check — all source files must exist
    missing: list[str] = []
    for relpath in FILES_TO_BUNDLE:
        if not (REPO_ROOT / relpath).exists():
            missing.append(relpath)
    if missing:
        print("ERROR: missing files:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    print(f"Creating {out_path.name}...")
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Source code — stripped of the 'src/' prefix so it imports
        #    cleanly when extracted at the bundle root.
        for relpath in FILES_TO_BUNDLE:
            src = REPO_ROOT / relpath
            # 'src/apexfx/foo.py' → 'apexfx/foo.py' inside the zip
            arcname = relpath.replace("src/", "", 1)
            zf.write(src, arcname=arcname)
            print(f"  + {arcname}")

        # 2. Generated files at the bundle root
        zf.writestr("requirements.txt", REQUIREMENTS_TXT)
        zf.writestr("my_bot.py", MY_BOT_PY)
        zf.writestr("install.bat", INSTALL_BAT)
        zf.writestr("run.bat", RUN_BAT)
        zf.writestr("README.txt", README_TXT)
        print("  + requirements.txt")
        print("  + my_bot.py (template)")
        print("  + install.bat")
        print("  + run.bat")
        print("  + README.txt")

    size_kb = out_path.stat().st_size / 1024
    print()
    print(f"✓ Created: {out_path}")
    print(f"  Size: {size_kb:.1f} KB")
    print()
    print("Next: copy this zip to your Windows machine, unpack, follow README.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
