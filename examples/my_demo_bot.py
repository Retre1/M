"""Your personal trading bot — edit values, then run.

How to use this file
====================
1. Edit the CONFIG section below with YOUR MT5 demo credentials.
2. (Optional) Add Telegram bot token + chat ID for alerts.
3. (Optional) Tune strategy parameters — defaults are sane for $1k demo.
4. Run::

       python examples/my_demo_bot.py

The bot connects to your MT5 terminal, watches the symbols, and trades
on bar close.  Stop with Ctrl-C or by creating a ``.kill_switch`` file
in the working directory.

What you need before running
============================
* MT5 terminal installed (Windows) — desktop app or running on a VPS
* You've logged into your demo account at least once in the terminal
* AutoTrading enabled in MT5 (the green "Algo Trading" button)
* ``pip install MetaTrader5`` done in this venv
* On non-Windows: MT5 doesn't work natively — use a Windows VPS
  (see MT5_SETUP.md)

Getting your demo credentials
=============================
After logging in to your broker's demo:
  * **Login** — the 8-9 digit account number (shown in MT5 title bar)
  * **Password** — the *trading* password you set during registration
  * **Server** — Tools → Options → Server tab in MT5 shows it,
                 looks like "MetaQuotes-Demo" or "RoboForex-DemoPro"

This file lives in version control with PLACEHOLDER values.  Replace
them locally; never commit your real password.  Better: put real
secrets in env vars and use BotConfig.from_env() — see end of file.
"""

from __future__ import annotations

from apexfx.aggressive import (
    BotConfig,
    Mt5LoginConfig,
    TelegramSettings,
    run_bot,
)


# ============================================================================
# CONFIG — EDIT THESE VALUES
# ============================================================================

config = BotConfig(
    # ---- MT5 demo credentials ---------------------------------------------
    # Replace with YOUR demo account info (from broker welcome email or
    # the MT5 "Tools → Options → Server" / title bar).
    mt5=Mt5LoginConfig(
        login=12345678,                    # your demo account number
        password="YOUR-DEMO-PASSWORD",     # trading password (not investor)
        server="MetaQuotes-Demo",          # broker server name
    ),

    # ---- What to trade -----------------------------------------------------
    # Use the EXACT symbol names from your broker's Market Watch.
    # Some brokers add suffixes — "EURUSDp", "EURUSD.", "EURUSDcent" etc.
    symbols=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],

    timeframe="H4",                         # M1/M5/M15/H1/H4/D1
    deposit_currency="USD",                 # your account base currency

    # ---- Strategy parameters ----------------------------------------------
    # Defaults are tuned for $1k demo on H4.  Reduce risk_per_unit on
    # micro-live (0.005 = 0.5% per unit is conservative-aggressive).
    entry_period=20,                        # Donchian channel for entries
    exit_period=10,                         # Donchian channel for exits
    ema_period=200,                         # Trend filter EMA
    use_trend_filter=True,                  # Long only above EMA200, short below
    atr_period=20,                          # ATR lookback for sizing/stops
    risk_per_unit_pct=0.015,                # 1.5% of equity per unit
    stop_atr_mult=2.0,                      # Hard stop = 2N ATR
    pyramid_atr_mult=0.5,                   # Add unit every +0.5N profit
    max_units=4,                            # Max pyramid depth

    # ---- Risk limits ------------------------------------------------------
    # Bot stops trading (kill switch triggers) when these breach.
    daily_loss_pct=0.08,                    # 8% of equity in one day
    weekly_loss_pct=0.20,                   # 20% in one week
    monthly_dd_pct=0.35,                    # 35% from all-time high
    max_consecutive_failed_orders=3,        # Broker rejecting orders

    # ---- Execution settings -----------------------------------------------
    magic_number=770125,                    # Unique tag for our orders
    deviation_points=20,                    # Max slippage in points
    poll_interval_s=60.0,                   # How often to check for new bars

    # ---- Telegram alerts (optional) ---------------------------------------
    # Leave empty to disable.  To enable:
    # 1. @BotFather in Telegram → /newbot
    # 2. Save the token
    # 3. Message your new bot
    # 4. Open: https://api.telegram.org/bot<TOKEN>/getUpdates
    #    Find result[0].message.chat.id
    telegram=TelegramSettings(
        bot_token="",                       # e.g. "1234567890:AAHabc..."
        chat_id="",                         # e.g. "123456789"
    ),

    # ---- Logging ----------------------------------------------------------
    log_level="INFO",                       # DEBUG / INFO / WARNING / ERROR
)


# ============================================================================
# RUN — usually no need to edit below
# ============================================================================

if __name__ == "__main__":
    print(f"Starting bot: {config.summary()}")
    print("Stop with Ctrl-C or `touch .kill_switch`.")
    run_bot(config)


# ============================================================================
# Alternative: secrets from env vars (recommended for production)
# ============================================================================
#
# If you don't want to type real credentials into this file:
#
#     export APEXFX_MT5_LOGIN=12345678
#     export APEXFX_MT5_PASSWORD="your-password"
#     export APEXFX_MT5_SERVER="MetaQuotes-Demo"
#     export APEXFX_TELEGRAM_TOKEN="..."  # optional
#     export APEXFX_TELEGRAM_CHAT_ID="..."  # optional
#
# Then replace the config = BotConfig(...) block above with:
#
#     config = BotConfig.from_env()
