"""Launch the Donchian Turtle strategy on MT5.

This is the single production entrypoint.  It wires:
  * Mt5Client (with credentials from env)
  * DonchianTurtle strategy (with TurtleConfig from env)
  * KillSwitch + CircuitBreaker (risk safety)
  * TelegramNotifier (event alerts)
  * TurtleRunner (main loop)

Setup
-----
1. Install MT5 terminal on Windows VPS, log in to broker account
2. ``pip install MetaTrader5``
3. Set env vars (see ``MT5_SETUP.md`` for full list)
4. ``python scripts/run_mt5_turtle.py``

The script blocks until kill switch fires or Ctrl-C.  Use systemd / Windows
service / nssm to keep it running 24/7.

Demo mode
---------
There's no explicit demo flag — demo vs live is determined by which account
the MT5 terminal is logged into.  ALWAYS start with demo:
   - Open broker registration
   - Choose "Open demo account" not real
   - Log MT5 terminal in
   - Run this script

Switching to live = just log out demo, log in live in MT5 terminal, restart
script.  Account number changes, code is identical.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make package importable when running as a script
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from apexfx.aggressive.alerts.telegram import (  # noqa: E402
    NullNotifier, TelegramConfig, TelegramNotifier,
)
from apexfx.aggressive.exchanges.mt5_client import (  # noqa: E402
    Mt5Client, Mt5Credentials,
)
from apexfx.aggressive.risk.circuit_breaker import (  # noqa: E402
    CircuitBreaker, CircuitBreakerConfig,
)
from apexfx.aggressive.risk.kill_switch import KillSwitch  # noqa: E402
from apexfx.aggressive.strategies.donchian_turtle import (  # noqa: E402
    DonchianTurtle, TurtleConfig,
)
from apexfx.aggressive.strategies.turtle_runner import TurtleRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _build_mt5_credentials_from_env() -> Mt5Credentials | None:
    """Build MT5 creds from env vars, or None to attach to running terminal."""
    login = os.environ.get("APEXFX_MT5_LOGIN", "").strip()
    password = os.environ.get("APEXFX_MT5_PASSWORD", "").strip()
    server = os.environ.get("APEXFX_MT5_SERVER", "").strip()
    if not (login and password and server):
        return None  # Attach to already-running terminal
    return Mt5Credentials(
        login=int(login), password=password, server=server,
        terminal_path=os.environ.get("APEXFX_MT5_PATH") or None,
    )


def _build_turtle_config_from_env() -> TurtleConfig:
    """Read strategy hyperparameters from env, with defaults."""
    return TurtleConfig(
        entry_period=int(os.environ.get("APEXFX_ENTRY_PERIOD", "20")),
        exit_period=int(os.environ.get("APEXFX_EXIT_PERIOD", "10")),
        ema_period=int(os.environ.get("APEXFX_EMA_PERIOD", "200")),
        use_trend_filter=os.environ.get("APEXFX_TREND_FILTER", "true").lower() != "false",
        atr_period=int(os.environ.get("APEXFX_ATR_PERIOD", "20")),
        risk_per_unit_pct=float(os.environ.get("APEXFX_RISK_PER_UNIT", "0.015")),
        stop_atr_mult=float(os.environ.get("APEXFX_STOP_ATR_MULT", "2.0")),
        pyramid_atr_mult=float(os.environ.get("APEXFX_PYRAMID_ATR_MULT", "0.5")),
        max_units=int(os.environ.get("APEXFX_MAX_UNITS", "4")),
    )


def _build_breaker_config_from_env() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        daily_loss_pct=float(os.environ.get("APEXFX_BREAKER_DAILY_PCT", "0.08")),
        weekly_loss_pct=float(os.environ.get("APEXFX_BREAKER_WEEKLY_PCT", "0.20")),
        monthly_dd_pct=float(os.environ.get("APEXFX_BREAKER_MONTHLY_PCT", "0.35")),
        max_consecutive_failed_orders=int(
            os.environ.get("APEXFX_BREAKER_MAX_FAILS", "3")
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Donchian Turtle on MT5.",
    )
    parser.add_argument(
        "--symbols", nargs="+",
        default=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        help="Symbols to trade (broker-specific naming).  Default majors.",
    )
    parser.add_argument(
        "--timeframe", default="H4",
        help="Bar timeframe (M1/M5/M15/H1/H4/D1).  Default H4.",
    )
    parser.add_argument(
        "--poll-seconds", type=float, default=60.0,
        help="How often to wake up and check for new bars.  Default 60s.",
    )
    parser.add_argument(
        "--deposit-currency", default="USD",
        help="Account deposit currency for equity reads.  Default USD.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle and exit (for cron-driven setups or smoke tests).",
    )
    parser.add_argument(
        "--magic", type=int, default=770125,
        help="MT5 magic number tag.  Use different per bot if running multiple.",
    )
    parser.add_argument(
        "--deviation", type=int, default=20,
        help="Max price deviation in points for market orders.",
    )
    args = parser.parse_args()

    # Logging — keep it simple
    logging.basicConfig(
        level=os.environ.get("APEXFX_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Build components
    mt5_creds = _build_mt5_credentials_from_env()
    exchange = Mt5Client(
        credentials=mt5_creds, magic=args.magic, deviation_points=args.deviation,
    )

    strategy = DonchianTurtle(_build_turtle_config_from_env())
    kill = KillSwitch()
    breaker = CircuitBreaker(
        config=_build_breaker_config_from_env(), kill_switch=kill,
    )
    tg_cfg = TelegramConfig.from_env()
    notifier = TelegramNotifier(tg_cfg) if tg_cfg else NullNotifier()

    runner = TurtleRunner(
        exchange=exchange,
        symbols=args.symbols,
        timeframe=args.timeframe,
        strategy=strategy,
        kill_switch=kill,
        breaker=breaker,
        notifier=notifier,
        deposit_currency=args.deposit_currency,
    )

    try:
        if args.once:
            runner.run_once()
        else:
            runner.run_forever(poll_interval_s=args.poll_seconds)
    finally:
        exchange.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
