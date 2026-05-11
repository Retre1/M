"""High-level ``run_bot()`` — one function to wire everything and run.

Public entry point for end users.  Hides the assembly of Mt5Client,
strategy, risk engine, notifier, and runner behind a single call.

Usage::

    from apexfx.aggressive import run_bot, BotConfig, Mt5LoginConfig

    run_bot(BotConfig(
        mt5=Mt5LoginConfig(login=12345, password="x", server="MetaQuotes-Demo"),
        symbols=["EURUSD", "GBPUSD"],
        risk_per_unit_pct=0.005,
    ))

That's the whole user-facing API.  Everything else is internal.

Resilience
----------
The runner is wrapped in a try/except loop that catches ``ExchangeError``
and triggers a reconnect via ``ResilientMt5Connection``.  Strategy state
(pyramid counts, last bar times) is held in memory and survives reconnect
because we keep the ``TurtleRunner`` instance alive.

To stop the bot: Ctrl-C or create the kill-switch file.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from apexfx.aggressive.alerts.telegram import (
    NullNotifier, TelegramConfig, TelegramNotifier,
)
from apexfx.aggressive.config import BotConfig
from apexfx.aggressive.exchanges.base import ExchangeError
from apexfx.aggressive.live.connection import ResilientMt5Connection
from apexfx.aggressive.risk.circuit_breaker import CircuitBreaker
from apexfx.aggressive.risk.kill_switch import KillSwitch
from apexfx.aggressive.strategies.donchian_turtle import DonchianTurtle
from apexfx.aggressive.strategies.turtle_runner import TurtleRunner
from apexfx.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def run_bot(
    config: BotConfig,
    *,
    once: bool = False,
    mt5_module: Any | None = None,
) -> None:
    """Bring up the whole stack and run.

    Parameters
    ----------
    config : BotConfig
        All settings — credentials, strategy params, risk limits, alerts.
        See ``BotConfig`` docstring.
    once : bool
        If True, run a single cycle and return.  Useful for smoke tests
        and cron-driven setups.  Default False = run forever.
    mt5_module : Any | None
        For tests — inject a mock MetaTrader5 module.  Production code
        passes None (uses the real package).
    """
    # 1. Logging — set up before anything that might warn
    setup_logging(level=config.log_level, fmt="console")
    logger.info("ApexFX MT5 bot starting", config=config.summary())

    # 2. Notifier (Telegram or no-op)
    notifier: TelegramNotifier | NullNotifier
    if config.telegram.enabled:
        notifier = TelegramNotifier(TelegramConfig(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id,
        ))
        logger.info("Telegram alerts enabled")
    else:
        notifier = NullNotifier()
        logger.info("Telegram alerts disabled (no token/chat_id)")

    # 3. Resilient MT5 connection
    connection = ResilientMt5Connection(
        config=config, notifier=notifier, mt5_module=mt5_module,
    )
    try:
        connection.connect()
    except ExchangeError as exc:
        logger.error("Could not establish initial MT5 connection — aborting",
                     error=str(exc))
        raise

    # 4. Risk engine
    kill_switch = KillSwitch(flag_path=config.kill_switch_path)
    breaker = CircuitBreaker(
        config=config.build_breaker_config(),
        kill_switch=kill_switch,
        state_path=config.breaker_state_path,
    )

    # 5. Strategy + runner
    strategy = DonchianTurtle(config.build_turtle_config())
    runner = TurtleRunner(
        exchange=connection.client,
        symbols=config.symbols,
        timeframe=config.timeframe,
        strategy=strategy,
        kill_switch=kill_switch,
        breaker=breaker,
        notifier=notifier,
        deposit_currency=config.deposit_currency,
    )

    # 6. Startup notification
    notifier.send(
        f"🚀 *ApexFX bot started*\n"
        f"Account: `{config.mt5.login}@{config.mt5.server}`\n"
        f"Symbols: `{', '.join(config.symbols)}`\n"
        f"Timeframe: `{config.timeframe}`\n"
        f"Risk/unit: `{config.risk_per_unit_pct:.2%}`"
    )

    # 7. Run loop
    try:
        if once:
            runner.run_once()
        else:
            _run_forever_resilient(runner, connection, config, kill_switch, notifier)
    finally:
        notifier.send("🛑 ApexFX bot stopped")
        connection.shutdown()


# ---------------------------------------------------------------------------
# Internal — resilient main loop
# ---------------------------------------------------------------------------


def _run_forever_resilient(
    runner: TurtleRunner,
    connection: ResilientMt5Connection,
    config: BotConfig,
    kill_switch: KillSwitch,
    notifier: TelegramNotifier | NullNotifier,
) -> None:
    """Run cycles forever, reconnecting on exchange errors.

    Differs from ``TurtleRunner.run_forever`` in one critical way: when
    the underlying ``ExchangeError`` indicates a lost connection, we
    call ``connection.reconnect()`` and rebind the runner's exchange
    reference.  This keeps strategy state intact across the reconnect.
    """
    logger.info(
        "Entering main loop",
        poll_interval_s=config.poll_interval_s,
    )
    while True:
        # Kill switch check is non-negotiable
        if kill_switch.is_active():
            state = kill_switch.state()
            logger.warning("Kill switch active — stopping loop",
                           reason=state.reason)
            notifier.notify_kill_switch(state.reason)
            return

        try:
            runner.run_once()
        except ExchangeError as exc:
            logger.error("Cycle failed with ExchangeError — reconnecting",
                         error=str(exc))
            notifier.notify_health_failure(
                component="cycle", error=str(exc),
            )
            if connection.reconnect():
                # Rebind runner to new client instance
                runner._exchange = connection.client  # type: ignore[attr-defined]
                logger.info("Reconnect successful, runner rebinding")
            else:
                # Reconnect failed — wait full reconnect interval before retry
                time.sleep(config.reconnect_interval_s)
                continue
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — exiting cleanly")
            return
        except Exception as exc:
            # Unexpected error — log full traceback, notify, continue
            logger.exception("Unexpected error in cycle", error=str(exc))
            notifier.notify_health_failure(
                component="cycle", error=f"unexpected: {exc}",
            )

        time.sleep(config.poll_interval_s)


def setup_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Stub: most projects already have setup_logging.  We provide a
    minimal version so this module is self-contained."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
