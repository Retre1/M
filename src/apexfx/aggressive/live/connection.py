"""Resilient MT5 connection — wraps Mt5Client with auto-reconnect.

Why we need this
----------------
The bare ``Mt5Client`` raises on connection loss.  In a long-running
bot (24/7 for weeks) that means one transient network blip → script
dies → systemd/NSSM restarts it → small window where no signals can
fire.  Acceptable for retail, but we can do better with a 50-line
wrapper.

Failure modes the wrapper handles
---------------------------------
1. **Terminal not running** — sleeps, retries, alerts after N failures.
2. **Broker disconnect** (terminal still up, but no quote feed) — re-init
   the MT5 session, which forces broker reconnect.
3. **Transient API call failure** — the underlying ``Mt5Client`` retries
   inside ``order_send``; we don't double-up here.

Backoff
-------
Exponential up to 5 minutes max.  Even at the cap, daily ATR-based
strategy on H4 only misses 5 bars if disconnected the entire day —
acceptable.

State preservation
------------------
On reconnect we DO NOT reset the kill switch, breaker, or per-symbol
pyramid state.  Those are managed by their own files on disk; the
connection manager only re-creates the ``Mt5Client`` instance.
"""

from __future__ import annotations

import time
from typing import Any

from apexfx.aggressive.alerts.telegram import NullNotifier, TelegramNotifier
from apexfx.aggressive.config import BotConfig
from apexfx.aggressive.exchanges.base import ExchangeError
from apexfx.aggressive.exchanges.mt5_client import Mt5Client, Mt5Credentials
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# Max sleep between reconnect attempts.  We don't want to spam the broker
# during an outage but we also don't want to be down for hours.
_MAX_BACKOFF_S = 300.0
_INITIAL_BACKOFF_S = 5.0
_BACKOFF_MULTIPLIER = 2.0

# How many consecutive failures before we alert the user via Telegram.
# We don't alert on EVERY failure — random network blips would spam.
_ALERT_AFTER_N_FAILURES = 3


class ResilientMt5Connection:
    """Mt5Client wrapper with auto-reconnect on failure.

    Usage::

        conn = ResilientMt5Connection(config, notifier)
        conn.connect()  # initial connect with retries
        try:
            while True:
                client = conn.client  # always valid (raises if not connected)
                # ... use client ...
        finally:
            conn.shutdown()

    The ``client`` property always returns a working ``Mt5Client`` or
    raises ``ExchangeError`` if connection couldn't be established
    after ``max_initial_retries`` attempts.  After successful initial
    connect, transient failures are handled internally by ``reconnect()``.
    """

    def __init__(
        self,
        config: BotConfig,
        notifier: TelegramNotifier | NullNotifier | None = None,
        *,
        max_initial_retries: int = 5,
        mt5_module: object | None = None,
    ) -> None:
        self._config = config
        self._notifier = notifier or NullNotifier()
        self._max_initial = max_initial_retries
        self._mt5_module = mt5_module  # passed through for tests
        self._client: Mt5Client | None = None
        self._consecutive_failures = 0

    @property
    def client(self) -> Mt5Client:
        """Return the live client, or raise if not connected."""
        if self._client is None:
            raise ExchangeError("Not connected — call connect() first")
        return self._client

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish initial connection with up to ``max_initial_retries`` tries.

        Raises ``ExchangeError`` if every attempt fails — calling code
        should treat this as a startup-blocker (don't continue without
        connection).
        """
        backoff = _INITIAL_BACKOFF_S
        creds = Mt5Credentials(
            login=self._config.mt5.login,
            password=self._config.mt5.password,
            server=self._config.mt5.server,
            terminal_path=self._config.mt5.terminal_path,
        )
        last_err: Exception | None = None
        for attempt in range(1, self._max_initial + 1):
            try:
                self._client = Mt5Client(
                    credentials=creds,
                    magic=self._config.magic_number,
                    deviation_points=self._config.deviation_points,
                    mt5_module=self._mt5_module,
                )
                logger.info("MT5 connected", attempt=attempt)
                self._consecutive_failures = 0
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning(
                    "MT5 connect failed",
                    attempt=attempt, max_attempts=self._max_initial,
                    error=str(exc), next_backoff_s=backoff,
                )
                if attempt < self._max_initial:
                    time.sleep(backoff)
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_S)

        # All attempts failed
        msg = f"Could not connect to MT5 after {self._max_initial} attempts: {last_err}"
        self._notifier.notify_health_failure(component="mt5_connect", error=msg)
        raise ExchangeError(msg)

    def reconnect(self) -> bool:
        """Attempt a single reconnect.  Returns True on success.

        Use after a transient failure.  Does NOT loop — caller decides
        whether to retry.  Increments the consecutive-failure counter
        for alerting heuristics.
        """
        try:
            if self._client is not None:
                try:
                    self._client.shutdown()
                except Exception:  # noqa: BLE001
                    pass  # Ignore shutdown errors during a reconnect
            self._client = None
            self.connect()
            if self._consecutive_failures >= _ALERT_AFTER_N_FAILURES:
                # Connection restored — notify so user knows it recovered
                self._notifier.send(
                    f"✅ MT5 reconnected after {self._consecutive_failures} failures"
                )
            self._consecutive_failures = 0
            return True
        except Exception as exc:  # noqa: BLE001
            self._consecutive_failures += 1
            logger.error("Reconnect failed",
                         consecutive=self._consecutive_failures, error=str(exc))
            if self._consecutive_failures == _ALERT_AFTER_N_FAILURES:
                self._notifier.notify_health_failure(
                    component="mt5_reconnect",
                    error=f"{self._consecutive_failures} consecutive failures: {exc}",
                )
            return False

    def shutdown(self) -> None:
        """Close the connection cleanly.  Safe to call multiple times."""
        if self._client is not None:
            try:
                self._client.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Shutdown raised", error=str(exc))
            self._client = None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Quick liveness probe: try a cheap read; reconnect on failure.

        Returns True if the connection is healthy (after a reconnect if
        necessary), False if we're disconnected after attempting recovery.
        """
        if self._client is None:
            return self.reconnect()
        try:
            self._client.get_balance(self._config.deposit_currency)
            return True
        except ExchangeError as exc:
            logger.warning("Health check failed — attempting reconnect", error=str(exc))
            return self.reconnect()
