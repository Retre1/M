"""Circuit breaker — tracks PnL drawdowns and trips the kill switch.

The breaker is a stateful object that observes equity over time and
fires the kill switch when any of these thresholds breaches:

  * **Daily loss** — equity drop from day-start > X% (default 8%)
  * **Weekly loss** — equity drop from week-start > Y% (default 20%)
  * **Monthly DD** — equity drop from all-time-high > Z% (default 35%)
  * **Consecutive failed orders** — N in a row (default 3)

State is persisted to a JSON file so a bot restart doesn't reset the
day-start equity (which would let you sneak past daily limits via
restart-loop).

Why these specific thresholds (defaults)
----------------------------------------
For aggressive Crypto Turtle on $1k:
  * Daily 8%  = $80, lets you take 4 unit-losses (1.5% × 4 = 6%) plus
    fees / slippage room. If you exceed this, your strategy decayed
    or market is regime-shifting — pause and reassess.
  * Weekly 20% = $200, allows for 2-3 bad days before stopping.
  * Monthly 35% = $350, the "max DD I'm willing to see before
    fundamentally questioning the strategy".
  * Failed orders 3 = exchange is rejecting us, likely auth / rate
    limit / margin issue — stop placing orders.

These are tuned for "aggressive but not suicidal".  Reduce by half for
conservative variant.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from apexfx.aggressive.risk.kill_switch import KillSwitch
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Threshold tuning for the breaker — separate from the breaker itself
    so the same logic can run with different limits in tests / production.
    """

    daily_loss_pct: float = 0.08       # 8% of day-start equity
    weekly_loss_pct: float = 0.20      # 20% of week-start equity
    monthly_dd_pct: float = 0.35       # 35% from all-time high
    max_consecutive_failed_orders: int = 3

    def __post_init__(self) -> None:
        for name in ("daily_loss_pct", "weekly_loss_pct", "monthly_dd_pct"):
            v = getattr(self, name)
            if not 0 < v < 1:
                raise ValueError(f"{name} must be in (0, 1), got {v}")
        if self.max_consecutive_failed_orders < 1:
            raise ValueError("max_consecutive_failed_orders must be ≥ 1")


# ---------------------------------------------------------------------------
# Persisted state
# ---------------------------------------------------------------------------


@dataclass
class BreakerState:
    """JSON-serializable state of the circuit breaker."""

    day_start_equity: float = 0.0       # Equity at start of current day
    day_start_date: str = ""            # ISO date string for tracking day rollover
    week_start_equity: float = 0.0
    week_start_iso_week: str = ""       # "YYYY-WW" identifier
    all_time_high_equity: float = 0.0
    consecutive_failed_orders: int = 0
    last_updated: float = 0.0           # Unix epoch seconds

    def to_dict(self) -> dict:
        return {
            "day_start_equity": self.day_start_equity,
            "day_start_date": self.day_start_date,
            "week_start_equity": self.week_start_equity,
            "week_start_iso_week": self.week_start_iso_week,
            "all_time_high_equity": self.all_time_high_equity,
            "consecutive_failed_orders": self.consecutive_failed_orders,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BreakerState":
        return cls(**data)


# ---------------------------------------------------------------------------
# Trigger result — what triggered the breaker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TripResult:
    tripped: bool
    reason: str = ""
    threshold: float = 0.0
    observed: float = 0.0


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Watches equity over time, trips kill switch on threshold breach.

    Usage::

        breaker = CircuitBreaker(config, kill_switch, state_path=".breaker.json")
        breaker.observe_equity(current_equity_in_usdt)
        # If kill switch was tripped, breaker.state.consecutive_failed_orders
        # may still be 0 — daily/weekly check is independent of order failures.

        try:
            order = exchange.place_order(req)
            breaker.notify_order_success()
        except OrderRejectedError:
            breaker.notify_order_failure()
    """

    def __init__(
        self,
        config: CircuitBreakerConfig,
        kill_switch: KillSwitch,
        state_path: str | Path = ".breaker_state.json",
    ) -> None:
        self._config = config
        self._kill = kill_switch
        self._state_path = Path(state_path)
        self._state = self._load_state()

    @property
    def state(self) -> BreakerState:
        return self._state

    # ------------------------------------------------------------------
    # Equity observation — main entry point
    # ------------------------------------------------------------------

    def observe_equity(self, equity: float, *, now: datetime | None = None) -> TripResult:
        """Update internal state with current equity and check thresholds.

        ``now`` is injected for testability — production code passes None
        to use the wall clock.

        Returns a ``TripResult`` describing what (if anything) tripped.
        Side effect: if a threshold trips, the kill switch is triggered.
        """
        if equity <= 0:
            # Can't compute % drawdown on zero / negative equity — likely
            # we read balance during a transient.  Don't update state.
            return TripResult(tripped=False)

        ts = now or datetime.now(tz=UTC)
        self._roll_over_periods(equity, ts)

        # Update peak (for monthly DD)
        self._state.all_time_high_equity = max(
            self._state.all_time_high_equity, equity,
        )
        self._state.last_updated = time.time()

        # -- Daily check --
        daily_dd = (
            (self._state.day_start_equity - equity) / self._state.day_start_equity
            if self._state.day_start_equity > 0 else 0.0
        )
        if daily_dd > self._config.daily_loss_pct:
            return self._trip(
                "daily_loss_limit", self._config.daily_loss_pct, daily_dd,
            )

        # -- Weekly check --
        weekly_dd = (
            (self._state.week_start_equity - equity) / self._state.week_start_equity
            if self._state.week_start_equity > 0 else 0.0
        )
        if weekly_dd > self._config.weekly_loss_pct:
            return self._trip(
                "weekly_loss_limit", self._config.weekly_loss_pct, weekly_dd,
            )

        # -- Monthly drawdown from all-time high --
        peak = self._state.all_time_high_equity
        monthly_dd = (peak - equity) / peak if peak > 0 else 0.0
        if monthly_dd > self._config.monthly_dd_pct:
            return self._trip(
                "monthly_drawdown", self._config.monthly_dd_pct, monthly_dd,
            )

        self._save_state()
        return TripResult(tripped=False)

    # ------------------------------------------------------------------
    # Order outcome notifications
    # ------------------------------------------------------------------

    def notify_order_success(self) -> None:
        """Reset the consecutive-failure counter."""
        if self._state.consecutive_failed_orders > 0:
            self._state.consecutive_failed_orders = 0
            self._save_state()

    def notify_order_failure(self) -> TripResult:
        """Increment failure counter; trip if threshold exceeded."""
        self._state.consecutive_failed_orders += 1
        threshold = self._config.max_consecutive_failed_orders
        if self._state.consecutive_failed_orders >= threshold:
            return self._trip(
                "consecutive_failed_orders",
                float(threshold), float(self._state.consecutive_failed_orders),
            )
        self._save_state()
        return TripResult(tripped=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _roll_over_periods(self, equity: float, now: datetime) -> None:
        """If we're in a new day/week vs stored state, reset that period's
        anchor to the current equity.  Run BEFORE drawdown checks so the
        first observation of a new day doesn't false-trip on yesterday's
        loss."""
        today_iso = now.date().isoformat()
        if self._state.day_start_date != today_iso:
            logger.info("Circuit breaker: new day rollover",
                        prev_date=self._state.day_start_date,
                        new_date=today_iso, equity=equity)
            self._state.day_start_equity = equity
            self._state.day_start_date = today_iso

        # ISO week id "YYYY-Www" — Monday is start of week
        iso_year, iso_week, _ = now.isocalendar()
        week_id = f"{iso_year}-W{iso_week:02d}"
        if self._state.week_start_iso_week != week_id:
            logger.info("Circuit breaker: new week rollover",
                        prev_week=self._state.week_start_iso_week,
                        new_week=week_id, equity=equity)
            self._state.week_start_equity = equity
            self._state.week_start_iso_week = week_id

    def _trip(self, reason: str, threshold: float, observed: float) -> TripResult:
        """Fire the kill switch and return a structured trip result."""
        message = (
            f"{reason}: observed {observed:.2%} >= threshold {threshold:.2%}"
        )
        logger.error("Circuit breaker TRIPPED", reason=message)
        self._kill.trigger(message)
        # Persist state so post-mortem can reconstruct
        self._save_state()
        return TripResult(
            tripped=True, reason=reason,
            threshold=threshold, observed=observed,
        )

    def _load_state(self) -> BreakerState:
        if not self._state_path.exists():
            return BreakerState()
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            return BreakerState.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Circuit breaker state file corrupt; starting fresh",
                           error=str(exc))
            return BreakerState()

    def _save_state(self) -> None:
        try:
            self._state_path.write_text(
                json.dumps(self._state.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to persist circuit breaker state",
                         error=str(exc))
