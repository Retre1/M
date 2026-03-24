"""Losing streak cooldown and tilt detection."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class CooldownManager:
    """
    Enforces trading pauses after losing streaks (overtrading prevention)
    and detects rapid drawdown acceleration ("tilt").
    """

    def __init__(
        self,
        after_n_losses: int = 3,
        duration_minutes: int = 60,
        tilt_drawdown_pct: float = 0.02,
        tilt_window_minutes: int = 30,
    ) -> None:
        self._max_losses = after_n_losses
        self._duration = timedelta(minutes=duration_minutes)
        self._tilt_dd_threshold = tilt_drawdown_pct
        self._tilt_window = timedelta(minutes=tilt_window_minutes)

        self._consecutive_losses: int = 0
        self._cooldown_until: datetime | None = None
        self._trade_results: deque[tuple[datetime, float]] = deque(maxlen=100)
        self._portfolio_snapshots: deque[tuple[datetime, float]] = deque(maxlen=1000)

    @property
    def is_active(self) -> bool:
        """Check if cooldown is currently in effect."""
        if self._cooldown_until is None:
            return False
        return datetime.now(UTC) < self._cooldown_until

    @property
    def remaining(self) -> timedelta | None:
        """Time remaining in cooldown, or None if not active."""
        if not self.is_active or self._cooldown_until is None:
            return None
        return self._cooldown_until - datetime.now(UTC)

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    def record_trade(self, pnl: float) -> None:
        """Record a trade result and check for cooldown triggers."""
        now = datetime.now(UTC)
        self._trade_results.append((now, pnl))

        if pnl < 0:
            self._consecutive_losses += 1
            logger.debug(
                "Losing trade recorded",
                consecutive_losses=self._consecutive_losses,
            )

            if self._consecutive_losses >= self._max_losses:
                self._activate_cooldown("consecutive_losses")
        else:
            self._consecutive_losses = 0

    def record_portfolio_value(self, value: float) -> None:
        """Record portfolio value for tilt detection."""
        now = datetime.now(UTC)
        self._portfolio_snapshots.append((now, value))
        self._check_tilt()

    def _check_tilt(self) -> None:
        """Detect rapid drawdown acceleration within the tilt window."""
        if len(self._portfolio_snapshots) < 2:
            return

        now = datetime.now(UTC)
        cutoff = now - self._tilt_window

        # Find the portfolio value at the start of the tilt window
        start_value = None
        for ts, val in self._portfolio_snapshots:
            if ts >= cutoff:
                start_value = val
                break

        if start_value is None or start_value <= 0:
            return

        current_value = self._portfolio_snapshots[-1][1]
        window_drawdown = (start_value - current_value) / start_value

        if window_drawdown >= self._tilt_dd_threshold:
            self._activate_cooldown("tilt_detection")

    def _activate_cooldown(self, reason: str) -> None:
        """Activate the cooldown period."""
        now = datetime.now(UTC)
        self._cooldown_until = now + self._duration
        self._consecutive_losses = 0

        logger.warning(
            "COOLDOWN ACTIVATED",
            reason=reason,
            until=self._cooldown_until.isoformat(),
            duration_minutes=self._duration.total_seconds() / 60,
        )

    def force_reset(self) -> None:
        """Force-reset the cooldown (manual override)."""
        self._cooldown_until = None
        self._consecutive_losses = 0
        logger.info("Cooldown manually reset")
