"""Real-time drawdown tracking and breach detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DrawdownState:
    """Current drawdown state snapshot."""
    current_drawdown: float  # as fraction (0.03 = 3%)
    peak_value: float
    current_value: float
    drawdown_duration: int  # bars since drawdown started
    max_drawdown_ever: float
    warning_level: str  # "normal", "warning", "critical", "breached"


class DrawdownMonitor:
    """
    Tracks portfolio equity peak and current drawdown in real time.
    Triggers alerts and actions at configurable thresholds.
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.05,
        warning_threshold: float = 0.03,
        critical_threshold: float = 0.04,
    ) -> None:
        self._max_dd = max_drawdown_pct
        self._warning = warning_threshold
        self._critical = critical_threshold

        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._drawdown_start_step: int = 0
        self._current_step: int = 0
        self._max_dd_ever: float = 0.0
        self._in_drawdown: bool = False
        self._drawdown_history: list[tuple[datetime, float]] = []

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a fraction."""
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - self._current_value) / self._peak_value

    @property
    def is_breached(self) -> bool:
        return self.current_drawdown >= self._max_dd

    @property
    def is_critical(self) -> bool:
        return self.current_drawdown >= self._critical

    @property
    def is_warning(self) -> bool:
        return self.current_drawdown >= self._warning

    def update(self, portfolio_value: float) -> DrawdownState:
        """Update with latest portfolio value and return current state."""
        self._current_step += 1
        self._current_value = portfolio_value

        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._in_drawdown = False
        elif not self._in_drawdown:
            self._in_drawdown = True
            self._drawdown_start_step = self._current_step

        dd = self.current_drawdown
        self._max_dd_ever = max(self._max_dd_ever, dd)

        # Record history periodically
        now = datetime.now(UTC)
        self._drawdown_history.append((now, dd))
        if len(self._drawdown_history) > 10_000:
            self._drawdown_history = self._drawdown_history[-5_000:]

        # Determine warning level
        if dd >= self._max_dd:
            level = "breached"
            logger.error(
                "MAX DRAWDOWN BREACHED",
                drawdown=round(dd, 4),
                limit=self._max_dd,
                portfolio=portfolio_value,
            )
        elif dd >= self._critical:
            level = "critical"
            logger.warning("Drawdown critical", drawdown=round(dd, 4))
        elif dd >= self._warning:
            level = "warning"
            logger.info("Drawdown warning", drawdown=round(dd, 4))
        else:
            level = "normal"

        duration = self._current_step - self._drawdown_start_step if self._in_drawdown else 0

        return DrawdownState(
            current_drawdown=dd,
            peak_value=self._peak_value,
            current_value=self._current_value,
            drawdown_duration=duration,
            max_drawdown_ever=self._max_dd_ever,
            warning_level=level,
        )

    def reset(self, initial_value: float) -> None:
        """Reset the monitor."""
        self._peak_value = initial_value
        self._current_value = initial_value
        self._current_step = 0
        self._max_dd_ever = 0.0
        self._in_drawdown = False
        self._drawdown_history = []
