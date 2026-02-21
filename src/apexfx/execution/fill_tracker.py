"""Execution quality tracking: slippage, fill time, rejection rate."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FillRecord:
    """Record of a single order fill."""
    timestamp: datetime
    symbol: str
    direction: str
    expected_price: float
    actual_price: float
    volume: float
    fill_time_ms: float
    slippage_pips: float
    order_type: str  # "market" or "limit"


@dataclass
class ExecutionQuality:
    """Aggregated execution quality metrics."""
    avg_slippage_pips: float
    max_slippage_pips: float
    avg_fill_time_ms: float
    fill_rate: float  # fraction of successful fills
    rejection_rate: float
    n_fills: int
    limit_improvement_pips: float  # avg price improvement from limit orders


class FillTracker:
    """Tracks and reports execution quality metrics."""

    def __init__(self, pip_value: float = 0.0001, max_history: int = 1000) -> None:
        self._pip_value = pip_value
        self._fills: deque[FillRecord] = deque(maxlen=max_history)
        self._rejections: int = 0
        self._total_attempts: int = 0

    def record_fill(
        self,
        symbol: str,
        direction: str,
        expected_price: float,
        actual_price: float,
        volume: float,
        fill_time_ms: float,
        order_type: str = "market",
    ) -> FillRecord:
        """Record a successful order fill."""
        self._total_attempts += 1

        slippage = (actual_price - expected_price) / self._pip_value
        if direction == "SELL":
            slippage = -slippage  # Positive slippage = worse for us

        record = FillRecord(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            direction=direction,
            expected_price=expected_price,
            actual_price=actual_price,
            volume=volume,
            fill_time_ms=fill_time_ms,
            slippage_pips=slippage,
            order_type=order_type,
        )

        self._fills.append(record)

        if abs(slippage) > 1.0:
            logger.warning(
                "High slippage detected",
                slippage_pips=round(slippage, 2),
                symbol=symbol,
                direction=direction,
            )

        return record

    def record_rejection(self) -> None:
        """Record a rejected order."""
        self._total_attempts += 1
        self._rejections += 1

    def get_quality_report(self) -> ExecutionQuality:
        """Generate execution quality report."""
        if len(self._fills) == 0:
            return ExecutionQuality(
                avg_slippage_pips=0.0,
                max_slippage_pips=0.0,
                avg_fill_time_ms=0.0,
                fill_rate=0.0,
                rejection_rate=0.0,
                n_fills=0,
                limit_improvement_pips=0.0,
            )

        slippages = np.array([f.slippage_pips for f in self._fills])
        fill_times = np.array([f.fill_time_ms for f in self._fills])

        # Limit order improvement
        limit_fills = [f for f in self._fills if f.order_type == "limit"]
        if limit_fills:
            improvements = [-f.slippage_pips for f in limit_fills]  # Negative slippage = improvement
            limit_improvement = float(np.mean(improvements))
        else:
            limit_improvement = 0.0

        total = max(self._total_attempts, 1)

        return ExecutionQuality(
            avg_slippage_pips=float(np.mean(slippages)),
            max_slippage_pips=float(np.max(np.abs(slippages))),
            avg_fill_time_ms=float(np.mean(fill_times)),
            fill_rate=len(self._fills) / total,
            rejection_rate=self._rejections / total,
            n_fills=len(self._fills),
            limit_improvement_pips=limit_improvement,
        )

    def get_recent_slippage(self, n: int = 20) -> float:
        """Get average slippage of recent fills."""
        recent = list(self._fills)[-n:]
        if not recent:
            return 0.0
        return float(np.mean([f.slippage_pips for f in recent]))
