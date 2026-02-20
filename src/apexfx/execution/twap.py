"""TWAP (Time-Weighted Average Price) execution for large orders.

Large orders should not be sent as a single market order — this causes
market impact and worse fill prices. TWAP splits the order into equal
slices executed over a time window.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TWAPSlice:
    """A single slice of a TWAP order."""
    volume: float
    scheduled_time: datetime
    executed: bool = False
    fill_price: float | None = None
    actual_volume: float = 0.0


@dataclass
class TWAPOrder:
    """A TWAP execution plan."""
    total_volume: float
    direction: int  # +1 or -1
    symbol: str
    n_slices: int
    interval_seconds: float
    slices: list[TWAPSlice] = field(default_factory=list)
    completed: bool = False
    vwap: float = 0.0  # volume-weighted average fill price

    @property
    def executed_volume(self) -> float:
        return sum(s.actual_volume for s in self.slices if s.executed)

    @property
    def remaining_volume(self) -> float:
        return self.total_volume - self.executed_volume


class TWAPExecutor:
    """Splits large orders into time-distributed slices.

    Configuration:
        volume_threshold: Orders above this (in lots) trigger TWAP.
        n_slices: Number of equal parts to split the order into.
        interval_seconds: Time between slices.
        max_deviation_pct: Max price deviation before aborting remaining slices.
    """

    def __init__(
        self,
        volume_threshold: float = 1.0,
        n_slices: int = 5,
        interval_seconds: float = 30.0,
        max_deviation_pct: float = 0.005,
    ) -> None:
        self._threshold = volume_threshold
        self._n_slices = n_slices
        self._interval = interval_seconds
        self._max_deviation = max_deviation_pct
        self._active_order: TWAPOrder | None = None

    @property
    def is_active(self) -> bool:
        return self._active_order is not None and not self._active_order.completed

    @property
    def active_order(self) -> TWAPOrder | None:
        return self._active_order

    def should_use_twap(self, volume: float) -> bool:
        """Check if the order is large enough to require TWAP."""
        return volume >= self._threshold

    def create_plan(
        self,
        volume: float,
        direction: int,
        symbol: str,
    ) -> TWAPOrder:
        """Create a TWAP execution plan.

        Splits the volume into n_slices equal parts scheduled
        at regular intervals from now.
        """
        n = max(2, min(self._n_slices, int(volume / 0.1)))  # At least 0.1 lots per slice
        slice_volume = volume / n

        now = datetime.now(timezone.utc)
        slices = []
        for i in range(n):
            scheduled = now + timedelta(seconds=self._interval * i)
            slices.append(TWAPSlice(
                volume=round(slice_volume, 2),
                scheduled_time=scheduled,
            ))

        order = TWAPOrder(
            total_volume=volume,
            direction=direction,
            symbol=symbol,
            n_slices=n,
            interval_seconds=self._interval,
            slices=slices,
        )

        self._active_order = order
        logger.info(
            "TWAP plan created",
            symbol=symbol,
            total_volume=volume,
            n_slices=n,
            slice_volume=round(slice_volume, 2),
            interval_s=self._interval,
        )

        return order

    async def execute_plan(
        self,
        order: TWAPOrder,
        execute_fn,
        get_price_fn,
    ) -> TWAPOrder:
        """Execute a TWAP plan by calling execute_fn for each slice.

        Args:
            order: The TWAP plan to execute.
            execute_fn: async fn(direction, volume) -> (success, fill_price)
            get_price_fn: fn() -> current_price
        """
        initial_price = get_price_fn()
        total_cost = 0.0
        total_filled = 0.0

        for i, slice_order in enumerate(order.slices):
            if order.completed:
                break

            # Check price deviation before each slice
            current_price = get_price_fn()
            deviation = abs(current_price - initial_price) / initial_price

            if deviation > self._max_deviation:
                logger.warning(
                    "TWAP aborted: price deviation exceeded",
                    deviation=f"{deviation:.4f}",
                    limit=f"{self._max_deviation:.4f}",
                    slices_executed=i,
                    slices_total=order.n_slices,
                )
                break

            # Execute this slice
            success, fill_price = await execute_fn(
                order.direction, slice_order.volume
            )

            if success and fill_price is not None:
                slice_order.executed = True
                slice_order.fill_price = fill_price
                slice_order.actual_volume = slice_order.volume
                total_cost += fill_price * slice_order.volume
                total_filled += slice_order.volume
            else:
                logger.warning(
                    "TWAP slice failed",
                    slice_idx=i,
                    volume=slice_order.volume,
                )

            # Wait between slices (except for the last one)
            if i < len(order.slices) - 1:
                await asyncio.sleep(order.interval_seconds)

        # Compute VWAP
        if total_filled > 0:
            order.vwap = total_cost / total_filled

        order.completed = True
        self._active_order = None

        logger.info(
            "TWAP execution complete",
            symbol=order.symbol,
            filled=round(total_filled, 2),
            target=round(order.total_volume, 2),
            vwap=round(order.vwap, 5) if order.vwap else 0,
            slippage_from_initial=round(
                abs(order.vwap - initial_price) / initial_price * 10000, 2
            ) if order.vwap else 0,
        )

        return order

    def cancel(self) -> None:
        """Cancel the active TWAP order."""
        if self._active_order is not None:
            self._active_order.completed = True
            logger.info("TWAP order cancelled")
        self._active_order = None
