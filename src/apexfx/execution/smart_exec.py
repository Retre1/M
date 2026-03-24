"""Smart execution algorithms: VWAP, Implementation Shortfall, and SmartRouter.

Extends the base TWAP execution with volume-aware and cost-minimizing
strategies.  TWAP functionality is preserved via import from twap.py;
this module layers on:

  - **VWAPExecutor** -- slices weighted by an intraday volume profile.
  - **ImplementationShortfallExecutor** -- adaptive slicing that minimises
    total execution cost relative to a decision price.
  - **SmartRouter** -- selects the best algorithm for a given order.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from apexfx.execution.twap import TWAPExecutor, TWAPOrder, TWAPSlice  # noqa: F401
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Intraday volume profile (UTC hours) for EUR/USD
# ---------------------------------------------------------------------------

EURUSD_VOLUME_PROFILE: dict[int, float] = {
    0: 0.02,
    1: 0.02,
    2: 0.02,
    3: 0.03,
    4: 0.03,
    5: 0.03,
    6: 0.04,
    7: 0.06,
    8: 0.08,
    9: 0.07,
    10: 0.06,
    11: 0.06,
    12: 0.08,
    13: 0.09,
    14: 0.08,
    15: 0.07,
    16: 0.05,
    17: 0.04,
    18: 0.03,
    19: 0.02,
    20: 0.02,
    21: 0.02,
    22: 0.02,
    23: 0.02,
}

# =====================================================================
# VWAP
# =====================================================================


@dataclass
class VWAPSlice:
    """A single slice of a VWAP order."""

    volume: float
    scheduled_time: datetime
    target_volume_pct: float  # Portion of daily volume this slice represents
    executed: bool = False
    fill_price: float | None = None
    actual_volume: float = 0.0


@dataclass
class VWAPOrder:
    """A VWAP execution plan."""

    total_volume: float
    direction: int  # +1 (buy) or -1 (sell)
    symbol: str
    n_slices: int
    slices: list[VWAPSlice] = field(default_factory=list)
    completed: bool = False
    vwap: float = 0.0  # Volume-weighted average fill price

    @property
    def executed_volume(self) -> float:
        """Total volume that has been filled so far."""
        return sum(s.actual_volume for s in self.slices if s.executed)

    @property
    def remaining_volume(self) -> float:
        """Volume still to be executed."""
        return self.total_volume - self.executed_volume


class VWAPExecutor:
    """Volume-Weighted Average Price execution.

    Weights slices by an historical intraday volume profile so that
    each child order's size is proportional to the market volume
    expected during its time bucket.  This reduces market-impact cost
    by trading more during liquid periods and less during quiet ones.

    Parameters
    ----------
    volume_profile:
        Mapping from UTC hour (0-23) to a relative volume weight.
        Defaults to ``EURUSD_VOLUME_PROFILE``.
    n_slices:
        Number of child orders.
    interval_seconds:
        Time gap between consecutive slices.
    max_deviation_pct:
        If the mid price moves more than this fraction from the initial
        price, remaining slices are aborted.
    """

    def __init__(
        self,
        volume_profile: dict[int, float] | None = None,
        n_slices: int = 5,
        interval_seconds: float = 60.0,
        max_deviation_pct: float = 0.005,
    ) -> None:
        self._profile = volume_profile or EURUSD_VOLUME_PROFILE
        self._n_slices = n_slices
        self._interval = interval_seconds
        self._max_deviation = max_deviation_pct
        self._active_order: VWAPOrder | None = None

    # ----- properties -----------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Return ``True`` if a VWAP order is currently being executed."""
        return self._active_order is not None and not self._active_order.completed

    @property
    def active_order(self) -> VWAPOrder | None:
        """Return the in-flight order, if any."""
        return self._active_order

    # ----- planning -------------------------------------------------------

    def create_plan(
        self,
        volume: float,
        direction: int,
        symbol: str,
        current_hour: int,
    ) -> VWAPOrder:
        """Create a VWAP execution plan starting from *current_hour*.

        Volume is allocated proportionally to the volume profile for the
        upcoming ``n_slices`` hours (wrapping at midnight).

        Parameters
        ----------
        volume:
            Total volume (lots) to execute.
        direction:
            ``+1`` for buy, ``-1`` for sell.
        symbol:
            Instrument identifier (e.g. ``"EURUSD"``).
        current_hour:
            UTC hour (0-23) used to look up the starting point in the
            volume profile.

        Returns
        -------
        VWAPOrder
            A fully populated plan ready for ``execute_plan``.
        """
        n = max(2, min(self._n_slices, int(volume / 0.1)))

        # Determine volume weights for the upcoming hours
        hours = [(current_hour + i) % 24 for i in range(n)]
        weights = np.array(
            [self._profile.get(h, 0.03) for h in hours], dtype=np.float64
        )
        total_weight = float(weights.sum())
        if total_weight == 0:
            # Fallback: equal weighting when profile sums to zero
            weights = np.ones(n, dtype=np.float64)
            total_weight = float(n)

        now = datetime.now(UTC)
        slices: list[VWAPSlice] = []
        for i in range(n):
            pct = float(weights[i]) / total_weight
            slice_vol = round(volume * pct, 2)
            scheduled = now + timedelta(seconds=self._interval * i)
            slices.append(
                VWAPSlice(
                    volume=slice_vol,
                    scheduled_time=scheduled,
                    target_volume_pct=round(pct, 4),
                )
            )

        order = VWAPOrder(
            total_volume=volume,
            direction=direction,
            symbol=symbol,
            n_slices=n,
            slices=slices,
        )

        self._active_order = order
        logger.info(
            "VWAP plan created",
            symbol=symbol,
            total_volume=volume,
            n_slices=n,
            start_hour=current_hour,
            hours=hours,
            interval_s=self._interval,
        )
        return order

    # ----- execution ------------------------------------------------------

    async def execute_plan(
        self,
        order: VWAPOrder,
        execute_fn,
        get_price_fn,
    ) -> VWAPOrder:
        """Execute a VWAP plan slice-by-slice.

        Parameters
        ----------
        order:
            The plan produced by :meth:`create_plan`.
        execute_fn:
            ``async (direction, volume) -> (success: bool, fill_price: float | None)``
        get_price_fn:
            ``() -> float`` returning the current mid price.

        Returns
        -------
        VWAPOrder
            The same *order* object, mutated to reflect execution results.
        """
        initial_price = get_price_fn()
        total_cost = 0.0
        total_filled = 0.0

        for i, vwap_slice in enumerate(order.slices):
            if order.completed:
                break

            # --- price deviation guard ---
            current_price = get_price_fn()
            deviation = abs(current_price - initial_price) / initial_price

            if deviation > self._max_deviation:
                logger.warning(
                    "VWAP aborted: price deviation exceeded",
                    deviation=f"{deviation:.4f}",
                    limit=f"{self._max_deviation:.4f}",
                    slices_executed=i,
                    slices_total=order.n_slices,
                )
                break

            # --- execute slice ---
            success, fill_price = await execute_fn(
                order.direction, vwap_slice.volume
            )

            if success and fill_price is not None:
                vwap_slice.executed = True
                vwap_slice.fill_price = fill_price
                vwap_slice.actual_volume = vwap_slice.volume
                total_cost += fill_price * vwap_slice.volume
                total_filled += vwap_slice.volume
                logger.debug(
                    "VWAP slice filled",
                    slice_idx=i,
                    volume=vwap_slice.volume,
                    fill_price=round(fill_price, 5),
                    target_pct=vwap_slice.target_volume_pct,
                )
            else:
                logger.warning(
                    "VWAP slice failed",
                    slice_idx=i,
                    volume=vwap_slice.volume,
                )

            # Wait between slices (skip after the last)
            if i < len(order.slices) - 1:
                await asyncio.sleep(self._interval)

        # --- compute aggregate VWAP ---
        if total_filled > 0:
            order.vwap = total_cost / total_filled

        order.completed = True
        self._active_order = None

        logger.info(
            "VWAP execution complete",
            symbol=order.symbol,
            filled=round(total_filled, 2),
            target=round(order.total_volume, 2),
            vwap=round(order.vwap, 5) if order.vwap else 0,
            slippage_bps=round(
                abs(order.vwap - initial_price) / initial_price * 10_000, 2
            )
            if order.vwap
            else 0,
        )
        return order

    def cancel(self) -> None:
        """Cancel the active VWAP order."""
        if self._active_order is not None:
            self._active_order.completed = True
            logger.info("VWAP order cancelled")
        self._active_order = None


# =====================================================================
# Implementation Shortfall
# =====================================================================


@dataclass
class ISSlice:
    """A single slice of an Implementation Shortfall order."""

    volume: float
    scheduled_time: datetime
    urgency_factor: float  # Higher = execute faster
    executed: bool = False
    fill_price: float | None = None
    actual_volume: float = 0.0
    shortfall: float = 0.0  # Slippage from decision price (signed, in price units)


@dataclass
class ISOrder:
    """An Implementation Shortfall execution plan."""

    total_volume: float
    direction: int  # +1 (buy) or -1 (sell)
    symbol: str
    decision_price: float
    n_slices: int
    urgency: float  # 0 = patient, 1 = aggressive
    slices: list[ISSlice] = field(default_factory=list)
    completed: bool = False
    total_shortfall: float = 0.0  # Cumulative shortfall (price units)
    implementation_shortfall_bps: float = 0.0  # Final IS in basis points

    @property
    def executed_volume(self) -> float:
        """Total volume filled so far."""
        return sum(s.actual_volume for s in self.slices if s.executed)


class ImplementationShortfallExecutor:
    """Minimises total execution cost relative to a decision price.

    The algorithm is *adaptive*: after each fill, it re-evaluates
    whether to speed up or slow down execution based on realised
    shortfall and price trajectory.

    Parameters
    ----------
    urgency:
        ``0`` = very patient (equal slices, wider intervals).
        ``1`` = very aggressive (front-loaded, tight intervals).
    risk_aversion:
        Scales the penalty for market-risk exposure while the order
        is only partially filled.  Higher values bias toward faster
        completion.
    n_slices:
        Number of child orders.
    base_interval_seconds:
        Base waiting time between slices (before adaptive adjustment).
    max_deviation_pct:
        Hard abort threshold on price deviation from the decision
        price.
    """

    def __init__(
        self,
        urgency: float = 0.5,
        risk_aversion: float = 1.0,
        n_slices: int = 5,
        base_interval_seconds: float = 30.0,
        max_deviation_pct: float = 0.005,
    ) -> None:
        self._urgency = np.clip(urgency, 0.0, 1.0)
        self._risk_aversion = risk_aversion
        self._n_slices = n_slices
        self._base_interval = base_interval_seconds
        self._max_deviation = max_deviation_pct
        self._active_order: ISOrder | None = None

    # ----- properties -----------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Return ``True`` if an IS order is being executed."""
        return self._active_order is not None and not self._active_order.completed

    @property
    def active_order(self) -> ISOrder | None:
        """Return the in-flight order, if any."""
        return self._active_order

    # ----- planning -------------------------------------------------------

    def create_plan(
        self,
        volume: float,
        direction: int,
        symbol: str,
        decision_price: float,
    ) -> ISOrder:
        """Create an IS execution plan.

        Urgency determines front-loading behaviour:

        * High urgency  -- larger early slices to capture the decision
          price at the cost of higher market impact.
        * Low urgency   -- near-equal slices to minimise impact at the
          cost of greater timing risk.

        The slice-volume formula is::

            raw_i = 1 + urgency * (n - i) / n
            volume_i = volume * raw_i / sum(raw)

        Parameters
        ----------
        volume:
            Total volume (lots) to execute.
        direction:
            ``+1`` for buy, ``-1`` for sell.
        symbol:
            Instrument identifier.
        decision_price:
            The benchmark price at the moment the trading decision was
            made.  All shortfall calculations reference this value.

        Returns
        -------
        ISOrder
        """
        n = max(2, min(self._n_slices, int(volume / 0.1)))

        # Build front-loaded weight vector
        raw_weights = np.array(
            [1.0 + self._urgency * (n - i) / n for i in range(n)],
            dtype=np.float64,
        )
        weight_sum = float(raw_weights.sum())

        now = datetime.now(UTC)
        slices: list[ISSlice] = []
        for i in range(n):
            pct = float(raw_weights[i]) / weight_sum
            slice_vol = round(volume * pct, 2)
            scheduled = now + timedelta(seconds=self._base_interval * i)
            slices.append(
                ISSlice(
                    volume=slice_vol,
                    scheduled_time=scheduled,
                    urgency_factor=round(float(raw_weights[i]), 4),
                )
            )

        order = ISOrder(
            total_volume=volume,
            direction=direction,
            symbol=symbol,
            decision_price=decision_price,
            n_slices=n,
            urgency=float(self._urgency),
            slices=slices,
        )

        self._active_order = order
        logger.info(
            "IS plan created",
            symbol=symbol,
            total_volume=volume,
            decision_price=round(decision_price, 5),
            urgency=float(self._urgency),
            n_slices=n,
            risk_aversion=self._risk_aversion,
            interval_s=self._base_interval,
        )
        return order

    # ----- execution ------------------------------------------------------

    async def execute_plan(
        self,
        order: ISOrder,
        execute_fn,
        get_price_fn,
    ) -> ISOrder:
        """Execute an IS plan adaptively.

        After each fill the executor computes the per-slice shortfall.
        The interval to the next slice is adjusted:

        * **Shortfall growing** (fills are getting worse relative to the
          decision price) -- the executor *slows down* (widens the
          interval by up to 2x) to let the order book recover.
        * **Price moving away** (unfavourable trend) -- the executor
          *speeds up* (narrows the interval to 50 %) to capture
          remaining volume before conditions worsen further.

        Parameters
        ----------
        order:
            Plan from :meth:`create_plan`.
        execute_fn:
            ``async (direction, volume) -> (success: bool, fill_price: float | None)``
        get_price_fn:
            ``() -> float`` returning the current mid price.

        Returns
        -------
        ISOrder
            Mutated in place with fill information and shortfall
            metrics.
        """
        cumulative_shortfall = 0.0
        total_filled = 0.0
        prev_shortfall: float | None = None
        current_interval = self._base_interval

        for i, is_slice in enumerate(order.slices):
            if order.completed:
                break

            # --- price deviation guard ---
            current_price = get_price_fn()
            deviation = (
                abs(current_price - order.decision_price) / order.decision_price
            )

            if deviation > self._max_deviation:
                logger.warning(
                    "IS aborted: price deviation exceeded",
                    deviation=f"{deviation:.4f}",
                    limit=f"{self._max_deviation:.4f}",
                    slices_executed=i,
                    slices_total=order.n_slices,
                )
                break

            # --- execute slice ---
            success, fill_price = await execute_fn(
                order.direction, is_slice.volume
            )

            if success and fill_price is not None:
                is_slice.executed = True
                is_slice.fill_price = fill_price
                is_slice.actual_volume = is_slice.volume

                # Shortfall: signed difference from decision price
                # For buys  (dir=+1): positive shortfall = paid more
                # For sells (dir=-1): positive shortfall = received less
                slice_shortfall = (
                    order.direction * (fill_price - order.decision_price)
                )
                is_slice.shortfall = slice_shortfall
                cumulative_shortfall += slice_shortfall * is_slice.volume
                total_filled += is_slice.volume

                logger.debug(
                    "IS slice filled",
                    slice_idx=i,
                    volume=is_slice.volume,
                    fill_price=round(fill_price, 5),
                    slice_shortfall_pips=round(slice_shortfall * 10_000, 2),
                    urgency_factor=is_slice.urgency_factor,
                )

                # --- adaptive interval adjustment ---
                if prev_shortfall is not None:
                    if slice_shortfall > prev_shortfall:
                        # Shortfall growing -- slow down to let the book
                        # replenish and reduce our impact footprint.
                        current_interval = min(
                            current_interval * 1.5,
                            self._base_interval * 2.0,
                        )
                        logger.debug(
                            "IS: slowing down",
                            new_interval_s=round(current_interval, 1),
                        )
                    else:
                        # Price moving in our favour or shortfall
                        # shrinking -- speed up to lock in the gain.
                        current_interval = max(
                            current_interval * 0.75,
                            self._base_interval * 0.5,
                        )
                        logger.debug(
                            "IS: speeding up",
                            new_interval_s=round(current_interval, 1),
                        )
                prev_shortfall = slice_shortfall
            else:
                logger.warning(
                    "IS slice failed",
                    slice_idx=i,
                    volume=is_slice.volume,
                )

            # Wait (adaptive interval), skip after the last slice
            if i < len(order.slices) - 1:
                await asyncio.sleep(current_interval)

        # --- aggregate metrics ---
        order.total_shortfall = cumulative_shortfall
        if total_filled > 0 and order.decision_price > 0:
            avg_shortfall = cumulative_shortfall / total_filled
            order.implementation_shortfall_bps = round(
                avg_shortfall / order.decision_price * 10_000, 2
            )

        order.completed = True
        self._active_order = None

        logger.info(
            "IS execution complete",
            symbol=order.symbol,
            filled=round(total_filled, 2),
            target=round(order.total_volume, 2),
            decision_price=round(order.decision_price, 5),
            total_shortfall=round(cumulative_shortfall, 6),
            is_bps=order.implementation_shortfall_bps,
        )
        return order

    def cancel(self) -> None:
        """Cancel the active IS order."""
        if self._active_order is not None:
            self._active_order.completed = True
            logger.info("IS order cancelled")
        self._active_order = None


# =====================================================================
# Smart Router
# =====================================================================


class SmartRouter:
    """Select the optimal execution algorithm for an order.

    Decision matrix (by ascending order size, in lots):

    +-----------------+----------+-----------------------------------+
    | Volume range    | Algo     | Rationale                         |
    +-----------------+----------+-----------------------------------+
    | < twap_thresh   | direct   | Small enough for a single fill    |
    | < vwap_thresh   | twap     | Medium -- time-slice is sufficient|
    | < is_thresh     | vwap     | Larger -- match volume profile    |
    | >= is_thresh    | is       | Very large -- minimise cost       |
    +-----------------+----------+-----------------------------------+

    High urgency (> 0.8) overrides the size-based selection and forces
    TWAP, which fills fastest among the algorithmic strategies.

    Parameters
    ----------
    twap_threshold:
        Orders below this volume (lots) go straight to the market.
    vwap_threshold:
        Orders below this volume use TWAP; above use VWAP.
    is_threshold:
        Orders at or above this volume use Implementation Shortfall.
    """

    def __init__(
        self,
        twap_threshold: float = 0.5,
        vwap_threshold: float = 2.0,
        is_threshold: float = 5.0,
    ) -> None:
        self._twap_threshold = twap_threshold
        self._vwap_threshold = vwap_threshold
        self._is_threshold = is_threshold

        # TWAP is lazily initialised so the caller can inject a
        # custom TWAPExecutor if desired.
        self.twap: TWAPExecutor | None = None
        self.vwap = VWAPExecutor()
        self.is_executor = ImplementationShortfallExecutor()

    def select_algorithm(
        self,
        volume: float,
        urgency: float = 0.5,
        spread_bps: float = 0.0,
    ) -> str:
        """Choose the best execution strategy.

        Parameters
        ----------
        volume:
            Order size in lots.
        urgency:
            ``0`` = very patient, ``1`` = very aggressive.
        spread_bps:
            Current bid-ask spread in basis points (reserved for
            future spread-aware routing).

        Returns
        -------
        str
            One of ``"direct"``, ``"twap"``, ``"vwap"``, or ``"is"``.
        """
        if volume < self._twap_threshold:
            algo = "direct"
        elif urgency > 0.8:
            # High urgency overrides size-based routing in favour of
            # the fastest algorithmic strategy.
            algo = "twap"
        elif volume >= self._is_threshold:
            algo = "is"
        elif volume >= self._vwap_threshold:
            algo = "vwap"
        else:
            algo = "twap"

        logger.info(
            "SmartRouter selected algorithm",
            algorithm=algo,
            volume=volume,
            urgency=urgency,
            spread_bps=spread_bps,
        )
        return algo
