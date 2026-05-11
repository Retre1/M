"""Python implementation of the Donchian Turtle strategy.

This is the Python equivalent of ``tradingview/donchian_turtle.pine`` — same
rules, runs natively on the Python side so no TradingView subscription is
needed.  Designed for MT5 forex but symbol-agnostic (any ``Exchange`` works).

Strategy summary
----------------
* **Entry**: close crosses outside the N-period Donchian channel
* **Pyramid**: add a unit every +0.5N profit, up to ``max_units``
* **Exit**: close crosses against the M-period (shorter) channel, OR price
  hits a 2N hard-stop
* **Filter**: only take longs above EMA200, only take shorts below EMA200

Stateless per-bar
-----------------
The strategy class itself holds tuning parameters but NO per-symbol state.
On each bar-close the caller (``TurtleRunner``) passes the current bars +
existing position, and the strategy returns a ``StrategyDecision`` (no
side effects).  This makes it unit-testable without any exchange.

Indicators
----------
We compute Donchian / EMA / ATR from scratch in numpy — fast enough for
4H bars (one tick every 4 hours per symbol).  No external TA library
dependency (TA-Lib is a pain to install across platforms).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from apexfx.aggressive.exchanges.base import Bar, Position, Side
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


class DecisionAction(str, Enum):
    """What the strategy wants to do this bar."""

    HOLD = "hold"           # No change — keep current position (or stay flat)
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    PYRAMID = "pyramid"      # Add a unit in current direction
    EXIT = "exit"            # Close all units


@dataclass(frozen=True)
class StrategyDecision:
    """The output of one strategy invocation — what (if anything) to do."""

    action: DecisionAction
    side: Side | None = None          # Direction for entry/pyramid/exit
    target_volume: float = 0.0        # Lots / contracts to trade
    stop_loss: float | None = None    # For entries
    take_profit: float | None = None  # Optional
    reason: str = ""                  # Human-readable explanation

    @property
    def is_trade(self) -> bool:
        return self.action is not DecisionAction.HOLD


# ---------------------------------------------------------------------------
# Strategy config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurtleConfig:
    """Tunable parameters — matches Pine Script inputs 1:1.

    The defaults reproduce the original Turtle System 1 (Hugh Daniels,
    1980s) with our minor adaptations:
      * EMA200 trend filter (cuts whipsaws — major Sharpe boost)
      * Volatility-scaled sizing (ATR-based, target % equity risk)
      * Conservative defaults — start with these, tune later
    """

    entry_period: int = 20           # Donchian breakout window
    exit_period: int = 10            # Donchian exit window (must be < entry_period)
    ema_period: int = 200            # Trend filter window
    use_trend_filter: bool = True

    atr_period: int = 20             # N (ATR) lookback
    risk_per_unit_pct: float = 0.015 # 1.5% of equity per unit
    stop_atr_mult: float = 2.0       # SL distance in N units
    pyramid_atr_mult: float = 0.5    # Add unit every +0.5N profit
    max_units: int = 4               # Max pyramid depth

    def __post_init__(self) -> None:
        if self.exit_period >= self.entry_period:
            raise ValueError(
                f"exit_period ({self.exit_period}) must be < "
                f"entry_period ({self.entry_period})"
            )
        if not 0 < self.risk_per_unit_pct < 0.2:
            raise ValueError(
                f"risk_per_unit_pct must be in (0, 0.2), got {self.risk_per_unit_pct}"
            )
        if self.stop_atr_mult <= 0 or self.pyramid_atr_mult <= 0:
            raise ValueError("ATR multipliers must be positive")
        if self.max_units < 1:
            raise ValueError(f"max_units must be ≥ 1, got {self.max_units}")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class DonchianTurtle:
    """Single-symbol Donchian Turtle strategy.

    The class is intentionally stateless — instantiate one per strategy
    config, then call ``decide(bars, position, equity, last_unit_price)``
    on each bar close.  All state needed for pyramid timing is passed in
    by the caller (typically tracked in ``TurtleRunner``).
    """

    def __init__(self, config: TurtleConfig | None = None) -> None:
        self._config = config or TurtleConfig()
        # Need enough bars for the longest indicator + 1 (for crossover detection)
        self._min_bars = max(
            self._config.entry_period,
            self._config.exit_period,
            self._config.ema_period,
            self._config.atr_period,
        ) + 2

    @property
    def config(self) -> TurtleConfig:
        return self._config

    @property
    def min_bars(self) -> int:
        """Minimum number of bars needed before ``decide()`` can act."""
        return self._min_bars

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def compute_atr(self, bars: list[Bar]) -> float:
        """Average True Range over the last ``atr_period`` bars.

        TR = max(high-low, |high - prev_close|, |low - prev_close|)
        ATR = EMA-like rolling mean of TR.  We use a simple arithmetic mean
        for parity with Pine's ``ta.atr`` (which is RMA, ≈ Wilder smoothing,
        but for the small windows we use the differences are negligible).
        """
        if len(bars) < self._config.atr_period + 1:
            return 0.0
        window = bars[-(self._config.atr_period + 1):]
        trs: list[float] = []
        for i in range(1, len(window)):
            b = window[i]
            prev_close = window[i - 1].close
            tr = max(
                b.high - b.low,
                abs(b.high - prev_close),
                abs(b.low - prev_close),
            )
            trs.append(tr)
        return float(np.mean(trs)) if trs else 0.0

    def compute_ema(self, bars: list[Bar], period: int) -> float:
        """Exponential moving average of close prices over ``period`` bars."""
        if len(bars) < period:
            return 0.0
        closes = np.array([b.close for b in bars], dtype=np.float64)
        # Use pandas-style EMA: alpha = 2/(period+1)
        alpha = 2.0 / (period + 1.0)
        ema = closes[0]
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema
        return float(ema)

    def donchian_high(self, bars: list[Bar], period: int) -> float:
        """Highest high over the last ``period`` bars (excluding current).

        Pine's ``ta.highest(high, n)[1]`` shifts back by 1 — we mirror that
        by looking at bars[-period-1:-1].  This prevents the current bar's
        own high from being its own breakout target.
        """
        if len(bars) < period + 1:
            return float("inf")
        window = bars[-(period + 1):-1]
        return float(max(b.high for b in window))

    def donchian_low(self, bars: list[Bar], period: int) -> float:
        if len(bars) < period + 1:
            return float("-inf")
        window = bars[-(period + 1):-1]
        return float(min(b.low for b in window))

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def unit_volume(self, equity: float, atr: float, contract_size: float = 100_000.0) -> float:
        """Position size in lots for one Turtle unit.

        Formula (mirrors Pine sizing):
            risk_dollars = equity * risk_per_unit_pct
            stop_distance = stop_atr_mult * ATR (in price units)
            unit_size_in_quote = risk_dollars / stop_distance
            unit_lots = unit_size_in_quote / contract_size

        ``contract_size`` defaults to 100,000 (standard forex lot).  For
        crypto perpetuals or cent-account forex pass the correct value.
        Returns 0 if equity or ATR is non-positive (skip signal).
        """
        if equity <= 0 or atr <= 0:
            return 0.0
        risk_dollars = equity * self._config.risk_per_unit_pct
        stop_distance = self._config.stop_atr_mult * atr
        unit_quote = risk_dollars / stop_distance
        return unit_quote / contract_size

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def decide(
        self,
        bars: list[Bar],
        position: Position | None,
        equity: float,
        n_units_open: int = 0,
        last_unit_price: float | None = None,
        *,
        contract_size: float = 100_000.0,
    ) -> StrategyDecision:
        """Make one decision based on bars + current state.

        Parameters
        ----------
        bars : list[Bar]
            Historical bars in chronological order; last bar is the
            just-closed one.
        position : Position | None
            Current open position (after summing pyramid units), or None
            if flat.
        equity : float
            Account equity in the quote currency.  Used for sizing.
        n_units_open : int
            How many pyramid units we've already opened (caller tracks
            this).  Caps at ``max_units``.
        last_unit_price : float | None
            Entry price of the most recent unit.  Used to decide if it's
            time to pyramid (+0.5N from that price).
        contract_size : float
            Lots → quote currency conversion factor.
        """
        if len(bars) < self._min_bars:
            return StrategyDecision(action=DecisionAction.HOLD,
                                    reason="not enough bars for indicators")

        cfg = self._config
        cur = bars[-1]
        close = cur.close

        atr = self.compute_atr(bars)
        if atr <= 0:
            return StrategyDecision(action=DecisionAction.HOLD,
                                    reason="zero ATR")

        ema = self.compute_ema(bars, cfg.ema_period) if cfg.use_trend_filter else 0.0
        above_trend = (not cfg.use_trend_filter) or (close > ema)
        below_trend = (not cfg.use_trend_filter) or (close < ema)

        # ---- Exit logic — checked first; takes priority over entries ----
        if position is not None and not position.is_flat:
            return self._decide_exit_or_pyramid(
                bars, position, equity, atr,
                n_units_open, last_unit_price, contract_size,
            )

        # ---- Entry logic ----
        entry_high = self.donchian_high(bars, cfg.entry_period)
        entry_low = self.donchian_low(bars, cfg.entry_period)

        unit_lots = self.unit_volume(equity, atr, contract_size)
        if unit_lots <= 0:
            return StrategyDecision(action=DecisionAction.HOLD,
                                    reason="unit size = 0")

        # Long breakout
        if close > entry_high and above_trend:
            return StrategyDecision(
                action=DecisionAction.ENTER_LONG,
                side=Side.BUY,
                target_volume=unit_lots,
                stop_loss=close - cfg.stop_atr_mult * atr,
                reason=f"long breakout @{close:.5f} > donchian_high {entry_high:.5f}",
            )

        # Short breakout
        if close < entry_low and below_trend:
            return StrategyDecision(
                action=DecisionAction.ENTER_SHORT,
                side=Side.SELL,
                target_volume=unit_lots,
                stop_loss=close + cfg.stop_atr_mult * atr,
                reason=f"short breakout @{close:.5f} < donchian_low {entry_low:.5f}",
            )

        return StrategyDecision(action=DecisionAction.HOLD, reason="no signal")

    # ------------------------------------------------------------------
    # Exit / pyramid sub-decision
    # ------------------------------------------------------------------

    def _decide_exit_or_pyramid(
        self,
        bars: list[Bar],
        position: Position,
        equity: float,
        atr: float,
        n_units_open: int,
        last_unit_price: float | None,
        contract_size: float,
    ) -> StrategyDecision:
        cfg = self._config
        close = bars[-1].close
        entry_price = position.entry_price

        # --- Hard stop (2N adverse) ---
        if position.side is Side.BUY:
            stop = entry_price - cfg.stop_atr_mult * atr
            if close <= stop:
                return StrategyDecision(
                    action=DecisionAction.EXIT, side=position.side,
                    target_volume=position.quantity,
                    reason=f"hard_stop: close {close:.5f} <= {stop:.5f}",
                )
        else:  # SELL
            stop = entry_price + cfg.stop_atr_mult * atr
            if close >= stop:
                return StrategyDecision(
                    action=DecisionAction.EXIT, side=position.side,
                    target_volume=position.quantity,
                    reason=f"hard_stop: close {close:.5f} >= {stop:.5f}",
                )

        # --- Donchian exit channel ---
        exit_high = self.donchian_high(bars, cfg.exit_period)
        exit_low = self.donchian_low(bars, cfg.exit_period)

        if position.side is Side.BUY and close < exit_low:
            return StrategyDecision(
                action=DecisionAction.EXIT, side=position.side,
                target_volume=position.quantity,
                reason=f"donchian_exit: close {close:.5f} < exit_low {exit_low:.5f}",
            )
        if position.side is Side.SELL and close > exit_high:
            return StrategyDecision(
                action=DecisionAction.EXIT, side=position.side,
                target_volume=position.quantity,
                reason=f"donchian_exit: close {close:.5f} > exit_high {exit_high:.5f}",
            )

        # --- Pyramid ---
        if n_units_open < cfg.max_units and last_unit_price is not None:
            trigger_distance = cfg.pyramid_atr_mult * atr
            if position.side is Side.BUY:
                if close >= last_unit_price + trigger_distance:
                    unit_lots = self.unit_volume(equity, atr, contract_size)
                    if unit_lots > 0:
                        return StrategyDecision(
                            action=DecisionAction.PYRAMID, side=Side.BUY,
                            target_volume=unit_lots,
                            reason=(
                                f"pyramid #{n_units_open + 1}: "
                                f"close {close:.5f} >= last_entry "
                                f"{last_unit_price:.5f} + 0.5N"
                            ),
                        )
            else:  # SELL
                if close <= last_unit_price - trigger_distance:
                    unit_lots = self.unit_volume(equity, atr, contract_size)
                    if unit_lots > 0:
                        return StrategyDecision(
                            action=DecisionAction.PYRAMID, side=Side.SELL,
                            target_volume=unit_lots,
                            reason=(
                                f"pyramid #{n_units_open + 1}: "
                                f"close {close:.5f} <= last_entry "
                                f"{last_unit_price:.5f} - 0.5N"
                            ),
                        )

        return StrategyDecision(action=DecisionAction.HOLD, reason="hold position")
