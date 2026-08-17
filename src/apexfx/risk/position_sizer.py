"""Dynamic position sizing: risk per trade + Kelly scaling + hard caps."""

from __future__ import annotations

import numpy as np

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class PositionSizer:
    """Sizes a position from what it costs to be wrong.

    The size follows from the risk taken if the stop is hit::

        lots = equity * risk_per_trade * confidence * kelly
               ------------------------------------------
                        stop_distance * contract_size

    with ``stop_distance`` defaulting to ``atr_stop_mult * ATR``, matching the
    stop the engine actually places.

    **Why it works this way.** Sizing used to cap *notional* at
    ``max_position_pct`` of equity. On EURUSD 10% of $100k of notional is 0.1
    lots, and Kelly's warm-up value cut that to 0.02 — a trade risking 0.004%
    of equity against a 2xATR stop. Measured over a run: 671 of 1155 decisions
    rejected as "Position size computed to zero" and 0.045% average exposure.
    A backtest at that size cannot show edge or its absence, whatever its
    Sharpe reads. The parameter name promised risk control and delivered a
    notional cap two orders of magnitude below what it sounded like.

    ``max_position_pct`` still governs the fallback path, where no stop
    distance is known and risk-based sizing has no denominator.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        kelly_fraction: float = 0.5,
        min_trades_for_kelly: int = 30,
        vol_lookback_bars: int = 20,
        min_lot_size: float = 0.01,
        contract_size: float = 100_000.0,
        risk_per_trade_pct: float = 0.01,
        max_leverage: float = 10.0,
        atr_stop_mult: float = 2.0,
    ) -> None:
        self._max_pct = max_position_pct
        self._kelly_frac = kelly_fraction
        self._min_kelly_trades = min_trades_for_kelly
        self._vol_lookback = vol_lookback_bars
        self._min_lot = min_lot_size
        self._contract_size = contract_size
        self._risk_per_trade = risk_per_trade_pct
        self._max_leverage = max_leverage
        self._atr_stop_mult = atr_stop_mult

        self._trade_wins: int = 0
        self._trade_losses: int = 0
        self._avg_win: float = 0.0
        self._avg_loss: float = 0.0

    def update_trade_stats(self, trade_return: float) -> None:
        """Update running trade statistics for Kelly computation.

        Args:
            trade_return: Return on risk (PnL / notional value), NOT absolute PnL.
                          e.g. +0.02 for a 2% gain, -0.01 for a 1% loss.
                          Using returns instead of absolute PnL ensures Kelly fraction
                          is correctly computed regardless of position size.
        """
        if trade_return > 0:
            self._trade_wins += 1
            n = self._trade_wins
            self._avg_win = self._avg_win * ((n - 1) / n) + trade_return / n
        elif trade_return < 0:
            self._trade_losses += 1
            n = self._trade_losses
            self._avg_loss = self._avg_loss * ((n - 1) / n) + abs(trade_return) / n

    def compute(
        self,
        action: float,
        portfolio_value: float,
        current_price: float,
        current_atr: float | None = None,
        historical_atr: float | None = None,
        stop_distance: float | None = None,
    ) -> float:
        """
        Compute position size in lots.

        Args:
            action: model output in [-1, 1]
            portfolio_value: current portfolio value
            current_price: current market price
            current_atr: current ATR, used to derive the stop distance
            historical_atr: historical average ATR (fallback path only)
            stop_distance: stop distance in price units. Pass the stop that
                will actually be placed — sizing against a different one
                misstates the risk by exactly their ratio.

        Returns:
            Position size in lots (always positive; direction from action sign)
        """
        confidence = abs(action)
        if confidence <= 0.0:
            return 0.0

        lot_value = current_price * self._contract_size
        if lot_value <= 0:
            return 0.0

        kelly = self._compute_kelly()
        stop = self._stop_distance(stop_distance, current_atr)

        if stop is None:
            return self._notional_size(confidence, kelly, portfolio_value,
                                       current_atr, historical_atr, lot_value)

        # Risk-based sizing. No separate volatility scalar: a wider stop
        # already buys fewer lots for the same money, so applying the inverse
        # vol ratio here as well would scale by volatility twice.
        risk_amount = portfolio_value * self._risk_per_trade * confidence * kelly
        lots = risk_amount / (stop * self._contract_size)

        # A very tight stop would otherwise ask for an unbounded position.
        max_lots = portfolio_value * self._max_leverage / lot_value
        return self._round_to_lot_step(min(lots, max_lots))

    def _stop_distance(
        self, stop_distance: float | None, current_atr: float | None,
    ) -> float | None:
        """The stop the size is measured against, or None if none is known."""
        if stop_distance is not None and stop_distance > 0:
            return stop_distance
        if current_atr is not None and current_atr > 0:
            return self._atr_stop_mult * current_atr
        return None

    def _notional_size(
        self,
        confidence: float,
        kelly: float,
        portfolio_value: float,
        current_atr: float | None,
        historical_atr: float | None,
        lot_value: float,
    ) -> float:
        """Fallback for when no stop distance is known.

        Caps notional at ``max_position_pct`` of equity. This is the old
        behaviour, kept only because inventing a stop distance in order to
        size against it would be worse than sizing conservatively.
        """
        vol_scalar = self._volatility_adjustment(current_atr, historical_atr)
        max_value = portfolio_value * self._max_pct
        capped_value = min(confidence * kelly * vol_scalar * max_value, max_value)
        return self._round_to_lot_step(capped_value / lot_value)

    def _compute_kelly(self) -> float:
        """Compute half-Kelly criterion fraction."""
        total_trades = self._trade_wins + self._trade_losses

        if total_trades < self._min_kelly_trades:
            # Not enough data: use a conservative default
            return self._kelly_frac * 0.5  # Quarter Kelly until sufficient data

        win_rate = self._trade_wins / total_trades
        if self._avg_loss <= 0 or win_rate <= 0:
            return self._kelly_frac * 0.5

        # Kelly: f = (p * b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss
        b = self._avg_win / self._avg_loss
        q = 1.0 - win_rate
        kelly_full = (win_rate * b - q) / b

        # Apply fraction (half Kelly by default)
        kelly = kelly_full * self._kelly_frac

        # Clamp to [0, 1]
        return float(np.clip(kelly, 0.0, 1.0))

    @staticmethod
    def _volatility_adjustment(
        current_atr: float | None, historical_atr: float | None
    ) -> float:
        """Inverse volatility scaling: higher vol → smaller position."""
        if current_atr is None or historical_atr is None:
            return 1.0
        if current_atr <= 0 or historical_atr <= 0:
            return 1.0

        # Ratio: if current vol is 2x historical, scale position to 50%
        ratio = historical_atr / current_atr
        return float(np.clip(ratio, 0.25, 2.0))

    def _round_to_lot_step(self, lots: float) -> float:
        """Round to the nearest valid lot step."""
        if lots < self._min_lot:
            return 0.0
        return round(lots / self._min_lot) * self._min_lot
