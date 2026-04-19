"""BacktestEngine — event-driven bar-by-bar backtesting with full pipeline.

Simulates trading bar-by-bar using:
- FeaturePipeline for feature extraction
- A strategy (signal function) that receives features and returns actions
- RiskManager for all 10 risk gates
- Realistic execution with spread/slippage simulation

Supports:
- Walk-forward validation (train/test splits)
- Multiple strategies/signals
- Commission and slippage modeling
- ATR-based stop loss / take profit
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import numpy as np
import pandas as pd

from apexfx.backtest.result import BacktestResult, Trade
from apexfx.config.schema import AppConfig, RiskConfig
from apexfx.features.pipeline import FeaturePipeline
from apexfx.risk.risk_manager import MarketState, RiskManager
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class Strategy(Protocol):
    """Protocol for backtest strategies."""

    def on_bar(self, features: pd.Series, bar: pd.Series) -> float:
        """Return action in [-1, 1] given current features and bar data.

        Returns:
            action: -1.0 = max short, 0.0 = neutral, +1.0 = max long
        """
        ...


@dataclass
class OpenPosition:
    """Tracks an open position during backtesting."""
    symbol: str
    direction: int
    entry_price: float
    entry_time: datetime
    volume: float
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_bar_idx: int = 0
    notional: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_equity: float = 100_000.0
    commission_per_lot: float = 7.0  # $7 per round-trip lot
    slippage_pips: float = 0.5  # avg slippage in pips
    pip_value: float = 0.0001  # for EURUSD
    contract_size: int = 100_000
    spread_pips: float = 1.0  # simulated spread
    atr_stop_mult: float = 2.0  # stop loss = ATR * multiplier
    atr_tp_mult: float = 3.0  # take profit = ATR * multiplier
    use_trailing_stop: bool = True
    warmup_bars: int = 300  # bars to skip (feature warmup)
    symbol: str = "EURUSD"
    disable_risk: bool = False  # bypass risk manager (useful for strategy testing)
    default_volume: float = 0.1  # default lot size when risk manager is bypassed


class BacktestEngine:
    """Event-driven backtesting engine.

    Usage:
        engine = BacktestEngine(bars_df, strategy, config=BacktestConfig())
        result = engine.run()
        print(result.summary())
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        strategy: Strategy | Callable,
        config: BacktestConfig | None = None,
        pipeline: FeaturePipeline | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self._bars = bars.copy().reset_index(drop=True)
        self._strategy = strategy
        self._config = config or BacktestConfig()
        self._pipeline = pipeline or FeaturePipeline(normalize=True)

        # Risk manager
        if risk_config is None:
            app_cfg = AppConfig()
            risk_config = app_cfg.risk
        self._risk_manager = RiskManager(
            risk_config,
            initial_balance=self._config.initial_equity,
        )

        # State
        self._equity = self._config.initial_equity
        self._position: OpenPosition | None = None
        self._result = BacktestResult(initial_equity=self._config.initial_equity)

        # Precompute ATR
        self._atr = self._compute_atr(14)

    def run(self) -> BacktestResult:
        """Run the backtest. Returns BacktestResult with all metrics."""
        logger.info(
            "Starting backtest",
            bars=len(self._bars),
            warmup=self._config.warmup_bars,
            equity=self._config.initial_equity,
        )

        # 1. Compute features for all bars
        features_df = self._pipeline.compute(self._bars)

        # 2. Bar-by-bar simulation
        n = len(self._bars)
        prev_day = None

        for i in range(self._config.warmup_bars, n):
            bar = self._bars.iloc[i]
            features = features_df.iloc[i]
            timestamp = self._get_timestamp(bar, i)

            # Daily return tracking
            current_day = timestamp.date() if hasattr(timestamp, "date") else None
            if prev_day is not None and current_day != prev_day:
                daily_ret = (self._equity - self._day_start_equity) / self._day_start_equity
                self._risk_manager.record_daily_return(daily_ret)
                self._day_start_equity = self._equity
            if prev_day is None or current_day != prev_day:
                self._day_start_equity = self._equity
            prev_day = current_day

            # Update risk manager portfolio value
            self._risk_manager.update_portfolio(self._equity)

            # Update regime from features
            if "regime_label" in features.index:
                regime = features["regime_label"]
                if isinstance(regime, str):
                    self._risk_manager.set_regime(regime)

            # 2a. Check stop loss / take profit on open position
            if self._position is not None:
                self._check_exit_conditions(bar, i, timestamp)

            # 2b. Get signal from strategy
            action = self._call_strategy(features, bar)

            # 2c. Determine if we need to act
            should_trade = self._should_trade(action)

            if should_trade:
                atr_val = self._atr[i] if i < len(self._atr) else None

                if self._config.disable_risk:
                    # Bypass risk manager — execute directly
                    self._result.record_risk_decision(True)
                    self._execute_signal(
                        action=action,
                        volume=self._config.default_volume,
                        bar=bar,
                        bar_idx=i,
                        timestamp=timestamp,
                        atr=atr_val,
                    )
                else:
                    # 2d. Run through risk manager
                    market_state = MarketState(
                        current_price=float(bar["close"]),
                        current_spread=self._config.spread_pips * self._config.pip_value,
                        current_atr=atr_val,
                        historical_atr=np.nanmean(self._atr[max(0, i - 100): i]) if i > 0 else atr_val,
                        spread_limit=self._config.spread_pips * self._config.pip_value * 3,
                    )

                    current_pos = 0.0
                    if self._position is not None:
                        current_pos = self._position.direction * self._position.volume

                    risk_decision = self._risk_manager.evaluate_action(
                        action=action,
                        market_state=market_state,
                        uncertainty_score=None,
                        current_position=current_pos,
                    )

                    self._result.record_risk_decision(
                        risk_decision.approved,
                        risk_decision.reason if not risk_decision.approved else "",
                    )

                    if risk_decision.approved and risk_decision.position_size > 0:
                        self._execute_signal(
                            action=risk_decision.adjusted_action,
                            volume=risk_decision.position_size,
                            bar=bar,
                            bar_idx=i,
                            timestamp=timestamp,
                            atr=atr_val,
                        )

            # 2e. Update equity with unrealized P&L
            current_equity = self._compute_equity(bar)
            self._result.record_equity(timestamp, current_equity)

            # 2f. Record exposure
            exposure = 0.0
            if self._position is not None:
                notional = self._position.volume * self._config.contract_size * float(bar["close"])
                exposure = notional / current_equity if current_equity > 0 else 0
            self._result.record_exposure(timestamp, exposure)

        # 3. Close any remaining position at the end
        if self._position is not None:
            last_bar = self._bars.iloc[-1]
            last_ts = self._get_timestamp(last_bar, len(self._bars) - 1)
            self._close_position(last_bar, len(self._bars) - 1, last_ts, "backtest_end")

        # 4. Compute metrics
        self._result.compute_metrics()

        logger.info(
            "Backtest complete",
            trades=len(self._result.trades),
            final_equity=round(self._equity, 2),
            sharpe=round(self._result.metrics.get("sharpe_ratio", 0), 2),
        )

        return self._result

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _call_strategy(self, features: pd.Series, bar: pd.Series) -> float:
        """Call the strategy to get an action."""
        try:
            if callable(self._strategy) and not hasattr(self._strategy, "on_bar"):
                action = self._strategy(features, bar)
            else:
                action = self._strategy.on_bar(features, bar)
            return float(np.clip(action, -1.0, 1.0))
        except Exception as e:
            logger.debug("Strategy error", error=str(e))
            return 0.0

    def _should_trade(self, action: float) -> bool:
        """Determine if we need to act on this signal."""
        direction = np.sign(action)

        # If no position and signal is neutral, skip
        if self._position is None and abs(action) < 0.05:
            return False

        # If we have a position and signal reverses, we should trade
        if self._position is not None:
            if direction != 0 and direction != self._position.direction:
                return True
            # If signal goes neutral, close
            if abs(action) < 0.05:
                return True

        # Open new position
        if self._position is None and abs(action) >= 0.05:
            return True

        return False

    def _execute_signal(
        self,
        action: float,
        volume: float,
        bar: pd.Series,
        bar_idx: int,
        timestamp: datetime,
        atr: float | None,
    ) -> None:
        """Execute a trading signal — open, close, or reverse position."""
        direction = int(np.sign(action))

        # Close existing position if direction changes
        if self._position is not None:
            if direction != self._position.direction or direction == 0:
                self._close_position(bar, bar_idx, timestamp, "signal")
                if direction == 0:
                    return

        # Open new position
        if self._position is None and direction != 0:
            entry_price = float(bar["close"])
            # Apply slippage
            slip = self._config.slippage_pips * self._config.pip_value
            if direction > 0:
                entry_price += slip  # Buy higher
            else:
                entry_price -= slip  # Sell lower

            # Compute stop loss and take profit
            stop_loss = None
            take_profit = None
            if atr is not None and atr > 0:
                stop_distance = atr * self._config.atr_stop_mult
                tp_distance = atr * self._config.atr_tp_mult
                if direction > 0:
                    stop_loss = entry_price - stop_distance
                    take_profit = entry_price + tp_distance
                else:
                    stop_loss = entry_price + stop_distance
                    take_profit = entry_price - tp_distance

            notional = volume * self._config.contract_size * entry_price

            self._position = OpenPosition(
                symbol=self._config.symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=timestamp,
                volume=volume,
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_bar_idx=bar_idx,
                notional=notional,
            )

            # Deduct commission
            commission = volume * self._config.commission_per_lot
            self._equity -= commission

    def _close_position(
        self,
        bar: pd.Series,
        bar_idx: int,
        timestamp: datetime,
        reason: str,
    ) -> None:
        """Close the current position and record the trade."""
        if self._position is None:
            return

        exit_price = float(bar["close"])
        # Apply slippage
        slip = self._config.slippage_pips * self._config.pip_value
        if self._position.direction > 0:
            exit_price -= slip  # Sell lower
        else:
            exit_price += slip  # Cover higher

        # Compute P&L
        price_diff = exit_price - self._position.entry_price
        pnl = price_diff * self._position.direction * self._position.volume * self._config.contract_size

        # Deduct close commission
        commission = self._position.volume * self._config.commission_per_lot
        pnl -= commission

        # Update equity
        self._equity += pnl

        # Compute pnl_pct relative to equity at entry
        entry_equity = self._equity - pnl  # Approximate equity at entry
        pnl_pct = pnl / entry_equity if entry_equity > 0 else 0.0

        trade = Trade(
            entry_time=self._position.entry_time,
            exit_time=timestamp,
            symbol=self._position.symbol,
            direction=self._position.direction,
            entry_price=self._position.entry_price,
            exit_price=exit_price,
            volume=self._position.volume,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission * 2,  # round-trip
            bars_held=bar_idx - self._position.entry_bar_idx,
            exit_reason=reason,
        )

        self._result.record_trade(trade)
        self._risk_manager.record_trade(pnl, pnl_pct)

        logger.debug(
            "Trade closed",
            direction="LONG" if trade.direction > 0 else "SHORT",
            pnl=round(pnl, 2),
            bars_held=trade.bars_held,
            reason=reason,
        )

        self._position = None

    def _check_exit_conditions(
        self,
        bar: pd.Series,
        bar_idx: int,
        timestamp: datetime,
    ) -> None:
        """Check stop loss, take profit, and trailing stop."""
        if self._position is None:
            return

        high = float(bar["high"])
        low = float(bar["low"])

        # Stop loss hit?
        if self._position.stop_loss is not None:
            if self._position.direction > 0 and low <= self._position.stop_loss:
                self._close_position(bar, bar_idx, timestamp, "stop_loss")
                return
            if self._position.direction < 0 and high >= self._position.stop_loss:
                self._close_position(bar, bar_idx, timestamp, "stop_loss")
                return

        # Take profit hit?
        if self._position.take_profit is not None:
            if self._position.direction > 0 and high >= self._position.take_profit:
                self._close_position(bar, bar_idx, timestamp, "take_profit")
                return
            if self._position.direction < 0 and low <= self._position.take_profit:
                self._close_position(bar, bar_idx, timestamp, "take_profit")
                return

        # Trailing stop: move stop loss in profit direction
        if self._config.use_trailing_stop and self._position.stop_loss is not None:
            if self._position.direction > 0:
                # Long: trail stop up
                atr_val = self._atr[bar_idx] if bar_idx < len(self._atr) else None
                if atr_val and atr_val > 0:
                    new_stop = high - atr_val * self._config.atr_stop_mult
                    if new_stop > self._position.stop_loss:
                        self._position.stop_loss = new_stop
            else:
                # Short: trail stop down
                atr_val = self._atr[bar_idx] if bar_idx < len(self._atr) else None
                if atr_val and atr_val > 0:
                    new_stop = low + atr_val * self._config.atr_stop_mult
                    if new_stop < self._position.stop_loss:
                        self._position.stop_loss = new_stop

    def _compute_equity(self, bar: pd.Series) -> float:
        """Compute current equity including unrealized P&L."""
        if self._position is None:
            return self._equity

        price_diff = float(bar["close"]) - self._position.entry_price
        unrealized = (
            price_diff * self._position.direction
            * self._position.volume * self._config.contract_size
        )
        return self._equity + unrealized

    def _compute_atr(self, period: int = 14) -> np.ndarray:
        """Precompute ATR for all bars."""
        high = self._bars["high"].values
        low = self._bars["low"].values
        close = self._bars["close"].values

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1)),
            ),
        )
        tr[0] = high[0] - low[0]

        atr = np.full_like(tr, np.nan)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return atr

    @staticmethod
    def _get_timestamp(bar: pd.Series, idx: int) -> datetime:
        """Extract timestamp from bar."""
        if "time" in bar.index:
            ts = bar["time"]
            if isinstance(ts, pd.Timestamp):
                return ts.to_pydatetime()
            if isinstance(ts, datetime):
                return ts
        return datetime(2025, 1, 1, tzinfo=UTC)


# ------------------------------------------------------------------
# Walk-Forward Validation
# ------------------------------------------------------------------

def walk_forward_backtest(
    bars: pd.DataFrame,
    strategy_factory: Callable[..., Strategy | Callable],
    train_bars: int = 5000,
    test_bars: int = 2000,
    step_bars: int = 2000,
    config: BacktestConfig | None = None,
    pipeline: FeaturePipeline | None = None,
) -> list[BacktestResult]:
    """Run walk-forward validation: train → test → slide window.

    Args:
        bars: Full OHLCV history.
        strategy_factory: Callable that returns a Strategy given training data.
        train_bars: Number of bars for training window.
        test_bars: Number of bars for testing window.
        step_bars: Number of bars to slide forward each fold.
        config: Backtest configuration.
        pipeline: Feature pipeline (shared across folds).

    Returns:
        List of BacktestResult, one per fold.
    """
    results: list[BacktestResult] = []
    n = len(bars)
    fold = 0

    start = 0
    while start + train_bars + test_bars <= n:
        fold += 1
        train_end = start + train_bars
        test_end = min(train_end + test_bars, n)

        train_data = bars.iloc[start:train_end]
        test_data = bars.iloc[train_end:test_end]

        logger.info(
            f"Walk-forward fold {fold}",
            train=f"{start}:{train_end}",
            test=f"{train_end}:{test_end}",
        )

        # Create strategy from training data
        strategy = strategy_factory(train_data)

        # Run backtest on test data
        cfg = config or BacktestConfig()
        cfg.warmup_bars = min(cfg.warmup_bars, len(test_data) // 2)

        engine = BacktestEngine(
            bars=test_data,
            strategy=strategy,
            config=cfg,
            pipeline=pipeline,
        )
        result = engine.run()
        results.append(result)

        start += step_bars

    logger.info(f"Walk-forward complete: {len(results)} folds")
    return results
