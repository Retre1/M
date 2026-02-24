"""Paper Trading Loop — production-faithful simulation without real orders.

The paper trader is the mandatory bridge between backtesting and live trading.
It replays historical bars bar-by-bar through the **exact same pipeline** as
:class:`~apexfx.live.trading_loop.LiveTradingLoop`, so any bug that would
surface in production will surface here first.

Differences vs. live trading
-----------------------------
* No MT5 connection — data comes from :class:`~apexfx.data.data_store.DataStore`
  or a DataFrame passed directly.
* Order fills are *simulated* using configurable spread + slippage.
* All risk management, feature computation, and model inference are identical
  to the live loop — this is deliberate.

Typical usage
-------------
1. Run ``scripts/paper_trade.py`` for at least 3 months on recent unseen data.
2. Only proceed to live trading if Sharpe > 1.0, max DD < 10%.

OOS evaluation mode
-------------------
Pass ``oos_data`` to replay the sacred OOS set as the final pre-live check::

    guard = OOSGuard(full_data, oos_fraction=0.2)
    _, _ = guard.split()
    oos = guard.unlock_oos(reason="final pre-live paper trade check")
    loop = PaperTradingLoop(config, model_path="models/best/final_model", oos_data=oos)
    report = loop.run()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from apexfx.config.schema import AppConfig
from apexfx.env.reward import DifferentialSharpeReward
from apexfx.features.pipeline import FeaturePipeline
from apexfx.live.signal_generator import SignalGenerator
from apexfx.live.state_manager import StateManager
from apexfx.risk.risk_manager import MarketState, RiskManager
from apexfx.utils.logging import get_logger
from apexfx.utils.math_utils import atr
from apexfx.utils.metrics import compute_all_metrics

logger = get_logger(__name__)

_INITIAL_BALANCE = 100_000.0


@dataclass
class PaperTrade:
    """Record of a single paper trade."""
    bar_idx: int
    open_time: str
    close_time: str | None
    direction: int           # +1 long, -1 short
    open_price: float
    close_price: float | None
    volume: float
    pnl: float | None        # None while still open
    spread_cost: float
    slippage_cost: float


@dataclass
class PaperTradingReport:
    """Full performance report from a paper trading session."""
    n_bars: int
    initial_balance: float
    final_balance: float
    total_return_pct: float
    max_drawdown_pct: float
    metrics: dict[str, float]
    equity_curve: list[float]
    trades: list[PaperTrade]
    risk_vetoes: int          # how many signals the risk manager blocked
    spread_slippage_total: float  # total transaction costs

    def print_summary(self) -> None:
        sep = "=" * 60
        logger.info(sep)
        logger.info("PAPER TRADING RESULTS")
        logger.info(sep)
        logger.info(f"  Bars replayed:       {self.n_bars}")
        logger.info(f"  Initial balance:     ${self.initial_balance:,.2f}")
        logger.info(f"  Final balance:       ${self.final_balance:,.2f}")
        logger.info(f"  Total return:        {self.total_return_pct:+.2f}%")
        logger.info(f"  Max drawdown:        {self.max_drawdown_pct:.2f}%")
        logger.info(f"  Sharpe ratio:        {self.metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"  Sortino ratio:       {self.metrics.get('sortino_ratio', 0):.4f}")
        logger.info(f"  Calmar ratio:        {self.metrics.get('calmar_ratio', 0):.4f}")
        logger.info(f"  Win rate:            {self.metrics.get('win_rate', 0):.2%}")
        logger.info(f"  Profit factor:       {self.metrics.get('profit_factor', 0):.4f}")
        logger.info(f"  Total trades:        {len(self.trades)}")
        logger.info(f"  Risk vetoes:         {self.risk_vetoes}")
        logger.info(f"  Transaction costs:   ${self.spread_slippage_total:,.2f}")
        logger.info(sep)

        # Gate: warn if results are below minimum bar for going live
        sharpe = self.metrics.get("sharpe_ratio", 0)
        if sharpe < 1.0:
            logger.warning(
                "Sharpe ratio below 1.0 — do NOT proceed to live trading",
                sharpe=round(sharpe, 4),
            )
        if self.max_drawdown_pct > 10.0:
            logger.warning(
                "Max drawdown exceeds 10% — do NOT proceed to live trading",
                max_dd=round(self.max_drawdown_pct, 2),
            )
        if sharpe >= 1.0 and self.max_drawdown_pct <= 10.0:
            logger.info(
                "Paper trading gates passed — candidate for live deployment",
                sharpe=round(sharpe, 4),
                max_dd=round(self.max_drawdown_pct, 2),
            )

    def save(self, path: str | Path) -> None:
        """Persist the report as JSON for later analysis / dashboard display."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_bars": self.n_bars,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "risk_vetoes": self.risk_vetoes,
            "spread_slippage_total": self.spread_slippage_total,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "equity_curve": [float(e) for e in self.equity_curve],
            "trades": [
                {
                    "bar_idx": t.bar_idx,
                    "open_time": t.open_time,
                    "close_time": t.close_time,
                    "direction": t.direction,
                    "open_price": t.open_price,
                    "close_price": t.close_price,
                    "volume": t.volume,
                    "pnl": t.pnl,
                    "spread_cost": t.spread_cost,
                    "slippage_cost": t.slippage_cost,
                }
                for t in self.trades
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Paper trading report saved", path=str(path))


class PaperTradingLoop:
    """Replay historical bars through the live pipeline without real orders.

    Parameters
    ----------
    config:
        Full application config.
    model_path:
        Path to a trained SB3 model checkpoint (no ``.zip`` extension needed).
    symbol:
        Trading symbol — used for spread/slippage lookups from config.
    data:
        Pre-loaded DataFrame of OHLCV bars.  When None, data must be loaded
        externally and passed to :meth:`run`.
    oos_data:
        Optional separate OOS DataFrame. When provided, ``run()`` replays this
        instead of ``data``.
    initial_balance:
        Starting virtual equity (default $100,000).
    extra_slippage_pips:
        Additional slippage beyond the symbol's natural spread (default 0.5 pips).
    """

    def __init__(
        self,
        config: AppConfig,
        model_path: str | Path,
        symbol: str = "EURUSD",
        data: pd.DataFrame | None = None,
        oos_data: pd.DataFrame | None = None,
        initial_balance: float = _INITIAL_BALANCE,
        extra_slippage_pips: float = 0.5,
    ) -> None:
        self._config = config
        self._symbol = symbol
        self._data = data
        self._oos_data = oos_data
        self._initial_balance = initial_balance

        sym_cfg = config.symbols.symbols.get(symbol)
        if sym_cfg is None:
            raise ValueError(f"Symbol '{symbol}' not in config.symbols")
        self._sym_cfg = sym_cfg

        # Simulated transaction costs in price units
        self._spread = sym_cfg.spread_limit_pips * sym_cfg.pip_value
        self._slippage = extra_slippage_pips * sym_cfg.pip_value

        self._feature_pipeline = FeaturePipeline()
        self._signal_gen = SignalGenerator(str(model_path), device="cpu")
        self._risk_manager = RiskManager(config.risk)
        self._state = StateManager()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, data: pd.DataFrame | None = None) -> PaperTradingReport:
        """Replay bars and return a full performance report.

        Parameters
        ----------
        data:
            Override bars source (takes priority over constructor args).
        """
        bars_df = self._resolve_data(data)
        if bars_df is None or bars_df.empty:
            raise ValueError(
                "No data provided. Pass data= to run() or the constructor."
            )

        logger.info(
            "Paper trading started",
            symbol=self._symbol,
            n_bars=len(bars_df),
            initial_balance=self._initial_balance,
        )

        # --- Compute features for the entire session upfront ---
        features_df = self._feature_pipeline.compute(bars_df)
        n_features = min(self._feature_pipeline.n_features, 30)

        equity = self._initial_balance
        balance = self._initial_balance
        equity_curve: list[float] = [equity]
        step_returns: list[float] = []
        trades: list[PaperTrade] = []
        risk_vetoes = 0
        spread_slippage_total = 0.0

        # Position state
        position_direction = 0
        position_volume = 0.0
        position_entry_price = 0.0
        position_open_bar = 0
        position_open_time = ""

        obs_builder = self._signal_gen._obs_builder
        self._risk_manager.update_portfolio(equity)

        lookback = self._config.data.feature_window

        for bar_idx in range(lookback, len(features_df)):
            bar = bars_df.iloc[bar_idx]
            current_price = float(bar["close"])

            # --- Build observation ---
            try:
                unrealized_pnl = 0.0
                if position_direction != 0:
                    unrealized_pnl = (
                        (current_price - position_entry_price)
                        * position_volume
                        * self._sym_cfg.contract_size
                        * position_direction
                    )
                time_in_pos = bar_idx - position_open_bar if position_direction != 0 else 0

                obs = obs_builder.build(
                    features=features_df,
                    current_idx=bar_idx,
                    position=float(position_volume * position_direction),
                    unrealized_pnl=unrealized_pnl,
                    time_in_position=float(time_in_pos),
                    portfolio_value=equity,
                )
            except Exception as e:
                logger.debug("Observation build failed", bar_idx=bar_idx, error=str(e))
                equity_curve.append(equity)
                continue

            # --- Signal generation ---
            try:
                signal = self._signal_gen.generate(obs)
            except Exception as e:
                logger.debug("Signal generation failed", bar_idx=bar_idx, error=str(e))
                equity_curve.append(equity)
                continue

            # --- ATR for risk sizing ---
            atr_window = features_df.iloc[max(0, bar_idx - 14): bar_idx + 1]
            current_atr = None
            if len(atr_window) >= 14 and all(c in atr_window.columns for c in ("high", "low", "close")):
                atr_vals = atr(
                    atr_window["high"].values,
                    atr_window["low"].values,
                    atr_window["close"].values,
                    period=14,
                )
                if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]):
                    current_atr = float(atr_vals[-1])

            market_state = MarketState(
                current_price=current_price,
                current_spread=self._spread,
                current_atr=current_atr,
                spread_limit=self._spread,
            )

            self._risk_manager.update_portfolio(equity)
            risk_decision = self._risk_manager.evaluate_action(signal.action, market_state)

            if not risk_decision.approved and risk_decision.adjusted_action == 0.0:
                risk_vetoes += 1
                # Still update equity with unrealized
                equity = balance + (unrealized_pnl if position_direction != 0 else 0.0)
                step_return = (equity - equity_curve[-1]) / equity_curve[-1] if equity_curve[-1] > 0 else 0.0
                step_returns.append(step_return)
                equity_curve.append(equity)
                continue

            target_action = risk_decision.adjusted_action

            # --- Simulate execution ---
            if position_direction == 0 and abs(target_action) > 0.01:
                # Open new position
                direction = 1 if target_action > 0 else -1
                volume = self._compute_volume(risk_decision, equity)

                # Fill price includes spread + slippage
                if direction > 0:
                    fill_price = current_price + self._spread + self._slippage
                else:
                    fill_price = current_price - self._spread - self._slippage

                cost = (self._spread + self._slippage) * volume * self._sym_cfg.contract_size
                balance -= cost
                spread_slippage_total += cost

                position_direction = direction
                position_volume = volume
                position_entry_price = fill_price
                position_open_bar = bar_idx
                position_open_time = str(bar.get("time", bar_idx))

                trades.append(PaperTrade(
                    bar_idx=bar_idx,
                    open_time=position_open_time,
                    close_time=None,
                    direction=direction,
                    open_price=fill_price,
                    close_price=None,
                    volume=volume,
                    pnl=None,
                    spread_cost=self._spread * volume * self._sym_cfg.contract_size,
                    slippage_cost=self._slippage * volume * self._sym_cfg.contract_size,
                ))

            elif position_direction != 0 and abs(target_action) < 0.01:
                # Close position
                if position_direction > 0:
                    fill_price = current_price - self._spread - self._slippage
                else:
                    fill_price = current_price + self._spread + self._slippage

                pnl = (
                    (fill_price - position_entry_price)
                    * position_volume
                    * self._sym_cfg.contract_size
                    * position_direction
                )
                balance += pnl

                close_cost = (self._spread + self._slippage) * position_volume * self._sym_cfg.contract_size
                balance -= close_cost
                spread_slippage_total += close_cost

                self._risk_manager.record_trade(pnl, trade_return=pnl / equity if equity > 0 else 0.0)

                if trades and trades[-1].close_time is None:
                    trades[-1].close_price = fill_price
                    trades[-1].close_time = str(bar.get("time", bar_idx))
                    trades[-1].pnl = pnl

                position_direction = 0
                position_volume = 0.0
                position_entry_price = 0.0

            # Update equity
            unrealized_pnl = 0.0
            if position_direction != 0:
                unrealized_pnl = (
                    (current_price - position_entry_price)
                    * position_volume
                    * self._sym_cfg.contract_size
                    * position_direction
                )
            equity = balance + unrealized_pnl

            step_return = (equity - equity_curve[-1]) / equity_curve[-1] if equity_curve[-1] > 0 else 0.0
            step_returns.append(step_return)
            equity_curve.append(equity)

        # --- Force-close any open position at session end ---
        if position_direction != 0 and len(features_df) > 0:
            last_price = float(features_df.iloc[-1]["close"])
            fill_price = (
                last_price - self._spread
                if position_direction > 0
                else last_price + self._spread
            )
            pnl = (
                (fill_price - position_entry_price)
                * position_volume
                * self._sym_cfg.contract_size
                * position_direction
            )
            balance += pnl
            if trades and trades[-1].close_time is None:
                trades[-1].close_price = fill_price
                trades[-1].close_time = "session_end"
                trades[-1].pnl = pnl
            equity = balance

        # --- Performance metrics ---
        returns_arr = np.array(step_returns) if step_returns else np.array([0.0])
        metrics = compute_all_metrics(returns_arr)

        eq_arr = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq_arr)
        dd_series = (peaks - eq_arr) / peaks * 100
        max_dd_pct = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0

        total_return_pct = (equity - self._initial_balance) / self._initial_balance * 100

        report = PaperTradingReport(
            n_bars=len(features_df) - lookback,
            initial_balance=self._initial_balance,
            final_balance=float(equity),
            total_return_pct=float(total_return_pct),
            max_drawdown_pct=float(max_dd_pct),
            metrics=metrics,
            equity_curve=[float(e) for e in equity_curve],
            trades=trades,
            risk_vetoes=risk_vetoes,
            spread_slippage_total=float(spread_slippage_total),
        )

        report.print_summary()
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_data(self, override: pd.DataFrame | None) -> pd.DataFrame | None:
        if override is not None:
            return override
        if self._oos_data is not None:
            logger.info("Using OOS dataset for paper trading")
            return self._oos_data
        return self._data

    def _compute_volume(self, risk_decision, equity: float) -> float:
        """Derive lot size from risk decision or fall back to minimum."""
        lot_size = getattr(risk_decision, "lot_size", None)
        if lot_size and lot_size > 0:
            return float(lot_size)
        # Fallback: 1% risk at 0.01 pip stop → minimal position
        return self._sym_cfg.lot_step
