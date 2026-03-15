"""BacktestResult — comprehensive metrics and trade log from a backtest run.

Computes institutional-grade performance metrics:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis (max DD, duration, recovery)
- Trade statistics (win rate, profit factor, avg trade)
- Monthly/yearly returns breakdown
- Exposure analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

ANNUAL_TRADING_DAYS = 252
RISK_FREE_RATE = 0.05  # 5% annualized for 2024-2026 era


@dataclass
class Trade:
    """Record of a single completed trade."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    volume: float  # lots
    pnl: float  # absolute P&L in account currency
    pnl_pct: float  # return on equity at entry
    commission: float = 0.0
    slippage: float = 0.0
    bars_held: int = 0
    exit_reason: str = ""  # "signal", "stop_loss", "take_profit", "risk_close"


@dataclass
class DrawdownInfo:
    """Drawdown analysis."""
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    recovery_time_days: float = 0.0
    drawdown_series: list[float] = field(default_factory=list)


class BacktestResult:
    """Comprehensive backtest results with metrics, trades, and equity curve.

    Usage:
        result = BacktestResult(initial_equity=100000)
        result.record_equity(timestamp, equity)
        result.record_trade(trade)
        result.compute_metrics()
        print(result.summary())
    """

    def __init__(self, initial_equity: float = 100_000.0) -> None:
        self.initial_equity = initial_equity

        # Time series
        self.equity_curve: list[tuple[datetime, float]] = []
        self.returns_series: list[float] = []
        self.exposure_series: list[tuple[datetime, float]] = []  # (time, exposure_pct)

        # Trade log
        self.trades: list[Trade] = []

        # Risk manager stats
        self.risk_rejections: int = 0
        self.risk_approvals: int = 0
        self.risk_rejection_reasons: dict[str, int] = {}

        # Computed metrics (populated by compute_metrics)
        self.metrics: dict[str, float] = {}
        self.drawdown_info = DrawdownInfo()
        self.monthly_returns: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_equity(self, timestamp: datetime, equity: float) -> None:
        """Record a new equity data point."""
        self.equity_curve.append((timestamp, equity))
        if len(self.equity_curve) >= 2:
            prev_eq = self.equity_curve[-2][1]
            if prev_eq > 0:
                self.returns_series.append((equity - prev_eq) / prev_eq)
            else:
                self.returns_series.append(0.0)

    def record_trade(self, trade: Trade) -> None:
        """Record a completed trade."""
        self.trades.append(trade)

    def record_exposure(self, timestamp: datetime, exposure_pct: float) -> None:
        """Record current portfolio exposure."""
        self.exposure_series.append((timestamp, exposure_pct))

    def record_risk_decision(self, approved: bool, reason: str = "") -> None:
        """Record a risk manager decision."""
        if approved:
            self.risk_approvals += 1
        else:
            self.risk_rejections += 1
            self.risk_rejection_reasons[reason] = (
                self.risk_rejection_reasons.get(reason, 0) + 1
            )

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_metrics(self) -> dict[str, float]:
        """Compute all performance metrics from recorded data."""
        m: dict[str, float] = {}

        # Basic P&L
        final_equity = self.equity_curve[-1][1] if self.equity_curve else self.initial_equity
        m["initial_equity"] = self.initial_equity
        m["final_equity"] = final_equity
        m["net_pnl"] = final_equity - self.initial_equity
        m["total_return_pct"] = (final_equity / self.initial_equity - 1) * 100

        # Duration
        if len(self.equity_curve) >= 2:
            start = self.equity_curve[0][0]
            end = self.equity_curve[-1][0]
            m["duration_days"] = (end - start).total_seconds() / 86400
        else:
            m["duration_days"] = 0

        # Returns-based metrics
        returns = np.array(self.returns_series) if self.returns_series else np.array([0.0])

        m["total_bars"] = len(self.equity_curve)
        m["annual_return_pct"] = self._annualized_return(returns) * 100
        m["annual_volatility_pct"] = self._annualized_volatility(returns) * 100
        m["sharpe_ratio"] = self._sharpe_ratio(returns)
        m["sortino_ratio"] = self._sortino_ratio(returns)
        m["calmar_ratio"] = self._calmar_ratio(returns, final_equity)

        # Drawdown
        self._compute_drawdown()
        m["max_drawdown_pct"] = self.drawdown_info.max_drawdown_pct * 100
        m["max_drawdown_abs"] = self.drawdown_info.max_drawdown_abs
        m["max_drawdown_duration_days"] = self.drawdown_info.max_drawdown_duration_days
        m["avg_drawdown_pct"] = self.drawdown_info.avg_drawdown_pct * 100

        # Trade statistics
        self._compute_trade_stats(m)

        # Risk stats
        total_decisions = self.risk_approvals + self.risk_rejections
        m["risk_approvals"] = self.risk_approvals
        m["risk_rejections"] = self.risk_rejections
        m["risk_rejection_rate"] = (
            self.risk_rejections / total_decisions * 100 if total_decisions > 0 else 0
        )

        # Exposure
        if self.exposure_series:
            exposures = [e[1] for e in self.exposure_series]
            m["avg_exposure_pct"] = np.mean(exposures) * 100
            m["max_exposure_pct"] = np.max(exposures) * 100
            m["time_in_market_pct"] = (
                sum(1 for e in exposures if e > 0.001) / len(exposures) * 100
            )
        else:
            m["avg_exposure_pct"] = 0
            m["max_exposure_pct"] = 0
            m["time_in_market_pct"] = 0

        # Monthly returns
        self._compute_monthly_returns()

        self.metrics = m
        return m

    def _annualized_return(self, returns: np.ndarray) -> float:
        """Compute annualized return from daily returns."""
        if len(returns) < 2:
            return 0.0
        total = np.prod(1 + returns) - 1
        n_years = len(returns) / ANNUAL_TRADING_DAYS
        if n_years <= 0:
            return 0.0
        return (1 + total) ** (1 / n_years) - 1

    def _annualized_volatility(self, returns: np.ndarray) -> float:
        """Compute annualized volatility."""
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns, ddof=1) * np.sqrt(ANNUAL_TRADING_DAYS))

    def _sharpe_ratio(self, returns: np.ndarray) -> float:
        """Compute annualized Sharpe ratio."""
        ann_ret = self._annualized_return(returns)
        ann_vol = self._annualized_volatility(returns)
        if ann_vol < 1e-10:
            return 0.0
        return (ann_ret - RISK_FREE_RATE) / ann_vol

    def _sortino_ratio(self, returns: np.ndarray) -> float:
        """Compute Sortino ratio (downside deviation only)."""
        ann_ret = self._annualized_return(returns)
        downside = returns[returns < 0]
        if len(downside) < 2:
            return 0.0
        downside_vol = float(np.std(downside, ddof=1) * np.sqrt(ANNUAL_TRADING_DAYS))
        if downside_vol < 1e-10:
            return 0.0
        return (ann_ret - RISK_FREE_RATE) / downside_vol

    def _calmar_ratio(self, returns: np.ndarray, final_equity: float) -> float:
        """Compute Calmar ratio (annual return / max drawdown)."""
        ann_ret = self._annualized_return(returns)
        if self.drawdown_info.max_drawdown_pct < 1e-10:
            return 0.0
        return ann_ret / self.drawdown_info.max_drawdown_pct

    def _compute_drawdown(self) -> None:
        """Compute drawdown series and statistics."""
        if not self.equity_curve:
            return

        equities = np.array([e[1] for e in self.equity_curve])
        peak = np.maximum.accumulate(equities)
        dd = (peak - equities) / peak
        dd = np.nan_to_num(dd, nan=0.0)

        self.drawdown_info.drawdown_series = dd.tolist()
        self.drawdown_info.max_drawdown_pct = float(np.max(dd))
        self.drawdown_info.max_drawdown_abs = float(np.max(peak - equities))
        self.drawdown_info.avg_drawdown_pct = float(np.mean(dd[dd > 0])) if np.any(dd > 0) else 0.0
        self.drawdown_info.current_drawdown_pct = float(dd[-1])

        # Max drawdown duration (bars in drawdown)
        in_dd = dd > 0
        if np.any(in_dd):
            max_duration = 0
            current_duration = 0
            for v in in_dd:
                if v:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            self.drawdown_info.max_drawdown_duration_days = max_duration

    def _compute_trade_stats(self, m: dict) -> None:
        """Compute trade-level statistics."""
        m["total_trades"] = len(self.trades)

        if not self.trades:
            for key in [
                "win_rate", "profit_factor", "avg_trade_pnl",
                "avg_winner", "avg_loser", "largest_winner", "largest_loser",
                "avg_bars_held", "long_trades", "short_trades",
                "expectancy", "avg_trade_return_pct",
            ]:
                m[key] = 0.0
            return

        pnls = np.array([t.pnl for t in self.trades])
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]

        m["win_rate"] = len(winners) / len(pnls) * 100 if len(pnls) > 0 else 0
        m["avg_trade_pnl"] = float(np.mean(pnls))
        m["avg_trade_return_pct"] = float(np.mean([t.pnl_pct for t in self.trades])) * 100
        m["avg_winner"] = float(np.mean(winners)) if len(winners) > 0 else 0
        m["avg_loser"] = float(np.mean(losers)) if len(losers) > 0 else 0
        m["largest_winner"] = float(np.max(pnls)) if len(pnls) > 0 else 0
        m["largest_loser"] = float(np.min(pnls)) if len(pnls) > 0 else 0
        m["avg_bars_held"] = float(np.mean([t.bars_held for t in self.trades]))
        m["long_trades"] = sum(1 for t in self.trades if t.direction > 0)
        m["short_trades"] = sum(1 for t in self.trades if t.direction < 0)

        # Profit factor
        gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0
        gross_loss = float(np.abs(np.sum(losers))) if len(losers) > 0 else 0
        m["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy = avg win * win_rate - avg loss * loss_rate
        wr = len(winners) / len(pnls) if len(pnls) > 0 else 0
        lr = 1 - wr
        avg_w = float(np.mean(winners)) if len(winners) > 0 else 0
        avg_l = float(np.abs(np.mean(losers))) if len(losers) > 0 else 0
        m["expectancy"] = wr * avg_w - lr * avg_l

    def _compute_monthly_returns(self) -> None:
        """Compute monthly returns table."""
        if len(self.equity_curve) < 2:
            return

        df = pd.DataFrame(self.equity_curve, columns=["time", "equity"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        df["return"] = df["equity"].pct_change()

        monthly = df["equity"].resample("ME").last()
        monthly_ret = monthly.pct_change().dropna()

        if len(monthly_ret) > 0:
            self.monthly_returns = monthly_ret.to_frame("return")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of backtest results."""
        if not self.metrics:
            self.compute_metrics()
        m = self.metrics

        lines = [
            "=" * 60,
            "         APEXFX QUANTUM — BACKTEST RESULTS",
            "=" * 60,
            "",
            f"  Period:           {m.get('duration_days', 0):.0f} days",
            f"  Initial Equity:   ${m['initial_equity']:,.0f}",
            f"  Final Equity:     ${m['final_equity']:,.0f}",
            f"  Net P&L:          ${m['net_pnl']:,.2f}",
            f"  Total Return:     {m['total_return_pct']:.2f}%",
            "",
            "  --- Risk-Adjusted Returns ---",
            f"  Annual Return:    {m['annual_return_pct']:.2f}%",
            f"  Annual Vol:       {m['annual_volatility_pct']:.2f}%",
            f"  Sharpe Ratio:     {m['sharpe_ratio']:.2f}",
            f"  Sortino Ratio:    {m['sortino_ratio']:.2f}",
            f"  Calmar Ratio:     {m['calmar_ratio']:.2f}",
            "",
            "  --- Drawdown ---",
            f"  Max Drawdown:     {m['max_drawdown_pct']:.2f}%",
            f"  Max DD Duration:  {m['max_drawdown_duration_days']} bars",
            f"  Avg Drawdown:     {m['avg_drawdown_pct']:.2f}%",
            "",
            "  --- Trades ---",
            f"  Total Trades:     {m['total_trades']:.0f}",
            f"  Win Rate:         {m['win_rate']:.1f}%",
            f"  Profit Factor:    {m['profit_factor']:.2f}",
            f"  Expectancy:       ${m['expectancy']:.2f}",
            f"  Avg Trade P&L:    ${m['avg_trade_pnl']:.2f}",
            f"  Avg Winner:       ${m['avg_winner']:.2f}",
            f"  Avg Loser:        ${m['avg_loser']:.2f}",
            f"  Avg Bars Held:    {m['avg_bars_held']:.1f}",
            f"  Long/Short:       {m['long_trades']:.0f} / {m['short_trades']:.0f}",
            "",
            "  --- Exposure ---",
            f"  Time in Market:   {m['time_in_market_pct']:.1f}%",
            f"  Avg Exposure:     {m['avg_exposure_pct']:.1f}%",
            "",
            "  --- Risk Manager ---",
            f"  Approvals:        {m['risk_approvals']:.0f}",
            f"  Rejections:       {m['risk_rejections']:.0f} ({m['risk_rejection_rate']:.1f}%)",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        df = pd.DataFrame(self.equity_curve, columns=["time", "equity"])
        df["time"] = pd.to_datetime(df["time"])
        if self.drawdown_info.drawdown_series:
            df["drawdown"] = self.drawdown_info.drawdown_series[: len(df)]
        if self.returns_series:
            df["return"] = [0.0] + self.returns_series[: len(df) - 1]
        return df

    def trades_dataframe(self) -> pd.DataFrame:
        """Return trade log as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "symbol": t.symbol,
                "direction": "LONG" if t.direction > 0 else "SHORT",
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "volume": t.volume,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct * 100,
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
            })
        return pd.DataFrame(rows)
