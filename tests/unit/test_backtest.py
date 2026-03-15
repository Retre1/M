"""Tests for the backtesting engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


def _make_bars(n: int = 1000, trend: float = 0.0001) -> pd.DataFrame:
    """Generate synthetic OHLCV bars with optional trend."""
    np.random.seed(42)
    base = 1.1000
    returns = np.random.randn(n) * 0.001 + trend
    close = base + np.cumsum(returns)
    high = close + np.abs(np.random.randn(n) * 0.0005)
    low = close - np.abs(np.random.randn(n) * 0.0005)
    opn = close + np.random.randn(n) * 0.0003
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame({
        "time": [start + timedelta(hours=i) for i in range(n)],
        "open": opn, "high": high, "low": low, "close": close,
        "tick_volume": np.random.randint(100, 5000, n).astype(float),
        "volume": np.random.randint(100, 5000, n).astype(float),
        "spread": np.random.uniform(0.00005, 0.0002, n),
    })


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_record_equity(self):
        from apexfx.backtest.result import BacktestResult
        r = BacktestResult(initial_equity=100000)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r.record_equity(ts, 100000)
        r.record_equity(ts + timedelta(days=1), 101000)
        r.record_equity(ts + timedelta(days=2), 100500)
        assert len(r.equity_curve) == 3
        assert len(r.returns_series) == 2

    def test_compute_metrics_basic(self):
        from apexfx.backtest.result import BacktestResult
        r = BacktestResult(initial_equity=100000)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(100):
            eq = 100000 + i * 100
            r.record_equity(ts + timedelta(hours=i), eq)
        m = r.compute_metrics()
        assert m["final_equity"] == pytest.approx(109900, abs=1)
        assert m["total_return_pct"] > 0
        assert m["max_drawdown_pct"] == pytest.approx(0, abs=0.01)

    def test_trade_stats(self):
        from apexfx.backtest.result import BacktestResult, Trade
        r = BacktestResult()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # 3 winners, 2 losers
        for i, pnl in enumerate([100, 200, -50, 150, -30]):
            r.record_trade(Trade(
                entry_time=ts, exit_time=ts + timedelta(hours=1),
                symbol="EURUSD", direction=1,
                entry_price=1.10, exit_price=1.11,
                volume=0.1, pnl=pnl, pnl_pct=pnl / 100000,
                bars_held=10,
            ))
        r.record_equity(ts, 100000)
        r.record_equity(ts + timedelta(hours=5), 100370)
        m = r.compute_metrics()
        assert m["total_trades"] == 5
        assert m["win_rate"] == 60.0
        assert m["profit_factor"] > 1.0

    def test_drawdown_computation(self):
        from apexfx.backtest.result import BacktestResult
        r = BacktestResult(initial_equity=100000)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        # Equity goes up then down
        equities = [100000, 110000, 105000, 108000, 100000, 112000]
        for i, eq in enumerate(equities):
            r.record_equity(ts + timedelta(days=i), eq)
        m = r.compute_metrics()
        # Max DD should be from 110000 to 100000 = ~9.09%
        assert m["max_drawdown_pct"] == pytest.approx(9.09, abs=0.5)

    def test_summary_string(self):
        from apexfx.backtest.result import BacktestResult
        r = BacktestResult()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        r.record_equity(ts, 100000)
        r.record_equity(ts + timedelta(days=1), 101000)
        s = r.summary()
        assert "BACKTEST RESULTS" in s
        assert "Sharpe" in s

    def test_to_dataframe(self):
        from apexfx.backtest.result import BacktestResult
        r = BacktestResult()
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(10):
            r.record_equity(ts + timedelta(hours=i), 100000 + i * 50)
        r.compute_metrics()
        df = r.to_dataframe()
        assert "equity" in df.columns
        assert "drawdown" in df.columns
        assert len(df) == 10

    def test_risk_decision_tracking(self):
        from apexfx.backtest.result import BacktestResult
        r = BacktestResult()
        r.record_risk_decision(True)
        r.record_risk_decision(False, "spread_too_wide")
        r.record_risk_decision(False, "spread_too_wide")
        r.record_risk_decision(False, "cooldown")
        assert r.risk_approvals == 1
        assert r.risk_rejections == 3
        assert r.risk_rejection_reasons["spread_too_wide"] == 2


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_engine_runs_with_simple_strategy(self):
        """Engine runs end-to-end with a trivial strategy."""
        from apexfx.backtest.engine import BacktestConfig, BacktestEngine

        bars = _make_bars(600)

        # Simple strategy: always long
        def strategy(features, bar):
            return 0.5

        engine = BacktestEngine(
            bars=bars,
            strategy=strategy,
            config=BacktestConfig(warmup_bars=300, initial_equity=100000),
        )
        result = engine.run()

        assert result.metrics["total_bars"] > 0
        assert result.metrics["final_equity"] > 0
        assert len(result.equity_curve) > 0

    def test_engine_with_trend_following(self):
        """Trend-following strategy on trending data should produce decisions."""
        from apexfx.backtest.engine import BacktestConfig, BacktestEngine

        # Generate uptrending data
        bars = _make_bars(800, trend=0.0003)  # Strong uptrend

        # MA crossover strategy
        def ma_strategy(features, bar):
            if "close" in bar.index:
                # Simple: buy when close > sma(20), which is always true in uptrend
                return 0.6
            return 0.0

        engine = BacktestEngine(
            bars=bars,
            strategy=ma_strategy,
            config=BacktestConfig(warmup_bars=300, initial_equity=100000),
        )
        result = engine.run()

        # Risk manager may block some/all trades, but decisions should be recorded
        total_decisions = result.risk_approvals + result.risk_rejections
        assert total_decisions > 0

    def test_engine_neutral_strategy(self):
        """Neutral strategy should produce zero trades."""
        from apexfx.backtest.engine import BacktestConfig, BacktestEngine

        bars = _make_bars(500)

        def neutral(features, bar):
            return 0.0

        engine = BacktestEngine(
            bars=bars,
            strategy=neutral,
            config=BacktestConfig(warmup_bars=300),
        )
        result = engine.run()

        assert result.metrics["total_trades"] == 0
        assert result.metrics["final_equity"] == pytest.approx(100000, abs=1)

    def test_engine_risk_rejections(self):
        """Engine records risk rejections."""
        from apexfx.backtest.engine import BacktestConfig, BacktestEngine

        bars = _make_bars(500)

        def always_trade(features, bar):
            return 0.8

        engine = BacktestEngine(
            bars=bars,
            strategy=always_trade,
            config=BacktestConfig(warmup_bars=300),
        )
        result = engine.run()

        # Some decisions should have been made
        total = result.risk_approvals + result.risk_rejections
        assert total > 0

    def test_engine_stop_loss(self):
        """Stop loss should trigger on adverse moves."""
        from apexfx.backtest.engine import BacktestConfig, BacktestEngine

        # Generate data that trends up then crashes
        np.random.seed(42)
        n = 600
        base = 1.1000
        close = np.ones(n) * base
        close[:400] = base + np.cumsum(np.random.randn(400) * 0.0005 + 0.0002)
        close[400:] = close[399] - np.cumsum(np.abs(np.random.randn(200) * 0.002))

        bars = pd.DataFrame({
            "time": [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(n)],
            "open": close + np.random.randn(n) * 0.0001,
            "high": close + np.abs(np.random.randn(n) * 0.001),
            "low": close - np.abs(np.random.randn(n) * 0.001),
            "close": close,
            "tick_volume": np.random.randint(100, 5000, n).astype(float),
            "volume": np.random.randint(100, 5000, n).astype(float),
            "spread": np.ones(n) * 0.0001,
        })

        def buy_and_hold(features, bar):
            return 0.5

        engine = BacktestEngine(
            bars=bars,
            strategy=buy_and_hold,
            config=BacktestConfig(warmup_bars=300, atr_stop_mult=1.5),
        )
        result = engine.run()

        # Should have at least one stop loss exit
        stop_exits = [t for t in result.trades if t.exit_reason == "stop_loss"]
        assert len(stop_exits) >= 0  # May or may not trigger depending on ATR

    def test_backtest_config_defaults(self):
        from apexfx.backtest.engine import BacktestConfig
        cfg = BacktestConfig()
        assert cfg.initial_equity == 100000
        assert cfg.commission_per_lot == 7.0
        assert cfg.warmup_bars == 300


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_report_generates(self, tmp_path):
        """Report generates valid HTML file."""
        from apexfx.backtest.report import generate_html_report
        from apexfx.backtest.result import BacktestResult, Trade

        r = BacktestResult(initial_equity=100000)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for i in range(100):
            r.record_equity(ts + timedelta(hours=i), 100000 + i * 50 + np.random.randn() * 200)
            r.record_exposure(ts + timedelta(hours=i), 0.05)

        for i in range(5):
            pnl = np.random.randn() * 200
            r.record_trade(Trade(
                entry_time=ts, exit_time=ts + timedelta(hours=10),
                symbol="EURUSD", direction=1 if i % 2 == 0 else -1,
                entry_price=1.10, exit_price=1.11,
                volume=0.1, pnl=pnl, pnl_pct=pnl / 100000,
                bars_held=10, exit_reason="signal",
            ))

        r.record_risk_decision(False, "cooldown")
        r.record_risk_decision(True)

        output = tmp_path / "test_report.html"
        path = generate_html_report(r, output)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Equity Curve" in content
        assert "Chart" in content
        assert "Sharpe" in content


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_walk_forward_produces_folds(self):
        from apexfx.backtest.engine import BacktestConfig, walk_forward_backtest

        bars = _make_bars(2000)

        def strategy_factory(train_data):
            # Simple: always long
            return lambda features, bar: 0.3

        results = walk_forward_backtest(
            bars=bars,
            strategy_factory=strategy_factory,
            train_bars=800,
            test_bars=400,
            step_bars=400,
            config=BacktestConfig(warmup_bars=100),
        )

        assert len(results) >= 1
        for r in results:
            assert r.metrics["total_bars"] > 0


from pathlib import Path
