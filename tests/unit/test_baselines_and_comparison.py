"""Baselines, and the gate that makes them mean something.

The audit's central finding was not that the model was weak but that it lost to
buy & hold and nobody was checking. These cover the comparison harness and the
two guards that stop it producing a confident verdict from a run that cannot
support one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexfx.backtest.baselines import (
    BuyAndHold,
    DonchianBreakout,
    MACross,
    RandomStrategy,
    default_baselines,
)
from apexfx.backtest.comparison import ComparisonResult, StrategyScore
from apexfx.backtest.result import MIN_MEANINGFUL_VOLATILITY, BacktestResult


def _bar(close: float, high: float | None = None, low: float | None = None) -> pd.Series:
    return pd.Series({
        "close": close,
        "high": close if high is None else high,
        "low": close if low is None else low,
    })


class TestBuyAndHold:
    def test_always_long(self):
        strategy = BuyAndHold()
        assert all(strategy.on_bar(pd.Series(), _bar(1.1 + i)) == 1.0 for i in range(5))


class TestMACross:
    def test_flat_until_the_slow_window_fills(self):
        strategy = MACross(fast=2, slow=4)
        assert strategy.on_bar(pd.Series(), _bar(1.0)) == 0.0
        assert strategy.on_bar(pd.Series(), _bar(1.1)) == 0.0
        assert strategy.on_bar(pd.Series(), _bar(1.2)) == 0.0

    def test_long_when_fast_leads(self):
        strategy = MACross(fast=2, slow=4)
        for price in (1.0, 1.1, 1.2, 1.3):
            action = strategy.on_bar(pd.Series(), _bar(price))
        assert action == 1.0

    def test_short_when_fast_trails(self):
        strategy = MACross(fast=2, slow=4)
        for price in (1.3, 1.2, 1.1, 1.0):
            action = strategy.on_bar(pd.Series(), _bar(price))
        assert action == -1.0

    def test_long_only_holds_flat_instead_of_shorting(self):
        strategy = MACross(fast=2, slow=4, long_only=True)
        for price in (1.3, 1.2, 1.1, 1.0):
            action = strategy.on_bar(pd.Series(), _bar(price))
        assert action == 0.0

    def test_rejects_inverted_windows(self):
        with pytest.raises(ValueError, match="shorter"):
            MACross(fast=50, slow=20)


class TestDonchian:
    def test_no_lookahead_on_the_breakout_bar(self):
        """The channel must exclude the bar being judged.

        Including it lets the strategy trade on a high it has only just seen,
        which quietly inflates every backtest it appears in.
        """
        strategy = DonchianBreakout(lookback=3)
        for price in (1.0, 1.0, 1.0):
            strategy.on_bar(pd.Series(), _bar(price, high=1.0, low=1.0))

        # This bar makes a new high; the position may only turn long because
        # the *previous* three bars formed the channel.
        assert strategy.on_bar(pd.Series(), _bar(1.5, high=1.5, low=1.5)) == 1.0

    def test_breaks_short_on_a_new_low(self):
        strategy = DonchianBreakout(lookback=3)
        for price in (1.0, 1.0, 1.0):
            strategy.on_bar(pd.Series(), _bar(price, high=1.0, low=1.0))
        assert strategy.on_bar(pd.Series(), _bar(0.5, high=0.5, low=0.5)) == -1.0

    def test_long_only_flattens_instead_of_shorting(self):
        strategy = DonchianBreakout(lookback=3, long_only=True)
        for price in (1.0, 1.0, 1.0):
            strategy.on_bar(pd.Series(), _bar(price, high=1.0, low=1.0))
        assert strategy.on_bar(pd.Series(), _bar(0.5, high=0.5, low=0.5)) == 0.0

    def test_holds_position_between_breakouts(self):
        strategy = DonchianBreakout(lookback=2)
        for price in (1.0, 1.0):
            strategy.on_bar(pd.Series(), _bar(price, high=1.0, low=1.0))
        assert strategy.on_bar(pd.Series(), _bar(1.5, high=1.5, low=1.5)) == 1.0
        # No new extreme — the position stands.
        assert strategy.on_bar(pd.Series(), _bar(1.2, high=1.2, low=1.2)) == 1.0

    def test_rejects_degenerate_lookback(self):
        with pytest.raises(ValueError, match="at least 2"):
            DonchianBreakout(lookback=1)


class TestRandomStrategy:
    def test_reset_reproduces_the_sequence(self):
        strategy = RandomStrategy(seed=7)
        first = [strategy.on_bar(pd.Series(), _bar(1.0)) for _ in range(10)]
        strategy.reset()
        assert [strategy.on_bar(pd.Series(), _bar(1.0)) for _ in range(10)] == first


class TestDefaultSet:
    def test_names_are_unique(self):
        names = [b.name for b in default_baselines()]
        assert len(names) == len(set(names))

    def test_includes_buy_and_hold_and_random(self):
        names = {b.name for b in default_baselines()}
        assert {"buy_and_hold", "random"} <= names


def _score(name: str, **kwargs) -> StrategyScore:
    defaults = dict(
        total_return_pct=0.0, sharpe_ratio=0.0, max_drawdown_pct=0.0,
        profit_factor=1.0, n_trades=10, avg_exposure_pct=5.0,
        annual_volatility_pct=8.0,
    )
    defaults.update(kwargs)
    return StrategyScore(name=name, **defaults)


class TestComparisonVerdict:
    def test_candidate_must_out_sharpe_every_baseline(self):
        result = ComparisonResult(
            candidate=_score("model", sharpe_ratio=0.4),
            baselines=[_score("buy_and_hold", sharpe_ratio=0.65)],
        )
        assert not result.beats_best_baseline
        assert result.best_baseline.name == "buy_and_hold"

    def test_winning_candidate_is_recognised(self):
        result = ComparisonResult(
            candidate=_score("model", sharpe_ratio=1.1),
            baselines=[_score("buy_and_hold", sharpe_ratio=0.65)],
        )
        assert result.beats_best_baseline

    def test_random_is_excluded_from_the_bar_to_clear(self):
        """Random calibrates costs; beating it proves nothing."""
        result = ComparisonResult(
            candidate=_score("model", sharpe_ratio=0.1),
            baselines=[
                _score("buy_and_hold", sharpe_ratio=0.65),
                _score("random", sharpe_ratio=5.0),
            ],
        )
        assert result.best_baseline.name == "buy_and_hold"


class TestComparisonGuards:
    def test_negligible_exposure_is_flagged(self):
        """Run 5's shape: trades happened, size did not."""
        result = ComparisonResult(
            candidate=_score("model", sharpe_ratio=2.0, avg_exposure_pct=0.045),
            baselines=[_score("buy_and_hold", sharpe_ratio=0.65)],
        )
        assert not result.exposure_is_meaningful
        assert "too small" in result.summary()

    def test_real_exposure_passes(self):
        result = ComparisonResult(
            candidate=_score("model", avg_exposure_pct=12.0),
            baselines=[_score("buy_and_hold")],
        )
        assert result.exposure_is_meaningful

    def test_profitable_random_flags_the_cost_model(self):
        result = ComparisonResult(
            candidate=_score("model"),
            baselines=[_score("random", total_return_pct=4.0)],
        )
        assert not result.costs_look_charged
        assert "cost" in result.summary()

    def test_losing_random_is_the_expected_case(self):
        result = ComparisonResult(
            candidate=_score("model"),
            baselines=[_score("random", total_return_pct=-30.2)],
        )
        assert result.costs_look_charged


class TestAnnualisationIsInferred:
    """Annualising by a hardcoded 252 was wrong on every non-daily timeframe."""

    @staticmethod
    def _result_with(freq: str, n: int = 200) -> BacktestResult:
        result = BacktestResult(initial_equity=100_000.0)
        times = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
        rng = np.random.default_rng(0)
        equity = 100_000.0
        for t in times:
            equity *= 1 + rng.normal(0.0, 0.005)
            result.record_equity(t.to_pydatetime(), equity)
        return result

    def test_hourly_bars_infer_hourly_periods(self):
        result = self._result_with("h")
        assert result.periods_per_year() == pytest.approx(252 * 24, rel=0.01)

    def test_daily_bars_infer_daily_periods(self):
        result = self._result_with("D")
        assert result.periods_per_year() == pytest.approx(252, rel=0.01)

    def test_volatility_scales_with_the_inferred_period(self):
        hourly = self._result_with("h").compute_metrics()["annual_volatility_pct"]
        daily = self._result_with("D").compute_metrics()["annual_volatility_pct"]
        assert hourly == pytest.approx(daily * np.sqrt(24), rel=0.05)


class TestSharpeRefusesMeaninglessInput:
    def test_a_barely_trading_strategy_reports_no_sharpe(self):
        """Observed: 0.015% annual vol produced a "Sharpe" of -343.

        The risk-free subtraction divided by a near-zero denominator. Reporting
        0.0 says "no risk-adjusted return to speak of", which is true.
        """
        result = BacktestResult(initial_equity=100_000.0)
        times = pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC")
        equity = 100_000.0
        for i, t in enumerate(times):
            equity += 0.01 * (1 if i % 2 else -1)  # essentially flat
            result.record_equity(t.to_pydatetime(), equity)

        metrics = result.compute_metrics()
        assert metrics["annual_volatility_pct"] / 100 < MIN_MEANINGFUL_VOLATILITY
        assert metrics["sharpe_ratio"] == 0.0

    def test_a_real_return_series_still_gets_a_sharpe(self):
        result = BacktestResult(initial_equity=100_000.0)
        times = pd.date_range("2024-01-01", periods=500, freq="D", tz="UTC")
        rng = np.random.default_rng(3)
        equity = 100_000.0
        for t in times:
            equity *= 1 + rng.normal(0.0008, 0.01)
            result.record_equity(t.to_pydatetime(), equity)

        sharpe = result.compute_metrics()["sharpe_ratio"]
        assert sharpe != 0.0
        assert abs(sharpe) < 20, f"implausible Sharpe {sharpe}"


class TestSpreadIsCharged:
    """A round trip must pay the spread.

    Before this, ``spread_pips`` only fed the risk manager's spread check while
    fills used ``close +/- slippage``, so crossing the bid/ask cost nothing and
    every backtested trade was cheaper than the real one.
    """

    N_BARS = 160
    FLAT_PRICE = 1.10
    VOLUME = 0.1

    @staticmethod
    def _flat_bars(n: int, price: float) -> pd.DataFrame:
        """A price that never moves, so any P&L is pure cost.

        Flat bars also give ATR = 0, which disables the stop and take-profit
        legs — the round trip is opened and closed by signal alone.
        """
        close = np.full(n, price)
        return pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(n, 500),
            "spread": np.full(n, 0.0001),
        })

    def _round_trip(self, spread_pips: float, slippage_pips: float = 0.0):
        """One long round trip on a flat price; returns (trade, config)."""
        from apexfx.backtest.engine import BacktestConfig, BacktestEngine

        class _OneRoundTrip:
            """Long for a stretch, then flat — exactly one open and one close."""

            def __init__(self, enter_at: int, exit_at: int) -> None:
                self._i = 0
                self._enter, self._exit = enter_at, exit_at

            def on_bar(self, features, bar):  # noqa: ARG002
                self._i += 1
                return 1.0 if self._enter <= self._i < self._exit else 0.0

        # Risk sizing is bypassed on purpose: this measures the cost model, and
        # the sizer currently rejects most signals ("Position size computed to
        # zero"), which would leave nothing to measure.
        config = BacktestConfig(
            warmup_bars=20,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            disable_risk=True,
            default_volume=self.VOLUME,
        )
        engine = BacktestEngine(
            bars=self._flat_bars(self.N_BARS, self.FLAT_PRICE),
            strategy=_OneRoundTrip(enter_at=40, exit_at=80),
            config=config,
        )
        result = engine.run()
        assert len(result.trades) == 1, f"expected one trade, got {len(result.trades)}"
        return result.trades[0], config

    def test_a_flat_market_loses_exactly_the_spread(self):
        """Price never moves, so the whole loss is the cost of crossing."""
        trade, config = self._round_trip(spread_pips=2.0)

        expected_spread_cost = (
            config.spread_pips * config.pip_value * self.VOLUME * config.contract_size
        )
        assert trade.pnl == pytest.approx(
            -(expected_spread_cost + self._reported_commission(config)), rel=1e-6,
        )

    def test_zero_spread_costs_only_commission(self):
        trade, config = self._round_trip(spread_pips=0.0)
        assert trade.pnl == pytest.approx(-self._reported_commission(config), rel=1e-6)

    def _reported_commission(self, config) -> float:
        """Commission visible in ``trade.pnl`` — the close leg only.

        The engine deducts the entry commission straight from equity and only
        the exit commission from ``trade.pnl``. Account equity is therefore
        right, but the per-trade figure understates the round trip by one
        commission — and profit_factor is computed from those trade P&Ls.
        Pinned here rather than changed: correcting it means moving both legs
        into the close, which is a change to trade accounting.
        """
        return self.VOLUME * config.commission_per_lot

    def test_cost_scales_with_the_spread(self):
        narrow, _ = self._round_trip(spread_pips=1.0)
        wide, config = self._round_trip(spread_pips=3.0)

        per_pip = config.pip_value * self.VOLUME * config.contract_size
        assert (narrow.pnl - wide.pnl) == pytest.approx(2 * per_pip, rel=1e-6)

    def test_default_config_charges_a_retail_spread(self):
        from apexfx.backtest.engine import BacktestConfig

        assert BacktestConfig().spread_pips >= 1.5, (
            "retail MT5 on EURUSD runs 1.5-3.0 pips; a lower default flatters "
            "every backtest"
        )
