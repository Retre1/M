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
from apexfx.backtest.comparison import (
    MIN_TRADES_PER_FOLD,
    ComparisonResult,
    FoldComparison,
    StrategyScore,
    compare_across_folds,
)
from apexfx.backtest.engine import BacktestConfig
from apexfx.backtest.result import MIN_MEANINGFUL_VOLATILITY, BacktestResult
from apexfx.features.pipeline import FeaturePipeline


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
        """Commission visible in ``trade.pnl`` — both legs.

        This used to report the close leg only: the entry commission came out
        of equity at open and never reached ``trade.pnl``, so every round trip
        looked one commission cheaper than it was. Equity was right and the
        per-trade figure was not, which mattered because profit_factor — a
        gate-2 metric — is computed from those trade P&Ls.
        """
        return 2 * self.VOLUME * config.commission_per_lot

    def test_cost_scales_with_the_spread(self):
        narrow, _ = self._round_trip(spread_pips=1.0)
        wide, config = self._round_trip(spread_pips=3.0)

        per_pip = config.pip_value * self.VOLUME * config.contract_size
        assert (narrow.pnl - wide.pnl) == pytest.approx(2 * per_pip, rel=1e-6)

    def test_equity_moves_by_exactly_the_reported_trade_pnl(self):
        """The invariant the split legs broke.

        The entry commission left equity at open and never reached
        ``trade.pnl``, so the account and the trade report disagreed by one
        commission on every round trip — and profit_factor is built from the
        trade side.
        """
        from apexfx.backtest.engine import BacktestConfig, BacktestEngine

        class _OneRoundTrip:
            def __init__(self) -> None:
                self._i = 0

            def on_bar(self, features, bar):  # noqa: ARG002
                self._i += 1
                return 1.0 if 40 <= self._i < 80 else 0.0

        config = BacktestConfig(
            warmup_bars=20, spread_pips=2.0, slippage_pips=0.0,
            disable_risk=True, default_volume=self.VOLUME,
        )
        result = BacktestEngine(
            bars=self._flat_bars(self.N_BARS, self.FLAT_PRICE),
            strategy=_OneRoundTrip(),
            config=config,
        ).run()

        assert len(result.trades) == 1
        # equity_curve holds (timestamp, equity) pairs.
        equity_change = result.equity_curve[-1][1] - config.initial_equity
        assert equity_change == pytest.approx(result.trades[0].pnl, rel=1e-9)

    def test_default_config_charges_a_retail_spread(self):
        from apexfx.backtest.engine import BacktestConfig

        assert BacktestConfig().spread_pips >= 1.5, (
            "retail MT5 on EURUSD runs 1.5-3.0 pips; a lower default flatters "
            "every backtest"
        )


class TestFoldComparisonVerdict:
    """The gate reads per-trade expectancy in R, not a share of winning folds.

    The retired rule was "beat the best baseline in 80% of folds". It collapsed
    each fold to one bit, threw away the sample size, and the 80% itself was
    asserted rather than derived.
    """

    @staticmethod
    def _comparison(
        *, mean_r: float, n_per_fold: int, n_folds: int = 9, wins: int | None = None,
        sd: float = 1.0, seed: int = 0, **kwargs,
    ) -> FoldComparison:
        """Build a comparison with a known true mean R and a chosen win count."""
        rng = np.random.default_rng(seed)
        trades = [rng.normal(mean_r, sd, n_per_fold) for _ in range(n_folds)]
        wins = n_folds if wins is None else wins
        frame = pd.DataFrame({
            "model": [1.0] * wins + [0.0] * (n_folds - wins),
            "buy_and_hold": [0.5] * n_folds,
        })
        return FoldComparison(
            candidate_name="model",
            fold_sharpe=frame,
            fold_returns=kwargs.pop("fold_returns", {}),
            fold_trade_r={"model": trades, "buy_and_hold": trades},
            **kwargs,
        )

    def test_a_real_edge_on_a_large_sample_passes(self):
        c = self._comparison(mean_r=0.25, n_per_fold=200)
        assert c.p_value < 0.05
        assert c.passes_gate

    def test_the_same_edge_on_the_projects_actual_sample_does_not(self):
        """165 trades over 9 folds is ~18 each — below the floor entirely."""
        c = self._comparison(mean_r=0.25, n_per_fold=18)
        assert c.evaluable_folds == []
        assert not c.passes_gate
        assert "no evidence here either way" in c.summary()

    def test_no_edge_is_not_significant(self):
        c = self._comparison(mean_r=0.0, n_per_fold=200)
        assert c.p_value > 0.05
        assert not c.passes_gate

    def test_expectancy_alone_is_not_enough(self):
        """Positive mean R while losing to buy & hold on most folds fails."""
        c = self._comparison(mean_r=0.25, n_per_fold=200, wins=4)
        assert c.p_value < 0.05
        assert c.win_rate_p_value > 0.05
        assert not c.passes_gate

    def test_consistency_alone_is_not_enough(self):
        """Winning every fold while making nothing per trade also fails."""
        c = self._comparison(mean_r=0.0, n_per_fold=200, wins=9)
        assert c.win_rate_p_value < 0.05
        assert not c.passes_gate


class TestMinimumTradesPerFold:
    @staticmethod
    def _with_counts(counts: list[int], mean_r: float = 0.2) -> FoldComparison:
        rng = np.random.default_rng(1)
        trades = [rng.normal(mean_r, 1.0, n) for n in counts]
        frame = pd.DataFrame({
            "model": [1.0] * len(counts), "buy_and_hold": [0.5] * len(counts),
        })
        return FoldComparison(
            candidate_name="model", fold_sharpe=frame, fold_returns={},
            fold_trade_r={"model": trades, "buy_and_hold": trades},
        )

    def test_thin_folds_are_excluded_not_counted_as_losses(self):
        """A fold nobody traded is missing evidence, not evidence of failure."""
        c = self._with_counts([100, 5, 100])
        assert c.evaluable_folds == [0, 2]
        assert c.underpowered_folds == [1]

    def test_excluded_trades_do_not_enter_the_statistic(self):
        c = self._with_counts([100, 5, 100])
        assert c.n_trades == 200

    def test_the_exclusion_is_reported_not_silent(self):
        assert "excluded for under" in self._with_counts([100, 5, 100]).summary()

    def test_the_floor_is_the_documented_constant(self):
        just_under = self._with_counts([MIN_TRADES_PER_FOLD - 1] * 3)
        just_over = self._with_counts([MIN_TRADES_PER_FOLD] * 3)
        assert just_under.evaluable_folds == []
        assert len(just_over.evaluable_folds) == 3


class TestPowerIsReported:
    """Answers "could this sample have detected the edge being claimed?"."""

    @staticmethod
    def _sample(n_per_fold: int, mean_r: float = 0.0, sd: float = 1.0):
        rng = np.random.default_rng(2)
        trades = [rng.normal(mean_r, sd, n_per_fold) for _ in range(9)]
        frame = pd.DataFrame({"model": [1.0] * 9, "buy_and_hold": [0.5] * 9})
        return FoldComparison(
            candidate_name="model", fold_sharpe=frame, fold_returns={},
            fold_trade_r={"model": trades, "buy_and_hold": trades},
        )

    def test_a_bigger_sample_detects_a_smaller_edge(self):
        small = self._sample(40).minimum_detectable_r()
        large = self._sample(1000).minimum_detectable_r()
        assert large < small

    def test_the_detectable_effect_shrinks_as_the_root_of_n(self):
        """Quadrupling the trades roughly halves the detectable effect."""
        base = self._sample(100).minimum_detectable_r()
        quad = self._sample(400).minimum_detectable_r()
        assert quad == pytest.approx(base / 2, rel=0.12)

    def test_a_null_result_on_a_thin_sample_is_flagged_as_uninformative(self):
        summary = self._sample(35, mean_r=0.05).summary()
        assert "says nothing about the strategy" in summary

    def test_no_verdict_without_a_usable_sample(self):
        rng = np.random.default_rng(3)
        c = FoldComparison(
            candidate_name="model",
            fold_sharpe=pd.DataFrame({"model": [1.0], "buy_and_hold": [0.5]}),
            fold_returns={},
            fold_trade_r={"model": [rng.normal(0, 1, 40)], "buy_and_hold": []},
        )
        flat = FoldComparison(
            candidate_name="model",
            fold_sharpe=pd.DataFrame({"model": [1.0], "buy_and_hold": [0.5]}),
            fold_returns={},
            fold_trade_r={"model": [np.full(40, 0.1)], "buy_and_hold": []},
        )
        assert c.minimum_detectable_r() is not None
        # Zero variance: no standard error, so no detectable-effect statement.
        assert flat.minimum_detectable_r() is None
        assert flat.t_statistic is None


class TestWinRateThresholdIsDerived:
    """The 80% line is replaced by an exact binomial test."""

    @staticmethod
    def _wins(wins: int, n_folds: int = 9) -> FoldComparison:
        rng = np.random.default_rng(4)
        trades = [rng.normal(0.1, 1.0, 100) for _ in range(n_folds)]
        frame = pd.DataFrame({
            "model": [1.0] * wins + [0.0] * (n_folds - wins),
            "buy_and_hold": [0.5] * n_folds,
        })
        return FoldComparison(
            candidate_name="model", fold_sharpe=frame, fold_returns={},
            fold_trade_r={"model": trades, "buy_and_hold": trades},
        )

    def test_eight_of_nine_clears_alpha(self):
        assert self._wins(8).win_rate_p_value == pytest.approx(0.0195, abs=1e-3)

    def test_seven_of_nine_does_not(self):
        """7/9 is 78% — under the old rule a near miss, here plainly not significant."""
        assert self._wins(7).win_rate_p_value == pytest.approx(0.0898, abs=1e-3)

    def test_the_old_eighty_percent_line_sat_below_the_bar(self):
        """At 9 folds alpha = 0.05 demands 8 wins, i.e. 88.9%, not 80%."""
        assert self._wins(8).win_rate > 0.8
        assert self._wins(7).win_rate < 0.8
        assert self._wins(8).win_rate_p_value < 0.05 < self._wins(7).win_rate_p_value

    def test_the_optimism_of_the_p_value_is_stated(self):
        assert "optimistic" in self._wins(8).summary()


class TestFoldComparisonOverfitting:
    @staticmethod
    def _returns(n_obs: int = 400, seed: int = 0) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {
            name: rng.normal(0.0, 0.01, n_obs)
            for name in ("model", "buy_and_hold", "ma_cross")
        }

    @staticmethod
    def _comparison(returns: dict[str, np.ndarray]) -> FoldComparison:
        frame = pd.DataFrame({name: [0.0] for name in returns})
        # Trades are supplied so the summary gets past the "no evidence" guard;
        # PBO itself is computed from the bar returns, not from these.
        rng = np.random.default_rng(9)
        trades = {name: [rng.normal(0.0, 1.0, 60)] for name in returns}
        return FoldComparison(
            candidate_name="model", fold_sharpe=frame, fold_returns=returns,
            fold_trade_r=trades,
        )

    def test_pbo_is_reported_when_there_is_enough_history(self):
        pbo = self._comparison(self._returns()).probability_of_overfitting()
        assert pbo is not None
        assert 0.0 <= pbo <= 1.0

    def test_too_short_a_history_yields_no_verdict(self):
        """Better no number than one computed from four observations."""
        assert self._comparison(self._returns(n_obs=4)).probability_of_overfitting() is None

    def test_a_single_strategy_cannot_be_selected_between(self):
        """PBO measures a choice; with one column there was no choice to make."""
        single = {"model": np.random.default_rng(0).normal(0, 0.01, 400)}
        assert self._comparison(single).probability_of_overfitting() is None

    def test_no_returns_at_all_yields_no_verdict(self):
        """An all-skipped run leaves nothing to compute over — and min() over an
        empty sequence raises, so this path needs its own guard."""
        assert self._comparison({}).probability_of_overfitting() is None

    def test_series_of_different_lengths_are_truncated_not_rejected(self):
        """Folds get skipped per strategy, so the series can come out ragged."""
        returns = self._returns()
        returns["ma_cross"] = returns["ma_cross"][:250]
        assert self._comparison(returns).probability_of_overfitting() is not None

    def test_pbo_reaches_the_summary(self):
        assert "PBO" in self._comparison(self._returns()).summary()

    def test_pbo_survives_a_run_that_produced_no_verdict(self):
        """The realistic thin-fold case: no fold clears the trade floor, so
        there is no verdict — but PBO is computed from bar returns and does not
        depend on the trade count, so suppressing it would lose a diagnostic
        that is still valid."""
        returns = self._returns()
        thin = FoldComparison(
            candidate_name="model",
            fold_sharpe=pd.DataFrame({name: [0.0] for name in returns}),
            fold_returns=returns,
            fold_trade_r={name: [np.zeros(3)] for name in returns},
        )
        summary = thin.summary()
        assert thin.evaluable_folds == []
        assert "no evidence here either way" in summary
        assert "PBO" in summary


class TestCompareAcrossFolds:
    """The end-to-end path: a real engine run per fold per strategy.

    The folds are computed once for the class. Each run recomputes the feature
    pipeline over its segment, so re-running per test would cost more than the
    rest of the unit suite combined.
    """

    N_BARS = 900
    N_FOLDS = 3
    WARMUP = 40

    @staticmethod
    def _bars(n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 1.10 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
        return pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": np.full(n, 500),
            "spread": np.full(n, 0.0002),
        })

    @classmethod
    def _config(cls) -> BacktestConfig:
        # Risk sizing is bypassed for the same reason as the spread tests: the
        # sizer rejects most signals, and a fold in which nobody traded scores
        # every strategy at zero, which tests nothing.
        return BacktestConfig(
            warmup_bars=cls.WARMUP, disable_risk=True, default_volume=0.1,
        )

    @pytest.fixture(scope="class")
    @classmethod
    def comparison(cls) -> FoldComparison:
        return compare_across_folds(
            cls._bars(cls.N_BARS),
            BuyAndHold(),
            n_folds=cls.N_FOLDS,
            config=cls._config(),
            candidate_name="model",
            baselines=[MACross(fast=5, slow=20), RandomStrategy(seed=1)],
            pipeline=FeaturePipeline.lightweight(),
        )

    def test_every_fold_scores_every_strategy(self, comparison):
        assert len(comparison.fold_sharpe) == self.N_FOLDS
        assert set(comparison.fold_sharpe.columns) == {
            "model", "ma_cross_5_20", "random",
        }

    def test_folds_are_scored_separately_not_pooled(self, comparison):
        """Identical Sharpe on every fold would mean the segments were never
        actually split — the spread across conditions is the whole point."""
        assert comparison.fold_sharpe["model"].nunique() > 1

    def test_returns_are_collected_for_pbo(self, comparison):
        lengths = {name: len(r) for name, r in comparison.fold_returns.items()}
        assert set(lengths) == set(comparison.fold_sharpe.columns)
        assert min(lengths.values()) > 0

    def test_the_verdict_is_reported_over_folds_not_one_number(self, comparison):
        assert 0.0 <= comparison.win_rate <= 1.0
        assert f"of {self.N_FOLDS} folds" in comparison.summary()

    def test_a_fold_shorter_than_the_warmup_is_skipped(self):
        """Otherwise it contributes an empty backtest as a zero Sharpe."""
        comparison = compare_across_folds(
            self._bars(self.WARMUP * 4),
            BuyAndHold(),
            n_folds=8,  # 20 bars per fold, under the 40-bar warmup
            config=self._config(),
            baselines=[MACross(fast=5, slow=20)],
            pipeline=FeaturePipeline.lightweight(),
        )
        assert len(comparison.fold_sharpe) == 0
        assert comparison.win_rate == 0.0, "a run with no folds is not a clean sweep"
        assert not comparison.passes_gate

    def test_strategies_are_reset_between_folds(self):
        """A moving average carrying prices across a segment boundary would be
        fitting one fold with the previous fold's data."""

        class _CountingCandidate:
            name = "model"

            def __init__(self) -> None:
                self.resets = 0

            def reset(self) -> None:
                self.resets += 1

            def on_bar(self, features, bar):  # noqa: ARG002
                return 1.0

        candidate = _CountingCandidate()
        comparison = compare_across_folds(
            self._bars(self.N_BARS),
            candidate,
            n_folds=self.N_FOLDS,
            config=self._config(),
            baselines=[BuyAndHold()],
            pipeline=FeaturePipeline.lightweight(),
        )
        assert candidate.resets == len(comparison.fold_sharpe) == self.N_FOLDS
