"""Tests for trivial trading baselines.

The baselines themselves are simple by design — the tests' job is to
*lock in* their semantics so future refactors don't silently change
what the comparison report measures.

Test plan
---------
* Each baseline produces actions in {-1, 0, +1} after evaluate snap.
* B&H is always +1.
* MA cross flips at the canonical golden/death cross.
* Donchian goes long on a fresh high, flat (or short) on a fresh low,
  holds in between.
* Random with the same seed is reproducible across reset() calls.
* evaluate_on_data: monotonic uptrend → B&H positive return; flat
  series → ~0 return; spread cost is correctly deducted on flips;
  no look-ahead (action at bar i sees only prices ≤ i).
* Profit factor / Sharpe / max DD all sane on the trivial cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexfx.eval.baselines import (
    BaselineExecConfig,
    BuyAndHoldBaseline,
    DonchianBaseline,
    MACrossBaseline,
    RandomBaseline,
    evaluate_on_data,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _ohlc_uptrend(n: int = 200, drift: float = 0.0005, seed: int = 1) -> pd.DataFrame:
    """OHLC where close steadily drifts upward."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=drift, scale=0.0003, size=n)
    close = 1.10 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0002, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0002, n)))
    return pd.DataFrame({"close": close, "high": high, "low": low})


def _ohlc_flat(n: int = 200) -> pd.DataFrame:
    close = np.full(n, 1.10)
    return pd.DataFrame({"close": close, "high": close.copy(), "low": close.copy()})


def _ohlc_downtrend(n: int = 200, drift: float = -0.0005, seed: int = 2) -> pd.DataFrame:
    return _ohlc_uptrend(n=n, drift=drift, seed=seed)


# ---------------------------------------------------------------------------
# BuyAndHoldBaseline
# ---------------------------------------------------------------------------


class TestBuyAndHold:
    def test_always_long(self) -> None:
        b = BuyAndHoldBaseline()
        for n in (1, 5, 100):
            close = np.linspace(1.0, 1.5, n)
            assert b.predict_action(close, close, close) == 1.0

    def test_uptrend_yields_positive_return(self) -> None:
        df = _ohlc_uptrend(n=300, drift=0.0005, seed=11)
        result = evaluate_on_data(BuyAndHoldBaseline(), df)
        assert result.total_return_pct > 0.0
        # Over 300 bars at drift 0.05%/bar, expect notional ≥ 5%
        assert result.total_return_pct > 5.0

    def test_downtrend_yields_negative_return(self) -> None:
        df = _ohlc_downtrend(n=300, drift=-0.0005, seed=12)
        result = evaluate_on_data(BuyAndHoldBaseline(), df)
        assert result.total_return_pct < 0.0

    def test_only_one_trade_open(self) -> None:
        # B&H opens one direction-change trade on bar 1 (flat→long) and never closes
        df = _ohlc_uptrend(n=100)
        result = evaluate_on_data(BuyAndHoldBaseline(), df)
        assert result.n_trades == 1


# ---------------------------------------------------------------------------
# MACrossBaseline
# ---------------------------------------------------------------------------


class TestMACross:
    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            MACrossBaseline(fast=0, slow=10)
        with pytest.raises(ValueError):
            MACrossBaseline(fast=10, slow=10)
        with pytest.raises(ValueError):
            MACrossBaseline(fast=20, slow=5)

    def test_flat_until_warmup(self) -> None:
        b = MACrossBaseline(fast=5, slow=10)
        close = np.linspace(1.0, 1.1, 9)  # one less than slow
        action = b.predict_action(close, close, close)
        assert action == 0.0

    def test_uptrend_goes_long(self) -> None:
        b = MACrossBaseline(fast=5, slow=20)
        close = np.linspace(1.0, 1.5, 50)  # rising
        action = b.predict_action(close, close, close)
        assert action == 1.0

    def test_downtrend_goes_short(self) -> None:
        b = MACrossBaseline(fast=5, slow=20)
        close = np.linspace(1.5, 1.0, 50)  # falling
        action = b.predict_action(close, close, close)
        assert action == -1.0

    def test_flips_at_crossover(self) -> None:
        b = MACrossBaseline(fast=3, slow=6)
        # First half down, second half sharply up
        close = np.concatenate([np.linspace(1.5, 1.0, 50), np.linspace(1.0, 2.0, 50)])
        actions = [b.predict_action(close[:i], close[:i], close[:i]) for i in range(7, 100)]
        # Should have at least one transition from -1 → +1
        sign_changes = sum(
            1 for a, b_ in zip(actions, actions[1:], strict=False) if a != b_ and a != 0 and b_ != 0
        )
        assert sign_changes >= 1


# ---------------------------------------------------------------------------
# DonchianBaseline
# ---------------------------------------------------------------------------


class TestDonchian:
    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError):
            DonchianBaseline(window=0)
        with pytest.raises(ValueError):
            DonchianBaseline(window=-5)

    def test_flat_until_warmup(self) -> None:
        b = DonchianBaseline(window=10)
        close = np.linspace(1.0, 1.1, 9)
        action = b.predict_action(close, close, close)
        assert action == 0.0

    def test_breakout_goes_long(self) -> None:
        b = DonchianBaseline(window=10, tol_pct=0.0001)
        close = np.array([1.10] * 9 + [1.105])  # close breaks above the prior 10-bar high
        high = close.copy()
        low = np.array([1.09] * 10)
        action = b.predict_action(close, high, low)
        assert action == 1.0

    def test_breakdown_goes_short(self) -> None:
        b = DonchianBaseline(window=10, tol_pct=0.0001)
        close = np.array([1.10] * 9 + [1.095])
        high = np.array([1.11] * 10)
        low = np.array([1.10] * 9 + [1.095])
        action = b.predict_action(close, high, low)
        assert action == -1.0

    def test_long_only_does_not_short(self) -> None:
        b = DonchianBaseline(window=10, tol_pct=0.0001, long_only=True)
        close = np.array([1.10] * 9 + [1.095])
        high = np.array([1.11] * 10)
        low = np.array([1.10] * 9 + [1.095])
        action = b.predict_action(close, high, low)
        assert action == 0.0  # exit, not short

    def test_holds_in_bracket(self) -> None:
        b = DonchianBaseline(window=10, tol_pct=0.0)
        # Establish a long via fresh high
        close_break = np.array([1.10] * 9 + [1.20])
        high_break = close_break.copy()
        low_break = np.array([1.05] * 10)
        b.predict_action(close_break, high_break, low_break)
        # Now sit inside the bracket — should hold long
        close_inside = np.array([1.10] * 9 + [1.15])
        action = b.predict_action(close_inside, high_break, low_break)
        assert action == 1.0

    def test_reset_clears_state(self) -> None:
        b = DonchianBaseline(window=5, tol_pct=0.0)
        close = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.10])
        high = close.copy()
        low = np.array([0.95] * 6)
        b.predict_action(close, high, low)
        assert b._last_action == 1.0
        b.reset()
        assert b._last_action == 0.0


# ---------------------------------------------------------------------------
# RandomBaseline
# ---------------------------------------------------------------------------


class TestRandom:
    def test_invalid_p_flat_raises(self) -> None:
        with pytest.raises(ValueError):
            RandomBaseline(p_flat=-0.1)
        with pytest.raises(ValueError):
            RandomBaseline(p_flat=1.1)

    def test_actions_are_in_set(self) -> None:
        b = RandomBaseline(seed=42)
        seen = set()
        for _ in range(200):
            a = b.predict_action(
                np.array([1.10]), np.array([1.10]), np.array([1.10])
            )
            seen.add(a)
            assert a in (-1.0, 0.0, 1.0)
        # All three actions should appear over 200 calls
        assert seen == {-1.0, 0.0, 1.0}

    def test_reset_makes_reproducible(self) -> None:
        b = RandomBaseline(seed=7)
        first_seq = [
            b.predict_action(np.array([1.0]), np.array([1.0]), np.array([1.0]))
            for _ in range(20)
        ]
        b.reset()
        second_seq = [
            b.predict_action(np.array([1.0]), np.array([1.0]), np.array([1.0]))
            for _ in range(20)
        ]
        assert first_seq == second_seq

    def test_random_loses_money_on_realistic_costs(self) -> None:
        # With 2 pip spread and noisy prices, random should be strongly negative
        df = _ohlc_uptrend(n=500, drift=0.0, seed=99)  # No drift — pure noise
        result = evaluate_on_data(
            RandomBaseline(seed=99),
            df,
            config=BaselineExecConfig(transaction_cost_pips=2.0),
        )
        assert result.total_return_pct < 0.0


# ---------------------------------------------------------------------------
# evaluate_on_data — execution model
# ---------------------------------------------------------------------------


class TestEvaluateOnData:
    def test_missing_close_raises(self) -> None:
        df = pd.DataFrame({"price": [1.0, 1.1]})
        with pytest.raises(ValueError):
            evaluate_on_data(BuyAndHoldBaseline(), df)

    def test_short_data_returns_neutral(self) -> None:
        df = pd.DataFrame({"close": [1.0]})
        result = evaluate_on_data(BuyAndHoldBaseline(), df)
        assert result.n_trades == 0
        assert result.total_return_pct == 0.0

    def test_high_low_default_to_close_when_missing(self) -> None:
        df = pd.DataFrame({"close": np.linspace(1.0, 1.1, 50)})
        # Should not raise
        result = evaluate_on_data(BuyAndHoldBaseline(), df)
        assert result.n_bars == 50

    def test_flat_series_zero_return_after_costs(self) -> None:
        df = _ohlc_flat(n=100)
        result = evaluate_on_data(BuyAndHoldBaseline(), df)
        # Flat prices → no PnL, but B&H opens a position with spread cost on bar 1
        # Cost = 2 pips on $1.10 = ~0.018%. Expect a small negative return.
        assert -0.05 < result.total_return_pct < 0.0

    def test_no_look_ahead(self) -> None:
        """Baseline at bar i must only see prices up to i — verify by failing fast."""

        class CheatingBaseline(BuyAndHoldBaseline):
            def predict_action(self, close_history, high_history, low_history):
                # If runner gives us future data we'll record the length
                CheatingBaseline.last_len = len(close_history)
                return 1.0

        df = _ohlc_uptrend(n=10)
        evaluate_on_data(CheatingBaseline(), df)
        # On the last call, history length should equal n - 1 (we use bars 0..n-2)
        assert CheatingBaseline.last_len == len(df) - 1

    def test_spread_cost_reduces_return(self) -> None:
        """High spread should produce a worse return than zero spread."""
        df = _ohlc_uptrend(n=300, drift=0.0005, seed=21)
        b = MACrossBaseline(fast=5, slow=20)
        cheap = evaluate_on_data(b, df, config=BaselineExecConfig(transaction_cost_pips=0.0))
        b2 = MACrossBaseline(fast=5, slow=20)
        expensive = evaluate_on_data(
            b2, df, config=BaselineExecConfig(transaction_cost_pips=10.0)
        )
        assert cheap.total_return_pct >= expensive.total_return_pct

    def test_metrics_are_sane_on_uptrend(self) -> None:
        df = _ohlc_uptrend(n=300, drift=0.0008, seed=31)
        result = evaluate_on_data(BuyAndHoldBaseline(), df)
        m = result.metrics
        assert m["total_return"] > 0.0
        assert m["max_drawdown"] >= 0.0
        assert m["max_drawdown"] <= 1.0
        # Win rate over 100% is impossible
        assert 0.0 <= m["win_rate"] <= 1.0

    def test_sharpe_increases_with_drift(self) -> None:
        df_low = _ohlc_uptrend(n=400, drift=0.0001, seed=41)
        df_high = _ohlc_uptrend(n=400, drift=0.0010, seed=41)
        sr_low = evaluate_on_data(BuyAndHoldBaseline(), df_low).sharpe_ratio
        sr_high = evaluate_on_data(BuyAndHoldBaseline(), df_high).sharpe_ratio
        assert sr_high > sr_low

    def test_equity_curve_length_matches_bars(self) -> None:
        df = _ohlc_uptrend(n=50, seed=51)
        result = evaluate_on_data(BuyAndHoldBaseline(), df)
        # equity_curve has bar count entries: initial + n-1 step updates
        assert len(result.equity_curve) == len(df)
