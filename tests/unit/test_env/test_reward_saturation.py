"""Tests for reward saturation / scale fix (audit finding #1).

The old implementation used ``np.clip(reward, -10, 10)`` with
``reward_scale=1000``, which in practice saturated every non-trivial bar and
zeroed the policy gradient.  After Patch #3 we:

    * reduce ``reward_scale`` defaults from 1000 → 100
    * replace the hard clip with ``tanh(_SAT_SCALE * reward)`` soft-saturation

These tests lock in:

    * reward is always finite and in (-1, 1) (never exactly ±10 from a clip)
    * gradient-preserving behaviour — for typical H4 returns the emitted
      reward stays well inside the linear region of ``tanh`` (|r| < 0.9)
    * sign/ordering preserved (monotonicity of saturation)
    * extreme inputs soft-clamp instead of producing infinities / NaN
"""

from __future__ import annotations

import numpy as np

from apexfx.env.reward import (
    LogReturnReward,
    OnlineSharpeTracker,
    TradingReward,
    _soft_saturate,
)


class TestSoftSaturate:
    def test_finite_zero_maps_to_zero(self) -> None:
        assert _soft_saturate(0.0) == 0.0

    def test_monotone_increasing(self) -> None:
        pts = np.linspace(-10, 10, 21)
        vals = [_soft_saturate(float(x)) for x in pts]
        assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:], strict=False))

    def test_bounded_within_one(self) -> None:
        """Saturation is bounded at ±1 (may hit the float boundary for huge inputs)."""
        for x in [-1000.0, -10.0, 10.0, 1000.0, 1e9, -1e9]:
            assert abs(_soft_saturate(x)) <= 1.0

    def test_bounded_strictly_below_one_for_moderate_inputs(self) -> None:
        """For inputs in the 'normal reward' band, output is strictly inside (-1, 1)."""
        for x in [-5.0, -2.0, -1.0, 0.5, 1.0, 2.0, 5.0]:
            assert abs(_soft_saturate(x)) < 1.0

    def test_nan_and_inf_handled(self) -> None:
        assert _soft_saturate(float("nan")) == 0.0
        # tanh(inf)=1 but we explicitly guard non-finite
        assert _soft_saturate(float("inf")) == 0.0
        assert _soft_saturate(float("-inf")) == 0.0

    def test_gradient_preserving_in_typical_band(self) -> None:
        """A typical scaled reward (|r|<1) should sit in the linear-ish zone."""
        # dtanh/dx at 0 is 1; at x=0.5 it's ~0.786.  We require the mapping
        # is not flat: f(0.5) - f(0.1) must be noticeably positive.
        assert _soft_saturate(0.5) - _soft_saturate(0.1) > 0.1


class TestTradingRewardScaling:
    def test_default_scale_is_100_not_1000(self) -> None:
        """Regression: default reward_scale must stay at 100 (finding #1)."""
        rw = TradingReward()
        assert rw.reward_scale == 100.0

    def test_reward_bounded(self) -> None:
        """Output always finite and within ±1."""
        rw = TradingReward()
        rw.set_trade_info(0.5, 1, 100.0, 5)
        for prev, cur in [
            (100000, 100100),
            (100000, 99900),
            (100000, 150000),    # huge positive — would have clipped to +10
            (100000, 50000),     # huge negative — would have clipped to -10
        ]:
            r = rw.compute(float(cur), float(prev))
            assert np.isfinite(r)
            assert -1.0 <= r <= 1.0
        # At the typical-H4 scale, reward must NOT be stuck at ±1 (else we're
        # back to the original saturation pathology).
        rw.reset()
        rw.set_trade_info(0.5, 1, 100.0, 5)
        r_typical = rw.compute(100100.0, 100000.0)
        assert abs(r_typical) < 0.9

    def test_reward_does_not_saturate_on_typical_h4_bar(self) -> None:
        """Typical ±0.1% H4 move must land inside the linear band."""
        rw = TradingReward()
        rw.set_trade_info(0.5, 1, 50.0, 5)
        # +0.1% move (100.1 pip-equivalent) — very typical for H4 EURUSD
        r = rw.compute(100100.0, 100000.0)
        # Not saturated — |r| comfortably below 0.9
        assert abs(r) < 0.9

    def test_blowup_returns_finite_soft_clamped(self) -> None:
        """Catastrophic blow-up (portfolio ≤ 0) returns bounded finite reward."""
        rw = TradingReward()
        rw.set_trade_info(0.0, 0, 0.0, 0)
        r = rw.compute(0.0, 100000.0)
        assert np.isfinite(r)
        assert -1.0 < r < 0.0  # negative but not a hard -10

    def test_sign_preserved_after_scale_change(self) -> None:
        rw = TradingReward()
        rw.set_trade_info(0.5, 1, 100.0, 5)
        r_pos = rw.compute(100100.0, 100000.0)
        rw.reset()
        rw.set_trade_info(0.5, 1, -100.0, 5)
        r_neg = rw.compute(99900.0, 100000.0)
        assert r_pos > 0
        assert r_neg < 0

    def test_asymmetric_loss_preserved(self) -> None:
        rw = TradingReward(loss_weight=2.0)
        rw.set_trade_info(0.0, 0, 0.0, 0)
        pos = rw.compute(100100.0, 100000.0)
        rw.reset()
        rw.set_trade_info(0.0, 0, 0.0, 0)
        neg = rw.compute(99900.0, 100000.0)
        assert abs(neg) > abs(pos)


class TestLogReturnReward:
    def test_default_scale_100(self) -> None:
        r = LogReturnReward()
        assert r.reward_scale == 100.0

    def test_bounded(self) -> None:
        r = LogReturnReward()
        for prev, cur in [(100.0, 101.0), (100.0, 99.0), (100.0, 200.0), (100.0, 1.0)]:
            val = r.compute(float(cur), float(prev))
            assert np.isfinite(val)
            assert -1.0 <= val <= 1.0
        # Typical small move must remain unsaturated
        val_typical = r.compute(100.1, 100.0)
        assert abs(val_typical) < 0.9

    def test_blowup_bounded(self) -> None:
        r = LogReturnReward()
        val = r.compute(0.0, 100.0)
        assert np.isfinite(val)
        assert -1.0 < val < 0.0


class TestOnlineSharpeTracker:
    def test_empty_returns_zero(self) -> None:
        t = OnlineSharpeTracker()
        assert t.value() == 0.0

    def test_single_update_returns_zero(self) -> None:
        t = OnlineSharpeTracker()
        t.update(100.0)
        assert t.value() == 0.0

    def test_positive_drift_gives_positive_sharpe(self) -> None:
        t = OnlineSharpeTracker(periods_per_year=252)
        rng = np.random.RandomState(0)
        val = 100.0
        t.update(val)
        for _ in range(500):
            # ~0.1% mean drift + 0.2% noise → clearly positive Sharpe
            val *= float(np.exp(0.001 + 0.002 * rng.randn()))
            t.update(val)
        assert t.value() > 0.5

    def test_negative_drift_gives_negative_sharpe(self) -> None:
        t = OnlineSharpeTracker(periods_per_year=252)
        rng = np.random.RandomState(1)
        val = 100.0
        t.update(val)
        for _ in range(500):
            val *= float(np.exp(-0.001 + 0.002 * rng.randn()))
            t.update(val)
        assert t.value() < -0.5

    def test_rolling_window_forgets(self) -> None:
        t = OnlineSharpeTracker(periods_per_year=252, window=10)
        for v in range(1, 50):
            t.update(float(100 + v))
        assert t.n_samples() == 10

    def test_zero_variance_returns_zero(self) -> None:
        t = OnlineSharpeTracker()
        for _ in range(50):
            t.update(100.0)  # flat, zero log-returns
        assert t.value() == 0.0

    def test_nan_values_are_skipped(self) -> None:
        """Non-finite portfolio values must not poison the series."""
        t = OnlineSharpeTracker()
        t.update(100.0)
        t.update(float("nan"))  # should be ignored
        t.update(101.0)
        # Only 1 valid return so Sharpe rounds to 0 (single sample, ddof=1).
        assert np.isfinite(t.value())

    def test_reset_clears_state(self) -> None:
        t = OnlineSharpeTracker()
        for v in [100.0, 101.0, 102.0]:
            t.update(v)
        assert t.n_samples() > 0
        t.reset()
        assert t.n_samples() == 0
        assert t.value() == 0.0
