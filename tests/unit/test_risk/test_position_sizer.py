"""Tests for position sizer.

The governing defect these pin down: ``max_position_pct`` capped *notional*
at 10% of equity. On EURUSD that is 0.1 lots before Kelly and 0.02 after, and
with a 2xATR stop the trade then risks 0.004% of equity. The measured effect
was 671 of 1155 signals rejected as "Position size computed to zero" and an
average exposure of 0.045% — a run that cannot demonstrate edge or its absence.

Sizing now starts from the risk taken if the stop is hit, which is the figure
the parameter always read as.
"""


import pytest

from apexfx.risk.position_sizer import PositionSizer


class TestPositionSizer:
    def test_zero_action(self):
        sizer = PositionSizer()
        lots = sizer.compute(0.0, 100_000, 1.1, 0.001, 0.001)
        assert lots == 0.0

    def test_half_action(self):
        sizer = PositionSizer()
        full_lots = sizer.compute(1.0, 100_000, 1.1, 0.001, 0.001)
        half_lots = sizer.compute(0.5, 100_000, 1.1, 0.001, 0.001)
        assert half_lots <= full_lots

    def test_volatility_scaling(self):
        """Higher vol → wider stop → fewer lots, without a separate scalar."""
        sizer = PositionSizer()
        lots_low_vol = sizer.compute(1.0, 100_000, 1.1, 0.0005, 0.001)
        lots_high_vol = sizer.compute(1.0, 100_000, 1.1, 0.002, 0.001)
        assert lots_low_vol >= lots_high_vol

    def test_min_lot_size(self):
        sizer = PositionSizer(min_lot_size=0.01)
        lots = sizer.compute(0.001, 100_000, 1.1, 0.001, 0.001)
        assert lots == 0.0 or lots >= 0.01

    def test_trade_stats_update(self):
        sizer = PositionSizer()
        sizer.update_trade_stats(500.0)
        sizer.update_trade_stats(-200.0)
        assert sizer._trade_wins == 1
        assert sizer._trade_losses == 1


class TestRiskPerTradeGovernsSize:
    """The size follows from what a stopped-out trade costs."""

    EQUITY = 100_000.0
    PRICE = 1.08
    ATR = 0.0010
    CONTRACT = 100_000.0

    @staticmethod
    def _sizer(**kwargs) -> PositionSizer:
        # Kelly is pinned at its warmed-up value so these tests measure sizing,
        # not the Kelly warm-up schedule.
        defaults = dict(risk_per_trade_pct=0.01, atr_stop_mult=2.0, kelly_fraction=1.0)
        defaults.update(kwargs)
        sizer = PositionSizer(**defaults)
        sizer._trade_wins, sizer._trade_losses = 60, 40
        sizer._avg_win, sizer._avg_loss = 0.02, 0.01
        return sizer

    def _risk_if_stopped(self, lots: float, atr: float | None = None) -> float:
        stop_distance = 2.0 * (self.ATR if atr is None else atr)
        return lots * self.CONTRACT * stop_distance

    def test_a_full_confidence_trade_risks_the_configured_fraction(self):
        """The headline number: 1% of equity, not 0.004%."""
        sizer = self._sizer()
        lots = sizer.compute(1.0, self.EQUITY, self.PRICE, self.ATR, self.ATR)
        kelly = sizer._compute_kelly()
        expected = self.EQUITY * 0.01 * kelly
        assert self._risk_if_stopped(lots) == pytest.approx(expected, rel=0.02)

    def test_risk_scales_with_confidence_not_with_price(self):
        sizer = self._sizer()
        cheap = sizer.compute(1.0, self.EQUITY, 0.65, self.ATR, self.ATR)
        dear = sizer.compute(1.0, self.EQUITY, 1.60, self.ATR, self.ATR)
        # Risk is set by the stop distance, which does not depend on the quote.
        assert cheap == pytest.approx(dear, rel=0.02)

    def test_a_wider_stop_buys_fewer_lots_for_the_same_risk(self):
        sizer = self._sizer()
        tight = sizer.compute(1.0, self.EQUITY, self.PRICE, 0.0005, self.ATR)
        wide = sizer.compute(1.0, self.EQUITY, self.PRICE, 0.0020, self.ATR)
        assert tight > wide
        assert self._risk_if_stopped(tight, 0.0005) == pytest.approx(
            self._risk_if_stopped(wide, 0.0020), rel=0.05,
        )

    def test_an_explicit_stop_distance_overrides_the_atr_default(self):
        """The risk manager computes a regime-aware stop; sizing must use the
        same one, or the position is sized against a stop nobody will place."""
        sizer = self._sizer()
        default = sizer.compute(1.0, self.EQUITY, self.PRICE, self.ATR, self.ATR)
        wider = sizer.compute(
            1.0, self.EQUITY, self.PRICE, self.ATR, self.ATR, stop_distance=0.0080,
        )
        assert wider < default

    def test_signals_that_used_to_round_to_zero_now_trade(self):
        """|action| = 0.43 was below the old rounding floor. 671 of 1155
        decisions were rejected this way."""
        sizer = self._sizer()
        assert sizer.compute(0.43, self.EQUITY, self.PRICE, self.ATR, self.ATR) > 0.0

    def test_exposure_clears_the_measurability_bar(self):
        """ComparisonResult calls anything under 0.5% of equity unmeasurable."""
        sizer = self._sizer()
        lots = sizer.compute(0.5, self.EQUITY, self.PRICE, self.ATR, self.ATR)
        notional = lots * self.PRICE * self.CONTRACT
        assert notional / self.EQUITY > 0.005


class TestLeverageCap:
    EQUITY = 100_000.0
    PRICE = 1.08

    def test_a_tight_stop_cannot_buy_unlimited_leverage(self):
        """Risk-based sizing divides by the stop; a near-zero stop would
        otherwise ask for an unbounded position."""
        sizer = PositionSizer(risk_per_trade_pct=0.01, max_leverage=5.0)
        lots = sizer.compute(1.0, self.EQUITY, self.PRICE, 1e-7, 1e-7)
        notional = lots * self.PRICE * 100_000
        assert notional <= self.EQUITY * 5.0 * 1.001

    def test_a_lower_cap_binds_harder(self):
        loose = PositionSizer(risk_per_trade_pct=0.01, max_leverage=20.0)
        tight = PositionSizer(risk_per_trade_pct=0.01, max_leverage=2.0)
        args = (1.0, self.EQUITY, self.PRICE, 1e-6, 1e-6)
        assert tight.compute(*args) < loose.compute(*args)


class TestFallbackWhenNoStopIsKnown:
    """Without ATR there is no stop distance, so risk-based sizing has no
    denominator. The old notional cap still applies there rather than
    guessing a stop."""

    def test_missing_atr_falls_back_to_the_notional_cap(self):
        sizer = PositionSizer(max_position_pct=0.10)
        lots = sizer.compute(1.0, 100_000, 1.1, None, None)
        assert 0.0 < lots <= 100_000 * 0.10 / (1.1 * 100_000) + 0.01

    def test_a_non_positive_atr_is_treated_as_unknown(self):
        sizer = PositionSizer(max_position_pct=0.10)
        assert sizer.compute(1.0, 100_000, 1.1, 0.0, 0.0) == pytest.approx(
            sizer.compute(1.0, 100_000, 1.1, None, None),
        )

    def test_a_worthless_lot_yields_no_position(self):
        sizer = PositionSizer()
        assert sizer.compute(1.0, 100_000, 0.0, 0.001, 0.001) == 0.0
