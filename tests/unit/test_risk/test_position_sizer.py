"""Tests for position sizer."""


from apexfx.risk.position_sizer import PositionSizer


class TestPositionSizer:
    def test_zero_action(self):
        sizer = PositionSizer()
        lots = sizer.compute(0.0, 100_000, 1.1, 0.001, 0.001)
        assert lots == 0.0

    def test_max_action(self):
        sizer = PositionSizer(max_position_pct=0.10)
        lots = sizer.compute(1.0, 100_000, 1.1, 0.001, 0.001)
        max_lots = 100_000 * 0.10 / (1.1 * 100_000)
        assert lots <= max_lots + 0.01  # rounding tolerance

    def test_half_action(self):
        sizer = PositionSizer(max_position_pct=0.10)
        full_lots = sizer.compute(1.0, 100_000, 1.1, 0.001, 0.001)
        half_lots = sizer.compute(0.5, 100_000, 1.1, 0.001, 0.001)
        assert half_lots <= full_lots

    def test_volatility_scaling(self):
        sizer = PositionSizer()
        # Higher current vol → smaller position
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
