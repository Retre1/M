"""Phase 3 tests: Professional Trading Intelligence.

Tests for:
- CalendarProvider & FundamentalExtractor
- StructureExtractor (swing detection, structure breaks)
- Break-even stop logic
- Position layers (pyramiding)
- Expanded observation space
- News-aware gating (via HiveMind)
- Enhanced TradingReward
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Test: CalendarProvider
# ---------------------------------------------------------------------------

class TestCalendarProvider:
    def _make_provider(self):
        from apexfx.data.calendar_provider import CalendarEvent, CalendarProvider

        provider = CalendarProvider()
        events = [
            CalendarEvent(
                time_utc=datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc),
                currency="USD",
                name="Non-Farm Payrolls",
                impact="high",
                actual=216.0,
                forecast=170.0,
                previous=199.0,
            ),
            CalendarEvent(
                time_utc=datetime(2024, 1, 10, 13, 30, tzinfo=timezone.utc),
                currency="USD",
                name="CPI",
                impact="high",
                actual=3.4,
                forecast=3.2,
                previous=3.1,
            ),
            CalendarEvent(
                time_utc=datetime(2024, 1, 11, 8, 0, tzinfo=timezone.utc),
                currency="EUR",
                name="ECB Rate Decision",
                impact="high",
                actual=4.5,
                forecast=4.5,
                previous=4.5,
            ),
        ]
        provider.add_events(events)
        return provider

    def test_get_events_filters_by_currency(self):
        provider = self._make_provider()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)

        usd_events = provider.get_events(start, end, currencies=["USD"])
        assert len(usd_events) == 2
        assert all(e.currency == "USD" for e in usd_events)

    def test_surprise_scores_computed(self):
        provider = self._make_provider()
        # NFP had actual=216, forecast=170 → big positive surprise
        nfp = provider._events[0]
        assert nfp.surprise_score > 0  # positive surprise

    def test_next_event(self):
        provider = self._make_provider()
        as_of = datetime(2024, 1, 6, tzinfo=timezone.utc)
        nxt = provider.next_event(as_of, currencies=["USD"])
        assert nxt is not None
        assert "CPI" in nxt.name

    def test_is_hawkish(self):
        from apexfx.data.calendar_provider import CalendarEvent
        # Higher NFP = hawkish
        e = CalendarEvent(
            time_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
            currency="USD", name="Non-Farm Payrolls", impact="high",
            actual=300.0, forecast=200.0,
        )
        assert e.is_hawkish is True

        # Higher unemployment = dovish
        e2 = CalendarEvent(
            time_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
            currency="USD", name="Unemployment Rate", impact="high",
            actual=4.5, forecast=4.0,
        )
        assert e2.is_hawkish is False  # higher unemployment = dovish


# ---------------------------------------------------------------------------
# Test: FundamentalExtractor
# ---------------------------------------------------------------------------

class TestFundamentalExtractor:
    def _make_bars(self, n=100):
        dates = pd.date_range("2024-01-05 12:00", periods=n, freq="h", tz="UTC")
        np.random.seed(42)
        close = 1.1000 + np.cumsum(np.random.randn(n) * 0.001)
        return pd.DataFrame({
            "time": dates,
            "open": close - 0.0005,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        })

    def test_feature_count(self):
        from apexfx.features.fundamental import FundamentalExtractor
        ext = FundamentalExtractor()
        assert len(ext.feature_names) == 8

    def test_output_shape(self):
        from apexfx.data.calendar_provider import CalendarEvent
        from apexfx.features.fundamental import FundamentalExtractor

        bars = self._make_bars(100)
        ext = FundamentalExtractor(base_currency="EUR", quote_currency="USD")

        events = [
            CalendarEvent(
                time_utc=datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc),
                currency="USD", name="Non-Farm Payrolls", impact="high",
                actual=216.0, forecast=170.0,
            ),
        ]
        ext.set_events(events)
        result = ext.extract(bars)

        assert result.shape == (100, 8)
        assert list(result.columns) == ext.feature_names

    def test_surprise_score_nonzero_after_event(self):
        from apexfx.data.calendar_provider import CalendarEvent
        from apexfx.features.fundamental import FundamentalExtractor

        bars = self._make_bars(100)
        ext = FundamentalExtractor(base_currency="EUR", quote_currency="USD")
        # Need at least 3 events of same name for proper std computation
        events = [
            CalendarEvent(
                time_utc=datetime(2024, 1, 5, 13, 30, tzinfo=timezone.utc),
                currency="USD", name="CPI", impact="high",
                actual=3.5, forecast=3.0,
            ),
            CalendarEvent(
                time_utc=datetime(2023, 12, 12, 13, 30, tzinfo=timezone.utc),
                currency="USD", name="CPI", impact="high",
                actual=3.2, forecast=3.1,
            ),
            CalendarEvent(
                time_utc=datetime(2023, 11, 14, 13, 30, tzinfo=timezone.utc),
                currency="USD", name="CPI", impact="high",
                actual=3.3, forecast=3.2,
            ),
        ]
        ext.set_events(events)
        result = ext.extract(bars)

        # After the event (bar ~1.5h later), surprise score should be nonzero
        # Event at 13:30, bars start at 12:00, so bar 2 (14:00) should show it
        post_event_vals = result["news_surprise_score"].iloc[2:20]
        assert post_event_vals.abs().sum() > 0

    def test_no_events_returns_zeros(self):
        from apexfx.features.fundamental import FundamentalExtractor
        bars = self._make_bars(50)
        ext = FundamentalExtractor()
        result = ext.extract(bars)
        # rate_differential may be nonzero, but surprise/bias should be zero
        assert result["news_surprise_score"].abs().sum() == 0.0


# ---------------------------------------------------------------------------
# Test: StructureExtractor
# ---------------------------------------------------------------------------

class TestStructureExtractor:
    def _make_trending_bars(self, n=100):
        """Create uptrending data with clear swing highs/lows."""
        np.random.seed(42)
        # Uptrend with some pullbacks
        base = np.linspace(1.0, 1.05, n)
        noise = np.random.randn(n) * 0.002
        # Add some oscillation for swing points
        swing = 0.005 * np.sin(np.linspace(0, 8 * np.pi, n))
        close = base + noise + swing
        high = close + abs(np.random.randn(n) * 0.001)
        low = close - abs(np.random.randn(n) * 0.001)
        return pd.DataFrame({
            "open": close - 0.0003,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        })

    def test_feature_count(self):
        from apexfx.features.structure import StructureExtractor
        ext = StructureExtractor()
        assert len(ext.feature_names) == 8

    def test_output_shape(self):
        from apexfx.features.structure import StructureExtractor
        bars = self._make_trending_bars(200)
        ext = StructureExtractor(swing_period=5)
        result = ext.extract(bars)
        assert result.shape == (200, 8)

    def test_swing_detection(self):
        from apexfx.features.structure import StructureExtractor
        # Create data with obvious swing high at index 10
        n = 30
        high = np.ones(n) * 1.0
        high[10] = 1.05  # obvious swing high
        is_swing = StructureExtractor._detect_swing_highs(high, period=3)
        assert is_swing[10] is True or is_swing[10] == True  # noqa

    def test_structure_trend_bullish(self):
        from apexfx.features.structure import StructureExtractor
        ext = StructureExtractor()
        # HH + HL pattern
        recent_highs = [(10, 1.02), (20, 1.04)]  # higher high
        recent_lows = [(15, 1.00), (25, 1.01)]   # higher low
        trend = ext._compute_structure_trend(recent_highs, recent_lows)
        assert trend == 1.0  # bullish

    def test_structure_trend_bearish(self):
        from apexfx.features.structure import StructureExtractor
        ext = StructureExtractor()
        # LH + LL pattern
        recent_highs = [(10, 1.04), (20, 1.02)]  # lower high
        recent_lows = [(15, 1.01), (25, 0.99)]   # lower low
        trend = ext._compute_structure_trend(recent_highs, recent_lows)
        assert trend == -1.0  # bearish

    def test_breakout_strength_uses_volume(self):
        from apexfx.features.structure import StructureExtractor
        bars = self._make_trending_bars(200)
        # Set a volume spike at bar 100
        bars.loc[bars.index[100], "volume"] = 10000
        ext = StructureExtractor(swing_period=5)
        result = ext.extract(bars)
        # Bar 100 should have above-average breakout_strength
        if not np.isnan(result["breakout_strength"].iloc[100]):
            assert result["breakout_strength"].iloc[100] > 1.0


# ---------------------------------------------------------------------------
# Test: Break-Even Stop
# ---------------------------------------------------------------------------

class TestBreakEvenStop:
    def test_breakeven_activates_after_profit(self):
        from apexfx.env.forex_env import AdaptiveStopLoss

        stop = AdaptiveStopLoss(
            atr_multiplier=2.0, pip_value=0.0001,
            breakeven_atr_mult=1.5,
        )
        stop.set_entry(1.1000)

        atr = 0.0050  # 50 pips
        # Price moves up by 1.5 ATR = 0.0075 → trigger break-even
        stop.update(1.1000, 1, atr)  # initial
        assert stop.breakeven_activated is False

        stop.update(1.1080, 1, atr)  # +80 pips > 1.5 * 50 = 75 pips
        assert stop.breakeven_activated is True

    def test_breakeven_locks_stop_at_entry(self):
        from apexfx.env.forex_env import AdaptiveStopLoss

        stop = AdaptiveStopLoss(
            atr_multiplier=2.0, pip_value=0.0001,
            breakeven_atr_mult=1.0,
        )
        stop.set_entry(1.1000)

        atr = 0.0050
        # Move up to trigger break-even
        stop.update(1.1060, 1, atr)  # +60 pips > 1.0 * 50
        assert stop.breakeven_activated is True
        # Stop should be at or above entry
        assert stop.stop_price >= 1.1000

    def test_breakeven_short_position(self):
        from apexfx.env.forex_env import AdaptiveStopLoss

        stop = AdaptiveStopLoss(
            atr_multiplier=2.0, pip_value=0.0001,
            breakeven_atr_mult=1.0,
        )
        stop.set_entry(1.1000)

        atr = 0.0050
        # Short: price goes down
        stop.update(1.0940, -1, atr)  # 60 pips profit on short
        assert stop.breakeven_activated is True
        # Stop should be at or below entry
        assert stop.stop_price <= 1.1000

    def test_no_breakeven_before_threshold(self):
        from apexfx.env.forex_env import AdaptiveStopLoss

        stop = AdaptiveStopLoss(
            atr_multiplier=2.0, pip_value=0.0001,
            breakeven_atr_mult=2.0,
        )
        stop.set_entry(1.1000)

        atr = 0.0050
        # Only 50 pips profit, need 2.0 * 50 = 100
        stop.update(1.1050, 1, atr)
        assert stop.breakeven_activated is False


# ---------------------------------------------------------------------------
# Test: Position Layers
# ---------------------------------------------------------------------------

class TestPositionLayers:
    def _make_env(self):
        from apexfx.env.forex_env import ForexTradingEnv

        n = 500
        np.random.seed(42)
        close = 1.1 + np.cumsum(np.random.randn(n) * 0.0005)
        data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close - 0.0002,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        })
        env = ForexTradingEnv(
            data=data, episode_length=200, lookback=10,
            n_market_features=5, max_position_layers=3,
        )
        return env

    def test_initial_no_layers(self):
        env = self._make_env()
        env.reset()
        assert len(env._position_layers) == 0

    def test_open_creates_one_layer(self):
        env = self._make_env()
        env.reset()
        # Take a long action
        env.step(np.array([0.5], dtype=np.float32))
        env.step(np.array([0.5], dtype=np.float32))
        if env._position_direction != 0:
            assert len(env._position_layers) >= 1

    def test_close_clears_layers(self):
        env = self._make_env()
        env.reset()
        env.step(np.array([0.5], dtype=np.float32))
        env.step(np.array([0.5], dtype=np.float32))
        # Close
        env.step(np.array([0.0], dtype=np.float32))
        env.step(np.array([0.0], dtype=np.float32))
        assert len(env._position_layers) == 0


# ---------------------------------------------------------------------------
# Test: Expanded Observation Space
# ---------------------------------------------------------------------------

class TestExpandedObservation:
    def test_observation_has_new_keys(self):
        from apexfx.env.forex_env import ForexTradingEnv

        n = 500
        np.random.seed(42)
        close = 1.1 + np.cumsum(np.random.randn(n) * 0.0005)
        data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
            "open": close - 0.0002,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "volume": np.random.randint(100, 1000, n),
        })
        env = ForexTradingEnv(
            data=data, episode_length=200, lookback=10,
            n_market_features=5,
        )
        obs, _ = env.reset()

        assert "fundamental_features" in obs
        assert "structure_features" in obs
        assert obs["fundamental_features"].shape == (8,)
        assert obs["structure_features"].shape == (8,)
        assert obs["position_state"].shape == (8,)


# ---------------------------------------------------------------------------
# Test: HiveMind with new features
# ---------------------------------------------------------------------------

class TestHiveMindPhase3:
    def test_forward_with_fundamental_and_structure(self):
        from apexfx.models.ensemble.hive_mind import HiveMind

        batch, seq = 2, 10
        hm = HiveMind(
            n_continuous_vars=5,
            n_known_future_vars=3,
            d_model=16,
            d_trend_features=4,
            d_reversion_features=4,
            d_breakout_features=4,
            d_regime_features=6,
            d_fundamental_features=8,
            d_structure_features=8,
        )

        out = hm(
            market_features=torch.randn(batch, seq, 5),
            time_features=torch.randn(batch, seq, 3),
            trend_features=torch.randn(batch, 4),
            reversion_features=torch.randn(batch, 4),
            regime_features=torch.randn(batch, 6),
            breakout_features=torch.randn(batch, 4),
            fundamental_features=torch.randn(batch, 8),
            structure_features=torch.randn(batch, 8),
        )

        assert out.action.shape == (batch, 1)
        assert out.gating_weights.shape == (batch, 3)

    def test_forward_without_optional_features(self):
        from apexfx.models.ensemble.hive_mind import HiveMind

        batch, seq = 2, 10
        hm = HiveMind(
            n_continuous_vars=5,
            n_known_future_vars=3,
            d_model=16,
            d_trend_features=4,
            d_reversion_features=4,
            d_breakout_features=4,
            d_regime_features=6,
        )

        # Should work without fundamental/structure features (backward compat)
        out = hm(
            market_features=torch.randn(batch, seq, 5),
            time_features=torch.randn(batch, seq, 3),
            trend_features=torch.randn(batch, 4),
            reversion_features=torch.randn(batch, 4),
            regime_features=torch.randn(batch, 6),
        )

        assert out.action.shape == (batch, 1)

    def test_gradient_flows_through_fundamental(self):
        from apexfx.models.ensemble.hive_mind import HiveMind

        batch, seq = 2, 10
        hm = HiveMind(
            n_continuous_vars=5,
            n_known_future_vars=3,
            d_model=16,
            d_trend_features=4,
            d_reversion_features=4,
            d_breakout_features=4,
            d_regime_features=6,
            d_fundamental_features=8,
            d_structure_features=8,
        )

        fundamental = torch.randn(batch, 8, requires_grad=True)
        out = hm(
            market_features=torch.randn(batch, seq, 5),
            time_features=torch.randn(batch, seq, 3),
            trend_features=torch.randn(batch, 4),
            reversion_features=torch.randn(batch, 4),
            regime_features=torch.randn(batch, 6),
            fundamental_features=fundamental,
        )

        loss = out.action.sum()
        loss.backward()
        # Fundamental features should get gradients (flows through gating)
        assert fundamental.grad is not None
        assert fundamental.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Test: Enhanced TradingReward
# ---------------------------------------------------------------------------

class TestTradingRewardPhase3:
    def test_hold_winner_bonus(self):
        from apexfx.env.reward import TradingReward

        reward_fn = TradingReward(hold_winner_bonus=0.5)
        reward_fn._peak = 100000

        # Simulate holding a winning position for 10 bars
        reward_fn.set_trade_info(
            action=0.5, direction=1,
            unrealized_pnl=500, time_in_position=10,
        )

        r1 = reward_fn.compute(100100, 100000)

        # Compare with no winner bonus
        reward_fn2 = TradingReward(hold_winner_bonus=0.0)
        reward_fn2._peak = 100000
        reward_fn2.set_trade_info(
            action=0.5, direction=1,
            unrealized_pnl=500, time_in_position=10,
        )
        r2 = reward_fn2.compute(100100, 100000)

        assert r1 > r2  # winner bonus should increase reward

    def test_quick_cut_bonus(self):
        from apexfx.env.reward import TradingReward

        reward_fn = TradingReward(quick_cut_bonus=1.0)
        reward_fn._peak = 100000

        # Simulate: was losing, closed within 5 bars
        reward_fn.set_trade_info(action=0.5, direction=1, unrealized_pnl=-100, time_in_position=3)
        reward_fn.set_trade_info(action=0.0, direction=0, unrealized_pnl=0, time_in_position=3)

        r1 = reward_fn.compute(99950, 99960)

        reward_fn2 = TradingReward(quick_cut_bonus=0.0)
        reward_fn2._peak = 100000
        reward_fn2.set_trade_info(action=0.5, direction=1, unrealized_pnl=-100, time_in_position=3)
        reward_fn2.set_trade_info(action=0.0, direction=0, unrealized_pnl=0, time_in_position=3)

        r2 = reward_fn2.compute(99950, 99960)
        assert r1 > r2  # quick cut bonus should help

    def test_news_trade_penalty(self):
        from apexfx.env.reward import TradingReward

        reward_fn = TradingReward(news_trade_penalty=1.0)
        reward_fn._peak = 100000

        # Opening during news = penalty
        reward_fn.set_trade_info(
            action=0.5, direction=1, unrealized_pnl=0, time_in_position=0,
            news_active=True,
        )
        r1 = reward_fn.compute(100010, 100000)

        reward_fn2 = TradingReward(news_trade_penalty=0.0)
        reward_fn2._peak = 100000
        reward_fn2.set_trade_info(
            action=0.5, direction=1, unrealized_pnl=0, time_in_position=0,
            news_active=True,
        )
        r2 = reward_fn2.compute(100010, 100000)
        assert r1 < r2  # news penalty should decrease reward

    def test_structure_confirm_bonus(self):
        from apexfx.env.reward import TradingReward

        reward_fn = TradingReward(structure_confirm_bonus=1.0)
        reward_fn._peak = 100000

        # Entry aligned with structure break
        reward_fn.set_trade_info(
            action=0.5, direction=1, unrealized_pnl=0, time_in_position=0,
            structure_aligned=True,
        )
        r1 = reward_fn.compute(100010, 100000)

        reward_fn2 = TradingReward(structure_confirm_bonus=0.0)
        reward_fn2._peak = 100000
        reward_fn2.set_trade_info(
            action=0.5, direction=1, unrealized_pnl=0, time_in_position=0,
            structure_aligned=True,
        )
        r2 = reward_fn2.compute(100010, 100000)
        assert r1 > r2  # structure bonus should increase reward


# ---------------------------------------------------------------------------
# Test: Config Schema
# ---------------------------------------------------------------------------

class TestConfigSchema:
    def test_calendar_config_defaults(self):
        from apexfx.config.schema import CalendarConfig
        cfg = CalendarConfig()
        assert cfg.enabled is True
        assert "USD" in cfg.currencies

    def test_structure_config_defaults(self):
        from apexfx.config.schema import StructureConfig
        cfg = StructureConfig()
        assert cfg.swing_period == 5
        assert cfg.confluence_atr_mult == 1.0

    def test_position_management_config(self):
        from apexfx.config.schema import PositionManagementConfig
        cfg = PositionManagementConfig()
        assert cfg.max_layers == 3
        assert cfg.breakeven_atr_mult == 1.5

    def test_data_config_has_calendar(self):
        from apexfx.config.schema import DataConfig
        cfg = DataConfig()
        assert hasattr(cfg, "calendar")
        assert cfg.calendar.enabled is True

    def test_model_config_has_structure(self):
        from apexfx.config.schema import ModelConfig
        cfg = ModelConfig()
        assert hasattr(cfg, "structure")
        assert hasattr(cfg, "position_management")
