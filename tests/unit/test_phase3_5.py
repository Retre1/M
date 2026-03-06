"""Phase 3.5 tests: Calendar API + Rule-Based Trade Filters.

Tests for:
- CalendarFetcher XML/HTML parsing
- Numeric value parsing (K, M, %)
- ET→UTC time conversion
- StrategyFilter (all 6+ rules)
- TradeFilterWrapper (gym wrapper)
- Config schema additions
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test: CalendarFetcher XML parsing
# ---------------------------------------------------------------------------


class TestCalendarFetcherXML:
    """Test XML feed parsing for Forex Factory calendar."""

    _SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<weeklyevents>
  <event>
    <title>Non-Farm Employment Change</title>
    <country>USD</country>
    <date>01-05-2024</date>
    <time>8:30am</time>
    <impact>High</impact>
    <forecast>170K</forecast>
    <previous>199K</previous>
  </event>
  <event>
    <title>CPI y/y</title>
    <country>USD</country>
    <date>01-10-2024</date>
    <time>8:30am</time>
    <impact>High</impact>
    <forecast>3.2%</forecast>
    <previous>3.1%</previous>
    <actual>3.4%</actual>
  </event>
  <event>
    <title>Trade Balance</title>
    <country>EUR</country>
    <date>01-08-2024</date>
    <time>5:00am</time>
    <impact>Medium</impact>
    <forecast>12.3B</forecast>
    <previous>11.1B</previous>
  </event>
  <event>
    <title>Bank Holiday</title>
    <country>JPY</country>
    <date>01-08-2024</date>
    <time>All Day</time>
    <impact>Holiday</impact>
  </event>
</weeklyevents>"""

    def test_parse_xml_event_count(self):
        from apexfx.data.calendar_fetcher import CalendarFetcher

        fetcher = CalendarFetcher()
        events = fetcher.parse_xml(self._SAMPLE_XML)
        assert len(events) == 4

    def test_parse_xml_currencies(self):
        from apexfx.data.calendar_fetcher import CalendarFetcher

        fetcher = CalendarFetcher()
        events = fetcher.parse_xml(self._SAMPLE_XML)
        currencies = {e.currency for e in events}
        assert "USD" in currencies
        assert "EUR" in currencies

    def test_parse_xml_impact_normalized(self):
        from apexfx.data.calendar_fetcher import CalendarFetcher

        fetcher = CalendarFetcher()
        events = fetcher.parse_xml(self._SAMPLE_XML)
        nfp = next(e for e in events if "Non-Farm" in e.name)
        assert nfp.impact == "high"

        trade_bal = next(e for e in events if "Trade Balance" in e.name)
        assert trade_bal.impact == "medium"

    def test_parse_xml_time_utc_conversion(self):
        from apexfx.data.calendar_fetcher import CalendarFetcher

        fetcher = CalendarFetcher()
        events = fetcher.parse_xml(self._SAMPLE_XML)
        nfp = next(e for e in events if "Non-Farm" in e.name)
        # 8:30 AM Eastern in January (EST = UTC-5) → 13:30 UTC
        assert nfp.time_utc.hour == 13
        assert nfp.time_utc.minute == 30
        assert nfp.time_utc.tzinfo == timezone.utc


class TestNumericParsing:
    """Test parsing of Forex Factory numeric values."""

    def test_parse_k_suffix(self):
        from apexfx.data.calendar_fetcher import _parse_ff_numeric

        assert _parse_ff_numeric("216K") == 216.0

    def test_parse_percent(self):
        from apexfx.data.calendar_fetcher import _parse_ff_numeric

        assert _parse_ff_numeric("3.4%") == 3.4

    def test_parse_negative(self):
        from apexfx.data.calendar_fetcher import _parse_ff_numeric

        assert _parse_ff_numeric("-0.5%") == -0.5

    def test_parse_billion(self):
        from apexfx.data.calendar_fetcher import _parse_ff_numeric

        assert _parse_ff_numeric("12.3B") == 12_300_000.0

    def test_parse_none(self):
        from apexfx.data.calendar_fetcher import _parse_ff_numeric

        assert _parse_ff_numeric(None) is None
        assert _parse_ff_numeric("") is None
        assert _parse_ff_numeric("-") is None
        assert _parse_ff_numeric("N/A") is None

    def test_parse_plain_number(self):
        from apexfx.data.calendar_fetcher import _parse_ff_numeric

        assert _parse_ff_numeric("42.5") == 42.5


class TestETtoUTC:
    """Test Eastern Time to UTC conversion."""

    def test_winter_time(self):
        from apexfx.data.calendar_fetcher import _et_to_utc

        # January (EST = UTC-5): 8:30 ET → 13:30 UTC
        et = datetime(2024, 1, 5, 8, 30)
        utc = _et_to_utc(et)
        assert utc.hour == 13
        assert utc.minute == 30

    def test_summer_time(self):
        from apexfx.data.calendar_fetcher import _et_to_utc

        # July (EDT = UTC-4): 8:30 ET → 12:30 UTC
        et = datetime(2024, 7, 5, 8, 30)
        utc = _et_to_utc(et)
        assert utc.hour == 12
        assert utc.minute == 30

    def test_utc_timezone(self):
        from apexfx.data.calendar_fetcher import _et_to_utc

        et = datetime(2024, 1, 5, 8, 30)
        utc = _et_to_utc(et)
        assert utc.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# Test: StrategyFilter
# ---------------------------------------------------------------------------


class TestStrategyFilter:
    """Test rule-based strategy filter."""

    def _make_obs(
        self,
        news_active: float = 0.0,
        time_to_event: float = 1.0,
        fundamental_bias: float = 0.0,
        conflicting: float = 0.0,
        break_bull: float = 0.0,
        break_bear: float = 0.0,
    ) -> dict[str, np.ndarray]:
        fund = np.zeros(8, dtype=np.float32)
        fund[1] = news_active
        fund[2] = time_to_event
        fund[3] = fundamental_bias
        fund[7] = conflicting

        struct = np.zeros(8, dtype=np.float32)
        struct[2] = break_bull
        struct[3] = break_bear

        return {
            "fundamental_features": fund,
            "structure_features": struct,
            "position_state": np.zeros(8, dtype=np.float32),
        }

    def test_news_blackout_blocks_entry(self):
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter()
        obs = self._make_obs(news_active=1.0, break_bull=1.0, fundamental_bias=1.0)
        decision = sf.check(obs, proposed_action=0.5, current_position=0.0)
        assert not decision.allowed
        assert "news_blackout" in decision.reason

    def test_news_blackout_allows_exit(self):
        """During news, closing a position should be allowed."""
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter()
        obs = self._make_obs(news_active=1.0)
        # Action=0 is closing/staying flat — should be allowed
        decision = sf.check(obs, proposed_action=0.0, current_position=0.5)
        assert decision.allowed

    def test_conflicting_signals_force_close(self):
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter(exit_on_conflict=True)
        obs = self._make_obs(conflicting=1.0)
        decision = sf.check(obs, proposed_action=0.5, current_position=0.3)
        assert decision.force_close
        assert "conflicting_signals" in decision.reason

    def test_minimum_bias_blocks_entry(self):
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter(min_fundamental_bias=0.3)
        obs = self._make_obs(fundamental_bias=0.1, break_bull=1.0)
        decision = sf.check(obs, proposed_action=0.5, current_position=0.0)
        assert not decision.allowed
        assert "insufficient_bias" in decision.reason

    def test_structure_confirm_blocks_without_bos(self):
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter(require_structure_confirm=True, min_fundamental_bias=0.0)
        obs = self._make_obs(fundamental_bias=1.0, break_bull=0.0, break_bear=0.0)
        decision = sf.check(obs, proposed_action=0.5, current_position=0.0)
        assert not decision.allowed
        assert "no_structure_confirm" in decision.reason

    def test_direction_alignment_blocks_against_bias(self):
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter(
            block_against_bias=True,
            min_bias_for_direction=0.5,
            require_structure_confirm=False,
            min_fundamental_bias=0.0,
        )
        # Bias is positive (bullish) but action is sell
        obs = self._make_obs(fundamental_bias=1.0)
        decision = sf.check(obs, proposed_action=-0.5, current_position=0.0)
        assert not decision.allowed
        assert "against_fundamental_bias" in decision.reason

    def test_pre_news_scaling(self):
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter(
            pre_news_time_threshold=0.1,
            reduce_scale_pre_news=0.5,
        )
        obs = self._make_obs(time_to_event=0.05)  # Event approaching
        decision = sf.check(obs, proposed_action=0.0, current_position=0.5)
        assert decision.allowed
        assert decision.scale == 0.5
        assert "pre_news" in decision.reason

    def test_all_rules_pass(self):
        """When conditions are met, filter allows the trade."""
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter(
            min_fundamental_bias=0.3,
            require_structure_confirm=True,
        )
        obs = self._make_obs(
            fundamental_bias=1.0,
            break_bull=1.0,
            time_to_event=0.5,
        )
        decision = sf.check(obs, proposed_action=0.5, current_position=0.0)
        assert decision.allowed
        assert decision.scale == 1.0

    def test_disabled_filter_allows_all(self):
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter(enabled=False)
        obs = self._make_obs(news_active=1.0, conflicting=1.0)
        decision = sf.check(obs, proposed_action=0.5, current_position=0.5)
        assert decision.allowed
        assert decision.scale == 1.0

    def test_imminent_event_blocks_entry(self):
        from apexfx.env.trade_filter import StrategyFilter

        sf = StrategyFilter(
            time_to_event_threshold=0.01,
            require_structure_confirm=False,
            min_fundamental_bias=0.0,
        )
        obs = self._make_obs(time_to_event=0.005)  # Event very close
        decision = sf.check(obs, proposed_action=0.5, current_position=0.0)
        assert not decision.allowed
        assert "event_imminent" in decision.reason


# ---------------------------------------------------------------------------
# Test: TradeFilterWrapper
# ---------------------------------------------------------------------------


class TestTradeFilterWrapper:
    """Test gym wrapper for strategy filter."""

    def _make_mock_env(self):
        """Create a minimal mock environment."""
        import gymnasium as gym
        from gymnasium import spaces

        class MockEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = spaces.Dict({
                    "fundamental_features": spaces.Box(-10, 10, (8,)),
                    "structure_features": spaces.Box(-10, 10, (8,)),
                    "position_state": spaces.Box(-10, 10, (8,)),
                    "market_features": spaces.Box(-10, 10, (5,)),
                })
                self.action_space = spaces.Box(-1, 1, (1,))
                self._step_count = 0
                self._last_action = None

            def reset(self, **kwargs):
                self._step_count = 0
                obs = {
                    "fundamental_features": np.zeros(8, dtype=np.float32),
                    "structure_features": np.zeros(8, dtype=np.float32),
                    "position_state": np.zeros(8, dtype=np.float32),
                    "market_features": np.zeros(5, dtype=np.float32),
                }
                return obs, {}

            def step(self, action):
                self._step_count += 1
                self._last_action = float(action) if np.isscalar(action) else float(action[0])
                obs = {
                    "fundamental_features": np.zeros(8, dtype=np.float32),
                    "structure_features": np.zeros(8, dtype=np.float32),
                    "position_state": np.zeros(8, dtype=np.float32),
                    "market_features": np.zeros(5, dtype=np.float32),
                }
                return obs, 0.0, self._step_count >= 10, False, {}

        return MockEnv()

    def test_wrapper_passes_through_when_no_filters_triggered(self):
        from apexfx.env.trade_filter import StrategyFilter
        from apexfx.env.wrappers import TradeFilterWrapper

        env = self._make_mock_env()
        sf = StrategyFilter(enabled=False)
        wrapped = TradeFilterWrapper(env, sf)

        obs, info = wrapped.reset()
        obs, reward, terminated, truncated, info = wrapped.step(np.float32(0.5))
        # Should pass through since filter is disabled
        assert env._last_action == pytest.approx(0.5, abs=0.01)

    def test_wrapper_blocks_action_during_news(self):
        from apexfx.env.trade_filter import StrategyFilter
        from apexfx.env.wrappers import TradeFilterWrapper

        env = self._make_mock_env()
        sf = StrategyFilter(
            require_structure_confirm=False,
            min_fundamental_bias=0.0,
        )
        wrapped = TradeFilterWrapper(env, sf)

        obs, info = wrapped.reset()

        # Manually set observation to have news_active=1
        obs["fundamental_features"][1] = 1.0  # news_impact_active
        obs["fundamental_features"][3] = 1.0  # fundamental_bias
        wrapped._last_obs = obs

        obs, reward, terminated, truncated, info = wrapped.step(np.float32(0.8))
        # Action should be blocked (set to 0 since no position)
        assert env._last_action == pytest.approx(0.0, abs=0.01)
        assert "filter_reason" in info
        assert "news_blackout" in info["filter_reason"]

    def test_wrapper_tracks_stats(self):
        from apexfx.env.trade_filter import StrategyFilter
        from apexfx.env.wrappers import TradeFilterWrapper

        env = self._make_mock_env()
        sf = StrategyFilter(enabled=False)
        wrapped = TradeFilterWrapper(env, sf)

        wrapped.reset()
        wrapped.step(np.float32(0.5))
        wrapped.step(np.float32(-0.5))

        stats = wrapped.filter_stats
        assert stats["total_actions"] == 2


# ---------------------------------------------------------------------------
# Test: Config Schema
# ---------------------------------------------------------------------------


class TestConfigSchemaPhase35:
    """Test new config additions."""

    def test_calendar_config_has_auto_fetch(self):
        from apexfx.config.schema import CalendarConfig

        cfg = CalendarConfig()
        assert cfg.auto_fetch is True
        assert cfg.fetch_interval_hours == 1
        assert cfg.source == "forex_factory"

    def test_strategy_filter_config_defaults(self):
        from apexfx.config.schema import StrategyFilterConfig

        cfg = StrategyFilterConfig()
        assert cfg.enabled is True
        assert cfg.min_fundamental_bias == 0.3
        assert cfg.require_structure_confirm is True
        assert cfg.exit_on_conflict is True
        assert cfg.block_against_bias is True

    def test_risk_config_has_strategy_filter(self):
        from apexfx.config.schema import RiskConfig

        cfg = RiskConfig()
        assert hasattr(cfg, "strategy_filter")
        assert cfg.strategy_filter.enabled is True

    def test_strategy_filter_config_custom(self):
        from apexfx.config.schema import StrategyFilterConfig

        cfg = StrategyFilterConfig(
            min_fundamental_bias=0.5,
            require_structure_confirm=False,
            news_blackout_threshold=0.8,
        )
        assert cfg.min_fundamental_bias == 0.5
        assert cfg.require_structure_confirm is False
        assert cfg.news_blackout_threshold == 0.8


# ---------------------------------------------------------------------------
# Test: CSV Export
# ---------------------------------------------------------------------------


class TestCSVExport:
    """Test CalendarFetcher CSV save/load roundtrip."""

    def test_save_csv_roundtrip(self, tmp_path):
        from apexfx.data.calendar_fetcher import CalendarFetcher
        from apexfx.data.calendar_provider import CalendarEvent, CalendarProvider

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
        ]

        csv_path = tmp_path / "test_calendar.csv"
        fetcher = CalendarFetcher()
        fetcher.save_csv(events, csv_path)

        # Load back with CalendarProvider
        provider = CalendarProvider()
        loaded = provider.load(str(csv_path))
        assert len(loaded) == 2
        assert loaded[0].name == "Non-Farm Payrolls"
        assert loaded[0].currency == "USD"
        assert loaded[0].actual == 216.0
        assert loaded[0].forecast == 170.0
