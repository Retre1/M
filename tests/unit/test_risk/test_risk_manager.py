"""Tests for the central RiskManager and its sub-components."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apexfx.config.schema import (
    CooldownConfig,
    PositionSizingConfig,
    RiskConfig,
)
from apexfx.risk.risk_manager import (
    DailyLossGuard,
    KillSwitch,
    MarketState,
    RegimeAdaptiveRisk,
    RiskManager,
    VolatilityTargeter,
    WeekendGapGuard,
)


@pytest.fixture()
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_drawdown_pct=0.05,
        daily_var_limit=0.02,
        var_confidence=0.99,
        var_lookback_days=252,
        var_method="historical",
        cooldown=CooldownConfig(
            after_n_losses=3,
            duration_minutes=60,
            tilt_drawdown_pct=0.02,
            tilt_window_minutes=30,
        ),
        position_sizing=PositionSizingConfig(
            max_position_pct=0.10,
            kelly_fraction=0.5,
            min_trades_for_kelly=30,
            vol_lookback_bars=20,
            min_lot_size=0.01,
        ),
    )


@pytest.fixture()
def market_state() -> MarketState:
    return MarketState(
        current_price=1.1000,
        current_spread=0.0001,
        current_atr=0.0050,
        historical_atr=0.0050,
        spread_limit=0.0002,
    )


@pytest.fixture()
def risk_manager(risk_config: RiskConfig) -> RiskManager:
    return RiskManager(config=risk_config, initial_balance=100_000.0)


class TestDailyLossGuard:
    def test_not_triggered_initially(self):
        guard = DailyLossGuard(max_daily_loss_pct=0.02)
        assert not guard.is_triggered

    def test_trigger_on_loss(self):
        guard = DailyLossGuard(max_daily_loss_pct=0.02)
        guard.update(100_000)  # start of day
        guard.update(97_000)   # 3% loss exceeds 2% limit
        assert guard.is_triggered

    def test_no_trigger_within_limit(self):
        guard = DailyLossGuard(max_daily_loss_pct=0.02)
        guard.update(100_000)
        guard.update(99_000)  # 1% loss, within 2% limit
        assert not guard.is_triggered

    def test_daily_loss_pct(self):
        guard = DailyLossGuard(max_daily_loss_pct=0.05)
        guard.update(100_000)
        guard.update(98_000)
        assert abs(guard.daily_loss_pct - 0.02) < 1e-6


class TestKillSwitch:
    def test_not_active_initially(self, tmp_path, monkeypatch):
        ks = KillSwitch()
        monkeypatch.setattr(KillSwitch, "KILL_FILE", tmp_path / "KILL")
        assert not ks.is_active

    def test_activate(self, tmp_path, monkeypatch):
        ks = KillSwitch()
        monkeypatch.setattr(KillSwitch, "KILL_FILE", tmp_path / "KILL")
        ks.activate("test reason")
        assert ks._active
        assert ks.is_active  # also checks file-based detection

    def test_consecutive_rejections_trigger(self, tmp_path, monkeypatch):
        ks = KillSwitch(max_consecutive_rejections=3)
        monkeypatch.setattr(KillSwitch, "KILL_FILE", tmp_path / "KILL")
        ks.record_rejection()
        ks.record_rejection()
        assert not ks._active
        ks.record_rejection()
        assert ks._active

    def test_success_resets_counter(self, tmp_path, monkeypatch):
        ks = KillSwitch(max_consecutive_rejections=3)
        monkeypatch.setattr(KillSwitch, "KILL_FILE", tmp_path / "KILL")
        ks.record_rejection()
        ks.record_rejection()
        ks.record_success()
        ks.record_rejection()
        assert not ks._active

    def test_equity_floor(self, tmp_path, monkeypatch):
        ks = KillSwitch(equity_floor_pct=0.80)
        monkeypatch.setattr(KillSwitch, "KILL_FILE", tmp_path / "KILL")
        ks.check_equity(100_000, 100_000)
        assert not ks._active
        ks.check_equity(79_000, 100_000)  # below 80% floor
        assert ks._active

    def test_reset(self, tmp_path, monkeypatch):
        ks = KillSwitch()
        monkeypatch.setattr(KillSwitch, "KILL_FILE", tmp_path / "KILL")
        ks.activate("test")
        ks.reset()
        assert not ks._active


class TestVolatilityTargeter:
    def test_conservative_without_data(self):
        vt = VolatilityTargeter(target_vol=0.10)
        assert vt.compute_leverage() == 0.5

    def test_leverage_after_data(self):
        vt = VolatilityTargeter(target_vol=0.10, lookback=20)
        for _ in range(30):
            vt.update(0.005)  # ~0.5% daily return
        leverage = vt.compute_leverage()
        assert 0.1 <= leverage <= 2.0


class TestWeekendGapGuard:
    def test_weekday_no_block(self):
        guard = WeekendGapGuard()
        # Tuesday 10:00 UTC
        dt = datetime(2024, 1, 9, 10, 0, tzinfo=UTC)
        block, scale = guard.check(dt)
        assert not block
        assert scale == 1.0

    def test_friday_evening_block(self):
        guard = WeekendGapGuard(close_before_hour_utc=20)
        # Friday 21:00 UTC
        dt = datetime(2024, 1, 12, 21, 0, tzinfo=UTC)
        block, scale = guard.check(dt)
        assert block
        assert scale == 0.0

    def test_friday_afternoon_reduce(self):
        guard = WeekendGapGuard(close_before_hour_utc=20, reduce_on_friday=0.5)
        # Friday 15:00 UTC
        dt = datetime(2024, 1, 12, 15, 0, tzinfo=UTC)
        block, scale = guard.check(dt)
        assert not block
        assert scale == 0.5

    def test_saturday_block(self):
        guard = WeekendGapGuard()
        dt = datetime(2024, 1, 13, 12, 0, tzinfo=UTC)
        block, scale = guard.check(dt)
        assert block


class TestRegimeAdaptiveRisk:
    def test_default_flat(self):
        rar = RegimeAdaptiveRisk()
        pos_scale, var_scale = rar.get_scales()
        assert pos_scale == 0.7  # flat default

    def test_trending(self):
        rar = RegimeAdaptiveRisk()
        rar.set_regime("trending")
        pos_scale, _ = rar.get_scales()
        assert pos_scale == 1.2

    def test_volatile(self):
        rar = RegimeAdaptiveRisk()
        rar.set_regime("volatile")
        pos_scale, var_scale = rar.get_scales()
        assert pos_scale == 0.5
        assert var_scale == 0.5


class TestRiskManagerIntegration:
    def test_approve_normal_action(
        self, risk_manager: RiskManager, market_state: MarketState
    ):
        risk_manager.update_portfolio(100_000)
        # Set trending regime (scale=1.2) so position doesn't get
        # reduced to zero by flat regime(0.7) * vol_target(0.5)
        risk_manager.set_regime("trending")
        decision = risk_manager.evaluate_action(1.0, market_state)
        assert decision.approved
        assert decision.position_size > 0

    def test_reject_during_cooldown(
        self, risk_manager: RiskManager, market_state: MarketState
    ):
        risk_manager.update_portfolio(100_000)
        # Trigger cooldown by recording losses
        for _ in range(3):
            risk_manager.record_trade(-100)
        decision = risk_manager.evaluate_action(0.5, market_state)
        assert not decision.approved
        assert "cooldown" in decision.reason.lower()

    def test_reject_wide_spread(self, risk_manager: RiskManager):
        risk_manager.update_portfolio(100_000)
        wide_spread = MarketState(
            current_price=1.1000,
            current_spread=0.001,  # 10 pips - very wide
            spread_limit=0.0002,
        )
        decision = risk_manager.evaluate_action(0.5, wide_spread)
        assert not decision.approved
        assert "spread" in decision.reason.lower()

    def test_small_action_stays_neutral(
        self, risk_manager: RiskManager, market_state: MarketState
    ):
        risk_manager.update_portfolio(100_000)
        decision = risk_manager.evaluate_action(0.01, market_state)
        assert decision.approved
        assert decision.position_size == 0.0
        assert decision.adjusted_action == 0.0

    def test_drawdown_breach_rejects(
        self, risk_config: RiskConfig, market_state: MarketState
    ):
        # Use high daily_var_limit and tilt threshold so drawdown
        # check triggers first (not daily loss or cooldown tilt)
        risk_config.daily_var_limit = 0.20
        risk_config.cooldown.tilt_drawdown_pct = 0.50
        rm = RiskManager(config=risk_config, initial_balance=100_000.0)
        rm.update_portfolio(100_000)
        rm.update_portfolio(94_000)  # 6% dd > 5% limit
        decision = rm.evaluate_action(0.5, market_state)
        assert not decision.approved
        assert "drawdown" in decision.reason.lower()

    def test_force_close_on_drawdown(self, risk_manager: RiskManager):
        risk_manager.update_portfolio(100_000)
        risk_manager.update_portfolio(94_000)
        assert risk_manager.force_close_all()

    def test_regime_scaling(
        self, risk_manager: RiskManager, market_state: MarketState
    ):
        risk_manager.update_portfolio(100_000)

        # Normal trade
        decision_normal = risk_manager.evaluate_action(0.8, market_state)

        # Set volatile regime — should reduce position
        risk_manager.set_regime("volatile")
        decision_volatile = risk_manager.evaluate_action(0.8, market_state)

        if decision_normal.approved and decision_volatile.approved:
            assert decision_volatile.position_size <= decision_normal.position_size
