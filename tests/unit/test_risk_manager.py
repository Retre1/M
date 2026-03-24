"""Unit tests for KillSwitch, DailyLossGuard, and RiskManager.evaluate_action."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apexfx.config.schema import (
    CooldownConfig,
    PositionSizingConfig,
    RiskConfig,
    StressTestConfig,
    StrategyFilterConfig,
)
from apexfx.risk.risk_manager import (
    DailyLossGuard,
    KillSwitch,
    MarketState,
    RiskDecision,
    RiskManager,
    VolatilityTargeter,
    WeekendGapGuard,
)


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    @pytest.fixture(autouse=True)
    def _use_tmp_kill_file(self, tmp_path: Path):
        """Patch KILL_FILE to a temp location for test isolation."""
        self._kill_file = tmp_path / "KILL_SWITCH"
        with patch.object(KillSwitch, "KILL_FILE", self._kill_file):
            yield

    def test_not_active_by_default(self):
        ks = KillSwitch()
        assert not ks.is_active
        assert ks.reason == ""

    def test_activate_sets_active(self):
        ks = KillSwitch()
        ks.activate("test reason")
        assert ks.is_active
        assert ks.reason == "test reason"
        # Kill file should be created
        assert self._kill_file.exists()

    def test_manual_kill_file_activates(self):
        ks = KillSwitch()
        self._kill_file.parent.mkdir(parents=True, exist_ok=True)
        self._kill_file.write_text("manual kill")
        assert ks.is_active
        assert "Manual kill switch" in ks.reason

    def test_record_rejection_triggers_after_n(self):
        ks = KillSwitch(max_consecutive_rejections=3)
        ks.record_rejection()
        ks.record_rejection()
        assert not ks._active  # Check internal, not property (avoids file check)
        ks.record_rejection()
        assert ks._active

    def test_record_success_resets_rejection_counter(self):
        ks = KillSwitch(max_consecutive_rejections=3)
        ks.record_rejection()
        ks.record_rejection()
        ks.record_success()
        ks.record_rejection()
        assert not ks._active

    def test_check_equity_triggers_below_floor(self):
        ks = KillSwitch(equity_floor_pct=0.80)
        ks.check_equity(equity=75_000.0, initial_balance=100_000.0)
        assert ks._active

    def test_check_equity_ok_above_floor(self):
        ks = KillSwitch(equity_floor_pct=0.80)
        ks.check_equity(equity=85_000.0, initial_balance=100_000.0)
        assert not ks._active

    def test_reset_clears_state_and_file(self):
        ks = KillSwitch()
        ks.activate("testing reset")
        assert ks._active
        assert self._kill_file.exists()
        ks.reset()
        assert not ks._active
        assert ks.reason == ""
        assert not self._kill_file.exists()


# ---------------------------------------------------------------------------
# DailyLossGuard
# ---------------------------------------------------------------------------


class TestDailyLossGuard:
    def test_initially_not_triggered(self):
        dlg = DailyLossGuard(max_daily_loss_pct=0.02)
        assert not dlg.is_triggered
        assert dlg.daily_loss_pct == 0.0

    def test_update_sets_day_start_on_first_call(self):
        dlg = DailyLossGuard(max_daily_loss_pct=0.02)
        result = dlg.update(100_000.0)
        assert result is True  # Trading allowed
        assert dlg._day_start_equity == 100_000.0

    def test_triggers_when_loss_exceeds_limit(self):
        dlg = DailyLossGuard(max_daily_loss_pct=0.02)
        dlg.update(100_000.0)
        result = dlg.update(97_000.0)  # 3% loss
        assert result is False
        assert dlg.is_triggered

    def test_does_not_trigger_below_limit(self):
        dlg = DailyLossGuard(max_daily_loss_pct=0.02)
        dlg.update(100_000.0)
        result = dlg.update(99_000.0)  # 1% loss
        assert result is True
        assert not dlg.is_triggered

    def test_daily_loss_pct_property(self):
        dlg = DailyLossGuard(max_daily_loss_pct=0.05)
        dlg.update(100_000.0)
        dlg.update(98_000.0)
        assert abs(dlg.daily_loss_pct - 0.02) < 1e-10

    def test_division_by_zero_day_start_none(self):
        dlg = DailyLossGuard()
        assert dlg.daily_loss_pct == 0.0

    def test_division_by_zero_day_start_zero(self):
        dlg = DailyLossGuard()
        dlg._day_start_equity = 0.0
        assert dlg.daily_loss_pct == 0.0

    def test_new_day_resets(self):
        dlg = DailyLossGuard(max_daily_loss_pct=0.02)
        dlg.update(100_000.0)
        dlg.update(97_000.0)  # triggers
        assert dlg.is_triggered

        # Simulate new day by changing internal date
        dlg._current_date = None
        result = dlg.update(97_000.0)
        # New day — day_start is set to 97k, no loss yet
        assert result is True
        assert not dlg.is_triggered


# ---------------------------------------------------------------------------
# VolatilityTargeter
# ---------------------------------------------------------------------------


class TestVolatilityTargeter:
    def test_conservative_with_insufficient_data(self):
        vt = VolatilityTargeter(target_vol=0.10, lookback=60)
        assert vt.compute_leverage() == 0.5

    def test_leverage_scales_to_target(self):
        vt = VolatilityTargeter(target_vol=0.10, lookback=60, annualization=252)
        rng = np.random.default_rng(42)
        for _ in range(100):
            vt.update(rng.normal(0, 0.005))
        lev = vt.compute_leverage()
        assert 0.1 <= lev <= 2.0


# ---------------------------------------------------------------------------
# WeekendGapGuard
# ---------------------------------------------------------------------------


class TestWeekendGapGuard:
    def test_midweek_no_block(self):
        wg = WeekendGapGuard()
        # Wednesday 10:00 UTC
        dt = datetime(2024, 1, 3, 10, 0, tzinfo=UTC)
        block, scale = wg.check(utc_now=dt)
        assert not block
        assert scale == 1.0

    def test_friday_late_blocks(self):
        wg = WeekendGapGuard(close_before_hour_utc=20)
        # Friday 21:00 UTC
        dt = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
        block, scale = wg.check(utc_now=dt)
        assert block
        assert scale == 0.0

    def test_saturday_blocks(self):
        wg = WeekendGapGuard()
        # Saturday
        dt = datetime(2024, 1, 6, 12, 0, tzinfo=UTC)
        block, scale = wg.check(utc_now=dt)
        assert block

    def test_friday_afternoon_reduces(self):
        wg = WeekendGapGuard(close_before_hour_utc=20, reduce_on_friday=0.5)
        # Friday 15:00 UTC (afternoon but before close hour)
        dt = datetime(2024, 1, 5, 15, 0, tzinfo=UTC)
        block, scale = wg.check(utc_now=dt)
        assert not block
        assert scale == 0.5


# ---------------------------------------------------------------------------
# RiskManager — helper to build minimal config
# ---------------------------------------------------------------------------


def _make_risk_config(**overrides) -> RiskConfig:
    """Build a minimal RiskConfig for testing."""
    defaults = dict(
        daily_var_limit=0.02,
        max_drawdown_pct=0.05,
        var_confidence=0.99,
        var_lookback_days=252,
        var_method="historical",
        position_sizing=PositionSizingConfig(),
        cooldown=CooldownConfig(),
        stress_test=StressTestConfig(enabled=False),
        strategy_filter=StrategyFilterConfig(enabled=False),
    )
    defaults.update(overrides)
    return RiskConfig(**defaults)


def _make_market_state(**overrides) -> MarketState:
    defaults = dict(
        current_price=1.1000,
        current_spread=0.00010,
        current_atr=0.0050,
        historical_atr=0.0045,
        spread_limit=0.00020,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


# ---------------------------------------------------------------------------
# RiskManager.evaluate_action
# ---------------------------------------------------------------------------


class TestRiskManagerEvaluateAction:
    @pytest.fixture(autouse=True)
    def _patch_kill_file(self, tmp_path: Path):
        with patch.object(KillSwitch, "KILL_FILE", tmp_path / "KILL_SWITCH"):
            yield

    def _make_rm(self, **config_overrides) -> RiskManager:
        cfg = _make_risk_config(**config_overrides)
        rm = RiskManager(config=cfg, initial_balance=100_000.0)
        # Seed with enough returns so VaR doesn't block
        rng = np.random.default_rng(42)
        for _ in range(300):
            rm.record_daily_return(rng.normal(0, 0.005))
        rm.update_portfolio(100_000.0)
        return rm

    def test_kill_switch_blocks(self):
        rm = self._make_rm()
        rm.kill_switch.activate("test")
        ms = _make_market_state()
        decision = rm.evaluate_action(0.5, ms)
        assert not decision.approved
        assert "Kill switch" in decision.reason

    def test_daily_loss_guard_blocks(self):
        rm = self._make_rm()
        rm.daily_loss_guard.update(100_000.0)
        rm.daily_loss_guard.update(97_000.0)  # 3% > 2% limit
        ms = _make_market_state()
        decision = rm.evaluate_action(0.5, ms)
        assert not decision.approved
        assert "Daily loss" in decision.reason

    def test_spread_too_wide_blocks(self):
        rm = self._make_rm()
        ms = _make_market_state(current_spread=0.0005, spread_limit=0.0002)
        decision = rm.evaluate_action(0.5, ms)
        assert not decision.approved
        assert "Spread" in decision.reason

    def test_small_action_returns_neutral(self):
        rm = self._make_rm()
        ms = _make_market_state()
        decision = rm.evaluate_action(0.01, ms)
        assert decision.approved
        assert decision.position_size == 0.0
        assert "too small" in decision.reason

    def test_approved_action_has_position_size(self):
        rm = self._make_rm()
        ms = _make_market_state()
        decision = rm.evaluate_action(0.8, ms)
        # Should be approved with some position size
        # (exact size depends on PositionSizer internals)
        if decision.approved:
            assert decision.position_size > 0
            assert decision.adjusted_action == 0.8

    def test_cooldown_blocks(self):
        rm = self._make_rm()
        # Trigger cooldown by recording consecutive losses
        for _ in range(5):
            rm.record_trade(-500.0, trade_return=-0.005)
        ms = _make_market_state()
        decision = rm.evaluate_action(0.5, ms)
        if rm.cooldown.is_active:
            assert not decision.approved
            assert "Cooldown" in decision.reason

    def test_uncertainty_scaling_reduces_position(self):
        rm = self._make_rm()
        ms = _make_market_state()
        # Compare position with and without uncertainty
        d_no_unc = rm.evaluate_action(0.8, ms, uncertainty_score=None)
        d_high_unc = rm.evaluate_action(0.8, ms, uncertainty_score=0.9)
        if d_no_unc.approved and d_high_unc.approved:
            # High uncertainty should produce smaller or equal position
            assert d_high_unc.var_scale <= d_no_unc.var_scale

    def test_weekend_block(self):
        rm = self._make_rm()
        ms = _make_market_state()
        # Mock weekend
        friday_late = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
        with patch.object(rm.weekend_guard, "check", return_value=(True, 0.0)):
            decision = rm.evaluate_action(0.5, ms)
            assert not decision.approved
            assert "Weekend" in decision.reason

    def test_drawdown_breach_blocks(self):
        rm = self._make_rm()
        ms = _make_market_state()
        rm.drawdown._breached = True
        with patch.object(rm.drawdown, "is_breached", new_callable=lambda: property(lambda self: True)):
            decision = rm.evaluate_action(0.5, ms)
            assert not decision.approved

    def test_checks_passed_populated(self):
        rm = self._make_rm()
        ms = _make_market_state()
        decision = rm.evaluate_action(0.01, ms)
        assert len(decision.checks_passed) > 0

    def test_risk_decision_fields(self):
        decision = RiskDecision(
            approved=True,
            adjusted_action=0.5,
            position_size=0.10,
            reason="All checks passed",
            checks_passed=["a", "b"],
            checks_failed=[],
            var_scale=0.8,
        )
        assert decision.var_scale == 0.8
        assert decision.checks_failed == []


# ---------------------------------------------------------------------------
# RiskManager.force_close_all
# ---------------------------------------------------------------------------


class TestForceCloseAll:
    @pytest.fixture(autouse=True)
    def _patch_kill_file(self, tmp_path: Path):
        with patch.object(KillSwitch, "KILL_FILE", tmp_path / "KILL_SWITCH"):
            yield

    def test_force_close_on_drawdown_breach(self):
        rm = RiskManager(config=_make_risk_config(), initial_balance=100_000.0)
        with patch.object(type(rm.drawdown), "is_breached", new_callable=lambda: property(lambda self: True)):
            assert rm.force_close_all() is True

    def test_force_close_on_kill_switch(self):
        rm = RiskManager(config=_make_risk_config(), initial_balance=100_000.0)
        rm.kill_switch.activate("test")
        assert rm.force_close_all() is True

    def test_force_close_on_daily_loss(self):
        rm = RiskManager(config=_make_risk_config(), initial_balance=100_000.0)
        rm.daily_loss_guard.update(100_000.0)
        rm.daily_loss_guard.update(97_000.0)
        assert rm.force_close_all() is True

    def test_no_force_close_normal(self):
        rm = RiskManager(config=_make_risk_config(), initial_balance=100_000.0)
        rm.update_portfolio(100_000.0)
        with patch.object(rm.weekend_guard, "check", return_value=(False, 1.0)):
            assert rm.force_close_all() is False


# ---------------------------------------------------------------------------
# RiskManager.compute_dynamic_stop
# ---------------------------------------------------------------------------


class TestDynamicStop:
    @pytest.fixture(autouse=True)
    def _patch_kill_file(self, tmp_path: Path):
        with patch.object(KillSwitch, "KILL_FILE", tmp_path / "KILL_SWITCH"):
            yield

    def test_trending_regime_uses_trailing(self):
        rm = RiskManager(config=_make_risk_config(), initial_balance=100_000.0)
        stop = rm.compute_dynamic_stop(
            uncertainty_score=0.0, regime="trending", current_atr=0.005
        )
        assert stop.trailing is True
        assert stop.stop_distance is not None

    def test_volatile_regime_wider_stops(self):
        rm = RiskManager(config=_make_risk_config(), initial_balance=100_000.0)
        trend_stop = rm.compute_dynamic_stop(0.0, "trending", 0.005)
        vol_stop = rm.compute_dynamic_stop(0.0, "volatile", 0.005)
        assert vol_stop.atr_mult > trend_stop.atr_mult

    def test_no_atr_returns_none_distance(self):
        rm = RiskManager(config=_make_risk_config(), initial_balance=100_000.0)
        stop = rm.compute_dynamic_stop(0.0, "flat", current_atr=None)
        assert stop.stop_distance is None
