"""Tests for src/apexfx/risk/cooldown.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from apexfx.risk.cooldown import CooldownManager


class TestInitialState:
    def test_not_active_initially(self):
        cm = CooldownManager()
        assert cm.is_active is False

    def test_zero_consecutive_losses(self):
        cm = CooldownManager()
        assert cm.consecutive_losses == 0

    def test_remaining_is_none_initially(self):
        cm = CooldownManager()
        assert cm.remaining is None


class TestConsecutiveLosses:
    def test_two_losses_below_threshold(self):
        cm = CooldownManager(after_n_losses=3)
        cm.record_trade(-10.0)
        cm.record_trade(-20.0)
        assert cm.consecutive_losses == 2
        assert cm.is_active is False

    def test_three_losses_triggers_cooldown(self):
        cm = CooldownManager(after_n_losses=3, duration_minutes=60)
        cm.record_trade(-10.0)
        cm.record_trade(-20.0)
        cm.record_trade(-30.0)
        assert cm.is_active is True

    def test_consecutive_losses_reset_after_cooldown_activation(self):
        cm = CooldownManager(after_n_losses=3)
        cm.record_trade(-10.0)
        cm.record_trade(-20.0)
        cm.record_trade(-30.0)
        # _activate_cooldown resets consecutive losses to 0
        assert cm.consecutive_losses == 0

    def test_winning_trade_resets_losses(self):
        cm = CooldownManager(after_n_losses=3)
        cm.record_trade(-10.0)
        cm.record_trade(-20.0)
        assert cm.consecutive_losses == 2
        cm.record_trade(50.0)  # win
        assert cm.consecutive_losses == 0
        assert cm.is_active is False


class TestCooldownExpiry:
    def test_cooldown_expires_after_duration(self):
        cm = CooldownManager(after_n_losses=3, duration_minutes=60)
        cm.record_trade(-10.0)
        cm.record_trade(-20.0)
        cm.record_trade(-30.0)
        assert cm.is_active is True

        # Move time forward past duration
        future = datetime.now(UTC) + timedelta(minutes=61)
        with patch("apexfx.risk.cooldown.datetime") as mock_dt:
            mock_dt.now.return_value = future
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            assert cm.is_active is False

    def test_remaining_property_during_cooldown(self):
        cm = CooldownManager(after_n_losses=3, duration_minutes=60)
        cm.record_trade(-10.0)
        cm.record_trade(-20.0)
        cm.record_trade(-30.0)
        remaining = cm.remaining
        assert remaining is not None
        assert remaining.total_seconds() > 0
        assert remaining.total_seconds() <= 3600


class TestTiltDetection:
    def test_rapid_drawdown_triggers_cooldown(self):
        cm = CooldownManager(
            after_n_losses=100,  # high threshold so only tilt triggers
            tilt_drawdown_pct=0.02,
            tilt_window_minutes=30,
        )
        # Record portfolio declining 3% within the tilt window
        cm.record_portfolio_value(10000.0)
        cm.record_portfolio_value(9700.0)  # 3% drop
        assert cm.is_active is True


class TestForceReset:
    def test_force_reset_clears_cooldown(self):
        cm = CooldownManager(after_n_losses=3, duration_minutes=60)
        cm.record_trade(-10.0)
        cm.record_trade(-20.0)
        cm.record_trade(-30.0)
        assert cm.is_active is True

        cm.force_reset()
        assert cm.is_active is False
        assert cm.consecutive_losses == 0
        assert cm.remaining is None
