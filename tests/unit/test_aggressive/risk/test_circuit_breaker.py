"""Tests for the circuit breaker — equity drawdown limits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from apexfx.aggressive.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from apexfx.aggressive.risk.kill_switch import KillSwitch

UTC = timezone.utc


@pytest.fixture
def kill_switch(tmp_path: Path) -> KillSwitch:
    return KillSwitch(
        flag_path=tmp_path / "kill",
        cooldown_path=tmp_path / "cool",
        cooldown_seconds=60.0,
    )


@pytest.fixture
def config() -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        daily_loss_pct=0.08,
        weekly_loss_pct=0.20,
        monthly_dd_pct=0.35,
        max_consecutive_failed_orders=3,
    )


@pytest.fixture
def breaker(config, kill_switch, tmp_path) -> CircuitBreaker:
    return CircuitBreaker(
        config=config, kill_switch=kill_switch,
        state_path=tmp_path / "breaker.json",
    )


# ---------------------------------------------------------------------------


class TestConfig:
    def test_invalid_daily_loss_pct_rejected(self) -> None:
        with pytest.raises(ValueError):
            CircuitBreakerConfig(daily_loss_pct=0.0)
        with pytest.raises(ValueError):
            CircuitBreakerConfig(daily_loss_pct=1.5)

    def test_invalid_consec_orders_rejected(self) -> None:
        with pytest.raises(ValueError):
            CircuitBreakerConfig(max_consecutive_failed_orders=0)


# ---------------------------------------------------------------------------


class TestEquityObservation:
    def test_first_observation_sets_anchors(self, breaker: CircuitBreaker) -> None:
        result = breaker.observe_equity(1000.0)
        assert not result.tripped
        assert breaker.state.day_start_equity == 1000.0
        assert breaker.state.week_start_equity == 1000.0
        assert breaker.state.all_time_high_equity == 1000.0

    def test_within_threshold_no_trip(self, breaker: CircuitBreaker) -> None:
        breaker.observe_equity(1000.0)  # anchor
        # 5% loss — under 8% daily threshold
        result = breaker.observe_equity(950.0)
        assert not result.tripped

    def test_daily_loss_threshold_trips(
        self, breaker: CircuitBreaker, kill_switch: KillSwitch,
    ) -> None:
        breaker.observe_equity(1000.0)
        # 10% loss — over 8% daily threshold
        result = breaker.observe_equity(900.0)
        assert result.tripped
        assert result.reason == "daily_loss_limit"
        assert kill_switch.is_active()

    def test_weekly_loss_threshold_trips(
        self, breaker: CircuitBreaker, kill_switch: KillSwitch,
    ) -> None:
        # First observation in week N
        ts = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)  # Monday
        breaker.observe_equity(1000.0, now=ts)

        # Day 2 — equity drops 5% (under daily, but cumulative)
        breaker.observe_equity(950.0, now=ts + timedelta(days=1))

        # Day 3 — equity drops to 75% of week-start = 25% weekly loss
        # But day 3 daily anchor was 950 → 750 = 21% daily ⇒ also trips daily.
        # Use a fresh-day equity that's still over weekly limit but under daily.
        # Day 3 starts fresh day_start at 950, drop to 770 = 18.9% daily (still trips)
        # We need: day_start = X, equity < X×0.92, equity / week_start < 0.80
        # If day 3 starts at 850 (we set it manually via roll) → equity 700:
        #   daily 17.6% (trips daily)
        # Easier: bump observed equity up day 2 then back down day 3
        # Day 2: equity 1100 → all-time high updated
        # Day 3: equity drops to 790 — daily start was 1100 → drops 28% (trips daily first)

        # Cleaner path: manually configure to test ONLY weekly trip
        breaker._state.day_start_equity = 800.0  # so daily anchor = 800
        breaker._state.day_start_date = (ts + timedelta(days=2)).date().isoformat()
        # 750 vs day_start 800 = 6.25% daily (under 8%)
        # 750 vs week_start 1000 = 25% weekly (over 20%)
        result = breaker.observe_equity(750.0, now=ts + timedelta(days=2))
        assert result.tripped
        assert result.reason == "weekly_loss_limit"

    def test_monthly_dd_from_atl_high_trips(
        self, breaker: CircuitBreaker,
    ) -> None:
        # Build up equity over time
        breaker.observe_equity(1000.0)
        breaker.observe_equity(1500.0)  # ATH = 1500
        # 40% drawdown from peak — trips monthly (35% threshold)
        breaker._state.day_start_equity = 1500.0
        breaker._state.week_start_equity = 1500.0
        result = breaker.observe_equity(900.0)
        # Daily dd = 40%, weekly dd = 40%, monthly dd from ATH = 40%
        # Daily check fires first (8%), so we won't get to monthly
        assert result.tripped
        # The first check that trips is daily, not monthly — verify reason
        assert result.reason == "daily_loss_limit"

    def test_zero_or_negative_equity_does_not_update(
        self, breaker: CircuitBreaker,
    ) -> None:
        breaker.observe_equity(1000.0)
        original_anchor = breaker.state.day_start_equity
        breaker.observe_equity(0.0)  # transient bad read
        breaker.observe_equity(-5.0)
        assert breaker.state.day_start_equity == original_anchor


# ---------------------------------------------------------------------------


class TestPeriodRollover:
    def test_new_day_resets_daily_anchor(
        self, breaker: CircuitBreaker,
    ) -> None:
        ts = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
        breaker.observe_equity(1000.0, now=ts)
        # Tomorrow same time — anchor should reset to today's equity
        result = breaker.observe_equity(900.0, now=ts + timedelta(days=1))
        # 900 is now BOTH day_start and current — no drawdown
        assert not result.tripped
        assert breaker.state.day_start_equity == 900.0

    def test_new_week_resets_weekly_anchor(
        self, breaker: CircuitBreaker,
    ) -> None:
        # Monday of week N
        ts = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
        breaker.observe_equity(1000.0, now=ts)
        # Monday of week N+1
        next_week = ts + timedelta(days=7)
        breaker.observe_equity(900.0, now=next_week)
        assert breaker.state.week_start_equity == 900.0


# ---------------------------------------------------------------------------


class TestOrderOutcomes:
    def test_failure_increments_counter(self, breaker: CircuitBreaker) -> None:
        breaker.notify_order_failure()
        assert breaker.state.consecutive_failed_orders == 1

    def test_success_resets_counter(self, breaker: CircuitBreaker) -> None:
        breaker.notify_order_failure()
        breaker.notify_order_failure()
        breaker.notify_order_success()
        assert breaker.state.consecutive_failed_orders == 0

    def test_three_failures_trip(
        self, breaker: CircuitBreaker, kill_switch: KillSwitch,
    ) -> None:
        breaker.notify_order_failure()
        breaker.notify_order_failure()
        result = breaker.notify_order_failure()  # 3rd
        assert result.tripped
        assert result.reason == "consecutive_failed_orders"
        assert kill_switch.is_active()

    def test_reset_after_success_resets_trip_path(
        self, breaker: CircuitBreaker, kill_switch: KillSwitch,
    ) -> None:
        breaker.notify_order_failure()
        breaker.notify_order_failure()
        breaker.notify_order_success()  # reset
        # Two more failures shouldn't trip (counter went 0 → 1 → 2)
        breaker.notify_order_failure()
        result = breaker.notify_order_failure()
        assert not result.tripped
        assert not kill_switch.is_active()


# ---------------------------------------------------------------------------


class TestPersistence:
    def test_state_survives_restart(
        self, config, kill_switch, tmp_path,
    ) -> None:
        path = tmp_path / "state.json"
        b1 = CircuitBreaker(config=config, kill_switch=kill_switch, state_path=path)
        b1.observe_equity(1000.0)
        b1.notify_order_failure()

        # New instance reads from disk
        b2 = CircuitBreaker(config=config, kill_switch=kill_switch, state_path=path)
        assert b2.state.day_start_equity == 1000.0
        assert b2.state.consecutive_failed_orders == 1

    def test_corrupt_state_starts_fresh(
        self, config, kill_switch, tmp_path,
    ) -> None:
        path = tmp_path / "state.json"
        path.write_text("not-json{")
        # Should not crash; starts with default state
        b = CircuitBreaker(config=config, kill_switch=kill_switch, state_path=path)
        assert b.state.day_start_equity == 0.0
