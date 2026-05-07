"""Tests for the kill switch — file-based emergency halt."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from apexfx.aggressive.risk.kill_switch import KillSwitch


@pytest.fixture
def ks(tmp_path: Path) -> KillSwitch:
    """Fresh kill switch using tmp files — no cross-test contamination."""
    return KillSwitch(
        flag_path=tmp_path / "flag",
        cooldown_path=tmp_path / "cooldown",
        cooldown_seconds=60.0,
    )


class TestInactive:
    def test_initially_inactive(self, ks: KillSwitch) -> None:
        assert ks.is_active() is False
        st = ks.state()
        assert st.active is False
        assert st.reason == ""

    def test_state_no_files(self, ks: KillSwitch) -> None:
        st = ks.state()
        assert st.cooldown_remaining_s == 0.0


class TestTrigger:
    def test_trigger_creates_flag(self, ks: KillSwitch) -> None:
        ks.trigger("daily_loss_limit")
        assert ks.is_active()
        st = ks.state()
        assert st.active is True
        assert st.reason == "daily_loss_limit"

    def test_trigger_does_not_overwrite_existing(self, ks: KillSwitch) -> None:
        ks.trigger("first_reason")
        ks.trigger("second_reason")  # should be a no-op
        assert ks.state().reason == "first_reason"

    def test_trigger_writes_reason_to_file(
        self, ks: KillSwitch, tmp_path: Path,
    ) -> None:
        ks.trigger("monthly_drawdown")
        contents = (tmp_path / "flag").read_text()
        assert contents == "monthly_drawdown"


class TestDisarm:
    def test_disarm_clears_active(self, ks: KillSwitch) -> None:
        ks.trigger("test")
        ks.disarm()
        # Flag is gone — but cooldown is now active, so still active
        assert ks.is_active() is True
        st = ks.state()
        assert st.reason == "re-arm cooldown"
        assert st.cooldown_remaining_s > 0

    def test_disarm_when_not_triggered_noop(self, ks: KillSwitch) -> None:
        # No exception, no side effect
        ks.disarm()
        assert ks.is_active() is False

    def test_active_until_cooldown_expires(self, ks: KillSwitch, tmp_path: Path) -> None:
        ks.trigger("x")
        ks.disarm()
        # Manually expire the cooldown by writing past timestamp
        (tmp_path / "cooldown").write_text(f"{time.time() - 1:.1f}")
        assert ks.is_active() is False


class TestForceClear:
    def test_force_clear_removes_both(self, ks: KillSwitch, tmp_path: Path) -> None:
        ks.trigger("x")
        ks.disarm()
        # Both files should exist after disarm
        assert (tmp_path / "cooldown").exists()
        ks.force_clear()
        assert not (tmp_path / "flag").exists()
        assert not (tmp_path / "cooldown").exists()
        assert ks.is_active() is False


class TestEdgeCases:
    def test_corrupt_cooldown_treated_as_expired(
        self, ks: KillSwitch, tmp_path: Path,
    ) -> None:
        (tmp_path / "cooldown").write_text("not-a-number")
        # Should not crash; treats as expired and cleans up
        assert ks.is_active() is False
        assert not (tmp_path / "cooldown").exists()

    def test_state_handles_unreadable_flag(
        self, ks: KillSwitch, tmp_path: Path,
    ) -> None:
        # Flag exists but is empty (legitimate manual touch)
        (tmp_path / "flag").touch()
        st = ks.state()
        assert st.active is True
        assert st.reason == "manual"
