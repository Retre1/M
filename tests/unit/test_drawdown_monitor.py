"""Tests for src/apexfx/risk/drawdown_monitor.py."""

from __future__ import annotations

import pytest

from apexfx.risk.drawdown_monitor import DrawdownMonitor, DrawdownState


class TestInitialState:
    def test_zero_drawdown_initially(self):
        dm = DrawdownMonitor()
        assert dm.current_drawdown == 0.0

    def test_not_breached_initially(self):
        dm = DrawdownMonitor()
        assert dm.is_breached is False
        assert dm.is_critical is False
        assert dm.is_warning is False


class TestPeakTracking:
    def test_new_peak_no_drawdown(self):
        dm = DrawdownMonitor()
        state = dm.update(10000.0)
        assert state.current_drawdown == 0.0
        assert state.peak_value == 10000.0

        state = dm.update(10500.0)
        assert state.current_drawdown == 0.0
        assert state.peak_value == 10500.0

    def test_peak_value_tracking(self):
        dm = DrawdownMonitor()
        dm.update(10000.0)
        dm.update(10500.0)
        dm.update(10200.0)  # decline, but peak stays at 10500
        assert dm._peak_value == 10500.0


class TestDrawdownCalculation:
    def test_drawdown_after_decline(self):
        dm = DrawdownMonitor()
        dm.update(10000.0)
        state = dm.update(9500.0)
        # Drawdown = (10000 - 9500) / 10000 = 0.05
        assert state.current_drawdown == pytest.approx(0.05)
        assert state.current_value == 9500.0

    def test_drawdown_resets_on_new_peak(self):
        dm = DrawdownMonitor()
        dm.update(10000.0)
        dm.update(9500.0)
        state = dm.update(10100.0)  # new peak
        assert state.current_drawdown == 0.0
        assert state.peak_value == 10100.0


class TestWarningLevels:
    def test_normal_level(self):
        dm = DrawdownMonitor(warning_threshold=0.03, critical_threshold=0.04, max_drawdown_pct=0.05)
        dm.update(10000.0)
        state = dm.update(9800.0)  # 2% drawdown
        assert state.warning_level == "normal"

    def test_warning_level(self):
        dm = DrawdownMonitor(warning_threshold=0.03, critical_threshold=0.04, max_drawdown_pct=0.05)
        dm.update(10000.0)
        state = dm.update(9700.0)  # 3% drawdown
        assert state.warning_level == "warning"

    def test_critical_level(self):
        dm = DrawdownMonitor(warning_threshold=0.03, critical_threshold=0.04, max_drawdown_pct=0.05)
        dm.update(10000.0)
        state = dm.update(9600.0)  # 4% drawdown
        assert state.warning_level == "critical"

    def test_breached_level(self):
        dm = DrawdownMonitor(warning_threshold=0.03, critical_threshold=0.04, max_drawdown_pct=0.05)
        dm.update(10000.0)
        state = dm.update(9500.0)  # 5% drawdown
        assert state.warning_level == "breached"


class TestBooleanProperties:
    def test_is_breached(self):
        dm = DrawdownMonitor(max_drawdown_pct=0.05)
        dm.update(10000.0)
        dm.update(9400.0)  # 6% drawdown
        assert dm.is_breached is True

    def test_is_critical(self):
        dm = DrawdownMonitor(critical_threshold=0.04)
        dm.update(10000.0)
        dm.update(9550.0)  # 4.5% drawdown
        assert dm.is_critical is True

    def test_is_warning(self):
        dm = DrawdownMonitor(warning_threshold=0.03)
        dm.update(10000.0)
        dm.update(9650.0)  # 3.5% drawdown
        assert dm.is_warning is True


class TestReset:
    def test_reset_method(self):
        dm = DrawdownMonitor()
        dm.update(10000.0)
        dm.update(9500.0)
        assert dm.current_drawdown > 0

        dm.reset(10000.0)
        assert dm.current_drawdown == 0.0
        assert dm._peak_value == 10000.0
        assert dm._current_step == 0
        assert dm._max_dd_ever == 0.0
        assert dm._drawdown_history == []


class TestHistoryTruncation:
    def test_history_truncated_at_10k(self):
        dm = DrawdownMonitor()
        dm.update(10000.0)
        # Push 10001 updates to exceed 10K history limit
        for i in range(10001):
            dm.update(10000.0 - (i % 100))
        # After truncation, history should be trimmed to 5000
        assert len(dm._drawdown_history) <= 10_000


class TestDrawdownState:
    def test_state_fields(self):
        dm = DrawdownMonitor(warning_threshold=0.03, critical_threshold=0.04, max_drawdown_pct=0.05)
        dm.update(10000.0)
        state = dm.update(9700.0)
        assert isinstance(state, DrawdownState)
        assert state.current_drawdown == pytest.approx(0.03)
        assert state.peak_value == 10000.0
        assert state.current_value == 9700.0
        assert state.max_drawdown_ever == pytest.approx(0.03)
        assert state.warning_level == "warning"
