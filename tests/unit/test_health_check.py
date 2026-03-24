"""Tests for src/apexfx/live/health_check.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from apexfx.live.health_check import HealthCheck, HealthStatus


class TestHealthCheckNoMT5:
    """Tests when mt5_client is None."""

    def test_no_mt5_client_mt5_connected_true(self):
        hc = HealthCheck(mt5_client=None)
        status = hc.check()
        # With no MT5 client, mt5_ok stays True (no check performed)
        assert status.mt5_connected is True

    def test_no_mt5_no_tick_data(self):
        hc = HealthCheck(mt5_client=None)
        status = hc.check()
        assert status.data_fresh is False
        assert "No tick data received yet" in status.issues


class TestDataFreshness:
    """Tests for tick data freshness checks."""

    def test_fresh_tick_data(self):
        hc = HealthCheck(mt5_client=None, max_tick_age_s=30.0)
        # Tick received just now
        hc.update_tick_time(datetime.now(UTC))
        status = hc.check()
        assert status.data_fresh is True
        assert status.last_tick_age_s < 2.0  # allow small time delta

    def test_stale_tick_data_reports_issue(self):
        hc = HealthCheck(mt5_client=None, max_tick_age_s=30.0)
        # Tick received 60 seconds ago
        old_time = datetime.now(UTC) - timedelta(seconds=60)
        hc.update_tick_time(old_time)
        status = hc.check()
        assert status.data_fresh is False
        assert any("Stale data" in issue for issue in status.issues)

    def test_no_tick_data(self):
        hc = HealthCheck(mt5_client=None)
        status = hc.check()
        assert status.data_fresh is False
        assert any("No tick data" in issue for issue in status.issues)


class TestInferenceLatency:
    """Tests for inference latency monitoring."""

    def test_high_inference_latency_reports_issue(self):
        hc = HealthCheck(mt5_client=None, max_inference_latency_ms=1000.0)
        hc.update_inference_latency(1500.0)
        # Also provide fresh tick so that only latency issue appears
        hc.update_tick_time(datetime.now(UTC))
        status = hc.check()
        assert any("High inference latency" in issue for issue in status.issues)

    def test_normal_inference_latency_no_issue(self):
        hc = HealthCheck(mt5_client=None, max_inference_latency_ms=1000.0)
        hc.update_inference_latency(500.0)
        hc.update_tick_time(datetime.now(UTC))
        status = hc.check()
        assert not any("inference" in issue.lower() for issue in status.issues)


class TestOverallHealth:
    """Tests for overall_healthy flag."""

    @patch("apexfx.live.health_check.os.statvfs")
    def test_overall_healthy_when_all_pass(self, mock_statvfs):
        # Mock disk usage to be low
        mock_stat = MagicMock()
        mock_stat.f_bavail = 90
        mock_stat.f_blocks = 100
        mock_statvfs.return_value = mock_stat

        hc = HealthCheck(mt5_client=None, max_tick_age_s=30.0, max_memory_mb=99999)
        hc.update_tick_time(datetime.now(UTC))
        hc.update_inference_latency(100.0)
        status = hc.check()
        assert status.overall_healthy is True
        assert status.issues == []

    def test_overall_unhealthy_when_no_tick_data(self):
        hc = HealthCheck(mt5_client=None)
        status = hc.check()
        assert status.overall_healthy is False

    def test_overall_unhealthy_when_stale_data(self):
        hc = HealthCheck(mt5_client=None, max_tick_age_s=30.0)
        hc.update_tick_time(datetime.now(UTC) - timedelta(seconds=60))
        status = hc.check()
        assert status.overall_healthy is False


class TestHealthStatus:
    """Tests for the HealthStatus dataclass."""

    def test_health_status_fields(self):
        status = HealthStatus(
            mt5_connected=True,
            data_fresh=True,
            last_tick_age_s=1.0,
            inference_latency_ms=50.0,
            memory_usage_mb=512.0,
            disk_usage_pct=45.0,
            overall_healthy=True,
            issues=[],
        )
        assert status.mt5_connected is True
        assert status.data_fresh is True
        assert status.inference_latency_ms == 50.0
