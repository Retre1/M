"""System health monitoring: connection, data freshness, latency, resources."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from apexfx.utils.logging import get_logger

if TYPE_CHECKING:
    from apexfx.data.mt5_client import MT5Client

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """System health snapshot."""
    mt5_connected: bool
    data_fresh: bool
    last_tick_age_s: float
    inference_latency_ms: float
    memory_usage_mb: float
    disk_usage_pct: float
    overall_healthy: bool
    issues: list[str]


class HealthCheck:
    """Monitors system health and triggers alerts."""

    def __init__(
        self,
        mt5_client: MT5Client | None = None,
        max_tick_age_s: float = 30.0,
        max_inference_latency_ms: float = 1000.0,
        max_memory_mb: float = 4096.0,
    ) -> None:
        self._mt5 = mt5_client
        self._max_tick_age = max_tick_age_s
        self._max_inference_latency = max_inference_latency_ms
        self._max_memory = max_memory_mb

        self._last_tick_time: datetime | None = None
        self._last_inference_ms: float = 0.0

    def update_tick_time(self, tick_time: datetime) -> None:
        self._last_tick_time = tick_time

    def update_inference_latency(self, latency_ms: float) -> None:
        self._last_inference_ms = latency_ms

    def check(self) -> HealthStatus:
        """Run all health checks and return status."""
        issues: list[str] = []

        # MT5 connection
        mt5_ok = True
        if self._mt5 is not None:
            try:
                self._mt5._ensure_connected()
            except ConnectionError:
                mt5_ok = False
                issues.append("MT5 disconnected")

        # Data freshness
        data_fresh = True
        tick_age = 0.0
        if self._last_tick_time:
            tick_age = (datetime.now(UTC) - self._last_tick_time).total_seconds()
            if tick_age > self._max_tick_age:
                data_fresh = False
                issues.append(f"Stale data: last tick {tick_age:.0f}s ago")
        else:
            data_fresh = False
            issues.append("No tick data received yet")

        # Inference latency
        if self._last_inference_ms > self._max_inference_latency:
            issues.append(f"High inference latency: {self._last_inference_ms:.0f}ms")

        # Memory usage
        try:
            import resource
            mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB on macOS
        except Exception:
            mem_usage = 0.0

        if mem_usage > self._max_memory:
            issues.append(f"High memory: {mem_usage:.0f}MB")

        # Disk usage
        disk_pct = 0.0
        try:
            stat = os.statvfs(".")
            disk_pct = (1 - stat.f_bavail / stat.f_blocks) * 100
            if disk_pct > 90:
                issues.append(f"Disk nearly full: {disk_pct:.0f}%")
        except Exception:
            pass

        overall = mt5_ok and data_fresh and len(issues) == 0

        if issues:
            logger.warning("Health issues detected", issues=issues)

        return HealthStatus(
            mt5_connected=mt5_ok,
            data_fresh=data_fresh,
            last_tick_age_s=tick_age,
            inference_latency_ms=self._last_inference_ms,
            memory_usage_mb=mem_usage,
            disk_usage_pct=disk_pct,
            overall_healthy=overall,
            issues=issues,
        )
