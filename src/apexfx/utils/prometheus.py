"""Prometheus metrics exporter for live trading monitoring.

Exposes key trading, risk, and system metrics via an HTTP endpoint
that Prometheus can scrape. Start with `start_metrics_server(port)`.

Metrics exposed:
  - apexfx_equity: Current portfolio equity
  - apexfx_pnl_total: Cumulative realized PnL
  - apexfx_drawdown_pct: Current drawdown from peak (%)
  - apexfx_trade_count: Total number of closed trades
  - apexfx_position_direction: Current position direction (+1/-1/0)
  - apexfx_position_volume: Current position volume (lots)
  - apexfx_bar_processing_seconds: Bar processing latency histogram
  - apexfx_inference_seconds: Model inference latency histogram
  - apexfx_fill_slippage_pips: Execution slippage histogram
  - apexfx_consecutive_failures: Circuit breaker failure counter
  - apexfx_kill_switch_active: Kill switch status (0/1)
  - apexfx_health_status: Overall system health (0/1)
  - apexfx_tick_age_seconds: Age of the last received tick
  - apexfx_memory_usage_mb: Process memory usage in MB
  - apexfx_model_version: Current model version
"""

from __future__ import annotations

import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class _Gauge:
    """Thread-safe gauge metric."""

    def __init__(self, name: str, help_text: str, labels: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help = help_text
        self.labels = labels
        self._value: float = 0.0
        self._labeled_values: dict[tuple[str, ...], float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **label_values: str) -> None:
        with self._lock:
            if label_values:
                key = tuple(label_values.get(l, "") for l in self.labels)
                self._labeled_values[key] = value
            else:
                self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def format(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            if self._labeled_values:
                for label_vals, val in self._labeled_values.items():
                    label_str = ",".join(
                        f'{l}="{v}"' for l, v in zip(self.labels, label_vals)
                    )
                    lines.append(f"{self.name}{{{label_str}}} {val}")
            else:
                lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


class _Histogram:
    """Thread-safe histogram metric with fixed buckets."""

    def __init__(self, name: str, help_text: str, buckets: tuple[float, ...] = ()) -> None:
        self.name = name
        self.help = help_text
        self.buckets = buckets or (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        self._counts: list[int] = [0] * len(self.buckets)
        self._count = 0
        self._sum = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            for i, b in enumerate(self.buckets):
                if value <= b:
                    self._counts[i] += 1

    def format(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self._lock:
            cumulative = 0
            for i, b in enumerate(self.buckets):
                cumulative += self._counts[i]
                lines.append(f'{self.name}_bucket{{le="{b}"}} {cumulative}')
            lines.append(f'{self.name}_bucket{{le="+Inf"}} {self._count}')
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global metric instances
# ---------------------------------------------------------------------------

equity = _Gauge("apexfx_equity", "Current portfolio equity")
pnl_total = _Gauge("apexfx_pnl_total", "Cumulative realized PnL")
drawdown_pct = _Gauge("apexfx_drawdown_pct", "Current drawdown from peak (%)")
trade_count = _Gauge("apexfx_trade_count", "Total number of closed trades")
position_direction = _Gauge("apexfx_position_direction", "Current position direction (+1/-1/0)")
position_volume = _Gauge("apexfx_position_volume", "Current position volume (lots)")
consecutive_failures = _Gauge("apexfx_consecutive_failures", "Circuit breaker consecutive failure count")
kill_switch_active = _Gauge("apexfx_kill_switch_active", "Kill switch status (0=off, 1=active)")
health_status = _Gauge("apexfx_health_status", "Overall system health (0=unhealthy, 1=healthy)")
tick_age_seconds = _Gauge("apexfx_tick_age_seconds", "Age of last received tick in seconds")
memory_usage_mb = _Gauge("apexfx_memory_usage_mb", "Process memory usage in MB")
model_version = _Gauge("apexfx_model_version", "Current model version number")

bar_processing_seconds = _Histogram(
    "apexfx_bar_processing_seconds",
    "Bar processing latency in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
inference_seconds = _Histogram(
    "apexfx_inference_seconds",
    "Model inference latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
fill_slippage_pips = _Histogram(
    "apexfx_fill_slippage_pips",
    "Execution slippage in pips",
    buckets=(0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)

_ALL_METRICS: list[_Gauge | _Histogram] = [
    equity, pnl_total, drawdown_pct, trade_count,
    position_direction, position_volume,
    consecutive_failures, kill_switch_active,
    health_status, tick_age_seconds, memory_usage_mb, model_version,
    bar_processing_seconds, inference_seconds, fill_slippage_pips,
]


# ---------------------------------------------------------------------------
# HTTP server for /metrics endpoint
# ---------------------------------------------------------------------------

class _MetricsHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves Prometheus metrics."""

    def do_GET(self) -> None:
        if self.path == "/metrics":
            body = "\n\n".join(m.format() for m in _ALL_METRICS) + "\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default request logging
        pass


_server: HTTPServer | None = None


def start_metrics_server(port: int = 9090, host: str = "0.0.0.0") -> None:
    """Start the Prometheus metrics HTTP server in a background daemon thread."""
    global _server
    if _server is not None:
        return

    _server = HTTPServer((host, port), _MetricsHandler)
    thread = threading.Thread(target=_server.serve_forever, daemon=True, name="prometheus-metrics")
    thread.start()
    logger.info("Prometheus metrics server started", host=host, port=port)


def stop_metrics_server() -> None:
    """Stop the Prometheus metrics server."""
    global _server
    if _server is not None:
        _server.shutdown()
        _server = None
