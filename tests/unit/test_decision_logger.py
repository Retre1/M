"""Tests for XAI Decision Logger (Sprint 2, Improvement 2)."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from apexfx.live.decision_logger import DecisionLogger, DecisionRecord


# ---------------------------------------------------------------------------
# Helpers — lightweight stand-in for TradingSignal (avoids heavy imports)
# ---------------------------------------------------------------------------

@dataclass
class _FakeSignal:
    action: float = 0.65
    confidence: float = 0.65
    trend_agent_action: float = 0.3
    reversion_agent_action: float = 0.2
    breakout_agent_action: float = 0.15
    gating_weights: tuple[float, ...] = (0.5, 0.3, 0.2)
    regime: str = "trending"
    timestamp: datetime = field(default_factory=lambda: datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC))
    inference_time_ms: float = 4.2
    uncertainty_score: float = 0.15
    position_scale: float = 0.925
    recommended_stop_atr_mult: float = 1.725
    top_features: list[tuple[str, float]] = field(default_factory=lambda: [
        ("rsi_14", 0.12), ("ema_cross", 0.09), ("atr_norm", 0.07),
    ])


def _make_signal(**overrides) -> _FakeSignal:
    return _FakeSignal(**overrides)


# ---------------------------------------------------------------------------
# DecisionRecord.from_signal
# ---------------------------------------------------------------------------


class TestDecisionRecordFromSignal:
    def test_basic_creation(self):
        sig = _make_signal()
        rec = DecisionRecord.from_signal(sig, portfolio_value=100_000.0, position=0.5)

        assert rec.action == 0.65
        assert rec.confidence == 0.65
        assert rec.regime == "trending"
        assert rec.trend_action == 0.3
        assert rec.reversion_action == 0.2
        assert rec.breakout_action == 0.15
        assert rec.gating_weight_trend == 0.5
        assert rec.gating_weight_reversion == 0.3
        assert rec.gating_weight_breakout == 0.2
        assert rec.uncertainty == 0.15
        assert rec.position_scale == 0.925
        assert rec.stop_mult == 1.725
        assert rec.portfolio_value == 100_000.0
        assert rec.current_position == 0.5
        assert rec.inference_time_ms == 4.2

    def test_top_features_json(self):
        sig = _make_signal()
        rec = DecisionRecord.from_signal(sig, portfolio_value=50_000, position=0.0)
        features = json.loads(rec.top_features_json)
        assert len(features) == 3
        assert features[0]["name"] == "rsi_14"

    def test_empty_top_features(self):
        sig = _make_signal(top_features=[])
        rec = DecisionRecord.from_signal(sig, portfolio_value=50_000, position=0.0)
        features = json.loads(rec.top_features_json)
        assert features == []

    def test_timestamp_is_iso(self):
        sig = _make_signal()
        rec = DecisionRecord.from_signal(sig, portfolio_value=50_000, position=0.0)
        # Should be parseable as ISO
        dt = datetime.fromisoformat(rec.timestamp)
        assert dt.year == 2025


# ---------------------------------------------------------------------------
# DecisionLogger — log + flush cycle
# ---------------------------------------------------------------------------


class TestLogFlushCycle:
    def test_manual_flush(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db, buffer_size=1000) as dl:
            sig = _make_signal()
            rec = DecisionRecord.from_signal(sig, 100_000, 0.0)
            dl.log(rec)
            # Still in buffer
            dl.flush()
            # Now in DB — query should return it
            results = dl.query(limit=10)
        assert len(results) == 1
        assert results[0].action == 0.65

    def test_auto_flush_on_buffer_size(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db, buffer_size=5) as dl:
            for i in range(5):
                sig = _make_signal(action=float(i) / 10)
                rec = DecisionRecord.from_signal(sig, 100_000, 0.0)
                dl.log(rec)
            # After 5 logs with buffer_size=5, buffer should have been flushed
            # Query without explicit flush should find records
            results = dl.query(limit=100)
        assert len(results) == 5

    def test_multiple_flushes(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db, buffer_size=3) as dl:
            for i in range(7):
                sig = _make_signal(action=float(i) / 10)
                rec = DecisionRecord.from_signal(sig, 100_000, 0.0)
                dl.log(rec)
            # 7 records: 2 auto-flushes (at 3 and 6), 1 remaining in buffer
            results = dl.query(limit=100)
        assert len(results) == 7


# ---------------------------------------------------------------------------
# Query with date range and limit
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_with_date_range(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db, buffer_size=1000) as dl:
            t1 = datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC)
            t2 = datetime(2025, 6, 15, 10, 0, 0, tzinfo=UTC)
            t3 = datetime(2025, 6, 30, 10, 0, 0, tzinfo=UTC)

            for ts in [t1, t2, t3]:
                sig = _make_signal(timestamp=ts)
                rec = DecisionRecord.from_signal(sig, 100_000, 0.0)
                dl.log(rec)

            # Query only June 10-20
            results = dl.query(
                start=datetime(2025, 6, 10, tzinfo=UTC),
                end=datetime(2025, 6, 20, tzinfo=UTC),
            )
            assert len(results) == 1
            assert "2025-06-15" in results[0].timestamp

    def test_query_with_start_only(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db, buffer_size=1000) as dl:
            t1 = datetime(2025, 1, 1, tzinfo=UTC)
            t2 = datetime(2025, 12, 1, tzinfo=UTC)
            for ts in [t1, t2]:
                sig = _make_signal(timestamp=ts)
                dl.log(DecisionRecord.from_signal(sig, 100_000, 0.0))

            results = dl.query(start=datetime(2025, 6, 1, tzinfo=UTC))
            assert len(results) == 1

    def test_query_with_limit(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db, buffer_size=1000) as dl:
            for i in range(20):
                sig = _make_signal()
                dl.log(DecisionRecord.from_signal(sig, 100_000, 0.0))
            results = dl.query(limit=5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------


class TestExportCSV:
    def test_export_creates_file(self, tmp_path: Path):
        db = tmp_path / "test.db"
        csv_path = tmp_path / "export" / "decisions.csv"

        with DecisionLogger(db, buffer_size=1000) as dl:
            for i in range(3):
                sig = _make_signal(action=0.1 * i)
                dl.log(DecisionRecord.from_signal(sig, 100_000, 0.0))
            dl.export_csv(csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 1 header + 3 data rows
        assert len(rows) == 4
        assert rows[0][0] == "timestamp"


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_basic(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db, buffer_size=1000) as dl:
            now = datetime.now(UTC)
            for regime in ["trending", "trending", "flat"]:
                sig = _make_signal(
                    regime=regime,
                    timestamp=now,
                    confidence=0.8,
                    uncertainty_score=0.2,
                )
                sig.inference_time_ms = 5.0
                dl.log(DecisionRecord.from_signal(sig, 100_000, 0.0))

            stats = dl.get_stats(hours=1)

        assert stats["total"] == 3
        assert stats["avg_confidence"] == 0.8
        assert stats["avg_uncertainty"] == 0.2
        assert stats["regime_distribution"]["trending"] == 2
        assert stats["regime_distribution"]["flat"] == 1
        assert stats["avg_inference_ms"] == 5.0

    def test_stats_empty_db(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db) as dl:
            stats = dl.get_stats(hours=24)
        assert stats["total"] == 0
        assert stats["regime_distribution"] == {}


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_enter_exit(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db) as dl:
            sig = _make_signal()
            dl.log(DecisionRecord.from_signal(sig, 100_000, 0.0))
        # After exit, buffer should be flushed — reopen and check
        dl2 = DecisionLogger(db)
        results = dl2.query(limit=10)
        dl2.close()
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_logging(self, tmp_path: Path):
        db = tmp_path / "test.db"
        n_threads = 4
        records_per_thread = 50

        with DecisionLogger(db, buffer_size=10) as dl:
            errors: list[Exception] = []

            def worker(thread_id: int):
                try:
                    for i in range(records_per_thread):
                        sig = _make_signal(action=thread_id + i * 0.001)
                        rec = DecisionRecord.from_signal(sig, 100_000, 0.0)
                        dl.log(rec)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"Thread errors: {errors}"
            results = dl.query(limit=10_000)

        assert len(results) == n_threads * records_per_thread


# ---------------------------------------------------------------------------
# Empty DB queries
# ---------------------------------------------------------------------------


class TestEmptyDB:
    def test_query_empty(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db) as dl:
            results = dl.query()
        assert results == []

    def test_export_csv_empty(self, tmp_path: Path):
        db = tmp_path / "test.db"
        csv_path = tmp_path / "empty.csv"
        with DecisionLogger(db) as dl:
            dl.export_csv(csv_path)
        assert csv_path.exists()
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # Header only

    def test_stats_empty(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with DecisionLogger(db) as dl:
            stats = dl.get_stats()
        assert stats["total"] == 0
