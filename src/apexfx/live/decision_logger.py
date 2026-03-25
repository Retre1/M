"""XAI Decision Logger — structured post-mortem records for every trade decision.

Stores WHY the bot made each decision: gating weights, top features, agent actions,
uncertainty. Records are buffered and batch-inserted into SQLite for efficient I/O.

Usage::

    with DecisionLogger("data/decisions.db") as dl:
        record = DecisionRecord.from_signal(signal, portfolio_value=100_000, position=0.5)
        dl.log(record)
"""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    action          REAL    NOT NULL,
    confidence      REAL    NOT NULL,
    regime          TEXT    NOT NULL,
    trend_action    REAL    NOT NULL,
    reversion_action REAL   NOT NULL,
    breakout_action REAL    NOT NULL,
    gating_weight_trend     REAL NOT NULL,
    gating_weight_reversion REAL NOT NULL,
    gating_weight_breakout  REAL NOT NULL,
    top_features_json       TEXT NOT NULL,
    uncertainty     REAL    NOT NULL,
    position_scale  REAL    NOT NULL,
    stop_mult       REAL    NOT NULL,
    portfolio_value REAL    NOT NULL,
    current_position REAL   NOT NULL,
    inference_time_ms REAL  NOT NULL
)
"""

_INSERT_SQL = """
INSERT INTO decisions (
    timestamp, action, confidence, regime,
    trend_action, reversion_action, breakout_action,
    gating_weight_trend, gating_weight_reversion, gating_weight_breakout,
    top_features_json, uncertainty, position_scale, stop_mult,
    portfolio_value, current_position, inference_time_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_QUERY_SQL = """
SELECT
    timestamp, action, confidence, regime,
    trend_action, reversion_action, breakout_action,
    gating_weight_trend, gating_weight_reversion, gating_weight_breakout,
    top_features_json, uncertainty, position_scale, stop_mult,
    portfolio_value, current_position, inference_time_ms
FROM decisions
{where}
ORDER BY timestamp DESC
LIMIT ?
"""


@dataclass
class DecisionRecord:
    """Immutable record of a single trade decision with full XAI context."""

    timestamp: str
    action: float
    confidence: float
    regime: str
    trend_action: float
    reversion_action: float
    breakout_action: float
    gating_weight_trend: float
    gating_weight_reversion: float
    gating_weight_breakout: float
    top_features_json: str
    uncertainty: float
    position_scale: float
    stop_mult: float
    portfolio_value: float
    current_position: float
    inference_time_ms: float

    @staticmethod
    def from_signal(
        signal,
        portfolio_value: float,
        position: float,
    ) -> DecisionRecord:
        """Build a DecisionRecord from a TradingSignal plus portfolio context.

        Parameters
        ----------
        signal:
            A ``TradingSignal`` (or ``MTFTradingSignal``) instance.
        portfolio_value:
            Current equity / portfolio value.
        position:
            Signed current position (volume * direction).
        """
        gw = signal.gating_weights
        top_features_json = json.dumps(
            [{"name": name, "importance": round(imp, 6)} for name, imp in signal.top_features]
            if signal.top_features
            else []
        )
        return DecisionRecord(
            timestamp=signal.timestamp.isoformat(),
            action=signal.action,
            confidence=signal.confidence,
            regime=signal.regime,
            trend_action=signal.trend_agent_action,
            reversion_action=signal.reversion_agent_action,
            breakout_action=signal.breakout_agent_action,
            gating_weight_trend=gw[0] if len(gw) > 0 else 0.0,
            gating_weight_reversion=gw[1] if len(gw) > 1 else 0.0,
            gating_weight_breakout=gw[2] if len(gw) > 2 else 0.0,
            top_features_json=top_features_json,
            uncertainty=signal.uncertainty_score,
            position_scale=signal.position_scale,
            stop_mult=signal.recommended_stop_atr_mult,
            portfolio_value=portfolio_value,
            current_position=position,
            inference_time_ms=signal.inference_time_ms,
        )

    def to_row(self) -> tuple:
        """Return values tuple matching ``_INSERT_SQL`` column order."""
        return (
            self.timestamp,
            self.action,
            self.confidence,
            self.regime,
            self.trend_action,
            self.reversion_action,
            self.breakout_action,
            self.gating_weight_trend,
            self.gating_weight_reversion,
            self.gating_weight_breakout,
            self.top_features_json,
            self.uncertainty,
            self.position_scale,
            self.stop_mult,
            self.portfolio_value,
            self.current_position,
            self.inference_time_ms,
        )


class DecisionLogger:
    """Thread-safe, buffered SQLite logger for trade-decision records.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Created if it does not exist.
    buffer_size:
        Number of records to buffer before auto-flushing to disk.
    """

    def __init__(self, db_path: str | Path, buffer_size: int = 100) -> None:
        self._db_path = Path(db_path)
        self._buffer_size = buffer_size
        self._buffer: list[DecisionRecord] = []
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open connection with WAL mode for concurrent read/write
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

        logger.info(
            "DecisionLogger initialised",
            db_path=str(self._db_path),
            buffer_size=buffer_size,
        )

    # --- Context manager --------------------------------------------------

    def __enter__(self) -> DecisionLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # --- Public API -------------------------------------------------------

    def log(self, record: DecisionRecord) -> None:
        """Add a record to the buffer; auto-flush when buffer_size is reached."""
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._buffer_size:
                self._flush_unlocked()

    def flush(self) -> None:
        """Flush all buffered records to SQLite."""
        with self._lock:
            self._flush_unlocked()

    def query(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[DecisionRecord]:
        """Query decision records with optional time range filter.

        Parameters
        ----------
        start:
            Inclusive start time (UTC).
        end:
            Inclusive end time (UTC).
        limit:
            Maximum number of records to return (most recent first).
        """
        # Flush pending records so query sees them
        self.flush()

        conditions: list[str] = []
        params: list = []

        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start.isoformat())
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end.isoformat())

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = _QUERY_SQL.format(where=where)
        params.append(limit)

        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

        return [
            DecisionRecord(
                timestamp=row[0],
                action=row[1],
                confidence=row[2],
                regime=row[3],
                trend_action=row[4],
                reversion_action=row[5],
                breakout_action=row[6],
                gating_weight_trend=row[7],
                gating_weight_reversion=row[8],
                gating_weight_breakout=row[9],
                top_features_json=row[10],
                uncertainty=row[11],
                position_scale=row[12],
                stop_mult=row[13],
                portfolio_value=row[14],
                current_position=row[15],
                inference_time_ms=row[16],
            )
            for row in rows
        ]

    def export_csv(self, path: Path) -> None:
        """Export all records to a CSV file."""
        self.flush()

        with self._lock:
            cursor = self._conn.execute(
                "SELECT timestamp, action, confidence, regime, "
                "trend_action, reversion_action, breakout_action, "
                "gating_weight_trend, gating_weight_reversion, gating_weight_breakout, "
                "top_features_json, uncertainty, position_scale, stop_mult, "
                "portfolio_value, current_position, inference_time_ms "
                "FROM decisions ORDER BY timestamp"
            )
            rows = cursor.fetchall()

        headers = [
            "timestamp", "action", "confidence", "regime",
            "trend_action", "reversion_action", "breakout_action",
            "gating_weight_trend", "gating_weight_reversion", "gating_weight_breakout",
            "top_features_json", "uncertainty", "position_scale", "stop_mult",
            "portfolio_value", "current_position", "inference_time_ms",
        ]

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        logger.info("Exported decisions to CSV", path=str(path), n_records=len(rows))

    def get_stats(self, hours: int = 24) -> dict:
        """Summary statistics for the last *hours* of decisions.

        Returns
        -------
        dict with keys: total, avg_confidence, avg_uncertainty,
        regime_distribution (dict[str, int]), avg_inference_ms.
        """
        self.flush()

        cutoff = datetime.now(UTC)
        # Compute cutoff by subtracting hours worth of seconds
        from datetime import timedelta

        cutoff = (cutoff - timedelta(hours=hours)).isoformat()

        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*), AVG(confidence), AVG(uncertainty), AVG(inference_time_ms) "
                "FROM decisions WHERE timestamp >= ?",
                (cutoff,),
            )
            row = cursor.fetchone()
            total = row[0] or 0
            avg_confidence = row[1] or 0.0
            avg_uncertainty = row[2] or 0.0
            avg_inference_ms = row[3] or 0.0

            cursor = self._conn.execute(
                "SELECT regime, COUNT(*) FROM decisions WHERE timestamp >= ? GROUP BY regime",
                (cutoff,),
            )
            regime_dist = {r[0]: r[1] for r in cursor.fetchall()}

        return {
            "total": total,
            "avg_confidence": round(avg_confidence, 4),
            "avg_uncertainty": round(avg_uncertainty, 4),
            "regime_distribution": regime_dist,
            "avg_inference_ms": round(avg_inference_ms, 2),
        }

    def close(self) -> None:
        """Flush remaining records and close the database connection."""
        with self._lock:
            self._flush_unlocked()
            self._conn.close()
        logger.info("DecisionLogger closed")

    # --- Internal ---------------------------------------------------------

    def _flush_unlocked(self) -> None:
        """Batch-insert buffered records. Caller must hold ``self._lock``."""
        if not self._buffer:
            return
        try:
            self._conn.executemany(_INSERT_SQL, [r.to_row() for r in self._buffer])
            self._conn.commit()
            logger.debug("Flushed decision records", count=len(self._buffer))
        except Exception as e:
            logger.error("Decision flush failed", error=str(e))
        finally:
            self._buffer.clear()
