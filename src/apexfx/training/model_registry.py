"""Model version registry — tracks live models, promotion history, and enables rollback.

Uses SQLite for atomic persistence. Each promoted model gets a version record with
validation metrics and timestamp. Supports rollback to any previous version.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS model_versions (
    version_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_path      TEXT    NOT NULL,
    promoted_at     TEXT    NOT NULL,
    validation_sharpe REAL  NOT NULL,
    validation_return REAL  NOT NULL DEFAULT 0.0,
    is_active       INTEGER NOT NULL DEFAULT 0,
    notes           TEXT    NOT NULL DEFAULT ''
);
"""


@dataclass
class ModelVersion:
    """A single registered model version."""

    version_id: int
    model_path: str
    promoted_at: datetime
    validation_sharpe: float
    validation_return: float
    is_active: bool
    notes: str


def _row_to_version(row: sqlite3.Row) -> ModelVersion:
    """Convert a database row to a ModelVersion dataclass."""
    return ModelVersion(
        version_id=row["version_id"],
        model_path=row["model_path"],
        promoted_at=datetime.fromisoformat(row["promoted_at"]),
        validation_sharpe=row["validation_sharpe"],
        validation_return=row["validation_return"],
        is_active=bool(row["is_active"]),
        notes=row["notes"],
    )


class ModelRegistry:
    """Persistent model version registry backed by SQLite.

    Thread-safe via an internal lock; uses WAL mode for concurrent reads.
    """

    def __init__(self, db_path: str | Path = "data/model_registry.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        logger.info("Model registry initialised", db_path=str(self._db_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model_path: str,
        validation_sharpe: float,
        validation_return: float = 0.0,
        notes: str = "",
    ) -> ModelVersion:
        """Register a new model version and make it the active model.

        Deactivates any currently active version first.
        """
        now = datetime.now(UTC).isoformat()
        with self._lock:
            # Deactivate current active version
            self._conn.execute(
                "UPDATE model_versions SET is_active = 0 WHERE is_active = 1"
            )
            cursor = self._conn.execute(
                """\
                INSERT INTO model_versions
                    (model_path, promoted_at, validation_sharpe, validation_return, is_active, notes)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (model_path, now, validation_sharpe, validation_return, notes),
            )
            self._conn.commit()
            version_id = cursor.lastrowid

        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            promoted_at=datetime.fromisoformat(now),
            validation_sharpe=validation_sharpe,
            validation_return=validation_return,
            is_active=True,
            notes=notes,
        )
        logger.info(
            "Model registered",
            version_id=version_id,
            model_path=model_path,
            validation_sharpe=round(validation_sharpe, 4),
        )
        return version

    def get_active(self) -> ModelVersion | None:
        """Return the currently active model version, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM model_versions WHERE is_active = 1"
            ).fetchone()
        if row is None:
            return None
        return _row_to_version(row)

    def get_version(self, version_id: int) -> ModelVersion | None:
        """Return a specific version by ID, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM model_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_version(row)

    def get_history(self, limit: int = 20) -> list[ModelVersion]:
        """Return recent model versions, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM model_versions ORDER BY version_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_version(r) for r in rows]

    def rollback(self, to_version_id: int) -> ModelVersion:
        """Roll back to a previous model version.

        Deactivates the current active version and activates the target.
        Raises ValueError if the target version does not exist.
        """
        with self._lock:
            target = self._conn.execute(
                "SELECT * FROM model_versions WHERE version_id = ?",
                (to_version_id,),
            ).fetchone()
            if target is None:
                raise ValueError(f"Version {to_version_id} not found")

            self._conn.execute(
                "UPDATE model_versions SET is_active = 0 WHERE is_active = 1"
            )
            self._conn.execute(
                "UPDATE model_versions SET is_active = 1 WHERE version_id = ?",
                (to_version_id,),
            )
            self._conn.commit()

        version = _row_to_version(target)
        version.is_active = True
        logger.info(
            "Model rolled back",
            to_version_id=to_version_id,
            model_path=version.model_path,
        )
        return version

    def deactivate_current(self) -> None:
        """Deactivate the current active model (emergency stop)."""
        with self._lock:
            self._conn.execute(
                "UPDATE model_versions SET is_active = 0 WHERE is_active = 1"
            )
            self._conn.commit()
        logger.warning("All active models deactivated (emergency stop)")

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
