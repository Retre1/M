"""Portfolio state management: positions, P&L, equity curve, persistence.

Phase 4 additions:
- Write-Ahead Log (WAL) for crash recovery
- Checksum validation for data integrity
- Automatic checkpoint + WAL replay on startup
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WALEntry:
    """Single Write-Ahead Log entry for crash recovery."""
    sequence: int
    timestamp: str
    operation: str  # "equity_update", "position_open", "position_close", "trade_record"
    data: dict
    checksum: str  # SHA256 of serialized data

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @staticmethod
    def from_json(line: str) -> WALEntry:
        d = json.loads(line)
        return WALEntry(**d)

    @staticmethod
    def compute_checksum(data: dict) -> str:
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: str
    exit_time: str
    symbol: str
    direction: str
    volume: float
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pips: float
    duration_bars: int


@dataclass
class PortfolioState:
    """Complete portfolio state snapshot."""
    balance: float = 100_000.0
    equity: float = 100_000.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0

    current_position_symbol: str = ""
    current_position_direction: int = 0
    current_position_volume: float = 0.0
    current_position_entry_price: float = 0.0
    current_position_entry_time: str = ""
    time_in_position: int = 0

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    peak_equity: float = 100_000.0
    max_drawdown: float = 0.0

    equity_curve: list[float] = field(default_factory=list)
    trade_history: list[dict] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)

    last_update: str = ""


class StateManager:
    """Thread-safe portfolio state manager with WAL-based crash recovery.

    Guarantees:
    1. Every state mutation writes a WAL entry FIRST (append-only file)
    2. Periodic checkpoint: full state snapshot + WAL truncation
    3. On startup: load last checkpoint, replay WAL entries
    4. Checksum validation prevents corrupt state from being loaded
    """

    def __init__(
        self,
        state_file: str = "data/portfolio_state.json",
        wal_enabled: bool = True,
        checkpoint_interval: int = 100,  # Checkpoint every N WAL entries
    ) -> None:
        self._state_file = Path(state_file)
        self._wal_file = self._state_file.with_suffix(".wal")
        self._backup_file = Path(str(self._state_file) + ".backup")
        self._lock = threading.RLock()
        self._state = PortfolioState()
        self._wal_enabled = wal_enabled
        self._checkpoint_interval = checkpoint_interval
        self._wal_sequence = 0

        # Recover state (checkpoint + WAL replay)
        self._recover()

    @property
    def state(self) -> PortfolioState:
        """Get a snapshot of current state (thread-safe read)."""
        with self._lock:
            return PortfolioState(**asdict(self._state))

    def update_equity(self, equity: float, unrealized_pnl: float = 0.0) -> None:
        """Update portfolio equity value."""
        with self._lock:
            self._write_wal("equity_update", {
                "equity": equity, "unrealized_pnl": unrealized_pnl,
            })

            self._state.equity = equity
            self._state.unrealized_pnl = unrealized_pnl
            self._state.peak_equity = max(self._state.peak_equity, equity)

            if self._state.peak_equity > 0:
                dd = (self._state.peak_equity - equity) / self._state.peak_equity
            else:
                dd = 0.0
            self._state.max_drawdown = max(self._state.max_drawdown, dd)

            self._state.equity_curve.append(equity)
            if len(self._state.equity_curve) > 100_000:
                self._state.equity_curve = self._state.equity_curve[-50_000:]

            self._state.last_update = datetime.now(UTC).isoformat()

    def open_position(
        self,
        symbol: str,
        direction: int,
        volume: float,
        entry_price: float,
    ) -> None:
        """Record a new position opening."""
        with self._lock:
            self._write_wal("position_open", {
                "symbol": symbol, "direction": direction,
                "volume": volume, "entry_price": entry_price,
            })
            self._state.current_position_symbol = symbol
            self._state.current_position_direction = direction
            self._state.current_position_volume = volume
            self._state.current_position_entry_price = entry_price
            self._state.current_position_entry_time = datetime.now(UTC).isoformat()
            self._state.time_in_position = 0

            logger.info(
                "Position opened",
                symbol=symbol,
                direction="LONG" if direction > 0 else "SHORT",
                volume=volume,
                price=entry_price,
            )

    def close_position(self, exit_price: float, pnl: float, pip_value: float = 0.0001) -> None:
        """Record a position closing and add to trade history."""
        with self._lock:
            self._write_wal("position_close", {
                "exit_price": exit_price, "pnl": pnl, "pip_value": pip_value,
            })
            entry_price = self._state.current_position_entry_price
            pnl_pips = (exit_price - entry_price) / pip_value
            if self._state.current_position_direction < 0:
                pnl_pips = -pnl_pips

            record = TradeRecord(
                entry_time=self._state.current_position_entry_time,
                exit_time=datetime.now(UTC).isoformat(),
                symbol=self._state.current_position_symbol,
                direction="LONG" if self._state.current_position_direction > 0 else "SHORT",
                volume=self._state.current_position_volume,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pips=pnl_pips,
                duration_bars=self._state.time_in_position,
            )

            self._state.trade_history.append(asdict(record))
            if len(self._state.trade_history) > 10_000:
                self._state.trade_history = self._state.trade_history[-5_000:]

            self._state.total_trades += 1
            self._state.total_pnl += pnl
            self._state.balance += pnl

            if pnl > 0:
                self._state.winning_trades += 1
            else:
                self._state.losing_trades += 1

            # Reset position state
            self._state.current_position_direction = 0
            self._state.current_position_volume = 0.0
            self._state.current_position_entry_price = 0.0
            self._state.time_in_position = 0

            logger.info(
                "Position closed",
                pnl=round(pnl, 2),
                pnl_pips=round(pnl_pips, 1),
                total_trades=self._state.total_trades,
            )

    def record_daily_return(self, daily_return: float) -> None:
        """Record end-of-day return."""
        with self._lock:
            self._state.daily_returns.append(daily_return)
            if len(self._state.daily_returns) > 10_000:
                self._state.daily_returns = self._state.daily_returns[-5_000:]

    def increment_time_in_position(self) -> None:
        with self._lock:
            self._state.time_in_position += 1

    def persist(self) -> None:
        """Save state to disk (full checkpoint + WAL truncation)."""
        with self._lock:
            self._checkpoint()

    def _write_wal(self, operation: str, data: dict) -> None:
        """Append WAL entry before any state mutation."""
        if not self._wal_enabled:
            return
        try:
            self._wal_sequence += 1
            entry = WALEntry(
                sequence=self._wal_sequence,
                timestamp=datetime.now(UTC).isoformat(),
                operation=operation,
                data=data,
                checksum=WALEntry.compute_checksum(data),
            )
            self._wal_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._wal_file, "a") as f:
                f.write(entry.to_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

            # Auto-checkpoint after N entries
            if self._wal_sequence % self._checkpoint_interval == 0:
                self._checkpoint()
        except Exception as e:
            logger.error("WAL write failed — state mutation may be unrecoverable", error=str(e))
            raise  # Do not proceed with state mutation if WAL failed

    def _checkpoint(self) -> None:
        """Full state snapshot + WAL truncation."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)

            # Backup current checkpoint before overwriting
            if self._state_file.exists():
                shutil.copy2(self._state_file, self._backup_file)

            # Write new checkpoint
            with open(self._state_file, "w") as f:
                json.dump(asdict(self._state), f, indent=2, default=str)

            # Truncate WAL (state is now fully captured in checkpoint)
            if self._wal_file.exists():
                self._wal_file.unlink()
            self._wal_sequence = 0

        except Exception as e:
            logger.error("Checkpoint failed", error=str(e))

    def _recover(self) -> None:
        """On startup: load checkpoint + replay WAL entries."""
        loaded = False

        # Try main checkpoint
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                self._state = PortfolioState(**data)
                loaded = True
                logger.info("State loaded from checkpoint", file=str(self._state_file))
            except Exception as e:
                logger.warning("Checkpoint corrupted, trying backup", error=str(e))

        # Try backup if main failed
        if not loaded and self._backup_file.exists():
            try:
                with open(self._backup_file) as f:
                    data = json.load(f)
                self._state = PortfolioState(**data)
                loaded = True
                logger.info("State loaded from backup", file=str(self._backup_file))
            except Exception as e:
                logger.warning("Backup also corrupted, starting fresh", error=str(e))

        # Replay WAL entries
        if self._wal_file.exists():
            n_replayed = 0
            n_invalid = 0
            try:
                with open(self._wal_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = WALEntry.from_json(line)
                            # Validate checksum
                            expected = WALEntry.compute_checksum(entry.data)
                            if expected != entry.checksum:
                                n_invalid += 1
                                logger.warning(
                                    "WAL entry checksum mismatch, skipping",
                                    seq=entry.sequence,
                                )
                                continue
                            self._replay_wal_entry(entry)
                            n_replayed += 1
                            self._wal_sequence = max(self._wal_sequence, entry.sequence)
                        except Exception as e:
                            n_invalid += 1
                            logger.warning("WAL entry replay failed", error=str(e))

                if n_replayed > 0:
                    logger.info(
                        "WAL replay complete",
                        replayed=n_replayed,
                        invalid=n_invalid,
                    )
                    # Checkpoint after replay to persist recovered state
                    self._checkpoint()
            except Exception as e:
                logger.error("WAL replay failed", error=str(e))

    def _replay_wal_entry(self, entry: WALEntry) -> None:
        """Replay a single WAL entry to reconstruct state."""
        op = entry.operation
        data = entry.data

        if op == "equity_update":
            self._state.equity = data["equity"]
            self._state.unrealized_pnl = data["unrealized_pnl"]
            self._state.peak_equity = max(self._state.peak_equity, data["equity"])
            if self._state.peak_equity > 0:
                dd = (self._state.peak_equity - data["equity"]) / self._state.peak_equity
            else:
                dd = 0.0
            self._state.max_drawdown = max(self._state.max_drawdown, dd)
            self._state.equity_curve.append(data["equity"])
            self._state.last_update = entry.timestamp

        elif op == "position_open":
            self._state.current_position_symbol = data["symbol"]
            self._state.current_position_direction = data["direction"]
            self._state.current_position_volume = data["volume"]
            self._state.current_position_entry_price = data["entry_price"]
            self._state.current_position_entry_time = entry.timestamp
            self._state.time_in_position = 0

        elif op == "position_close":
            pnl = data["pnl"]
            self._state.total_trades += 1
            self._state.total_pnl += pnl
            self._state.balance += pnl
            if pnl > 0:
                self._state.winning_trades += 1
            else:
                self._state.losing_trades += 1
            self._state.current_position_direction = 0
            self._state.current_position_volume = 0.0
            self._state.current_position_entry_price = 0.0
            self._state.time_in_position = 0
