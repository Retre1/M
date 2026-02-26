"""Portfolio state management: positions, P&L, equity curve, persistence."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


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
    """Thread-safe portfolio state manager with disk persistence."""

    def __init__(self, state_file: str = "data/portfolio_state.json") -> None:
        self._state_file = Path(state_file)
        self._lock = threading.RLock()
        self._state = PortfolioState()

        # Try to load persisted state
        self._load_state()

    @property
    def state(self) -> PortfolioState:
        """Get a snapshot of current state (thread-safe read)."""
        with self._lock:
            return PortfolioState(**asdict(self._state))

    def update_equity(self, equity: float, unrealized_pnl: float = 0.0) -> None:
        """Update portfolio equity value."""
        with self._lock:
            self._state.equity = equity
            self._state.unrealized_pnl = unrealized_pnl
            self._state.peak_equity = max(self._state.peak_equity, equity)

            dd = (self._state.peak_equity - equity) / self._state.peak_equity
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
        """Save state to disk."""
        with self._lock:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump(asdict(self._state), f, indent=2, default=str)

    def _load_state(self) -> None:
        """Load state from disk if available."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    data = json.load(f)
                self._state = PortfolioState(**data)
                logger.info("State loaded from disk", file=str(self._state_file))
            except Exception as e:
                logger.warning("Failed to load state, starting fresh", error=str(e))
