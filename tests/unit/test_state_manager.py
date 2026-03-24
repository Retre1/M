"""Unit tests for StateManager — WAL, checkpoint, recovery, division-by-zero protection."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest

from apexfx.live.state_manager import PortfolioState, StateManager, WALEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for state files."""
    return tmp_path / "state"


@pytest.fixture
def state_file(tmp_state_dir: Path) -> str:
    return str(tmp_state_dir / "portfolio_state.json")


@pytest.fixture
def sm(state_file: str) -> StateManager:
    """Create a fresh StateManager with WAL enabled."""
    return StateManager(state_file=state_file, wal_enabled=True, checkpoint_interval=50)


@pytest.fixture
def sm_no_wal(state_file: str) -> StateManager:
    """StateManager with WAL disabled."""
    return StateManager(state_file=state_file, wal_enabled=False, checkpoint_interval=50)


# ---------------------------------------------------------------------------
# WALEntry unit tests
# ---------------------------------------------------------------------------


class TestWALEntry:
    def test_compute_checksum_deterministic(self):
        data = {"equity": 100_000.0, "unrealized_pnl": 0.0}
        cs1 = WALEntry.compute_checksum(data)
        cs2 = WALEntry.compute_checksum(data)
        assert cs1 == cs2
        assert len(cs1) == 16

    def test_compute_checksum_key_order_insensitive(self):
        data_a = {"a": 1, "b": 2}
        data_b = {"b": 2, "a": 1}
        assert WALEntry.compute_checksum(data_a) == WALEntry.compute_checksum(data_b)

    def test_roundtrip_json(self):
        data = {"equity": 99_000.0}
        entry = WALEntry(
            sequence=1,
            timestamp="2024-01-01T00:00:00",
            operation="equity_update",
            data=data,
            checksum=WALEntry.compute_checksum(data),
        )
        line = entry.to_json()
        recovered = WALEntry.from_json(line)
        assert recovered.sequence == 1
        assert recovered.data == data
        assert recovered.checksum == entry.checksum


# ---------------------------------------------------------------------------
# PortfolioState defaults
# ---------------------------------------------------------------------------


class TestPortfolioState:
    def test_default_values(self):
        ps = PortfolioState()
        assert ps.balance == 100_000.0
        assert ps.equity == 100_000.0
        assert ps.total_trades == 0
        assert ps.equity_curve == []
        assert ps.max_drawdown == 0.0

    def test_serialization_roundtrip(self):
        ps = PortfolioState(balance=50_000.0, total_trades=5)
        d = asdict(ps)
        ps2 = PortfolioState(**d)
        assert ps2.balance == 50_000.0
        assert ps2.total_trades == 5


# ---------------------------------------------------------------------------
# StateManager: update_equity
# ---------------------------------------------------------------------------


class TestUpdateEquity:
    def test_equity_updates_state(self, sm: StateManager):
        sm.update_equity(101_000.0, unrealized_pnl=1_000.0)
        s = sm.state
        assert s.equity == 101_000.0
        assert s.unrealized_pnl == 1_000.0

    def test_peak_equity_tracks_max(self, sm: StateManager):
        sm.update_equity(110_000.0)
        sm.update_equity(105_000.0)
        s = sm.state
        assert s.peak_equity == 110_000.0

    def test_drawdown_computed_correctly(self, sm: StateManager):
        sm.update_equity(100_000.0)
        sm.update_equity(95_000.0)
        s = sm.state
        expected_dd = 5_000.0 / 100_000.0
        assert abs(s.max_drawdown - expected_dd) < 1e-10

    def test_division_by_zero_peak_equity_zero(self, sm: StateManager):
        """If peak_equity is 0, drawdown should not raise ZeroDivisionError."""
        sm._state.peak_equity = 0.0
        # This should not raise
        sm.update_equity(0.0)
        s = sm.state
        assert s.max_drawdown == 0.0

    def test_equity_curve_appended(self, sm: StateManager):
        sm.update_equity(100_500.0)
        sm.update_equity(101_000.0)
        s = sm.state
        assert s.equity_curve == [100_500.0, 101_000.0]

    def test_equity_curve_truncated_at_limit(self, sm: StateManager):
        """Equity curve should be truncated when it exceeds 100k entries."""
        sm._state.equity_curve = list(range(100_000))
        sm.update_equity(999.0)
        s = sm.state
        # After truncation: curve should be significantly smaller than 100k
        assert len(s.equity_curve) <= 50_001
        assert len(s.equity_curve) >= 50_000


# ---------------------------------------------------------------------------
# StateManager: open_position / close_position
# ---------------------------------------------------------------------------


class TestPositionLifecycle:
    def test_open_position(self, sm: StateManager):
        sm.open_position("EURUSD", direction=1, volume=0.10, entry_price=1.1050)
        s = sm.state
        assert s.current_position_symbol == "EURUSD"
        assert s.current_position_direction == 1
        assert s.current_position_volume == 0.10
        assert s.current_position_entry_price == 1.1050
        assert s.time_in_position == 0

    def test_close_position_updates_stats(self, sm: StateManager):
        sm.open_position("EURUSD", direction=1, volume=0.10, entry_price=1.1000)
        sm.close_position(exit_price=1.1050, pnl=50.0, pip_value=0.0001)
        s = sm.state
        assert s.total_trades == 1
        assert s.winning_trades == 1
        assert s.losing_trades == 0
        assert s.total_pnl == 50.0
        assert s.balance == 100_050.0
        # Position cleared
        assert s.current_position_direction == 0
        assert s.current_position_volume == 0.0

    def test_close_losing_trade(self, sm: StateManager):
        sm.open_position("EURUSD", direction=1, volume=0.10, entry_price=1.1000)
        sm.close_position(exit_price=1.0950, pnl=-50.0, pip_value=0.0001)
        s = sm.state
        assert s.losing_trades == 1
        assert s.winning_trades == 0
        assert s.total_pnl == -50.0

    def test_close_short_position_pnl_pips(self, sm: StateManager):
        sm.open_position("EURUSD", direction=-1, volume=0.10, entry_price=1.1000)
        sm.close_position(exit_price=1.0950, pnl=50.0, pip_value=0.0001)
        s = sm.state
        # For SHORT: pnl_pips = -((exit - entry) / pip_value) = -((1.0950 - 1.1000) / 0.0001) = 50 pips
        last_trade = s.trade_history[-1]
        assert last_trade["pnl_pips"] == pytest.approx(50.0, rel=1e-5)

    def test_trade_history_truncated(self, sm: StateManager):
        sm._state.trade_history = [{"fake": i} for i in range(10_000)]
        sm.open_position("EURUSD", direction=1, volume=0.01, entry_price=1.0)
        sm.close_position(exit_price=1.0, pnl=0.0)
        s = sm.state
        assert len(s.trade_history) <= 5_001

    def test_increment_time_in_position(self, sm: StateManager):
        sm.open_position("EURUSD", direction=1, volume=0.01, entry_price=1.0)
        sm.increment_time_in_position()
        sm.increment_time_in_position()
        assert sm.state.time_in_position == 2


# ---------------------------------------------------------------------------
# StateManager: WAL write and checkpoint
# ---------------------------------------------------------------------------


class TestWAL:
    def test_wal_file_created_on_mutation(self, sm: StateManager, state_file: str):
        sm.update_equity(99_000.0)
        wal_path = Path(state_file).with_suffix(".wal")
        assert wal_path.exists()
        content = wal_path.read_text()
        assert "equity_update" in content

    def test_wal_disabled_no_file(self, sm_no_wal: StateManager, state_file: str):
        sm_no_wal.update_equity(99_000.0)
        wal_path = Path(state_file).with_suffix(".wal")
        assert not wal_path.exists()

    def test_auto_checkpoint_truncates_wal(self, state_file: str):
        """WAL should be truncated after checkpoint_interval entries."""
        sm = StateManager(state_file=state_file, wal_enabled=True, checkpoint_interval=5)
        wal_path = Path(state_file).with_suffix(".wal")

        for i in range(5):
            sm.update_equity(100_000.0 + i)

        # After 5 entries, checkpoint should have fired and WAL should be truncated
        assert not wal_path.exists() or wal_path.read_text().strip() == ""

    def test_persist_creates_checkpoint(self, sm: StateManager, state_file: str):
        sm.update_equity(99_500.0)
        sm.persist()
        sf = Path(state_file)
        assert sf.exists()
        data = json.loads(sf.read_text())
        assert data["equity"] == 99_500.0

    def test_persist_truncates_wal(self, sm: StateManager, state_file: str):
        sm.update_equity(99_000.0)
        wal_path = Path(state_file).with_suffix(".wal")
        assert wal_path.exists()
        sm.persist()
        assert not wal_path.exists()


# ---------------------------------------------------------------------------
# StateManager: Recovery
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_recover_from_checkpoint(self, state_file: str):
        """Create state, persist, then recover in a new StateManager."""
        sm1 = StateManager(state_file=state_file, wal_enabled=True)
        sm1.update_equity(88_000.0)
        sm1.open_position("GBPUSD", direction=-1, volume=0.50, entry_price=1.2500)
        sm1.persist()

        sm2 = StateManager(state_file=state_file, wal_enabled=True)
        s = sm2.state
        assert s.equity == 88_000.0
        assert s.current_position_symbol == "GBPUSD"
        assert s.current_position_direction == -1

    def test_recover_from_wal_replay(self, state_file: str):
        """Recovery should replay WAL entries after the last checkpoint."""
        sm1 = StateManager(state_file=state_file, wal_enabled=True, checkpoint_interval=1000)
        sm1.persist()  # empty checkpoint

        sm1.update_equity(95_000.0, unrealized_pnl=-5_000.0)
        sm1.open_position("EURUSD", direction=1, volume=0.20, entry_price=1.0800)

        # Do NOT persist — so WAL has the entries but checkpoint is stale
        sm2 = StateManager(state_file=state_file, wal_enabled=True, checkpoint_interval=1000)
        s = sm2.state
        assert s.equity == 95_000.0
        assert s.current_position_symbol == "EURUSD"
        assert s.current_position_direction == 1

    def test_recover_from_backup_when_main_corrupt(self, state_file: str):
        """If main checkpoint is corrupt, recovery should fall back to backup."""
        sm1 = StateManager(state_file=state_file, wal_enabled=True)
        sm1.update_equity(77_000.0)
        sm1.persist()

        # Now corrupt the main file
        backup_path = Path(state_file + ".backup")
        main_path = Path(state_file)

        # Backup should exist after a second persist
        sm1.update_equity(78_000.0)
        sm1.persist()
        assert backup_path.exists()

        # Corrupt main file
        main_path.write_text("NOT_VALID_JSON!!!")

        sm2 = StateManager(state_file=state_file, wal_enabled=True)
        s = sm2.state
        # Should have loaded from backup (which has the state from first persist)
        assert s.equity == 77_000.0

    def test_recover_fresh_when_no_files(self, state_file: str):
        """If no checkpoint or WAL, start fresh with defaults."""
        sm = StateManager(state_file=state_file, wal_enabled=True)
        s = sm.state
        assert s.balance == 100_000.0
        assert s.equity == 100_000.0
        assert s.total_trades == 0

    def test_wal_checksum_mismatch_skips_entry(self, state_file: str):
        """WAL entries with bad checksums should be skipped during replay."""
        sm1 = StateManager(state_file=state_file, wal_enabled=True, checkpoint_interval=1000)
        sm1.persist()  # empty checkpoint

        # Manually write a bad WAL entry
        wal_path = Path(state_file).with_suffix(".wal")
        wal_path.parent.mkdir(parents=True, exist_ok=True)
        bad_entry = WALEntry(
            sequence=1,
            timestamp="2024-01-01T00:00:00",
            operation="equity_update",
            data={"equity": 50_000.0, "unrealized_pnl": 0.0},
            checksum="BAD_CHECKSUM_1234",
        )
        wal_path.write_text(bad_entry.to_json() + "\n")

        sm2 = StateManager(state_file=state_file, wal_enabled=True, checkpoint_interval=1000)
        # Bad entry should be skipped, equity remains default
        assert sm2.state.equity == 100_000.0


# ---------------------------------------------------------------------------
# StateManager: record_daily_return
# ---------------------------------------------------------------------------


class TestDailyReturn:
    def test_record_daily_return(self, sm: StateManager):
        sm.record_daily_return(0.01)
        sm.record_daily_return(-0.005)
        assert sm.state.daily_returns == [0.01, -0.005]

    def test_daily_returns_truncated(self, sm: StateManager):
        sm._state.daily_returns = list(range(10_000))
        sm.record_daily_return(0.001)
        assert len(sm.state.daily_returns) <= 5_001


# ---------------------------------------------------------------------------
# Thread safety (basic check)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_state_property_returns_copy(self, sm: StateManager):
        """Modifying the returned state should not affect internal state."""
        s = sm.state
        s.balance = 0.0
        assert sm.state.balance == 100_000.0
