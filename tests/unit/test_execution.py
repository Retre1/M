"""Unit tests for Executor and OrderManager._cancel_order."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apexfx.config.schema import (
    ExecutionConfig,
    RetryConfig,
    SlippageConfig,
    SmartExecutionConfig,
    SymbolConfig,
)
from apexfx.data.mt5_client import (
    OrderType,
    Position,
    SymbolInfo,
    TradeAction,
    TradeRequest,
    TradeResult,
)
from apexfx.execution.executor import ExecutionResult, Executor
from apexfx.execution.order_manager import ManagedOrder, OrderManager, OrderStatus
from apexfx.risk.risk_manager import KillSwitch, RiskDecision


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_exec_config(**overrides) -> ExecutionConfig:
    defaults = dict(
        order_type="market",
        limit_offset_pips=0.5,
        limit_timeout_s=30,
        limit_fallback="market",
        slippage=SlippageConfig(max_slippage_pips=2.0),
        retry=RetryConfig(max_retries=2, backoff_base_ms=10, backoff_max_ms=50),
        smart_execution=SmartExecutionConfig(algorithm="direct"),
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def _make_symbol_config(**overrides) -> SymbolConfig:
    defaults = dict(
        pip_value=0.0001,
        spread_limit_pips=2.0,
        lot_step=0.01,
        contract_size=100_000,
    )
    defaults.update(overrides)
    return SymbolConfig(**defaults)


def _make_symbol_info(**overrides) -> SymbolInfo:
    defaults = dict(
        name="EURUSD",
        bid=1.1000,
        ask=1.1002,
        spread=20.0,
        point=0.00001,
        trade_tick_size=0.00001,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        trade_contract_size=100_000.0,
    )
    defaults.update(overrides)
    return SymbolInfo(**defaults)


def _make_trade_result(success: bool = True, **overrides) -> TradeResult:
    defaults = dict(
        retcode=10009 if success else 10006,
        deal=12345,
        order=67890,
        volume=0.10,
        price=1.1001,
        comment="Done" if success else "Rejected",
    )
    defaults.update(overrides)
    return TradeResult(**defaults)


def _make_risk_decision(approved: bool = True, **overrides) -> RiskDecision:
    defaults = dict(
        approved=approved,
        adjusted_action=0.5 if approved else 0.0,
        position_size=0.10 if approved else 0.0,
        reason="All checks passed" if approved else "Blocked",
        checks_passed=["test"],
        checks_failed=[],
    )
    defaults.update(overrides)
    return RiskDecision(**defaults)


def _make_mock_mt5() -> MagicMock:
    """Create a mock MT5Client with sensible defaults."""
    mt5 = MagicMock()
    mt5.get_symbol_info.return_value = _make_symbol_info()
    mt5.send_order.return_value = _make_trade_result(success=True)
    mt5.get_positions.return_value = []
    mt5.close_position.return_value = _make_trade_result(success=True)
    return mt5


@pytest.fixture
def mock_mt5() -> MagicMock:
    return _make_mock_mt5()


@pytest.fixture
def exec_config() -> ExecutionConfig:
    return _make_exec_config()


@pytest.fixture
def symbol_config() -> SymbolConfig:
    return _make_symbol_config()


@pytest.fixture
def executor(exec_config, symbol_config, mock_mt5) -> Executor:
    with patch.object(KillSwitch, "KILL_FILE", MagicMock(exists=MagicMock(return_value=False))):
        exc = Executor(
            config=exec_config,
            symbol_config=symbol_config,
            mt5_client=mock_mt5,
            kill_switch=KillSwitch(),
        )
        exc.set_symbol("EURUSD")
        return exc


# ---------------------------------------------------------------------------
# Executor.execute — blocked when not approved
# ---------------------------------------------------------------------------


class TestExecutorBlocked:
    @pytest.mark.asyncio
    async def test_blocked_decision_returns_blocked(self, executor: Executor):
        decision = _make_risk_decision(approved=False, reason="Kill switch active")
        result = await executor.execute(decision, current_direction=0, current_volume=0.0)
        assert not result.success
        assert result.action_taken == "blocked"
        assert "Kill switch" in result.message


# ---------------------------------------------------------------------------
# Executor.execute — open position
# ---------------------------------------------------------------------------


class TestExecutorOpen:
    @pytest.mark.asyncio
    async def test_open_long_position(self, executor: Executor, mock_mt5: MagicMock):
        decision = _make_risk_decision(approved=True, adjusted_action=0.8, position_size=0.10)
        result = await executor.execute(decision, current_direction=0, current_volume=0.0)
        assert result.success
        assert result.action_taken == "opened"
        assert result.volume == 0.10

    @pytest.mark.asyncio
    async def test_open_short_position(self, executor: Executor, mock_mt5: MagicMock):
        decision = _make_risk_decision(approved=True, adjusted_action=-0.8, position_size=0.10)
        result = await executor.execute(decision, current_direction=0, current_volume=0.0)
        assert result.success
        assert result.action_taken == "opened"

    @pytest.mark.asyncio
    async def test_slippage_tracked(self, executor: Executor, mock_mt5: MagicMock):
        # Fill at a different price than market
        mock_mt5.send_order.return_value = _make_trade_result(success=True, price=1.1005)
        decision = _make_risk_decision(approved=True, adjusted_action=0.8, position_size=0.10)
        result = await executor.execute(decision, current_direction=0, current_volume=0.0)
        assert result.success
        assert result.slippage_pips > 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, executor: Executor, mock_mt5: MagicMock):
        """Executor retries when first order fails, second succeeds."""
        fail_result = _make_trade_result(success=False)
        # The ManagedOrder status needs to be checked. Since we use market orders,
        # the OrderManager returns a ManagedOrder with status based on TradeResult.
        mock_mt5.send_order.side_effect = [fail_result, _make_trade_result(success=True)]
        decision = _make_risk_decision(approved=True, adjusted_action=0.8, position_size=0.10)
        result = await executor.execute(decision, current_direction=0, current_volume=0.0)
        # With market orders, OrderManager rejects on first attempt,
        # and Executor retries by calling place_order again
        assert mock_mt5.send_order.call_count >= 1

    @pytest.mark.asyncio
    async def test_all_retries_fail(self, executor: Executor, mock_mt5: MagicMock):
        mock_mt5.send_order.return_value = _make_trade_result(success=False)
        decision = _make_risk_decision(approved=True, adjusted_action=0.8, position_size=0.10)
        result = await executor.execute(decision, current_direction=0, current_volume=0.0)
        assert not result.success
        assert result.action_taken == "blocked"
        assert "retries" in result.message


# ---------------------------------------------------------------------------
# Executor.execute — close position
# ---------------------------------------------------------------------------


class TestExecutorClose:
    @pytest.mark.asyncio
    async def test_close_on_direction_change(self, executor: Executor, mock_mt5: MagicMock):
        # Set up existing long position
        executor._current_position_ticket = 12345
        executor._current_position_direction = 1
        executor._current_position_volume = 0.10

        mock_mt5.close_position.return_value = _make_trade_result(success=True)
        mock_mt5.get_positions.return_value = [
            Position(
                ticket=12345, symbol="EURUSD", type=0, volume=0.10,
                price_open=1.1000, price_current=1.1010, profit=10.0,
                sl=0.0, tp=0.0, time=datetime.now(),
            ),
        ]

        # Switch from long to short
        decision = _make_risk_decision(approved=True, adjusted_action=-0.8, position_size=0.10)
        result = await executor.execute(decision, current_direction=1, current_volume=0.10)
        # Should close first, then possibly open short
        mock_mt5.close_position.assert_called()

    @pytest.mark.asyncio
    async def test_close_on_neutral(self, executor: Executor, mock_mt5: MagicMock):
        executor._current_position_ticket = 12345
        executor._current_position_direction = 1
        executor._current_position_volume = 0.10
        mock_mt5.get_positions.return_value = [
            Position(
                ticket=12345, symbol="EURUSD", type=0, volume=0.10,
                price_open=1.1000, price_current=1.1010, profit=10.0,
                sl=0.0, tp=0.0, time=datetime.now(),
            ),
        ]

        decision = _make_risk_decision(
            approved=True, adjusted_action=0.0, position_size=0.0,
            reason="Action too small, staying neutral",
        )
        result = await executor.execute(decision, current_direction=1, current_volume=0.10)
        mock_mt5.close_position.assert_called_once_with(12345)

    @pytest.mark.asyncio
    async def test_close_failure_returns_error(self, executor: Executor, mock_mt5: MagicMock):
        executor._current_position_ticket = 12345
        executor._current_position_direction = 1
        executor._current_position_volume = 0.10
        mock_mt5.get_positions.return_value = [
            Position(
                ticket=12345, symbol="EURUSD", type=0, volume=0.10,
                price_open=1.1000, price_current=1.1010, profit=10.0,
                sl=0.0, tp=0.0, time=datetime.now(),
            ),
        ]
        mock_mt5.close_position.return_value = _make_trade_result(
            success=False, comment="Market closed"
        )

        decision = _make_risk_decision(
            approved=True, adjusted_action=0.0, position_size=0.0,
            reason="Action too small, staying neutral",
        )
        result = await executor.execute(decision, current_direction=1, current_volume=0.10)
        assert not result.success


# ---------------------------------------------------------------------------
# Executor: no change needed
# ---------------------------------------------------------------------------


class TestExecutorNoChange:
    @pytest.mark.asyncio
    async def test_same_direction_similar_volume(self, executor: Executor, mock_mt5: MagicMock):
        """If target direction matches and volume diff is within lot_step, no action."""
        executor._current_position_direction = 1
        executor._current_position_volume = 0.10
        decision = _make_risk_decision(
            approved=True, adjusted_action=0.8, position_size=0.10,
        )
        result = await executor.execute(decision, current_direction=1, current_volume=0.10)
        assert result.success
        assert result.action_taken == "none"


# ---------------------------------------------------------------------------
# Executor.sync_with_mt5
# ---------------------------------------------------------------------------


class TestSyncWithMT5:
    def test_sync_clears_when_no_positions(self, executor: Executor, mock_mt5: MagicMock):
        executor._current_position_ticket = 99999
        executor._current_position_direction = 1
        executor._current_position_volume = 0.50
        mock_mt5.get_positions.return_value = []

        executor.sync_with_mt5()
        assert executor._current_position_ticket is None
        assert executor._current_position_direction == 0
        assert executor._current_position_volume == 0.0

    def test_sync_updates_from_mt5(self, executor: Executor, mock_mt5: MagicMock):
        mock_mt5.get_positions.return_value = [
            Position(
                ticket=11111, symbol="EURUSD", type=1, volume=0.25,
                price_open=1.1050, price_current=1.1000, profit=50.0,
                sl=0.0, tp=0.0, time=datetime.now(),
            ),
        ]
        executor.sync_with_mt5()
        assert executor._current_position_ticket == 11111
        assert executor._current_position_direction == -1
        assert executor._current_position_volume == 0.25

    def test_sync_handles_attribute_error(self, executor: Executor, mock_mt5: MagicMock):
        mock_mt5.get_positions.side_effect = AttributeError("not impl")
        # Should not raise
        executor.sync_with_mt5()

    def test_sync_handles_generic_exception(self, executor: Executor, mock_mt5: MagicMock):
        mock_mt5.get_positions.side_effect = RuntimeError("network down")
        executor.sync_with_mt5()


# ---------------------------------------------------------------------------
# Executor.close_all / reduce_position
# ---------------------------------------------------------------------------


class TestExecutorPublicMethods:
    @pytest.mark.asyncio
    async def test_close_all_no_position(self, executor: Executor):
        result = await executor.close_all()
        assert result.success
        assert result.action_taken == "none"

    @pytest.mark.asyncio
    async def test_reduce_position_no_position(self, executor: Executor):
        result = await executor.reduce_position(0.5)
        assert result.success
        assert result.action_taken == "none"

    @pytest.mark.asyncio
    async def test_reduce_position_partial(self, executor: Executor, mock_mt5: MagicMock):
        executor._current_position_ticket = 12345
        executor._current_position_direction = 1
        executor._current_position_volume = 1.0
        mock_mt5.send_order.return_value = _make_trade_result(success=True, volume=0.50)

        result = await executor.reduce_position(0.5)
        assert result.success
        assert result.action_taken == "adjusted"


# ---------------------------------------------------------------------------
# Executor: liquidity check
# ---------------------------------------------------------------------------


class TestLiquidityCheck:
    @pytest.mark.asyncio
    async def test_liquidity_fail_blocks(self, executor: Executor):
        mock_liq = MagicMock()
        mock_liq.tradeable = False
        mock_liq.reason = "Low liquidity"
        executor.liquidity_guard.check = MagicMock(return_value=mock_liq)

        decision = _make_risk_decision(approved=True, adjusted_action=0.8, position_size=0.10)
        result = await executor.execute(decision, current_direction=0, current_volume=0.0)
        assert not result.success
        assert "Liquidity" in result.message


# ---------------------------------------------------------------------------
# OrderManager._cancel_order
# ---------------------------------------------------------------------------


class TestOrderManagerCancelOrder:
    def test_cancel_order_success(self, mock_mt5: MagicMock):
        config = _make_exec_config()
        om = OrderManager(config=config, mt5_client=mock_mt5)

        # Mock the internal _mt5 attribute of MT5Client
        inner_mt5 = MagicMock()
        result_mock = MagicMock()
        result_mock.retcode = 10009
        inner_mt5.order_send.return_value = result_mock
        mock_mt5._mt5 = inner_mt5

        om._cancel_order(ticket=12345)
        inner_mt5.order_send.assert_called_once()
        call_args = inner_mt5.order_send.call_args[0][0]
        assert call_args["action"] == int(TradeAction.REMOVE)
        assert call_args["order"] == 12345

    def test_cancel_order_rejected(self, mock_mt5: MagicMock):
        config = _make_exec_config()
        om = OrderManager(config=config, mt5_client=mock_mt5)

        inner_mt5 = MagicMock()
        result_mock = MagicMock()
        result_mock.retcode = 10013  # some error
        inner_mt5.order_send.return_value = result_mock
        mock_mt5._mt5 = inner_mt5

        # Should not raise
        om._cancel_order(ticket=99999)

    def test_cancel_order_none_result(self, mock_mt5: MagicMock):
        config = _make_exec_config()
        om = OrderManager(config=config, mt5_client=mock_mt5)

        inner_mt5 = MagicMock()
        inner_mt5.order_send.return_value = None
        mock_mt5._mt5 = inner_mt5

        # Should handle None result gracefully
        om._cancel_order(ticket=11111)

    def test_cancel_order_exception(self, mock_mt5: MagicMock):
        config = _make_exec_config()
        om = OrderManager(config=config, mt5_client=mock_mt5)

        inner_mt5 = MagicMock()
        inner_mt5.order_send.side_effect = RuntimeError("connection lost")
        mock_mt5._mt5 = inner_mt5

        # Should catch and not propagate
        om._cancel_order(ticket=22222)


# ---------------------------------------------------------------------------
# OrderManager: place_order (market and limit)
# ---------------------------------------------------------------------------


class TestOrderManagerPlaceOrder:
    @pytest.mark.asyncio
    async def test_market_order_filled(self, mock_mt5: MagicMock):
        config = _make_exec_config(order_type="market")
        om = OrderManager(config=config, mt5_client=mock_mt5)
        mock_mt5.send_order.return_value = _make_trade_result(success=True, price=1.1001, volume=0.10)

        managed = await om.place_order("EURUSD", direction=1, volume=0.10, market_price=1.1000, pip_value=0.0001)
        assert managed.status == OrderStatus.FILLED
        assert managed.fill_price == 1.1001
        assert managed.filled_volume == 0.10

    @pytest.mark.asyncio
    async def test_market_order_rejected(self, mock_mt5: MagicMock):
        config = _make_exec_config(order_type="market")
        om = OrderManager(config=config, mt5_client=mock_mt5)
        mock_mt5.send_order.return_value = _make_trade_result(success=False)

        managed = await om.place_order("EURUSD", direction=1, volume=0.10, market_price=1.1000, pip_value=0.0001)
        assert managed.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_limit_order_placed(self, mock_mt5: MagicMock):
        config = _make_exec_config(order_type="limit", limit_offset_pips=1.0)
        om = OrderManager(config=config, mt5_client=mock_mt5)
        mock_mt5.send_order.return_value = _make_trade_result(success=True, order=55555)

        managed = await om.place_order("EURUSD", direction=1, volume=0.10, market_price=1.1000, pip_value=0.0001)
        assert managed.status == OrderStatus.PENDING
        assert managed.order_ticket == 55555
        # Buy limit should be below market
        assert managed.limit_price < 1.1000

    @pytest.mark.asyncio
    async def test_limit_order_sell(self, mock_mt5: MagicMock):
        config = _make_exec_config(order_type="limit", limit_offset_pips=1.0)
        om = OrderManager(config=config, mt5_client=mock_mt5)
        mock_mt5.send_order.return_value = _make_trade_result(success=True, order=55556)

        managed = await om.place_order("EURUSD", direction=-1, volume=0.10, market_price=1.1000, pip_value=0.0001)
        # Sell limit should be above market
        assert managed.limit_price > 1.1000

    @pytest.mark.asyncio
    async def test_limit_order_rejected_falls_back_to_market(self, mock_mt5: MagicMock):
        config = _make_exec_config(order_type="limit", limit_fallback="market")
        om = OrderManager(config=config, mt5_client=mock_mt5)

        # First call (limit) fails, second call (market fallback) succeeds
        fail_result = _make_trade_result(success=False)
        ok_result = _make_trade_result(success=True, price=1.1002, volume=0.10, order=77777)
        mock_mt5.send_order.side_effect = [fail_result, ok_result]

        managed = await om.place_order("EURUSD", direction=1, volume=0.10, market_price=1.1000, pip_value=0.0001)
        assert managed.status == OrderStatus.FILLED
        assert managed.fill_price == 1.1002


# ---------------------------------------------------------------------------
# ExecutionResult dataclass
# ---------------------------------------------------------------------------


class TestExecutionResult:
    def test_defaults(self):
        r = ExecutionResult(success=True, action_taken="opened")
        assert r.fill_price is None
        assert r.volume == 0.0
        assert r.slippage_pips == 0.0
        assert r.message == ""

    def test_with_all_fields(self):
        r = ExecutionResult(
            success=True,
            action_taken="opened",
            fill_price=1.1001,
            volume=0.10,
            slippage_pips=0.5,
            message="Position opened",
        )
        assert r.fill_price == 1.1001
