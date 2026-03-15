"""Order execution engine: translates risk-approved signals into MT5 orders."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from apexfx.config.schema import ExecutionConfig, SymbolConfig
from apexfx.data.mt5_client import MT5Client, OrderType, TradeAction, TradeRequest
from apexfx.execution.fill_tracker import FillTracker
from apexfx.execution.liquidity_guard import LiquidityGuard
from apexfx.execution.order_manager import OrderManager
from apexfx.execution.smart_exec import SmartRouter
from apexfx.risk.risk_manager import KillSwitch, MarketState, RiskDecision
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of an execution attempt."""
    success: bool
    action_taken: str  # "opened", "closed", "adjusted", "blocked", "none"
    fill_price: float | None = None
    volume: float = 0.0
    slippage_pips: float = 0.0
    message: str = ""


class Executor:
    """
    Order execution engine.
    Takes RiskDecision and translates into actual MT5 orders.
    """

    def __init__(
        self,
        config: ExecutionConfig,
        symbol_config: SymbolConfig,
        mt5_client: MT5Client,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self._config = config
        self._symbol_config = symbol_config
        self._mt5 = mt5_client
        self._symbol = ""

        self.liquidity_guard = LiquidityGuard(config, symbol_config, mt5_client)
        self.order_manager = OrderManager(config, mt5_client)
        self.fill_tracker = FillTracker(pip_value=symbol_config.pip_value)

        self._kill_switch = kill_switch

        # Smart order router for algorithmic execution
        se_cfg = getattr(config, "smart_execution", None)
        if se_cfg and se_cfg.algorithm != "direct":
            self._smart_router = SmartRouter(
                twap_threshold=se_cfg.twap_threshold_lots,
                vwap_threshold=se_cfg.vwap_threshold_lots,
                is_threshold=se_cfg.is_threshold_lots,
            )
            self._smart_urgency = se_cfg.urgency
        else:
            self._smart_router: SmartRouter | None = None
            self._smart_urgency = 0.5

        self._current_position_ticket: int | None = None
        self._current_position_direction: int = 0
        self._current_position_volume: float = 0.0

    def set_symbol(self, symbol: str) -> None:
        self._symbol = symbol

    def sync_with_mt5(self) -> None:
        """Reconcile internal state with MT5 positions.

        Queries MT5 for actual open positions and updates internal tracking
        state accordingly.  This prevents state desync if MT5 closes a
        position externally (stop-out, margin call, admin action).
        """
        try:
            positions = self._mt5.get_positions(symbol=self._symbol)
            if not positions:
                if self._current_position_ticket is not None:
                    logger.warning("Position disappeared from MT5, syncing state")
                self._current_position_ticket = None
                self._current_position_direction = 0
                self._current_position_volume = 0.0
            else:
                pos = positions[0]
                self._current_position_ticket = pos.ticket
                self._current_position_volume = pos.volume
                self._current_position_direction = 1 if pos.type == 0 else -1
        except AttributeError:
            # MT5Client may not implement get_positions yet
            logger.debug("MT5Client.get_positions not available, skipping sync")
        except Exception as e:
            logger.warning("MT5 sync failed, using cached state", error=str(e))

    async def execute(
        self,
        risk_decision: RiskDecision,
        current_direction: int,
        current_volume: float,
    ) -> ExecutionResult:
        """
        Execute the risk-approved action.

        Args:
            risk_decision: Output from RiskManager.evaluate_action()
            current_direction: Current position direction (-1, 0, +1)
            current_volume: Current position volume in lots
        """
        if not risk_decision.approved:
            return ExecutionResult(
                success=False,
                action_taken="blocked",
                message=risk_decision.reason,
            )

        action = risk_decision.adjusted_action
        target_volume = risk_decision.position_size

        # Check liquidity before execution
        liquidity = self.liquidity_guard.check(self._symbol)
        if not liquidity.tradeable:
            return ExecutionResult(
                success=False,
                action_taken="blocked",
                message=f"Liquidity check failed: {liquidity.reason}",
            )

        # Determine target direction
        if abs(action) < 0.05:
            target_direction = 0
        else:
            target_direction = 1 if action > 0 else -1

        # --- Close position if direction changes ---
        if current_direction != 0 and target_direction != current_direction:
            close_result = await self._close_position()
            if not close_result.success:
                return close_result
            current_direction = 0
            current_volume = 0.0

        # --- Close if going neutral ---
        if target_direction == 0 and current_direction != 0:
            return await self._close_position()

        # --- Open new position ---
        if target_direction != 0 and current_direction == 0:
            return await self._open_position(target_direction, target_volume)

        # --- Adjust existing position ---
        if target_direction != 0 and current_direction == target_direction:
            volume_diff = target_volume - current_volume
            if abs(volume_diff) > self._symbol_config.lot_step:
                if volume_diff > 0:
                    return await self._open_position(target_direction, volume_diff)
                else:
                    return await self._partial_close(abs(volume_diff))

        return ExecutionResult(
            success=True,
            action_taken="none",
            message="No position change needed",
        )

    async def _open_position(
        self, direction: int, volume: float
    ) -> ExecutionResult:
        """Open a new position with retry logic."""
        # Sync with MT5 before opening to avoid stale state
        self.sync_with_mt5()

        info = self._mt5.get_symbol_info(self._symbol)
        market_price = info.ask if direction > 0 else info.bid

        for attempt in range(self._config.retry.max_retries):
            start_time = time.monotonic()

            managed = await self.order_manager.place_order(
                symbol=self._symbol,
                direction=direction,
                volume=volume,
                market_price=market_price,
                pip_value=self._symbol_config.pip_value,
            )

            fill_time_ms = (time.monotonic() - start_time) * 1000

            if managed.status.value == "filled":
                fill_price = managed.fill_price or market_price
                slippage = abs(fill_price - market_price) / self._symbol_config.pip_value

                # Check slippage limit
                if slippage > self._config.slippage.max_slippage_pips:
                    logger.warning(
                        "Slippage exceeded limit",
                        slippage=slippage,
                        limit=self._config.slippage.max_slippage_pips,
                    )

                self.fill_tracker.record_fill(
                    symbol=self._symbol,
                    direction="BUY" if direction > 0 else "SELL",
                    expected_price=market_price,
                    actual_price=fill_price,
                    volume=managed.filled_volume,
                    fill_time_ms=fill_time_ms,
                    order_type=self._config.order_type,
                )

                self._current_position_direction = direction
                self._current_position_volume = managed.filled_volume
                self._current_position_ticket = managed.order_ticket

                # Record successful fill in kill switch
                if self._kill_switch is not None:
                    self._kill_switch.record_success()

                return ExecutionResult(
                    success=True,
                    action_taken="opened",
                    fill_price=fill_price,
                    volume=managed.filled_volume,
                    slippage_pips=slippage,
                    message=f"Position opened at {fill_price}",
                )

            # Retry with backoff
            if attempt < self._config.retry.max_retries - 1:
                backoff_ms = min(
                    self._config.retry.backoff_base_ms * (2 ** attempt),
                    self._config.retry.backoff_max_ms,
                )
                logger.info(
                    "Order retry",
                    attempt=attempt + 1,
                    backoff_ms=backoff_ms,
                )
                await asyncio.sleep(backoff_ms / 1000)

        self.fill_tracker.record_rejection()
        # Record rejection in kill switch
        if self._kill_switch is not None:
            self._kill_switch.record_rejection()
        return ExecutionResult(
            success=False,
            action_taken="blocked",
            message=f"Order failed after {self._config.retry.max_retries} retries",
        )

    async def close_all(self) -> ExecutionResult:
        """Public method: close the entire current position."""
        return await self._close_position()

    async def reduce_position(self, fraction: float = 0.5) -> ExecutionResult:
        """Public method: reduce current position by a fraction.

        Args:
            fraction: Fraction to close (0.5 = close half).
        """
        if self._current_position_volume <= 0:
            return ExecutionResult(success=True, action_taken="none", message="No position to reduce")
        close_volume = self._current_position_volume * fraction
        close_volume = max(close_volume, self._symbol_config.lot_step)
        return await self._partial_close(close_volume)

    async def _close_position(self) -> ExecutionResult:
        """Close the current position."""
        # Sync with MT5 before closing to avoid stale state
        self.sync_with_mt5()

        if self._current_position_ticket is None:
            return ExecutionResult(success=True, action_taken="none", message="No position to close")

        result = self._mt5.close_position(self._current_position_ticket)

        if result.success:
            self._current_position_direction = 0
            self._current_position_volume = 0.0
            self._current_position_ticket = None
            return ExecutionResult(
                success=True,
                action_taken="closed",
                fill_price=result.price,
                volume=result.volume,
                message="Position closed",
            )

        return ExecutionResult(
            success=False,
            action_taken="blocked",
            message=f"Close failed: {result.comment}",
        )

    async def _partial_close(self, volume: float) -> ExecutionResult:
        """Partially close the current position."""
        if self._current_position_ticket is None:
            return ExecutionResult(success=True, action_taken="none")

        close_dir = -self._current_position_direction
        info = self._mt5.get_symbol_info(self._symbol)
        price = info.bid if close_dir < 0 else info.ask

        order_type = OrderType.SELL if close_dir < 0 else OrderType.BUY
        request = TradeRequest(
            action=TradeAction.DEAL,
            symbol=self._symbol,
            volume=volume,
            type=order_type,
            price=price,
            comment="ApexFX Partial Close",
        )

        result = self._mt5.send_order(request)

        if result.success:
            self._current_position_volume -= volume
            if self._current_position_volume <= 0:
                self._current_position_direction = 0
                self._current_position_ticket = None

            return ExecutionResult(
                success=True,
                action_taken="adjusted",
                fill_price=result.price,
                volume=volume,
                message=f"Partially closed {volume} lots",
            )

        return ExecutionResult(
            success=False,
            action_taken="blocked",
            message=f"Partial close failed: {result.comment}",
        )
