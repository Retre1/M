"""Limit order management: placement, monitoring, timeout, and fallback."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from apexfx.config.schema import ExecutionConfig
from apexfx.data.mt5_client import MT5Client, OrderType, TradeAction, TradeRequest
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


@dataclass
class ManagedOrder:
    """A limit order being managed by the order manager."""
    symbol: str
    direction: int  # +1 buy, -1 sell
    volume: float
    limit_price: float
    created_at: datetime
    timeout_s: int
    order_ticket: int | None = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    filled_volume: float = 0.0


class OrderManager:
    """
    Manages limit order lifecycle:
    - Place limit order at better price than market
    - Monitor for fill within timeout
    - Cancel and retry at market if timeout expires
    """

    def __init__(
        self,
        config: ExecutionConfig,
        mt5_client: MT5Client,
    ) -> None:
        self._config = config
        self._mt5 = mt5_client
        self._active_orders: dict[int, ManagedOrder] = {}

    async def place_order(
        self,
        symbol: str,
        direction: int,
        volume: float,
        market_price: float,
        pip_value: float,
    ) -> ManagedOrder:
        """
        Place a new order (limit or market based on config).
        Returns the ManagedOrder tracking object.
        """
        if self._config.order_type == "limit":
            return await self._place_limit_order(
                symbol, direction, volume, market_price, pip_value
            )
        else:
            return await self._place_market_order(symbol, direction, volume, market_price)

    async def _place_limit_order(
        self,
        symbol: str,
        direction: int,
        volume: float,
        market_price: float,
        pip_value: float,
    ) -> ManagedOrder:
        """Place a limit order at a better price than market."""
        offset = self._config.limit_offset_pips * pip_value

        if direction > 0:
            # Buy: place limit below market
            limit_price = market_price - offset
            order_type = OrderType.BUY_LIMIT
        else:
            # Sell: place limit above market
            limit_price = market_price + offset
            order_type = OrderType.SELL_LIMIT

        managed = ManagedOrder(
            symbol=symbol,
            direction=direction,
            volume=volume,
            limit_price=limit_price,
            created_at=datetime.now(timezone.utc),
            timeout_s=self._config.limit_timeout_s,
        )

        request = TradeRequest(
            action=TradeAction.PENDING,
            symbol=symbol,
            volume=volume,
            type=order_type,
            price=limit_price,
        )

        result = self._mt5.send_order(request)

        if result.success:
            managed.order_ticket = result.order
            managed.status = OrderStatus.PENDING
            self._active_orders[result.order] = managed
            logger.info(
                "Limit order placed",
                symbol=symbol,
                direction="BUY" if direction > 0 else "SELL",
                price=limit_price,
                volume=volume,
                ticket=result.order,
            )
        else:
            managed.status = OrderStatus.REJECTED
            logger.warning(
                "Limit order rejected",
                symbol=symbol,
                comment=result.comment,
            )

            # Fallback to market order if configured
            if self._config.limit_fallback == "market":
                return await self._place_market_order(symbol, direction, volume, market_price)

        return managed

    async def _place_market_order(
        self,
        symbol: str,
        direction: int,
        volume: float,
        price: float,
    ) -> ManagedOrder:
        """Place a market order."""
        order_type = OrderType.BUY if direction > 0 else OrderType.SELL

        managed = ManagedOrder(
            symbol=symbol,
            direction=direction,
            volume=volume,
            limit_price=price,
            created_at=datetime.now(timezone.utc),
            timeout_s=0,
        )

        request = TradeRequest(
            action=TradeAction.DEAL,
            symbol=symbol,
            volume=volume,
            type=order_type,
            price=price,
        )

        result = self._mt5.send_order(request)

        if result.success:
            managed.status = OrderStatus.FILLED
            managed.fill_price = result.price
            managed.filled_volume = result.volume
            managed.order_ticket = result.order
            logger.info(
                "Market order filled",
                symbol=symbol,
                direction="BUY" if direction > 0 else "SELL",
                price=result.price,
                volume=result.volume,
            )
        else:
            managed.status = OrderStatus.REJECTED
            logger.error("Market order rejected", comment=result.comment)

        return managed

    async def monitor_pending_orders(self) -> list[ManagedOrder]:
        """Check pending orders for fills or timeouts."""
        completed: list[ManagedOrder] = []
        now = datetime.now(timezone.utc)

        for ticket, order in list(self._active_orders.items()):
            if order.status != OrderStatus.PENDING:
                continue

            elapsed = (now - order.created_at).total_seconds()

            # Check if filled (via MT5 position check)
            positions = self._mt5.get_positions(order.symbol)
            is_filled = any(p.ticket == ticket for p in positions)

            if is_filled:
                order.status = OrderStatus.FILLED
                completed.append(order)
                del self._active_orders[ticket]
                logger.info("Limit order filled", ticket=ticket)
                continue

            # Check timeout
            if elapsed >= order.timeout_s:
                order.status = OrderStatus.TIMEOUT
                # Cancel the pending order
                self._cancel_order(ticket)

                if self._config.limit_fallback == "market":
                    logger.info("Limit order timeout, falling back to market", ticket=ticket)
                    info = self._mt5.get_symbol_info(order.symbol)
                    price = info.ask if order.direction > 0 else info.bid
                    market = await self._place_market_order(
                        order.symbol, order.direction, order.volume, price
                    )
                    completed.append(market)
                else:
                    logger.info("Limit order timeout, cancelled", ticket=ticket)
                    completed.append(order)

                del self._active_orders[ticket]

        return completed

    def _cancel_order(self, ticket: int) -> None:
        """Cancel a pending order."""
        try:
            request = TradeRequest(
                action=TradeAction.REMOVE,
                symbol="",
                volume=0,
                type=OrderType.BUY,
                price=0,
            )
            self._mt5.send_order(request)
            logger.debug("Order cancelled", ticket=ticket)
        except Exception as e:
            logger.error("Failed to cancel order", ticket=ticket, error=str(e))
