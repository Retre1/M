"""Tests for base exchange types — invariants over the dataclasses.

These types are passed across the entire aggressive trading stack, so
a wrong default or a missed validation here ripples everywhere.  The
tests pin down:

* Side.opposite / Side.sign — used in PnL math
* OrderStatus.is_terminal — used to decide whether to poll an order again
* OrderRequest validation — refuses LIMIT without price, refuses zero quantity
* Position.signed_quantity — the only place the +long / −short convention lives
* Bar / Ticker derived properties — must be safe on degenerate inputs
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from apexfx.aggressive.exchanges.base import (
    Bar,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Ticker,
    TimeInForce,
)

UTC = timezone.utc


class TestSide:
    def test_opposite_buy(self) -> None:
        assert Side.BUY.opposite is Side.SELL

    def test_opposite_sell(self) -> None:
        assert Side.SELL.opposite is Side.BUY

    def test_sign_buy(self) -> None:
        assert Side.BUY.sign == 1

    def test_sign_sell(self) -> None:
        assert Side.SELL.sign == -1


class TestOrderStatus:
    @pytest.mark.parametrize("status,expected", [
        (OrderStatus.PENDING, False),
        (OrderStatus.OPEN, False),
        (OrderStatus.PARTIALLY_FILLED, False),
        (OrderStatus.FILLED, True),
        (OrderStatus.CANCELED, True),
        (OrderStatus.REJECTED, True),
    ])
    def test_is_terminal(self, status: OrderStatus, expected: bool) -> None:
        assert status.is_terminal is expected


class TestOrderRequestValidation:
    def test_market_order_no_price_ok(self) -> None:
        req = OrderRequest(
            symbol="BTC-USDT-SWAP",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
        )
        assert req.price is None

    def test_limit_order_requires_price(self) -> None:
        with pytest.raises(ValueError, match="LIMIT order requires price"):
            OrderRequest(
                symbol="BTC-USDT-SWAP",
                side=Side.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.01,
                price=None,
            )

    def test_zero_quantity_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            OrderRequest(
                symbol="BTC-USDT-SWAP",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=0.0,
            )

    def test_negative_quantity_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            OrderRequest(
                symbol="BTC-USDT-SWAP",
                side=Side.BUY,
                order_type=OrderType.MARKET,
                quantity=-0.01,
            )

    def test_optional_sl_tp_default_none(self) -> None:
        req = OrderRequest(
            symbol="BTC-USDT-SWAP",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.01,
        )
        assert req.stop_loss is None
        assert req.take_profit is None
        assert req.reduce_only is False

    def test_default_tif_gtc(self) -> None:
        req = OrderRequest(
            symbol="BTC-USDT-SWAP",
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            price=50000.0,
        )
        assert req.time_in_force is TimeInForce.GTC


class TestPosition:
    def test_long_signed_quantity_positive(self) -> None:
        p = Position(
            symbol="BTC-USDT-SWAP",
            side=Side.BUY,
            quantity=0.05,
            entry_price=50000.0,
            leverage=5.0,
            unrealized_pnl=10.0,
            timestamp=datetime.now(tz=UTC),
        )
        assert p.signed_quantity == 0.05
        assert not p.is_flat

    def test_short_signed_quantity_negative(self) -> None:
        p = Position(
            symbol="BTC-USDT-SWAP",
            side=Side.SELL,
            quantity=0.05,
            entry_price=50000.0,
            leverage=5.0,
            unrealized_pnl=-2.0,
            timestamp=datetime.now(tz=UTC),
        )
        assert p.signed_quantity == -0.05

    def test_zero_qty_is_flat(self) -> None:
        p = Position(
            symbol="BTC-USDT-SWAP",
            side=Side.BUY,
            quantity=0.0,
            entry_price=0.0,
            leverage=1.0,
            unrealized_pnl=0.0,
            timestamp=datetime.now(tz=UTC),
        )
        assert p.is_flat
        assert p.signed_quantity == 0.0


class TestBar:
    def test_range_pct_normal(self) -> None:
        bar = Bar(
            timestamp=datetime.now(tz=UTC),
            open=100.0, high=110.0, low=90.0, close=105.0, volume=1000.0,
        )
        assert bar.range_pct == pytest.approx((110 - 90) / 105)

    def test_range_pct_zero_close_safe(self) -> None:
        bar = Bar(
            timestamp=datetime.now(tz=UTC),
            open=0.0, high=0.0, low=0.0, close=0.0, volume=0.0,
        )
        assert bar.range_pct == 0.0


class TestTicker:
    def test_mid_normal(self) -> None:
        t = Ticker(
            symbol="BTC", last_price=50100, bid=50000, ask=50100,
            timestamp=datetime.now(tz=UTC),
        )
        assert t.mid == pytest.approx(50050.0)

    def test_mid_falls_back_to_last_when_book_empty(self) -> None:
        t = Ticker(
            symbol="BTC", last_price=50100, bid=0, ask=0,
            timestamp=datetime.now(tz=UTC),
        )
        assert t.mid == 50100.0

    def test_spread_pct_zero_safe(self) -> None:
        t = Ticker(
            symbol="BTC", last_price=0, bid=0, ask=0,
            timestamp=datetime.now(tz=UTC),
        )
        assert t.spread_pct == 0.0


class TestOrder:
    def test_remaining_quantity(self) -> None:
        o = Order(
            order_id="123", client_order_id=None, symbol="BTC-USDT-SWAP",
            side=Side.BUY, order_type=OrderType.LIMIT, status=OrderStatus.OPEN,
            quantity=0.10, filled_quantity=0.03, avg_fill_price=50000.0,
            price=49000.0, timestamp=datetime.now(tz=UTC),
        )
        assert o.remaining_quantity == pytest.approx(0.07)

    def test_remaining_quantity_clamped_at_zero(self) -> None:
        # Filled > quantity shouldn't ever happen but if exchange double-counts,
        # we don't return a negative remaining (would crash sizing math).
        o = Order(
            order_id="123", client_order_id=None, symbol="X",
            side=Side.BUY, order_type=OrderType.MARKET, status=OrderStatus.FILLED,
            quantity=0.10, filled_quantity=0.15, avg_fill_price=50000.0,
            price=None, timestamp=datetime.now(tz=UTC),
        )
        assert o.remaining_quantity == 0.0
