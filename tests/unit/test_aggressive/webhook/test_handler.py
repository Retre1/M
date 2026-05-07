"""Tests for SignalHandler — TV alert → OKX order translation.

We use a hand-rolled mock ``Exchange`` so tests:
  * never touch the network
  * deterministically control what ``place_order`` / ``get_position`` return
  * verify the exact ``OrderRequest`` we built (assert on captured args)
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from apexfx.aggressive.exchanges.base import (
    InsufficientFundsError,
    Order,
    OrderRequest,
    OrderRejectedError,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from apexfx.aggressive.webhook.handler import (
    SignalHandler,
    alert_side_to_okx_side,
    exit_side_to_okx_side,
    tv_symbol_to_okx,
)
from apexfx.aggressive.webhook.models import (
    AlertAction,
    AlertSide,
    ExitReason,
    TradingViewAlert,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alert(**overrides) -> TradingViewAlert:
    base = {
        "action": "entry", "symbol": "BTCUSDT.P", "side": "long",
        "account": "main", "unit": 1, "size": 0.05,
        "price": 50000.0, "sl": 48500.0,
    }
    base.update(overrides)
    return TradingViewAlert.model_validate(base)


def _make_order(**overrides) -> Order:
    base = dict(
        order_id="ord-1", client_order_id="apx-x",
        symbol="BTC-USDT-SWAP", side=Side.BUY,
        order_type=OrderType.MARKET, status=OrderStatus.OPEN,
        quantity=0.05, filled_quantity=0.0, avg_fill_price=0.0,
        price=None, timestamp=datetime.now(tz=UTC),
    )
    base.update(overrides)
    return Order(**base)


def _make_exchange_mock(position: Position | None = None) -> MagicMock:
    ex = MagicMock()
    ex.place_order.return_value = _make_order()
    ex.get_position.return_value = position
    return ex


# ---------------------------------------------------------------------------
# Symbol translation
# ---------------------------------------------------------------------------


class TestSymbolTranslation:
    @pytest.mark.parametrize("tv,okx", [
        ("BTCUSDT.P", "BTC-USDT-SWAP"),
        ("ETHUSDT.P", "ETH-USDT-SWAP"),
        ("SOLUSDT.P", "SOL-USDT-SWAP"),
        ("BTC-USDT-SWAP", "BTC-USDT-SWAP"),  # already OKX form
        ("BTCUSDT", "BTC-USDT-SWAP"),       # TV without .P
        ("BTCUSDC.P", "BTC-USDC-SWAP"),     # USDC
        ("btcusdt.p", "BTC-USDT-SWAP"),     # lowercase
    ])
    def test_translation_table(self, tv: str, okx: str) -> None:
        assert tv_symbol_to_okx(tv) == okx

    def test_alert_side_long_to_buy(self) -> None:
        assert alert_side_to_okx_side(AlertSide.LONG) is Side.BUY

    def test_alert_side_short_to_sell(self) -> None:
        assert alert_side_to_okx_side(AlertSide.SHORT) is Side.SELL

    def test_exit_side_long_to_sell(self) -> None:
        # Closing a LONG = SELL the contracts
        assert exit_side_to_okx_side(AlertSide.LONG) is Side.SELL

    def test_exit_side_short_to_buy(self) -> None:
        assert exit_side_to_okx_side(AlertSide.SHORT) is Side.BUY


# ---------------------------------------------------------------------------
# Entry handling
# ---------------------------------------------------------------------------


class TestEntryHandling:
    def test_entry_places_market_order_with_sl(self) -> None:
        ex = _make_exchange_mock()
        h = SignalHandler(exchange=ex)
        result = h.handle(_make_alert())

        assert result.success is True
        assert result.order is not None
        assert ex.place_order.called

        req: OrderRequest = ex.place_order.call_args[0][0]
        assert req.symbol == "BTC-USDT-SWAP"
        assert req.side is Side.BUY
        assert req.order_type is OrderType.MARKET
        assert req.quantity == 0.05
        assert req.stop_loss == 48500.0
        assert req.reduce_only is False

    def test_pyramid_does_not_attach_sl(self) -> None:
        ex = _make_exchange_mock()
        h = SignalHandler(exchange=ex)
        h.handle(_make_alert(action="pyramid", unit=2, sl=None))

        req: OrderRequest = ex.place_order.call_args[0][0]
        assert req.stop_loss is None
        assert req.reduce_only is False  # pyramid is still opening

    def test_short_entry_uses_sell_side(self) -> None:
        ex = _make_exchange_mock()
        h = SignalHandler(exchange=ex)
        h.handle(_make_alert(side="short"))

        req: OrderRequest = ex.place_order.call_args[0][0]
        assert req.side is Side.SELL

    def test_entry_missing_size_rejected(self) -> None:
        # Construct an alert with no size — bypass model validation by
        # building manually
        alert = _make_alert(size=0.05)
        # Force size None via copy-with-update
        alert = TradingViewAlert.model_construct(**{**alert.model_dump(), "size": None})
        ex = _make_exchange_mock()
        h = SignalHandler(exchange=ex)
        result = h.handle(alert)
        assert result.success is False
        assert result.rejection_reason == "missing_size"
        assert not ex.place_order.called


# ---------------------------------------------------------------------------
# Exit handling
# ---------------------------------------------------------------------------


class TestExitHandling:
    def _exit_alert(self, **kw) -> TradingViewAlert:
        return _make_alert(action="exit", reason="donchian_exit", **kw)

    def _make_position(self, side: Side, qty: float) -> Position:
        return Position(
            symbol="BTC-USDT-SWAP", side=side, quantity=qty,
            entry_price=50000.0, leverage=5.0, unrealized_pnl=0.0,
            timestamp=datetime.now(tz=UTC),
        )

    def test_exit_long_places_reduce_only_sell(self) -> None:
        pos = self._make_position(Side.BUY, 0.10)
        ex = _make_exchange_mock(position=pos)
        h = SignalHandler(exchange=ex)
        result = h.handle(self._exit_alert(side="long"))

        assert result.success is True
        req: OrderRequest = ex.place_order.call_args[0][0]
        assert req.side is Side.SELL
        assert req.reduce_only is True
        assert req.quantity == 0.10

    def test_exit_short_places_reduce_only_buy(self) -> None:
        pos = self._make_position(Side.SELL, 0.07)
        ex = _make_exchange_mock(position=pos)
        h = SignalHandler(exchange=ex)
        result = h.handle(self._exit_alert(side="short"))

        req: OrderRequest = ex.place_order.call_args[0][0]
        assert req.side is Side.BUY
        assert req.reduce_only is True
        assert req.quantity == 0.07

    def test_exit_with_no_open_position_rejected(self) -> None:
        ex = _make_exchange_mock(position=None)
        h = SignalHandler(exchange=ex)
        result = h.handle(self._exit_alert())
        assert result.success is False
        assert result.rejection_reason == "flat"
        assert not ex.place_order.called

    def test_exit_with_zero_position_rejected(self) -> None:
        pos = Position(
            symbol="BTC-USDT-SWAP", side=Side.BUY, quantity=0.0,
            entry_price=0, leverage=1, unrealized_pnl=0,
            timestamp=datetime.now(tz=UTC),
        )
        ex = _make_exchange_mock(position=pos)
        h = SignalHandler(exchange=ex)
        result = h.handle(self._exit_alert())
        assert result.success is False
        assert result.rejection_reason == "flat"


# ---------------------------------------------------------------------------
# Risk gating
# ---------------------------------------------------------------------------


class TestRiskGating:
    def test_risk_check_can_block(self) -> None:
        ex = _make_exchange_mock()
        h = SignalHandler(
            exchange=ex,
            risk_check_fn=lambda alert, ex: "daily_loss_limit",
        )
        result = h.handle(_make_alert())
        assert result.success is False
        assert result.rejection_reason == "daily_loss_limit"
        assert not ex.place_order.called

    def test_risk_check_allows(self) -> None:
        ex = _make_exchange_mock()
        h = SignalHandler(exchange=ex, risk_check_fn=lambda a, ex: None)
        result = h.handle(_make_alert())
        assert result.success is True
        assert ex.place_order.called

    def test_no_risk_check_allows_by_default(self) -> None:
        # Without risk_check_fn, all signals pass through
        ex = _make_exchange_mock()
        h = SignalHandler(exchange=ex)
        result = h.handle(_make_alert())
        assert result.success is True


# ---------------------------------------------------------------------------
# Exchange error handling
# ---------------------------------------------------------------------------


class TestExchangeErrors:
    def test_insufficient_funds_caught(self) -> None:
        ex = MagicMock()
        ex.place_order.side_effect = InsufficientFundsError("low balance")
        h = SignalHandler(exchange=ex)
        result = h.handle(_make_alert())
        assert result.success is False
        assert result.rejection_reason == "insufficient_funds"

    def test_order_rejected_caught(self) -> None:
        ex = MagicMock()
        ex.place_order.side_effect = OrderRejectedError("bad price")
        h = SignalHandler(exchange=ex)
        result = h.handle(_make_alert())
        assert result.success is False
        assert result.rejection_reason == "exchange_rejected"
