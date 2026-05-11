"""Tests for the MT5 client — fully mocked, runs on any OS.

The MT5 client takes ``mt5_module`` as a constructor arg, so we inject a
MagicMock with the same surface area as the real ``MetaTrader5`` package.
No Windows, no terminal, no network — just behaviour assertions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from apexfx.aggressive.exchanges.base import (
    AuthenticationError,
    ExchangeError,
    InsufficientFundsError,
    OrderRejectedError,
    OrderRequest,
    OrderType,
    Side,
)
from apexfx.aggressive.exchanges.mt5_client import (
    Mt5Client,
    Mt5Credentials,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fake mt5 module
# ---------------------------------------------------------------------------


def _make_mt5_mock(*, init_ok: bool = True) -> MagicMock:
    """Build a MagicMock standing in for the ``MetaTrader5`` module."""
    m = MagicMock()
    m.initialize.return_value = init_ok
    m.last_error.return_value = (0, "ok")
    m.account_info.return_value = SimpleNamespace(
        login=12345, server="DemoBroker", currency="USD",
        balance=1000.0, equity=1000.0, margin_free=950.0,
        leverage=100, trade_mode=0,
    )
    m.symbol_info.return_value = SimpleNamespace(
        currency_base="EUR", currency_profit="USD",
        trade_contract_size=100000.0, point=0.00001,
        volume_step=0.01, volume_min=0.01, volume_max=100.0,
        visible=True, filling_mode=1,  # FOK supported
    )
    m.symbol_select.return_value = True
    m.positions_get.return_value = ()
    m.orders_get.return_value = ()
    m.history_orders_get.return_value = ()

    # MT5 constants used internally
    m.TRADE_ACTION_DEAL = 1
    m.TRADE_ACTION_PENDING = 5
    m.TRADE_ACTION_REMOVE = 8
    m.ORDER_TYPE_BUY = 0
    m.ORDER_TYPE_SELL = 1
    m.ORDER_TYPE_BUY_LIMIT = 2
    m.ORDER_TYPE_SELL_LIMIT = 3
    return m


def _mt5_tick(bid: float, ask: float) -> SimpleNamespace:
    return SimpleNamespace(
        bid=bid, ask=ask, last=ask,
        time=int(datetime(2026, 5, 1, tzinfo=UTC).timestamp()),
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_attach_to_running_terminal(self) -> None:
        mt5 = _make_mt5_mock()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        assert mt5.initialize.called
        # No login args when no credentials
        assert mt5.initialize.call_args.kwargs == {}
        client.shutdown()

    def test_initialize_with_creds(self) -> None:
        mt5 = _make_mt5_mock()
        creds = Mt5Credentials(login=12345, password="p", server="Broker")
        Mt5Client(credentials=creds, mt5_module=mt5)
        kwargs = mt5.initialize.call_args.kwargs
        assert kwargs["login"] == 12345
        assert kwargs["password"] == "p"
        assert kwargs["server"] == "Broker"

    def test_init_failure_with_auth_error(self) -> None:
        mt5 = _make_mt5_mock(init_ok=False)
        mt5.last_error.return_value = (5, "authorization failed")
        with pytest.raises(AuthenticationError):
            Mt5Client(credentials=None, mt5_module=mt5)

    def test_init_failure_generic(self) -> None:
        mt5 = _make_mt5_mock(init_ok=False)
        mt5.last_error.return_value = (1, "terminal not found")
        with pytest.raises(ExchangeError):
            Mt5Client(credentials=None, mt5_module=mt5)

    def test_idempotent_initialize(self) -> None:
        mt5 = _make_mt5_mock()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        # Second call should not re-invoke mt5.initialize
        client._initialize()
        assert mt5.initialize.call_count == 1

    def test_context_manager(self) -> None:
        mt5 = _make_mt5_mock()
        with Mt5Client(credentials=None, mt5_module=mt5) as client:
            assert client._initialized
        assert mt5.shutdown.called


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class TestMarketData:
    def _client(self, mt5: MagicMock) -> Mt5Client:
        return Mt5Client(credentials=None, mt5_module=mt5)

    def test_get_bars_invalid_interval(self) -> None:
        mt5 = _make_mt5_mock()
        client = self._client(mt5)
        with pytest.raises(ValueError, match="interval"):
            client.get_bars("EURUSD", "BogusTF")

    def test_get_bars_limit_out_of_range(self) -> None:
        mt5 = _make_mt5_mock()
        client = self._client(mt5)
        with pytest.raises(ValueError, match="limit"):
            client.get_bars("EURUSD", "H4", limit=0)
        with pytest.raises(ValueError, match="limit"):
            client.get_bars("EURUSD", "H4", limit=10000)

    def test_get_bars_parses_structured_array(self) -> None:
        mt5 = _make_mt5_mock()
        ts_base = int(datetime(2026, 5, 1, tzinfo=UTC).timestamp())
        # Structured numpy-like — list of records will do for the test
        rows = np.array(
            [
                (ts_base, 1.10, 1.105, 1.099, 1.103, 1000, 1, 1000),
                (ts_base + 3600, 1.103, 1.108, 1.102, 1.107, 1200, 1, 1200),
            ],
            dtype=[
                ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                ("close", "f8"), ("tick_volume", "i8"),
                ("spread", "i8"), ("real_volume", "i8"),
            ],
        )
        mt5.copy_rates_from_pos.return_value = rows
        client = self._client(mt5)
        bars = client.get_bars("EURUSD", "H4", limit=2)
        assert len(bars) == 2
        assert bars[0].open == 1.10
        assert bars[1].close == 1.107

    def test_get_bars_empty_raises(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.copy_rates_from_pos.return_value = None
        mt5.last_error.return_value = (1, "no data")
        client = self._client(mt5)
        with pytest.raises(ExchangeError):
            client.get_bars("EURUSD", "H4")

    def test_get_ticker(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.symbol_info_tick.return_value = _mt5_tick(bid=1.10, ask=1.1002)
        client = self._client(mt5)
        t = client.get_ticker("EURUSD")
        assert t.bid == 1.10
        assert t.ask == 1.1002

    def test_get_symbol_info_cached(self) -> None:
        mt5 = _make_mt5_mock()
        client = self._client(mt5)
        info1 = client.get_symbol_info("EURUSD")
        info2 = client.get_symbol_info("EURUSD")
        # Second call should not re-fetch (cached)
        assert mt5.symbol_info.call_count == 2  # 1 for _ensure_visible + 1 for fetch
        assert info1 is info2

    def test_unknown_symbol_raises(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.symbol_info.return_value = None
        client = Mt5Client(credentials=None, mt5_module=_make_mt5_mock())
        # Replace mt5 after construction to trigger missing-symbol path
        client._mt5 = mt5
        with pytest.raises(ExchangeError):
            client.get_ticker("UNKNOWNXX")


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


class TestAccount:
    def test_get_balance(self) -> None:
        mt5 = _make_mt5_mock()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        b = client.get_balance()
        assert b.asset == "USD"
        assert b.equity == 1000.0
        assert b.available == 950.0

    def test_get_balance_no_info(self) -> None:
        mt5 = _make_mt5_mock()
        # account_info returns valid during init, then None on later read
        client = Mt5Client(credentials=None, mt5_module=mt5)
        mt5.account_info.return_value = None
        with pytest.raises(ExchangeError):
            client.get_balance()

    def test_get_positions_filters_by_magic(self) -> None:
        mt5 = _make_mt5_mock()
        ours = SimpleNamespace(
            symbol="EURUSD", type=0, volume=0.05, price_open=1.10,
            profit=5.0, magic=770125,
            time=int(datetime(2026, 5, 1, tzinfo=UTC).timestamp()),
        )
        other_bot = SimpleNamespace(
            symbol="GBPUSD", type=0, volume=0.10, price_open=1.30,
            profit=10.0, magic=999999,
            time=int(datetime(2026, 5, 1, tzinfo=UTC).timestamp()),
        )
        mt5.positions_get.return_value = (ours, other_bot)
        client = Mt5Client(credentials=None, mt5_module=mt5, magic=770125)
        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "EURUSD"

    def test_get_position_with_no_match_returns_none(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.positions_get.return_value = ()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        assert client.get_position("EURUSD") is None

    def test_get_position_hedging_aggregates(self) -> None:
        # Hedging mode: both long and short positions exist for same symbol
        mt5 = _make_mt5_mock()
        long_pos = SimpleNamespace(
            symbol="EURUSD", type=0, volume=0.10, price_open=1.10,
            profit=5.0, magic=770125,
            time=int(datetime(2026, 5, 1, tzinfo=UTC).timestamp()),
        )
        short_pos = SimpleNamespace(
            symbol="EURUSD", type=1, volume=0.04, price_open=1.105,
            profit=-2.0, magic=770125,
            time=int(datetime(2026, 5, 1, tzinfo=UTC).timestamp()),
        )
        mt5.positions_get.return_value = (long_pos, short_pos)
        client = Mt5Client(credentials=None, mt5_module=mt5, magic=770125)
        pos = client.get_position("EURUSD")
        # Net long 0.06
        assert pos is not None
        assert pos.side is Side.BUY
        assert pos.quantity == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


class TestOrders:
    def _success_result(self, **overrides) -> SimpleNamespace:
        base = dict(retcode=10009, order=12345, deal=67890,
                    volume=0.05, price=1.10, comment="")
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_market_order_success(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.symbol_info_tick.return_value = _mt5_tick(bid=1.10, ask=1.1002)
        mt5.order_send.return_value = self._success_result()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        order = client.place_order(OrderRequest(
            symbol="EURUSD", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=0.05,
        ))
        assert order.order_id == "12345"
        assert mt5.order_send.called
        # Inspect the request payload
        sent = mt5.order_send.call_args[0][0]
        assert sent["symbol"] == "EURUSD"
        assert sent["volume"] == 0.05
        assert sent["type"] == 0  # BUY
        assert sent["price"] == 1.1002  # ask for buy

    def test_market_order_attaches_sl_tp(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.symbol_info_tick.return_value = _mt5_tick(bid=1.10, ask=1.1002)
        mt5.order_send.return_value = self._success_result()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        client.place_order(OrderRequest(
            symbol="EURUSD", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=0.05,
            stop_loss=1.0950, take_profit=1.1200,
        ))
        sent = mt5.order_send.call_args[0][0]
        assert sent["sl"] == 1.0950
        assert sent["tp"] == 1.1200

    def test_insufficient_funds_mapped(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.symbol_info_tick.return_value = _mt5_tick(bid=1.10, ask=1.1002)
        mt5.order_send.return_value = self._success_result(
            retcode=10019, comment="no money",
        )
        client = Mt5Client(credentials=None, mt5_module=mt5)
        with pytest.raises(InsufficientFundsError):
            client.place_order(OrderRequest(
                symbol="EURUSD", side=Side.BUY,
                order_type=OrderType.MARKET, quantity=0.05,
            ))

    def test_rejection_mapped(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.symbol_info_tick.return_value = _mt5_tick(bid=1.10, ask=1.1002)
        mt5.order_send.return_value = self._success_result(
            retcode=10021, comment="price off-quote",
        )
        client = Mt5Client(credentials=None, mt5_module=mt5)
        with pytest.raises(OrderRejectedError):
            client.place_order(OrderRequest(
                symbol="EURUSD", side=Side.BUY,
                order_type=OrderType.MARKET, quantity=0.05,
            ))

    def test_volume_snapped_to_lot_step(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.symbol_info_tick.return_value = _mt5_tick(bid=1.10, ask=1.1002)
        mt5.order_send.return_value = self._success_result()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        # 0.057 should round DOWN to 0.05 (step 0.01)
        client.place_order(OrderRequest(
            symbol="EURUSD", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=0.057,
        ))
        sent = mt5.order_send.call_args[0][0]
        assert sent["volume"] == pytest.approx(0.05, abs=1e-9)

    def test_volume_below_min_rejected(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.symbol_info_tick.return_value = _mt5_tick(bid=1.10, ask=1.1002)
        client = Mt5Client(credentials=None, mt5_module=mt5)
        with pytest.raises(OrderRejectedError, match="below symbol minimum"):
            client.place_order(OrderRequest(
                symbol="EURUSD", side=Side.BUY,
                order_type=OrderType.MARKET, quantity=0.001,
            ))

    def test_cancel_invalid_ticket_raises_value_error(self) -> None:
        mt5 = _make_mt5_mock()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        with pytest.raises(ValueError):
            client.cancel_order("EURUSD", "not-a-number")

    def test_cancel_order_not_found_idempotent(self) -> None:
        mt5 = _make_mt5_mock()
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (1, "order not found")
        client = Mt5Client(credentials=None, mt5_module=mt5)
        # Should NOT raise
        client.cancel_order("EURUSD", "12345")


# ---------------------------------------------------------------------------
# Leverage (no-op)
# ---------------------------------------------------------------------------


class TestLeverage:
    def test_set_leverage_invalid(self) -> None:
        mt5 = _make_mt5_mock()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        with pytest.raises(ValueError):
            client.set_leverage("EURUSD", 0)

    def test_set_leverage_warns_only(self) -> None:
        # No exception; logs warning
        mt5 = _make_mt5_mock()
        client = Mt5Client(credentials=None, mt5_module=mt5)
        client.set_leverage("EURUSD", 100.0)  # No-op


# ---------------------------------------------------------------------------
# No MT5 module installed
# ---------------------------------------------------------------------------


class TestUnavailableMT5:
    def test_raises_import_error_when_no_module(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force the real-import path to None
        monkeypatch.setattr(
            "apexfx.aggressive.exchanges.mt5_client._real_mt5", None,
        )
        with pytest.raises(ImportError):
            Mt5Client(credentials=None)
