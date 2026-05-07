"""OKX client tests — mocked HTTP, no live calls.

The strategy of these tests is to mock the ``requests.Session`` underneath
the client.  This keeps tests:

* fast (no network)
* deterministic (no flakes from rate limits)
* runnable offline / in CI

What we lock down:

* Authentication required (rejects empty creds)
* Demo header set when ``demo=True``, absent when False
* Signature is HMAC-SHA256 base64 (round-trip math)
* Error code mapping: 401/403 → AuthenticationError; 51008 → InsufficientFundsError
* Retry on 429 / 5xx with backoff (asserts attempt count)
* OKX wrapping: ``code != "0"`` raises with mapped exception
* Order placement happy path and rejection path
* SymbolInfo cached after first fetch
* Bars come back oldest-first (we reverse OKX's newest-first response)
"""

from __future__ import annotations

import base64
import hmac
import json
from datetime import datetime, timezone
from hashlib import sha256
from unittest.mock import MagicMock, patch

import pytest
import requests

from apexfx.aggressive.exchanges.base import (
    AuthenticationError,
    ExchangeError,
    InsufficientFundsError,
    OrderRejectedError,
    OrderRequest,
    OrderType,
    Side,
)
from apexfx.aggressive.exchanges.okx_client import OkxClient

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ok_response(data: list | dict) -> dict:
    """Wrap data in OKX's standard ``{code, msg, data}`` envelope."""
    return {"code": "0", "msg": "", "data": data if isinstance(data, list) else [data]}


def _err_response(code: str, msg: str = "boom") -> dict:
    return {"code": code, "msg": msg, "data": []}


def _mock_http_response(status: int, json_body: dict, text: str = "") -> MagicMock:
    """Build a ``requests.Response``-shaped mock."""
    m = MagicMock(spec=requests.Response)
    m.status_code = status
    m.json.return_value = json_body
    m.text = text or json.dumps(json_body)
    return m


@pytest.fixture
def client() -> OkxClient:
    """Demo-mode client with placeholder creds."""
    return OkxClient(
        api_key="key", api_secret="secret", api_passphrase="pass",
        demo=True,
    )


# ---------------------------------------------------------------------------
# Construction / auth
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_key_rejected(self) -> None:
        with pytest.raises(AuthenticationError, match="api_key"):
            OkxClient(api_key="", api_secret="s", api_passphrase="p")

    def test_empty_secret_rejected(self) -> None:
        with pytest.raises(AuthenticationError):
            OkxClient(api_key="k", api_secret="", api_passphrase="p")

    def test_empty_passphrase_rejected(self) -> None:
        with pytest.raises(AuthenticationError):
            OkxClient(api_key="k", api_secret="s", api_passphrase="")


class TestSigning:
    def test_signature_matches_okx_spec(self, client: OkxClient) -> None:
        # OKX docs: sign = base64(HMAC-SHA256(secret, ts + method + path + body))
        ts = "2026-04-26T00:00:00.000Z"
        method = "GET"
        path = "/api/v5/account/balance"
        body = ""
        expected = base64.b64encode(
            hmac.new(b"secret", f"{ts}{method}{path}{body}".encode("utf-8"), sha256).digest()
        ).decode("utf-8")
        assert client._sign(ts, method, path, body) == expected

    def test_demo_header_set_when_demo_true(self, client: OkxClient) -> None:
        h = client._headers("GET", "/api/v5/account/balance")
        assert h.get("x-simulated-trading") == "1"

    def test_demo_header_absent_when_demo_false(self) -> None:
        c = OkxClient(api_key="k", api_secret="s", api_passphrase="p", demo=False)
        h = c._headers("GET", "/path")
        assert "x-simulated-trading" not in h

    def test_required_auth_headers_present(self, client: OkxClient) -> None:
        h = client._headers("GET", "/path")
        for k in (
            "OK-ACCESS-KEY",
            "OK-ACCESS-SIGN",
            "OK-ACCESS-TIMESTAMP",
            "OK-ACCESS-PASSPHRASE",
        ):
            assert k in h


# ---------------------------------------------------------------------------
# Error handling / retry
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_401_raises_auth_error_no_retry(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(401, {}, "Unauthorized"),
        )
        with pytest.raises(AuthenticationError):
            client._request("GET", "/path")
        # No retry on 401
        assert client._session.request.call_count == 1

    def test_403_raises_auth_error(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(403, {}, "Forbidden"),
        )
        with pytest.raises(AuthenticationError):
            client._request("GET", "/path")

    def test_429_retries_then_fails(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(429, {}, "Too Many"),
        )
        with patch("apexfx.aggressive.exchanges.okx_client.time.sleep"):
            with pytest.raises(ExchangeError):
                client._request("GET", "/path", max_retries=3)
        assert client._session.request.call_count == 3

    def test_5xx_retries(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(503, {}, "Service Unavailable"),
        )
        with patch("apexfx.aggressive.exchanges.okx_client.time.sleep"):
            with pytest.raises(ExchangeError):
                client._request("GET", "/path", max_retries=2)
        assert client._session.request.call_count == 2

    def test_400_no_retry(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(400, {}, "Bad Request"),
        )
        with pytest.raises(ExchangeError):
            client._request("GET", "/path", max_retries=3)
        assert client._session.request.call_count == 1

    def test_okx_error_code_51008_maps_insufficient_funds(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _err_response("51008", "balance low")),
        )
        with pytest.raises(InsufficientFundsError):
            client._request("GET", "/path")

    def test_okx_error_code_51000_maps_order_rejected(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _err_response("51000", "bad lot")),
        )
        with pytest.raises(OrderRejectedError):
            client._request("GET", "/path")

    def test_okx_retryable_code_50011_retries(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _err_response("50011", "busy")),
        )
        with patch("apexfx.aggressive.exchanges.okx_client.time.sleep"):
            with pytest.raises(ExchangeError):
                client._request("GET", "/path", max_retries=2)
        assert client._session.request.call_count == 2


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


class TestMarketData:
    def test_get_bars_invalid_interval_raises(self, client: OkxClient) -> None:
        with pytest.raises(ValueError, match="interval must be one of"):
            client.get_bars("BTC-USDT-SWAP", "7H")

    def test_get_bars_limit_out_of_range_raises(self, client: OkxClient) -> None:
        with pytest.raises(ValueError, match="limit"):
            client.get_bars("BTC-USDT-SWAP", "4H", limit=0)
        with pytest.raises(ValueError, match="limit"):
            client.get_bars("BTC-USDT-SWAP", "4H", limit=400)

    def test_get_bars_returns_oldest_first(self, client: OkxClient) -> None:
        # OKX returns newest-first; client must reverse so the strategy
        # iterates oldest→newest naturally.
        rows = [
            ["1700000200000", "100", "110", "90", "105", "1000", "0", "0", "1"],
            ["1700000100000", "99", "108", "89", "100", "950", "0", "0", "1"],
            ["1700000000000", "98", "105", "88", "99", "900", "0", "0", "1"],
        ]
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _ok_response(rows)),
        )
        bars = client.get_bars("BTC-USDT-SWAP", "4H", limit=3)
        assert len(bars) == 3
        # First bar should be oldest (ts 1700000000000)
        assert bars[0].timestamp.timestamp() == 1700000000.0
        assert bars[-1].timestamp.timestamp() == 1700000200.0

    def test_get_ticker_parses_fields(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(
                200,
                _ok_response({"last": "50100", "bidPx": "50000", "askPx": "50100",
                              "ts": "1700000000000"}),
            ),
        )
        t = client.get_ticker("BTC-USDT-SWAP")
        assert t.last_price == 50100.0
        assert t.bid == 50000.0
        assert t.ask == 50100.0
        assert t.symbol == "BTC-USDT-SWAP"

    def test_get_symbol_info_cached(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(
                200,
                _ok_response({
                    "instId": "BTC-USDT-SWAP",
                    "ctValCcy": "BTC",
                    "settleCcy": "USDT",
                    "ctVal": "0.01",
                    "tickSz": "0.1",
                    "lotSz": "1",
                    "minSz": "1",
                    "lever": "100",
                }),
            ),
        )
        info1 = client.get_symbol_info("BTC-USDT-SWAP")
        info2 = client.get_symbol_info("BTC-USDT-SWAP")
        assert info1 is info2  # identical cached object
        assert client._session.request.call_count == 1
        assert info1.contract_size == 0.01
        assert info1.tick_size == 0.1
        assert info1.max_leverage == 100.0


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------


class TestAccount:
    def test_get_balance_returns_zero_for_unknown_asset(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _ok_response({"details": []})),
        )
        b = client.get_balance("USDT")
        assert b.equity == 0.0
        assert b.available == 0.0

    def test_get_balance_picks_correct_asset(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _ok_response({
                "details": [
                    {"ccy": "BTC", "eq": "0.5", "availEq": "0.5"},
                    {"ccy": "USDT", "eq": "1000.50", "availEq": "950.00"},
                ],
            })),
        )
        b = client.get_balance("USDT")
        assert b.equity == 1000.50
        assert b.available == 950.00

    def test_get_positions_skips_zero_qty(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _ok_response([
                {"instId": "BTC-USDT-SWAP", "pos": "0", "posSide": "long",
                 "avgPx": "0", "lever": "5", "upl": "0"},
                {"instId": "ETH-USDT-SWAP", "pos": "5", "posSide": "long",
                 "avgPx": "3000", "lever": "5", "upl": "10"},
            ])),
        )
        positions = client.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "ETH-USDT-SWAP"
        assert positions[0].side is Side.BUY

    def test_get_positions_short_decoded(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _ok_response([
                {"instId": "BTC-USDT-SWAP", "pos": "-3", "posSide": "net",
                 "avgPx": "50000", "lever": "5", "upl": "-50"},
            ])),
        )
        positions = client.get_positions()
        assert positions[0].side is Side.SELL
        assert positions[0].quantity == 3.0
        assert positions[0].signed_quantity == -3.0

    def test_get_position_returns_none_when_flat(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _ok_response([])),
        )
        assert client.get_position("BTC-USDT-SWAP") is None


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


class TestOrders:
    def test_place_market_order_happy_path(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _ok_response({
                "ordId": "ord-123",
                "clOrdId": "apx-xxx",
                "sCode": "0",
                "sMsg": "",
            })),
        )
        req = OrderRequest(
            symbol="BTC-USDT-SWAP", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=0.01,
        )
        order = client.place_order(req)
        assert order.order_id == "ord-123"
        assert order.side is Side.BUY
        assert order.order_type is OrderType.MARKET
        assert order.symbol == "BTC-USDT-SWAP"

    def test_place_order_rejection_raises(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _ok_response({
                "ordId": "",
                "clOrdId": "apx-xxx",
                "sCode": "51000",
                "sMsg": "Bad lot size",
            })),
        )
        req = OrderRequest(
            symbol="BTC-USDT-SWAP", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=0.01,
        )
        with pytest.raises(OrderRejectedError, match="Bad lot size"):
            client.place_order(req)

    def test_place_order_attaches_sl_tp(self, client: OkxClient) -> None:
        captured_body: dict | None = None

        def fake_request(*args, **kwargs):
            nonlocal captured_body
            captured_body = json.loads(kwargs["data"]) if kwargs.get("data") else None
            return _mock_http_response(200, _ok_response({
                "ordId": "ord-1", "clOrdId": "apx-y", "sCode": "0", "sMsg": "",
            }))

        client._session.request = MagicMock(side_effect=fake_request)  # type: ignore[method-assign]
        req = OrderRequest(
            symbol="BTC-USDT-SWAP", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=0.01,
            stop_loss=49000.0, take_profit=52000.0,
        )
        client.place_order(req)
        assert captured_body is not None
        assert "attachAlgoOrds" in captured_body
        algo = captured_body["attachAlgoOrds"][0]
        assert "slTriggerPx" in algo
        assert "tpTriggerPx" in algo

    def test_cancel_order_idempotent_on_already_canceled(self, client: OkxClient) -> None:
        # 51400 = order does not exist (already canceled or filled)
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _err_response("51400", "Order does not exist")),
        )
        # Should NOT raise — idempotent
        client.cancel_order("BTC-USDT-SWAP", "ord-123")

    def test_cancel_order_propagates_other_errors(self, client: OkxClient) -> None:
        client._session.request = MagicMock(  # type: ignore[method-assign]
            return_value=_mock_http_response(200, _err_response("51999", "weird error")),
        )
        with pytest.raises(OrderRejectedError):
            client.cancel_order("BTC-USDT-SWAP", "ord-123")


# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------


class TestLeverage:
    def test_set_leverage_validates_input(self, client: OkxClient) -> None:
        with pytest.raises(ValueError, match="leverage must be positive"):
            client.set_leverage("BTC-USDT-SWAP", 0)
        with pytest.raises(ValueError, match="leverage must be positive"):
            client.set_leverage("BTC-USDT-SWAP", -5)

    def test_set_leverage_calls_correct_endpoint(self, client: OkxClient) -> None:
        captured_body: dict | None = None

        def fake_request(*args, **kwargs):
            nonlocal captured_body
            captured_body = json.loads(kwargs["data"]) if kwargs.get("data") else None
            return _mock_http_response(200, _ok_response({}))

        client._session.request = MagicMock(side_effect=fake_request)  # type: ignore[method-assign]
        client.set_leverage("BTC-USDT-SWAP", 5)
        assert captured_body == {"instId": "BTC-USDT-SWAP", "lever": "5", "mgnMode": "cross"}
