"""OKX v5 REST client for USDT-M perpetual futures.

Why we use raw ``requests`` instead of an OKX SDK
-------------------------------------------------
The official ``python-okx`` SDK pulls heavy deps (pandas, asyncio loops),
has multi-version inconsistencies, and changes its public API between
minor releases.  For our needs — ~10 endpoints — a thin client is:

- 300 LOC vs 5,000 LOC of vendored code
- Independent of OKX SDK versioning
- Easy to mock in tests (we just stub ``_request``)
- Works identically against ``demo`` and ``live`` by switching base URL

Authentication — three secrets
------------------------------
OKX requires API key, secret, **and** a passphrase chosen at key creation.
The signature is HMAC-SHA256 over ``timestamp + method + path + body`` using
the secret, then base64-encoded.  Passphrase is sent unhashed in a header.

Demo trading
------------
OKX demo (``x-simulated-trading: 1`` header on the live API host) gives you
a paper account with the same exact endpoints, latency and limits as
production.  We expose this as ``OkxClient(demo=True)`` so the strategy can
run end-to-end without real funds.

Rate limits
-----------
OKX limits are per-endpoint (e.g. 20 req/2s for ``/orders``).  We don't
implement client-side throttling — instead we treat 429 as a recoverable
error with exponential backoff (3 attempts), which is what every retail
bot does in practice.  If you hit the limit consistently, your strategy
is overtrading.
"""

from __future__ import annotations

import base64
import hmac
import json
import time
import uuid
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

import requests

from apexfx.aggressive.exchanges.base import (
    AuthenticationError,
    Balance,
    Bar,
    Exchange,
    ExchangeError,
    InsufficientFundsError,
    Order,
    OrderRejectedError,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    RateLimitError,
    Side,
    SymbolInfo,
    Ticker,
    TimeInForce,
)
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OKX uses a single host for both live and demo; demo is unlocked via header.
OKX_BASE_URL = "https://www.okx.com"

# OKX returns timestamps as millisecond-precision strings.
_MS_PER_S = 1000.0

# Default per-call timeout for HTTP — long enough for slow responses, short
# enough to avoid hanging the strategy loop on a single bad endpoint.
_DEFAULT_TIMEOUT_S = 10.0

# Recoverable HTTP status codes — we retry these with backoff.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# OKX-specific error codes that aren't network errors but should still retry
# (transient-but-not-fatal — e.g. matching engine momentarily busy).
_RETRYABLE_OKX_CODES = {"50011", "50026", "63999"}


# ---------------------------------------------------------------------------
# Bar interval mapping
# ---------------------------------------------------------------------------

# OKX accepts: 1m/3m/5m/15m/30m/1H/2H/4H/6H/12H/1D/1W/1M
# We expose a small whitelist with normalized casing — anything outside
# raises so users get an immediate error instead of a confusing API rejection.
_VALID_INTERVALS = {
    "1m", "5m", "15m", "30m",
    "1H", "4H", "1D", "1W",
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OkxClient(Exchange):
    """OKX v5 REST client implementing the ``Exchange`` interface.

    Parameters
    ----------
    api_key : str
        Generated in OKX → Account → API.  Limit to ``Trade`` + ``Read``
        permissions; **do not** grant Withdraw to a trading bot.
    api_secret : str
        Shown once at key creation — store in env var, never commit.
    api_passphrase : str
        The 8-12 char passphrase you chose at key creation (NOT your
        login password).
    demo : bool, default ``True``
        If True, enables OKX simulated-trading mode.  Same endpoints,
        fake balance, real prices.  **Always start in demo.**
    timeout : float, default 10s
        HTTP timeout per call.

    Notes
    -----
    The client is sync.  For high-frequency reads consider the WebSocket
    streamer (in a future module).  At 4H bars we make ~6 API calls per
    bar per symbol — well within free-tier limits.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        demo: bool = True,
        timeout: float = _DEFAULT_TIMEOUT_S,
        base_url: str = OKX_BASE_URL,
    ) -> None:
        if not api_key or not api_secret or not api_passphrase:
            raise AuthenticationError(
                "OKX requires api_key, api_secret AND api_passphrase — "
                "the third is the passphrase you chose at key creation."
            )
        self._key = api_key
        self._secret = api_secret.encode("utf-8")
        self._passphrase = api_passphrase
        self._demo = demo
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        # Cache of SymbolInfo — these are static, fetch once
        self._symbol_cache: dict[str, SymbolInfo] = {}

    # ------------------------------------------------------------------
    # Internal: signing + HTTP
    # ------------------------------------------------------------------

    def _sign(self, timestamp: str, method: str, path: str, body: str) -> str:
        """OKX signature: HMAC_SHA256(secret, timestamp + method + path + body) → base64.

        ``timestamp`` must be ISO-8601 in UTC with millis precision (e.g.
        ``2026-04-26T12:34:56.789Z``).  ``method`` is uppercase HTTP verb.
        ``path`` is the URL path including query string but without host.
        ``body`` is the JSON-serialized body for POST or empty string for GET.
        """
        message = f"{timestamp}{method.upper()}{path}{body}"
        digest = hmac.new(self._secret, message.encode("utf-8"), sha256).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _headers(self, method: str, path: str, body: str = "") -> dict[str, str]:
        """Auth + content headers for one request.

        Demo flag is enabled via ``x-simulated-trading: 1`` — same auth,
        same path, OKX routes to the demo matching engine.
        """
        ts = (
            datetime.now(tz=UTC)
            .strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{datetime.now(tz=UTC).microsecond // 1000:03d}Z"
        )
        headers = {
            "OK-ACCESS-KEY": self._key,
            "OK-ACCESS-SIGN": self._sign(ts, method, path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._passphrase,
            "Content-Type": "application/json",
        }
        if self._demo:
            headers["x-simulated-trading"] = "1"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Single REST call with retry on transient failures.

        We treat 429 / 5xx and a small whitelist of OKX error codes as
        retryable.  Everything else (auth error, validation error,
        insufficient funds) raises immediately — those are bugs, not
        transient issues, and retrying makes them worse.
        """
        if params:
            path_with_query = f"{path}?{_urlencode(params)}"
        else:
            path_with_query = path

        body_str = json.dumps(body) if body else ""
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                headers = self._headers(method, path_with_query, body_str)
                resp = self._session.request(
                    method=method,
                    url=f"{self._base_url}{path_with_query}",
                    headers=headers,
                    data=body_str if body_str else None,
                    timeout=self._timeout,
                )

                # 401 / 403 → never retry, credentials wrong
                if resp.status_code in (401, 403):
                    raise AuthenticationError(
                        f"OKX rejected credentials: {resp.status_code} {resp.text[:200]}"
                    )
                # Retryable HTTP error
                if resp.status_code in _RETRYABLE_STATUS:
                    last_error = RateLimitError(
                        f"HTTP {resp.status_code} (attempt {attempt + 1}/{max_retries})"
                    )
                    _backoff_sleep(attempt)
                    continue
                # Other 4xx → permanent
                if resp.status_code >= 400:
                    raise ExchangeError(
                        f"OKX HTTP {resp.status_code}: {resp.text[:300]}"
                    )

                payload = resp.json()
                # OKX wraps everything in {code, msg, data}
                code = str(payload.get("code", ""))
                if code != "0":
                    if code in _RETRYABLE_OKX_CODES:
                        last_error = RateLimitError(
                            f"OKX code {code}: {payload.get('msg')}"
                        )
                        _backoff_sleep(attempt)
                        continue
                    # Map common codes to specific exceptions
                    msg = payload.get("msg") or "(no message)"
                    if code in ("51008", "51131"):  # insufficient balance
                        raise InsufficientFundsError(f"OKX {code}: {msg}")
                    if code.startswith("51") or code.startswith("59"):
                        raise OrderRejectedError(f"OKX {code}: {msg}")
                    raise ExchangeError(f"OKX {code}: {msg}")

                return payload  # type: ignore[no-any-return]

            except requests.RequestException as exc:
                last_error = exc
                _backoff_sleep(attempt)

        # All retries exhausted
        raise ExchangeError(
            f"OKX request failed after {max_retries} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        end_time: datetime | None = None,
    ) -> list[Bar]:
        if interval not in _VALID_INTERVALS:
            raise ValueError(
                f"interval must be one of {sorted(_VALID_INTERVALS)}, got {interval!r}"
            )
        if not 1 <= limit <= 300:
            raise ValueError(f"limit must be in [1, 300], got {limit}")

        params: dict[str, Any] = {
            "instId": symbol,
            "bar": interval,
            "limit": str(limit),
        }
        if end_time is not None:
            # OKX 'after' means "bars older than this timestamp", in ms
            params["after"] = str(int(end_time.timestamp() * _MS_PER_S))

        resp = self._request("GET", "/api/v5/market/candles", params=params)
        # OKX returns rows newest-first as
        # [ts, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
        bars: list[Bar] = []
        for row in resp["data"]:
            ts_ms = int(row[0])
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(ts_ms / _MS_PER_S, tz=UTC),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        # Return oldest-first for natural iteration in the strategy
        bars.reverse()
        return bars

    def get_ticker(self, symbol: str) -> Ticker:
        resp = self._request(
            "GET", "/api/v5/market/ticker", params={"instId": symbol}
        )
        rows = resp.get("data") or []
        if not rows:
            raise ExchangeError(f"No ticker data for {symbol}")
        row = rows[0]
        return Ticker(
            symbol=symbol,
            last_price=float(row["last"]),
            bid=float(row.get("bidPx") or 0),
            ask=float(row.get("askPx") or 0),
            timestamp=datetime.fromtimestamp(int(row["ts"]) / _MS_PER_S, tz=UTC),
        )

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]
        resp = self._request(
            "GET", "/api/v5/public/instruments",
            params={"instType": "SWAP", "instId": symbol},
        )
        rows = resp.get("data") or []
        if not rows:
            raise ExchangeError(f"No symbol info for {symbol}")
        row = rows[0]
        # OKX SWAP fields: ctVal = contract size in base ccy, lotSz = lot,
        # tickSz = price tick, minSz = minimum order in contracts, lever = max lev
        info = SymbolInfo(
            symbol=symbol,
            base_currency=str(row["ctValCcy"]),
            quote_currency=str(row["settleCcy"]),
            contract_size=float(row["ctVal"]),
            tick_size=float(row["tickSz"]),
            lot_size=float(row["lotSz"]),
            min_quantity=float(row["minSz"]),
            max_leverage=float(row.get("lever") or 1.0),
        )
        self._symbol_cache[symbol] = info
        return info

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_balance(self, asset: str = "USDT") -> Balance:
        resp = self._request("GET", "/api/v5/account/balance",
                             params={"ccy": asset})
        rows = resp.get("data") or []
        if not rows:
            raise ExchangeError(f"No balance data for {asset}")
        # OKX returns [{details: [{ccy, eq, availEq, ...}]}]
        details = rows[0].get("details") or []
        for d in details:
            if d.get("ccy") == asset:
                return Balance(
                    asset=asset,
                    equity=float(d.get("eq") or 0),
                    available=float(d.get("availEq") or d.get("cashBal") or 0),
                    timestamp=datetime.now(tz=UTC),
                )
        # Asset not in account → zero balance
        return Balance(
            asset=asset, equity=0.0, available=0.0,
            timestamp=datetime.now(tz=UTC),
        )

    def get_positions(self) -> list[Position]:
        resp = self._request("GET", "/api/v5/account/positions",
                             params={"instType": "SWAP"})
        positions: list[Position] = []
        for row in resp.get("data") or []:
            qty = float(row.get("pos") or 0)
            if qty == 0:
                continue
            # OKX uses positive qty for long, negative for short under net mode,
            # OR posSide field "long"/"short" under hedge mode
            side_str = row.get("posSide", "")
            if side_str in ("long", "short"):
                side = Side.BUY if side_str == "long" else Side.SELL
                quantity = abs(qty)
            else:
                side = Side.BUY if qty > 0 else Side.SELL
                quantity = abs(qty)
            positions.append(
                Position(
                    symbol=str(row["instId"]),
                    side=side,
                    quantity=quantity,
                    entry_price=float(row.get("avgPx") or 0),
                    leverage=float(row.get("lever") or 1),
                    unrealized_pnl=float(row.get("upl") or 0),
                    timestamp=datetime.now(tz=UTC),
                )
            )
        return positions

    def get_position(self, symbol: str) -> Position | None:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p
        return None

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, req: OrderRequest) -> Order:
        body: dict[str, Any] = {
            "instId": req.symbol,
            "tdMode": "cross",  # cross-margin (most retail-appropriate)
            "side": req.side.value,
            "ordType": _map_order_type(req.order_type, req.time_in_force),
            "sz": _format_qty(req.quantity),
        }
        if req.order_type is OrderType.LIMIT:
            body["px"] = _format_price(req.price)  # type: ignore[arg-type]
        if req.client_order_id:
            body["clOrdId"] = req.client_order_id
        else:
            body["clOrdId"] = _generate_client_order_id()
        if req.reduce_only:
            body["reduceOnly"] = "true"

        # Attach SL/TP as algo params on the same order (OKX's
        # ``attachAlgoOrds``) so we don't race a second API call.
        if req.stop_loss is not None or req.take_profit is not None:
            algo: dict[str, Any] = {}
            if req.stop_loss is not None:
                algo["slTriggerPx"] = _format_price(req.stop_loss)
                algo["slOrdPx"] = "-1"  # market on trigger
            if req.take_profit is not None:
                algo["tpTriggerPx"] = _format_price(req.take_profit)
                algo["tpOrdPx"] = "-1"
            body["attachAlgoOrds"] = [algo]

        resp = self._request("POST", "/api/v5/trade/order", body=body)
        rows = resp.get("data") or []
        if not rows:
            raise ExchangeError("OKX returned empty order response")
        row = rows[0]
        if str(row.get("sCode", "0")) != "0":
            msg = f"OKX order rejected ({row.get('sCode')}): {row.get('sMsg')}"
            if str(row.get("sCode")) in ("51008", "51131"):
                raise InsufficientFundsError(msg)
            raise OrderRejectedError(msg)

        return Order(
            order_id=str(row["ordId"]),
            client_order_id=body.get("clOrdId"),
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            status=OrderStatus.OPEN,
            quantity=req.quantity,
            filled_quantity=0.0,
            avg_fill_price=0.0,
            price=req.price,
            timestamp=datetime.now(tz=UTC),
        )

    def cancel_order(self, symbol: str, order_id: str) -> None:
        try:
            self._request(
                "POST", "/api/v5/trade/cancel-order",
                body={"instId": symbol, "ordId": order_id},
            )
        except OrderRejectedError as exc:
            # Idempotent: 51400/51401 mean already canceled or filled
            msg = str(exc)
            if "51400" in msg or "51401" in msg or "already" in msg.lower():
                logger.debug("cancel_order: order already terminal", order_id=order_id)
                return
            raise

    def get_order(self, symbol: str, order_id: str) -> Order:
        resp = self._request(
            "GET", "/api/v5/trade/order",
            params={"instId": symbol, "ordId": order_id},
        )
        rows = resp.get("data") or []
        if not rows:
            raise ExchangeError(f"Order {order_id} not found")
        return _parse_order(rows[0])

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        params: dict[str, Any] = {"instType": "SWAP"}
        if symbol:
            params["instId"] = symbol
        resp = self._request("GET", "/api/v5/trade/orders-pending", params=params)
        return [_parse_order(row) for row in (resp.get("data") or [])]

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    def set_leverage(self, symbol: str, leverage: float) -> None:
        if leverage <= 0:
            raise ValueError(f"leverage must be positive, got {leverage}")
        self._request(
            "POST", "/api/v5/account/set-leverage",
            body={
                "instId": symbol,
                "lever": str(leverage),
                "mgnMode": "cross",
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff: 0.5s, 1s, 2s.  Caps at attempt 3."""
    time.sleep(min(0.5 * (2 ** attempt), 4.0))


def _urlencode(params: dict[str, Any]) -> str:
    """Flat key=value&key=value; OKX doesn't need URL-encoding for our values."""
    return "&".join(f"{k}={v}" for k, v in params.items())


def _generate_client_order_id() -> str:
    """Short, alphanumeric, < 32 chars (OKX cap is 32, alphanumeric only)."""
    return "apx" + uuid.uuid4().hex[:24]


def _format_qty(qty: float) -> str:
    """OKX wants integer-typed strings for quantity in contracts.  Round
    to 8 decimals for safety; the exchange will then validate against
    ``lotSz`` and reject if not on grid."""
    return f"{qty:.8f}".rstrip("0").rstrip(".") or "0"


def _format_price(price: float) -> str:
    return f"{price:.8f}".rstrip("0").rstrip(".") or "0"


def _map_order_type(order_type: OrderType, tif: TimeInForce) -> str:
    """Translate (OrderType, TimeInForce) → OKX ``ordType``."""
    if order_type is OrderType.MARKET:
        return "market"
    # LIMIT — TIF determines the variant
    if tif is TimeInForce.IOC:
        return "ioc"
    if tif is TimeInForce.FOK:
        return "fok"
    return "limit"  # GTC


def _parse_order(row: dict[str, Any]) -> Order:
    """OKX order JSON → ``Order`` dataclass."""
    state = str(row.get("state", "live"))
    status_map = {
        "live": OrderStatus.OPEN,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELED,
        "mmp_canceled": OrderStatus.CANCELED,
    }
    status = status_map.get(state, OrderStatus.OPEN)
    side = Side.BUY if row.get("side") == "buy" else Side.SELL
    order_type_str = str(row.get("ordType", "limit"))
    order_type = OrderType.MARKET if order_type_str == "market" else OrderType.LIMIT
    return Order(
        order_id=str(row["ordId"]),
        client_order_id=str(row.get("clOrdId") or "") or None,
        symbol=str(row["instId"]),
        side=side,
        order_type=order_type,
        status=status,
        quantity=float(row.get("sz") or 0),
        filled_quantity=float(row.get("accFillSz") or 0),
        avg_fill_price=float(row.get("avgPx") or 0),
        price=float(row["px"]) if row.get("px") else None,
        timestamp=datetime.fromtimestamp(
            int(row.get("uTime") or row.get("cTime") or 0) / _MS_PER_S, tz=UTC,
        ),
    )
