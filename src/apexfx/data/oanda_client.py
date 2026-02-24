"""OANDA v20 REST API client — cross-platform alternative to MetaTrader 5.

Why OANDA instead of MT5
------------------------
MetaTrader 5 is **Windows-only**.  OANDA's REST API v20 works on any platform
(Linux, macOS, Windows, Docker containers) and supports both live and practice
accounts.  This client implements the same interface as :class:`MT5Client` so
the rest of the system (executor, data collection, live trading loop) can use
either broker without code changes.

Setup
-----
1. Create a free practice account at https://www.oanda.com
2. Generate an API token in My Services → API Access
3. Set environment variables (or put them in ``.env``)::

       OANDA_API_TOKEN=your-api-token
       OANDA_ACCOUNT_ID=your-account-id
       OANDA_ENVIRONMENT=practice   # or "live"

4. Install the dependency::

       pip install "apexfx-quantum[oanda]"

Instrument mapping
------------------
OANDA uses underscore notation: ``EURUSD`` → ``EUR_USD``.
The client converts automatically in both directions.

Limitations vs MT5
------------------
* Tick data: OANDA provides pricing stream (WebSocket), not MT5-style tick copy.
  Use the pricing stream endpoint or poll pricing at high frequency for ticks.
* Volume: Forex tick volume is from OANDA's internal order flow, not exchange volume.
* Speed: REST round-trip adds ~50–200 ms latency vs direct MT5 DLL call.
  Suitable for H1/H4/D1 timeframes; not suitable for M1 scalping.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from apexfx.data.mt5_client import Position, SymbolInfo, TradeResult
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# OANDA granularity mapping (Forex timeframes)
_OANDA_GRANULARITY: dict[str, str] = {
    "M1": "M1",
    "M5": "M5",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
    "D1": "D",
    "W1": "W",
    "MN1": "M",
}

_BASE_URLS = {
    "live": "https://api-fxtrade.oanda.com",
    "practice": "https://api-fxpractice.oanda.com",
}

_STREAM_URLS = {
    "live": "https://stream-fxtrade.oanda.com",
    "practice": "https://stream-fxpractice.oanda.com",
}


def _to_oanda_instrument(symbol: str) -> str:
    """Convert 'EURUSD' → 'EUR_USD'."""
    if "_" in symbol:
        return symbol
    if len(symbol) == 6:
        return f"{symbol[:3]}_{symbol[3:]}"
    return symbol


def _from_oanda_instrument(instrument: str) -> str:
    """Convert 'EUR_USD' → 'EURUSD'."""
    return instrument.replace("_", "")


@dataclass
class OANDAOrderRequest:
    """OANDA order request structure."""
    instrument: str
    units: int          # positive = buy, negative = sell
    order_type: str = "MARKET"   # MARKET | LIMIT | STOP
    price: float | None = None   # required for LIMIT/STOP orders
    stop_loss: float | None = None
    take_profit: float | None = None
    client_order_id: str = "ApexFX"


class OANDAClient:
    """Cross-platform OANDA v20 REST API broker client.

    Implements the same high-level interface as :class:`~apexfx.data.mt5_client.MT5Client`
    so :class:`~apexfx.execution.executor.Executor` and
    :class:`~apexfx.live.trading_loop.LiveTradingLoop` can use either broker
    without modifications.

    Parameters
    ----------
    api_token:
        OANDA API access token.  Defaults to ``OANDA_API_TOKEN`` env var.
    account_id:
        OANDA account ID.  Defaults to ``OANDA_ACCOUNT_ID`` env var.
    environment:
        ``"practice"`` (default) or ``"live"``.  Defaults to ``OANDA_ENVIRONMENT``.
    timeout:
        HTTP request timeout in seconds (default 10).
    """

    def __init__(
        self,
        api_token: str | None = None,
        account_id: str | None = None,
        environment: str | None = None,
        timeout: int = 10,
    ) -> None:
        self._token = api_token or os.environ.get("OANDA_API_TOKEN", "")
        self._account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "")
        self._env = (environment or os.environ.get("OANDA_ENVIRONMENT", "practice")).lower()
        self._timeout = timeout
        self._connected = False
        self._session: Any = None  # requests.Session

        if self._env not in _BASE_URLS:
            raise ValueError(f"environment must be 'live' or 'practice', got '{self._env}'")

        self._base_url = _BASE_URLS[self._env]

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Validate credentials by fetching account summary."""
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            })
        except ImportError:
            raise ImportError(
                "requests is required for OANDA: pip install 'apexfx-quantum[oanda]'"
            )

        resp = self._get(f"/v3/accounts/{self._account_id}/summary")
        balance = resp.get("account", {}).get("balance", "?")
        currency = resp.get("account", {}).get("currency", "?")
        self._connected = True
        logger.info(
            "OANDA connected",
            account=self._account_id,
            environment=self._env,
            balance=balance,
            currency=currency,
        )

    def disconnect(self) -> None:
        if self._session:
            self._session.close()
        self._connected = False
        logger.info("OANDA disconnected")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        from_dt: datetime | None = None,
        count: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data from OANDA.

        Parameters
        ----------
        symbol:
            Standard symbol (``EURUSD``, ``GBPUSD``, etc.).
        timeframe:
            Standard timeframe string (``M1``, ``H1``, ``D1``, etc.).
        from_dt:
            Start datetime (UTC).  When None, returns the most recent ``count`` bars.
        count:
            Number of candles to fetch (max 5000 per request).
        """
        self._ensure_connected()
        instrument = _to_oanda_instrument(symbol)
        granularity = _OANDA_GRANULARITY.get(timeframe, "H1")

        params: dict[str, Any] = {
            "granularity": granularity,
            "count": min(count, 5000),
            "price": "M",  # Mid prices
        }
        if from_dt is not None:
            params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            params.pop("count", None)

        resp = self._get(f"/v3/instruments/{instrument}/candles", params=params)
        candles = resp.get("candles", [])

        if not candles:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c.get("mid", {})
            rows.append({
                "time": pd.Timestamp(c["time"]).tz_localize(None)
                if pd.Timestamp(c["time"]).tzinfo is None
                else pd.Timestamp(c["time"]).tz_convert("UTC").tz_localize(None),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })

        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.sort_values("time").reset_index(drop=True)

    def get_ticks(
        self, symbol: str, from_dt: datetime, count: int = 1000
    ) -> pd.DataFrame:
        """Approximate tick data via high-frequency M1 candles.

        OANDA does not expose raw tick history via REST.  This method
        returns M1 OHLCV bars as a proxy — sufficient for feature computation
        and bar aggregation purposes.
        """
        return self.get_bars(symbol, "M1", from_dt=from_dt, count=count)

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get current bid/ask pricing for a symbol."""
        self._ensure_connected()
        instrument = _to_oanda_instrument(symbol)
        resp = self._get(
            f"/v3/accounts/{self._account_id}/pricing",
            params={"instruments": instrument},
        )
        prices = resp.get("prices", [])
        if not prices:
            raise ValueError(f"No pricing data returned for {symbol}")

        p = prices[0]
        bids = p.get("bids", [{}])
        asks = p.get("asks", [{}])
        bid = float(bids[0].get("price", 0)) if bids else 0.0
        ask = float(asks[0].get("price", 0)) if asks else 0.0

        # Fetch instrument spec for pip value etc.
        spec_resp = self._get(f"/v3/instruments/{instrument}")
        spec = spec_resp.get("instrument", {})
        pip_loc = int(spec.get("pipLocation", -4))
        pip_value = 10 ** pip_loc

        return SymbolInfo(
            name=symbol,
            bid=bid,
            ask=ask,
            spread=(ask - bid) / pip_value if pip_value > 0 else 0.0,
            point=pip_value,
            trade_tick_size=pip_value,
            volume_min=1,        # OANDA uses units, not lots; 1 unit minimum
            volume_max=100_000_000,
            volume_step=1,
            trade_contract_size=1,  # OANDA quotes in units already
        )

    # ------------------------------------------------------------------
    # Order execution (compatible with MT5Client interface)
    # ------------------------------------------------------------------

    def send_order(self, request: OANDAOrderRequest) -> TradeResult:
        """Submit a market or limit order to OANDA."""
        self._ensure_connected()
        instrument = _to_oanda_instrument(request.instrument)

        order_body: dict[str, Any] = {
            "order": {
                "type": request.order_type,
                "instrument": instrument,
                "units": str(request.units),
                "timeInForce": "FOK" if request.order_type == "MARKET" else "GTC",
            }
        }

        if request.order_type in ("LIMIT", "STOP") and request.price is not None:
            order_body["order"]["price"] = str(request.price)

        if request.stop_loss is not None:
            order_body["order"]["stopLossOnFill"] = {"price": str(request.stop_loss)}
        if request.take_profit is not None:
            order_body["order"]["takeProfitOnFill"] = {"price": str(request.take_profit)}

        try:
            resp = self._post(f"/v3/accounts/{self._account_id}/orders", body=order_body)
        except Exception as e:
            logger.error("OANDA order failed", error=str(e))
            return TradeResult(retcode=-1, deal=0, order=0, volume=0.0, price=0.0, comment=str(e))

        fill = resp.get("orderFillTransaction", {})
        trade_id = int(fill.get("tradeOpened", {}).get("tradeID", 0) or 0)
        fill_price = float(fill.get("price", 0) or 0)
        fill_units = abs(int(fill.get("units", 0) or 0))

        logger.info(
            "OANDA order filled",
            instrument=instrument,
            units=request.units,
            fill_price=fill_price,
            trade_id=trade_id,
        )

        # Translate to MT5-compatible TradeResult (retcode 10009 = success)
        return TradeResult(
            retcode=10009,
            deal=trade_id,
            order=trade_id,
            volume=float(fill_units),
            price=fill_price,
            comment="filled",
        )

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions, optionally filtered by symbol."""
        self._ensure_connected()

        if symbol:
            instrument = _to_oanda_instrument(symbol)
            resp = self._get(f"/v3/accounts/{self._account_id}/positions/{instrument}")
            raw = resp.get("position", {})
            return self._parse_position(raw, symbol)
        else:
            resp = self._get(f"/v3/accounts/{self._account_id}/openPositions")
            positions = []
            for raw in resp.get("positions", []):
                sym = _from_oanda_instrument(raw.get("instrument", ""))
                positions.extend(self._parse_position(raw, sym))
            return positions

    def close_position(self, ticket: int) -> TradeResult:
        """Close a position by OANDA trade ID."""
        self._ensure_connected()
        resp = self._put(
            f"/v3/accounts/{self._account_id}/trades/{ticket}/close"
        )
        fill = resp.get("orderFillTransaction", {})
        price = float(fill.get("price", 0) or 0)
        return TradeResult(
            retcode=10009,
            deal=ticket,
            order=ticket,
            volume=abs(float(fill.get("units", 0) or 0)),
            price=price,
            comment="closed",
        )

    def close_symbol_position(self, symbol: str) -> TradeResult:
        """Close ALL units of a symbol's position (long + short)."""
        self._ensure_connected()
        instrument = _to_oanda_instrument(symbol)
        body = {"longUnits": "ALL", "shortUnits": "ALL"}
        try:
            resp = self._put(
                f"/v3/accounts/{self._account_id}/positions/{instrument}/close",
                body=body,
            )
            price_long = float(
                (resp.get("longOrderFillTransaction") or {}).get("price", 0) or 0
            )
            price_short = float(
                (resp.get("shortOrderFillTransaction") or {}).get("price", 0) or 0
            )
            price = price_long or price_short
            return TradeResult(retcode=10009, deal=0, order=0, volume=0.0, price=price, comment="closed")
        except Exception as e:
            return TradeResult(retcode=-1, deal=0, order=0, volume=0.0, price=0.0, comment=str(e))

    # ------------------------------------------------------------------
    # Historical data download (bulk — for DataStore population)
    # ------------------------------------------------------------------

    def download_bars(
        self,
        symbol: str,
        timeframe: str,
        from_dt: datetime,
        to_dt: datetime | None = None,
        batch_size: int = 5000,
    ) -> pd.DataFrame:
        """Download historical bars in batches, respecting OANDA's 5000-candle limit.

        Parameters
        ----------
        symbol:
            Trading pair (``EURUSD``, etc.).
        timeframe:
            Timeframe string (``M1``, ``H1``, etc.).
        from_dt:
            Start datetime (UTC, inclusive).
        to_dt:
            End datetime (UTC, inclusive).  Defaults to now.
        batch_size:
            Candles per API request (max 5000).
        """
        to_dt = to_dt or datetime.now(timezone.utc)
        instrument = _to_oanda_instrument(symbol)
        granularity = _OANDA_GRANULARITY.get(timeframe, "H1")

        all_bars: list[pd.DataFrame] = []
        cursor = from_dt
        request_count = 0

        while cursor < to_dt:
            params = {
                "granularity": granularity,
                "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": to_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "count": batch_size,
                "price": "M",
            }
            resp = self._get(f"/v3/instruments/{instrument}/candles", params=params)
            candles = resp.get("candles", [])
            if not candles:
                break

            batch = self._candles_to_df(candles)
            if batch.empty:
                break

            all_bars.append(batch)
            last_time = batch["time"].max()
            if last_time <= cursor:
                break
            cursor = last_time.to_pydatetime() + pd.Timedelta(seconds=1)
            request_count += 1

            # Respect OANDA rate limit (120 req/sec for v20)
            time.sleep(0.05)

            logger.debug(
                "OANDA download batch",
                symbol=symbol,
                timeframe=timeframe,
                batch=request_count,
                bars=len(batch),
                cursor=str(cursor),
            )

        if not all_bars:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

        result = pd.concat(all_bars, ignore_index=True).drop_duplicates("time")
        result = result[result["time"] <= pd.Timestamp(to_dt, tz="UTC")]
        return result.sort_values("time").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        if not self._connected or self._session is None:
            raise ConnectionError("OANDAClient not connected. Call connect() first.")

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self._base_url}{endpoint}"
        resp = self._session.get(url, params=params, timeout=self._timeout)
        self._raise_for_status(resp)
        return resp.json()

    def _post(self, endpoint: str, body: dict) -> dict:
        import json
        url = f"{self._base_url}{endpoint}"
        resp = self._session.post(url, data=json.dumps(body), timeout=self._timeout)
        self._raise_for_status(resp)
        return resp.json()

    def _put(self, endpoint: str, body: dict | None = None) -> dict:
        import json
        url = f"{self._base_url}{endpoint}"
        data = json.dumps(body) if body else None
        resp = self._session.put(url, data=data, timeout=self._timeout)
        self._raise_for_status(resp)
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: Any) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise ConnectionError(
                f"OANDA API error {resp.status_code}: {detail}"
            )

    @staticmethod
    def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
        rows = []
        for c in candles:
            if not c.get("complete", True):
                continue
            mid = c.get("mid", {})
            rows.append({
                "time": pd.Timestamp(c["time"], tz="UTC"),
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(c.get("volume", 0)),
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

    def _parse_position(self, raw: dict, symbol: str) -> list[Position]:
        """Parse OANDA position dict into list of Position objects."""
        positions = []
        for side, pos_type in [("long", 0), ("short", 1)]:
            side_data = raw.get(side, {})
            units = int(side_data.get("units", 0) or 0)
            if units == 0:
                continue
            avg_price = float(side_data.get("averagePrice", 0) or 0)
            unrealized = float(side_data.get("unrealizedPL", 0) or 0)
            trade_ids = side_data.get("tradeIDs", [0])
            ticket = int(trade_ids[0]) if trade_ids else 0

            positions.append(Position(
                ticket=ticket,
                symbol=symbol,
                type=pos_type,
                volume=abs(units),
                price_open=avg_price,
                price_current=avg_price,  # would need separate pricing call for current
                profit=unrealized,
                sl=0.0,
                tp=0.0,
                time=datetime.now(timezone.utc),
            ))
        return positions
