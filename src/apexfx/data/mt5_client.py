"""MetaTrader 5 API wrapper with typed interface and error handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Any

import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# MT5 timeframe mapping
MT5_TIMEFRAMES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
    "W1": 32769,
    "MN1": 49153,
}


class OrderType(IntEnum):
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5


class TradeAction(IntEnum):
    DEAL = 1
    PENDING = 5
    SLTP = 6
    MODIFY = 7
    REMOVE = 8


@dataclass
class SymbolInfo:
    name: str
    bid: float
    ask: float
    spread: float
    point: float
    trade_tick_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    trade_contract_size: float


@dataclass
class TradeRequest:
    action: TradeAction
    symbol: str
    volume: float
    type: OrderType
    price: float
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 20
    magic: int = 234000
    comment: str = "ApexFX"
    type_filling: int = 2  # IOC


@dataclass
class TradeResult:
    retcode: int
    deal: int
    order: int
    volume: float
    price: float
    comment: str

    @property
    def success(self) -> bool:
        return self.retcode == 10009  # TRADE_RETCODE_DONE


@dataclass
class Position:
    ticket: int
    symbol: str
    type: int  # 0=buy, 1=sell
    volume: float
    price_open: float
    price_current: float
    profit: float
    sl: float
    tp: float
    time: datetime


class MT5Client:
    """Thread-safe MetaTrader 5 API wrapper."""

    def __init__(
        self,
        path: str | None = None,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
    ) -> None:
        self._path = path or os.environ.get("MT5_PATH", "")
        self._login = login or int(os.environ.get("MT5_LOGIN", "0"))
        self._password = password or os.environ.get("MT5_PASSWORD", "")
        self._server = server or os.environ.get("MT5_SERVER", "")
        self._connected = False
        self._mt5: Any = None

    def connect(self) -> None:
        """Initialize and connect to MT5 terminal."""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            logger.warning("MetaTrader5 package not available. Using mock mode.")
            self._mt5 = None
            return

        init_kwargs: dict[str, Any] = {}
        if self._path:
            init_kwargs["path"] = self._path
        if self._login:
            init_kwargs["login"] = self._login
        if self._password:
            init_kwargs["password"] = self._password
        if self._server:
            init_kwargs["server"] = self._server

        if not self._mt5.initialize(**init_kwargs):
            error = self._mt5.last_error()
            logger.error("MT5 initialization failed", error=error)
            raise ConnectionError(f"MT5 init failed: {error}")

        self._connected = True
        info = self._mt5.terminal_info()
        logger.info("MT5 connected", build=info.build if info else "unknown")

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def _ensure_connected(self) -> None:
        if not self._connected or self._mt5 is None:
            raise ConnectionError("MT5 not connected. Call connect() first.")

    def get_ticks(
        self, symbol: str, from_dt: datetime, count: int = 1000
    ) -> pd.DataFrame:
        """Get tick data from MT5."""
        self._ensure_connected()
        ticks = self._mt5.copy_ticks_from(symbol, from_dt, count, self._mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame(columns=["time", "bid", "ask", "last", "volume", "flags"])

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # Validate tick data: reject zero/negative prices and NaN
        pre_len = len(df)
        if "bid" in df.columns and "ask" in df.columns:
            df = df[
                (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])
            ].copy()
            df = df.dropna(subset=["bid", "ask"])
        if len(df) < pre_len:
            logger.warning(
                "Dropped invalid ticks",
                symbol=symbol,
                dropped=pre_len - len(df),
            )

        return df

    def get_ticks_range(
        self, symbol: str, from_dt: datetime, to_dt: datetime
    ) -> pd.DataFrame:
        """Get tick data in a time range."""
        self._ensure_connected()
        ticks = self._mt5.copy_ticks_range(symbol, from_dt, to_dt, self._mt5.COPY_TICKS_ALL)
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame(columns=["time", "bid", "ask", "last", "volume", "flags"])

        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        from_dt: datetime | None = None,
        count: int = 500,
    ) -> pd.DataFrame:
        """Get OHLCV bar data from MT5."""
        self._ensure_connected()
        tf = MT5_TIMEFRAMES.get(timeframe)
        if tf is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        if from_dt is not None:
            rates = self._mt5.copy_rates_from(symbol, tf, from_dt, count)
        else:
            rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            return pd.DataFrame(
                columns=["time", "open", "high", "low", "close", "tick_volume", "spread"]
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

        # Validate bar data integrity
        pre_len = len(df)
        df = df[
            (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["high"] >= df["low"])
            & (df["close"] >= df["low"])
            & (df["close"] <= df["high"])
        ].copy()
        if len(df) < pre_len:
            logger.warning(
                "Dropped invalid bars",
                symbol=symbol,
                dropped=pre_len - len(df),
            )

        return df

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Get current symbol information including bid/ask."""
        self._ensure_connected()
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol not found: {symbol}")

        tick = self._mt5.symbol_info_tick(symbol)
        bid = tick.bid if tick else 0.0
        ask = tick.ask if tick else 0.0

        return SymbolInfo(
            name=symbol,
            bid=bid,
            ask=ask,
            spread=(ask - bid) / info.point if info.point > 0 else 0,
            point=info.point,
            trade_tick_size=info.trade_tick_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_contract_size=info.trade_contract_size,
        )

    def send_order(self, request: TradeRequest) -> TradeResult:
        """Send a trade order to MT5."""
        self._ensure_connected()
        mt5_request = {
            "action": int(request.action),
            "symbol": request.symbol,
            "volume": request.volume,
            "type": int(request.type),
            "price": request.price,
            "sl": request.sl,
            "tp": request.tp,
            "deviation": request.deviation,
            "magic": request.magic,
            "comment": request.comment,
            "type_filling": request.type_filling,
            "type_time": 0,  # GTC
        }

        result = self._mt5.order_send(mt5_request)
        if result is None:
            error = self._mt5.last_error()
            logger.error("Order send failed", error=error)
            return TradeResult(
                retcode=-1, deal=0, order=0, volume=0, price=0, comment=str(error)
            )

        logger.info(
            "Order sent",
            retcode=result.retcode,
            symbol=request.symbol,
            volume=request.volume,
            type=request.type.name,
        )

        return TradeResult(
            retcode=result.retcode,
            deal=result.deal,
            order=result.order,
            volume=result.volume,
            price=result.price,
            comment=result.comment,
        )

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions."""
        self._ensure_connected()
        if symbol:
            positions = self._mt5.positions_get(symbol=symbol)
        else:
            positions = self._mt5.positions_get()

        if positions is None:
            return []

        return [
            Position(
                ticket=p.ticket,
                symbol=p.symbol,
                type=p.type,
                volume=p.volume,
                price_open=p.price_open,
                price_current=p.price_current,
                profit=p.profit,
                sl=p.sl,
                tp=p.tp,
                time=datetime.fromtimestamp(p.time),
            )
            for p in positions
        ]

    def get_order_book(self, symbol: str, depth: int = 10) -> dict:
        """Get L2 order book (Depth of Market) from MT5.

        Args:
            symbol: Instrument symbol.
            depth: Number of price levels on each side.

        Returns:
            Dictionary with 'bids' and 'asks', each a list of
            {'price': float, 'volume': float} entries.
        """
        self._ensure_connected()
        try:
            # Enable book subscription for this symbol
            self._mt5.market_book_add(symbol)
            book = self._mt5.market_book_get(symbol)
            if book is None:
                return {"bids": [], "asks": []}

            bids = []
            asks = []
            for entry in book:
                item = {"price": entry.price, "volume": float(entry.volume)}
                if entry.type == 1:  # BOOK_TYPE_SELL
                    asks.append(item)
                else:  # BOOK_TYPE_BUY
                    bids.append(item)

            # Limit to requested depth
            bids = sorted(bids, key=lambda x: -x["price"])[:depth]
            asks = sorted(asks, key=lambda x: x["price"])[:depth]

            return {"bids": bids, "asks": asks}
        except Exception as e:
            logger.warning("Order book fetch failed", symbol=symbol, error=str(e))
            return {"bids": [], "asks": []}
        finally:
            try:
                self._mt5.market_book_release(symbol)
            except Exception:
                pass

    def close_position(self, ticket: int) -> TradeResult:
        """Close a position by ticket."""
        self._ensure_connected()
        positions = self._mt5.positions_get(ticket=ticket)
        if not positions:
            return TradeResult(retcode=-1, deal=0, order=0, volume=0, price=0, comment="Position not found")

        pos = positions[0]
        close_type = OrderType.SELL if pos.type == 0 else OrderType.BUY
        tick = self._mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if close_type == OrderType.SELL else tick.ask

        request = TradeRequest(
            action=TradeAction.DEAL,
            symbol=pos.symbol,
            volume=pos.volume,
            type=close_type,
            price=price,
            comment="ApexFX Close",
        )
        return self.send_order(request)
