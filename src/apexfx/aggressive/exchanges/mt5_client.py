"""MetaTrader 5 client implementing the ``Exchange`` protocol.

Why MT5 + Python
----------------
MT5 has a first-party Python package (``MetaTrader5`` on PyPI) that talks to
a *locally running* MT5 terminal via shared memory.  It's not REST — the
terminal handles broker connection, you handle strategy logic.

This means:

* **No webhook server needed** — the terminal is the data source AND the
  execution venue.  Just run Python on the same machine.
* **No TradingView subscription needed** — MT5 has its own bars/ticks.
* **Demo and live are the same code path** — pick the account at terminal
  login, not at API level.

Platform constraint
-------------------
``MetaTrader5`` package is **Windows-only**.  On Mac/Linux:
  * Easiest: rent a Windows VPS (RoboForex VPS, AEZA Windows, AWS Workspaces)
  * Or use Wine — works but fragile, not recommended
  * Or use the unofficial ``mt5linux`` bridge (relays calls to a Windows
    terminal over TCP) — works for development

For retail $1k production: rent a $5-10/mo Windows VPS, run MT5 + this
Python client there.  Latency to broker is what matters, not your local
machine.

Symbol conventions
------------------
Brokers add suffixes:
  * Standard account: ``EURUSD``
  * Cent account (RoboForex): ``EURUSDp``, ``EURUSDc``
  * ECN account (FxPro): ``EURUSD.ecn``
  * Pro accounts (IC Markets): ``EURUSD.``

The client doesn't normalize — pass the broker's exact symbol name to all
methods.  The strategy config file lists symbols per broker.

Magic number
------------
Every order we send carries a ``magic`` integer so we can filter "our"
positions from manual trades or other bots.  Default 770125 (= "APX1" in
ASCII offset).  Tests can override.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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
# MT5 module import — soft, so non-Windows can still import this file
# ---------------------------------------------------------------------------

try:
    import MetaTrader5 as _real_mt5  # type: ignore[import-untyped]
    _MT5_AVAILABLE = True
except ImportError:
    _real_mt5 = None  # type: ignore[assignment]
    _MT5_AVAILABLE = False


# Default magic number — "APX1" mapped to integer.  Used to tag every
# order we place so we can distinguish them from manual trades.
_DEFAULT_MAGIC = 770125


# Timeframe mapping — strategy speaks strings, MT5 wants constants.
# We keep our own dict instead of importing MT5 constants because mt5
# may be None on import-time (non-Windows machines).
_TIMEFRAME_CODES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 16385,
    "H4": 16388,
    "D1": 16408,
    "W1": 32769,
}


# MT5 trade return codes — only the few we map specifically.
_RETCODE_DONE = 10009            # TRADE_RETCODE_DONE — success
_RETCODE_DONE_PARTIAL = 10010    # success but partial fill
_RETCODE_NO_MONEY = 10019        # not enough margin
_RETCODE_TIMEOUT = 10008
_RETCODE_PRICE_OFF = 10021       # price too far / off-quote
_RETCODE_INVALID = 10013         # invalid request


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mt5Credentials:
    """Login info for the broker account.

    On a properly-configured Windows VPS where MT5 terminal is already
    logged in and running, you can skip these — just call
    ``Mt5Client(credentials=None)`` and it will attach to the running
    terminal.  Provide them if you want to FORCE the terminal to log in
    to a specific account (useful for switching demo/live programmatically).
    """

    login: int                 # MT5 account number
    password: str
    server: str                # e.g. "RoboForex-ECN", "FxPro-MT5"
    terminal_path: str | None = None  # Path to terminal64.exe; None ⇒ auto-detect


class Mt5Client(Exchange):
    """Local MT5 terminal client.

    Construction connects to (or starts) the terminal.  Always call
    ``shutdown()`` when done — the terminal connection holds resources.

    Parameters
    ----------
    credentials : Mt5Credentials | None
        Login info.  ``None`` ⇒ attach to whatever terminal is currently
        running and logged in.
    magic : int
        Order tag used to filter our trades.  Different bots / strategies
        on the same account should use different magic numbers.
    deviation_points : int
        Max price-deviation in points (= ticks) accepted for market orders.
        20 is a forgiving default; tighten to 5-10 for volatile assets.
    mt5_module : object | None
        For tests — inject a mock with the same interface as the real
        ``MetaTrader5`` module.  Production uses the real one.
    """

    def __init__(
        self,
        credentials: Mt5Credentials | None = None,
        *,
        magic: int = _DEFAULT_MAGIC,
        deviation_points: int = 20,
        mt5_module: object | None = None,
    ) -> None:
        self._mt5 = mt5_module if mt5_module is not None else _real_mt5
        if self._mt5 is None:
            raise ImportError(
                "MetaTrader5 package not available.  Install with `pip install "
                "MetaTrader5` on Windows, or pass mt5_module=<mock> in tests."
            )
        self._magic = magic
        self._deviation = deviation_points
        self._credentials = credentials
        self._initialized = False
        self._symbol_info_cache: dict[str, SymbolInfo] = {}

        self._initialize()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Connect to (or attach to) the MT5 terminal.

        Idempotent — multiple calls are no-ops after the first success.
        Raises ``AuthenticationError`` on login failure so callers can
        distinguish "creds wrong" from "terminal not installed".
        """
        if self._initialized:
            return

        if self._credentials is None:
            ok = self._mt5.initialize()
        else:
            c = self._credentials
            kwargs: dict[str, Any] = {
                "login": c.login,
                "password": c.password,
                "server": c.server,
            }
            if c.terminal_path:
                kwargs["path"] = c.terminal_path
            ok = self._mt5.initialize(**kwargs)

        if not ok:
            err = self._last_error()
            # Login errors are auth, anything else is generic
            if "auth" in err.lower() or "login" in err.lower() or "password" in err.lower():
                raise AuthenticationError(f"MT5 login failed: {err}")
            raise ExchangeError(f"MT5 initialize failed: {err}")

        self._initialized = True
        info = self._mt5.account_info()
        if info is not None:
            logger.info(
                "MT5 connected",
                login=info.login, server=info.server,
                currency=info.currency, balance=info.balance,
                trade_mode=info.trade_mode,
            )

    def shutdown(self) -> None:
        """Close terminal connection.  Safe to call multiple times."""
        if self._initialized and self._mt5 is not None:
            self._mt5.shutdown()
            self._initialized = False

    def __enter__(self) -> "Mt5Client":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()

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
        tf_code = _TIMEFRAME_CODES.get(interval)
        if tf_code is None:
            raise ValueError(
                f"interval must be one of {sorted(_TIMEFRAME_CODES)}, got {interval!r}"
            )
        if not 1 <= limit <= 5000:
            raise ValueError(f"limit must be in [1, 5000], got {limit}")

        self._ensure_symbol_visible(symbol)

        if end_time is None:
            rates = self._mt5.copy_rates_from_pos(symbol, tf_code, 0, limit)
        else:
            rates = self._mt5.copy_rates_from(symbol, tf_code, end_time, limit)

        if rates is None or len(rates) == 0:
            raise ExchangeError(
                f"copy_rates returned nothing for {symbol} {interval}: "
                f"{self._last_error()}"
            )

        # MT5 returns a numpy structured array with fields:
        #   time, open, high, low, close, tick_volume, spread, real_volume
        # Oldest-first (good — matches our convention)
        bars: list[Bar] = []
        for row in rates:
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(int(row["time"]), tz=UTC),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["tick_volume"]),
                )
            )
        return bars

    def get_ticker(self, symbol: str) -> Ticker:
        self._ensure_symbol_visible(symbol)
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ExchangeError(f"No tick for {symbol}: {self._last_error()}")
        return Ticker(
            symbol=symbol,
            last_price=float(tick.last) if getattr(tick, "last", 0) else float(tick.bid),
            bid=float(tick.bid),
            ask=float(tick.ask),
            timestamp=datetime.fromtimestamp(int(tick.time), tz=UTC),
        )

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        self._ensure_symbol_visible(symbol)
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise ExchangeError(f"No symbol info for {symbol}: {self._last_error()}")
        acct = self._mt5.account_info()
        max_lev = float(getattr(acct, "leverage", 1)) if acct else 1.0

        si = SymbolInfo(
            symbol=symbol,
            base_currency=str(getattr(info, "currency_base", "") or ""),
            quote_currency=str(getattr(info, "currency_profit", "") or ""),
            contract_size=float(info.trade_contract_size),
            tick_size=float(info.point),
            lot_size=float(info.volume_step),
            min_quantity=float(info.volume_min),
            max_leverage=max_lev,
        )
        self._symbol_info_cache[symbol] = si
        return si

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_balance(self, asset: str = "USD") -> Balance:
        """Account equity in the *deposit currency*.

        Note: ``asset`` is informational only — MT5 has one balance per
        account in one deposit currency.  We don't filter by asset.
        Caller should check ``balance.asset == expected_asset``.
        """
        info = self._mt5.account_info()
        if info is None:
            raise ExchangeError(f"Cannot read account_info: {self._last_error()}")
        return Balance(
            asset=str(info.currency),
            equity=float(info.equity),
            available=float(info.margin_free),
            timestamp=datetime.now(tz=UTC),
        )

    def get_positions(self) -> list[Position]:
        positions = self._mt5.positions_get()
        if positions is None:
            return []
        result: list[Position] = []
        for pos in positions:
            if pos.magic != self._magic:
                # Skip positions opened by other bots / manual trading
                continue
            # MT5: type 0 = BUY, 1 = SELL
            side = Side.BUY if pos.type == 0 else Side.SELL
            result.append(
                Position(
                    symbol=str(pos.symbol),
                    side=side,
                    quantity=float(pos.volume),
                    entry_price=float(pos.price_open),
                    leverage=0.0,  # MT5 doesn't expose per-position leverage
                    unrealized_pnl=float(pos.profit),
                    timestamp=datetime.fromtimestamp(int(pos.time), tz=UTC),
                )
            )
        return result

    def get_position(self, symbol: str) -> Position | None:
        # MT5 supports filtering on the call — saves filtering here.
        positions = self._mt5.positions_get(symbol=symbol)
        if not positions:
            return None
        # Filter by magic + sum volumes (broker may have hedging mode → multiple
        # positions in same symbol)
        ours = [p for p in positions if p.magic == self._magic]
        if not ours:
            return None
        # If multiple positions exist, aggregate into a single signed total.
        # Netting mode usually gives one; hedging mode may give 2 (long+short)
        # in which case we report whichever has bigger volume.
        if len(ours) == 1:
            p = ours[0]
            return Position(
                symbol=symbol,
                side=Side.BUY if p.type == 0 else Side.SELL,
                quantity=float(p.volume),
                entry_price=float(p.price_open),
                leverage=0.0,
                unrealized_pnl=float(p.profit),
                timestamp=datetime.fromtimestamp(int(p.time), tz=UTC),
            )
        # Multiple — pick dominant
        total_buy = sum(p.volume for p in ours if p.type == 0)
        total_sell = sum(p.volume for p in ours if p.type == 1)
        if total_buy >= total_sell:
            volume = total_buy - total_sell
            side = Side.BUY
        else:
            volume = total_sell - total_buy
            side = Side.SELL
        if volume == 0:
            return None
        avg_price = sum(p.price_open * p.volume for p in ours) / sum(
            p.volume for p in ours
        )
        return Position(
            symbol=symbol, side=side, quantity=float(volume),
            entry_price=float(avg_price), leverage=0.0,
            unrealized_pnl=float(sum(p.profit for p in ours)),
            timestamp=datetime.now(tz=UTC),
        )

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, req: OrderRequest) -> Order:
        self._ensure_symbol_visible(req.symbol)
        info = self.get_symbol_info(req.symbol)

        # Quantity in MT5 = lots (already)
        volume = self._snap_to_lot(req.quantity, info.lot_size)
        if volume < info.min_quantity:
            raise OrderRejectedError(
                f"Volume {volume} below symbol minimum {info.min_quantity}"
            )

        # Determine action + type
        if req.order_type is OrderType.MARKET:
            action = self._action_code("DEAL")
            order_type = self._order_type_code("BUY" if req.side is Side.BUY else "SELL")
            price = self._market_price(req.symbol, req.side)
        else:
            action = self._action_code("PENDING")
            if req.price is None:
                raise ValueError("LIMIT order needs price")
            price = req.price
            order_type = self._order_type_code(
                f"{'BUY' if req.side is Side.BUY else 'SELL'}_LIMIT"
            )

        request: dict[str, Any] = {
            "action": action,
            "symbol": req.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": req.client_order_id or "apexfx-turtle",
            "type_time": self._tif_code(req.time_in_force),
            "type_filling": self._guess_filling_mode(req.symbol),
        }

        if req.stop_loss is not None:
            request["sl"] = req.stop_loss
        if req.take_profit is not None:
            request["tp"] = req.take_profit

        # Reduce-only: in MT5, to close a position you target its ticket.
        # We don't take ticket in OrderRequest, so reduce_only orders rely
        # on the strategy passing the right side+volume; in MT5 netting
        # mode an opposite-direction trade just reduces exposure.

        result = self._mt5.order_send(request)
        if result is None:
            raise ExchangeError(f"order_send returned None: {self._last_error()}")

        retcode = int(result.retcode)
        if retcode in (_RETCODE_DONE, _RETCODE_DONE_PARTIAL):
            return self._build_order_from_result(req, result, info)

        # Error mapping
        comment = str(getattr(result, "comment", ""))
        if retcode == _RETCODE_NO_MONEY:
            raise InsufficientFundsError(f"MT5 {retcode}: {comment}")
        if retcode == _RETCODE_TIMEOUT:
            raise RateLimitError(f"MT5 timeout {retcode}: {comment}")
        raise OrderRejectedError(f"MT5 retcode {retcode}: {comment}")

    def cancel_order(self, symbol: str, order_id: str) -> None:
        """Cancel a pending order by its MT5 ticket number.

        Idempotent: 'order not found' is treated as no-op (already
        canceled or filled).
        """
        try:
            ticket = int(order_id)
        except ValueError as exc:
            raise ValueError(f"order_id must be a numeric MT5 ticket, got {order_id!r}") from exc

        request = {
            "action": self._action_code("REMOVE"),
            "order": ticket,
        }
        result = self._mt5.order_send(request)
        if result is None:
            err = self._last_error()
            if "not found" in err.lower():
                return  # idempotent
            raise ExchangeError(f"cancel_order failed: {err}")

        retcode = int(result.retcode)
        if retcode == _RETCODE_DONE:
            return
        # Not-found-style retcodes — treat as success
        if retcode in (10005, 10006):  # ORDER_NOT_FOUND, ORDER_CLOSED
            return
        raise OrderRejectedError(
            f"cancel_order retcode {retcode}: {getattr(result, 'comment', '')}"
        )

    def get_order(self, symbol: str, order_id: str) -> Order:
        try:
            ticket = int(order_id)
        except ValueError as exc:
            raise ValueError(f"order_id must be numeric MT5 ticket") from exc

        orders = self._mt5.orders_get(ticket=ticket)
        if orders:
            return _parse_order(orders[0])
        # Order may have already been filled — look in history
        history = self._mt5.history_orders_get(ticket=ticket)
        if history:
            return _parse_order(history[0], from_history=True)
        raise ExchangeError(f"Order {order_id} not found")

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        if symbol:
            orders = self._mt5.orders_get(symbol=symbol)
        else:
            orders = self._mt5.orders_get()
        if not orders:
            return []
        return [
            _parse_order(o)
            for o in orders
            if o.magic == self._magic
        ]

    def set_leverage(self, symbol: str, leverage: float) -> None:
        """MT5 has account-level leverage, not per-symbol.  This is a no-op
        with a warning so the strategy code stays portable."""
        if leverage <= 0:
            raise ValueError(f"leverage must be positive, got {leverage}")
        logger.warning(
            "set_leverage called on MT5 — has no effect; "
            "leverage is account-level, set in broker terminal",
            symbol=symbol, leverage=leverage,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_symbol_visible(self, symbol: str) -> None:
        """MT5 requires symbols to be 'visible' (selected) before read/trade.

        Idempotent — safe to call before every market-data fetch.
        """
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise ExchangeError(
                f"Unknown symbol {symbol} on this broker: {self._last_error()}"
            )
        if not info.visible:
            if not self._mt5.symbol_select(symbol, True):
                raise ExchangeError(
                    f"Cannot enable symbol {symbol}: {self._last_error()}"
                )

    def _market_price(self, symbol: str, side: Side) -> float:
        """Current ask (for buy) or bid (for sell) — required by MT5
        order_send even for market orders."""
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ExchangeError(f"No tick for {symbol}")
        return float(tick.ask) if side is Side.BUY else float(tick.bid)

    def _snap_to_lot(self, volume: float, lot_step: float) -> float:
        """Round volume to nearest valid lot step (floor)."""
        if lot_step <= 0:
            return volume
        return round((volume // lot_step) * lot_step, 8)

    def _guess_filling_mode(self, symbol: str) -> int:
        """Pick a filling type the broker supports.

        Brokers expose ``filling_mode`` as a bitmask of allowed modes
        (FOK / IOC / RETURN).  We prefer FOK (all-or-nothing), fall back
        to IOC (partial OK), then RETURN.
        """
        info = self._mt5.symbol_info(symbol)
        if info is None:
            return 0
        mode = int(getattr(info, "filling_mode", 0))
        # Constants per MT5 docs:
        #   FOK = 1, IOC = 2, RETURN = 4 (bitmask in filling_mode)
        if mode & 1:  # FOK supported
            return 0  # ORDER_FILLING_FOK
        if mode & 2:  # IOC supported
            return 1  # ORDER_FILLING_IOC
        return 2  # ORDER_FILLING_RETURN

    def _build_order_from_result(
        self, req: OrderRequest, result: Any, info: SymbolInfo,
    ) -> Order:
        """Construct our Order dataclass from MT5's TradeResult."""
        order_id = str(int(result.order))
        deal_id = str(int(getattr(result, "deal", 0))) if hasattr(result, "deal") else ""

        return Order(
            order_id=order_id,
            client_order_id=req.client_order_id or deal_id or None,
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            status=OrderStatus.FILLED if req.order_type is OrderType.MARKET
                   else OrderStatus.OPEN,
            quantity=float(result.volume) if hasattr(result, "volume") else req.quantity,
            filled_quantity=float(result.volume) if req.order_type is OrderType.MARKET
                            else 0.0,
            avg_fill_price=float(result.price) if hasattr(result, "price") else 0.0,
            price=req.price,
            timestamp=datetime.now(tz=UTC),
        )

    # -- MT5 constants accessed via module to keep tests mockable --

    def _action_code(self, name: str) -> int:
        mapping = {
            "DEAL": getattr(self._mt5, "TRADE_ACTION_DEAL", 1),
            "PENDING": getattr(self._mt5, "TRADE_ACTION_PENDING", 5),
            "MODIFY": getattr(self._mt5, "TRADE_ACTION_MODIFY", 7),
            "REMOVE": getattr(self._mt5, "TRADE_ACTION_REMOVE", 8),
        }
        return int(mapping[name])

    def _order_type_code(self, name: str) -> int:
        mapping = {
            "BUY": getattr(self._mt5, "ORDER_TYPE_BUY", 0),
            "SELL": getattr(self._mt5, "ORDER_TYPE_SELL", 1),
            "BUY_LIMIT": getattr(self._mt5, "ORDER_TYPE_BUY_LIMIT", 2),
            "SELL_LIMIT": getattr(self._mt5, "ORDER_TYPE_SELL_LIMIT", 3),
            "BUY_STOP": getattr(self._mt5, "ORDER_TYPE_BUY_STOP", 4),
            "SELL_STOP": getattr(self._mt5, "ORDER_TYPE_SELL_STOP", 5),
        }
        return int(mapping[name])

    def _tif_code(self, tif: TimeInForce) -> int:
        # MT5 time-types: GTC=0, DAY=1, SPECIFIED=2, SPECIFIED_DAY=3
        if tif is TimeInForce.GTC:
            return 0
        if tif is TimeInForce.IOC:
            return 0  # MT5 has no IOC at type_time level; use filling instead
        return 0

    def _last_error(self) -> str:
        """Get last MT5 error as readable string."""
        err = self._mt5.last_error()
        if isinstance(err, tuple) and len(err) >= 2:
            return f"{err[0]}: {err[1]}"
        return str(err)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_order(o: Any, *, from_history: bool = False) -> Order:
    """MT5 order/historyOrder object → our ``Order`` dataclass.

    ``from_history=True`` marks orders that came from ``history_orders_get``
    rather than ``orders_get`` — these are always in a terminal state.
    """
    # MT5 order type codes: 0/1 = market buy/sell, 2/3 = limit buy/sell, 4/5 = stop
    is_market = o.type in (0, 1)
    is_limit = o.type in (2, 3)
    side = Side.BUY if o.type in (0, 2, 4) else Side.SELL

    # State mapping:
    # MT5 order states: 0 STARTED, 1 PLACED, 2 CANCELED, 3 PARTIAL,
    #                   4 FILLED, 5 REJECTED, 6 EXPIRED, ...
    state = int(getattr(o, "state", 1))
    state_map = {
        0: OrderStatus.PENDING,
        1: OrderStatus.OPEN,
        2: OrderStatus.CANCELED,
        3: OrderStatus.PARTIALLY_FILLED,
        4: OrderStatus.FILLED,
        5: OrderStatus.REJECTED,
        6: OrderStatus.CANCELED,
    }
    status = state_map.get(state, OrderStatus.OPEN)
    if from_history and status is OrderStatus.OPEN:
        # Anything in history is terminal
        status = OrderStatus.FILLED

    return Order(
        order_id=str(int(o.ticket)),
        client_order_id=str(getattr(o, "comment", "")) or None,
        symbol=str(o.symbol),
        side=side,
        order_type=OrderType.MARKET if is_market else OrderType.LIMIT,
        status=status,
        quantity=float(o.volume_initial),
        filled_quantity=float(getattr(o, "volume_initial", 0)) - float(
            getattr(o, "volume_current", o.volume_initial)
        ),
        avg_fill_price=float(getattr(o, "price_current", 0)),
        price=float(o.price_open) if is_limit else None,
        timestamp=datetime.fromtimestamp(
            int(getattr(o, "time_setup", time.time())), tz=UTC,
        ),
    )
