"""Abstract exchange interface — what every exchange (OKX, Bybit, ...) must implement.

Why an abstract layer if we picked OKX?
----------------------------------------
Two reasons:

1. **Testability.** The strategy logic should be testable without hitting a
   real API. A mock exchange is one ``Protocol`` satisfaction away.

2. **Optionality.** If OKX KYC fails or the user wants to migrate to OKX
   demo / Bybit / Hyperliquid, the strategy code does not change — only
   the exchange client does.  Without an interface this becomes a rewrite.

The interface deliberately exposes the **minimum** surface area we actually
need for Turtle-style trading: market data, account balance, place/cancel
orders, query positions.  Streaming (WebSocket) is in a sibling protocol
because not all clients need it (backtest, paper trade, unit tests).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Side(str, Enum):
    """Direction of an order or position."""

    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for long (buy), -1 for short (sell). Useful for PnL math."""
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    """Order types we actually use.  Stop / TP variants are encoded as
    separate fields on the order request, not as a different ``OrderType``,
    because OKX/Bybit treat them as conditional on a base order."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    """Lifecycle states.  Mirrors OKX's ``state`` field with additions for
    locally-tracked unsubmitted/rejected orders."""

    PENDING = "pending"          # Locally created, not yet sent
    OPEN = "open"                # Live on the exchange, unfilled
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        """True once the order can no longer change (filled/canceled/rejected)."""
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


class TimeInForce(str, Enum):
    """How long an order remains active."""

    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel — fill what you can, cancel the rest
    FOK = "fok"  # Fill or kill — all or nothing


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """OHLC + volume bar.  Timestamp is the bar's *open* time in UTC."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range_pct(self) -> float:
        """Bar range as percent of close — useful for vol filters."""
        if self.close <= 0:
            return 0.0
        return (self.high - self.low) / self.close


@dataclass(frozen=True)
class Ticker:
    """Latest ticker snapshot.  Used for entry-price and slippage estimates."""

    symbol: str
    last_price: float
    bid: float
    ask: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid > 0 and self.ask > 0 else self.last_price

    @property
    def spread_pct(self) -> float:
        """Spread as percent of mid — for slippage budgets."""
        m = self.mid
        if m <= 0:
            return 0.0
        return (self.ask - self.bid) / m


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Balance:
    """Account balance snapshot — only the metric we need for sizing.

    On OKX USDT-M perpetual the relevant number is ``equity`` (mark-to-market
    USD value of the futures wallet) — we use this for position-size
    calculations to make our risk fraction adapt to PnL accumulation.
    """

    asset: str            # "USDT" for our case
    equity: float         # Total equity including unrealized PnL
    available: float      # Free margin (= equity − used margin)
    timestamp: datetime


@dataclass(frozen=True)
class Position:
    """Open position on a single symbol/contract.

    Quantity convention: positive = long, negative = short.  Zero ⇒ no
    open position even if the row exists (some exchanges keep zeroed
    rows for previously-traded symbols).
    """

    symbol: str
    side: Side
    quantity: float       # Always positive; combine with ``side`` for direction
    entry_price: float
    leverage: float
    unrealized_pnl: float
    timestamp: datetime

    @property
    def signed_quantity(self) -> float:
        """+qty for long, −qty for short.  Convenience for arithmetic."""
        return self.quantity * self.side.sign

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0.0


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderRequest:
    """Everything needed to submit a new order.

    ``client_order_id`` is locally-generated — letting the exchange echo it
    back is how we correlate fills to the strategy decision that produced
    them, without trusting wall-clock timing.
    """

    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: float | None = None        # Required for LIMIT
    stop_loss: float | None = None    # Optional protective SL price
    take_profit: float | None = None  # Optional TP target
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False         # Closing trade only — won't open new exposure
    client_order_id: str | None = None
    leverage: float | None = None     # If set, exchange will adjust margin mode

    def __post_init__(self) -> None:
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise ValueError("LIMIT order requires price")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")


@dataclass(frozen=True)
class Order:
    """Order state as returned by the exchange.  Read-only snapshot."""

    order_id: str                 # Exchange-assigned id
    client_order_id: str | None
    symbol: str
    side: Side
    order_type: OrderType
    status: OrderStatus
    quantity: float
    filled_quantity: float
    avg_fill_price: float
    price: float | None
    timestamp: datetime

    @property
    def remaining_quantity(self) -> float:
        return max(self.quantity - self.filled_quantity, 0.0)


# ---------------------------------------------------------------------------
# Symbol metadata (instrument spec)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolInfo:
    """Static info about a tradable instrument.  Cached after first fetch.

    These fields drive the position-size rounder: we must round ``quantity``
    to a multiple of ``lot_size`` and ``price`` to ``tick_size``, otherwise
    the exchange rejects the order.
    """

    symbol: str
    base_currency: str            # "BTC" in BTC-USDT-SWAP
    quote_currency: str           # "USDT"
    contract_size: float          # e.g. 0.01 BTC per contract on OKX BTC-USDT-SWAP
    tick_size: float              # Min price increment
    lot_size: float               # Min quantity increment
    min_quantity: float           # Min order size (in contracts)
    max_leverage: float


# ---------------------------------------------------------------------------
# Exchange interface
# ---------------------------------------------------------------------------


class Exchange(Protocol):
    """The minimum interface every exchange client must implement.

    All methods are synchronous — async streaming is in
    ``StreamingExchange`` because not every consumer (backtest, mock for
    tests) needs WebSockets.  Real-time updates aren't required for a 4H
    strategy that only acts on bar-close.
    """

    # -- Market data --

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
        end_time: datetime | None = None,
    ) -> list[Bar]:
        """Fetch historical OHLC bars.  ``interval`` is exchange-specific
        (e.g. ``"4H"`` for OKX).  ``end_time`` lets us page backward."""

    @abstractmethod
    def get_ticker(self, symbol: str) -> Ticker:
        """Latest mark/bid/ask for ``symbol``."""

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """Static contract spec.  Cache the result — it rarely changes."""

    # -- Account --

    @abstractmethod
    def get_balance(self, asset: str = "USDT") -> Balance:
        """Account equity and free margin in ``asset``."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """All open positions across all symbols.  Empty list ⇒ flat."""

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Specific position, or ``None`` if flat."""

    # -- Orders --

    @abstractmethod
    def place_order(self, req: OrderRequest) -> Order:
        """Submit a new order.  Returns the exchange-assigned ``Order``
        with its ``order_id``.  Raises ``ExchangeError`` on rejection."""

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> None:
        """Cancel a still-open order.  Idempotent — canceling a filled
        or already-canceled order is a no-op (no exception)."""

    @abstractmethod
    def get_order(self, symbol: str, order_id: str) -> Order:
        """Latest known state of a specific order."""

    @abstractmethod
    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """All non-terminal orders, optionally filtered by symbol."""

    # -- Leverage / margin mode --

    @abstractmethod
    def set_leverage(self, symbol: str, leverage: float) -> None:
        """Adjust leverage for a symbol.  May fail if there's an open
        position on a different leverage."""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExchangeError(Exception):
    """Base for all exchange-related errors.  Wraps both API-rejection
    errors and network-layer failures so the strategy doesn't have to
    distinguish them — at the strategy level, both mean "abort this
    decision and retry later".
    """


class AuthenticationError(ExchangeError):
    """API credentials missing/wrong/expired.  Non-recoverable inside
    a single run — caller must intervene."""


class InsufficientFundsError(ExchangeError):
    """Order rejected because available margin < required.  Recoverable
    by sizing down."""


class RateLimitError(ExchangeError):
    """Hit the exchange's request quota.  Recoverable by waiting; the
    OKX client implements exponential backoff for these."""


class OrderRejectedError(ExchangeError):
    """Order rejected by the exchange for reasons other than funds
    (price out of bounds, lot size, post-only conflict, etc.)."""
