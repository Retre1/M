"""Signal handler: ``TradingViewAlert`` → OKX order execution.

This is the **policy layer** between the dumb webhook receiver (which only
parses + authenticates) and the OKX exchange client (which only places
orders).  All business decisions live here:

  • Translate TV symbol format → OKX symbol format (``BTCUSDT.P`` → ``BTC-USDT-SWAP``)
  • Decide order type (always MARKET for entries — slippage is acceptable
    cost for guaranteed fill on a Donchian breakout)
  • Compute SL/TP attached to the order
  • Reject signals that would breach risk limits (delegated to risk engine)
  • Map exit signals to ``reduce_only=True`` orders so we never accidentally
    flip direction

Why separate from ``server.py``
-------------------------------
The Flask server is dumb plumbing — receive request, validate, hand off,
return response.  Putting business logic there means every test needs a
Flask test client.  This file is pure functions over well-typed data,
so it's testable with a mock ``Exchange`` and no HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass

from apexfx.aggressive.exchanges.base import (
    Exchange,
    InsufficientFundsError,
    Order,
    OrderRejectedError,
    OrderRequest,
    OrderType,
    Side,
)
from apexfx.aggressive.webhook.models import (
    AlertAction,
    AlertSide,
    TradingViewAlert,
)
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Symbol translation
# ---------------------------------------------------------------------------


def tv_symbol_to_okx(tv_symbol: str) -> str:
    """Map TradingView ticker → OKX instrument id.

    TradingView convention varies by data provider:
      * Crypto perp:    ``BTCUSDT.P`` (.P suffix marks perpetual)
      * Crypto spot:    ``BTCUSDT``
      * OKX direct:     ``BTC-USDT-SWAP`` (already correct form)

    We keep this in one function because (a) the rules are exchange-specific
    and (b) bugs here go undetected — a wrong instrument ID at TV side
    plus wrong translation here gives an ID that *exists somewhere* and
    you trade the wrong thing.

    Examples
    --------
    >>> tv_symbol_to_okx("BTCUSDT.P")
    'BTC-USDT-SWAP'
    >>> tv_symbol_to_okx("ETHUSDT.P")
    'ETH-USDT-SWAP'
    >>> tv_symbol_to_okx("BTC-USDT-SWAP")
    'BTC-USDT-SWAP'
    >>> tv_symbol_to_okx("SOLUSDT")
    'SOL-USDT-SWAP'
    """
    sym = tv_symbol.upper().strip()

    # Already in OKX format
    if sym.endswith("-SWAP"):
        return sym

    # Strip TradingView suffixes
    suffixes = (".P", ".PERP")
    for suffix in suffixes:
        if sym.endswith(suffix):
            sym = sym[: -len(suffix)]
            break

    # USDT-margined assumed (we don't trade Coin-M)
    # Split BASEUSDT into BASE-USDT
    if sym.endswith("USDT"):
        base = sym[:-4]
        return f"{base}-USDT-SWAP"
    if sym.endswith("USDC"):
        base = sym[:-4]
        return f"{base}-USDC-SWAP"

    # Fallback — assume the user typed the OKX form already
    return sym


def alert_side_to_okx_side(alert_side: AlertSide) -> Side:
    """Map TV ``long``/``short`` → OKX ``buy``/``sell``."""
    return Side.BUY if alert_side is AlertSide.LONG else Side.SELL


def exit_side_to_okx_side(alert_side: AlertSide) -> Side:
    """For an EXIT signal, the OKX order side is the *opposite* of the
    position side.  E.g. exiting a long requires a SELL reduce-only.
    """
    return Side.SELL if alert_side is AlertSide.LONG else Side.BUY


# ---------------------------------------------------------------------------
# Handler result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandlerResult:
    """What happened when we processed a signal — used by the server to
    construct the HTTP response."""

    success: bool
    order: Order | None
    message: str
    rejection_reason: str | None = None


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class SignalHandler:
    """Translate validated TV alerts into OKX orders.

    Wraps an ``Exchange`` instance.  Stateless — every call is independent.
    Risk checks are NOT done here; they're injected as a callback so the
    risk engine remains the single source of truth for "should this trade
    happen?"
    """

    def __init__(
        self,
        exchange: Exchange,
        risk_check_fn=None,
    ) -> None:
        """
        Parameters
        ----------
        exchange : Exchange
            Implementation to send orders to (real OKX or mock for tests).
        risk_check_fn : callable, optional
            ``f(alert: TradingViewAlert, exchange: Exchange) -> str | None``.
            Returns ``None`` to allow the trade, or a string reason to reject.
            If omitted, no risk check is performed (test-only mode).
        """
        self._exchange = exchange
        self._risk_check = risk_check_fn

    def handle(self, alert: TradingViewAlert) -> HandlerResult:
        """Main entry point.  Idempotency / dedup is the caller's job."""
        # 1. Risk gate
        if self._risk_check is not None:
            reason = self._risk_check(alert, self._exchange)
            if reason is not None:
                logger.warning(
                    "Signal rejected by risk engine",
                    action=alert.action.value,
                    symbol=alert.symbol,
                    reason=reason,
                )
                return HandlerResult(
                    success=False, order=None,
                    message=f"Risk rejected: {reason}",
                    rejection_reason=reason,
                )

        # 2. Dispatch by action
        try:
            if alert.is_entry_or_pyramid():
                return self._handle_entry(alert)
            return self._handle_exit(alert)
        except InsufficientFundsError as exc:
            logger.error("Insufficient funds for signal",
                         symbol=alert.symbol, error=str(exc))
            return HandlerResult(
                success=False, order=None,
                message=f"Insufficient funds: {exc}",
                rejection_reason="insufficient_funds",
            )
        except OrderRejectedError as exc:
            logger.error("Order rejected by exchange",
                         symbol=alert.symbol, error=str(exc))
            return HandlerResult(
                success=False, order=None,
                message=f"Exchange rejected: {exc}",
                rejection_reason="exchange_rejected",
            )

    # ------------------------------------------------------------------

    def _handle_entry(self, alert: TradingViewAlert) -> HandlerResult:
        """Entry or pyramid: open or add to a position."""
        if alert.size is None:
            return HandlerResult(
                success=False, order=None,
                message="Entry/pyramid alert missing 'size' field",
                rejection_reason="missing_size",
            )

        okx_symbol = tv_symbol_to_okx(alert.symbol)
        side = alert_side_to_okx_side(alert.side)

        # SL only on first unit (subsequent pyramid units inherit the
        # original stop, since OKX moves the stop trigger to position-avg).
        sl = alert.sl if alert.action is AlertAction.ENTRY else None

        req = OrderRequest(
            symbol=okx_symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=alert.size,
            stop_loss=sl,
            reduce_only=False,
        )

        order = self._exchange.place_order(req)
        logger.info(
            "Entry order placed",
            action=alert.action.value,
            symbol=okx_symbol,
            side=side.value,
            unit=alert.unit,
            order_id=order.order_id,
        )
        return HandlerResult(
            success=True, order=order,
            message=f"{alert.action.value} #{alert.unit} on {okx_symbol}",
        )

    def _handle_exit(self, alert: TradingViewAlert) -> HandlerResult:
        """Exit: close all units of the position (reduce_only)."""
        okx_symbol = tv_symbol_to_okx(alert.symbol)

        # Read the current position size from the exchange — Pine doesn't
        # tell us how many contracts to close, just "close all".
        position = self._exchange.get_position(okx_symbol)
        if position is None or position.is_flat:
            logger.warning(
                "Exit signal but no open position — skipping",
                symbol=okx_symbol,
            )
            return HandlerResult(
                success=False, order=None,
                message="No open position to exit",
                rejection_reason="flat",
            )

        # The exit side is the OPPOSITE of the position side
        exit_side = position.side.opposite

        req = OrderRequest(
            symbol=okx_symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            reduce_only=True,  # Critical: never flip direction
        )

        order = self._exchange.place_order(req)
        logger.info(
            "Exit order placed",
            symbol=okx_symbol,
            side=exit_side.value,
            quantity=position.quantity,
            reason=alert.reason.value if alert.reason else "unknown",
            order_id=order.order_id,
        )
        return HandlerResult(
            success=True, order=order,
            message=f"exit on {okx_symbol} (reason={alert.reason.value if alert.reason else 'n/a'})",
        )
