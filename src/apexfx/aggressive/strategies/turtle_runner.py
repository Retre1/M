"""Main strategy loop — connects MT5 → DonchianTurtle → risk engine → alerts.

Lifecycle
---------
1. ``run_once()``: do a single pass — check kill switch, update equity, for
   each symbol fetch bars, run strategy, place orders.  Returns immediately.
2. ``run_forever(poll_interval_s)``: loop ``run_once`` forever until the kill
   switch fires or an unrecoverable exception.

Bar-close detection
-------------------
We avoid acting mid-bar (the "calc_on_every_tick" pitfall from Pine).
After each fetch we remember the last bar's timestamp per symbol and only
act when the *latest* bar's timestamp differs from the previous run.  This
means actions happen exactly once per bar close, regardless of how often
the poll loop wakes up.

State tracking
--------------
For pyramid logic the strategy needs to know:
  * how many units we've opened in the current direction
  * the entry price of the most recent unit

Both are tracked in memory by this runner.  On restart we lose them — the
strategy will treat the current position as "1 unit at avg_price" which
underestimates pyramid count but doesn't break trading (worst case: we
miss one pyramid opportunity right after restart).

For longer-term persistence we'd checkpoint to JSON, but for retail this
is over-engineering — the loss of one pyramid trigger per restart is
trivial compared to the simplicity gain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from apexfx.aggressive.alerts.telegram import NullNotifier, TelegramNotifier
from apexfx.aggressive.exchanges.base import (
    Exchange,
    ExchangeError,
    OrderRequest,
    OrderType,
    Position,
    Side,
)
from apexfx.aggressive.risk.circuit_breaker import CircuitBreaker
from apexfx.aggressive.risk.kill_switch import KillSwitch
from apexfx.aggressive.strategies.donchian_turtle import (
    DecisionAction,
    DonchianTurtle,
    StrategyDecision,
    TurtleConfig,
)
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-symbol pyramid state
# ---------------------------------------------------------------------------


@dataclass
class _SymbolState:
    """Per-symbol bookkeeping for pyramid + bar-close dedup."""

    last_bar_time: datetime | None = None
    n_units_open: int = 0
    last_unit_price: float | None = None
    direction: Side | None = None

    def reset(self) -> None:
        self.n_units_open = 0
        self.last_unit_price = None
        self.direction = None


@dataclass
class RunnerStats:
    """Counters for visibility / health checks."""

    bars_processed: int = 0
    decisions_total: int = 0
    decisions_hold: int = 0
    orders_placed: int = 0
    orders_failed: int = 0
    risk_rejections: int = 0
    cycles: int = 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TurtleRunner:
    """Orchestrator: poll MT5 for each symbol, decide, execute.

    Parameters
    ----------
    exchange : Exchange
        Any implementation of the protocol — MT5 in production, mock in tests.
    symbols : list[str]
        Universe to trade.  Must match broker's exact symbol naming.
    timeframe : str
        Bar interval string — must be a key in ``_TIMEFRAME_CODES`` of the
        exchange.  Default ``"H4"`` matches our Pine Script default.
    strategy : DonchianTurtle | None
        Strategy instance.  If None, uses default ``TurtleConfig``.
    kill_switch, breaker : risk components.  Optional — if omitted, runner
        runs without those safety layers (test mode).
    notifier : TelegramNotifier | NullNotifier
        Where to send event alerts.
    """

    def __init__(
        self,
        exchange: Exchange,
        symbols: list[str],
        *,
        timeframe: str = "H4",
        strategy: DonchianTurtle | None = None,
        kill_switch: KillSwitch | None = None,
        breaker: CircuitBreaker | None = None,
        notifier: TelegramNotifier | NullNotifier | None = None,
        deposit_currency: str = "USD",
    ) -> None:
        if not symbols:
            raise ValueError("must specify at least one symbol")
        self._exchange = exchange
        self._symbols = symbols
        self._timeframe = timeframe
        self._strategy = strategy or DonchianTurtle()
        self._kill = kill_switch
        self._breaker = breaker
        self._notifier = notifier or NullNotifier()
        self._deposit_currency = deposit_currency
        self._state: dict[str, _SymbolState] = {
            sym: _SymbolState() for sym in symbols
        }
        self._stats = RunnerStats()

    @property
    def stats(self) -> RunnerStats:
        return self._stats

    @property
    def state(self) -> dict[str, _SymbolState]:
        return self._state

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_forever(self, poll_interval_s: float = 60.0) -> None:
        """Run until the kill switch fires or KeyboardInterrupt.

        ``poll_interval_s`` should be << the bar interval — 60s for H4
        gives 240 wake-ups per bar, plenty to catch the close.  Drop to
        10s for M5/M15 strategies.
        """
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        logger.info("TurtleRunner starting",
                    symbols=self._symbols, timeframe=self._timeframe,
                    poll_interval_s=poll_interval_s)
        try:
            while True:
                try:
                    self.run_once()
                except Exception as exc:
                    logger.exception("Cycle failed — continuing", error=str(exc))
                    self._stats.orders_failed += 1
                    # Notify but don't crash — transient errors are normal
                    self._notifier.notify_health_failure(
                        component="turtle_runner", error=str(exc),
                    )
                if self._kill is not None and self._kill.is_active():
                    logger.warning("Kill switch active — stopping run loop",
                                   state=self._kill.state())
                    self._notifier.notify_kill_switch(self._kill.state().reason)
                    return
                time.sleep(poll_interval_s)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — exiting cleanly")

    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """One scan over all symbols.  Idempotent — same bar won't trigger twice."""
        self._stats.cycles += 1

        # Pre-flight: kill switch trumps everything
        if self._kill is not None and self._kill.is_active():
            return

        # Update equity for circuit breaker
        equity = self._equity_or_zero()
        if self._breaker is not None and equity > 0:
            trip = self._breaker.observe_equity(equity)
            if trip.tripped:
                # Kill switch already fired by the breaker
                self._notifier.notify_kill_switch(trip.reason)
                return

        # Process each symbol
        for symbol in self._symbols:
            try:
                self._process_symbol(symbol, equity)
            except ExchangeError as exc:
                # Per-symbol failure: log, notify breaker, continue with rest
                logger.error("Symbol cycle failed",
                             symbol=symbol, error=str(exc))
                self._stats.orders_failed += 1
                if self._breaker is not None:
                    self._breaker.notify_order_failure()

    # ------------------------------------------------------------------

    def _process_symbol(self, symbol: str, equity: float) -> None:
        """Fetch bars for one symbol, run strategy, execute if needed."""
        # Need enough bars for indicators + headroom
        bars = self._exchange.get_bars(
            symbol, self._timeframe, limit=max(self._strategy.min_bars + 10, 300),
        )
        if not bars:
            return

        latest_bar_time = bars[-1].timestamp
        state = self._state[symbol]

        # Bar-close dedup
        if state.last_bar_time is not None and state.last_bar_time >= latest_bar_time:
            return  # Same bar as last cycle — no action
        state.last_bar_time = latest_bar_time
        self._stats.bars_processed += 1

        # Read current position from exchange — source of truth for
        # direction + total volume.  Local state tracks pyramid count.
        position = self._exchange.get_position(symbol)
        self._sync_local_state_with_exchange(state, position)

        # Determine contract_size for the symbol (mostly relevant for forex)
        symbol_info = self._exchange.get_symbol_info(symbol)
        contract_size = symbol_info.contract_size

        # Strategy decision
        decision: StrategyDecision = self._strategy.decide(
            bars=bars, position=position, equity=equity,
            n_units_open=state.n_units_open,
            last_unit_price=state.last_unit_price,
            contract_size=contract_size,
        )
        self._stats.decisions_total += 1

        if not decision.is_trade:
            self._stats.decisions_hold += 1
            return

        # Execute
        self._execute_decision(symbol, decision, state)

    def _sync_local_state_with_exchange(
        self, state: _SymbolState, position: Position | None,
    ) -> None:
        """Reconcile pyramid count with what the exchange says.

        If we restarted and exchange shows a position but our local state
        is empty, assume 1 unit (best-effort).  If exchange is flat but
        we thought we had units, reset state.
        """
        if position is None or position.is_flat:
            if state.n_units_open > 0:
                logger.info("Position closed externally — resetting state",
                            symbol=position.symbol if position else "?")
                state.reset()
            return

        if state.direction is None:
            # First time seeing this position (probably post-restart)
            state.direction = position.side
            state.n_units_open = 1
            state.last_unit_price = position.entry_price
            logger.info("Synced local state from exchange position",
                        symbol=position.symbol, side=position.side.value,
                        entry_price=position.entry_price)

    # ------------------------------------------------------------------

    def _execute_decision(
        self, symbol: str, decision: StrategyDecision, state: _SymbolState,
    ) -> None:
        """Translate a StrategyDecision into an OrderRequest and submit."""
        if decision.action is DecisionAction.EXIT:
            req = OrderRequest(
                symbol=symbol,
                side=decision.side.opposite,  # type: ignore[union-attr]
                order_type=OrderType.MARKET,
                quantity=decision.target_volume,
                reduce_only=True,
            )
            order = self._safe_place(req)
            if order is None:
                return
            self._notifier.notify_exit(
                symbol=symbol,
                side=decision.side.value if decision.side else "?",
                price=order.avg_fill_price or 0.0,
                reason=decision.reason,
            )
            state.reset()
            self._stats.orders_placed += 1
            if self._breaker is not None:
                self._breaker.notify_order_success()
            return

        # Entry or pyramid
        if decision.side is None:
            return

        req = OrderRequest(
            symbol=symbol, side=decision.side,
            order_type=OrderType.MARKET,
            quantity=decision.target_volume,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            reduce_only=False,
        )
        order = self._safe_place(req)
        if order is None:
            return

        self._stats.orders_placed += 1
        if self._breaker is not None:
            self._breaker.notify_order_success()
        fill_price = order.avg_fill_price or 0.0

        if decision.action in (DecisionAction.ENTER_LONG, DecisionAction.ENTER_SHORT):
            state.direction = decision.side
            state.n_units_open = 1
            state.last_unit_price = fill_price
            self._notifier.notify_entry(
                symbol=symbol, side=decision.side.value, unit=1,
                size=decision.target_volume, price=fill_price,
                sl=decision.stop_loss,
            )
        elif decision.action is DecisionAction.PYRAMID:
            state.n_units_open += 1
            state.last_unit_price = fill_price
            self._notifier.notify_pyramid(
                symbol=symbol, side=decision.side.value,
                unit=state.n_units_open,
                size=decision.target_volume, price=fill_price,
            )

    def _safe_place(self, req: OrderRequest) -> object | None:
        """Place an order with structured error handling.

        Notifies on rejection, increments stats, updates breaker.  Returns
        the order on success, None on failure.
        """
        try:
            return self._exchange.place_order(req)  # type: ignore[return-value]
        except ExchangeError as exc:
            logger.warning("Order rejected", symbol=req.symbol,
                           side=req.side.value, error=str(exc))
            self._stats.orders_failed += 1
            self._notifier.notify_order_rejected(
                symbol=req.symbol, side=req.side.value, reason=str(exc),
            )
            if self._breaker is not None:
                trip = self._breaker.notify_order_failure()
                if trip.tripped:
                    self._notifier.notify_kill_switch(trip.reason)
            return None

    # ------------------------------------------------------------------

    def _equity_or_zero(self) -> float:
        """Read equity from exchange, return 0 on transient failure.

        We don't crash the loop on a balance-fetch hiccup — log it and
        skip risk checks for this cycle.  Persistent failures will trip
        the breaker's consecutive-failed-orders counter via order
        failures anyway.
        """
        try:
            return self._exchange.get_balance(self._deposit_currency).equity
        except ExchangeError as exc:
            logger.warning("Equity fetch failed — skipping risk check",
                           error=str(exc))
            return 0.0
