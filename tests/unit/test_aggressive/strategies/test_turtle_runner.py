"""Tests for the TurtleRunner — main strategy loop wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from apexfx.aggressive.alerts.telegram import NullNotifier
from apexfx.aggressive.exchanges.base import (
    Balance,
    Bar,
    ExchangeError,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    SymbolInfo,
)
from apexfx.aggressive.risk.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig,
)
from apexfx.aggressive.risk.kill_switch import KillSwitch
from apexfx.aggressive.strategies.donchian_turtle import (
    DecisionAction,
    DonchianTurtle,
    StrategyDecision,
    TurtleConfig,
)
from apexfx.aggressive.strategies.turtle_runner import TurtleRunner

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bars(n: int = 250, base_price: float = 1.10) -> list[Bar]:
    base_time = datetime(2026, 5, 1, tzinfo=UTC)
    return [
        Bar(timestamp=base_time + timedelta(hours=4 * i),
            open=base_price, high=base_price * 1.001,
            low=base_price * 0.999, close=base_price, volume=100.0)
        for i in range(n)
    ]


def _balance(equity: float = 1000.0) -> Balance:
    return Balance(
        asset="USD", equity=equity, available=equity * 0.9,
        timestamp=datetime.now(tz=UTC),
    )


def _symbol_info(symbol: str = "EURUSD") -> SymbolInfo:
    return SymbolInfo(
        symbol=symbol, base_currency="EUR", quote_currency="USD",
        contract_size=100_000.0, tick_size=0.00001,
        lot_size=0.01, min_quantity=0.01, max_leverage=100.0,
    )


def _order(symbol: str = "EURUSD", side: Side = Side.BUY,
           qty: float = 0.05, fill_price: float = 1.10) -> Order:
    return Order(
        order_id="999", client_order_id=None, symbol=symbol, side=side,
        order_type=OrderType.MARKET, status=OrderStatus.FILLED,
        quantity=qty, filled_quantity=qty, avg_fill_price=fill_price,
        price=None, timestamp=datetime.now(tz=UTC),
    )


def _mock_exchange(
    bars: list[Bar] | None = None,
    position: Position | None = None,
    equity: float = 1000.0,
) -> MagicMock:
    ex = MagicMock()
    ex.get_bars.return_value = bars if bars is not None else _bars()
    ex.get_position.return_value = position
    ex.get_balance.return_value = _balance(equity)
    ex.get_symbol_info.return_value = _symbol_info()
    ex.place_order.return_value = _order()
    return ex


# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_symbols_rejected(self) -> None:
        with pytest.raises(ValueError):
            TurtleRunner(exchange=MagicMock(), symbols=[])

    def test_initial_state_clean(self) -> None:
        runner = TurtleRunner(
            exchange=_mock_exchange(), symbols=["EURUSD"],
        )
        st = runner.state["EURUSD"]
        assert st.last_bar_time is None
        assert st.n_units_open == 0
        assert st.direction is None


# ---------------------------------------------------------------------------


class TestRunOnceHold:
    def test_no_signal_no_order(self) -> None:
        ex = _mock_exchange()
        runner = TurtleRunner(exchange=ex, symbols=["EURUSD"])
        runner.run_once()
        # No order placed — flat bars give HOLD
        assert not ex.place_order.called
        assert runner.stats.decisions_hold > 0

    def test_kill_switch_active_skips_everything(self, tmp_path) -> None:
        ex = _mock_exchange()
        kill = KillSwitch(
            flag_path=tmp_path / "kill", cooldown_path=tmp_path / "cool",
        )
        kill.trigger("test")
        runner = TurtleRunner(
            exchange=ex, symbols=["EURUSD"], kill_switch=kill,
        )
        runner.run_once()
        assert not ex.get_bars.called
        assert not ex.place_order.called


# ---------------------------------------------------------------------------


class TestRunOnceTrade:
    def test_entry_signal_places_order_and_updates_state(self) -> None:
        ex = _mock_exchange()

        # Patch strategy to deterministically return ENTER_LONG
        fake_strategy = MagicMock(spec=DonchianTurtle)
        fake_strategy.min_bars = 50
        fake_strategy.decide.return_value = StrategyDecision(
            action=DecisionAction.ENTER_LONG, side=Side.BUY,
            target_volume=0.05, stop_loss=1.095,
            reason="long breakout",
        )
        # The runner re-derives equity; ensure strategy receives correct args
        runner = TurtleRunner(
            exchange=ex, symbols=["EURUSD"], strategy=fake_strategy,
        )
        runner.run_once()
        assert ex.place_order.called
        st = runner.state["EURUSD"]
        assert st.direction is Side.BUY
        assert st.n_units_open == 1
        assert st.last_unit_price == 1.10  # from fake order fill_price

    def test_exit_signal_resets_state(self) -> None:
        pos = Position(
            symbol="EURUSD", side=Side.BUY, quantity=0.10,
            entry_price=1.10, leverage=0.0, unrealized_pnl=0.0,
            timestamp=datetime.now(tz=UTC),
        )
        ex = _mock_exchange(position=pos)
        ex.place_order.return_value = _order(side=Side.SELL)

        fake_strategy = MagicMock(spec=DonchianTurtle)
        fake_strategy.min_bars = 50
        fake_strategy.decide.return_value = StrategyDecision(
            action=DecisionAction.EXIT, side=Side.BUY,
            target_volume=0.10, reason="donchian_exit",
        )
        runner = TurtleRunner(
            exchange=ex, symbols=["EURUSD"], strategy=fake_strategy,
        )
        # Pre-set state to simulate prior entry
        runner.state["EURUSD"].direction = Side.BUY
        runner.state["EURUSD"].n_units_open = 2
        runner.state["EURUSD"].last_unit_price = 1.10

        runner.run_once()
        # State should reset after exit
        assert runner.state["EURUSD"].direction is None
        assert runner.state["EURUSD"].n_units_open == 0

    def test_pyramid_increments_unit_count(self) -> None:
        pos = Position(
            symbol="EURUSD", side=Side.BUY, quantity=0.05,
            entry_price=1.10, leverage=0.0, unrealized_pnl=2.0,
            timestamp=datetime.now(tz=UTC),
        )
        ex = _mock_exchange(position=pos)
        ex.place_order.return_value = _order(fill_price=1.11)

        fake_strategy = MagicMock(spec=DonchianTurtle)
        fake_strategy.min_bars = 50
        fake_strategy.decide.return_value = StrategyDecision(
            action=DecisionAction.PYRAMID, side=Side.BUY,
            target_volume=0.05, reason="pyramid trigger",
        )
        runner = TurtleRunner(
            exchange=ex, symbols=["EURUSD"], strategy=fake_strategy,
        )
        runner.state["EURUSD"].direction = Side.BUY
        runner.state["EURUSD"].n_units_open = 1
        runner.state["EURUSD"].last_unit_price = 1.10

        runner.run_once()
        assert runner.state["EURUSD"].n_units_open == 2
        assert runner.state["EURUSD"].last_unit_price == 1.11


# ---------------------------------------------------------------------------


class TestBarCloseDedup:
    def test_same_bar_only_processed_once(self) -> None:
        ex = _mock_exchange()
        fake_strategy = MagicMock(spec=DonchianTurtle)
        fake_strategy.min_bars = 50
        fake_strategy.decide.return_value = StrategyDecision(
            action=DecisionAction.HOLD,
        )
        runner = TurtleRunner(
            exchange=ex, symbols=["EURUSD"], strategy=fake_strategy,
        )
        runner.run_once()
        first_decisions = fake_strategy.decide.call_count
        # Run again with SAME bars — should skip strategy call
        runner.run_once()
        assert fake_strategy.decide.call_count == first_decisions

    def test_new_bar_triggers_new_decision(self) -> None:
        bars = _bars()
        ex = _mock_exchange(bars=bars)
        fake_strategy = MagicMock(spec=DonchianTurtle)
        fake_strategy.min_bars = 50
        fake_strategy.decide.return_value = StrategyDecision(
            action=DecisionAction.HOLD,
        )
        runner = TurtleRunner(
            exchange=ex, symbols=["EURUSD"], strategy=fake_strategy,
        )
        runner.run_once()
        # Append a new bar to simulate time passing
        bars.append(Bar(
            timestamp=bars[-1].timestamp + timedelta(hours=4),
            open=1.10, high=1.101, low=1.099, close=1.10, volume=100.0,
        ))
        ex.get_bars.return_value = bars
        runner.run_once()
        # Strategy called a second time
        assert fake_strategy.decide.call_count == 2


# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_exchange_error_does_not_crash_loop(self) -> None:
        ex = _mock_exchange()
        ex.get_bars.side_effect = ExchangeError("network blip")
        runner = TurtleRunner(exchange=ex, symbols=["EURUSD"])
        # Should NOT raise — error logged, stats incremented, continue
        runner.run_once()
        assert runner.stats.orders_failed > 0

    def test_order_rejection_notifies_breaker(self, tmp_path) -> None:
        ex = _mock_exchange()
        ex.place_order.side_effect = ExchangeError("rejected")

        kill = KillSwitch(
            flag_path=tmp_path / "kill", cooldown_path=tmp_path / "cool",
        )
        breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                daily_loss_pct=0.5, weekly_loss_pct=0.5,
                monthly_dd_pct=0.5, max_consecutive_failed_orders=2,
            ),
            kill_switch=kill,
            state_path=tmp_path / "breaker.json",
        )

        fake_strategy = MagicMock(spec=DonchianTurtle)
        fake_strategy.min_bars = 50
        fake_strategy.decide.return_value = StrategyDecision(
            action=DecisionAction.ENTER_LONG, side=Side.BUY,
            target_volume=0.05, stop_loss=1.095,
        )
        runner = TurtleRunner(
            exchange=ex, symbols=["EURUSD"], strategy=fake_strategy,
            kill_switch=kill, breaker=breaker,
        )
        # First failure
        runner.run_once()
        assert breaker.state.consecutive_failed_orders == 1
        # Second failure — should trip
        bars = ex.get_bars.return_value.copy()
        bars.append(Bar(
            timestamp=bars[-1].timestamp + timedelta(hours=4),
            open=1.10, high=1.101, low=1.099, close=1.10, volume=100.0,
        ))
        ex.get_bars.return_value = bars
        runner.run_once()
        assert kill.is_active()

    def test_balance_failure_continues_with_zero_equity(self) -> None:
        ex = _mock_exchange()
        ex.get_balance.side_effect = ExchangeError("transient")
        runner = TurtleRunner(exchange=ex, symbols=["EURUSD"])
        # Should NOT crash; just runs with equity=0 → no orders placed
        runner.run_once()
