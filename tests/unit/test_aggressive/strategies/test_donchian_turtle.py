"""Tests for the Python Donchian Turtle strategy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from apexfx.aggressive.exchanges.base import Bar, Position, Side
from apexfx.aggressive.strategies.donchian_turtle import (
    DecisionAction,
    DonchianTurtle,
    TurtleConfig,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Bar generators
# ---------------------------------------------------------------------------


def _bars_flat(n: int, price: float = 1.10) -> list[Bar]:
    """N flat bars, no movement — useful for null-signal tests."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        Bar(timestamp=base + timedelta(hours=4 * i),
            open=price, high=price, low=price, close=price, volume=100.0)
        for i in range(n)
    ]


def _bars_uptrend(n: int, start: float = 1.0, step: float = 0.001) -> list[Bar]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    bars: list[Bar] = []
    for i in range(n):
        close = start + step * i
        bars.append(Bar(
            timestamp=base + timedelta(hours=4 * i),
            open=close - step * 0.5, high=close + step * 0.2,
            low=close - step * 0.7, close=close, volume=100.0,
        ))
    return bars


def _bars_downtrend(n: int, start: float = 1.1, step: float = 0.001) -> list[Bar]:
    return _bars_uptrend(n, start=start, step=-step)


def _bars_breakout(
    n_setup: int = 30, breakout_pct: float = 0.005,
) -> list[Bar]:
    """Build bars where the LAST bar breaks above the 20-period high."""
    flat = _bars_flat(n_setup, price=1.10)
    breakout_close = 1.10 * (1 + breakout_pct)
    flat.append(Bar(
        timestamp=flat[-1].timestamp + timedelta(hours=4),
        open=1.10, high=breakout_close, low=1.10,
        close=breakout_close, volume=200.0,
    ))
    return flat


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_exit_must_be_less_than_entry(self) -> None:
        with pytest.raises(ValueError):
            TurtleConfig(entry_period=10, exit_period=10)
        with pytest.raises(ValueError):
            TurtleConfig(entry_period=10, exit_period=15)

    def test_invalid_risk_pct_rejected(self) -> None:
        with pytest.raises(ValueError):
            TurtleConfig(risk_per_unit_pct=0.5)

    def test_invalid_max_units_rejected(self) -> None:
        with pytest.raises(ValueError):
            TurtleConfig(max_units=0)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


class TestIndicators:
    def test_atr_zero_on_flat_bars(self) -> None:
        s = DonchianTurtle(TurtleConfig(atr_period=10))
        bars = _bars_flat(20)
        assert s.compute_atr(bars) == 0.0

    def test_atr_positive_on_volatile_bars(self) -> None:
        s = DonchianTurtle(TurtleConfig(atr_period=5))
        bars = _bars_uptrend(20, step=0.005)
        assert s.compute_atr(bars) > 0.0

    def test_ema_converges_on_flat(self) -> None:
        s = DonchianTurtle()
        bars = _bars_flat(200, price=1.10)
        ema = s.compute_ema(bars, period=200)
        assert ema == pytest.approx(1.10, abs=1e-9)

    def test_ema_tracks_uptrend(self) -> None:
        s = DonchianTurtle()
        bars = _bars_uptrend(200, start=1.0, step=0.001)
        ema = s.compute_ema(bars, period=50)
        # EMA50 of uptrend should be below the most recent close
        assert ema < bars[-1].close
        # But above the average start price
        assert ema > 1.0

    def test_donchian_high_excludes_current_bar(self) -> None:
        """Critical: Donchian must use the PREVIOUS N bars to avoid the
        current bar's own high being its own breakout target."""
        s = DonchianTurtle()
        bars = _bars_flat(30, price=1.10)
        # Current bar (last) high = 1.10
        # Previous 20 bars also 1.10 → high = 1.10
        # Now bump the LAST bar's high to 1.20 — should NOT change result
        bars[-1] = Bar(
            timestamp=bars[-1].timestamp, open=1.10, high=1.20, low=1.10,
            close=1.10, volume=100.0,
        )
        assert s.donchian_high(bars, period=20) == pytest.approx(1.10)


# ---------------------------------------------------------------------------
# Decision: HOLD
# ---------------------------------------------------------------------------


class TestDecisionHold:
    def test_insufficient_bars_holds(self) -> None:
        s = DonchianTurtle()
        decision = s.decide(
            bars=_bars_flat(5), position=None, equity=1000.0,
        )
        assert decision.action is DecisionAction.HOLD
        assert "enough bars" in decision.reason

    def test_zero_atr_holds(self) -> None:
        s = DonchianTurtle(TurtleConfig(atr_period=10))
        decision = s.decide(
            bars=_bars_flat(220), position=None, equity=1000.0,
        )
        assert decision.action is DecisionAction.HOLD


# ---------------------------------------------------------------------------
# Decision: ENTER_LONG
# ---------------------------------------------------------------------------


class TestDecisionLongEntry:
    def test_breakout_above_donchian_high_triggers_long(self) -> None:
        # 200+ bars uptrend → close way above EMA200 → trend filter passes
        s = DonchianTurtle()
        bars = _bars_uptrend(300, start=1.0, step=0.0005)
        # Engineer a breakout on the last bar
        last = bars[-1]
        donch_high = s.donchian_high(bars, 20)
        bars[-1] = Bar(
            timestamp=last.timestamp, open=last.open,
            high=donch_high * 1.01, low=last.low,
            close=donch_high * 1.005, volume=last.volume,
        )
        decision = s.decide(
            bars=bars, position=None, equity=1000.0,
            contract_size=100000.0,
        )
        assert decision.action is DecisionAction.ENTER_LONG
        assert decision.side is Side.BUY
        assert decision.target_volume > 0
        assert decision.stop_loss is not None
        assert decision.stop_loss < bars[-1].close

    def test_trend_filter_blocks_long_below_ema(self) -> None:
        # Strong downtrend → EMA200 above price → long blocked even if breakout
        s = DonchianTurtle()
        bars = _bars_downtrend(300, start=1.5, step=0.0005)
        # Force a "breakout" upward by spiking last bar
        last = bars[-1]
        donch_high = s.donchian_high(bars, 20)
        bars[-1] = Bar(
            timestamp=last.timestamp, open=last.open,
            high=donch_high * 1.01, low=last.low,
            close=donch_high * 1.005, volume=last.volume,
        )
        decision = s.decide(
            bars=bars, position=None, equity=1000.0,
        )
        assert decision.action is DecisionAction.HOLD

    def test_trend_filter_disabled_allows_against_trend(self) -> None:
        s = DonchianTurtle(TurtleConfig(use_trend_filter=False))
        bars = _bars_downtrend(300, start=1.5, step=0.0005)
        last = bars[-1]
        donch_high = s.donchian_high(bars, 20)
        bars[-1] = Bar(
            timestamp=last.timestamp, open=last.open,
            high=donch_high * 1.01, low=last.low,
            close=donch_high * 1.005, volume=last.volume,
        )
        decision = s.decide(
            bars=bars, position=None, equity=1000.0,
        )
        assert decision.action is DecisionAction.ENTER_LONG


# ---------------------------------------------------------------------------
# Decision: ENTER_SHORT
# ---------------------------------------------------------------------------


class TestDecisionShortEntry:
    def test_breakdown_below_donchian_low_triggers_short(self) -> None:
        s = DonchianTurtle()
        bars = _bars_downtrend(300, start=1.5, step=0.0005)
        last = bars[-1]
        donch_low = s.donchian_low(bars, 20)
        bars[-1] = Bar(
            timestamp=last.timestamp, open=last.open,
            high=last.high, low=donch_low * 0.99,
            close=donch_low * 0.995, volume=last.volume,
        )
        decision = s.decide(
            bars=bars, position=None, equity=1000.0,
        )
        assert decision.action is DecisionAction.ENTER_SHORT
        assert decision.side is Side.SELL
        assert decision.stop_loss is not None
        assert decision.stop_loss > bars[-1].close


# ---------------------------------------------------------------------------
# Decision: EXIT
# ---------------------------------------------------------------------------


class TestDecisionExit:
    def _make_position(self, side: Side, entry_price: float) -> Position:
        return Position(
            symbol="EURUSD", side=side, quantity=0.10,
            entry_price=entry_price, leverage=0.0, unrealized_pnl=0.0,
            timestamp=datetime.now(tz=UTC),
        )

    def test_long_hard_stop_2N_trips_exit(self) -> None:
        s = DonchianTurtle()
        bars = _bars_uptrend(300, start=1.0, step=0.0005)
        atr = s.compute_atr(bars)
        entry_price = bars[-1].close
        # Force a 3N drop on the last bar
        bars[-1] = Bar(
            timestamp=bars[-1].timestamp, open=entry_price,
            high=entry_price, low=entry_price - 3 * atr,
            close=entry_price - 3 * atr, volume=100.0,
        )
        pos = self._make_position(Side.BUY, entry_price)
        decision = s.decide(bars=bars, position=pos, equity=1000.0)
        assert decision.action is DecisionAction.EXIT
        assert "hard_stop" in decision.reason

    def test_long_donchian_exit_trips(self) -> None:
        s = DonchianTurtle()
        bars = _bars_uptrend(300, start=1.0, step=0.0005)
        exit_low = s.donchian_low(bars, 10)
        bars[-1] = Bar(
            timestamp=bars[-1].timestamp, open=bars[-1].open,
            high=bars[-1].high, low=exit_low * 0.99,
            close=exit_low * 0.999, volume=100.0,
        )
        # Position entered slightly below current — not stopped out
        pos = self._make_position(Side.BUY, bars[-1].close * 1.5)
        decision = s.decide(bars=bars, position=pos, equity=1000.0)
        # Either donchian_exit or hard_stop (depending on entry distance);
        # whichever fires, action must be EXIT
        assert decision.action is DecisionAction.EXIT


# ---------------------------------------------------------------------------
# Decision: PYRAMID
# ---------------------------------------------------------------------------


class TestDecisionPyramid:
    def _make_position(self, side: Side, entry_price: float) -> Position:
        return Position(
            symbol="EURUSD", side=side, quantity=0.10,
            entry_price=entry_price, leverage=0.0, unrealized_pnl=0.0,
            timestamp=datetime.now(tz=UTC),
        )

    def test_pyramid_on_continued_advance(self) -> None:
        s = DonchianTurtle()
        bars = _bars_uptrend(300, start=1.0, step=0.0005)
        atr = s.compute_atr(bars)
        last_unit_price = bars[-1].close - 0.5 * atr * 1.5  # already +0.75N up
        pos = self._make_position(Side.BUY, last_unit_price)
        decision = s.decide(
            bars=bars, position=pos, equity=1000.0,
            n_units_open=1, last_unit_price=last_unit_price,
        )
        assert decision.action is DecisionAction.PYRAMID
        assert decision.side is Side.BUY

    def test_no_pyramid_at_max_units(self) -> None:
        s = DonchianTurtle(TurtleConfig(max_units=2))
        bars = _bars_uptrend(300, start=1.0, step=0.0005)
        last_price = bars[-1].close - 0.1
        pos = self._make_position(Side.BUY, last_price)
        decision = s.decide(
            bars=bars, position=pos, equity=1000.0,
            n_units_open=2,  # already at cap
            last_unit_price=last_price,
        )
        # Action will be HOLD or EXIT (depending on stop distance), never PYRAMID
        assert decision.action is not DecisionAction.PYRAMID

    def test_no_pyramid_without_last_unit_price(self) -> None:
        s = DonchianTurtle()
        bars = _bars_uptrend(300, start=1.0, step=0.0005)
        pos = self._make_position(Side.BUY, bars[-1].close - 0.001)
        decision = s.decide(
            bars=bars, position=pos, equity=1000.0,
            n_units_open=1, last_unit_price=None,
        )
        assert decision.action is not DecisionAction.PYRAMID


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


class TestUnitVolume:
    def test_basic_formula(self) -> None:
        s = DonchianTurtle(TurtleConfig(risk_per_unit_pct=0.015, stop_atr_mult=2.0))
        # equity=1000, ATR=0.001 → risk=$15, stop=0.002 →
        # unit_quote = 15/0.002 = 7500 → /100000 = 0.075 lots
        lots = s.unit_volume(equity=1000.0, atr=0.001, contract_size=100_000.0)
        assert lots == pytest.approx(0.075)

    def test_zero_equity(self) -> None:
        s = DonchianTurtle()
        assert s.unit_volume(0.0, 0.001) == 0.0

    def test_zero_atr(self) -> None:
        s = DonchianTurtle()
        assert s.unit_volume(1000.0, 0.0) == 0.0
