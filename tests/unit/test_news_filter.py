"""Tests for src/apexfx/risk/news_filter.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from apexfx.risk.news_filter import NewsEvent, NewsFilter


def _make_event(
    minutes_from_now: float = 0,
    currency: str = "USD",
    impact: str = "high",
    name: str = "NFP",
    base_time: datetime | None = None,
) -> NewsEvent:
    """Helper to create a NewsEvent relative to a base time."""
    base = base_time or datetime.now(UTC)
    return NewsEvent(
        time_utc=base + timedelta(minutes=minutes_from_now),
        currency=currency,
        impact=impact,
        name=name,
    )


class TestNoEvents:
    def test_can_trade_with_no_events(self):
        nf = NewsFilter()
        can_trade, scale, reason = nf.check(datetime.now(UTC))
        assert can_trade is True
        assert scale == 1.0
        assert reason == ""


class TestBlackoutWindow:
    def test_during_blackout_before_event(self):
        nf = NewsFilter(blackout_before_min=15, blackout_after_min=10)
        now = datetime.now(UTC)
        # Event in 10 minutes (within 15-min blackout_before)
        event = _make_event(minutes_from_now=10, base_time=now)
        nf.add_event(event)
        can_trade, scale, reason = nf.check(now)
        assert can_trade is False
        assert scale == 0.0
        assert "blackout" in reason.lower()

    def test_during_blackout_after_event(self):
        nf = NewsFilter(blackout_before_min=15, blackout_after_min=10)
        now = datetime.now(UTC)
        # Event was 5 minutes ago (within 10-min blackout_after)
        event = _make_event(minutes_from_now=-5, base_time=now)
        nf.add_event(event)
        can_trade, scale, reason = nf.check(now)
        assert can_trade is False
        assert scale == 0.0

    def test_after_blackout_after_can_trade(self):
        nf = NewsFilter(blackout_before_min=15, blackout_after_min=10)
        now = datetime.now(UTC)
        # Event was 15 minutes ago (past 10-min blackout_after)
        event = _make_event(minutes_from_now=-15, base_time=now)
        nf.add_event(event)
        can_trade, scale, reason = nf.check(now)
        assert can_trade is True
        assert scale == 1.0


class TestReduceBeforeWindow:
    def test_in_reduce_before_window(self):
        nf = NewsFilter(
            blackout_before_min=15,
            blackout_after_min=10,
            reduce_before_min=60,
            reduce_scale=0.5,
        )
        now = datetime.now(UTC)
        # Event in 30 minutes — past blackout_before (15) but within reduce_before (60)
        event = _make_event(minutes_from_now=30, base_time=now)
        nf.add_event(event)
        can_trade, scale, reason = nf.check(now)
        assert can_trade is True
        assert scale == 0.5
        assert "reduction" in reason.lower()


class TestCurrencyFiltering:
    def test_only_relevant_currencies_blocked(self):
        nf = NewsFilter(blackout_before_min=15)
        now = datetime.now(UTC)
        # USD event in 5 minutes
        event = _make_event(minutes_from_now=5, currency="USD", base_time=now)
        nf.add_event(event)

        # Trading EUR/GBP — USD event should NOT block
        can_trade, scale, _ = nf.check(now, symbol_currencies=["EUR", "GBP"])
        assert can_trade is True
        assert scale == 1.0

    def test_matching_currency_is_blocked(self):
        nf = NewsFilter(blackout_before_min=15)
        now = datetime.now(UTC)
        event = _make_event(minutes_from_now=5, currency="USD", base_time=now)
        nf.add_event(event)

        # Trading EUR/USD — USD event should block
        can_trade, scale, _ = nf.check(now, symbol_currencies=["EUR", "USD"])
        assert can_trade is False


class TestImpactFiltering:
    def test_only_high_impact_filtered(self):
        nf = NewsFilter(blackout_before_min=15)
        now = datetime.now(UTC)
        # Low-impact event in 5 minutes — should NOT block
        event = _make_event(minutes_from_now=5, impact="low", base_time=now)
        nf.add_event(event)
        can_trade, scale, _ = nf.check(now)
        assert can_trade is True
        assert scale == 1.0

    def test_medium_impact_not_filtered(self):
        nf = NewsFilter(blackout_before_min=15)
        now = datetime.now(UTC)
        event = _make_event(minutes_from_now=5, impact="medium", base_time=now)
        nf.add_event(event)
        can_trade, scale, _ = nf.check(now)
        assert can_trade is True


class TestEventManagement:
    def test_clear_old_events(self):
        nf = NewsFilter(blackout_after_min=10)
        now = datetime.now(UTC)
        # Event 1 hour ago — well past blackout_after
        old_event = _make_event(minutes_from_now=-60, base_time=now)
        # Event in 30 minutes — still relevant
        future_event = _make_event(minutes_from_now=30, base_time=now)
        nf.add_events([old_event, future_event])
        assert len(nf._events) == 2

        nf.clear_old_events(before=now)
        assert len(nf._events) == 1
        assert nf._events[0].time_utc == future_event.time_utc

    def test_add_events_sorts_by_time(self):
        nf = NewsFilter()
        now = datetime.now(UTC)
        e1 = _make_event(minutes_from_now=60, name="Later", base_time=now)
        e2 = _make_event(minutes_from_now=10, name="Sooner", base_time=now)
        nf.add_events([e1, e2])
        assert nf._events[0].name == "Sooner"
        assert nf._events[1].name == "Later"
