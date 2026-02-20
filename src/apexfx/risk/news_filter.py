"""News/economic calendar filter — reduces or blocks trading around major events.

High-impact events (NFP, FOMC, ECB rate decisions) cause extreme volatility
and unpredictable moves. Trading through these events is essentially gambling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NewsEvent:
    """A scheduled economic event."""
    time_utc: datetime
    currency: str
    impact: str  # "high", "medium", "low"
    name: str


class NewsFilter:
    """Filter trading around major economic events.

    Maintains a calendar of upcoming high-impact events and
    blocks/reduces trading in the blackout window around them.

    Usage:
        filter = NewsFilter(blackout_before_min=15, blackout_after_min=10)
        filter.add_event(NewsEvent(...))
        can_trade, scale = filter.check(utc_now)
    """

    def __init__(
        self,
        blackout_before_min: int = 15,
        blackout_after_min: int = 10,
        reduce_before_min: int = 60,
        reduce_scale: float = 0.5,
    ) -> None:
        self._blackout_before = timedelta(minutes=blackout_before_min)
        self._blackout_after = timedelta(minutes=blackout_after_min)
        self._reduce_before = timedelta(minutes=reduce_before_min)
        self._reduce_scale = reduce_scale
        self._events: list[NewsEvent] = []

        # Common recurring high-impact events (for auto-scheduling)
        self._recurring_events = [
            # US events (typically first Friday of month, FOMC every 6 weeks)
            "Non-Farm Payrolls",
            "FOMC Rate Decision",
            "CPI",
            "Core PCE",
            "GDP",
            "Retail Sales",
            # ECB
            "ECB Rate Decision",
            "ECB Press Conference",
            # BOE
            "BOE Rate Decision",
            # BOJ
            "BOJ Rate Decision",
        ]

    def add_event(self, event: NewsEvent) -> None:
        """Add an economic event to the calendar."""
        self._events.append(event)
        self._events.sort(key=lambda e: e.time_utc)

    def add_events(self, events: list[NewsEvent]) -> None:
        """Add multiple events."""
        self._events.extend(events)
        self._events.sort(key=lambda e: e.time_utc)

    def clear_old_events(self, before: datetime | None = None) -> None:
        """Remove events that have already passed."""
        cutoff = before or datetime.now(timezone.utc)
        self._events = [
            e for e in self._events
            if e.time_utc + self._blackout_after > cutoff
        ]

    def check(
        self,
        utc_now: datetime | None = None,
        symbol_currencies: list[str] | None = None,
    ) -> tuple[bool, float, str]:
        """Check if trading is allowed right now.

        Args:
            utc_now: Current UTC time.
            symbol_currencies: Currencies in the traded pair (e.g. ["EUR", "USD"]).
                If None, all events are considered.

        Returns:
            (can_trade, position_scale, reason)
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        for event in self._events:
            # Filter by currency if specified
            if symbol_currencies and event.currency not in symbol_currencies:
                continue

            # Only filter high-impact events
            if event.impact != "high":
                continue

            time_to_event = event.time_utc - utc_now
            time_since_event = utc_now - event.time_utc

            # Hard blackout: no trading
            if (
                timedelta(0) <= time_to_event <= self._blackout_before
                or timedelta(0) <= time_since_event <= self._blackout_after
            ):
                reason = (
                    f"News blackout: {event.name} ({event.currency}) "
                    f"at {event.time_utc.strftime('%H:%M UTC')}"
                )
                logger.info(reason)
                return False, 0.0, reason

            # Reduced exposure window
            if timedelta(0) <= time_to_event <= self._reduce_before:
                reason = (
                    f"Pre-news reduction: {event.name} in "
                    f"{int(time_to_event.total_seconds() / 60)}min"
                )
                return True, self._reduce_scale, reason

        return True, 1.0, ""

    @property
    def upcoming_events(self) -> list[NewsEvent]:
        """Get upcoming events sorted by time."""
        now = datetime.now(timezone.utc)
        return [e for e in self._events if e.time_utc > now]

    @property
    def next_event(self) -> NewsEvent | None:
        upcoming = self.upcoming_events
        return upcoming[0] if upcoming else None
