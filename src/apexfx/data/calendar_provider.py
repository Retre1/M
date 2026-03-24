"""Economic calendar provider for fundamental analysis features.

Loads historical economic calendar data (NFP, CPI, FOMC, etc.)
for backtesting and live trading. Computes surprise scores.

Expected CSV schema:
    datetime_utc, currency, event_name, impact, actual, forecast, previous
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalendarEvent:
    """A single economic calendar event with surprise scoring."""

    time_utc: datetime
    currency: str
    name: str
    impact: str  # "high", "medium", "low"
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    surprise_score: float = 0.0  # (actual - forecast) / historical_std

    @property
    def has_data(self) -> bool:
        return self.actual is not None and self.forecast is not None

    @property
    def raw_surprise(self) -> float:
        """Raw surprise = actual - forecast."""
        if not self.has_data:
            return 0.0
        return self.actual - self.forecast

    @property
    def is_hawkish(self) -> bool:
        """Positive surprise in inflation/employment → hawkish (currency bullish)."""
        hawkish_events = {
            "CPI", "Core CPI", "Core PCE", "PPI",
            "Non-Farm Payrolls", "NFP", "Employment Change",
            "Retail Sales", "GDP", "PMI", "ISM Manufacturing",
            "ISM Services", "Average Hourly Earnings",
        }
        dovish_events = {
            "Unemployment Rate", "Unemployment Claims",
            "Initial Jobless Claims", "Continuing Claims",
        }
        name_upper = self.name.upper()
        for evt in hawkish_events:
            if evt.upper() in name_upper:
                return self.raw_surprise > 0
        for evt in dovish_events:
            if evt.upper() in name_upper:
                return self.raw_surprise < 0  # lower unemployment = hawkish
        # Rate decisions: higher rate = hawkish
        if "RATE" in name_upper and "DECISION" in name_upper:
            return self.raw_surprise > 0
        return self.raw_surprise > 0  # default


class CalendarProvider:
    """Loads and serves economic calendar data.

    Usage::

        provider = CalendarProvider()
        provider.load("data/economic_calendar.csv")
        events = provider.get_events(start_dt, end_dt, currencies=["USD"])
    """

    def __init__(self) -> None:
        self._events: list[CalendarEvent] = []
        self._by_currency: dict[str, list[CalendarEvent]] = {}
        self._surprise_stds: dict[str, float] = {}

    def load(self, path: str | Path) -> list[CalendarEvent]:
        """Load calendar from CSV file.

        Expected columns: datetime_utc, currency, event_name, impact, actual, forecast, previous
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Calendar file not found, using empty calendar", path=str(path))
            return []

        df = pd.read_csv(path)

        # Normalize column names
        col_map = {}
        for col in df.columns:
            lower = col.strip().lower().replace(" ", "_")
            col_map[col] = lower
        df = df.rename(columns=col_map)

        required = {"datetime_utc", "currency", "event_name", "impact"}
        missing = required - set(df.columns)
        if missing:
            logger.error("Missing required columns", missing=missing)
            return []

        events = []
        for _, row in df.iterrows():
            try:
                time_utc = pd.to_datetime(row["datetime_utc"], utc=True).to_pydatetime()
            except Exception:
                continue

            actual = self._parse_float(row.get("actual"))
            forecast = self._parse_float(row.get("forecast"))
            previous = self._parse_float(row.get("previous"))

            event = CalendarEvent(
                time_utc=time_utc,
                currency=str(row["currency"]).strip().upper(),
                name=str(row["event_name"]).strip(),
                impact=str(row["impact"]).strip().lower(),
                actual=actual,
                forecast=forecast,
                previous=previous,
            )
            events.append(event)

        events.sort(key=lambda e: e.time_utc)
        self._events = events
        self._build_index()
        self._compute_surprise_scores()

        logger.info(
            "Calendar loaded",
            n_events=len(events),
            date_range=f"{events[0].time_utc:%Y-%m-%d} to {events[-1].time_utc:%Y-%m-%d}"
            if events else "empty",
        )
        return events

    def add_events(self, events: list[CalendarEvent]) -> None:
        """Add events programmatically (for live trading)."""
        self._events.extend(events)
        self._events.sort(key=lambda e: e.time_utc)
        self._build_index()
        self._compute_surprise_scores()

    def get_events(
        self,
        start: datetime,
        end: datetime,
        currencies: list[str] | None = None,
        impact: str | None = "high",
    ) -> list[CalendarEvent]:
        """Get events in a time range, optionally filtered by currency and impact."""
        result = []
        for event in self._events:
            if event.time_utc < start:
                continue
            if event.time_utc > end:
                break
            if currencies and event.currency not in currencies:
                continue
            if impact and event.impact != impact:
                continue
            result.append(event)
        return result

    def get_recent_events(
        self,
        as_of: datetime,
        lookback_hours: int = 24,
        currencies: list[str] | None = None,
    ) -> list[CalendarEvent]:
        """Get events in the last N hours relative to as_of."""
        start = as_of - timedelta(hours=lookback_hours)
        return self.get_events(start, as_of, currencies=currencies, impact=None)

    def next_event(
        self,
        as_of: datetime,
        currencies: list[str] | None = None,
        impact: str = "high",
    ) -> CalendarEvent | None:
        """Get the next upcoming event after as_of."""
        for event in self._events:
            if event.time_utc <= as_of:
                continue
            if currencies and event.currency not in currencies:
                continue
            if impact and event.impact != impact:
                continue
            return event
        return None

    def _build_index(self) -> None:
        """Build currency index for fast lookup."""
        self._by_currency.clear()
        for event in self._events:
            self._by_currency.setdefault(event.currency, []).append(event)

    def _compute_surprise_scores(self) -> None:
        """Compute normalized surprise scores using historical std per event name."""
        import numpy as np

        # Group by event name, compute std of raw surprises
        surprises_by_name: dict[str, list[float]] = {}
        for event in self._events:
            if event.has_data:
                surprises_by_name.setdefault(event.name, []).append(event.raw_surprise)

        stds: dict[str, float] = {}
        for name, surprises in surprises_by_name.items():
            if len(surprises) >= 3:
                std = float(np.std(surprises))
                stds[name] = max(std, 1e-10)  # prevent division by zero
            else:
                stds[name] = abs(float(np.mean(surprises))) + 1e-10 if surprises else 1.0

        self._surprise_stds = stds

        # Assign normalized surprise scores
        for event in self._events:
            if event.has_data and event.name in stds:
                event.surprise_score = event.raw_surprise / stds[event.name]

    @staticmethod
    def _parse_float(val) -> float | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            # Handle percentage strings like "3.5%"
            s = str(val).strip().replace("%", "").replace(",", "")
            if s in ("", "-", "nan", "NaN"):
                return None
            return float(s)
        except (ValueError, TypeError):
            return None
