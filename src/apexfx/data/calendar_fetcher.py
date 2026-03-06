"""Forex Factory economic calendar fetcher.

Downloads and parses economic calendar data for use in fundamental
analysis features. Supports two modes:

1. **XML feed** (live/weekly): Fast, structured, no API key needed.
   Source: ``https://nfs.faireconomy.media/ff_calendar_thisweek.xml``

2. **HTML scraper** (historical): Parses Forex Factory archive pages
   for backtesting over long date ranges.

Usage::

    fetcher = CalendarFetcher()

    # Current week (XML)
    events = fetcher.fetch_current_week()

    # Historical range (HTML)
    events = fetcher.fetch_range(date(2020, 1, 1), date(2024, 12, 31))

    # Save to CSV for CalendarProvider
    fetcher.save_csv(events, "data/economic_calendar.csv")
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

from apexfx.data.calendar_provider import CalendarEvent
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Forex Factory XML feed URL
_FF_XML_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"

# Forex Factory HTML calendar base URL
_FF_HTML_URL = "https://www.forexfactory.com/calendar"

# Eastern Time offset (UTC-5 standard, UTC-4 DST)
# Forex Factory times are in Eastern Time
_ET_OFFSET_STANDARD = timedelta(hours=-5)
_ET_OFFSET_DST = timedelta(hours=-4)

# US DST: second Sunday in March to first Sunday in November
_DST_TRANSITIONS_CACHE: dict[int, tuple[date, date]] = {}


def _get_dst_dates(year: int) -> tuple[date, date]:
    """Get US DST start/end dates for a given year."""
    if year in _DST_TRANSITIONS_CACHE:
        return _DST_TRANSITIONS_CACHE[year]

    # Second Sunday in March
    march1 = date(year, 3, 1)
    days_to_sunday = (6 - march1.weekday()) % 7
    first_sunday_march = march1 + timedelta(days=days_to_sunday)
    second_sunday_march = first_sunday_march + timedelta(weeks=1)

    # First Sunday in November
    nov1 = date(year, 11, 1)
    days_to_sunday_nov = (6 - nov1.weekday()) % 7
    first_sunday_nov = nov1 + timedelta(days=days_to_sunday_nov)

    _DST_TRANSITIONS_CACHE[year] = (second_sunday_march, first_sunday_nov)
    return second_sunday_march, first_sunday_nov


def _is_us_dst(d: date) -> bool:
    """Check if a date falls in US DST."""
    dst_start, dst_end = _get_dst_dates(d.year)
    return dst_start <= d < dst_end


def _et_to_utc(dt: datetime) -> datetime:
    """Convert Eastern Time datetime to UTC."""
    if _is_us_dst(dt.date()):
        offset = _ET_OFFSET_DST
    else:
        offset = _ET_OFFSET_STANDARD
    return (dt - offset).replace(tzinfo=timezone.utc)


def _parse_ff_numeric(value: str | None) -> float | None:
    """Parse Forex Factory numeric values (handles K, M, B, % suffixes).

    Examples:
        "216K" → 216.0 (thousands — stored as-is for comparison with forecast)
        "3.4%" → 3.4
        "-0.5%" → -0.5
        "1.234M" → 1234.0
        "" → None
    """
    if value is None:
        return None

    s = value.strip()
    if not s or s in ("-", "—", "N/A", "n/a"):
        return None

    # Remove HTML entities and whitespace
    s = s.replace("&nbsp;", "").replace(",", "").strip()

    multiplier = 1.0
    if s.endswith("%"):
        s = s[:-1].strip()
    elif s.endswith("K"):
        s = s[:-1].strip()
        # Keep K values as-is for proper surprise computation
        # (forecast and actual both use K notation)
    elif s.endswith("M"):
        s = s[:-1].strip()
        multiplier = 1000.0
    elif s.endswith("B"):
        s = s[:-1].strip()
        multiplier = 1_000_000.0
    elif s.endswith("T"):
        s = s[:-1].strip()
        multiplier = 1_000_000_000.0

    try:
        return float(s) * multiplier
    except (ValueError, TypeError):
        return None


def _parse_ff_impact(impact_str: str) -> str:
    """Normalize impact string to lowercase standard."""
    s = impact_str.strip().lower()
    if "high" in s or s == "red" or "holiday" in s:
        return "high"
    if "medium" in s or s == "orange" or s == "ora":
        return "medium"
    if "low" in s or s == "yellow" or s == "yel":
        return "low"
    return "low"


def _parse_ff_time(time_str: str, event_date: date) -> datetime | None:
    """Parse Forex Factory time string to datetime.

    Forex Factory times are in Eastern Time.
    Handles formats: "8:30am", "2:00pm", "All Day", "Tentative", ""
    """
    s = time_str.strip().lower()
    if not s or s in ("all day", "tentative", "day 1", "day 2", "day 3"):
        # Use midnight for all-day / tentative events
        return datetime(event_date.year, event_date.month, event_date.day, 0, 0)

    # Parse "8:30am" / "2:00pm" format
    match = re.match(r"(\d{1,2}):(\d{2})(am|pm)", s)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = match.group(3)

    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0

    return datetime(event_date.year, event_date.month, event_date.day, hour, minute)


class CalendarFetcher:
    """Downloads economic calendar data from Forex Factory.

    Supports XML feed (current week) and HTML scraping (historical).
    Respects rate limits with configurable delays between requests.
    """

    def __init__(
        self,
        request_delay_s: float = 2.0,
        user_agent: str = "ApexFX-Quantum/1.0",
        timeout_s: int = 30,
    ) -> None:
        self._delay = request_delay_s
        self._headers = {"User-Agent": user_agent}
        self._timeout = timeout_s
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """Enforce delay between HTTP requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str) -> str:
        """Make a rate-limited GET request."""
        import requests

        self._rate_limit()
        logger.debug("Fetching URL", url=url)

        response = requests.get(
            url,
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.text

    # ------------------------------------------------------------------
    # XML feed (current week)
    # ------------------------------------------------------------------

    def fetch_current_week(self) -> list[CalendarEvent]:
        """Fetch this week's calendar from the XML feed.

        This is the fastest and most reliable method for live trading.
        """
        try:
            xml_content = self._get(_FF_XML_URL)
            events = self.parse_xml(xml_content)
            logger.info("Fetched current week calendar", n_events=len(events))
            return events
        except Exception as e:
            logger.error("Failed to fetch current week calendar", error=str(e))
            return []

    def fetch_week(self, target_date: date) -> list[CalendarEvent]:
        """Fetch calendar for the week containing target_date.

        Uses XML feed for current week, HTML scraper for historical.
        """
        today = date.today()
        # Check if target_date is in the current week
        week_start = today - timedelta(days=today.weekday())  # Monday
        week_end = week_start + timedelta(days=6)  # Sunday

        if week_start <= target_date <= week_end:
            return self.fetch_current_week()
        else:
            return self._fetch_week_html(target_date)

    def parse_xml(self, xml_content: str) -> list[CalendarEvent]:
        """Parse Forex Factory XML feed into CalendarEvent list.

        Expected format:
        <weeklyevents>
          <event>
            <title>Non-Farm Employment Change</title>
            <country>USD</country>
            <date>01-05-2024</date>
            <time>8:30am</time>
            <impact>High</impact>
            <forecast>170K</forecast>
            <previous>199K</previous>
          </event>
        </weeklyevents>
        """
        events: list[CalendarEvent] = []

        try:
            root = ElementTree.fromstring(xml_content)
        except ElementTree.ParseError as e:
            logger.error("Failed to parse XML", error=str(e))
            return events

        for event_elem in root.findall(".//event"):
            try:
                title = (event_elem.findtext("title") or "").strip()
                country = (event_elem.findtext("country") or "").strip().upper()
                date_str = (event_elem.findtext("date") or "").strip()
                time_str = (event_elem.findtext("time") or "").strip()
                impact_str = (event_elem.findtext("impact") or "").strip()
                forecast_str = event_elem.findtext("forecast")
                previous_str = event_elem.findtext("previous")
                actual_str = event_elem.findtext("actual")

                if not title or not country or not date_str:
                    continue

                # Parse date (MM-DD-YYYY format)
                try:
                    event_date = datetime.strptime(date_str, "%m-%d-%Y").date()
                except ValueError:
                    # Try alternative formats
                    try:
                        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        logger.debug("Cannot parse date", date_str=date_str)
                        continue

                # Parse time
                event_dt = _parse_ff_time(time_str, event_date)
                if event_dt is None:
                    continue

                # Convert to UTC
                time_utc = _et_to_utc(event_dt)

                event = CalendarEvent(
                    time_utc=time_utc,
                    currency=country,
                    name=title,
                    impact=_parse_ff_impact(impact_str),
                    actual=_parse_ff_numeric(actual_str),
                    forecast=_parse_ff_numeric(forecast_str),
                    previous=_parse_ff_numeric(previous_str),
                )
                events.append(event)

            except Exception as e:
                logger.debug("Skipping event due to parse error", error=str(e))
                continue

        events.sort(key=lambda e: e.time_utc)
        return events

    # ------------------------------------------------------------------
    # HTML scraper (historical)
    # ------------------------------------------------------------------

    def _fetch_week_html(self, target_date: date) -> list[CalendarEvent]:
        """Fetch calendar for a specific week from Forex Factory HTML."""
        # Forex Factory URL format: /calendar?week=jan6.2024
        month_abbr = target_date.strftime("%b").lower()
        url = f"{_FF_HTML_URL}?week={month_abbr}{target_date.day}.{target_date.year}"

        try:
            html_content = self._get(url)
            events = self.parse_html(html_content, target_date)
            return events
        except Exception as e:
            logger.error(
                "Failed to fetch week HTML",
                date=str(target_date),
                error=str(e),
            )
            return []

    def parse_html(self, html_content: str, week_date: date) -> list[CalendarEvent]:
        """Parse Forex Factory HTML calendar page.

        This is a best-effort parser for the FF calendar HTML structure.
        FF uses a table with class 'calendar__table' containing rows
        with class 'calendar__row'.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")
        events: list[CalendarEvent] = []

        # Find calendar table
        table = soup.find("table", class_="calendar__table")
        if table is None:
            logger.debug("No calendar table found in HTML")
            return events

        current_date = week_date
        rows = table.find_all("tr", class_="calendar__row")

        for row in rows:
            try:
                # Check for date cell (indicates new day)
                date_cell = row.find("td", class_="calendar__date")
                if date_cell:
                    date_text = date_cell.get_text(strip=True)
                    if date_text:
                        parsed_date = self._parse_ff_html_date(date_text, week_date.year)
                        if parsed_date:
                            current_date = parsed_date

                # Get event data cells
                currency_cell = row.find("td", class_="calendar__currency")
                event_cell = row.find("td", class_="calendar__event")
                impact_cell = row.find("td", class_="calendar__impact")
                time_cell = row.find("td", class_="calendar__time")
                actual_cell = row.find("td", class_="calendar__actual")
                forecast_cell = row.find("td", class_="calendar__forecast")
                previous_cell = row.find("td", class_="calendar__previous")

                if not event_cell or not currency_cell:
                    continue

                title = event_cell.get_text(strip=True)
                currency = currency_cell.get_text(strip=True).upper()

                if not title or not currency:
                    continue

                # Parse impact from span class (e.g., "icon--ff-impact-red")
                impact = "low"
                if impact_cell:
                    impact_span = impact_cell.find("span")
                    if impact_span:
                        cls = " ".join(impact_span.get("class", []))
                        if "red" in cls or "high" in cls:
                            impact = "high"
                        elif "ora" in cls or "medium" in cls:
                            impact = "medium"
                        elif "yel" in cls or "low" in cls:
                            impact = "low"

                # Parse time
                time_text = time_cell.get_text(strip=True) if time_cell else ""
                event_dt = _parse_ff_time(time_text, current_date)
                if event_dt is None:
                    continue

                time_utc = _et_to_utc(event_dt)

                event = CalendarEvent(
                    time_utc=time_utc,
                    currency=currency,
                    name=title,
                    impact=impact,
                    actual=_parse_ff_numeric(
                        actual_cell.get_text(strip=True) if actual_cell else None
                    ),
                    forecast=_parse_ff_numeric(
                        forecast_cell.get_text(strip=True) if forecast_cell else None
                    ),
                    previous=_parse_ff_numeric(
                        previous_cell.get_text(strip=True) if previous_cell else None
                    ),
                )
                events.append(event)

            except Exception as e:
                logger.debug("Skipping HTML row", error=str(e))
                continue

        events.sort(key=lambda e: e.time_utc)
        return events

    @staticmethod
    def _parse_ff_html_date(text: str, year: int) -> date | None:
        """Parse FF HTML date like 'Mon Jan 6' into a date."""
        # Remove day-of-week prefix
        parts = text.split()
        if len(parts) < 2:
            return None

        # Try various formats
        for fmt in ("%b %d", "%b%d"):
            try:
                clean = " ".join(parts[-2:])  # Take last two parts (month + day)
                parsed = datetime.strptime(clean, "%b %d")
                return date(year, parsed.month, parsed.day)
            except ValueError:
                continue

        return None

    # ------------------------------------------------------------------
    # Historical range fetch
    # ------------------------------------------------------------------

    def fetch_range(
        self,
        start: date,
        end: date,
        progress_callback: callable | None = None,
    ) -> list[CalendarEvent]:
        """Fetch calendar events for a date range (week by week).

        This makes multiple HTTP requests with rate limiting.
        For long ranges (years), this can take several minutes.

        Args:
            start: Start date.
            end: End date.
            progress_callback: Optional callback(current_week, total_weeks).
        """
        all_events: list[CalendarEvent] = []
        seen_keys: set[tuple] = set()  # Deduplicate events

        # Generate Monday dates for each week in range
        current = start - timedelta(days=start.weekday())  # Align to Monday
        weeks = []
        while current <= end:
            weeks.append(current)
            current += timedelta(weeks=1)

        logger.info(
            "Fetching historical calendar",
            start=str(start),
            end=str(end),
            n_weeks=len(weeks),
        )

        for i, week_monday in enumerate(weeks):
            events = self.fetch_week(week_monday)

            for event in events:
                # Filter to requested range
                event_date = event.time_utc.date() if hasattr(event.time_utc, "date") else event.time_utc
                if isinstance(event_date, datetime):
                    event_date = event_date.date()
                if event_date < start or event_date > end:
                    continue

                # Deduplicate by (time, currency, name)
                key = (event.time_utc, event.currency, event.name)
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_events.append(event)

            if progress_callback:
                progress_callback(i + 1, len(weeks))

            if (i + 1) % 10 == 0:
                logger.info(
                    "Calendar fetch progress",
                    weeks_done=i + 1,
                    total=len(weeks),
                    events_so_far=len(all_events),
                )

        all_events.sort(key=lambda e: e.time_utc)
        logger.info("Historical calendar fetch complete", n_events=len(all_events))
        return all_events

    # ------------------------------------------------------------------
    # CSV export/import
    # ------------------------------------------------------------------

    def save_csv(self, events: list[CalendarEvent], path: str | Path) -> Path:
        """Save events to CSV compatible with CalendarProvider.load().

        CSV columns: datetime_utc, currency, event_name, impact, actual, forecast, previous
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for event in events:
            records.append({
                "datetime_utc": event.time_utc.strftime("%Y-%m-%d %H:%M:%S%z")
                if event.time_utc.tzinfo
                else event.time_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "currency": event.currency,
                "event_name": event.name,
                "impact": event.impact,
                "actual": event.actual if event.actual is not None else "",
                "forecast": event.forecast if event.forecast is not None else "",
                "previous": event.previous if event.previous is not None else "",
            })

        df = pd.DataFrame(records)
        df.to_csv(path, index=False)
        logger.info("Calendar saved to CSV", path=str(path), n_events=len(records))
        return path

    def update_csv(
        self,
        csv_path: str | Path,
        start: date | None = None,
    ) -> Path:
        """Incrementally update a CSV file with latest events.

        If the CSV exists, reads the last date and fetches from there.
        Otherwise creates from scratch starting from `start` or 30 days ago.
        """
        csv_path = Path(csv_path)

        existing_events: list[CalendarEvent] = []
        last_date = start or (date.today() - timedelta(days=30))

        if csv_path.exists():
            from apexfx.data.calendar_provider import CalendarProvider

            provider = CalendarProvider()
            existing_events = provider.load(str(csv_path))
            if existing_events:
                last_dt = max(e.time_utc for e in existing_events)
                last_date = last_dt.date()
                logger.info("Updating from last date", last_date=str(last_date))

        # Fetch new events
        new_events = self.fetch_range(last_date, date.today())

        # Merge (deduplicate by key)
        seen: set[tuple] = set()
        merged: list[CalendarEvent] = []

        for event in existing_events + new_events:
            key = (event.time_utc, event.currency, event.name)
            if key not in seen:
                seen.add(key)
                merged.append(event)

        merged.sort(key=lambda e: e.time_utc)
        return self.save_csv(merged, csv_path)
