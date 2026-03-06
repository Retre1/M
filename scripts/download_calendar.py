"""Download historical economic calendar data from Forex Factory.

Downloads week-by-week calendar data and saves to CSV for backtesting.

Usage::

    # Download last 5 years
    python scripts/download_calendar.py --start 2020-01-01 --end 2024-12-31

    # Update existing CSV with latest data
    python scripts/download_calendar.py --update --output data/economic_calendar.csv

    # Custom output path
    python scripts/download_calendar.py --start 2023-01-01 --output data/calendar_2023.csv
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from apexfx.data.calendar_fetcher import CalendarFetcher
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


def _progress(current: int, total: int) -> None:
    """Print progress bar."""
    pct = current / total * 100
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:.0f}% ({current}/{total} weeks)", end="", flush=True)
    if current == total:
        print()  # newline at end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download economic calendar from Forex Factory"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 2 years ago.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/economic_calendar.csv",
        help="Output CSV path. Default: data/economic_calendar.csv",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update existing CSV with latest data.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between requests in seconds. Default: 2.0",
    )

    args = parser.parse_args()

    fetcher = CalendarFetcher(request_delay_s=args.delay)

    if args.update:
        print(f"Updating {args.output}...")
        result_path = fetcher.update_csv(
            args.output,
            start=datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None,
        )
        print(f"Updated: {result_path}")
        return

    # Parse dates
    end_date = (
        datetime.strptime(args.end, "%Y-%m-%d").date()
        if args.end
        else date.today()
    )
    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d").date()
        if args.start
        else date(end_date.year - 2, end_date.month, end_date.day)
    )

    print(f"Downloading calendar: {start_date} → {end_date}")
    print(f"Output: {args.output}")
    print(f"Request delay: {args.delay}s")
    print()

    events = fetcher.fetch_range(start_date, end_date, progress_callback=_progress)

    print(f"\nTotal events: {len(events)}")

    # Stats
    high_impact = sum(1 for e in events if e.impact == "high")
    currencies = set(e.currency for e in events)
    print(f"High-impact events: {high_impact}")
    print(f"Currencies: {', '.join(sorted(currencies))}")

    # Save
    result_path = fetcher.save_csv(events, args.output)
    print(f"\nSaved to: {result_path}")


if __name__ == "__main__":
    main()
