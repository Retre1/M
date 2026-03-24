"""CFTC Commitment of Traders (COT) data fetcher.

Downloads weekly COT reports from CFTC and extracts positioning data
for major FX futures. COT data reveals institutional positioning
(commercial hedgers vs speculative traders) which is a strong
contrarian/trend-following signal.

Data source: https://www.cftc.gov/dea/futures/deacmelf.htm
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# CFTC CME FX futures — maps FX pair to CME contract code
COT_CURRENCY_MAP: dict[str, str] = {
    "EUR": "099741",   # Euro FX
    "GBP": "096742",   # British Pound
    "JPY": "097741",   # Japanese Yen
    "CHF": "092741",   # Swiss Franc
    "AUD": "232741",   # Australian Dollar
    "NZD": "112741",   # New Zealand Dollar
    "CAD": "090741",   # Canadian Dollar
    "MXN": "095741",   # Mexican Peso
}

# CFTC bulk download URLs (current year + archive)
COT_CURRENT_YEAR_URL = (
    "https://www.cftc.gov/dea/newcot/deacmelf.zip"
)
COT_ARCHIVE_URL = (
    "https://www.cftc.gov/files/dea/history/deacmelf{year}.zip"
)


@dataclass
class COTRecord:
    """Single COT report record for a currency."""
    report_date: datetime
    currency: str
    # Commercial (hedgers)
    commercial_long: int = 0
    commercial_short: int = 0
    commercial_net: int = 0
    # Non-commercial (speculators)
    speculative_long: int = 0
    speculative_short: int = 0
    speculative_net: int = 0
    # Total open interest
    open_interest: int = 0
    # Derived metrics
    spec_net_pct: float = 0.0  # spec net / open interest
    commercial_net_pct: float = 0.0


@dataclass
class COTData:
    """Parsed COT data for all tracked currencies."""
    records: dict[str, list[COTRecord]] = field(default_factory=dict)
    last_update: datetime | None = None


class COTFetcher:
    """Fetches and parses CFTC Commitment of Traders reports.

    Usage:
        fetcher = COTFetcher()
        data = fetcher.fetch_current()
        eur_records = data.records.get("EUR", [])
        latest = eur_records[-1] if eur_records else None
    """

    def __init__(
        self,
        cache_dir: str = "data/cot",
        timeout_s: float = 30.0,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout_s
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "Mozilla/5.0 (ApexFX/1.0; Financial Research)"
        )

    def fetch_current(self) -> COTData:
        """Fetch current year COT data from CFTC."""
        return self._fetch_url(COT_CURRENT_YEAR_URL)

    def fetch_historical(self, year: int) -> COTData:
        """Fetch historical COT data for a specific year."""
        url = COT_ARCHIVE_URL.format(year=year)
        return self._fetch_url(url)

    def _fetch_url(self, url: str) -> COTData:
        """Download and parse a COT zip file."""
        try:
            logger.info("Fetching COT data", url=url)
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()

            return self._parse_zip(resp.content)

        except requests.RequestException as e:
            logger.error("COT fetch failed", url=url, error=str(e))
            return COTData()

    def _parse_zip(self, zip_bytes: bytes) -> COTData:
        """Parse a CFTC zip file containing COT CSV data."""
        data = COTData(last_update=datetime.now(UTC))

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for name in zf.namelist():
                    if name.endswith(".txt") or name.endswith(".csv"):
                        with zf.open(name) as f:
                            content = f.read().decode("utf-8", errors="replace")
                            self._parse_csv(content, data)
        except (zipfile.BadZipFile, KeyError) as e:
            logger.error("COT zip parse failed", error=str(e))

        return data

    def _parse_csv(self, content: str, data: COTData) -> None:
        """Parse COT CSV content and populate data records."""
        reader = csv.DictReader(io.StringIO(content))

        for row in reader:
            # Match by CFTC contract code
            contract_code = row.get("CFTC_Contract_Market_Code", "").strip()

            currency = None
            for ccy, code in COT_CURRENCY_MAP.items():
                if contract_code == code:
                    currency = ccy
                    break

            if currency is None:
                continue

            try:
                report_date_str = row.get("As_of_Date_In_Form_YYMMDD", "")
                if not report_date_str:
                    report_date_str = row.get("Report_Date_as_YYYY-MM-DD", "")

                if len(report_date_str) == 6:
                    # YYMMDD format
                    report_date = datetime.strptime(report_date_str, "%y%m%d")
                elif "-" in report_date_str:
                    report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
                else:
                    continue

                report_date = report_date.replace(tzinfo=UTC)

                # Parse positioning columns
                comm_long = self._parse_int(
                    row.get("Comm_Positions_Long_All", "0")
                )
                comm_short = self._parse_int(
                    row.get("Comm_Positions_Short_All", "0")
                )
                spec_long = self._parse_int(
                    row.get("NonComm_Positions_Long_All", "0")
                )
                spec_short = self._parse_int(
                    row.get("NonComm_Positions_Short_All", "0")
                )
                oi = self._parse_int(
                    row.get("Open_Interest_All", "0")
                )

                comm_net = comm_long - comm_short
                spec_net = spec_long - spec_short
                spec_net_pct = spec_net / oi if oi > 0 else 0.0
                comm_net_pct = comm_net / oi if oi > 0 else 0.0

                record = COTRecord(
                    report_date=report_date,
                    currency=currency,
                    commercial_long=comm_long,
                    commercial_short=comm_short,
                    commercial_net=comm_net,
                    speculative_long=spec_long,
                    speculative_short=spec_short,
                    speculative_net=spec_net,
                    open_interest=oi,
                    spec_net_pct=spec_net_pct,
                    commercial_net_pct=comm_net_pct,
                )

                if currency not in data.records:
                    data.records[currency] = []
                data.records[currency].append(record)

            except (ValueError, KeyError):
                continue

        # Sort records by date
        for ccy in data.records:
            data.records[ccy].sort(key=lambda r: r.report_date)

    @staticmethod
    def _parse_int(s: str) -> int:
        """Parse an integer, handling commas and whitespace."""
        try:
            return int(s.strip().replace(",", ""))
        except (ValueError, AttributeError):
            return 0

    def to_dataframe(self, data: COTData, currency: str) -> pd.DataFrame:
        """Convert COT records to a pandas DataFrame."""
        records = data.records.get(currency, [])
        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            rows.append({
                "time": r.report_date,
                "cot_spec_net": r.speculative_net,
                "cot_spec_net_pct": r.spec_net_pct,
                "cot_comm_net": r.commercial_net,
                "cot_comm_net_pct": r.commercial_net_pct,
                "cot_open_interest": r.open_interest,
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def save_csv(self, data: COTData, path: str | Path | None = None) -> None:
        """Save all COT data to CSV."""
        if path is None:
            path = self._cache_dir / "cot_data.csv"
        path = Path(path)

        rows = []
        for ccy, records in data.records.items():
            for r in records:
                rows.append({
                    "date": r.report_date.strftime("%Y-%m-%d"),
                    "currency": r.currency,
                    "spec_long": r.speculative_long,
                    "spec_short": r.speculative_short,
                    "spec_net": r.speculative_net,
                    "spec_net_pct": round(r.spec_net_pct, 4),
                    "comm_long": r.commercial_long,
                    "comm_short": r.commercial_short,
                    "comm_net": r.commercial_net,
                    "open_interest": r.open_interest,
                })

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("COT data saved", path=str(path), records=len(df))

    def load_csv(self, path: str | Path | None = None) -> COTData:
        """Load COT data from cached CSV."""
        if path is None:
            path = self._cache_dir / "cot_data.csv"
        path = Path(path)

        if not path.exists():
            return COTData()

        df = pd.read_csv(path)
        data = COTData(last_update=datetime.now(UTC))

        for _, row in df.iterrows():
            ccy = row["currency"]
            record = COTRecord(
                report_date=datetime.strptime(row["date"], "%Y-%m-%d").replace(
                    tzinfo=UTC
                ),
                currency=ccy,
                speculative_long=int(row.get("spec_long", 0)),
                speculative_short=int(row.get("spec_short", 0)),
                speculative_net=int(row.get("spec_net", 0)),
                spec_net_pct=float(row.get("spec_net_pct", 0)),
                commercial_long=int(row.get("comm_long", 0)),
                commercial_short=int(row.get("comm_short", 0)),
                commercial_net=int(row.get("comm_net", 0)),
                open_interest=int(row.get("open_interest", 0)),
            )
            if ccy not in data.records:
                data.records[ccy] = []
            data.records[ccy].append(record)

        return data
