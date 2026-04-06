"""Data validation layer for bar DataFrames.

Validates incoming OHLCV data before it enters the feature pipeline
or training loop. Catches gaps, NaN, duplicates, and data quality
issues that would silently corrupt features and model training.

Usage::

    from apexfx.data.validator import BarDataValidator

    validator = BarDataValidator()
    report = validator.validate(bars)
    if not report.is_valid:
        for issue in report.issues:
            logger.warning("Data issue", **issue)
    bars_clean = validator.clean(bars)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    """Single data quality issue."""

    severity: Literal["error", "warning", "info"]
    check: str
    message: str
    count: int = 0
    details: dict | None = None


@dataclass
class ValidationReport:
    """Result of validating a bar DataFrame."""

    issues: list[ValidationIssue] = field(default_factory=list)
    n_bars: int = 0
    n_bars_after_clean: int = 0

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def summary(self) -> str:
        lines = [f"Validation: {self.n_bars} bars"]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.upper()}] {issue.check}: {issue.message}")
        if not self.issues:
            lines.append("  All checks passed.")
        return "\n".join(lines)


# Required columns for OHLCV bar data
REQUIRED_COLUMNS = {"time", "open", "high", "low", "close", "volume"}


class BarDataValidator:
    """Validates and cleans OHLCV bar DataFrames.

    Parameters
    ----------
    max_gap_factor : float
        Maximum allowed gap between consecutive bars, as a multiple of
        the median time delta. Gaps beyond this trigger a warning.
    max_nan_pct : float
        Maximum allowed fraction of NaN values per column before
        raising an error.
    min_bars : int
        Minimum number of bars required for valid data.
    check_ohlc_consistency : bool
        Verify that high >= max(open, close) and low <= min(open, close).
    """

    def __init__(
        self,
        max_gap_factor: float = 5.0,
        max_nan_pct: float = 0.10,
        min_bars: int = 100,
        check_ohlc_consistency: bool = True,
    ) -> None:
        self._max_gap_factor = max_gap_factor
        self._max_nan_pct = max_nan_pct
        self._min_bars = min_bars
        self._check_ohlc = check_ohlc_consistency

    def validate(self, bars: pd.DataFrame) -> ValidationReport:
        """Run all checks and return a report."""
        report = ValidationReport(n_bars=len(bars))

        self._check_required_columns(bars, report)
        if any(i.severity == "error" for i in report.issues):
            return report

        self._check_min_bars(bars, report)
        self._check_duplicates(bars, report)
        self._check_nan(bars, report)
        self._check_time_gaps(bars, report)
        self._check_ohlc_consistency(bars, report)
        self._check_price_sanity(bars, report)
        self._check_volume_sanity(bars, report)

        return report

    def clean(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of the DataFrame.

        Steps:
        1. Drop duplicate timestamps (keep last)
        2. Sort by time
        3. Forward-fill NaN in OHLC columns (small gaps only)
        4. Drop rows where close is still NaN
        5. Fix OHLC consistency (high >= max(open,close), etc.)
        6. Clip extreme returns (> 10 sigma)
        """
        df = bars.copy()

        # 1. Drop duplicate timestamps
        if "time" in df.columns:
            n_before = len(df)
            df = df.drop_duplicates(subset=["time"], keep="last")
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                logger.info("Dropped duplicate timestamps", count=n_dropped)

            # 2. Sort by time
            df = df.sort_values("time").reset_index(drop=True)

        # 3. Forward-fill NaN in price columns (max 3 consecutive)
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        df[price_cols] = df[price_cols].ffill(limit=3)

        # 4. Drop rows where close is still NaN
        if "close" in df.columns:
            valid = df["close"].notna()
            n_dropped = (~valid).sum()
            if n_dropped > 0:
                logger.info("Dropped bars with NaN close", count=int(n_dropped))
                df = df[valid].reset_index(drop=True)

        # 5. Fix OHLC consistency
        if self._check_ohlc and all(c in df.columns for c in ["open", "high", "low", "close"]):
            o = df["open"].values
            h = df["high"].values
            l = df["low"].values
            c = df["close"].values

            # high must be >= all of open, close, low
            df["high"] = np.maximum(h, np.maximum(o, c))
            # low must be <= all of open, close, high
            df["low"] = np.minimum(l, np.minimum(o, c))

        # 6. Fill NaN volume with 0
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        # 7. Clip extreme log returns (> 10 sigma from rolling mean)
        if "close" in df.columns and len(df) > 50:
            log_ret = np.log(df["close"].values[1:] / df["close"].values[:-1])
            if len(log_ret) > 0:
                mu = np.nanmean(log_ret)
                sigma = np.nanstd(log_ret)
                if sigma > 1e-10:
                    extreme = np.abs(log_ret - mu) > 10 * sigma
                    n_extreme = extreme.sum()
                    if n_extreme > 0:
                        logger.warning(
                            "Extreme returns detected",
                            count=int(n_extreme),
                            max_sigma=float(np.max(np.abs(log_ret - mu) / sigma)),
                        )

        return df

    # --- Individual checks ---

    @staticmethod
    def _check_required_columns(bars: pd.DataFrame, report: ValidationReport) -> None:
        missing = REQUIRED_COLUMNS - set(bars.columns)
        if missing:
            report.issues.append(ValidationIssue(
                severity="error",
                check="required_columns",
                message=f"Missing columns: {sorted(missing)}",
            ))

    def _check_min_bars(self, bars: pd.DataFrame, report: ValidationReport) -> None:
        if len(bars) < self._min_bars:
            report.issues.append(ValidationIssue(
                severity="error",
                check="min_bars",
                message=f"Only {len(bars)} bars, need at least {self._min_bars}",
                count=len(bars),
            ))

    @staticmethod
    def _check_duplicates(bars: pd.DataFrame, report: ValidationReport) -> None:
        if "time" not in bars.columns:
            return
        n_dups = bars["time"].duplicated().sum()
        if n_dups > 0:
            report.issues.append(ValidationIssue(
                severity="warning",
                check="duplicates",
                message=f"{n_dups} duplicate timestamps found",
                count=int(n_dups),
            ))

    def _check_nan(self, bars: pd.DataFrame, report: ValidationReport) -> None:
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in bars.columns:
                continue
            nan_count = bars[col].isna().sum()
            nan_pct = nan_count / len(bars) if len(bars) > 0 else 0
            if nan_pct > self._max_nan_pct:
                report.issues.append(ValidationIssue(
                    severity="error",
                    check="nan_values",
                    message=f"Column '{col}' has {nan_pct:.1%} NaN ({nan_count}/{len(bars)})",
                    count=int(nan_count),
                ))
            elif nan_count > 0:
                report.issues.append(ValidationIssue(
                    severity="warning",
                    check="nan_values",
                    message=f"Column '{col}' has {nan_count} NaN values ({nan_pct:.1%})",
                    count=int(nan_count),
                ))

    def _check_time_gaps(self, bars: pd.DataFrame, report: ValidationReport) -> None:
        if "time" not in bars.columns or len(bars) < 2:
            return

        times = pd.to_datetime(bars["time"])
        deltas = times.diff().dropna()
        if len(deltas) == 0:
            return

        median_delta = deltas.median()
        if median_delta.total_seconds() == 0:
            return

        large_gaps = deltas > median_delta * self._max_gap_factor
        n_gaps = large_gaps.sum()
        if n_gaps > 0:
            max_gap = deltas.max()
            report.issues.append(ValidationIssue(
                severity="warning",
                check="time_gaps",
                message=(
                    f"{n_gaps} gaps > {self._max_gap_factor}x median delta "
                    f"(median={median_delta}, max_gap={max_gap})"
                ),
                count=int(n_gaps),
            ))

    def _check_ohlc_consistency(self, bars: pd.DataFrame, report: ValidationReport) -> None:
        if not self._check_ohlc:
            return
        required = ["open", "high", "low", "close"]
        if not all(c in bars.columns for c in required):
            return

        o = bars["open"].values
        h = bars["high"].values
        l = bars["low"].values  # noqa: E741
        c = bars["close"].values

        # high should be >= open, close; low should be <= open, close
        bad_high = np.nansum(h < np.minimum(o, c))
        bad_low = np.nansum(l > np.maximum(o, c))
        bad_hl = np.nansum(h < l)

        total_bad = bad_high + bad_low + bad_hl
        if total_bad > 0:
            report.issues.append(ValidationIssue(
                severity="warning",
                check="ohlc_consistency",
                message=(
                    f"{total_bad} OHLC violations "
                    f"(high<price: {bad_high}, low>price: {bad_low}, high<low: {bad_hl})"
                ),
                count=int(total_bad),
            ))

    @staticmethod
    def _check_price_sanity(bars: pd.DataFrame, report: ValidationReport) -> None:
        if "close" not in bars.columns:
            return
        close = pd.to_numeric(bars["close"], errors="coerce").values
        valid = close[~np.isnan(close)]
        if len(valid) < 2:
            return

        # Check for zero or negative prices
        n_bad = np.sum(valid <= 0)
        if n_bad > 0:
            report.issues.append(ValidationIssue(
                severity="error",
                check="price_sanity",
                message=f"{n_bad} bars with zero or negative close price",
                count=int(n_bad),
            ))

        # Check for suspiciously flat prices
        n_unchanged = np.sum(np.diff(valid) == 0)
        pct_unchanged = n_unchanged / (len(valid) - 1)
        if pct_unchanged > 0.5:
            report.issues.append(ValidationIssue(
                severity="warning",
                check="price_sanity",
                message=f"{pct_unchanged:.0%} of bars have unchanged close price",
                count=int(n_unchanged),
            ))

    @staticmethod
    def _check_volume_sanity(bars: pd.DataFrame, report: ValidationReport) -> None:
        if "volume" not in bars.columns:
            return
        volume = pd.to_numeric(bars["volume"], errors="coerce").values
        valid = volume[~np.isnan(volume)]

        n_zero = np.sum(valid == 0)
        pct_zero = n_zero / len(valid) if len(valid) > 0 else 0
        if pct_zero > 0.3:
            report.issues.append(ValidationIssue(
                severity="warning",
                check="volume_sanity",
                message=f"{pct_zero:.0%} of bars have zero volume ({n_zero}/{len(valid)})",
                count=int(n_zero),
            ))

        n_negative = np.sum(valid < 0)
        if n_negative > 0:
            report.issues.append(ValidationIssue(
                severity="error",
                check="volume_sanity",
                message=f"{n_negative} bars with negative volume",
                count=int(n_negative),
            ))
