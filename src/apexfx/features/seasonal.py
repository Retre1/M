"""Seasonal pattern features for FX markets.

FX markets exhibit well-documented seasonal patterns:
- Month-of-year effects (January USD strength, summer doldrums)
- Day-of-week effects (higher volatility Mon/Fri)
- Time-of-day / session effects (London session highest volume)
- Month-end rebalancing flows (pension/mutual fund portfolio adjustments)
- Quarter-end effects (window dressing, tax-related flows)
- Holiday effects (reduced liquidity, wider spreads)

These patterns are not alpha signals on their own but provide useful context
for position sizing and risk management.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Historical average daily ranges by month (EURUSD, pips, approximate)
# Source: empirical analysis of FX markets over 20+ years
_MONTHLY_AVG_RANGE_EURUSD = {
    1: 95,   # January — moderate (post-holiday return)
    2: 90,   # February — moderate
    3: 100,  # March — higher (quarter-end flows)
    4: 85,   # April — slightly below average
    5: 85,   # May — "sell in May" period begins
    6: 80,   # June — quarter-end, moderate
    7: 70,   # July — summer doldrums (lowest)
    8: 75,   # August — summer, thin liquidity
    9: 95,   # September — return from summer, higher vol
    10: 100, # October — historically volatile
    11: 90,  # November — moderate
    12: 75,  # December — holiday season, thin liquidity
}

# Day-of-week volatility multipliers (relative to average)
_DOW_VOL_MULTIPLIER = {
    0: 0.95,  # Monday — lower (Asia-only start)
    1: 1.05,  # Tuesday — above average
    2: 1.05,  # Wednesday — above average (mid-week data releases)
    3: 1.10,  # Thursday — highest (ECB/major data days)
    4: 0.85,  # Friday — lower (early close, weekend risk)
}

# Known FX seasonality — month tends to favor direction for USD
# +1 = USD typically strengthens, -1 = USD typically weakens, 0 = neutral
_USD_SEASONAL_BIAS = {
    1: +0.3,   # January — USD tends to strengthen (new year flows)
    2: +0.1,   # February — slight USD strength
    3: -0.1,   # March — quarter-end rebalancing
    4: -0.2,   # April — tax season, some USD selling
    5: +0.1,   # May — neutral to slight USD bid
    6: 0.0,    # June — quarter-end, mixed
    7: -0.1,   # July — summer, slight USD weakness
    8: 0.0,    # August — neutral
    9: +0.2,   # September — fiscal year start, USD bid
    10: -0.1,  # October — mixed
    11: +0.2,  # November — USD tends stronger
    12: -0.2,  # December — year-end rebalancing, USD sells
}


class SeasonalExtractor(BaseFeatureExtractor):
    """Extracts seasonal pattern features from bar timestamps.

    8 features:
    - seasonal_month_vol: Historical average volatility for this month (normalized)
    - seasonal_dow_vol: Day-of-week volatility multiplier
    - seasonal_usd_bias: Historical USD directional bias for this month
    - seasonal_month_end: Proximity to month-end (0 = far, 1 = last 2 days)
    - seasonal_quarter_end: Proximity to quarter-end (0 = far, 1 = last 3 days)
    - seasonal_summer: Summer doldrums indicator (June-August)
    - seasonal_year_end: Year-end effect indicator (Dec 15 - Jan 5)
    - seasonal_session: Trading session ID (0=Asia, 1=London, 2=NY, 3=overlap)
    """

    def __init__(self) -> None:
        pass

    @property
    def feature_names(self) -> list[str]:
        return [
            "seasonal_month_vol",
            "seasonal_dow_vol",
            "seasonal_usd_bias",
            "seasonal_month_end",
            "seasonal_quarter_end",
            "seasonal_summer",
            "seasonal_year_end",
            "seasonal_session",
        ]

    def extract(
        self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Extract seasonal features from bar timestamps."""
        n = len(bars)
        result = pd.DataFrame(index=bars.index)

        # Need 'time' column
        if "time" not in bars.columns:
            for name in self.feature_names:
                result[name] = 0.0
            return result

        times = pd.to_datetime(bars["time"])

        months = times.dt.month.values
        days = times.dt.day.values
        dows = times.dt.dayofweek.values  # 0=Mon, 6=Sun
        hours = times.dt.hour.values

        # Days in each month for month-end detection
        days_in_month = times.dt.days_in_month.values

        # 1. Monthly average volatility (normalized to 0-1 range)
        max_range = max(_MONTHLY_AVG_RANGE_EURUSD.values())
        result["seasonal_month_vol"] = [
            _MONTHLY_AVG_RANGE_EURUSD.get(m, 85) / max_range for m in months
        ]

        # 2. Day-of-week volatility multiplier
        result["seasonal_dow_vol"] = [
            _DOW_VOL_MULTIPLIER.get(d, 1.0) for d in dows
        ]

        # 3. USD seasonal bias
        result["seasonal_usd_bias"] = [
            _USD_SEASONAL_BIAS.get(m, 0.0) for m in months
        ]

        # 4. Month-end proximity (0 = far, 1 = last 2 days)
        days_to_end = days_in_month - days
        result["seasonal_month_end"] = np.clip(1.0 - days_to_end / 2.0, 0.0, 1.0)

        # 5. Quarter-end proximity (0 = far, 1 = last 3 days of Mar/Jun/Sep/Dec)
        is_quarter_end_month = np.isin(months, [3, 6, 9, 12])
        qe_proximity = np.where(
            is_quarter_end_month,
            np.clip(1.0 - days_to_end / 3.0, 0.0, 1.0),
            0.0,
        )
        result["seasonal_quarter_end"] = qe_proximity

        # 6. Summer doldrums (June-August)
        result["seasonal_summer"] = np.where(
            np.isin(months, [6, 7, 8]), 1.0, 0.0
        )

        # 7. Year-end effect (Dec 15 - Jan 5)
        year_end = np.where(
            (months == 12) & (days >= 15), 1.0,
            np.where((months == 1) & (days <= 5), 1.0, 0.0),
        )
        result["seasonal_year_end"] = year_end

        # 8. Trading session (encoded as 0-3)
        # 0=Asia (0-7 UTC), 1=London (7-12 UTC), 2=NY (12-17 UTC), 3=overlap (17-21)
        sessions = np.zeros(n)
        sessions = np.where((hours >= 0) & (hours < 7), 0, sessions)   # Asia
        sessions = np.where((hours >= 7) & (hours < 12), 1, sessions)  # London
        sessions = np.where((hours >= 12) & (hours < 17), 2, sessions) # NY
        sessions = np.where(hours >= 17, 3, sessions)                  # Late/overlap
        result["seasonal_session"] = sessions / 3.0  # Normalize to [0, 1]

        return result
