"""Fundamental analysis feature extractor.

Computes features from economic calendar data that allow the RL agent to learn:
1. When high-impact news creates directional opportunity (surprise score)
2. Whether to avoid trading (pre-news uncertainty)
3. Accumulated fundamental bias from recent events
4. Rate differentials between currencies
5. Whether recent signals conflict (uncertainty)

The agent LEARNS from these features — they're not hardcoded rules.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from apexfx.data.calendar_provider import CalendarEvent, CalendarProvider
from apexfx.features import BaseFeatureExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Maps currency to whether positive surprise is bullish (+1) or bearish (-1)
# for the currency itself (not the pair)
_CURRENCY_DIRECTION: dict[str, int] = {
    "USD": 1, "EUR": 1, "GBP": 1, "JPY": 1,
    "AUD": 1, "NZD": 1, "CAD": 1, "CHF": 1,
}


class FundamentalExtractor(BaseFeatureExtractor):
    """Extracts 8 features from economic calendar data.

    Features:
        news_surprise_score: Rolling weighted surprise from recent events (24h decay)
        news_impact_active: 1.0 if high-impact event in last 30min
        time_to_next_event: Normalized time to next high-impact event [0, 1]
        fundamental_bias: EMA-smoothed accumulated directional bias
        rate_differential: Interest rate diff between base/quote currency
        hawkish_dovish_score: Net hawkish (+1) vs dovish (-1) signal
        event_volatility_ratio: Post-event ATR / pre-event ATR
        conflicting_signals: 1.0 if recent events have opposing bias
    """

    def __init__(
        self,
        base_currency: str = "EUR",
        quote_currency: str = "USD",
        impact_window_min: int = 30,
        surprise_decay_hours: int = 24,
        bias_ema_halflife: int = 48,  # bars
        base_rate: float = 0.0,  # will be overridden
        quote_rate: float = 0.0,
    ) -> None:
        self._base_ccy = base_currency.upper()
        self._quote_ccy = quote_currency.upper()
        self._currencies = [self._base_ccy, self._quote_ccy]
        self._impact_window = timedelta(minutes=impact_window_min)
        self._surprise_decay_hours = surprise_decay_hours
        self._bias_halflife = bias_ema_halflife
        self._base_rate = base_rate
        self._quote_rate = quote_rate

        self._calendar: CalendarProvider | None = None
        self._events: list[CalendarEvent] = []

    def set_calendar(self, calendar: CalendarProvider) -> None:
        """Inject calendar provider."""
        self._calendar = calendar

    def set_events(self, events: list[CalendarEvent]) -> None:
        """Inject events directly (for testing or live).

        Also computes surprise scores if they haven't been set
        (i.e. events were not loaded through CalendarProvider).
        """
        self._events = sorted(events, key=lambda e: e.time_utc)
        # Compute surprise scores if all are at default (0.0)
        needs_scoring = any(e.has_data for e in self._events) and all(
            e.surprise_score == 0.0 for e in self._events if e.has_data
        )
        if needs_scoring:
            self._compute_surprise_scores()

    def _compute_surprise_scores(self) -> None:
        """Compute normalized surprise scores (mirrors CalendarProvider logic)."""
        surprises_by_name: dict[str, list[float]] = {}
        for event in self._events:
            if event.has_data:
                surprises_by_name.setdefault(event.name, []).append(event.raw_surprise)

        stds: dict[str, float] = {}
        for name, surprises in surprises_by_name.items():
            if len(surprises) >= 3:
                std = float(np.std(surprises))
                stds[name] = max(std, 1e-10)
            else:
                stds[name] = abs(float(np.mean(surprises))) + 1e-10 if surprises else 1.0

        for event in self._events:
            if event.has_data and event.name in stds:
                event.surprise_score = event.raw_surprise / stds[event.name]

    def set_rates(self, base_rate: float, quote_rate: float) -> None:
        """Set interest rates for rate differential computation."""
        self._base_rate = base_rate
        self._quote_rate = quote_rate

    @property
    def feature_names(self) -> list[str]:
        return [
            "news_surprise_score",
            "news_impact_active",
            "time_to_next_event",
            "fundamental_bias",
            "rate_differential",
            "hawkish_dovish_score",
            "event_volatility_ratio",
            "conflicting_signals",
        ]

    def extract(
        self,
        bars: pd.DataFrame,
        ticks: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        n = len(bars)
        result = pd.DataFrame(index=bars.index)
        for col in self.feature_names:
            result[col] = 0.0

        # Need time column for calendar matching
        if "time" not in bars.columns:
            logger.debug("No 'time' column — fundamental features will be zero")
            return result

        # Get all events for the data range
        events = self._get_events_for_range(bars)
        if not events:
            # Still compute rate differential even without events
            result["rate_differential"] = self._base_rate - self._quote_rate
            return result

        # Precompute ATR for volatility ratio
        atr_values = self._compute_atr(bars)

        # Running state for bias accumulation
        bias_ema = 0.0
        alpha = 2.0 / (self._bias_halflife + 1)

        for i in range(n):
            row = bars.iloc[i]
            try:
                bar_time = pd.Timestamp(row["time"])
                if bar_time.tzinfo is None:
                    bar_time = bar_time.tz_localize("UTC")
                bar_dt = bar_time.to_pydatetime()
            except Exception:
                continue

            # 1. News surprise score (decaying sum of recent surprises)
            surprise = self._compute_surprise_score(bar_dt, events)
            result.iloc[i, result.columns.get_loc("news_surprise_score")] = surprise

            # 2. News impact active (high-impact event in last N minutes)
            impact_active = self._is_impact_active(bar_dt, events)
            result.iloc[i, result.columns.get_loc("news_impact_active")] = impact_active

            # 3. Time to next event (normalized)
            time_to_next = self._time_to_next_event(bar_dt, events)
            result.iloc[i, result.columns.get_loc("time_to_next_event")] = time_to_next

            # 4. Fundamental bias (EMA of directional surprises)
            directional = self._compute_directional_surprise(bar_dt, events)
            bias_ema = alpha * directional + (1 - alpha) * bias_ema
            result.iloc[i, result.columns.get_loc("fundamental_bias")] = np.clip(bias_ema, -3.0, 3.0)

            # 5. Rate differential
            result.iloc[i, result.columns.get_loc("rate_differential")] = (
                self._base_rate - self._quote_rate
            )

            # 6. Hawkish/dovish score
            hawk_dove = self._compute_hawkish_dovish(bar_dt, events)
            result.iloc[i, result.columns.get_loc("hawkish_dovish_score")] = hawk_dove

            # 7. Event volatility ratio
            if i >= 14 and atr_values is not None:
                vol_ratio = self._compute_event_vol_ratio(bar_dt, events, atr_values, i)
                result.iloc[i, result.columns.get_loc("event_volatility_ratio")] = vol_ratio

            # 8. Conflicting signals
            conflict = self._compute_conflicting_signals(bar_dt, events)
            result.iloc[i, result.columns.get_loc("conflicting_signals")] = conflict

        return result

    def _get_events_for_range(self, bars: pd.DataFrame) -> list[CalendarEvent]:
        """Get relevant events for the data range."""
        if self._events:
            return self._events

        if self._calendar is None:
            return []

        try:
            start = pd.Timestamp(bars.iloc[0]["time"])
            end = pd.Timestamp(bars.iloc[-1]["time"])
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            if end.tzinfo is None:
                end = end.tz_localize("UTC")

            # Expand range to catch pre/post event effects
            start_dt = (start - timedelta(days=1)).to_pydatetime()
            end_dt = (end + timedelta(days=1)).to_pydatetime()

            return self._calendar.get_events(
                start_dt, end_dt,
                currencies=self._currencies,
                impact=None,  # include all, we filter per-feature
            )
        except Exception as e:
            logger.debug("Failed to get calendar events", error=str(e))
            return []

    def _compute_surprise_score(
        self, bar_dt: datetime, events: list[CalendarEvent],
    ) -> float:
        """Rolling weighted surprise with exponential decay over 24h."""
        total = 0.0
        decay_secs = self._surprise_decay_hours * 3600

        for event in reversed(events):
            if event.time_utc > bar_dt:
                continue
            elapsed = (bar_dt - event.time_utc).total_seconds()
            if elapsed > decay_secs:
                break
            if not event.has_data or event.impact != "high":
                continue

            decay = np.exp(-elapsed / (decay_secs / 3))  # 3 half-lives in window
            total += event.surprise_score * decay

        return float(np.clip(total, -5.0, 5.0))

    def _is_impact_active(
        self, bar_dt: datetime, events: list[CalendarEvent],
    ) -> float:
        """1.0 if high-impact event occurred in the last impact_window minutes."""
        for event in reversed(events):
            if event.time_utc > bar_dt:
                continue
            if event.impact != "high":
                continue
            elapsed = bar_dt - event.time_utc
            if elapsed <= self._impact_window:
                return 1.0
            if elapsed > timedelta(hours=2):
                break
        return 0.0

    def _time_to_next_event(
        self, bar_dt: datetime, events: list[CalendarEvent],
    ) -> float:
        """Normalized time to next high-impact event. 0 = imminent, 1 = far away."""
        for event in events:
            if event.time_utc <= bar_dt:
                continue
            if event.impact != "high":
                continue
            if event.currency not in self._currencies:
                continue
            minutes = (event.time_utc - bar_dt).total_seconds() / 60
            return float(np.clip(minutes / 1440.0, 0.0, 1.0))  # normalize by 1 day
        return 1.0  # no upcoming event = far away

    def _compute_directional_surprise(
        self, bar_dt: datetime, events: list[CalendarEvent],
    ) -> float:
        """Directional surprise for the pair: base vs quote currency impact."""
        score = 0.0
        lookback = timedelta(hours=self._surprise_decay_hours)

        for event in reversed(events):
            if event.time_utc > bar_dt:
                continue
            if (bar_dt - event.time_utc) > lookback:
                break
            if not event.has_data or event.impact != "high":
                continue

            elapsed_h = (bar_dt - event.time_utc).total_seconds() / 3600
            decay = np.exp(-elapsed_h / 8)  # 8-hour half-life

            # Positive surprise in base currency → pair goes up
            # Positive surprise in quote currency → pair goes down
            if event.currency == self._base_ccy:
                score += event.surprise_score * decay
            elif event.currency == self._quote_ccy:
                score -= event.surprise_score * decay

        return float(score)

    def _compute_hawkish_dovish(
        self, bar_dt: datetime, events: list[CalendarEvent],
    ) -> float:
        """Net hawkish/dovish score from recent events. +1 hawkish, -1 dovish."""
        score = 0.0
        lookback = timedelta(hours=48)

        for event in reversed(events):
            if event.time_utc > bar_dt:
                continue
            if (bar_dt - event.time_utc) > lookback:
                break
            if not event.has_data:
                continue

            elapsed_h = (bar_dt - event.time_utc).total_seconds() / 3600
            decay = np.exp(-elapsed_h / 24)

            direction = 1.0 if event.is_hawkish else -1.0
            weight = 1.0 if event.impact == "high" else 0.3

            if event.currency == self._base_ccy:
                score += direction * weight * decay
            elif event.currency == self._quote_ccy:
                score -= direction * weight * decay

        return float(np.clip(score, -2.0, 2.0))

    def _compute_event_vol_ratio(
        self,
        bar_dt: datetime,
        events: list[CalendarEvent],
        atr_values: np.ndarray,
        bar_idx: int,
    ) -> float:
        """Ratio of post-event volatility to pre-event baseline."""
        # Check if a high-impact event happened in the last few bars
        for event in reversed(events):
            if event.time_utc > bar_dt:
                continue
            if event.impact != "high":
                continue
            elapsed_h = (bar_dt - event.time_utc).total_seconds() / 3600
            if elapsed_h > 4:
                break

            # Compare current ATR to rolling baseline
            if bar_idx >= 14 and not np.isnan(atr_values[bar_idx]):
                baseline = np.nanmean(atr_values[max(0, bar_idx - 14):bar_idx])
                if baseline > 0:
                    ratio = atr_values[bar_idx] / baseline
                    return float(np.clip(ratio, 0.0, 5.0))

        return 1.0  # no event → ratio = 1 (normal)

    def _compute_conflicting_signals(
        self, bar_dt: datetime, events: list[CalendarEvent],
    ) -> float:
        """1.0 if recent events give opposing signals (uncertainty)."""
        lookback = timedelta(hours=24)
        directions: list[float] = []

        for event in reversed(events):
            if event.time_utc > bar_dt:
                continue
            if (bar_dt - event.time_utc) > lookback:
                break
            if not event.has_data or event.impact != "high":
                continue

            d = 1.0 if event.is_hawkish else -1.0
            if event.currency == self._quote_ccy:
                d = -d  # quote currency has inverse effect on pair
            directions.append(d)

        if len(directions) < 2:
            return 0.0

        # If directions have both positive and negative → conflicting
        has_positive = any(d > 0 for d in directions)
        has_negative = any(d < 0 for d in directions)
        return 1.0 if (has_positive and has_negative) else 0.0

    @staticmethod
    def _compute_atr(bars: pd.DataFrame, period: int = 14) -> np.ndarray | None:
        """Compute ATR from OHLC."""
        if not all(c in bars.columns for c in ("high", "low", "close")):
            return None

        high = bars["high"].values.astype(float)
        low = bars["low"].values.astype(float)
        close = bars["close"].values.astype(float)

        n = len(high)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr_vals = np.full(n, np.nan)
        if n >= period:
            atr_vals[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period

        return atr_vals
