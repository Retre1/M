"""Multi-Timeframe synthetic data generation.

Generates aligned D1/H1/M5 DataFrames from a single M5-resolution
synthetic price series via resampling. Ensures timestamp consistency
across all three timeframes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apexfx.data.synthetic import SyntheticDataGenerator, RegimeParams
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MTFSyntheticData:
    """Container for aligned multi-timeframe synthetic data."""

    d1: pd.DataFrame
    h1: pd.DataFrame
    m5: pd.DataFrame


class MTFSyntheticGenerator:
    """Generate aligned synthetic D1/H1/M5 data.

    Strategy:
    1. Generate a high-resolution M5 series (base timeframe)
    2. Resample to H1 (12 M5 bars = 1 H1 bar)
    3. Resample to D1 (24 H1 bars = 1 D1 bar, or 288 M5 bars)

    This ensures all timeframes are perfectly aligned in time and
    reflect the same underlying price dynamics.
    """

    def __init__(self, seed: int = 42) -> None:
        self._generator = SyntheticDataGenerator(seed=seed)
        self._rng = np.random.default_rng(seed)

    def generate(
        self,
        n_h1_bars: int = 5000,
        mu: float = 0.0001,
        sigma: float = 0.01,
        noise_std: float = 0.0,
        black_swan_intensity: float = 0.0,
        black_swan_magnitude_std: float = 5.0,
        initial_price: float = 1.1000,
    ) -> MTFSyntheticData:
        """Generate aligned MTF synthetic data.

        Parameters
        ----------
        n_h1_bars : int
            Number of H1 bars to generate (primary timeframe).
            M5 bars = n_h1_bars * 12, D1 bars = n_h1_bars / 24.
        mu, sigma : float
            GBM drift and volatility.
        noise_std : float
            Additional Gaussian noise (0 = clean).
        black_swan_intensity : float
            Probability of black swan per M5 bar.
        black_swan_magnitude_std : float
            Magnitude of black swan shocks.
        initial_price : float
            Starting price.

        Returns
        -------
        MTFSyntheticData
            Aligned D1, H1, M5 DataFrames.
        """
        # Generate at M5 resolution (12 M5 bars = 1 H1 bar)
        n_m5_bars = n_h1_bars * 12

        logger.info(
            "Generating MTF synthetic data",
            n_m5_bars=n_m5_bars,
            n_h1_bars=n_h1_bars,
            n_d1_bars=n_h1_bars // 24,
        )

        # Use regime switching for realistic dynamics at M5 level
        m5_data = self._generator.generate_regime_switching(
            n_steps=n_m5_bars,
            initial_price=initial_price,
            regimes=[
                RegimeParams(mu=0.001, sigma=0.008, duration_mean=2400, name="trend_up"),
                RegimeParams(mu=-0.001, sigma=0.008, duration_mean=2400, name="trend_down"),
                RegimeParams(mu=0.0, sigma=0.004, duration_mean=3600, name="flat"),
                RegimeParams(mu=0.0, sigma=0.012, duration_mean=1800, name="volatile"),
            ],
            dt=1.0 / (252 * 24 * 12),  # M5 resolution
        )

        # Set proper M5 timestamps (5 min intervals)
        m5_data["time"] = pd.date_range(
            start="2023-01-02 00:00:00",
            periods=len(m5_data),
            freq="5min",
            tz="UTC",
        )

        # Apply noise if requested
        if noise_std > 0:
            m5_data = self._generator.add_noise(m5_data, noise_std)

        # Apply black swans at M5 level
        if black_swan_intensity > 0:
            m5_data = self._generator.inject_black_swans(
                m5_data, black_swan_intensity, black_swan_magnitude_std,
            )

        # Resample M5 → H1
        h1_data = self._resample(m5_data, "1h")

        # Resample M5 → D1
        d1_data = self._resample(m5_data, "1D")

        logger.info(
            "MTF synthetic data ready",
            d1_bars=len(d1_data),
            h1_bars=len(h1_data),
            m5_bars=len(m5_data),
        )

        return MTFSyntheticData(d1=d1_data, h1=h1_data, m5=m5_data)

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample OHLCV data to a lower frequency.

        Uses proper OHLC aggregation:
        - open: first
        - high: max
        - low: min
        - close: last
        - volume: sum
        """
        # Set time as index for resampling
        temp = df.set_index("time")

        agg = temp.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })

        # Drop incomplete bars (NaN)
        agg = agg.dropna()

        # Tick count: use count of source bars
        tick_counts = temp.resample(rule).size()
        agg["tick_count"] = tick_counts.loc[agg.index].values

        # Copy regime label if present
        if "regime" in temp.columns:
            # Take the mode (most common regime in the window)
            regime_mode = temp["regime"].resample(rule).apply(
                lambda x: x.mode().iloc[0] if len(x) > 0 else 0
            )
            agg["regime"] = regime_mode.loc[agg.index].values

        # Reset index to get time column back
        result = agg.reset_index()

        return result


def resample_real_data(
    h1_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample real H1 data to D1 and approximate M5.

    For real data where we only have H1:
    - D1: Proper OHLCV resample from H1
    - M5: Interpolated from H1 (synthetic intra-hour structure)

    Parameters
    ----------
    h1_data : pd.DataFrame
        Real H1 bars with 'time' column.

    Returns
    -------
    tuple
        (d1_data, m5_data) DataFrames
    """
    # H1 → D1: proper resample
    d1_data = MTFSyntheticGenerator._resample(h1_data, "1D")

    # H1 → M5: interpolate (each H1 bar → 12 M5 bars)
    m5_rows = []
    rng = np.random.default_rng(42)

    for idx in range(len(h1_data)):
        row = h1_data.iloc[idx]
        h1_open = row["open"]
        h1_close = row["close"]
        h1_high = row["high"]
        h1_low = row["low"]
        h1_vol = row["volume"] if "volume" in row.index else 1000
        h1_time = pd.Timestamp(row["time"])

        # Generate 12 M5 bars within this H1 bar
        # Use random walk from open to close, constrained by high/low
        n_sub = 12
        prices = np.linspace(h1_open, h1_close, n_sub + 1)
        noise = rng.normal(0, abs(h1_close - h1_open) * 0.1 + 1e-8, n_sub + 1)
        prices = prices + noise
        prices[0] = h1_open
        prices[-1] = h1_close

        # Clamp to H1 high/low
        prices = np.clip(prices, h1_low, h1_high)

        for j in range(n_sub):
            m5_time = h1_time + pd.Timedelta(minutes=5 * j)
            m5_open = prices[j]
            m5_close = prices[j + 1]
            m5_high = max(m5_open, m5_close) * (1 + rng.uniform(0, 0.0005))
            m5_low = min(m5_open, m5_close) * (1 - rng.uniform(0, 0.0005))
            m5_high = min(m5_high, h1_high)
            m5_low = max(m5_low, h1_low)
            m5_vol = max(1, h1_vol / n_sub * rng.lognormal(0, 0.3))

            m5_rows.append({
                "time": m5_time,
                "open": m5_open,
                "high": m5_high,
                "low": m5_low,
                "close": m5_close,
                "volume": int(m5_vol),
                "tick_count": max(1, int(rng.integers(10, 100))),
            })

    m5_data = pd.DataFrame(m5_rows)

    logger.info(
        "Real data resampled to MTF",
        d1_bars=len(d1_data),
        h1_bars=len(h1_data),
        m5_bars=len(m5_data),
    )

    return d1_data, m5_data
