"""Synthetic market data generation for curriculum learning.

Includes GARCH-like volatility clustering, realistic volume autocorrelation,
session-dependent activity, and clustered black swan events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeParams:
    """Parameters for a single market regime."""
    mu: float  # drift
    sigma: float  # volatility
    duration_mean: int  # average bars in this regime
    name: str = "neutral"


@dataclass
class GARCHParams:
    """GARCH(1,1) parameters for volatility clustering."""
    omega: float = 2e-6       # long-run variance weight
    alpha: float = 0.08       # shock impact (ARCH)
    beta: float = 0.90        # persistence (GARCH)
    initial_var: float = 1e-4  # starting variance


class SyntheticDataGenerator:
    """Generate synthetic Forex-like data with configurable properties.

    Improvements over basic GBM:
    - GARCH(1,1) volatility clustering
    - Session-dependent volume patterns (Tokyo/London/NY)
    - Volume autocorrelation
    - Clustered black swans (crisis chains)
    - Support/resistance level generation for breakout training
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_gbm(
        self,
        n_steps: int,
        initial_price: float = 1.1000,
        mu: float = 0.0001,
        sigma: float = 0.01,
        dt: float = 1.0 / (252 * 24),  # Hourly
    ) -> pd.DataFrame:
        """Generate price series via Geometric Brownian Motion."""
        z_samples = self._rng.standard_normal(n_steps)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_samples

        log_prices = np.zeros(n_steps + 1)
        log_prices[0] = np.log(initial_price)
        log_prices[1:] = log_prices[0] + np.cumsum(log_returns)

        prices = np.exp(log_prices)
        return self._prices_to_ohlcv(prices)

    def generate_garch(
        self,
        n_steps: int,
        initial_price: float = 1.1000,
        mu: float = 0.0,
        garch: GARCHParams | None = None,
        dt: float = 1.0 / (252 * 24),
    ) -> pd.DataFrame:
        """Generate price series with GARCH(1,1) volatility clustering.

        Unlike GBM, this produces realistic volatility clusters where
        high-vol periods follow high-vol periods (like real markets).
        """
        garch = garch or GARCHParams()

        z_samples = self._rng.standard_normal(n_steps)
        log_returns = np.zeros(n_steps)
        variances = np.zeros(n_steps)
        variances[0] = garch.initial_var

        for t in range(n_steps):
            sigma_t = np.sqrt(variances[t])
            log_returns[t] = (mu - 0.5 * variances[t]) * dt + sigma_t * np.sqrt(dt) * z_samples[t]
            if t < n_steps - 1:
                variances[t + 1] = (
                    garch.omega
                    + garch.alpha * log_returns[t] ** 2
                    + garch.beta * variances[t]
                )

        log_prices = np.zeros(n_steps + 1)
        log_prices[0] = np.log(initial_price)
        log_prices[1:] = log_prices[0] + np.cumsum(log_returns)

        prices = np.exp(log_prices)
        df = self._prices_to_ohlcv(prices)
        df["realized_vol"] = np.concatenate([[np.nan], np.sqrt(variances)])[:len(df)]
        return df

    def generate_regime_switching(
        self,
        n_steps: int,
        initial_price: float = 1.1000,
        regimes: list[RegimeParams] | None = None,
        dt: float = 1.0 / (252 * 24),
        use_garch: bool = True,
    ) -> pd.DataFrame:
        """Generate data with alternating market regimes.

        When use_garch=True, volatility within each regime follows GARCH(1,1)
        dynamics, creating realistic vol clustering.
        """
        if regimes is None:
            regimes = [
                RegimeParams(mu=0.001, sigma=0.008, duration_mean=200, name="trend_up"),
                RegimeParams(mu=-0.001, sigma=0.008, duration_mean=200, name="trend_down"),
                RegimeParams(mu=0.0, sigma=0.015, duration_mean=150, name="volatile"),
                RegimeParams(mu=0.0, sigma=0.005, duration_mean=300, name="flat"),
            ]

        log_returns = np.zeros(n_steps)
        regime_labels = np.zeros(n_steps, dtype=int)
        current_step = 0

        # GARCH state persists across regimes for smooth vol transitions
        current_var = 1e-4

        while current_step < n_steps:
            regime_idx = self._rng.integers(0, len(regimes))
            regime = regimes[regime_idx]
            duration = int(self._rng.exponential(regime.duration_mean))
            duration = max(10, min(duration, n_steps - current_step))

            end_step = current_step + duration
            z_samples = self._rng.standard_normal(duration)

            if use_garch:
                # GARCH within regime — vol targets regime.sigma but clusters
                target_var = regime.sigma ** 2
                for i in range(duration):
                    t = current_step + i
                    sigma_t = np.sqrt(current_var)
                    log_returns[t] = (
                        (regime.mu - 0.5 * current_var) * dt + sigma_t * np.sqrt(dt) * z_samples[i]
                    )
                    # GARCH update with mean-reversion to regime target
                    current_var = (
                        0.02 * target_var
                        + 0.08 * log_returns[t] ** 2
                        + 0.90 * current_var
                    )
            else:
                log_returns[current_step:end_step] = (
                    (regime.mu - 0.5 * regime.sigma**2) * dt
                    + regime.sigma * np.sqrt(dt) * z_samples
                )

            regime_labels[current_step:end_step] = regime_idx
            current_step = end_step

        log_prices = np.zeros(n_steps + 1)
        log_prices[0] = np.log(initial_price)
        log_prices[1:] = log_prices[0] + np.cumsum(log_returns)

        prices = np.exp(log_prices)
        df = self._prices_to_ohlcv(prices, session_volumes=True)
        df["regime"] = np.concatenate([[regime_labels[0]], regime_labels])[:len(df)]
        return df

    def inject_black_swans(
        self,
        df: pd.DataFrame,
        intensity: float = 0.001,
        magnitude_std: float = 5.0,
        cluster_prob: float = 0.4,
        cluster_decay: float = 0.7,
    ) -> pd.DataFrame:
        """Inject fat-tail events (jumps) into price data.

        Now supports clustered events: a black swan can trigger follow-up
        shocks with decaying magnitude (like real crises where one event
        causes a chain reaction).

        Args:
            intensity: Probability of a black swan per bar.
            magnitude_std: Standard deviation of shock magnitude.
            cluster_prob: Probability that a shock triggers a follow-up.
            cluster_decay: Magnitude decay factor for follow-up shocks.
        """
        if intensity <= 0:
            return df

        result = df.copy()
        n = len(result)

        # Poisson process for initial event times
        n_events = self._rng.poisson(intensity * n)
        if n_events == 0:
            return result

        event_indices = self._rng.integers(1, n, size=n_events)
        event_magnitudes = self._rng.standard_normal(n_events) * magnitude_std

        # Build full event list including clusters
        all_events: list[tuple[int, float]] = []
        for idx, magnitude in zip(event_indices, event_magnitudes, strict=False):
            all_events.append((int(idx), float(magnitude)))

            # Cluster generation: same direction, decaying magnitude
            current_mag = magnitude
            current_idx = int(idx)
            while self._rng.random() < cluster_prob and current_idx + 1 < n:
                current_mag *= cluster_decay
                current_idx += self._rng.integers(1, 5)  # 1-4 bars later
                if current_idx < n:
                    all_events.append((current_idx, current_mag))

        # Sort by index and apply
        all_events.sort(key=lambda x: x[0])
        for idx, magnitude in all_events:
            shock_factor = 1 + magnitude * 0.001
            result.loc[idx:, "close"] *= shock_factor
            result.loc[idx:, "open"] *= shock_factor
            result.loc[idx:, "high"] *= shock_factor
            result.loc[idx:, "low"] *= shock_factor

            if magnitude > 0:
                result.loc[idx, "high"] = result.loc[idx, "close"] * 1.002
            else:
                result.loc[idx, "low"] = result.loc[idx, "close"] * 0.998

            result.loc[idx, "volume"] = int(result.loc[idx, "volume"] * max(1.0, abs(magnitude)))

        return result

    def add_noise(self, df: pd.DataFrame, noise_std: float = 0.005) -> pd.DataFrame:
        """Add Gaussian noise to price data."""
        if noise_std <= 0:
            return df

        result = df.copy()
        n = len(result)

        for col in ["open", "high", "low", "close"]:
            noise = self._rng.normal(0, noise_std, n) * result[col].values
            result[col] = result[col] + noise

        # Ensure high >= max(open, close) and low <= min(open, close)
        result["high"] = result[["open", "high", "close"]].max(axis=1)
        result["low"] = result[["open", "low", "close"]].min(axis=1)

        return result

    def generate_spread(
        self,
        n: int,
        base_spread_pips: float = 1.0,
        pip_value: float = 0.0001,
    ) -> np.ndarray:
        """Generate realistic spread series with session-dependent widening."""
        base = base_spread_pips * pip_value
        # Log-normal spread variation
        spreads = base * self._rng.lognormal(0, 0.3, n)
        return np.clip(spreads, base * 0.5, base * 10)

    def generate_support_resistance(
        self,
        df: pd.DataFrame,
        n_levels: int = 5,
    ) -> pd.DataFrame:
        """Add synthetic support/resistance levels for breakout training.

        Creates horizontal price levels that price tends to bounce off,
        with occasional breakouts through them. Useful for training
        the BreakoutAgent.
        """
        result = df.copy()
        close = result["close"].values
        n = len(close)

        # Find local extremes as level candidates
        window = max(20, n // (n_levels * 4))
        levels: list[float] = []
        for i in range(window, n - window, window):
            local_high = np.max(close[i - window : i + window])
            local_low = np.min(close[i - window : i + window])
            levels.extend([local_high, local_low])

        # Deduplicate (merge levels within 0.1% of each other)
        levels = sorted(set(levels))
        merged: list[float] = []
        for level in levels:
            if not merged or abs(level - merged[-1]) / merged[-1] > 0.001:
                merged.append(level)

        # Keep top n_levels
        if len(merged) > n_levels:
            indices = np.linspace(0, len(merged) - 1, n_levels, dtype=int)
            merged = [merged[i] for i in indices]

        # Compute distances to nearest support/resistance
        support_dist = np.full(n, np.nan)
        resist_dist = np.full(n, np.nan)
        for i in range(n):
            price = close[i]
            below = [level for level in merged if level < price]
            above = [level for level in merged if level > price]
            if below:
                support_dist[i] = (price - max(below)) / price
            if above:
                resist_dist[i] = (min(above) - price) / price

        result["nearest_support_distance"] = support_dist
        result["nearest_resistance_distance"] = resist_dist
        return result

    def _prices_to_ohlcv(
        self,
        prices: np.ndarray,
        session_volumes: bool = False,
    ) -> pd.DataFrame:
        """Convert a price series into OHLCV DataFrame.

        When session_volumes=True, volume follows realistic session patterns
        with autocorrelation (London/NY overlap = highest volume).
        """
        n = len(prices)
        noise_scale = np.std(np.diff(np.log(prices))) * 0.3

        opens = prices.copy()
        closes = prices.copy()
        highs = prices * (1 + np.abs(self._rng.normal(0, noise_scale, n)))
        lows = prices * (1 - np.abs(self._rng.normal(0, noise_scale, n)))

        times = pd.date_range(
            start="2020-01-01", periods=n, freq="h", tz="UTC"
        )

        if session_volumes:
            # Session-dependent base volume (London-NY overlap is peak)
            session_mult = np.ones(n)
            for i in range(n):
                hour = times[i].hour if i < len(times) else 12
                if 12 <= hour <= 16:
                    session_mult[i] = 2.0    # London-NY overlap
                elif 7 <= hour <= 16:
                    session_mult[i] = 1.5    # London
                elif 13 <= hour <= 21:
                    session_mult[i] = 1.3    # New York
                elif 0 <= hour <= 9:
                    session_mult[i] = 0.7    # Tokyo
                else:
                    session_mult[i] = 0.3    # Off hours

            # Autocorrelated volume (vol clusters like real markets)
            base_vol = self._rng.lognormal(10, 1.0, n) * session_mult
            volumes = np.zeros(n)
            volumes[0] = base_vol[0]
            for i in range(1, n):
                volumes[i] = 0.6 * volumes[i - 1] + 0.4 * base_vol[i]
        else:
            volumes = self._rng.lognormal(10, 1.5, n)

        return pd.DataFrame({
            "time": times[:n],
            "open": opens[:n],
            "high": highs[:n],
            "low": lows[:n],
            "close": closes[:n],
            "volume": volumes[:n],
            "tick_count": self._rng.integers(50, 500, n),
        })
