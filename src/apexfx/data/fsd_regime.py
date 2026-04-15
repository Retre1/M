"""FSD (Functional Skewness Dispersion) regime detection.

Based on: "Functional Skewness Dispersion and the Cross-Section of Stock Returns"

Computes rolling skewness dispersion across feature cross-sections to identify
market regime transitions (risk-on / risk-off / neutral). The FSD signal is
appended as additional features to the observation space.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd
from scipy.stats import skew

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class FSDRegime(IntEnum):
    """Market regime classified by skewness dispersion."""

    RISK_OFF = 0
    NEUTRAL = 1
    RISK_ON = 2


@dataclass(frozen=True)
class FSDConfig:
    """Configuration for FSD regime detector."""

    window: int = 252
    n_quantiles: int = 5
    update_freq: int = 20


class FSDRegimeDetector:
    """Detect market regimes via functional skewness dispersion.

    The detector computes rolling skewness for each feature column, then
    measures the cross-sectional dispersion of those skewness values.
    High dispersion → regime transition / stress. Low dispersion → stable regime.
    """

    def __init__(self, config: FSDConfig | None = None) -> None:
        self.config = config or FSDConfig()
        self._cache_step: int = -1
        self._cached_regime: FSDRegime = FSDRegime.NEUTRAL
        self._cached_dispersion: float = 0.0

    def compute_skewness_dispersion(self, returns: pd.DataFrame) -> pd.Series:
        """Compute rolling cross-sectional skewness dispersion.

        Args:
            returns: DataFrame of asset/feature returns, shape (T, N).

        Returns:
            Series of skewness dispersion values, length T.
        """
        window = self.config.window
        n = len(returns)

        if n < window:
            logger.warning("Insufficient data for FSD", n_bars=n, window=window)
            return pd.Series(np.zeros(n), index=returns.index)

        dispersions = np.full(n, np.nan)

        for i in range(window, n):
            chunk = returns.iloc[i - window : i]
            col_skew = chunk.apply(lambda c: skew(c.dropna(), bias=False))
            dispersions[i] = col_skew.std()

        result = pd.Series(dispersions, index=returns.index)
        return result.ffill().fillna(0.0)

    def classify_regime(
        self, dispersion: float, thresholds: tuple[float, float] = (0.3, 0.7)
    ) -> FSDRegime:
        """Classify regime based on dispersion quantile.

        Args:
            dispersion: Current skewness dispersion value.
            thresholds: (low, high) quantile thresholds.

        Returns:
            FSDRegime enum value.
        """
        low, high = thresholds
        if dispersion < low:
            return FSDRegime.RISK_ON
        elif dispersion > high:
            return FSDRegime.RISK_OFF
        return FSDRegime.NEUTRAL

    def get_features(self, returns: pd.DataFrame, step: int) -> np.ndarray:
        """Get FSD features for the current step.

        Returns a 3-element array: [dispersion, regime_onehot_0, regime_onehot_1].
        """
        if step == self._cache_step:
            regime = self._cached_regime
            disp = self._cached_dispersion
        else:
            disp_series = self.compute_skewness_dispersion(returns.iloc[: step + 1])
            disp = float(disp_series.iloc[-1]) if len(disp_series) > 0 else 0.0
            regime = self.classify_regime(disp)
            self._cache_step = step
            self._cached_regime = regime
            self._cached_dispersion = disp

        onehot = np.zeros(3, dtype=np.float32)
        onehot[regime] = 1.0
        return np.concatenate([[disp], onehot])
