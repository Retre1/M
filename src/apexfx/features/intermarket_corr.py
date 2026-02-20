"""Rolling intermarket correlation features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor


class IntermarketCorrExtractor(BaseFeatureExtractor):
    """
    Computes rolling Pearson and Spearman correlations between the target
    pair and intermarket instruments (DXY, Gold, US Treasuries).
    """

    def __init__(
        self,
        instruments: list[str] | None = None,
        windows: list[int] | None = None,
    ) -> None:
        self._instruments = instruments or ["DXY", "XAUUSD", "US10Y"]
        self._windows = windows or [20, 60, 252]

    @property
    def feature_names(self) -> list[str]:
        names = []
        for inst in self._instruments:
            for w in self._windows:
                names.append(f"corr_{inst}_{w}")
                names.append(f"rank_corr_{inst}_{w}")
            names.append(f"beta_{inst}")
        return names

    def extract(self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None) -> pd.DataFrame:
        result = pd.DataFrame(index=bars.index)
        for col in self.feature_names:
            result[col] = np.nan

        close = bars["close"].values
        log_returns_target = np.diff(np.log(close), prepend=0)

        for inst in self._instruments:
            col_name = f"{inst}_close"
            if col_name not in bars.columns:
                continue

            inst_close = bars[col_name].values
            if np.all(np.isnan(inst_close)) or np.all(inst_close == 0):
                continue

            log_returns_inst = np.diff(np.log(np.maximum(inst_close, 1e-10)), prepend=0)

            for w in self._windows:
                pearson = self._rolling_correlation(log_returns_target, log_returns_inst, w)
                spearman = self._rolling_rank_correlation(log_returns_target, log_returns_inst, w)
                result[f"corr_{inst}_{w}"] = pearson
                result[f"rank_corr_{inst}_{w}"] = spearman

            # Rolling beta (regression coefficient)
            result[f"beta_{inst}"] = self._rolling_beta(
                log_returns_target, log_returns_inst, max(self._windows)
            )

        return result

    @staticmethod
    def _rolling_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Rolling Pearson correlation."""
        n = len(x)
        result = np.full(n, np.nan)

        for i in range(window, n):
            x_w = x[i - window : i]
            y_w = y[i - window : i]

            x_std = np.std(x_w, ddof=1)
            y_std = np.std(y_w, ddof=1)

            if x_std < 1e-10 or y_std < 1e-10:
                result[i] = 0.0
                continue

            cov = np.mean((x_w - np.mean(x_w)) * (y_w - np.mean(y_w)))
            result[i] = cov / (x_std * y_std)

        return result

    @staticmethod
    def _rolling_rank_correlation(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Rolling Spearman rank correlation."""
        n = len(x)
        result = np.full(n, np.nan)

        for i in range(window, n):
            x_w = x[i - window : i]
            y_w = y[i - window : i]

            # Rank the values
            x_ranks = np.argsort(np.argsort(x_w)).astype(float)
            y_ranks = np.argsort(np.argsort(y_w)).astype(float)

            x_std = np.std(x_ranks, ddof=1)
            y_std = np.std(y_ranks, ddof=1)

            if x_std < 1e-10 or y_std < 1e-10:
                result[i] = 0.0
                continue

            cov = np.mean((x_ranks - np.mean(x_ranks)) * (y_ranks - np.mean(y_ranks)))
            result[i] = cov / (x_std * y_std)

        return result

    @staticmethod
    def _rolling_beta(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        """Rolling regression beta: x = alpha + beta * y."""
        n = len(x)
        result = np.full(n, np.nan)

        for i in range(window, n):
            x_w = x[i - window : i]
            y_w = y[i - window : i]

            y_var = np.var(y_w, ddof=1)
            if y_var < 1e-10:
                result[i] = 0.0
                continue

            cov = np.cov(x_w, y_w, ddof=1)[0, 1]
            result[i] = cov / y_var

        return result
