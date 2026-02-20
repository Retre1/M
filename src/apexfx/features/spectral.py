"""Fourier and Wavelet transforms for cycle decomposition and noise filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft, fftfreq

from apexfx.features import BaseFeatureExtractor


class SpectralExtractor(BaseFeatureExtractor):
    """
    Decomposes price series into frequency components via FFT and
    multi-resolution components via Wavelet transform.
    """

    def __init__(
        self,
        fft_window: int = 256,
        top_n_cycles: int = 3,
        wavelet: str = "db4",
        wavelet_level: int = 4,
    ) -> None:
        self._fft_window = fft_window
        self._top_n = top_n_cycles
        self._wavelet = wavelet
        self._wavelet_level = wavelet_level

    @property
    def feature_names(self) -> list[str]:
        names = []
        for i in range(1, self._top_n + 1):
            names.extend([f"fft_period_{i}", f"fft_amplitude_{i}"])
        names.append("fft_dominant_period")
        for i in range(1, self._wavelet_level + 1):
            names.append(f"wavelet_energy_d{i}")
        names.append("wavelet_energy_approx")
        names.append("wavelet_trend")
        return names

    def extract(self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None) -> pd.DataFrame:
        n = len(bars)
        result = pd.DataFrame(index=bars.index)
        for col in self.feature_names:
            result[col] = np.nan

        close = bars["close"].values
        log_close = np.log(close)
        # Detrend: use differenced log prices
        detrended = np.diff(log_close, prepend=log_close[0])

        for i in range(self._fft_window, n):
            window = detrended[i - self._fft_window : i]

            # --- FFT ---
            periods, amplitudes = self._compute_fft(window)

            for j in range(min(self._top_n, len(periods))):
                result.iloc[i, result.columns.get_loc(f"fft_period_{j + 1}")] = periods[j]
                result.iloc[i, result.columns.get_loc(f"fft_amplitude_{j + 1}")] = amplitudes[j]

            if len(periods) > 0:
                result.iloc[i, result.columns.get_loc("fft_dominant_period")] = periods[0]

            # --- Wavelet ---
            try:
                coeffs = pywt.wavedec(window, self._wavelet, level=self._wavelet_level)
            except ValueError:
                continue

            # Approximation coefficient energy
            approx_energy = np.sum(coeffs[0] ** 2) / len(coeffs[0])
            result.iloc[i, result.columns.get_loc("wavelet_energy_approx")] = approx_energy

            # Detail coefficient energies at each level
            for level_idx in range(1, min(len(coeffs), self._wavelet_level + 1)):
                detail = coeffs[level_idx]
                energy = np.sum(detail**2) / len(detail) if len(detail) > 0 else 0
                col_name = f"wavelet_energy_d{level_idx}"
                result.iloc[i, result.columns.get_loc(col_name)] = energy

            # Wavelet trend: reconstruction from approximation coefficients only
            trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            try:
                trend_signal = pywt.waverec(trend_coeffs, self._wavelet)
                # Slope of the trend
                if len(trend_signal) >= 2:
                    result.iloc[i, result.columns.get_loc("wavelet_trend")] = (
                        trend_signal[-1] - trend_signal[-2]
                    )
            except ValueError:
                pass

        return result

    def _compute_fft(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute FFT and return top-N (period, amplitude) pairs sorted by amplitude."""
        n = len(signal)
        # Apply Hann window to reduce spectral leakage
        windowed = signal * np.hanning(n)

        yf = fft(windowed)
        frequencies = fftfreq(n, d=1.0)  # d=1 bar

        # Only positive frequencies, skip DC component
        pos_mask = frequencies > 0
        pos_freq = frequencies[pos_mask]
        pos_amp = 2.0 / n * np.abs(yf[pos_mask])

        if len(pos_amp) == 0:
            return np.array([]), np.array([])

        # Sort by amplitude (descending)
        sorted_idx = np.argsort(pos_amp)[::-1]
        top_idx = sorted_idx[: self._top_n]

        periods = 1.0 / pos_freq[top_idx]  # Convert frequency to period (in bars)
        amplitudes = pos_amp[top_idx]

        return periods, amplitudes
