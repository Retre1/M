"""Fourier and Wavelet transforms for cycle decomposition and noise filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

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

    # A Hann window has a coherent gain of 0.5: windowing halves the amplitude
    # of every component. Without dividing it back out, fft_amplitude_N reported
    # exactly half the true amplitude (measured 0.498 for a unit sine).
    _HANN_COHERENT_GAIN = 0.5

    def _compute_fft(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-N spectral *peaks* as (period, amplitude) pairs.

        Peaks, not bins. Taking the N largest bins looked equivalent but is not:
        the Hann window spreads each component over its neighbours, so the
        largest bins cluster around one peak. On a mixture of a period-20
        component (amplitude 1.0) and a period-8 component (amplitude 0.3) the
        old code returned periods 19.69, 21.33 and 18.29 — the same peak three
        times — and never reported the period-8 component at all. Features 2
        and 3 were leakage artefacts rather than independent information.
        """
        n = len(signal)
        # Apply Hann window to reduce spectral leakage
        windowed = signal * np.hanning(n)

        yf = fft(windowed)
        frequencies = fftfreq(n, d=1.0)  # d=1 bar

        # Only positive frequencies, skip DC component
        pos_mask = frequencies > 0
        pos_freq = frequencies[pos_mask]
        pos_amp = 2.0 / n * np.abs(yf[pos_mask]) / self._HANN_COHERENT_GAIN

        if len(pos_amp) == 0:
            return np.array([]), np.array([])

        # Local maxima only, so each reported component is a distinct cycle.
        peak_idx, _ = find_peaks(pos_amp)
        if len(peak_idx) == 0:
            # Monotonic spectrum (very short or heavily damped window) — fall
            # back to the strongest bin so the feature is still populated.
            peak_idx = np.array([int(np.argmax(pos_amp))])

        strongest = peak_idx[np.argsort(pos_amp[peak_idx])[::-1]][: self._top_n]

        periods = 1.0 / pos_freq[strongest]  # frequency -> period in bars
        amplitudes = pos_amp[strongest]

        return periods, amplitudes
