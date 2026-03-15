"""COT (Commitment of Traders) features — institutional positioning signals.

CFTC COT data shows how commercial hedgers and speculative traders are
positioned in FX futures. Key signals:
- Extreme speculative positioning → contrarian reversal signal
- Commercial hedger positioning → smart money indicator
- Change in open interest → market participation momentum
- Spec net as % of OI → normalized positioning
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.features import BaseFeatureExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class COTExtractor(BaseFeatureExtractor):
    """Extracts trading signals from CFTC COT positioning data.

    7 features per currency pair:
    - cot_spec_net_norm: Speculative net normalized to [-1, 1] (z-score of 52-week range)
    - cot_spec_change: Week-over-week change in spec net
    - cot_comm_net_norm: Commercial net normalized (smart money signal)
    - cot_oi_change: Week-over-week open interest change (participation)
    - cot_spec_extreme: 1 if spec at 52-week extreme (contrarian signal)
    - cot_momentum: 4-week momentum of spec net
    - cot_divergence: Divergence between price trend and positioning
    """

    def __init__(
        self,
        base_currency: str = "EUR",
        quote_currency: str = "USD",
        z_score_lookback: int = 52,  # weeks
        extreme_threshold: float = 2.0,  # z-score for extreme detection
    ) -> None:
        self._base = base_currency
        self._quote = quote_currency
        self._z_lookback = z_score_lookback
        self._extreme_threshold = extreme_threshold

    @property
    def feature_names(self) -> list[str]:
        return [
            "cot_spec_net_norm",
            "cot_spec_change",
            "cot_comm_net_norm",
            "cot_oi_change",
            "cot_spec_extreme",
            "cot_momentum",
            "cot_divergence",
        ]

    def extract(
        self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Extract COT features aligned to bar timestamps.

        Expects bars to have columns: 'cot_spec_net', 'cot_comm_net',
        'cot_open_interest' (forward-filled from weekly COT data).
        If these columns are missing, returns all NaN.
        """
        n = len(bars)
        result = pd.DataFrame(index=bars.index)

        # Initialize with NaN
        for name in self.feature_names:
            result[name] = np.nan

        # Check for COT columns
        required = ["cot_spec_net", "cot_comm_net", "cot_open_interest"]
        if not all(col in bars.columns for col in required):
            return result

        spec_net = bars["cot_spec_net"].values.astype(float)
        comm_net = bars["cot_comm_net"].values.astype(float)
        oi = bars["cot_open_interest"].values.astype(float)

        # 1. Speculative net normalized (z-score over lookback)
        spec_norm = self._rolling_zscore(spec_net, self._z_lookback * 5)  # ~5 bars per week
        result["cot_spec_net_norm"] = np.clip(spec_norm, -3, 3) / 3.0  # Scale to [-1, 1]

        # 2. Spec change (week-over-week, approximated by 5-bar diff)
        spec_diff = np.zeros(n)
        spec_diff[5:] = spec_net[5:] - spec_net[:-5]
        oi_safe = np.maximum(oi, 1)
        result["cot_spec_change"] = spec_diff / oi_safe

        # 3. Commercial net normalized
        comm_norm = self._rolling_zscore(comm_net, self._z_lookback * 5)
        result["cot_comm_net_norm"] = np.clip(comm_norm, -3, 3) / 3.0

        # 4. Open interest change
        oi_diff = np.zeros(n)
        oi_diff[5:] = oi[5:] - oi[:-5]
        result["cot_oi_change"] = oi_diff / oi_safe

        # 5. Speculative extreme (contrarian signal)
        result["cot_spec_extreme"] = (np.abs(spec_norm) > self._extreme_threshold).astype(float)

        # 6. Momentum (4-week spec net change)
        momentum = np.zeros(n)
        offset = 20  # ~4 weeks of H1 bars
        if n > offset:
            momentum[offset:] = spec_net[offset:] - spec_net[:-offset]
        result["cot_momentum"] = momentum / (oi_safe + 1e-10)

        # 7. Divergence: price up but spec reducing → bearish divergence
        if "close" in bars.columns:
            close = bars["close"].values
            price_change = np.zeros(n)
            spec_change_20 = np.zeros(n)
            if n > 20:
                price_change[20:] = close[20:] - close[:-20]
                spec_change_20[20:] = spec_net[20:] - spec_net[:-20]
            # Divergence = sign mismatch between price and positioning
            price_dir = np.sign(price_change)
            spec_dir = np.sign(spec_change_20)
            result["cot_divergence"] = -price_dir * spec_dir  # -1 = divergent, +1 = convergent
        else:
            result["cot_divergence"] = 0.0

        return result

    @staticmethod
    def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling z-score."""
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(window, n):
            window_data = arr[i - window: i]
            mean = np.nanmean(window_data)
            std = np.nanstd(window_data, ddof=1)
            if std > 1e-10:
                result[i] = (arr[i] - mean) / std
            else:
                result[i] = 0.0
        return result
