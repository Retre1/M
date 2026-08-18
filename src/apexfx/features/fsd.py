"""Cross-sectional skewness dispersion as a regime signal.

FSD is the dispersion of skewness *across instruments*: when the individual
return distributions of a basket disagree in shape, the market is pricing
different assets under different regimes. That is the quantity
``HiveMindGating_v2`` conditions its gating on.

**The constraint that governs this module.** Dispersion across one instrument
is not a quantity: ``std`` of a single value is undefined, so a single-symbol
frame produces exactly zero at every bar, the regime is constant, and anything
downstream conditioned on it is dead weight. That failure is silent and looks
like a working feature, so this extractor refuses to fake it — with fewer than
``MIN_INSTRUMENTS`` series it emits zeros *and warns*, rather than presenting a
constant as a signal.

The basket comes from the intermarket merge (DXY, gold, SPX and so on), which
lands in the bars frame as ``{instrument}_close`` columns. Those columns are the
reason FSD is computable at all while the system trades one pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from apexfx.data.fsd_regime import FSDRegime, FSDRegimeDetector
from apexfx.features import BaseFeatureExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)

# Below this, cross-sectional dispersion is not defined in any useful sense.
MIN_INSTRUMENTS = 2
# Quantiles of the dispersion's own history that separate the regimes.
QUANTILE_THRESHOLDS = (0.3, 0.7)
# Bars of dispersion history before a quantile means anything.
WARMUP_BARS = 100


class FSDExtractor(BaseFeatureExtractor):
    """Dispersion plus a one-hot of the regime it implies.

    Produces exactly the four columns ``GatingV2Config.d_fsd`` expects.
    """

    def __init__(self, detector: FSDRegimeDetector | None = None) -> None:
        self._detector = detector or FSDRegimeDetector()

    @property
    def feature_names(self) -> list[str]:
        return [
            "fsd_dispersion",
            "fsd_regime_risk_on",
            "fsd_regime_neutral",
            "fsd_regime_risk_off",
        ]

    def extract(
        self, bars: pd.DataFrame, ticks: pd.DataFrame | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        returns = self._basket_returns(bars)
        out = pd.DataFrame(index=bars.index, columns=self.feature_names, dtype=np.float64)

        if returns.shape[1] < MIN_INSTRUMENTS:
            logger.warning(
                "FSD needs a basket, not one instrument — emitting zeros. "
                "Anything conditioned on these columns is inert until "
                "intermarket data is merged.",
                n_instruments=returns.shape[1],
                required=MIN_INSTRUMENTS,
            )
            out[:] = 0.0
            return out

        dispersion = self._detector.compute_skewness_dispersion(returns)
        out["fsd_dispersion"] = dispersion.to_numpy()

        # The detector reports zero until its own rolling window fills. Those
        # zeros are not low-dispersion readings, they are absence of a reading,
        # and leaving them in drags the quantile baseline to zero — which made
        # the RISK_ON side unreachable because nothing is below zero.
        warmup = int(self._detector.config.window)
        usable = dispersion.copy()
        usable.iloc[:warmup] = np.nan

        regimes = self._classify(usable)
        for value in FSDRegime:
            column = f"fsd_regime_{value.name.lower()}"
            out[column] = (regimes == int(value)).astype(np.float64)

        return out

    @staticmethod
    def _classify(dispersion: pd.Series) -> np.ndarray:
        """Regime from where dispersion sits in its own history.

        ``FSDRegimeDetector.classify_regime`` documents its 0.3 / 0.7 cut-offs
        as quantiles but compares the raw dispersion against them. Measured
        dispersion on an intermarket basket runs roughly 0 to 0.3, so every bar
        lands in one class and all three regime bits come out constant — a
        third variant of the inert-input failure, this one hidden behind a
        docstring that describes the intended behaviour rather than the code's.

        The cut-offs are therefore taken as quantiles of the dispersion seen
        *so far*: an expanding window, shifted by one bar, so the label for bar
        *t* never uses bar *t*'s own value or anything after it. A trailing
        window would be look-ahead by the back door — the same defect the
        feature extractors were audited for.
        """
        low_q, high_q = QUANTILE_THRESHOLDS
        low = dispersion.expanding(min_periods=WARMUP_BARS).quantile(low_q).shift(1)
        high = dispersion.expanding(min_periods=WARMUP_BARS).quantile(high_q).shift(1)

        regimes = np.full(len(dispersion), int(FSDRegime.NEUTRAL), dtype=np.int64)
        values = dispersion.to_numpy()
        # Before the warm-up fills there is no history to rank against, and
        # NEUTRAL is the honest label for "not yet known".
        known = (
            low.notna().to_numpy()
            & high.notna().to_numpy()
            & ~np.isnan(values)
        )
        regimes[known & (values < low.to_numpy())] = int(FSDRegime.RISK_ON)
        regimes[known & (values > high.to_numpy())] = int(FSDRegime.RISK_OFF)
        return regimes

    @staticmethod
    def _basket_returns(bars: pd.DataFrame) -> pd.DataFrame:
        """Log returns of the traded instrument and every merged one.

        ``{inst}_close`` is the shape the intermarket merge produces; the target
        instrument's own ``close`` joins the basket because the regime it sits
        in is part of what the dispersion measures.
        """
        series: dict[str, pd.Series] = {}
        if "close" in bars.columns:
            series["target"] = bars["close"]
        for column in bars.columns:
            if column.endswith("_close") and column != "close":
                series[column[: -len("_close")]] = bars[column]

        if not series:
            return pd.DataFrame(index=bars.index)

        prices = pd.DataFrame(series, index=bars.index).astype(np.float64)
        # Guard against non-positive quotes before taking logs.
        prices = prices.where(prices > 0)
        return np.log(prices).diff().fillna(0.0)
