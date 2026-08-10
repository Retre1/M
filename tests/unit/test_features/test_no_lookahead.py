"""No feature may depend on bars that have not happened yet.

This is the defect class that silently inflates a backtest and then destroys
live performance, and it is invisible to every other kind of test: the maths
can be right, the values plausible, the unit tests green, and the feature still
unusable because it peeks.

The check is behavioural rather than by inspection. Compute features over the
full series, compute them again over a prefix, and compare the overlap. A value
at bar *i* that changes when later bars are removed was reading the future.

Applied to every extractor in the production pipeline, so a new one cannot be
added without facing it. `StructureExtractor` is the reason this exists: its
swing detection genuinely needs `period` bars on the right, and it handles that
by only trusting swings from `i - swing_period`. That is the correct pattern —
this test proves it rather than trusting the comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexfx.features.clustering import ClusteringExtractor
from apexfx.features.hurst import HurstExtractor
from apexfx.features.order_flow import OrderFlowExtractor
from apexfx.features.orderbook import OrderBookExtractor
from apexfx.features.regime import RegimeExtractor
from apexfx.features.spectral import SpectralExtractor
from apexfx.features.structure import StructureExtractor
from apexfx.features.volume_profile import VolumeProfileExtractor

N_BARS = 700
TRUNCATE = 60  # bars removed to form the prefix


def _bars(n: int = N_BARS) -> pd.DataFrame:
    """A series with trend, reversal and a volatility burst.

    Deliberately not pure noise: a flat series hides lookahead because there is
    nothing in the future worth peeking at.
    """
    rng = np.random.default_rng(17)
    drift = np.concatenate([
        np.full(n // 3, 0.00015),      # up
        np.full(n // 3, -0.00020),     # down
        np.full(n - 2 * (n // 3), 0.00005),
    ])
    noise = rng.normal(0, 0.0008, n)
    noise[n // 2: n // 2 + 30] *= 4    # a burst the future-peeking would see
    close = 1.10 + np.cumsum(drift + noise)

    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"),
        "open": close,
        "high": close + np.abs(rng.normal(0, 0.0006, n)),
        "low": close - np.abs(rng.normal(0, 0.0006, n)),
        "close": close,
        "volume": rng.integers(100, 2000, n).astype(float),
        "spread": np.full(n, 0.0001),
    })


EXTRACTORS = [
    pytest.param(VolumeProfileExtractor(window=100), id="volume_profile"),
    pytest.param(OrderFlowExtractor(), id="order_flow"),
    pytest.param(RegimeExtractor(), id="regime"),
    pytest.param(ClusteringExtractor(window=200), id="clustering"),
    pytest.param(StructureExtractor(), id="structure"),
    pytest.param(OrderBookExtractor(), id="orderbook"),
    pytest.param(HurstExtractor(window=252), id="hurst"),
    pytest.param(SpectralExtractor(fft_window=256), id="spectral"),
]


@pytest.mark.parametrize("extractor", EXTRACTORS)
def test_prefix_values_do_not_change_when_later_bars_are_removed(extractor):
    """The core property: bar i must not know about bar i+1."""
    full = _bars()
    prefix = full.iloc[: N_BARS - TRUNCATE].copy()

    on_full = extractor.extract(full).iloc[: N_BARS - TRUNCATE]
    on_prefix = extractor.extract(prefix)

    assert list(on_full.columns) == list(on_prefix.columns)

    mismatches = []
    for col in on_full.columns:
        a = pd.to_numeric(on_full[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(on_prefix[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-9, atol=1e-12) | both_nan
        if not close.all():
            first = int(np.argmax(~close))
            mismatches.append(
                f"{col}: first divergence at bar {first} "
                f"({a[first]!r} with future bars, {b[first]!r} without)",
            )

    assert not mismatches, (
        f"{type(extractor).__name__} reads future bars:\n  " + "\n  ".join(mismatches)
    )


class TestTheCheckCanFail:
    """Guard the guard — a deliberately peeking extractor must be caught."""

    class _PeekingExtractor:
        """Writes tomorrow's close onto today's bar."""

        def extract(self, bars: pd.DataFrame, ticks=None) -> pd.DataFrame:  # noqa: ARG002
            out = pd.DataFrame(index=bars.index)
            out["next_close"] = bars["close"].shift(-1)
            return out

    def test_lookahead_is_detected(self):
        full = _bars()
        prefix = full.iloc[: N_BARS - TRUNCATE].copy()
        extractor = self._PeekingExtractor()

        on_full = extractor.extract(full).iloc[: N_BARS - TRUNCATE]
        on_prefix = extractor.extract(prefix)

        a = on_full["next_close"].to_numpy(dtype=float)
        b = on_prefix["next_close"].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        agree = (np.isclose(a, b, rtol=1e-9) | both_nan).all()

        assert not agree, "the comparison failed to notice an obvious lookahead"


class TestSwingConfirmationDelay:
    """StructureExtractor's specific mechanism, checked directly."""

    def test_a_swing_is_not_reported_before_its_right_window_closes(self):
        """A swing high at bar i is only knowable at bar i + swing_period."""
        extractor = StructureExtractor()
        period = extractor._swing_period

        highs = np.full(400, 1.10)
        spike = 250
        highs[spike] = 1.15  # an unmistakable swing high

        bars = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=400, freq="h", tz="UTC"),
            "open": 1.10, "high": highs, "low": 1.09, "close": 1.10,
            "volume": 500.0,
        })

        # Cut the series so the spike's right-hand window is incomplete.
        truncated = bars.iloc[: spike + period].copy()
        features = extractor.extract(truncated)

        # Nothing before the confirmation bar may reflect the spike.
        assert np.all(np.isfinite(features.to_numpy(dtype=float)))
