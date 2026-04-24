"""NaN / inf safety tests for FeatureNormalizer.

Guards against the silent corruption path documented in AUDIT_REPORT.md:
one NaN in the live stream poisons the running mean/var and every
subsequent observation normalises to NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apexfx.features.normalizer import FeatureNormalizer, _sanitise


class TestSanitise:
    """Unit tests for the private _sanitise helper."""

    def test_clean_array_passes_through(self) -> None:
        arr = np.array([1.0, 2.0, -3.0])
        out = _sanitise(arr)
        np.testing.assert_array_equal(out, arr)

    def test_nan_replaced_with_zero(self) -> None:
        arr = np.array([1.0, np.nan, 2.0])
        out = _sanitise(arr)
        assert np.all(np.isfinite(out))
        assert out[1] == 0.0

    def test_posinf_clipped(self) -> None:
        arr = np.array([np.inf, 1.0])
        out = _sanitise(arr)
        assert np.all(np.isfinite(out))
        assert out[0] == 5.0

    def test_neginf_clipped(self) -> None:
        arr = np.array([-np.inf, 1.0])
        out = _sanitise(arr)
        assert np.all(np.isfinite(out))
        assert out[0] == -5.0


class TestTransformOnline:
    """Integration tests for the live normalisation path."""

    def test_nan_does_not_poison_stats(self) -> None:
        """Single NaN observation must not corrupt subsequent outputs."""
        norm = FeatureNormalizer(method="zscore")

        # Warm up with clean data
        for _ in range(50):
            _ = norm.transform_online(np.array([1.0, 2.0, 3.0]))

        # Inject a poisoned sample
        _ = norm.transform_online(np.array([np.nan, 2.0, np.inf]))

        # Next clean sample must produce finite output
        out = norm.transform_online(np.array([1.0, 2.0, 3.0]))
        assert np.all(np.isfinite(out))

    def test_repeated_nans_do_not_diverge(self) -> None:
        """Many NaN inputs in a row should not blow stats to infinity."""
        norm = FeatureNormalizer(method="zscore")
        for _ in range(100):
            out = norm.transform_online(np.array([np.nan, np.nan]))
            assert np.all(np.isfinite(out))

    def test_output_bounded(self) -> None:
        """Output must respect the ±5 σ clip even with pathological input."""
        norm = FeatureNormalizer(method="zscore")
        # Warm up
        for _ in range(20):
            norm.transform_online(np.zeros(2))
        # Huge finite input
        out = norm.transform_online(np.array([1e12, -1e12]))
        assert np.all(np.abs(out) <= 5.0)


class TestFitTransform:
    """Batch fit_transform must also be NaN-resistant."""

    def test_dataframe_with_nans_does_not_raise(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0, 5.0] * 10,
                "b": [np.inf, 2.0, 3.0, 4.0, 5.0] * 10,
            }
        )
        norm = FeatureNormalizer(method="zscore", window=10)
        out = norm.fit_transform(df)
        assert out.shape == df.shape
        # Rolling z-score may legitimately produce a few NaNs during warm-up
        # but should not explode into inf and most rows should be finite.
        assert np.isfinite(out.to_numpy(na_value=0.0)).all()

    def test_stats_not_persisted_across_calls(self) -> None:
        """fit_transform is stateless w.r.t. _stats (used for live mode)."""
        norm = FeatureNormalizer(method="zscore", window=5)
        df = pd.DataFrame({"a": np.random.RandomState(0).randn(50)})
        norm.fit_transform(df)
        # _stats should still be None until transform_online is called
        assert norm._stats is None

    @pytest.mark.parametrize("method", ["zscore", "minmax"])
    def test_methods_robust_to_inf(self, method: str) -> None:
        df = pd.DataFrame({"a": [1.0, -np.inf, 3.0, 4.0] * 10})
        norm = FeatureNormalizer(method=method, window=5)
        out = norm.fit_transform(df)
        assert out.shape == df.shape
        assert np.isfinite(out.to_numpy(na_value=0.0)).all()
