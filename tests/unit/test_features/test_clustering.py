"""Tests for Clustering (DBSCAN) extractor."""

import numpy as np
import pandas as pd

from apexfx.features.clustering import ClusteringExtractor


class TestClusteringExtractor:
    def test_feature_names(self):
        ext = ClusteringExtractor()
        assert ext.feature_names == [
            "nearest_support_distance", "nearest_resistance_distance",
            "in_liquidity_zone", "n_clusters", "cluster_density",
        ]

    def test_output_shape(self, sample_bars):
        ext = ClusteringExtractor(window=50)
        result = ext.extract(sample_bars)
        assert len(result) == len(sample_bars)

    def test_warmup_is_nan(self, sample_bars):
        window = 50
        ext = ClusteringExtractor(window=window)
        result = ext.extract(sample_bars)
        assert result["n_clusters"].iloc[:window].isna().all()

    def test_distances_non_negative(self, sample_bars):
        ext = ClusteringExtractor(window=50)
        result = ext.extract(sample_bars)
        for col in ["nearest_support_distance", "nearest_resistance_distance"]:
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col} should be non-negative"

    def test_in_liquidity_zone_binary(self, sample_bars):
        ext = ClusteringExtractor(window=50)
        result = ext.extract(sample_bars)
        valid = result["in_liquidity_zone"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})
