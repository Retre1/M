"""Feature engineering pipeline orchestrator."""

from __future__ import annotations

import pandas as pd

from apexfx.features import BaseFeatureExtractor
from apexfx.features.clustering import ClusteringExtractor
from apexfx.features.fundamental import FundamentalExtractor
from apexfx.features.hurst import HurstExtractor
from apexfx.features.intermarket_corr import IntermarketCorrExtractor
from apexfx.features.normalizer import FeatureNormalizer
from apexfx.features.order_flow import OrderFlowExtractor
from apexfx.features.orderbook import OrderBookExtractor
from apexfx.features.regime import RegimeExtractor
from apexfx.features.spectral import SpectralExtractor
from apexfx.features.structure import StructureExtractor
from apexfx.features.volume_profile import VolumeProfileExtractor
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """Orchestrates all feature extractors and produces the final feature matrix."""

    def __init__(
        self,
        extractors: list[BaseFeatureExtractor] | None = None,
        normalizer: FeatureNormalizer | None = None,
        normalize: bool = True,
    ) -> None:
        self._extractors = extractors or self._default_extractors()
        self._normalizer = normalizer or FeatureNormalizer(method="zscore", window=252)
        self._normalize = normalize
        self._feature_names: list[str] = []

    @staticmethod
    def _default_extractors() -> list[BaseFeatureExtractor]:
        extractors = [
            VolumeProfileExtractor(window=100),
            OrderFlowExtractor(),
            HurstExtractor(window=252),
            SpectralExtractor(fft_window=256),
            IntermarketCorrExtractor(),
            RegimeExtractor(),
            ClusteringExtractor(window=200),
            FundamentalExtractor(),
            StructureExtractor(),
            OrderBookExtractor(),
        ]

        # Conditionally add SentimentExtractor (requires transformers package)
        try:
            from apexfx.features.sentiment import SentimentExtractor
            extractors.append(SentimentExtractor())
        except ImportError:
            logger.debug("SentimentExtractor not available (transformers not installed)")

        return extractors

    @property
    def feature_names(self) -> list[str]:
        if not self._feature_names:
            for extractor in self._extractors:
                self._feature_names.extend(extractor.feature_names)
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def compute(
        self,
        bars: pd.DataFrame,
        ticks: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run all extractors and concatenate results.
        Returns a DataFrame with the original bar columns plus all feature columns.
        """
        feature_dfs: list[pd.DataFrame] = []

        for extractor in self._extractors:
            name = extractor.__class__.__name__
            logger.debug("Running extractor", extractor=name)
            try:
                features = extractor.extract(bars, ticks)
                feature_dfs.append(features)
                logger.debug(
                    "Extractor complete",
                    extractor=name,
                    n_features=len(features.columns),
                )
            except Exception as e:
                logger.error("Extractor failed", extractor=name, error=str(e))
                # Create empty DataFrame with expected columns
                empty = pd.DataFrame(index=bars.index)
                for col in extractor.feature_names:
                    empty[col] = float("nan")
                feature_dfs.append(empty)

        # Concatenate all features
        result = pd.concat([bars] + feature_dfs, axis=1)

        # Handle duplicated column names (from bars that already have feature columns)
        result = result.loc[:, ~result.columns.duplicated(keep="last")]

        # Forward-fill NaN at the beginning (from rolling windows)
        feature_cols = [c for c in result.columns if c not in bars.columns]
        result[feature_cols] = result[feature_cols].ffill()

        # Normalize features (excluding bar columns and regime labels)
        if self._normalize:
            exclude_cols = list(bars.columns) + [
                "regime_label", "hurst_regime", "in_liquidity_zone",
                "delta_divergence",
                # Phase 3: binary/categorical features should not be z-scored
                "news_impact_active", "conflicting_signals",
                "structure_break_bull", "structure_break_bear",
                "structure_trend", "retest_signal",
                # Phase 4: orderbook features that are already normalized
                "bid_ask_imbalance", "book_pressure",
                # Phase 4: sentiment features that are already bounded
                "headline_count",
            ]
            normalize_cols = [c for c in feature_cols if c not in exclude_cols]
            if normalize_cols:
                result[normalize_cols] = self._normalizer.fit_transform(
                    result[normalize_cols]
                )

        logger.info(
            "Feature pipeline complete",
            n_bars=len(result),
            n_features=len(feature_cols),
        )

        return result

    def compute_incremental(
        self,
        new_bar: pd.Series,
        history: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute features for a single new bar given history.
        Used in live trading for real-time feature computation.
        """
        # Append new bar to history
        combined = pd.concat([history, new_bar.to_frame().T], ignore_index=True)

        # Temporarily disable normalization — we normalize manually below
        saved_normalize = self._normalize
        self._normalize = False
        try:
            features = self.compute(combined)
        finally:
            self._normalize = saved_normalize

        # Get the last row and normalize online
        last_features = features.iloc[-1].copy()

        if saved_normalize:
            available = [n for n in self.feature_names if n in last_features.index]
            if available:
                feature_vals = last_features[available].values.astype(float)
                normalized = self._normalizer.transform_online(feature_vals)
                for i, name in enumerate(available):
                    last_features[name] = normalized[i]

        return last_features

    @property
    def normalizer(self) -> FeatureNormalizer:
        return self._normalizer
