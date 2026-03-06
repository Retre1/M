"""Feature importance analysis and selection using gradient boosting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Ranks features by predictive importance for next-bar direction.

    Uses GradientBoosting to evaluate which features actually predict price
    movement, then selects the top-N for RL training.  This removes noise
    features (e.g. unstable spectral components) and keeps only features
    with real predictive signal.

    Usage::

        selector = FeatureSelector(top_n=15)
        selector.fit(features_df)           # learn importance
        selected = selector.transform(features_df)  # filter columns
        # or one-shot:
        selected = selector.fit_transform(features_df)
    """

    def __init__(
        self,
        top_n: int = 15,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_bars: int = 500,
        forward_return_bars: int = 1,
        seed: int = 42,
    ) -> None:
        self.top_n = top_n
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._min_bars = min_bars
        self._forward_return_bars = forward_return_bars
        self._seed = seed

        self._selected_features: list[str] = []
        self._importance_scores: dict[str, float] = {}
        self._is_fitted: bool = False

    # ── Columns that should never be used as model input ──────────
    _EXCLUDE_COLS = frozenset({
        "time", "open", "high", "low", "close", "volume", "tick_count",
        "regime_label", "hurst_regime", "in_liquidity_zone",
        "delta_divergence", "poc_price",
    })

    # ── public API ────────────────────────────────────────────────

    @property
    def selected_features(self) -> list[str]:
        """Feature names selected after fit(), ordered by importance (desc)."""
        return list(self._selected_features)

    @property
    def importance_scores(self) -> dict[str, float]:
        """All feature importance scores from the last fit()."""
        return dict(self._importance_scores)

    def fit(self, features: pd.DataFrame) -> FeatureSelector:
        """Compute feature importance by training GBM on direction prediction.

        Args:
            features: DataFrame with OHLCV + all extracted feature columns.
                      Must contain a ``close`` column for label construction.

        Returns:
            self (for chaining).
        """
        feature_cols = self._get_candidate_columns(features)
        if not feature_cols:
            logger.warning("No candidate feature columns found — skipping selection")
            return self

        # Build binary label: 1 if close[t + forward] > close[t], else 0
        close = features["close"].values
        labels = np.zeros(len(close), dtype=np.int32)
        for i in range(len(close) - self._forward_return_bars):
            if close[i + self._forward_return_bars] > close[i]:
                labels[i] = 1

        X = features[feature_cols].copy()
        y = labels

        # Drop rows with NaN (from rolling-window warm-up)
        valid_mask = X.notna().all(axis=1)
        # Exclude last `forward_return_bars` rows (label undefined)
        valid_mask.iloc[-self._forward_return_bars :] = False

        X = X.loc[valid_mask]
        y = y[valid_mask.values]

        if len(X) < self._min_bars:
            logger.warning(
                "Not enough valid bars for feature selection",
                n_valid=len(X),
                min_required=self._min_bars,
            )
            # Fallback: return first top_n columns
            self._selected_features = feature_cols[: self.top_n]
            self._is_fitted = True
            return self

        # Train / validation split (time-ordered, no shuffle)
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y[:split], y[split:]

        logger.info(
            "Training GradientBoosting for feature importance",
            n_features=len(feature_cols),
            n_train=len(X_train),
            n_val=len(X_val),
        )

        model = GradientBoostingClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=0.8,
            max_features=0.8,
            random_state=self._seed,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=0,
        )

        model.fit(X_train, y_train)

        # Extract importance (based on impurity reduction)
        raw_importance = model.feature_importances_
        self._importance_scores = {
            col: float(imp) for col, imp in zip(feature_cols, raw_importance)
        }

        # Sort by importance descending
        ranked = sorted(
            self._importance_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        self._selected_features = [name for name, _ in ranked[: self.top_n]]

        # Log results
        logger.info("Feature importance ranking (top %d):", self.top_n)
        for rank, (name, score) in enumerate(ranked[: self.top_n], 1):
            logger.info("  %2d. %-35s  %.4f", rank, name, score)

        if len(ranked) > self.top_n:
            dropped = [name for name, _ in ranked[self.top_n :]]
            logger.info("Dropped %d features: %s", len(dropped), dropped)

        # Validation accuracy as sanity check
        val_acc = model.score(X_val, y_val)
        logger.info("Validation accuracy: %.4f (baseline ~0.50)", val_acc)

        self._is_fitted = True
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Keep only selected feature columns (plus OHLCV/time).

        Args:
            features: full feature DataFrame.

        Returns:
            Filtered DataFrame with OHLCV + selected features only.
        """
        if not self._is_fitted:
            raise RuntimeError("FeatureSelector.fit() must be called before transform()")

        # Always keep bar columns
        bar_cols = [c for c in ("time", "open", "high", "low", "close", "volume", "tick_count")
                    if c in features.columns]

        # Keep selected features that exist in this DataFrame
        keep = [c for c in self._selected_features if c in features.columns]

        return features[bar_cols + keep].copy()

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit on features, then return filtered DataFrame."""
        self.fit(features)
        return self.transform(features)

    # ── internals ─────────────────────────────────────────────────

    def _get_candidate_columns(self, df: pd.DataFrame) -> list[str]:
        """Return numeric columns that are valid candidates for importance ranking."""
        candidates = []
        for col in df.columns:
            if col in self._EXCLUDE_COLS:
                continue
            if df[col].dtype.kind not in ("f", "i"):
                continue
            # Skip columns that are all NaN or constant
            if df[col].isna().all() or df[col].nunique() <= 1:
                continue
            candidates.append(col)
        return candidates
