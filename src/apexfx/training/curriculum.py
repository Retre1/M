"""Curriculum learning — progressive training from filtered to full real data.

Stages train exclusively on real market data with increasing difficulty:

1. **Filtered**: remove extreme volatility events for stable initial learning.
2. **Full**: all market conditions including volatility spikes and regime shifts.
3. **Augmented**: noise-augmented real data for robustness and overfitting prevention.

Synthetic data generation has been removed in favour of real-data-only training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from apexfx.config.schema import CurriculumStage, DataAugmentationConfig, TrainingConfig
from apexfx.data.mtf_synthetic import resample_real_data
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------


@dataclass
class StageData:
    """Container for a curriculum stage's training data."""

    stage: CurriculumStage
    data: pd.DataFrame
    stage_idx: int


@dataclass
class MTFStageData(StageData):
    """Container for MTF curriculum stage data (D1 + H1 + M5)."""

    d1_data: pd.DataFrame = None
    h1_data: pd.DataFrame = None
    m5_data: pd.DataFrame = None


# ------------------------------------------------------------------
# Manager
# ------------------------------------------------------------------


class CurriculumManager:
    """Manage real-data curriculum with progressive difficulty.

    Stage types
    -----------
    * ``data_source="real"`` (default) — use historical OHLCV data.
    * ``filter_quantile`` — if set, remove the most extreme *q* bars.
    * ``augmentation``     — if set, inject Gaussian noise + price shift.

    Synthetic data is no longer generated.  If ``real_data`` is ``None``
    the manager raises immediately so the user knows data is required.
    """

    def __init__(
        self,
        config: TrainingConfig,
        real_data: pd.DataFrame | None = None,
        seed: int = 42,
        mtf_enabled: bool = False,
    ) -> None:
        self._config = config
        self._real_data = real_data
        self._mtf_enabled = mtf_enabled
        self._rng = np.random.RandomState(seed)
        self._current_stage: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_stages(self) -> int:
        return len(self._config.curriculum.stages)

    @property
    def current_stage_idx(self) -> int:
        return self._current_stage

    def get_stage(self, stage_idx: int) -> StageData | MTFStageData:
        """Prepare data for a specific curriculum stage."""
        if stage_idx >= self.n_stages:
            raise ValueError(f"Stage {stage_idx} does not exist (max {self.n_stages - 1}).")

        stage = self._config.curriculum.stages[stage_idx]
        logger.info(
            "Preparing curriculum stage",
            stage=stage_idx,
            name=stage.name,
            description=stage.description,
            data_source=stage.data_source,
            mtf=self._mtf_enabled,
        )

        if self._real_data is None or len(self._real_data) == 0:
            raise ValueError(
                "Real data is required for training. "
                "Provide historical OHLCV data via the real_data parameter."
            )

        data = self._prepare_real_data(stage)

        if self._mtf_enabled:
            return self._build_mtf_stage(stage, stage_idx, data)

        logger.info("Stage data ready", stage=stage_idx, n_bars=len(data))
        return StageData(stage=stage, data=data, stage_idx=stage_idx)

    def stages(self) -> list[StageData | MTFStageData]:
        """Return data for every curriculum stage in order."""
        return [self.get_stage(i) for i in range(self.n_stages)]

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_real_data(self, stage: CurriculumStage) -> pd.DataFrame:
        """Copy real data, apply optional filter and augmentation."""
        data = self._real_data.copy()

        # 1. Filter extreme-volatility bars
        if stage.filter_quantile is not None:
            data = self._filter_extreme_bars(data, stage.filter_quantile)

        # 2. Noise augmentation
        if stage.augmentation is not None:
            data = self._augment_data(data, stage.augmentation)

        return data

    def _filter_extreme_bars(
        self,
        data: pd.DataFrame,
        quantile: float,
    ) -> pd.DataFrame:
        """Remove bars whose absolute return exceeds *quantile*.

        For example ``quantile=0.95`` drops the most volatile 5 % of bars,
        leaving a cleaner dataset for the warm-up stage.
        """
        if "close" not in data.columns:
            logger.warning("No 'close' column — skipping extreme-bar filter")
            return data

        returns = data["close"].pct_change().abs()
        threshold = returns.quantile(quantile)
        mask = returns <= threshold
        mask.iloc[0] = True  # always keep the first bar

        filtered = data[mask].reset_index(drop=True)
        n_removed = len(data) - len(filtered)
        logger.info(
            "Filtered extreme bars",
            original=len(data),
            kept=len(filtered),
            removed=n_removed,
            quantile=quantile,
            threshold=round(float(threshold), 6),
        )
        return filtered

    def _augment_data(
        self,
        data: pd.DataFrame,
        config: DataAugmentationConfig,
    ) -> pd.DataFrame:
        """Inject additive Gaussian noise + optional price-level shift.

        Augmentation improves robustness by exposing the model to slight
        variations of real market data, reducing over-fitting to exact
        historical price levels.
        """
        augmented = data.copy()
        numeric_cols = augmented.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return augmented

        # --- Additive Gaussian noise on all numeric columns ---
        if config.noise_std > 0:
            noise = self._rng.normal(
                0, config.noise_std, size=(len(augmented), len(numeric_cols))
            )
            augmented[numeric_cols] += noise
            logger.info(
                "Applied Gaussian noise augmentation",
                noise_std=config.noise_std,
                n_cols=len(numeric_cols),
            )

        # --- Random price-level shift (translation invariance) ---
        if config.price_shift_std > 0:
            price_cols = [c for c in ("open", "high", "low", "close") if c in augmented.columns]
            if price_cols:
                mean_price = float(augmented[price_cols].mean().mean())
                shift = float(self._rng.normal(0, config.price_shift_std)) * mean_price
                augmented[price_cols] += shift
                logger.info(
                    "Applied price-level shift",
                    shift=round(shift, 6),
                    mean_price=round(mean_price, 4),
                )

        return augmented

    # ------------------------------------------------------------------
    # Multi-Timeframe helpers
    # ------------------------------------------------------------------

    def _build_mtf_stage(
        self,
        stage: CurriculumStage,
        stage_idx: int,
        h1_data: pd.DataFrame,
    ) -> MTFStageData:
        """Re-sample H1 data to D1 and M5 for the MTF pipeline."""
        d1_data, m5_data = resample_real_data(h1_data)

        logger.info(
            "MTF real data ready",
            d1=len(d1_data),
            h1=len(h1_data),
            m5=len(m5_data),
        )

        return MTFStageData(
            stage=stage,
            data=h1_data,
            stage_idx=stage_idx,
            d1_data=d1_data,
            h1_data=h1_data,
            m5_data=m5_data,
        )
