"""Curriculum learning: progressive training from simple to complex data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from apexfx.config.schema import CurriculumStage, TrainingConfig
from apexfx.data.synthetic import RegimeParams, SyntheticDataGenerator
from apexfx.data.mtf_synthetic import MTFSyntheticGenerator, resample_real_data
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


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
class CurriculumManager:
    """
    Manages the three-stage curriculum learning process:

    Stage 1 — Synthetic Clean: Clear trending/mean-reverting regimes, no noise
    Stage 2 — Noisy + Simple Real: Noise-augmented synthetic + clean real data
    Stage 3 — Full Real + Black Swans: Real historical data with crisis events
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
        self._generator = SyntheticDataGenerator(seed=seed)
        self._mtf_generator = MTFSyntheticGenerator(seed=seed) if mtf_enabled else None
        self._mtf_enabled = mtf_enabled
        self._current_stage: int = 0

    @property
    def n_stages(self) -> int:
        return len(self._config.curriculum.stages)

    @property
    def current_stage_idx(self) -> int:
        return self._current_stage

    def get_stage(self, stage_idx: int) -> StageData | MTFStageData:
        """Generate or prepare data for a specific curriculum stage."""
        if stage_idx >= self.n_stages:
            raise ValueError(f"Stage {stage_idx} does not exist. Max: {self.n_stages - 1}")

        stage = self._config.curriculum.stages[stage_idx]
        logger.info(
            "Preparing curriculum stage",
            stage=stage_idx,
            name=stage.name,
            description=stage.description,
            data_source=stage.data_source,
            mtf=self._mtf_enabled,
        )

        if self._mtf_enabled:
            return self._get_mtf_stage(stage, stage_idx)

        if stage.data_source == "synthetic":
            data = self._generate_synthetic(stage)
        elif stage.data_source == "mixed":
            data = self._generate_mixed(stage)
        elif stage.data_source == "real":
            data = self._prepare_real(stage)
        else:
            raise ValueError(f"Unknown data source: {stage.data_source}")

        logger.info(
            "Stage data ready",
            stage=stage_idx,
            n_bars=len(data),
        )

        return StageData(stage=stage, data=data, stage_idx=stage_idx)

    def _get_mtf_stage(self, stage: CurriculumStage, stage_idx: int) -> MTFStageData:
        """Generate MTF data for a curriculum stage."""
        params = stage.synthetic_params

        if stage.data_source == "synthetic":
            mtf = self._mtf_generator.generate(
                n_h1_bars=max(2000, params.n_steps // 5),
                noise_std=params.noise_std,
                black_swan_intensity=params.black_swan_intensity,
                black_swan_magnitude_std=params.black_swan_magnitude_std,
            )
            logger.info(
                "MTF synthetic data ready",
                d1=len(mtf.d1), h1=len(mtf.h1), m5=len(mtf.m5),
            )
            return MTFStageData(
                stage=stage, data=mtf.h1, stage_idx=stage_idx,
                d1_data=mtf.d1, h1_data=mtf.h1, m5_data=mtf.m5,
            )

        elif stage.data_source == "mixed":
            # Generate synthetic MTF
            mtf_synth = self._mtf_generator.generate(
                n_h1_bars=max(1000, params.n_steps // 10),
                noise_std=params.noise_std,
            )
            if self._real_data is not None and len(self._real_data) > 0:
                real_h1 = self._real_data.head(params.n_steps // 2).copy().reset_index(drop=True)
                d1_real, m5_real = resample_real_data(real_h1)
                # Concatenate synthetic + real
                h1_combined = pd.concat([mtf_synth.h1, real_h1], ignore_index=True)
                d1_combined = pd.concat([mtf_synth.d1, d1_real], ignore_index=True)
                m5_combined = pd.concat([mtf_synth.m5, m5_real], ignore_index=True)
            else:
                h1_combined = mtf_synth.h1
                d1_combined = mtf_synth.d1
                m5_combined = mtf_synth.m5

            logger.info(
                "MTF mixed data ready",
                d1=len(d1_combined), h1=len(h1_combined), m5=len(m5_combined),
            )
            return MTFStageData(
                stage=stage, data=h1_combined, stage_idx=stage_idx,
                d1_data=d1_combined, h1_data=h1_combined, m5_data=m5_combined,
            )

        elif stage.data_source == "real":
            if self._real_data is None or len(self._real_data) == 0:
                logger.warning("No real data, falling back to synthetic MTF")
                return self._get_mtf_stage(
                    CurriculumStage(
                        name=stage.name, description=stage.description,
                        total_timesteps=stage.total_timesteps,
                        data_source="synthetic", synthetic_params=params,
                    ),
                    stage_idx,
                )
            real_h1 = self._real_data.copy()
            d1_real, m5_real = resample_real_data(real_h1)

            if params.black_swan_intensity > 0:
                real_h1 = self._generator.inject_black_swans(
                    real_h1, params.black_swan_intensity, params.black_swan_magnitude_std,
                )

            logger.info(
                "MTF real data ready",
                d1=len(d1_real), h1=len(real_h1), m5=len(m5_real),
            )
            return MTFStageData(
                stage=stage, data=real_h1, stage_idx=stage_idx,
                d1_data=d1_real, h1_data=real_h1, m5_data=m5_real,
            )

        raise ValueError(f"Unknown data source: {stage.data_source}")

    def stages(self) -> list[StageData]:
        """Iterate through all curriculum stages."""
        return [self.get_stage(i) for i in range(self.n_stages)]

    def _generate_synthetic(self, stage: CurriculumStage) -> pd.DataFrame:
        """Generate clean synthetic data for Stage 1."""
        params = stage.synthetic_params
        data = self._generator.generate_regime_switching(
            n_steps=params.n_steps,
            regimes=[
                RegimeParams(mu=0.001, sigma=0.008, duration_mean=200, name="trend_up"),
                RegimeParams(mu=-0.001, sigma=0.008, duration_mean=200, name="trend_down"),
                RegimeParams(mu=0.0, sigma=0.004, duration_mean=300, name="flat"),
                RegimeParams(mu=0.0, sigma=0.012, duration_mean=150, name="volatile"),
            ],
        )

        if params.noise_std > 0:
            data = self._generator.add_noise(data, params.noise_std)

        if params.black_swan_intensity > 0:
            data = self._generator.inject_black_swans(
                data, params.black_swan_intensity, params.black_swan_magnitude_std
            )

        return data

    def _generate_mixed(self, stage: CurriculumStage) -> pd.DataFrame:
        """Mix noisy synthetic with real data for Stage 2."""
        params = stage.synthetic_params

        # Generate noisy synthetic
        synthetic = self._generator.generate_regime_switching(
            n_steps=params.n_steps // 2,
        )
        synthetic = self._generator.add_noise(synthetic, params.noise_std)

        if self._real_data is not None and len(self._real_data) > 0:
            # Take the cleanest portion of real data
            real_portion = self._real_data.head(params.n_steps // 2).copy()
            real_portion = real_portion.reset_index(drop=True)

            # Combine: synthetic first, then real
            combined = pd.concat([synthetic, real_portion], ignore_index=True)
        else:
            # No real data: just use more synthetic
            extra = self._generator.generate_regime_switching(n_steps=params.n_steps // 2)
            extra = self._generator.add_noise(extra, params.noise_std * 2)
            combined = pd.concat([synthetic, extra], ignore_index=True)

        return combined

    def _prepare_real(self, stage: CurriculumStage) -> pd.DataFrame:
        """Prepare real data with optional black swan injection for Stage 3."""
        if self._real_data is None or len(self._real_data) == 0:
            logger.warning("No real data available, falling back to synthetic")
            return self._generate_synthetic(stage)

        data = self._real_data.copy()
        params = stage.synthetic_params

        if params.black_swan_intensity > 0:
            data = self._generator.inject_black_swans(
                data, params.black_swan_intensity, params.black_swan_magnitude_std
            )

        return data
