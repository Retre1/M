"""Pydantic v2 configuration models for the bias-free data pipeline.

All parameters are externalised to YAML via the existing config registry.
Each config class is frozen (immutable) after construction to prevent
accidental mutation during training.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# FSD (Forex Skewness Dispersion) configuration
# ---------------------------------------------------------------------------


class FSDConfig(BaseModel, frozen=True):
    """Configuration for the Forex Skewness Dispersion regime detector.

    Attributes:
        enabled: Whether to compute FSD features.
        window: Rolling window size for per-column skewness estimation.
            For H1 bars, 252 ≈ ~10.5 trading days of data.
        min_periods: Minimum observations before emitting a non-NaN value.
            Set to ``window // 2`` by default to allow warm-up.
        quantile_thresholds: (low, high) percentiles used to classify
            the regime into RISK_ON / NEUTRAL / RISK_OFF.
        output_cols: Column names appended to the output DataFrame.
    """

    enabled: bool = False
    window: int = 252
    min_periods: int | None = None  # defaults to window // 2
    quantile_thresholds: tuple[float, float] = (0.30, 0.70)
    output_cols: tuple[str, ...] = (
        "fsd_dispersion",
        "fsd_regime_risk_off",
        "fsd_regime_neutral",
        "fsd_regime_risk_on",
    )

    @field_validator("quantile_thresholds")
    @classmethod
    def _validate_thresholds(cls, v: tuple[float, float]) -> tuple[float, float]:
        low, high = v
        if not (0.0 < low < high < 1.0):
            raise ValueError(f"Thresholds must satisfy 0 < low < high < 1, got {v}")
        return v


# ---------------------------------------------------------------------------
# SBBTS (Schrödinger–Bass) synthetic generator configuration
# ---------------------------------------------------------------------------


class SBBTSPairParams(BaseModel, frozen=True):
    """Per-pair calibration overrides for the SBBTS generator.

    If omitted, the generator calibrates automatically from the real data.

    Attributes:
        mu: Annualised drift override.  ``None`` → calibrate.
        sigma: Annualised volatility override.  ``None`` → calibrate.
        bass_p: Bass innovation (external influence) parameter.
        bass_q: Bass imitation (internal influence) parameter.
        vol_of_vol: Volatility-of-volatility (ξ in Heston model).
        mean_reversion_kappa: Speed of vol mean-reversion (κ).
    """

    mu: float | None = None
    sigma: float | None = None
    bass_p: float = 0.03
    bass_q: float = 0.38
    vol_of_vol: float = 0.30
    mean_reversion_kappa: float = 2.0


class SBBTSConfig(BaseModel, frozen=True):
    """Configuration for the SBBTS synthetic trajectory generator.

    Attributes:
        enabled: Whether synthetic augmentation is active.
        ratio: Fraction of synthetic bars mixed into the final dataset.
            0.3 means 30 % synthetic, 70 % real.
        n_paths: Number of Monte-Carlo paths generated per call.
        dt: Simulation time-step expressed as fraction of a year.
            For H1 bars: 1 / (252 * 24) ≈ 1.65e-4.
        beta_levels: Multiple noise strengths to sweep during generation.
            Higher β → heavier tails and wilder paths.
        potential_depth: Depth of the double-well Schrödinger potential.
            Controls regime-switching intensity.
        hbar: Reduced Planck constant analogue (scales quantum diffusion).
        kl_threshold: Maximum acceptable KL divergence between the
            marginal return distributions of real vs. synthetic data.
        correlation_preserve: If True, generate correlated multi-pair paths
            via a Cholesky decomposition of the empirical correlation matrix.
        pair_params: Per-pair parameter overrides keyed by symbol name.
        seed: RNG seed for reproducibility.
    """

    enabled: bool = False
    ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    n_paths: int = Field(default=1000, ge=1)
    dt: float = Field(default=1.0 / (252 * 24), gt=0.0)
    beta_levels: tuple[float, ...] = (1.0,)
    potential_depth: float = 0.5
    hbar: float = 1.0
    kl_threshold: float = 0.05
    correlation_preserve: bool = True
    pair_params: dict[str, SBBTSPairParams] = Field(default_factory=dict)
    seed: int = 42


# ---------------------------------------------------------------------------
# Bias-free pipeline configuration
# ---------------------------------------------------------------------------


class GapProcedureConfig(BaseModel, frozen=True):
    """Gap-procedure settings for eliminating look-back bias (LIB).

    The gap procedure ensures that:
    - Signals are computed at close of bar *t*.
    - Returns are computed from open(t+1) → close(t+1).
    This eliminates the overlap between signal and return windows.

    Attributes:
        enabled: Activate the gap procedure.
        return_col: Name of the column that holds the unbiased return.
        signal_lag: Number of bars to lag signals (1 = standard gap).
    """

    enabled: bool = True
    return_col: str = "forward_return_gap"
    signal_lag: int = 1


class ExAnteFilterConfig(BaseModel, frozen=True):
    """Ex-ante (expanding-window) filtering to eliminate look-ahead bias (LAB).

    All winsorisation, clipping, and z-score normalisation are performed
    using only data available *up to and including* bar t-1.

    Attributes:
        enabled: Activate ex-ante filtering.
        winsor_quantiles: (lower, upper) quantiles for winsorisation.
        clip_zscore: Maximum absolute z-score after normalisation.
        min_history: Minimum number of bars before filtering starts.
            Bars before this threshold are left unnormalised.
    """

    enabled: bool = True
    winsor_quantiles: tuple[float, float] = (0.01, 0.99)
    clip_zscore: float = 5.0
    min_history: int = 50


class BiasFreePipelineConfig(BaseModel, frozen=True):
    """Master configuration for the bias-free data pipeline.

    Attributes:
        gap_procedure: LIB correction settings.
        ex_ante_filter: LAB correction settings.
        fsd: FSD regime detector settings.
        sbbts: Synthetic data generator settings.
        output_format: Serialisation format for processed data.
        output_compression: Parquet compression codec.
        n_jobs: Parallel workers for multi-pair processing.
            0 or 1 = sequential.  -1 = all CPUs.
    """

    gap_procedure: GapProcedureConfig = Field(default_factory=GapProcedureConfig)
    ex_ante_filter: ExAnteFilterConfig = Field(default_factory=ExAnteFilterConfig)
    fsd: FSDConfig = Field(default_factory=FSDConfig)
    sbbts: SBBTSConfig = Field(default_factory=SBBTSConfig)
    output_format: Literal["parquet", "csv"] = "parquet"
    output_compression: str = "snappy"
    n_jobs: int = 1
