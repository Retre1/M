"""Model inference → trading signal generation.

Extracts real agent actions and gating weights from the HiveMind ensemble
by hooking into the feature extractor's forward pass.

Supports both single-TF and multi-TF (MTF) modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from apexfx.env.obs_builder import ObservationBuilder
from apexfx.features.importance_tracker import FeatureImportanceTracker
from apexfx.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradingSignal:
    """Complete trading signal from model inference."""
    action: float                # [-1, 1]
    confidence: float            # abs(action)
    trend_agent_action: float    # individual agent contribution
    reversion_agent_action: float
    breakout_agent_action: float
    gating_weights: tuple[float, ...]  # (trend_w, reversion_w, breakout_w)
    regime: str                  # "trending", "mean_reverting", "flat"
    timestamp: datetime
    inference_time_ms: float
    variable_importance: list[float] = field(default_factory=list)
    uncertainty_score: float = 0.0           # MC Dropout uncertainty [0, 1]
    position_scale: float = 1.0              # Uncertainty-adjusted scale factor
    recommended_stop_atr_mult: float = 2.5   # Regime-aware stop multiplier
    top_features: list[tuple[str, float]] = field(default_factory=list)  # Top K by importance


class _HiveMindHook:
    """Forward hook to capture HiveMind intermediate outputs."""

    def __init__(self) -> None:
        self.trend_action: float = 0.0
        self.reversion_action: float = 0.0
        self.breakout_action: float = 0.0
        self.gating_weights: tuple[float, ...] = (0.33, 0.33, 0.34)
        self.variable_importance: list[float] = []
        self._handle = None

    def install(self, hive_mind) -> None:
        """Register a forward hook on the HiveMind module."""
        if self._handle is not None:
            return
        self._handle = hive_mind.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook_fn(self, module, input, output) -> None:
        """Capture outputs from HiveMind.forward()."""
        try:
            self.trend_action = float(output.trend_action[0, 0].detach().cpu())
            self.reversion_action = float(output.reversion_action[0, 0].detach().cpu())
            self.breakout_action = float(output.breakout_action[0, 0].detach().cpu())
            weights = output.gating_weights[0].detach().cpu().numpy()
            self.gating_weights = tuple(float(w) for w in weights)
            if output.variable_importance is not None:
                vi = output.variable_importance[0].detach().cpu()
                if vi.dim() > 1:
                    vi = vi.mean(dim=0)
                self.variable_importance = vi.numpy().tolist()
        except Exception as e:
            logger.debug("HiveMind action extraction failed", error=str(e))


class SignalGenerator:
    """
    Runs model inference and produces trading signals.
    Hooks into HiveMind to extract real agent actions and gating weights.
    """

    def __init__(
        self,
        model_path: str | Path,
        obs_builder: ObservationBuilder | None = None,
        device: str = "cpu",
        uncertainty_n_samples: int = 10,
        feature_names: list[str] | None = None,
        importance_ema_alpha: float = 0.01,
        importance_history_size: int = 1000,
        importance_top_k: int = 5,
    ) -> None:
        self._device = device
        self._obs_builder = obs_builder or ObservationBuilder()
        self._uncertainty_n_samples = uncertainty_n_samples
        self._importance_top_k = importance_top_k

        # Feature importance tracker (optional)
        self._importance_tracker: FeatureImportanceTracker | None = None
        if feature_names:
            self._importance_tracker = FeatureImportanceTracker(
                feature_names=feature_names,
                ema_alpha=importance_ema_alpha,
                history_size=importance_history_size,
            )
            logger.info(
                "Feature importance tracker created",
                n_features=len(feature_names),
                ema_alpha=importance_ema_alpha,
            )

        # Load SB3 model
        logger.info("Loading model", path=str(model_path))
        self._model = SAC.load(str(model_path), device=device)
        logger.info("Model loaded successfully")

        # Install HiveMind hook for agent-level introspection
        self._hook = _HiveMindHook()
        try:
            policy = self._model.policy
            if hasattr(policy, "features_extractor") and hasattr(
                policy.features_extractor, "hive_mind"
            ):
                self._hook.install(policy.features_extractor.hive_mind)
                logger.info("HiveMind forward hook installed")
        except Exception as e:
            logger.warning("Could not install HiveMind hook", error=str(e))

    @property
    def importance_tracker(self) -> FeatureImportanceTracker | None:
        """Access the feature importance tracker, if configured."""
        return self._importance_tracker

    def generate(
        self,
        observation: dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> TradingSignal:
        """
        Generate a trading signal from the current observation.

        The HiveMind hook automatically captures agent actions and
        gating weights during the SB3 predict() call.
        """
        start = time.monotonic()

        # Run inference — hook captures HiveMind internals
        action, _states = self._model.predict(observation, deterministic=deterministic)
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        inference_ms = (time.monotonic() - start) * 1000

        # Read captured values from hook
        trend_action = self._hook.trend_action
        reversion_action = self._hook.reversion_action
        breakout_action = self._hook.breakout_action
        gating_weights = self._hook.gating_weights

        # Determine regime from observation
        regime = "flat"
        regime_features = observation.get("regime_features", np.zeros(6))
        if regime_features[3] > 0.5:
            regime = "trending"
        elif regime_features[4] > 0.5:
            regime = "mean_reverting"

        # Update feature importance tracker
        top_features: list[tuple[str, float]] = []
        if (
            self._importance_tracker is not None
            and self._hook.variable_importance
        ):
            try:
                vi_tensor = torch.tensor(
                    self._hook.variable_importance, dtype=torch.float32,
                )
                self._importance_tracker.update(vi_tensor)
                top_features = self._importance_tracker.get_top_k(
                    self._importance_top_k,
                )
            except Exception as e:
                logger.debug("Importance tracker update failed", error=str(e))

        # Compute uncertainty via MC Dropout
        uncertainty_score = self._compute_uncertainty(observation)
        position_scale = max(0.1, 1.0 - 0.5 * uncertainty_score)
        stop_mult = self._compute_stop_mult(uncertainty_score, regime)

        signal = TradingSignal(
            action=action_value,
            confidence=abs(action_value),
            trend_agent_action=trend_action,
            reversion_agent_action=reversion_action,
            breakout_agent_action=breakout_action,
            gating_weights=gating_weights,
            regime=regime,
            timestamp=datetime.now(UTC),
            inference_time_ms=inference_ms,
            variable_importance=self._hook.variable_importance,
            uncertainty_score=uncertainty_score,
            position_scale=position_scale,
            recommended_stop_atr_mult=stop_mult,
            top_features=top_features,
        )

        gw_str = "/".join(f"{w:.2f}" for w in gating_weights)
        logger.debug(
            "Signal generated",
            action=round(signal.action, 4),
            confidence=round(signal.confidence, 4),
            trend=round(trend_action, 4),
            reversion=round(reversion_action, 4),
            breakout=round(breakout_action, 4),
            gating=gw_str,
            regime=signal.regime,
            uncertainty=round(uncertainty_score, 4),
            inference_ms=round(inference_ms, 2),
        )

        return signal

    def _compute_uncertainty(self, observation: dict[str, np.ndarray]) -> float:
        """Estimate uncertainty via MC Dropout (multiple stochastic forward passes)."""
        try:
            policy = self._model.policy
            extractor = getattr(policy, "features_extractor", None)
            if extractor is None or self._uncertainty_n_samples <= 1:
                return 0.0

            was_training = extractor.training
            extractor.train()  # Enable dropout

            actions = []
            for _ in range(self._uncertainty_n_samples):
                with torch.no_grad():
                    a, _ = self._model.predict(observation, deterministic=True)
                    val = float(a[0]) if isinstance(a, np.ndarray) else float(a)
                    actions.append(val)

            if not was_training:
                extractor.eval()

            if len(actions) < 2:
                return 0.0

            # Uncertainty = std of MC predictions, clamped to [0, 1]
            std = float(np.std(actions))
            return min(1.0, std / 0.5)  # Normalize: std=0.5 → uncertainty=1.0
        except Exception as e:
            logger.debug("MC uncertainty estimation failed", error=str(e))
            return 0.0

    @staticmethod
    def _compute_stop_mult(uncertainty_score: float, regime: str) -> float:
        """Compute recommended ATR multiplier for stop-loss based on regime + uncertainty."""
        base_mult = {
            "trending": 1.5,
            "mean_reverting": 3.0,
            "volatile": 4.0,
            "flat": 2.5,
        }
        mult = base_mult.get(regime, 2.5)
        # Higher uncertainty → wider stop
        mult *= (1.0 + 0.5 * uncertainty_score)
        return mult


@dataclass
class MTFTradingSignal(TradingSignal):
    """Extended trading signal with MTF-specific information."""
    tf_attention_weights: tuple[float, ...] = (0.33, 0.33, 0.34)  # D1, H1, M5


class _MTFHiveMindHook:
    """Forward hook for MTFHiveMind to capture timeframe attention."""

    def __init__(self) -> None:
        self.trend_action: float = 0.0
        self.reversion_action: float = 0.0
        self.breakout_action: float = 0.0
        self.gating_weights: tuple[float, ...] = (0.33, 0.33, 0.34)
        self.tf_attention_weights: tuple[float, ...] = (0.33, 0.33, 0.34)
        self.variable_importance: list[float] = []
        self._handle = None

    def install(self, hive_mind) -> None:
        if self._handle is not None:
            return
        self._handle = hive_mind.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook_fn(self, module, input, output) -> None:
        try:
            self.trend_action = float(output.trend_action[0, 0].detach().cpu())
            self.reversion_action = float(output.reversion_action[0, 0].detach().cpu())
            self.breakout_action = float(output.breakout_action[0, 0].detach().cpu())
            weights = output.gating_weights[0].detach().cpu().numpy()
            self.gating_weights = tuple(float(w) for w in weights)
            tf_weights = output.tf_attention_weights[0].detach().cpu().numpy()
            self.tf_attention_weights = tuple(float(w) for w in tf_weights)
            if output.variable_importance is not None:
                vi = output.variable_importance[0].detach().cpu()
                if vi.dim() > 1:
                    vi = vi.mean(dim=0)
                self.variable_importance = vi.numpy().tolist()
        except Exception as e:
            logger.debug("HiveMind action extraction failed", error=str(e))


class MTFSignalGenerator:
    """Multi-Timeframe signal generator.

    Same interface as SignalGenerator but expects MTF observations
    and produces MTFTradingSignal with timeframe attention weights.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        uncertainty_n_samples: int = 10,
    ) -> None:
        self._device = device
        self._uncertainty_n_samples = uncertainty_n_samples

        logger.info("Loading MTF model", path=str(model_path))
        self._model = SAC.load(str(model_path), device=device)
        logger.info("MTF model loaded successfully")

        # Install MTF hook
        self._hook = _MTFHiveMindHook()
        try:
            policy = self._model.policy
            if hasattr(policy, "features_extractor") and hasattr(
                policy.features_extractor, "hive_mind"
            ):
                self._hook.install(policy.features_extractor.hive_mind)
                logger.info("MTFHiveMind forward hook installed")
        except Exception as e:
            logger.warning("Could not install MTFHiveMind hook", error=str(e))

    def generate(
        self,
        observation: dict[str, np.ndarray],
        deterministic: bool = True,
    ) -> MTFTradingSignal:
        """Generate MTF trading signal."""
        start = time.monotonic()

        action, _states = self._model.predict(observation, deterministic=deterministic)
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        inference_ms = (time.monotonic() - start) * 1000

        # Determine regime
        regime = "flat"
        regime_features = observation.get("regime_features", np.zeros(6))
        if regime_features[3] > 0.5:
            regime = "trending"
        elif regime_features[4] > 0.5:
            regime = "mean_reverting"

        # Compute uncertainty via MC Dropout
        uncertainty_score = self._compute_uncertainty(observation)
        position_scale = max(0.1, 1.0 - 0.5 * uncertainty_score)
        stop_mult = SignalGenerator._compute_stop_mult(uncertainty_score, regime)

        signal = MTFTradingSignal(
            action=action_value,
            confidence=abs(action_value),
            trend_agent_action=self._hook.trend_action,
            reversion_agent_action=self._hook.reversion_action,
            breakout_agent_action=self._hook.breakout_action,
            gating_weights=self._hook.gating_weights,
            tf_attention_weights=self._hook.tf_attention_weights,
            regime=regime,
            timestamp=datetime.now(UTC),
            inference_time_ms=inference_ms,
            variable_importance=self._hook.variable_importance,
            uncertainty_score=uncertainty_score,
            position_scale=position_scale,
            recommended_stop_atr_mult=stop_mult,
        )

        tf_str = "/".join(f"{w:.2f}" for w in self._hook.tf_attention_weights)
        logger.debug(
            "MTF signal generated",
            action=round(signal.action, 4),
            tf_attention=tf_str,
            regime=signal.regime,
            uncertainty=round(uncertainty_score, 4),
            inference_ms=round(inference_ms, 2),
        )

        return signal

    def _compute_uncertainty(self, observation: dict[str, np.ndarray]) -> float:
        """Estimate uncertainty via MC Dropout (multiple stochastic forward passes)."""
        try:
            policy = self._model.policy
            extractor = getattr(policy, "features_extractor", None)
            if extractor is None or self._uncertainty_n_samples <= 1:
                return 0.0

            was_training = extractor.training
            extractor.train()

            actions = []
            for _ in range(self._uncertainty_n_samples):
                with torch.no_grad():
                    a, _ = self._model.predict(observation, deterministic=True)
                    val = float(a[0]) if isinstance(a, np.ndarray) else float(a)
                    actions.append(val)

            if not was_training:
                extractor.eval()

            if len(actions) < 2:
                return 0.0

            std = float(np.std(actions))
            return min(1.0, std / 0.5)
        except Exception as e:
            logger.debug("MC uncertainty estimation failed", error=str(e))
            return 0.0
