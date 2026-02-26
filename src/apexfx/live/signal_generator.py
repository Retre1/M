"""Model inference → trading signal generation.

Extracts real agent actions and gating weights from the HiveMind ensemble
by hooking into the feature extractor's forward pass.

Supports both single-TF and multi-TF (MTF) modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3 import SAC

from apexfx.env.obs_builder import ObservationBuilder
from apexfx.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

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
        except Exception:
            pass


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
    ) -> None:
        self._device = device
        self._obs_builder = obs_builder or ObservationBuilder()

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
        # regime_features: [hurst, realized_vol, trend_strength,
        #                   regime_trending, regime_mean_reverting, regime_flat]
        regime = "flat"
        regime_features = observation.get("regime_features", np.zeros(6))
        realized_vol = regime_features[1]
        if regime_features[3] > 0.5:
            regime = "trending"
        elif regime_features[4] > 0.5:
            regime = "mean_reverting"
        elif realized_vol > 2.0:
            regime = "volatile"

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
            inference_ms=round(inference_ms, 2),
        )

        return signal


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
        except Exception:
            pass


class MTFSignalGenerator:
    """Multi-Timeframe signal generator.

    Same interface as SignalGenerator but expects MTF observations
    and produces MTFTradingSignal with timeframe attention weights.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self._device = device

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
        realized_vol = regime_features[1]
        if regime_features[3] > 0.5:
            regime = "trending"
        elif regime_features[4] > 0.5:
            regime = "mean_reverting"
        elif realized_vol > 2.0:
            regime = "volatile"

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
        )

        tf_str = "/".join(f"{w:.2f}" for w in self._hook.tf_attention_weights)
        logger.debug(
            "MTF signal generated",
            action=round(signal.action, 4),
            tf_attention=tf_str,
            regime=signal.regime,
            inference_ms=round(inference_ms, 2),
        )

        return signal
