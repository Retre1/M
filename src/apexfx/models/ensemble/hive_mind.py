"""HiveMind Ensemble — the unified model connecting TFT, agents, and gating.

This is the top-level model that:
1. Runs features through TFT to get encoded market state
2. Feeds TFT output + specialized features to each agent
3. Runs the gating network to combine agent outputs
4. Provides SB3-compatible feature extraction interface

Now supports 3 agents: Trend, Reversion, and Breakout.
Also supports Multi-Timeframe (MTF) mode with shared TFT + CrossTimeframeFusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from apexfx.models.agents.breakout_agent import BreakoutAgent
from apexfx.models.agents.reversion_agent import ReversionAgent
from apexfx.models.agents.trend_agent import TrendAgent
from apexfx.models.ensemble.cross_tf_fusion import CrossTimeframeFusion
from apexfx.models.ensemble.gating_network import GatingNetwork
from apexfx.models.tft.tft_model import TemporalFusionTransformer

if TYPE_CHECKING:
    import gymnasium as gym


@dataclass
class HiveMindOutput:
    """Full output from the HiveMind ensemble."""
    action: torch.Tensor              # (batch, 1) — final combined action
    trend_action: torch.Tensor        # (batch, 1) — trend agent's action
    reversion_action: torch.Tensor    # (batch, 1) — reversion agent's action
    breakout_action: torch.Tensor     # (batch, 1) — breakout agent's action
    gating_weights: torch.Tensor      # (batch, 3) — meta-controller weights
    tft_attention: torch.Tensor       # attention weight matrices
    variable_importance: torch.Tensor  # TFT variable importance scores
    encoded_state: torch.Tensor       # (batch, d_model) — TFT encoded state


class HiveMind(nn.Module):
    """
    The full ensemble: TFT → Agents → Gating → Combined Action.
    Now with 3 specialist agents: Trend, Reversion, Breakout.
    """

    def __init__(
        self,
        n_continuous_vars: int,
        n_known_future_vars: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        d_trend_features: int = 8,
        d_reversion_features: int = 8,
        d_breakout_features: int = 10,
        d_regime_features: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # --- Temporal Fusion Transformer ---
        self.tft = TemporalFusionTransformer(
            n_continuous_vars=n_continuous_vars,
            n_known_future_vars=n_known_future_vars,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # --- Specialist Agents ---
        self.trend_agent = TrendAgent(
            d_tft=d_model,
            d_trend_features=d_trend_features,
            dropout=dropout,
        )

        self.reversion_agent = ReversionAgent(
            d_tft=d_model,
            d_reversion_features=d_reversion_features,
            dropout=dropout,
        )

        self.breakout_agent = BreakoutAgent(
            d_tft=d_model,
            d_breakout_features=d_breakout_features,
            dropout=dropout,
        )

        # --- Gating Network (Meta-Controller) — 3 agents ---
        self.gating = GatingNetwork(
            d_regime=d_regime_features,
            d_context=d_model,
            n_agents=3,
        )

    def forward(
        self,
        market_features: torch.Tensor,
        time_features: torch.Tensor,
        trend_features: torch.Tensor,
        reversion_features: torch.Tensor,
        regime_features: torch.Tensor,
        breakout_features: torch.Tensor | None = None,
    ) -> HiveMindOutput:
        """
        Full forward pass.

        Args:
            market_features: (batch, seq, n_continuous_vars) — historical market data
            time_features: (batch, seq_future, n_known_future_vars) — time encodings
            trend_features: (batch, d_trend_features) — for trend agent
            reversion_features: (batch, d_reversion_features) — for reversion agent
            regime_features: (batch, d_regime_features) — for gating network
            breakout_features: (batch, d_breakout_features) — for breakout agent
                If None, a zero tensor is used (backward compatible).
        """
        batch = market_features.size(0)

        # 1. TFT: encode market state
        tft_out = self.tft(
            x_past=market_features,
            x_future=time_features,
        )

        # 2. Agent outputs
        trend_action = self.trend_agent(tft_out.encoded_state, trend_features)
        reversion_action = self.reversion_agent(tft_out.encoded_state, reversion_features)

        if breakout_features is None:
            breakout_features = torch.zeros(
                batch, self.breakout_agent.d_input - self.d_model,
                device=market_features.device,
            )
        breakout_action = self.breakout_agent(tft_out.encoded_state, breakout_features)

        # 3. Gating: combine all 3 agents
        agent_outputs = torch.cat([trend_action, reversion_action, breakout_action], dim=-1)
        gating_weights, combined_action = self.gating(
            regime_features, tft_out.encoded_state, agent_outputs
        )

        return HiveMindOutput(
            action=combined_action,
            trend_action=trend_action,
            reversion_action=reversion_action,
            breakout_action=breakout_action,
            gating_weights=gating_weights,
            tft_attention=tft_out.attention_weights,
            variable_importance=tft_out.variable_importance,
            encoded_state=tft_out.encoded_state,
        )


class HiveMindExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible feature extractor wrapping the HiveMind.

    Converts the flat gymnasium observation into structured tensors
    for the HiveMind, then returns the encoded features for the
    SB3 actor-critic heads.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        n_continuous_vars: int = 30,
        n_known_future_vars: int = 5,
        d_model: int = 64,
        seq_len: int = 100,
        d_trend_features: int = 8,
        d_reversion_features: int = 8,
        d_breakout_features: int = 10,
        d_regime_features: int = 6,
    ) -> None:
        # encoded_state + 3 agent_actions + 3 gating weights + 4 position_state
        features_dim = d_model + 3 + 3 + 4
        super().__init__(observation_space, features_dim=features_dim)

        self.n_continuous_vars = n_continuous_vars
        self.n_known_future_vars = n_known_future_vars
        self.seq_len = seq_len
        self.d_trend_features = d_trend_features
        self.d_reversion_features = d_reversion_features
        self.d_breakout_features = d_breakout_features
        self.d_regime_features = d_regime_features

        self.hive_mind = HiveMind(
            n_continuous_vars=n_continuous_vars,
            n_known_future_vars=n_known_future_vars,
            d_model=d_model,
            d_trend_features=d_trend_features,
            d_reversion_features=d_reversion_features,
            d_breakout_features=d_breakout_features,
            d_regime_features=d_regime_features,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convert dict observation to HiveMind features for SB3 policy heads.

        Expected observation keys:
            market_features: (batch, seq_len * n_continuous_vars)
            time_features: (batch, seq_len * n_known_future_vars)
            trend_features: (batch, d_trend_features)
            reversion_features: (batch, d_reversion_features)
            regime_features: (batch, d_regime_features)
            breakout_features: (batch, d_breakout_features) [optional]
            position_state: (batch, 4)
        """
        batch = observations["market_features"].shape[0]

        # Reshape flat observations to structured tensors
        market = observations["market_features"].reshape(
            batch, self.seq_len, self.n_continuous_vars
        )
        time_feat = observations["time_features"].reshape(
            batch, self.seq_len, self.n_known_future_vars
        )
        trend = observations["trend_features"]
        reversion = observations["reversion_features"]
        regime = observations["regime_features"]
        position_state = observations["position_state"]

        # Breakout features (optional for backward compat)
        breakout = observations.get("breakout_features")

        # Run HiveMind
        out = self.hive_mind(
            market, time_feat, trend, reversion, regime,
            breakout_features=breakout,
        )

        # Concatenate for SB3: encoded state + agent actions + gating + position
        features = torch.cat([
            out.encoded_state,       # (batch, d_model)
            out.trend_action,        # (batch, 1)
            out.reversion_action,    # (batch, 1)
            out.breakout_action,     # (batch, 1)
            out.gating_weights,      # (batch, 3)
            position_state,          # (batch, 4)
        ], dim=-1)

        return features


# ---------------------------------------------------------------------------
# Multi-Timeframe HiveMind
# ---------------------------------------------------------------------------


@dataclass
class MTFHiveMindOutput:
    """Full output from the MTF HiveMind ensemble."""

    action: torch.Tensor              # (batch, 1) — final combined action
    trend_action: torch.Tensor        # (batch, 1)
    reversion_action: torch.Tensor    # (batch, 1)
    breakout_action: torch.Tensor     # (batch, 1)
    gating_weights: torch.Tensor      # (batch, 3) — agent weights
    tf_attention_weights: torch.Tensor  # (batch, 3) — timeframe attention
    tft_attention: torch.Tensor       # attention from H1 TFT pass
    variable_importance: torch.Tensor  # variable importance from H1 TFT pass
    encoded_state: torch.Tensor       # (batch, d_model) — fused state


class MTFHiveMind(nn.Module):
    """Multi-Timeframe HiveMind: Shared TFT + Timeframe Embedding + Fusion.

    Architecture:
        D1 bars → [Shared TFT + tf_embed=0] → d1_state ─┐
        H1 bars → [Shared TFT + tf_embed=1] → h1_state ─┤→ CrossTFFusion → fused → Agents → Gating
        M5 bars → [Shared TFT + tf_embed=2] → m5_state ─┘

    The TFT is shared across timeframes (parameter-efficient). A learned
    timeframe embedding is added to the input to distinguish D1/H1/M5.
    """

    def __init__(
        self,
        n_continuous_vars: int,
        n_known_future_vars: int = 5,
        d_model: int = 32,
        n_heads: int = 4,
        d_trend_features: int = 8,
        d_reversion_features: int = 8,
        d_breakout_features: int = 10,
        d_regime_features: int = 6,
        n_timeframes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_continuous_vars = n_continuous_vars
        self.n_known_future_vars = n_known_future_vars

        # Shared TFT with static_vars=1 for timeframe embedding
        self.tft = TemporalFusionTransformer(
            n_continuous_vars=n_continuous_vars,
            n_known_future_vars=n_known_future_vars,
            n_static_vars=1,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Timeframe embedding: 0=D1, 1=H1, 2=M5
        self.tf_embedding = nn.Embedding(n_timeframes, 1)

        # Cross-Timeframe Fusion
        self.fusion = CrossTimeframeFusion(
            d_model=d_model,
            n_timeframes=n_timeframes,
            dropout=dropout,
        )

        # Specialist Agents (same as HiveMind)
        self.trend_agent = TrendAgent(
            d_tft=d_model,
            d_trend_features=d_trend_features,
            dropout=dropout,
        )

        self.reversion_agent = ReversionAgent(
            d_tft=d_model,
            d_reversion_features=d_reversion_features,
            dropout=dropout,
        )

        self.breakout_agent = BreakoutAgent(
            d_tft=d_model,
            d_breakout_features=d_breakout_features,
            dropout=dropout,
        )

        # Gating Network
        self.gating = GatingNetwork(
            d_regime=d_regime_features,
            d_context=d_model,
            n_agents=3,
        )

    def _run_tft(
        self,
        market_features: torch.Tensor,
        time_features: torch.Tensor,
        tf_idx: int,
    ):
        """Run shared TFT for one timeframe.

        Parameters
        ----------
        market_features : (batch, seq, n_continuous_vars)
        time_features : (batch, seq, n_known_future_vars)
        tf_idx : int — timeframe index (0=D1, 1=H1, 2=M5)

        Returns
        -------
        TFTOutput
        """
        batch = market_features.size(0)
        device = market_features.device

        # Timeframe embedding as static variable
        tf_id = torch.full((batch,), tf_idx, dtype=torch.long, device=device)
        tf_static = self.tf_embedding(tf_id)  # (batch, 1)

        return self.tft(
            x_past=market_features,
            x_future=time_features,
            x_static=tf_static,
        )

    def forward(
        self,
        d1_market: torch.Tensor,
        d1_time: torch.Tensor,
        h1_market: torch.Tensor,
        h1_time: torch.Tensor,
        m5_market: torch.Tensor,
        m5_time: torch.Tensor,
        trend_features: torch.Tensor,
        reversion_features: torch.Tensor,
        regime_features: torch.Tensor,
        mtf_context: torch.Tensor | None = None,
        breakout_features: torch.Tensor | None = None,
    ) -> MTFHiveMindOutput:
        """Full MTF forward pass."""
        batch = h1_market.size(0)

        # Run shared TFT for each timeframe
        d1_out = self._run_tft(d1_market, d1_time, tf_idx=0)
        h1_out = self._run_tft(h1_market, h1_time, tf_idx=1)
        m5_out = self._run_tft(m5_market, m5_time, tf_idx=2)

        # Cross-Timeframe Fusion
        fused_state, tf_attn_weights = self.fusion(
            d1_out.encoded_state,
            h1_out.encoded_state,
            m5_out.encoded_state,
            mtf_context=mtf_context,
        )

        # Agent outputs (using fused state)
        trend_action = self.trend_agent(fused_state, trend_features)
        reversion_action = self.reversion_agent(fused_state, reversion_features)

        if breakout_features is None:
            breakout_features = torch.zeros(
                batch, self.breakout_agent.d_input - self.d_model,
                device=h1_market.device,
            )
        breakout_action = self.breakout_agent(fused_state, breakout_features)

        # Gating
        agent_outputs = torch.cat([trend_action, reversion_action, breakout_action], dim=-1)
        gating_weights, combined_action = self.gating(
            regime_features, fused_state, agent_outputs,
        )

        return MTFHiveMindOutput(
            action=combined_action,
            trend_action=trend_action,
            reversion_action=reversion_action,
            breakout_action=breakout_action,
            gating_weights=gating_weights,
            tf_attention_weights=tf_attn_weights,
            tft_attention=h1_out.attention_weights,
            variable_importance=h1_out.variable_importance,
            encoded_state=fused_state,
        )


class MTFHiveMindExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor for MTF HiveMind.

    Unpacks the MTF observation dict → reshapes → runs MTFHiveMind → returns
    feature vector for SB3 actor-critic heads.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        n_continuous_vars: int = 30,
        n_known_future_vars: int = 5,
        d_model: int = 32,
        d1_lookback: int = 5,
        h1_lookback: int = 20,
        m5_lookback: int = 20,
        d_trend_features: int = 8,
        d_reversion_features: int = 8,
        d_breakout_features: int = 10,
        d_regime_features: int = 6,
        n_mtf_context: int = 6,
    ) -> None:
        # features_dim = fused_state + 3 agent_actions + 3 gating
        # + 3 tf_attn + 4 position + 6 mtf_context
        features_dim = d_model + 3 + 3 + 3 + 4 + n_mtf_context
        super().__init__(observation_space, features_dim=features_dim)

        self.n_continuous_vars = n_continuous_vars
        self.n_known_future_vars = n_known_future_vars
        self.d1_lookback = d1_lookback
        self.h1_lookback = h1_lookback
        self.m5_lookback = m5_lookback
        self.n_mtf_context = n_mtf_context

        self.hive_mind = MTFHiveMind(
            n_continuous_vars=n_continuous_vars,
            n_known_future_vars=n_known_future_vars,
            d_model=d_model,
            d_trend_features=d_trend_features,
            d_reversion_features=d_reversion_features,
            d_breakout_features=d_breakout_features,
            d_regime_features=d_regime_features,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Unpack MTF obs dict → MTFHiveMind → SB3 feature vector."""
        batch = observations["h1_market_features"].shape[0]

        # Reshape flat vectors to (batch, seq, features)
        d1_market = observations["d1_market_features"].reshape(
            batch, self.d1_lookback, self.n_continuous_vars
        )
        d1_time = observations["d1_time_features"].reshape(
            batch, self.d1_lookback, self.n_known_future_vars
        )
        h1_market = observations["h1_market_features"].reshape(
            batch, self.h1_lookback, self.n_continuous_vars
        )
        h1_time = observations["h1_time_features"].reshape(
            batch, self.h1_lookback, self.n_known_future_vars
        )
        m5_market = observations["m5_market_features"].reshape(
            batch, self.m5_lookback, self.n_continuous_vars
        )
        m5_time = observations["m5_time_features"].reshape(
            batch, self.m5_lookback, self.n_known_future_vars
        )

        trend = observations["trend_features"]
        reversion = observations["reversion_features"]
        regime = observations["regime_features"]
        position_state = observations["position_state"]
        mtf_context = observations["mtf_context"]

        # Run MTF HiveMind
        out = self.hive_mind(
            d1_market, d1_time,
            h1_market, h1_time,
            m5_market, m5_time,
            trend, reversion, regime,
            mtf_context=mtf_context,
        )

        # Concatenate for SB3
        features = torch.cat([
            out.encoded_state,         # (batch, d_model)
            out.trend_action,          # (batch, 1)
            out.reversion_action,      # (batch, 1)
            out.breakout_action,       # (batch, 1)
            out.gating_weights,        # (batch, 3)
            out.tf_attention_weights,  # (batch, 3)
            position_state,            # (batch, 4)
            mtf_context,               # (batch, 6)
        ], dim=-1)

        return features
