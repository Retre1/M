"""HiveMind Ensemble — the unified model connecting TFT, agents, and gating.

Architecture:

1. TFT encodes market state from historical features.
2. Each specialist agent computes a **hidden representation** from TFT state
   + its domain-specific features.
3. **Cross-Agent Attention** allows agents to communicate — each agent's hidden
   attends to every other agent's hidden before producing a final action.
4. The **two-stream Gating Network** evaluates both market regime and agent
   proposals to produce mixing weights.
5. Weighted combination yields the final action for SB3.

Supports both single-timeframe and Multi-Timeframe (MTF) modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from apexfx.models.agents.breakout_agent import BreakoutAgent
from apexfx.models.agents.reversion_agent import ReversionAgent
from apexfx.models.agents.trend_agent import TrendAgent
from apexfx.models.ensemble.cross_agent_attention import CrossAgentAttention
from apexfx.models.ensemble.cross_tf_fusion import CrossTimeframeFusion
from apexfx.models.ensemble.gating_network import GatingNetwork
from apexfx.models.tft.tft_model import TemporalFusionTransformer


@dataclass
class HiveMindOutput:
    """Full output from the HiveMind ensemble."""

    action: torch.Tensor              # (batch, 1) — final combined action
    trend_action: torch.Tensor        # (batch, 1) — trend agent's action
    reversion_action: torch.Tensor    # (batch, 1) — reversion agent's action
    breakout_action: torch.Tensor     # (batch, 1) — breakout agent's action
    gating_weights: torch.Tensor      # (batch, 3) — meta-controller weights
    agent_attention: torch.Tensor     # (batch, 3, 3) — cross-agent attention
    tft_attention: torch.Tensor       # attention weight matrices
    variable_importance: torch.Tensor  # TFT variable importance scores
    encoded_state: torch.Tensor       # (batch, d_model) — TFT encoded state


class HiveMind(nn.Module):
    """TFT → Agent Encode → Cross-Agent Attention → Agent Act → Gating.

    The cross-agent attention step is the key innovation: agents share
    information with each other before committing to a final action.
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
        d_fundamental_features: int = 8,
        d_structure_features: int = 8,
        dropout: float = 0.1,
        # Cross-Agent Attention params
        d_cross_attn: int = 64,
        n_cross_attn_heads: int = 2,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_fundamental_features = d_fundamental_features
        self.d_structure_features = d_structure_features

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

        # Breakout agent receives structure features appended to breakout features
        self.breakout_agent = BreakoutAgent(
            d_tft=d_model,
            d_breakout_features=d_breakout_features + d_structure_features,
            dropout=dropout,
        )

        # --- Cross-Agent Attention ---
        self.cross_agent_attention = CrossAgentAttention(
            agent_hidden_dims=[
                self.trend_agent.hidden_dim,
                self.reversion_agent.hidden_dim,
                self.breakout_agent.hidden_dim,
            ],
            d_attn=d_cross_attn,
            n_heads=n_cross_attn_heads,
            dropout=dropout,
        )

        # --- Gating Network (Meta-Controller) ---
        # Gating sees regime + fundamental features for news-aware decisions
        self.gating = GatingNetwork(
            d_regime=d_regime_features + d_fundamental_features,
            d_context=d_model,
            n_agents=3,
            dropout=dropout,
        )

    def forward(
        self,
        market_features: torch.Tensor,
        time_features: torch.Tensor,
        trend_features: torch.Tensor,
        reversion_features: torch.Tensor,
        regime_features: torch.Tensor,
        breakout_features: torch.Tensor | None = None,
        fundamental_features: torch.Tensor | None = None,
        structure_features: torch.Tensor | None = None,
    ) -> HiveMindOutput:
        """Full forward pass with cross-agent communication.

        Args:
            market_features: (batch, seq, n_continuous_vars)
            time_features: (batch, seq, n_known_future_vars)
            trend_features: (batch, d_trend_features)
            reversion_features: (batch, d_reversion_features)
            regime_features: (batch, d_regime_features)
            breakout_features: (batch, d_breakout_features) or None
            fundamental_features: (batch, d_fundamental_features) or None
            structure_features: (batch, d_structure_features) or None
        """
        batch = market_features.size(0)
        device = market_features.device

        # 1. TFT — encode market state
        tft_out = self.tft(x_past=market_features, x_future=time_features)

        # 2a. Agent encode — hidden representations (before action heads)
        # Concat structure features with breakout features
        if breakout_features is None:
            brk_dim = self.breakout_agent.d_input - self.d_model - self.d_structure_features
            breakout_features = torch.zeros(batch, brk_dim, device=device)
        if structure_features is None:
            structure_features = torch.zeros(batch, self.d_structure_features, device=device)
        combined_breakout = torch.cat([breakout_features, structure_features], dim=-1)

        # Fundamental features default
        if fundamental_features is None:
            fundamental_features = torch.zeros(batch, self.d_fundamental_features, device=device)

        trend_hidden = self.trend_agent.encode(tft_out.encoded_state, trend_features)
        rev_hidden = self.reversion_agent.encode(tft_out.encoded_state, reversion_features)
        brk_hidden = self.breakout_agent.encode(tft_out.encoded_state, combined_breakout)

        # 2b. Cross-Agent Attention — agents communicate
        [trend_enriched, rev_enriched, brk_enriched], agent_attn = \
            self.cross_agent_attention([trend_hidden, rev_hidden, brk_hidden])

        # 2c. Agent act — final actions from enriched representations
        trend_action = self.trend_agent.act(trend_enriched)
        reversion_action = self.reversion_agent.act(
            rev_enriched, specialist_features=reversion_features,
        )
        breakout_action = self.breakout_agent.act(
            brk_enriched, specialist_features=combined_breakout,
        )

        # Cache for diversity regularization
        self._cached_trend_action = trend_action
        self._cached_reversion_action = reversion_action
        self._cached_breakout_action = breakout_action

        # 3. Gating — two-stream agent-aware meta-controller
        # Concat regime + fundamental features for news-aware gating
        gating_regime = torch.cat([regime_features, fundamental_features], dim=-1)

        agent_outputs = torch.cat(
            [trend_action, reversion_action, breakout_action], dim=-1,
        )
        gating_weights, combined_action = self.gating(
            gating_regime, tft_out.encoded_state, agent_outputs,
        )

        self._cached_gating_weights = gating_weights

        return HiveMindOutput(
            action=combined_action,
            trend_action=trend_action,
            reversion_action=reversion_action,
            breakout_action=breakout_action,
            gating_weights=gating_weights,
            agent_attention=agent_attn,
            tft_attention=tft_out.attention_weights,
            variable_importance=tft_out.variable_importance,
            encoded_state=tft_out.encoded_state,
        )


class HiveMindExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor wrapping the HiveMind.

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
        d_fundamental_features: int = 8,
        d_structure_features: int = 8,
    ) -> None:
        # encoded_state + 3 agent_actions + 3 gating weights + 8 position_state
        features_dim = d_model + 3 + 3 + 8
        super().__init__(observation_space, features_dim=features_dim)

        self.n_continuous_vars = n_continuous_vars
        self.n_known_future_vars = n_known_future_vars
        self.seq_len = seq_len
        self.d_trend_features = d_trend_features
        self.d_reversion_features = d_reversion_features
        self.d_breakout_features = d_breakout_features
        self.d_regime_features = d_regime_features
        self.d_fundamental_features = d_fundamental_features
        self.d_structure_features = d_structure_features

        self.hive_mind = HiveMind(
            n_continuous_vars=n_continuous_vars,
            n_known_future_vars=n_known_future_vars,
            d_model=d_model,
            d_trend_features=d_trend_features,
            d_reversion_features=d_reversion_features,
            d_breakout_features=d_breakout_features,
            d_regime_features=d_regime_features,
            d_fundamental_features=d_fundamental_features,
            d_structure_features=d_structure_features,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert dict observation → HiveMind → SB3 feature vector."""
        batch = observations["market_features"].shape[0]

        market = observations["market_features"].reshape(
            batch, self.seq_len, self.n_continuous_vars,
        )
        time_feat = observations["time_features"].reshape(
            batch, self.seq_len, self.n_known_future_vars,
        )
        trend = observations["trend_features"]
        reversion = observations["reversion_features"]
        regime = observations["regime_features"]
        position_state = observations["position_state"]
        breakout = observations.get("breakout_features")
        fundamental = observations.get("fundamental_features")
        structure = observations.get("structure_features")

        out = self.hive_mind(
            market, time_feat, trend, reversion, regime,
            breakout_features=breakout,
            fundamental_features=fundamental,
            structure_features=structure,
        )

        features = torch.cat([
            out.encoded_state,       # (batch, d_model)
            out.trend_action,        # (batch, 1)
            out.reversion_action,    # (batch, 1)
            out.breakout_action,     # (batch, 1)
            out.gating_weights,      # (batch, 3)
            position_state,          # (batch, 8)
        ], dim=-1)

        return features


# ---------------------------------------------------------------------------
# Multi-Timeframe HiveMind
# ---------------------------------------------------------------------------


@dataclass
class MTFHiveMindOutput:
    """Full output from the MTF HiveMind ensemble."""

    action: torch.Tensor              # (batch, 1)
    trend_action: torch.Tensor        # (batch, 1)
    reversion_action: torch.Tensor    # (batch, 1)
    breakout_action: torch.Tensor     # (batch, 1)
    gating_weights: torch.Tensor      # (batch, 3)
    agent_attention: torch.Tensor     # (batch, 3, 3) — cross-agent attention
    tf_attention_weights: torch.Tensor  # (batch, 3) — timeframe attention
    tft_attention: torch.Tensor       # attention from H1 TFT pass
    variable_importance: torch.Tensor  # variable importance from H1 TFT pass
    encoded_state: torch.Tensor       # (batch, d_model) — fused state


class MTFHiveMind(nn.Module):
    """Multi-Timeframe HiveMind with Cross-Agent Attention.

    Architecture::

        D1 bars → [Shared TFT + tf_embed=0] → d1_state ─┐
        H1 bars → [Shared TFT + tf_embed=1] → h1_state ─┤→ CrossTFFusion → fused_state
        M5 bars → [Shared TFT + tf_embed=2] → m5_state ─┘         │
                                                                    ↓
        fused_state + specialist_features → Agent.encode() per agent
                                                    │
                                             Cross-Agent Attention
                                                    │
                                             Agent.act() per agent
                                                    │
                                             Two-Stream Gating → action
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
        d_fundamental_features: int = 8,
        d_structure_features: int = 8,
        n_timeframes: int = 3,
        dropout: float = 0.1,
        d_cross_attn: int = 64,
        n_cross_attn_heads: int = 2,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_continuous_vars = n_continuous_vars
        self.n_known_future_vars = n_known_future_vars
        self.d_fundamental_features = d_fundamental_features
        self.d_structure_features = d_structure_features

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

        # Specialist Agents
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
            d_breakout_features=d_breakout_features + d_structure_features,
            dropout=dropout,
        )

        # Cross-Agent Attention
        self.cross_agent_attention = CrossAgentAttention(
            agent_hidden_dims=[
                self.trend_agent.hidden_dim,
                self.reversion_agent.hidden_dim,
                self.breakout_agent.hidden_dim,
            ],
            d_attn=d_cross_attn,
            n_heads=n_cross_attn_heads,
            dropout=dropout,
        )

        # Gating Network — sees regime + fundamental for news-aware decisions
        self.gating = GatingNetwork(
            d_regime=d_regime_features + d_fundamental_features,
            d_context=d_model,
            n_agents=3,
            dropout=dropout,
        )

    def _run_tft(
        self,
        market_features: torch.Tensor,
        time_features: torch.Tensor,
        tf_idx: int,
    ):
        """Run shared TFT for one timeframe."""
        batch = market_features.size(0)
        device = market_features.device
        tf_id = torch.full((batch,), tf_idx, dtype=torch.long, device=device)
        tf_static = self.tf_embedding(tf_id)
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
        fundamental_features: torch.Tensor | None = None,
        structure_features: torch.Tensor | None = None,
    ) -> MTFHiveMindOutput:
        """Full MTF forward pass with cross-agent communication."""
        batch = h1_market.size(0)
        device = h1_market.device

        # --- Multi-Timeframe TFT + Fusion ---
        d1_out = self._run_tft(d1_market, d1_time, tf_idx=0)
        h1_out = self._run_tft(h1_market, h1_time, tf_idx=1)
        m5_out = self._run_tft(m5_market, m5_time, tf_idx=2)

        fused_state, tf_attn_weights = self.fusion(
            d1_out.encoded_state,
            h1_out.encoded_state,
            m5_out.encoded_state,
            mtf_context=mtf_context,
        )

        # --- Agent encode ---
        if breakout_features is None:
            brk_dim = self.breakout_agent.d_input - self.d_model - self.d_structure_features
            breakout_features = torch.zeros(batch, brk_dim, device=device)
        if structure_features is None:
            structure_features = torch.zeros(batch, self.d_structure_features, device=device)
        combined_breakout = torch.cat([breakout_features, structure_features], dim=-1)

        if fundamental_features is None:
            fundamental_features = torch.zeros(batch, self.d_fundamental_features, device=device)

        trend_hidden = self.trend_agent.encode(fused_state, trend_features)
        rev_hidden = self.reversion_agent.encode(fused_state, reversion_features)
        brk_hidden = self.breakout_agent.encode(fused_state, combined_breakout)

        # --- Cross-Agent Attention ---
        [trend_enriched, rev_enriched, brk_enriched], agent_attn = \
            self.cross_agent_attention([trend_hidden, rev_hidden, brk_hidden])

        # --- Agent act ---
        trend_action = self.trend_agent.act(trend_enriched)
        reversion_action = self.reversion_agent.act(
            rev_enriched, specialist_features=reversion_features,
        )
        breakout_action = self.breakout_agent.act(
            brk_enriched, specialist_features=combined_breakout,
        )

        # Cache for diversity regularization
        self._cached_trend_action = trend_action
        self._cached_reversion_action = reversion_action
        self._cached_breakout_action = breakout_action

        # --- Gating (news-aware) ---
        gating_regime = torch.cat([regime_features, fundamental_features], dim=-1)
        agent_outputs = torch.cat(
            [trend_action, reversion_action, breakout_action], dim=-1,
        )
        gating_weights, combined_action = self.gating(
            gating_regime, fused_state, agent_outputs,
        )
        self._cached_gating_weights = gating_weights

        return MTFHiveMindOutput(
            action=combined_action,
            trend_action=trend_action,
            reversion_action=reversion_action,
            breakout_action=breakout_action,
            gating_weights=gating_weights,
            agent_attention=agent_attn,
            tf_attention_weights=tf_attn_weights,
            tft_attention=h1_out.attention_weights,
            variable_importance=h1_out.variable_importance,
            encoded_state=fused_state,
        )


class MTFHiveMindExtractor(BaseFeaturesExtractor):
    """SB3-compatible feature extractor for MTF HiveMind."""

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
        d_fundamental_features: int = 8,
        d_structure_features: int = 8,
        n_mtf_context: int = 6,
    ) -> None:
        # encoded_state + 3 agents + 3 gating + 3 tf_attn + 8 position + mtf_context
        features_dim = d_model + 3 + 3 + 3 + 8 + n_mtf_context
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
            d_fundamental_features=d_fundamental_features,
            d_structure_features=d_structure_features,
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Unpack MTF obs dict → MTFHiveMind → SB3 feature vector."""
        batch = observations["h1_market_features"].shape[0]

        d1_market = observations["d1_market_features"].reshape(
            batch, self.d1_lookback, self.n_continuous_vars,
        )
        d1_time = observations["d1_time_features"].reshape(
            batch, self.d1_lookback, self.n_known_future_vars,
        )
        h1_market = observations["h1_market_features"].reshape(
            batch, self.h1_lookback, self.n_continuous_vars,
        )
        h1_time = observations["h1_time_features"].reshape(
            batch, self.h1_lookback, self.n_known_future_vars,
        )
        m5_market = observations["m5_market_features"].reshape(
            batch, self.m5_lookback, self.n_continuous_vars,
        )
        m5_time = observations["m5_time_features"].reshape(
            batch, self.m5_lookback, self.n_known_future_vars,
        )

        trend = observations["trend_features"]
        reversion = observations["reversion_features"]
        regime = observations["regime_features"]
        position_state = observations["position_state"]
        mtf_context = observations["mtf_context"]
        fundamental = observations.get("fundamental_features")
        structure = observations.get("structure_features")

        out = self.hive_mind(
            d1_market, d1_time,
            h1_market, h1_time,
            m5_market, m5_time,
            trend, reversion, regime,
            mtf_context=mtf_context,
            fundamental_features=fundamental,
            structure_features=structure,
        )

        features = torch.cat([
            out.encoded_state,         # (batch, d_model)
            out.trend_action,          # (batch, 1)
            out.reversion_action,      # (batch, 1)
            out.breakout_action,       # (batch, 1)
            out.gating_weights,        # (batch, 3)
            out.tf_attention_weights,  # (batch, 3)
            position_state,            # (batch, 8)
            mtf_context,               # (batch, 6)
        ], dim=-1)

        return features
