"""Pydantic v2 frozen configs for World Model v2 and Gating v2.

All configuration is immutable after construction. Override via YAML
config files or constructor kwargs.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ── Backend selection ────────────────────────────────────────────────

class BackendType(str, Enum):
    """Dynamics backbone type for WorldModelHybrid."""
    MAMBA = "mamba"
    TENSOR_NETWORK = "tensor_network"
    QUANTUM_FEATURE_MAP = "quantum_feature_map"
    HYBRID = "hybrid"


# ── Backend-specific configs ─────────────────────────────────────────

class MambaBackendConfig(BaseModel, frozen=True):
    """Selective State Space (Mamba-style) dynamics backend."""
    d_state: int = 16          # SSM hidden state dimension
    d_conv: int = 4            # local convolution width
    expand_factor: int = 2     # inner dimension = d_model * expand_factor
    dt_rank: int = 0           # 0 = auto (ceil(d_model / 16))
    dt_min: float = 0.001
    dt_max: float = 0.1
    dropout: float = 0.1


class TensorNetworkBackendConfig(BaseModel, frozen=True):
    """Matrix Product State / PEPS tensor network backend."""
    bond_dim: int = 16         # MPS bond dimension
    n_layers: int = 2          # stacked MPS layers
    mode: Literal["mps", "peps"] = "mps"
    use_canonical: bool = True  # left-canonical form for stability
    dropout: float = 0.1


class QuantumFeatureMapConfig(BaseModel, frozen=True):
    """Quantum-inspired feature map + kernel backend."""
    n_qubits: int = 8
    n_layers: int = 4
    entanglement: Literal["linear", "full", "circular"] = "circular"
    encoding: Literal["angle", "amplitude", "iqp"] = "angle"
    measurement: Literal["expectation", "probabilities"] = "expectation"
    dropout: float = 0.1


class HybridBackendConfig(BaseModel, frozen=True):
    """Combines multiple backends with learned mixing weights."""
    backends: list[BackendType] = Field(
        default_factory=lambda: [BackendType.MAMBA, BackendType.TENSOR_NETWORK],
    )
    fusion: Literal["learned_gate", "concat", "attention"] = "learned_gate"


# ── World Model v2 ──────────────────────────────────────────────────

class WorldModelHybridConfig(BaseModel, frozen=True):
    """Configuration for WorldModelHybrid.

    The hybrid world model replaces the vanilla GRU dynamics with a
    pluggable backend (Mamba / TN / Quantum / Hybrid) and adds
    SBBTS-based imagination rollouts + DML-pricer integration.
    """
    # Architecture
    d_latent: int = 32
    d_hidden: int = 128
    d_action: int = 1
    n_ensemble: int = 5
    dropout: float = 0.1

    # Backend selection
    backend: BackendType = BackendType.MAMBA
    mamba: MambaBackendConfig = Field(default_factory=MambaBackendConfig)
    tensor_network: TensorNetworkBackendConfig = Field(
        default_factory=TensorNetworkBackendConfig,
    )
    quantum_feature_map: QuantumFeatureMapConfig = Field(
        default_factory=QuantumFeatureMapConfig,
    )
    hybrid: HybridBackendConfig = Field(default_factory=HybridBackendConfig)

    # Imagination rollouts
    imagination_horizon: int = 5
    sbbts_imagination: bool = True   # use SBBTS noise in imagination
    dml_pricer: bool = True          # attach DML jump-risk head

    # Training
    lr: float = 3e-4
    update_freq: int = 100
    batch_size: int = 256
    curiosity_weight: float = 0.01
    reward_pred_weight: float = 0.5
    done_pred_weight: float = 0.1


# ── Gating v2 ───────────────────────────────────────────────────────

class GatingV2Config(BaseModel, frozen=True):
    """Configuration for HiveMindGating_v2.

    Extends GatingNetwork with:
    - FSD regime conditioning (3 one-hot regime bits)
    - Entropy regularization to prevent gating collapse
    - Diversity loss across agent weights
    """
    # Architecture
    d_regime: int = 6
    d_context: int = 64
    n_agents: int = 3
    d_hidden: int = 64
    dropout: float = 0.1

    # FSD conditioning
    d_fsd: int = 4               # fsd_dispersion + 3 one-hot regime bits
    fsd_embed_dim: int = 16      # FSD embedding dimension

    # Anti-collapse mechanisms
    entropy_min: float = 0.12    # minimum gating entropy threshold
    entropy_weight: float = 0.1  # entropy regularization loss weight
    diversity_weight: float = 0.01  # diversity loss weight across agents
    temperature_init: float = 1.0
    temperature_min: float = 0.1
    temperature_max: float = 10.0

    # Meta-confidence
    confidence_dropout: float = 0.15  # higher dropout on confidence gate
