"""Pluggable dynamics backends for WorldModelHybrid.

Each backend implements the same interface:
    forward(z: Tensor, action: Tensor) -> Tensor
    — maps (latent, action) → next latent prediction.

Available backends:
    MambaBackend          — Selective State Space (S6-style)
    TensorNetworkBackend  — MPS / PEPS contraction
    QuantumFeatureMapBackend — Quantum-inspired feature map + kernel
    HybridBackend         — Learned fusion of multiple backends
"""

from apexfx.models.backends.mamba_backend import MambaBackend
from apexfx.models.backends.quantum_feature_map_backend import QuantumFeatureMapBackend
from apexfx.models.backends.tensor_network_backend import TensorNetworkBackend

__all__ = [
    "MambaBackend",
    "TensorNetworkBackend",
    "QuantumFeatureMapBackend",
]
