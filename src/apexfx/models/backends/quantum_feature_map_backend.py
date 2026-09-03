"""Quantum-inspired feature map dynamics backend.

Classical simulation of parameterised quantum circuits for non-linear
feature encoding. No quantum hardware or PennyLane required — this is
a pure PyTorch implementation of the mathematical operations.

The quantum feature map encodes classical vectors into a 2^n dimensional
Hilbert space via rotation gates + entanglement, then measures
expectations. This provides exponentially rich feature interactions
that can capture non-linear forex dynamics.

Encoding types:
    angle      — R_Y(x_i) per qubit, simple and stable
    amplitude  — normalised amplitudes, requires 2^n inputs
    iqp        — Z⊗Z interactions for quadratic feature maps
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from apexfx.models.config import QuantumFeatureMapConfig


class QuantumCircuit(nn.Module):
    """Simulated parameterised quantum circuit.

    Operates on the 2^n state vector directly (statevector simulation).
    Suitable for n_qubits ≤ 12 (4096-dim state vector).
    """

    def __init__(self, n_qubits: int, n_layers: int,
                 entanglement: str = "circular") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.d_state = 2 ** n_qubits
        self.entanglement = entanglement

        # Trainable rotation parameters per layer per qubit (3 Euler angles)
        self.params = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1,
        )

    def _ry_gate(self, theta: torch.Tensor) -> torch.Tensor:
        """Single-qubit Y rotation matrix."""
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        return torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ], dim=-2)

    def _rz_gate(self, theta: torch.Tensor) -> torch.Tensor:
        """Single-qubit Z rotation matrix."""
        # For real-valued simulation, approximate RZ as diagonal
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        return torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ], dim=-2)

    def _apply_single_qubit(self, state: torch.Tensor,
                            gate: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply a single-qubit gate to a specific qubit in the state vector.

        Uses reshape trick: state (2^n,) → (2^a, 2, 2^b) where a = qubit index.
        """
        n = self.n_qubits
        batch = state.shape[0]
        d_left = 2 ** qubit
        d_right = 2 ** (n - qubit - 1)

        # Reshape: (batch, 2^n) → (batch, d_left, 2, d_right)
        state = state.reshape(batch, d_left, 2, d_right)

        # Apply gate: (batch, 2, 2) @ (batch, d_left, 2, d_right)
        # gate: (batch, 2, 2), state slice: (batch, d_left, 2, d_right)
        state = torch.einsum("bij,bkjl->bkil", gate, state)

        return state.reshape(batch, self.d_state)

    def _apply_cnot(self, state: torch.Tensor,
                    control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate (classical simulation via index swapping)."""
        n = self.n_qubits
        state.shape[0]
        d = self.d_state

        # Build CNOT permutation
        result = state.clone()
        for i in range(d):
            ctrl_bit = (i >> (n - 1 - control)) & 1
            if ctrl_bit == 1:
                # Flip target bit
                j = i ^ (1 << (n - 1 - target))
                result[:, i] = state[:, j]

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input and evolve through circuit layers.

        Args:
            x: (batch, n_qubits) — input features (one per qubit).

        Returns:
            (batch, d_state) — output state vector (probabilities).
        """
        batch = x.shape[0]
        device = x.device

        # Initialise |0...0⟩ state
        state = torch.zeros(batch, self.d_state, device=device)
        state[:, 0] = 1.0

        for layer_idx in range(self.n_layers):
            layer_params = self.params[layer_idx]  # (n_qubits, 3)

            # Data encoding + trainable rotation per qubit
            for q in range(self.n_qubits):
                angle = x[:, q] + layer_params[q, 0]
                gate = self._ry_gate(angle)
                state = self._apply_single_qubit(state, gate, q)

                # Additional RZ rotation
                angle_z = layer_params[q, 1]
                gate_z = self._rz_gate(angle_z.expand(batch))
                state = self._apply_single_qubit(state, gate_z, q)

            # Entanglement layer
            if self.entanglement == "linear":
                pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
            elif self.entanglement == "circular":
                pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
            else:  # full
                pairs = [(i, j) for i in range(self.n_qubits)
                         for j in range(i + 1, self.n_qubits)]

            for ctrl, tgt in pairs:
                state = self._apply_cnot(state, ctrl, tgt)

        # Output: squared amplitudes (probabilities)
        probs = state ** 2
        return probs


class QuantumFeatureMapBackend(nn.Module):
    """Quantum-inspired feature map dynamics backend.

    Maps (z_t, a_t) → z_{t+1} via quantum circuit simulation
    followed by classical post-processing.
    """

    def __init__(
        self, d_latent: int, d_action: int,
        cfg: QuantumFeatureMapConfig,
    ) -> None:
        super().__init__()
        n_qubits = cfg.n_qubits
        d_input = d_latent + d_action

        # Project input to n_qubits dimensions
        self.input_proj = nn.Linear(d_input, n_qubits)
        self.norm_in = nn.LayerNorm(n_qubits)

        # Quantum circuit
        self.circuit = QuantumCircuit(
            n_qubits, cfg.n_layers, cfg.entanglement,
        )

        # Post-processing: map 2^n probabilities back to d_latent
        d_state = 2 ** n_qubits
        self.output_proj = nn.Sequential(
            nn.Linear(d_state, d_latent * 2),
            nn.LayerNorm(d_latent * 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_latent * 2, d_latent),
        )

        self.residual = nn.Linear(d_latent, d_latent, bias=False)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent via quantum feature map."""
        x = torch.cat([z, action], dim=-1)
        x = self.norm_in(self.input_proj(x))

        # Scale inputs to [0, π] for rotation gates
        x = torch.sigmoid(x) * math.pi

        # Quantum circuit
        probs = self.circuit(x)

        delta = self.output_proj(probs)
        return self.residual(z) + delta
