"""Tensor Network (MPS/PEPS) dynamics backend.

Implements Matrix Product State contraction for compact, expressive
dynamics modelling. MPS naturally captures local correlations with
bounded entanglement, making it well-suited for sequential financial
data where nearby time steps are most correlated.

For forex: think of bond dimension as "memory capacity" — higher bond
dim captures more complex temporal patterns but costs O(d * bond²).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from apexfx.models.config import TensorNetworkBackendConfig


class MPSLayer(nn.Module):
    """Single Matrix Product State contraction layer.

    Represents a linear map via a chain of 3-index tensors:
    T_1[i, α₁] · T_2[α₁, i, α₂] · ... · T_n[α_{n-1}, i]

    For dynamics: the input vector is contracted site-by-site through
    the MPS chain, producing an output vector of the same dimension.

    Args:
        d_phys: Physical dimension (input/output feature dim per site).
        bond_dim: Bond dimension controlling expressiveness.
        n_sites: Number of sites in the MPS chain.
        use_canonical: Use left-canonical initialisation for stability.
    """

    def __init__(
        self, d_phys: int, bond_dim: int, n_sites: int,
        use_canonical: bool = True,
    ) -> None:
        super().__init__()
        self.d_phys = d_phys
        self.bond_dim = bond_dim
        self.n_sites = n_sites

        # MPS tensors: site k has shape (bond_left, d_phys, bond_right)
        # Boundary sites have bond_dim=1 on the open end
        self.cores = nn.ParameterList()
        for k in range(n_sites):
            bl = 1 if k == 0 else bond_dim
            br = 1 if k == n_sites - 1 else bond_dim
            core = nn.Parameter(torch.randn(bl, d_phys, br) * (1.0 / (bl * br) ** 0.5))
            self.cores.append(core)

        if use_canonical:
            self._init_canonical()

    def _init_canonical(self) -> None:
        """Left-canonical initialisation via QR decomposition."""
        with torch.no_grad():
            for k in range(self.n_sites - 1):
                core = self.cores[k].data
                bl, dp, br = core.shape
                mat = core.reshape(bl * dp, br)
                q, r = torch.linalg.qr(mat)
                self.cores[k].data = q.reshape(bl, dp, -1)
                # Absorb R into next core
                next_core = self.cores[k + 1].data
                nbl, ndp, nbr = next_core.shape
                next_mat = next_core.reshape(nbl, ndp * nbr)
                # r might have different size from nbl — contract
                contracted = r @ next_mat[:r.shape[1]]
                self.cores[k + 1].data = contracted.reshape(-1, ndp, nbr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contract input through MPS chain.

        Args:
            x: (batch, n_sites * d_phys) — flattened input.

        Returns:
            (batch, n_sites * d_phys) — contracted output.
        """
        batch = x.shape[0]
        # Reshape to (batch, n_sites, d_phys)
        x_sites = x.reshape(batch, self.n_sites, self.d_phys)

        # Contract through the chain
        # Start: bond vector of dim 1
        result_sites = []
        bond_vec = torch.ones(batch, 1, device=x.device)

        for k in range(self.n_sites):
            core = self.cores[k]  # (bl, dp, br)
            x_k = x_sites[:, k, :]  # (batch, dp)

            # Contract: bond_vec @ core contracted with x_k
            # einsum: b=batch, l=bond_left, p=phys, r=bond_right
            contracted = torch.einsum("bl,lpr,bp->br", bond_vec, core, x_k)
            bond_vec = contracted

            # Output at this site: project bond back to physical
            # Use the core transposed for output
            out_k = torch.einsum("br,lpr->bp", contracted,
                                 core[:contracted.shape[1], :, :])
            result_sites.append(out_k)

        # Stack and flatten back
        result = torch.stack(result_sites, dim=1)  # (batch, n_sites, d_phys)
        return result.reshape(batch, -1)


class TensorNetworkBackend(nn.Module):
    """Tensor Network dynamics backend for WorldModelHybrid.

    Maps (z_t, a_t) → z_{t+1} using stacked MPS layers
    with residual connections and layer normalisation.
    """

    def __init__(
        self, d_latent: int, d_action: int,
        cfg: TensorNetworkBackendConfig,
    ) -> None:
        super().__init__()
        d_input = d_latent + d_action

        # Project to a dimension divisible by n_sites
        # Choose n_sites = bond_dim for balanced architecture
        n_sites = max(4, d_input // 4)
        d_phys = max(1, d_input // n_sites)
        d_padded = n_sites * d_phys

        self.input_proj = nn.Linear(d_input, d_padded)
        self.norm_in = nn.LayerNorm(d_padded)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(
                MPSLayer(d_phys, cfg.bond_dim, n_sites, cfg.use_canonical),
            )
            self.norms.append(nn.LayerNorm(d_padded))

        self.output_proj = nn.Linear(d_padded, d_latent)
        self.residual = nn.Linear(d_latent, d_latent, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next latent via MPS contraction."""
        x = torch.cat([z, action], dim=-1)
        x = self.norm_in(self.input_proj(x))

        for layer, norm in zip(self.layers, self.norms, strict=False):
            x = x + self.dropout(layer(x))
            x = norm(x)

        delta = self.output_proj(x)
        return self.residual(z) + delta
