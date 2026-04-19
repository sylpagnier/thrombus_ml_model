"""Implicit neural representation decoder for kinematics (optional SIREN path)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self, w0: float = 30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SIRENDecoder(nn.Module):
    """
    Maps latent node features + (x, y) coordinates to (u, v, p).
    Coordinates may retain ``requires_grad`` when the encoder passes a leaf tensor (e.g. hard-BC paths).

    Uses Sitzmann et al. SIREN weight initialization (not Kaiming) so sine activations stay stable.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.w0_initial = 30.0
        self.w0_hidden = 1.0

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            Sine(w0=self.w0_initial),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(w0=self.w0_hidden),
            nn.Linear(hidden_dim, 3),
        )
        self._initialize_siren()

    def _initialize_siren(self) -> None:
        """Sitzmann et al. SIREN init: first layer U(±1/n); later linears U(±√(6/n)/ω₀) for the matching sine."""
        with torch.no_grad():
            first = self.net[0]
            assert isinstance(first, nn.Linear)
            n_in = first.in_features
            first.weight.uniform_(-1.0 / n_in, 1.0 / n_in)
            if first.bias is not None:
                first.bias.zero_()

            # Linear before second sine: ω₀ = w0_hidden
            mid = self.net[2]
            assert isinstance(mid, nn.Linear)
            w_std = math.sqrt(6.0 / mid.in_features) / self.w0_hidden
            mid.weight.uniform_(-w_std, w_std)
            if mid.bias is not None:
                mid.bias.zero_()

            # Final readout after last sine: same scale as linear-after-sine in typical SIREN stacks
            last = self.net[4]
            assert isinstance(last, nn.Linear)
            w_std_out = math.sqrt(6.0 / last.in_features) / self.w0_hidden
            last.weight.uniform_(-w_std_out, w_std_out)
            if last.bias is not None:
                last.bias.zero_()

    def forward(self, z: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not pos.requires_grad:
            pos = pos.clone()
            pos.requires_grad_(True)
        z_pos = torch.cat([z, pos], dim=-1)
        uvp = self.net(z_pos)
        return uvp, pos
