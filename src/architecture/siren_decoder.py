"""Implicit neural representation decoder for kinematics (optional SIREN path)."""

from __future__ import annotations

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
    Coordinates are cloned and marked for grad so callers can use autograd on spatial derivatives.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            Sine(w0=30.0),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(w0=1.0),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = pos.clone()
        pos.requires_grad_(True)
        z_pos = torch.cat([z, pos], dim=-1)
        uvp = self.net(z_pos)
        return uvp, pos
