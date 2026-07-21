"""Spectral-normalized linear layer for Lipschitz-bounded DEQ stacks."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class SpectralLinear(nn.Module):
    """Linear layer with spectral norm for stable DEQ fixed-point solves."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        spectral_norm(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


__all__ = ["SpectralLinear"]
