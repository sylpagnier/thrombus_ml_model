"""LoRA parametrization and SpectralLinear for safe, dynamic low-rank adaptation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import spectral_norm


class LoRAParametrization(nn.Module):
    """Computes the low-rank additive weight delta applied to a frozen base weight."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scaling = alpha / rank

    def forward(self, original_weight: torch.Tensor) -> torch.Tensor:
        return original_weight + (self.lora_B @ self.lora_A) * self.scaling


class SpectralLinear(nn.Module):
    """Linear layer with spectral norm; supports computationally safe LoRA injection."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        spectral_norm(self.linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def inject_lora(self, rank: int = 4, alpha: float = 1.0) -> None:
        in_features = self.linear.in_features
        out_features = self.linear.out_features
        torch.nn.utils.remove_spectral_norm(self.linear)
        parametrize.register_parametrization(
            self.linear,
            "weight",
            LoRAParametrization(in_features, out_features, rank, alpha),
        )
        spectral_norm(self.linear)
        self.linear.parametrizations.weight.original.requires_grad = False


def inject_lora_to_spectral_linears(module: nn.Module, rank: int = 4, alpha: float = 1.0) -> int:
    """Call :meth:`SpectralLinear.inject_lora` on every nested :class:`SpectralLinear`. Returns count."""
    n = 0
    for child in module.modules():
        if isinstance(child, SpectralLinear):
            child.inject_lora(rank=rank, alpha=alpha)
            n += 1
    return n


def inject_lora_to_kinematics(
    kinematics_module: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    only: Optional[str] = None,
) -> int:
    """
    Attach LoRA to frozen kinematics layers (typically :class:`SpectralLinear` inside the DEQ stack).

    If ``only`` is set (substring), only module names containing that substring are considered.
    """
    n = 0
    for name, layer in kinematics_module.named_modules():
        if only is not None and only not in name:
            continue
        if isinstance(layer, SpectralLinear):
            layer.inject_lora(rank=rank, alpha=alpha)
            n += 1
    return n


__all__ = [
    "LoRAParametrization",
    "SpectralLinear",
    "inject_lora_to_spectral_linears",
    "inject_lora_to_kinematics",
]
