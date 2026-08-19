"""Parameter abstractions for the differentiable wall model.

Provides:
  - ParameterMap: Container holding the effective physical parameters (either [1] or [N]).
  - ParameterProvider: Abstract base class for supplying parameters.
  - GlobalPhysicsParameters (Level 1.1): Scalar nn.Parameters optimized globally.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ParameterMap:
    """Container for parameters required by the differentiable physics model.

    Each field can be a scalar tensor [1] (global) or a per-node tensor [N] (local).
    All tensors are in COMSOL/CGS units matching physics_wall_model.py.
    """
    da_scale: torch.Tensor       # Multiplier on surface Damkohler rate (default ~40.0)
    wake: torch.Tensor           # Stagnation wake strength (default ~8.0)
    lss: torch.Tensor            # Low-shear threshold in 1/s (default ~25.0)
    sgt_cgs: torch.Tensor        # Separation threshold in 1/(s*cm) (default ~ -750.0)
    tau_low: torch.Tensor        # Dimensionless softness margin for low-shear gate (default ~0.25)
    tau_sep: torch.Tensor        # Dimensionless softness margin for separation gate (default ~0.25)
    relax: torch.Tensor          # Growth front admission multiplier on lss (default ~2.0)
    phi_temp: torch.Tensor       # Temperature for soft clot readout sigmoid (default ~0.05)


class ParameterProvider(nn.Module, abc.ABC):
    """Abstract interface for producing physical parameters for a given graph."""

    @abc.abstractmethod
    def forward(self, data, num_nodes: int, device: torch.device) -> ParameterMap:
        """Return a ParameterMap for the specified graph."""
        pass


class GlobalPhysicsParameters(ParameterProvider):
    """Level 1.1: Learns global scalar parameters via unconstrained parameterizations.

    Uses smooth bijections (e.g. softplus or exp) so parameters strictly adhere to
    physical constraints (positivity, proper signs) during gradient descent.
    """

    def __init__(
        self,
        *,
        init_da_scale: float = 40.0,
        init_wake: float = 8.0,
        init_lss: float = 25.0,
        init_sgt_cgs: float = -750.0,
        init_tau_low: float = 0.25,
        init_tau_sep: float = 0.25,
        init_relax: float = 2.0,
        init_phi_temp: float = 0.05,
        trainable_keys: tuple[str, ...] = ("da_scale", "wake", "lss", "sgt_cgs", "tau_low", "relax"),
    ):
        super().__init__()
        self.trainable_keys = set(trainable_keys)

        # Internal raw parameters (mapped via softplus/exp to valid ranges)
        # da_scale > 0
        self.raw_da_scale = nn.Parameter(
            torch.tensor(self._inv_softplus(init_da_scale), dtype=torch.float32),
            requires_grad="da_scale" in self.trainable_keys,
        )
        # wake >= 0
        self.raw_wake = nn.Parameter(
            torch.tensor(self._inv_softplus(init_wake), dtype=torch.float32),
            requires_grad="wake" in self.trainable_keys,
        )
        # lss > 0
        self.raw_lss = nn.Parameter(
            torch.tensor(self._inv_softplus(init_lss), dtype=torch.float32),
            requires_grad="lss" in self.trainable_keys,
        )
        # sgt_cgs < 0  (parameterized as -softplus(raw))
        self.raw_sgt_cgs = nn.Parameter(
            torch.tensor(self._inv_softplus(abs(init_sgt_cgs)), dtype=torch.float32),
            requires_grad="sgt_cgs" in self.trainable_keys,
        )
        # tau_low > 0
        self.raw_tau_low = nn.Parameter(
            torch.tensor(self._inv_softplus(init_tau_low), dtype=torch.float32),
            requires_grad="tau_low" in self.trainable_keys,
        )
        # tau_sep > 0
        self.raw_tau_sep = nn.Parameter(
            torch.tensor(self._inv_softplus(init_tau_sep), dtype=torch.float32),
            requires_grad="tau_sep" in self.trainable_keys,
        )
        # relax > 0
        self.raw_relax = nn.Parameter(
            torch.tensor(self._inv_softplus(init_relax), dtype=torch.float32),
            requires_grad="relax" in self.trainable_keys,
        )
        # phi_temp > 0
        self.raw_phi_temp = nn.Parameter(
            torch.tensor(self._inv_softplus(init_phi_temp), dtype=torch.float32),
            requires_grad="phi_temp" in self.trainable_keys,
        )

    @staticmethod
    def _inv_softplus(y: float) -> float:
        if y <= 0:
            return -10.0
        return math.log(math.expm1(y)) if y < 20.0 else y

    def get_effective_parameters(self) -> dict[str, float]:
        """Return the current effective scalar values as a human-readable dict."""
        with torch.no_grad():
            return {
                "da_scale": float(F.softplus(self.raw_da_scale).item()),
                "wake": float(F.softplus(self.raw_wake).item()),
                "lss": float(F.softplus(self.raw_lss).item()),
                "sgt_cgs": -float(F.softplus(self.raw_sgt_cgs).item()),
                "tau_low": float(F.softplus(self.raw_tau_low).item()),
                "tau_sep": float(F.softplus(self.raw_tau_sep).item()),
                "relax": float(F.softplus(self.raw_relax).item()),
                "phi_temp": float(F.softplus(self.raw_phi_temp).item()),
            }

    def forward(self, data, num_nodes: int, device: torch.device) -> ParameterMap:
        return ParameterMap(
            da_scale=F.softplus(self.raw_da_scale).to(device),
            wake=F.softplus(self.raw_wake).to(device),
            lss=F.softplus(self.raw_lss).to(device),
            sgt_cgs=-F.softplus(self.raw_sgt_cgs).to(device),
            tau_low=F.softplus(self.raw_tau_low).to(device),
            tau_sep=F.softplus(self.raw_tau_sep).to(device),
            relax=F.softplus(self.raw_relax).to(device),
            phi_temp=F.softplus(self.raw_phi_temp).to(device),
        )
