"""Compatibility shim for deploy species initialization.

Legacy rung4 ladder code was archived; the active deploy path only needs
`resting_species_log_nd`.
"""

from __future__ import annotations

import torch

from src.utils import species_channels as sc


def resting_species_log_nd(data, device: torch.device) -> torch.Tensor:
    """Return baseline log1p species state [N, 12] used for deploy IC.

    Preference order:
    1) First timeline slice from `data.y[:, :, 4:16]` when available.
    2) Zeros fallback (resting prior).
    """
    if (
        hasattr(data, "y")
        and torch.is_tensor(data.y)
        and data.y.ndim >= 3
        and data.y.shape[-1] >= sc.Y_WIDTH
    ):
        out = data.y[0, :, sc.SPECIES_BLOCK].to(device=device, dtype=torch.float32)
        return torch.clamp(out, min=-10.0, max=8.0)
    n = int(getattr(data, "num_nodes", data.x.shape[0]))
    return torch.zeros(n, sc.SPECIES_BLOCK_WIDTH, device=device, dtype=torch.float32)

