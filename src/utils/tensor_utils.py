from __future__ import annotations

import torch


def as_tensor_like(
    value: float | int | torch.Tensor,
    *,
    like: torch.Tensor,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert Python scalars or tensors to a tensor on the same device/dtype as `like`."""
    if torch.is_tensor(value):
        out = value.to(device=like.device, dtype=like.dtype)
    else:
        out = torch.tensor(float(value), device=like.device, dtype=like.dtype)
    if requires_grad:
        out.requires_grad_(True)
    return out

