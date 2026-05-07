from __future__ import annotations

import torch


def to_t_nd(t: torch.Tensor, t_ref: float | torch.Tensor) -> torch.Tensor:
    """Convert physical time [s] to non-dimensional time.

    Kept intentionally tiny so every call site uses the same convention.
    """
    if not torch.is_tensor(t):
        raise TypeError("t must be a torch.Tensor")
    if torch.is_tensor(t_ref):
        t_ref_v = t_ref.to(device=t.device, dtype=t.dtype)
    else:
        t_ref_v = torch.tensor(float(t_ref), device=t.device, dtype=t.dtype)
    return t / torch.clamp(t_ref_v, min=1e-12)


def convective_time(d_bar: torch.Tensor, u_ref: torch.Tensor) -> torch.Tensor:
    """Convective reference time: t_conv = d_bar / u_ref."""
    return torch.clamp(d_bar, min=1e-12) / torch.clamp(u_ref, min=1e-12)


def time_ratio_global_to_convective(
    *,
    t_ref_global: float | torch.Tensor,
    d_bar: torch.Tensor,
    u_ref: torch.Tensor,
) -> torch.Tensor:
    """Chain-rule scaling factor t_ref_global / (d_bar/u_ref)."""
    t_conv = convective_time(d_bar, u_ref)
    if torch.is_tensor(t_ref_global):
        t_ref_v = t_ref_global.to(device=t_conv.device, dtype=t_conv.dtype)
    else:
        t_ref_v = torch.tensor(float(t_ref_global), device=t_conv.device, dtype=t_conv.dtype)
    return t_ref_v / torch.clamp(t_conv, min=1e-12)

