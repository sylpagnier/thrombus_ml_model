"""Shared Tier 3 time-axis handling (COMSOL exports vs training tensors)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.config import BiochemConfig


def _ensure_strictly_increasing_times(t: torch.Tensor) -> torch.Tensor:
    """Force strictly increasing timestamps (machine-representable Δt).

    Duplicate or decreasing ``data.t`` values make ``odeint`` hit ``dt`` underflow and
    make finite-difference ``d_pred/dt`` blow up to inf. A fixed tiny epsilon is not
    enough in float32 next to O(1–100) seconds (ULP ≫ 1e-9), so we use ``nextafter``.
    """
    dev = t.device
    dtype = t.dtype
    t1d = t.reshape(-1).contiguous().to(dtype).cpu()
    n = t1d.numel()
    if n <= 1:
        return t.reshape(-1).contiguous().to(dtype).to(dev)
    out = t1d.clone()
    pinf = torch.tensor(float("inf"), dtype=dtype)
    for i in range(1, n):
        lo = out[i - 1]
        if not bool((out[i] > lo).item()):
            out[i] = torch.nextafter(lo, pinf)
    return out.to(device=dev)


def resolve_tier3_times(data, bio_cfg: "BiochemConfig", device: torch.device) -> torch.Tensor:
    """Physical time stamps [s], length ``data.y.shape[0]``.

    Prefer ``data.t`` when it matches the trajectory length; otherwise interpolate
    uniformly to ``t[-1]`` (or ``bio_cfg.t_final``) and warn.
    """
    t_steps = int(data.y.shape[0])
    if hasattr(data, "t") and data.t is not None and data.t.numel() > 0:
        t = data.t.to(device=device, dtype=torch.float32).reshape(-1)
        if t.numel() == t_steps:
            return _ensure_strictly_increasing_times(t)
        t_last = float(t[-1].item()) if t.numel() else float(bio_cfg.t_final)
        warnings.warn(
            f"data.t length {t.numel()} != y time dim {t_steps}; "
            f"using linspace(0, {t_last:g}, {t_steps}). Re-export graphs with aligned t.",
            stacklevel=2,
        )
        return _ensure_strictly_increasing_times(
            torch.linspace(0.0, t_last, steps=t_steps, device=device, dtype=torch.float32)
        )
    return _ensure_strictly_increasing_times(
        torch.linspace(
            0.0, float(bio_cfg.t_final), steps=t_steps, device=device, dtype=torch.float32
        )
    )
