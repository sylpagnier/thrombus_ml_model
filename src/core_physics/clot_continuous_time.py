"""Step 3b continuous-time helpers: macro tau indexing and extrapolation pairs."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.config import BiochemConfig


def continuous_time_frac_enabled() -> bool:
    return (os.environ.get("CLOT_ML_USE_MACRO_TAU") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_tau_ref_s(data, bio_cfg: BiochemConfig | None = None) -> float:
    raw = (os.environ.get("CLOT_ML_TAU_REF_S") or "").strip()
    if raw:
        return max(float(raw), 1e-6)
    if hasattr(data, "t") and data.t is not None:
        t = data.t.to(dtype=torch.float32).reshape(-1)
        if t.numel() >= 2:
            return max(float(t[-1] - t[0]), 1e-6)
    if bio_cfg is not None:
        return max(float(bio_cfg.t_final), 1e-6)
    n = int(data.y.shape[0])
    return max(float(n - 1), 1e-6)


def macro_tau_at_index(
    data,
    time_index: int,
    *,
    tau_ref_s: float | None = None,
    bio_cfg: BiochemConfig | None = None,
) -> float:
    """Physical time / tau_ref (can exceed 1.0 when extrapolating past COMSOL export)."""
    ref = max(float(tau_ref_s if tau_ref_s is not None else resolve_tau_ref_s(data, bio_cfg)), 1e-6)
    if hasattr(data, "t") and data.t is not None:
        t = data.t.to(dtype=torch.float32).reshape(-1)
        n = int(t.numel())
        if n == 0:
            return 0.0
        t0 = float(t[0])
        if int(time_index) < n:
            ti = float(t[int(time_index)])
        else:
            dt = float(t[-1] - t[-2]) if n >= 2 else 1.0
            ti = float(t[-1]) + dt * float(int(time_index) - (n - 1))
        return (ti - t0) / ref
    n = int(data.y.shape[0])
    return float(time_index) / max(n - 1, 1)


def time_frac_for_rollout(
    data,
    time_index: int,
    *,
    bio_cfg: BiochemConfig | None = None,
    clamp_unit: bool = True,
) -> float:
    """Growth time coordinate: index-normalized or macro tau (Step 3b)."""
    if continuous_time_frac_enabled():
        tau = macro_tau_at_index(data, time_index, bio_cfg=bio_cfg)
        return min(1.0, tau) if clamp_unit else tau
    n = int(data.y.shape[0])
    return float(time_index) / max(n - 1, 1)


def sim_end_scale_from_env() -> float:
    raw = (os.environ.get("CLOT_ML_SIM_END_SCALE") or "1.0").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 1.0


def continuous_extrap_growth_enabled() -> bool:
    """Axis C: continue progressive growth past COMSOL export (requires macro tau)."""
    if not continuous_time_frac_enabled():
        return False
    raw = (os.environ.get("CLOT_ML_CONTINUOUS_EXTRAP") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def comsol_final_index(data) -> int:
    return max(int(data.y.shape[0]) - 1, 0)


def feature_time_index(data, virtual_t_out: int) -> int:
    """Clamp to labeled COMSOL window for species / GT feature lookups."""
    return max(0, min(int(virtual_t_out), comsol_final_index(data)))


def growth_time_frac(
    data,
    virtual_t_out: int,
    *,
    bio_cfg: BiochemConfig | None = None,
) -> float:
    """Growth clock coordinate; may exceed 1.0 when virtual index is past COMSOL end."""
    return time_frac_for_rollout(
        data, int(virtual_t_out), bio_cfg=bio_cfg, clamp_unit=False
    )


def growth_u_from_t_frac(
    t_frac: float,
    onset_frac: float,
    *,
    extrap: bool,
    sim_end_scale: float = 1.0,
    tau_comsol_end: float = 1.0,
) -> float:
    """Normalized growth ramp in-window; extends to u>1 in extrap segment when enabled."""
    onset = max(float(onset_frac), 0.0)
    tf = float(t_frac)
    if tf < onset:
        return 0.0
    t_end = max(float(tau_comsol_end), onset + 1e-6)
    if tf <= t_end + 1e-6:
        if onset > 0.0:
            return (tf - onset) / max(t_end - onset, 1e-6)
        return tf / t_end
    if not extrap:
        return 1.0
    scale = max(float(sim_end_scale), t_end + 1e-6)
    extra = (tf - t_end) / max(scale - t_end, 1e-6)
    return 1.0 + min(max(extra, 0.0), 1.0)


def extrap_frac_headroom() -> float:
    raw = (os.environ.get("CLOT_ML_EXTRAP_FRAC_HEADROOM") or "0.12").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.12


def extrapolated_t_out_max(data, *, sim_end_scale: float | None = None) -> int:
    """Last virtual t_out index when sim_end = scale * COMSOL span (Step 3b spike)."""
    n = int(data.y.shape[0])
    scale = float(sim_end_scale if sim_end_scale is not None else sim_end_scale_from_env())
    if scale <= 1.0 + 1e-6:
        return n - 1
    if hasattr(data, "t") and data.t is not None:
        t = data.t.to(dtype=torch.float32).reshape(-1)
        if t.numel() >= 2:
            dt = float(t[-1] - t[-2])
            extra_s = (scale - 1.0) * float(t[-1] - t[0])
            extra_steps = int(math.ceil(extra_s / max(dt, 1e-6)))
            return (n - 1) + max(extra_steps, 0)
    extra_steps = int(math.ceil((scale - 1.0) * max(n - 1, 1)))
    return (n - 1) + max(extra_steps, 0)


def rollout_time_indices(
    data,
    *,
    time_stride: int = 1,
    sim_end_scale: float | None = None,
) -> list[int]:
    n = int(data.y.shape[0])
    step = max(int(time_stride), 1)
    t_final = extrapolated_t_out_max(data, sim_end_scale=sim_end_scale)
    return list(range(0, t_final + 1, step))
