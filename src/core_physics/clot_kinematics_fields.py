"""Raw kinematic fields and COMSOL-aligned clot-risk scoring.

COMSOL wall clots on patient007 correlate with strong negative ``d(spf.sr,x)``
(≈ −800 1/(m·s)). The legacy prior used streamwise ``dshear/ds`` with per-graph
``max`` normalisation, which flattened that signal on late-time anchors.

This module exposes interpretable fields and scoring modes so tests, K11
triggers, and the bio_encoder prior share one definition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.utils.rheology import compute_shear_rate

if TYPE_CHECKING:
    from src.config import BiochemConfig


@dataclass
class ClotKinematicsFields:
    """SI kinematic quantities at one graph state (one time slice)."""

    gamma_si: torch.Tensor
    dshear_ds_phys: torch.Tensor
    dgamma_dx_phys: torch.Tensor
    dgamma_dy_phys: torch.Tensor
    is_low_shear: torch.Tensor
    is_separation_stream: torch.Tensor
    flux_path_stream: torch.Tensor
    flux_path_dx: torch.Tensor
    flux_path_dx_raw: torch.Tensor
    flux_stag: torch.Tensor
    wall_proximity: torch.Tensor
    adjacent_band: torch.Tensor


def _env_float(key: str, default: float) -> float:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_truthy(key: str, default: bool = False) -> bool:
    raw = (os.environ.get(key) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _sdf_nd_from_data(data, device: torch.device, n: int) -> torch.Tensor:
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.dim() == 2 and data.x.shape[1] > 2:
        return data.x[:, 2].to(device=device, dtype=torch.float32).clamp(min=0.0)
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        wall_soft = data.mask_wall.view(-1).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        return (1.0 - wall_soft).clamp(min=0.0)
    return torch.zeros(n, device=device, dtype=torch.float32)


def adjacent_band_mask(
    sdf_nd: torch.Tensor,
    wall_mask: torch.Tensor | None,
    *,
    peak_nd: float | None = None,
    sigma_nd: float | None = None,
) -> torch.Tensor:
    """Off-wall Gaussian band (K11 ``adjacent`` apply support), shape ``[N]`` bool."""
    d_peak = _env_float("BIOCHEM_K11_D_PEAK_ND", 0.008) if peak_nd is None else float(peak_nd)
    sigma = max(
        _env_float("BIOCHEM_K11_SIGMA_ND", 0.008) if sigma_nd is None else float(sigma_nd),
        1e-6,
    )
    sdf_cap = _env_float("BIOCHEM_K11_SDF_MAX_ND", 0.04)
    d = sdf_nd.reshape(-1).to(dtype=torch.float32).clamp(min=0.0)
    if wall_mask is None:
        off_wall = torch.ones_like(d)
    else:
        off_wall = (1.0 - wall_mask.reshape(-1).to(dtype=d.dtype).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    band = torch.exp(-0.5 * ((d - d_peak) / sigma) ** 2)
    if sdf_cap > 0.0:
        band = band * (d <= sdf_cap).to(dtype=d.dtype)
    return (band * off_wall) > 0.05


def compute_clot_kinematics_fields(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    bio_cfg: "BiochemConfig",
    props: dict,
) -> ClotKinematicsFields:
    """Build SI shear-rate and adhesion-proxy fields from graph kinematics."""
    u = u_nd.reshape(-1).to(dtype=torch.float32)
    v = v_nd.reshape(-1).to(dtype=torch.float32)
    n = int(u.shape[0])
    device = u.device

    du_dx = torch.sparse.mm(data.G_x, u.unsqueeze(1)).squeeze(1)
    du_dy = torch.sparse.mm(data.G_y, u.unsqueeze(1)).squeeze(1)
    dv_dx = torch.sparse.mm(data.G_x, v.unsqueeze(1)).squeeze(1)
    dv_dy = torch.sparse.mm(data.G_y, v.unsqueeze(1)).squeeze(1)
    gamma_dot_nd = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy, eps=1e-6)

    u_ref = props["u_ref"].to(device=device, dtype=torch.float32).reshape(-1)
    d_bar = props["d_bar"].to(device=device, dtype=torch.float32).reshape(-1)
    d_safe = torch.clamp(d_bar, min=1e-8)
    u_ref_safe = torch.clamp(u_ref, min=1e-8)
    scale_si = u_ref / d_safe
    gamma_si = gamma_dot_nd * scale_si

    gx = torch.sparse.mm(data.G_x, gamma_si.unsqueeze(1)).squeeze(1)
    gy = torch.sparse.mm(data.G_y, gamma_si.unsqueeze(1)).squeeze(1)
    dgamma_dx_phys = gx
    dgamma_dy_phys = gy

    vel_mag_nd = torch.sqrt(u * u + v * v) + 1e-8
    u_dir = u / vel_mag_nd
    v_dir = v / vel_mag_nd
    ds_stream = u_dir * gx + v_dir * gy
    dshear_ds_phys = ds_stream / d_safe

    T_ls = max(float(bio_cfg.soft_step_T_low_shear) * float(bio_cfg.soft_step_T_scale), 1e-6)
    T_gr = max(float(bio_cfg.soft_step_T_grad) * float(bio_cfg.soft_step_T_scale), 1e-6)
    is_low_shear = torch.sigmoid(((float(bio_cfg.lss) - gamma_si) / T_ls).clamp(-50.0, 50.0))
    is_separation_stream = torch.sigmoid(
        ((float(bio_cfg.sgt) - dshear_ds_phys) / T_gr).clamp(-50.0, 50.0)
    )

    sgt_ref = max(abs(float(bio_cfg.sgt)), 1e-6)
    flux_path_stream = (is_separation_stream * (torch.abs(dshear_ds_phys) / sgt_ref)).clamp(0.0, 5.0)

    dx_default = 800.0 if clot_prior_score_mode() == "legacy" else 35.0
    dx_thr = max(_env_float("BIOCHEM_PRIOR_DGAMMA_DX_THRESH", dx_default), 1e-6)
    dy_thr = max(_env_float("BIOCHEM_PRIOR_DGAMMA_DY_THRESH", dx_thr), 1e-6)
    w_dx = max(_env_float("BIOCHEM_PRIOR_W_DGAMMA_DX", 1.0), 0.0)
    w_dy = max(_env_float("BIOCHEM_PRIOR_W_DGAMMA_DY", 0.35), 0.0)

    is_sep_dx = torch.sigmoid(((-dx_thr) - dgamma_dx_phys) / T_gr).clamp(0.0, 1.0)
    is_sep_dy = torch.sigmoid(((-dy_thr) - dgamma_dy_phys) / T_gr).clamp(0.0, 1.0)
    flux_dx = (is_sep_dx * ((-dgamma_dx_phys).clamp(min=0.0)) / dx_thr).clamp(0.0, 5.0)
    flux_dy = (is_sep_dy * ((-dgamma_dy_phys).clamp(min=0.0)) / dy_thr).clamp(0.0, 5.0)
    flux_path_dx_raw = w_dx * flux_dx + w_dy * flux_dy
    flux_path_dx = flux_path_dx_raw.clamp(0.0, 5.0)

    beta_res = max(_env_float("BIOCHEM_PRIOR_RESIDENCE_BOOST", 0.5), 0.0)
    vel_mag_si = vel_mag_nd * u_ref
    residence = torch.exp(-(vel_mag_si / u_ref_safe).clamp(min=0.0, max=50.0))
    flux_stag = is_low_shear * (1.0 + beta_res * residence)

    sdf_nd = _sdf_nd_from_data(data, device, n)
    lambda_w = max(_env_float("BIOCHEM_PRIOR_WALL_DECAY_ND", 0.006), 1e-4)
    wall_proximity = torch.exp(-sdf_nd / lambda_w).clamp(0.0, 1.0)

    wall_mask = (
        data.mask_wall.view(-1).bool()
        if hasattr(data, "mask_wall") and data.mask_wall is not None
        else torch.zeros(n, dtype=torch.bool, device=device)
    )
    adj = adjacent_band_mask(sdf_nd, data.mask_wall if hasattr(data, "mask_wall") else None)

    return ClotKinematicsFields(
        gamma_si=gamma_si,
        dshear_ds_phys=dshear_ds_phys,
        dgamma_dx_phys=dgamma_dx_phys,
        dgamma_dy_phys=dgamma_dy_phys,
        is_low_shear=is_low_shear,
        is_separation_stream=is_separation_stream,
        flux_path_stream=flux_path_stream,
        flux_path_dx=flux_path_dx,
        flux_path_dx_raw=flux_path_dx_raw,
        flux_stag=flux_stag,
        wall_proximity=wall_proximity,
        adjacent_band=adj,
    )


def _normalize_field(
    field: torch.Tensor,
    norm_mask: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    mode = (mode or "adjacent_p95").strip().lower()
    if mode in ("max", "global_max", "legacy"):
        ref = field.detach().abs().max().clamp(min=1e-8)
        return (field / ref).clamp(0.0, 1.0)
    q = max(min(_env_float("BIOCHEM_PRIOR_NORM_QUANTILE", 0.95), 0.5), 0.5)
    m = norm_mask.view(-1).bool()
    if not bool(m.any().item()):
        ref = field.detach().abs().max().clamp(min=1e-8)
        return (field / ref).clamp(0.0, 1.0)
    ref = torch.quantile(field.detach()[m], q).clamp(min=1e-8)
    return (field / ref).clamp(0.0, 3.0)


def clot_prior_score_mode() -> str:
    if _env_truthy("BIOCHEM_PRIOR_COMSOL_ALIGNED"):
        return "comsol_hybrid"
    raw = (os.environ.get("BIOCHEM_PRIOR_SCORE_MODE") or "legacy").strip().lower()
    return raw if raw else "legacy"


def score_clot_risk_from_fields(
    fields: ClotKinematicsFields,
    bio_cfg: "BiochemConfig",
    *,
    norm_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(combined, path_channel, stag_channel)`` in ``[0, 1]``."""
    mode = clot_prior_score_mode()
    norm_scope = (os.environ.get("BIOCHEM_PRIOR_NORM_MASK") or "adjacent").strip().lower()
    if norm_mask is None:
        if norm_scope in ("adjacent", "adj", "band"):
            norm_mask = fields.adjacent_band
        elif norm_scope in ("wall", "wall_prox"):
            norm_mask = fields.wall_proximity > 0.5
        else:
            norm_mask = torch.ones_like(fields.gamma_si, dtype=torch.bool)

    w_path = min(max(_env_float("BIOCHEM_PRIOR_W_PATHOLOGICAL", 1.0), 0.0), 1.0)
    w_stag = min(max(_env_float("BIOCHEM_PRIOR_W_STAGNATION", 0.25), 0.0), 1.0)
    min_floor = max(_env_float("BIOCHEM_PRIOR_MIN_FLOOR", 1e-4), 0.0)
    total_power = max(_env_float("BIOCHEM_PRIOR_TOTAL_POWER", 1.5), 1e-6)
    stag_power = max(_env_float("BIOCHEM_PRIOR_STAGNATION_POWER", 1.5), 1e-6)

    if mode == "legacy":
        path_raw = fields.flux_path_stream
        stag_raw = fields.flux_stag
    else:
        w_stream = max(_env_float("BIOCHEM_PRIOR_W_STREAM_SEP", 0.45), 0.0)
        path_combined = (
            w_stream * fields.flux_path_stream + fields.flux_path_dx
        ).clamp(0.0, 5.0)
        if mode in ("comsol_dx", "dx_only"):
            path_raw = fields.flux_path_dx
        elif mode in ("comsol_hybrid", "hybrid", "comsol"):
            path_raw = torch.maximum(
                w_stream * fields.flux_path_stream, fields.flux_path_dx
            ).clamp(0.0, 5.0)
        else:
            path_raw = path_combined
        stag_raw = fields.flux_stag

    norm_mode = "max" if mode == "legacy" else (
        os.environ.get("BIOCHEM_PRIOR_NORM_MODE") or "adjacent_p95"
    ).strip().lower()
    path_norm = _normalize_field(path_raw, norm_mask, mode=norm_mode)
    stag_norm = _normalize_field(stag_raw, norm_mask, mode=norm_mode)
    stag_norm = stag_norm.pow(stag_power)

    path_w = (w_path * path_norm).clamp(0.0, 1.0)
    stag_w = (w_stag * stag_norm).clamp(0.0, 1.0)
    flux_total = (path_w + stag_w - path_w * stag_w).clamp(0.0, 1.0).pow(total_power)

    wall = fields.wall_proximity
    col_full = ((flux_total + min_floor) * wall).clamp(0.0, 1.0)
    col_path = ((path_w.pow(total_power) + min_floor) * wall).clamp(0.0, 1.0)
    col_stag = ((stag_w.pow(total_power) + min_floor) * wall).clamp(0.0, 1.0)
    return col_full, col_path, col_stag
