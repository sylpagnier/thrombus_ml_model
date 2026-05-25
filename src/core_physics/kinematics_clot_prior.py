"""Kinematics-derived clot-deposition prior (encoder + K11 mech trigger).

Scoring modes (``BIOCHEM_PRIOR_SCORE_MODE`` / ``BIOCHEM_PRIOR_COMSOL_ALIGNED``):

- ``legacy`` — streamwise separation + per-graph max norm (v2 behaviour).
- ``comsol_hybrid`` (recommended) — union of streamwise separation and
  **negative** ``dγ/dx`` gate (COMSOL ``d(spf.sr,x)`` ≲ −800 1/(m·s)), normalised with
  **p95 on the wall-adjacent band** so hotspots stay localized.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from src.core_physics.clot_kinematics_fields import (
    ClotKinematicsFields,
    adjacent_band_mask,
    clot_prior_score_mode,
    compute_clot_kinematics_fields,
    score_clot_risk_from_fields,
)

if TYPE_CHECKING:
    from src.config import BiochemConfig


def _max_neighbor_dilate_1d(v: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    flat = v.reshape(-1).contiguous()
    row = edge_index[0]
    col = edge_index[1]
    msg = torch.full_like(flat, float("-inf"))
    msg.scatter_reduce_(0, row, flat[col], reduce="amax", include_self=False)
    return torch.maximum(flat, msg).view(v.shape)


def _thrombus_corona_hop_count() -> int:
    try:
        return max(0, min(12, int(os.environ.get("BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS", "0"))))
    except ValueError:
        return 0


def _apply_thrombus_corona_dilation(
    cols: list[torch.Tensor],
    edge_index: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    hops = _thrombus_corona_hop_count()
    if hops <= 0 or edge_index.numel() == 0:
        return cols
    ei = edge_index.to(device=device)
    out_cols: list[torch.Tensor] = []
    for c in cols:
        flat = c.reshape(-1).contiguous()
        for _ in range(hops):
            flat = _max_neighbor_dilate_1d(flat, ei)
        out_cols.append(flat.reshape(c.shape).clamp(0.0, 1.0))
    return out_cols


def shear_rate_si(data, u_nd: torch.Tensor, v_nd: torch.Tensor, props: dict) -> torch.Tensor:
    """Physical shear rate [1/s] from ND velocities and graph sparse gradients."""
    from src.utils.rheology import compute_shear_rate

    u = u_nd.reshape(-1).to(dtype=torch.float32)
    v = v_nd.reshape(-1).to(dtype=torch.float32)
    du_dx = torch.sparse.mm(data.G_x, u.unsqueeze(1)).squeeze(1)
    du_dy = torch.sparse.mm(data.G_y, u.unsqueeze(1)).squeeze(1)
    dv_dx = torch.sparse.mm(data.G_x, v.unsqueeze(1)).squeeze(1)
    dv_dy = torch.sparse.mm(data.G_y, v.unsqueeze(1)).squeeze(1)
    gamma_dot_nd = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy, eps=1e-6)
    u_ref = props["u_ref"].to(device=u.device, dtype=torch.float32).reshape(-1)
    d_bar = props["d_bar"].to(device=u.device, dtype=torch.float32).reshape(-1)
    d_safe = torch.clamp(d_bar, min=1e-8)
    return gamma_dot_nd * (u_ref / d_safe)


def clot_prior_features(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    bio_cfg: "BiochemConfig",
    props: dict,
    n_features: int = 2,
) -> torch.Tensor:
    """Return ``[N, n_features]`` wall-localised clot-risk features."""
    n_features = max(1, int(n_features))
    fields = compute_clot_kinematics_fields(data, u_nd, v_nd, bio_cfg, props)
    col_full, col_path, col_stag = score_clot_risk_from_fields(fields, bio_cfg)

    cols = [col_full]
    if n_features >= 2:
        cols.append(col_path)
    if n_features >= 3:
        from src.core_physics.clot_kinematics_fields import _normalize_field

        dx_raw = (fields.flux_path_dx * fields.wall_proximity).clamp(0.0, 5.0)
        norm_mode = "max" if clot_prior_score_mode() == "legacy" else "adjacent_p95"
        cols.append(
            _normalize_field(dx_raw, fields.adjacent_band, mode=norm_mode).clamp(0.0, 1.0)
        )
    min_floor = max(float(os.environ.get("BIOCHEM_PRIOR_MIN_FLOOR", "1e-4") or "1e-4"), 0.0)
    pad_template = (min_floor * fields.wall_proximity).clamp(0.0, 1.0)
    while len(cols) < n_features:
        cols.append(pad_template)

    if (
        hasattr(data, "edge_index")
        and torch.is_tensor(data.edge_index)
        and data.edge_index.dim() == 2
        and data.edge_index.shape[1] > 0
    ):
        cols = _apply_thrombus_corona_dilation(cols, data.edge_index, device=col_full.device)

    return torch.stack(cols[:n_features], dim=1)


def clot_prior_score_flat(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    bio_cfg: "BiochemConfig",
    props: dict,
) -> torch.Tensor:
    """Single-channel prior ``[N]`` (column 0 of ``clot_prior_features``)."""
    return clot_prior_features(data, u_nd, v_nd, bio_cfg, props, n_features=1).squeeze(1)


def kinematics_clot_prior_score(
    kernels, data, u_nd: torch.Tensor, v_nd: torch.Tensor, props: dict
) -> torch.Tensor:
    return clot_prior_score_flat(data, u_nd, v_nd, kernels.cfg, props)


# Public aliases for diagnostics / tests
__all__ = [
    "ClotKinematicsFields",
    "adjacent_band_mask",
    "clot_prior_features",
    "clot_prior_score_flat",
    "clot_prior_score_mode",
    "compute_clot_kinematics_fields",
    "kinematics_clot_prior_score",
    "shear_rate_si",
    "score_clot_risk_from_fields",
]
