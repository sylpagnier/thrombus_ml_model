"""Wall-local clot phase field (phi) with capped GT mu and log-space Carreau blend.

Supervision support (``CLOT_PHI_MASK_MODE``):

- ``neighbor`` (default): mesh **k-hop** from ``mask_wall`` plus **k-hop** from GT clot
  seeds (capped ``mu_eff`` ≥ threshold). Matches the intended physics: nucleation at
  the wall / boundary layer and growth along the graph to adjacent clot nodes (same
  idea as ``BIOCHEM_PRIOR_THROMBUS_CORONA_HOPS`` dilation in ``kinematics_clot_prior``).
- ``sdf``: legacy shell ``sdf_nd <= CLOT_PHI_SDF_MAX_ND`` (wider, less specific).

Note: the **K10e forward** applies μ uplift in an off-wall SDF band; COMSOL GT can still
mark high ``mu_eff`` on ``mask_wall`` nodes (~95% of clot seeds on patient007).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BiochemConfig, NodeFeat, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.utils import species_channels as sc
from src.core_physics.clot_kinematics_fields import adjacent_band_mask, compute_clot_kinematics_fields, compute_shear_rate
from src.core_physics.kinematics_clot_prior import clot_prior_features, clot_prior_score_flat
from src.utils.rheology import carreau_yasuda_viscosity


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    return float(raw) if raw else float(default)


def clot_phi_mu_cap_si() -> float:
    return max(_env_float("CLOT_PHI_MU_CAP_SI", 0.10), 1e-6)


def clot_phi_mu_solid_si() -> float:
    return max(_env_float("CLOT_PHI_MU_SOLID_SI", clot_phi_mu_cap_si()), 1e-6)


def clot_phi_thresh_si(phys_cfg: PhysicsConfig) -> float:
    return max(_env_float("CLOT_PHI_THRESH_SI", 0.055), float(phys_cfg.mu_inf))


def cap_mu_eff_si(mu_si: torch.Tensor) -> torch.Tensor:
    cap = clot_phi_mu_cap_si()
    return mu_si.clamp(min=1e-8, max=cap)


def sdf_nd_from_data(data, device: torch.device, n: int) -> torch.Tensor:
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.dim() == 2 and data.x.shape[1] > 2:
        return data.x[:, NodeFeat.SDF.start].to(device=device, dtype=torch.float32).clamp(min=0.0)
    return torch.zeros(n, device=device, dtype=torch.float32)


def width_nd_from_data(data, device: torch.device, n: int) -> torch.Tensor:
    """Hydraulic lumen width [ND] from graph node features (ray-march prior)."""
    if (
        hasattr(data, "x")
        and torch.is_tensor(data.x)
        and data.x.dim() == 2
        and data.x.shape[1] > NodeFeat.WIDTH_ND.stop
    ):
        return data.x[:, NodeFeat.WIDTH_ND].reshape(-1).to(device=device, dtype=torch.float32).clamp(min=1e-6)
    if hasattr(data, "width_nd") and data.width_nd is not None:
        return data.width_nd.reshape(-1).to(device=device, dtype=torch.float32).clamp(min=1e-6)
    sdf = sdf_nd_from_data(data, device, n)
    return (2.0 * sdf).clamp(min=1e-6)


def _wall_mask_from_data(data, device: torch.device, n: int) -> torch.Tensor:
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        return data.mask_wall.view(-1).to(device=device).bool()
    return torch.zeros(n, device=device, dtype=torch.bool)


def clot_phi_center_exclude_frac() -> float:
    """Drop inner lumen nodes by SDF rank (centerline gap); 0 disables."""
    return max(0.0, min(_env_float("CLOT_PHI_CENTER_EXCLUDE_FRAC", 0.10), 0.95))


def clot_phi_shear_min_frac() -> float:
    """Keep nodes with ``gamma_dot >= frac * gamma_ref_max`` (0 disables)."""
    return max(0.0, min(_env_float("CLOT_PHI_SHEAR_MIN_FRAC", 0.0), 1.0))


def clot_phi_shear_wall_exempt() -> bool:
    """If true, ``mask_wall`` nodes skip the shear cutoff (trim off-wall stagnation only)."""
    return _env_bool("CLOT_PHI_SHEAR_WALL_EXEMPT", True)


def clot_phi_shear_ref_time_index(data) -> int:
    """Time index for GT ``u,v`` shear reference (default ``-1`` = final frame)."""
    raw = (os.environ.get("CLOT_PHI_SHEAR_REF_TIME") or "-1").strip()
    try:
        ti = int(raw)
    except ValueError:
        ti = -1
    t_last = int(data.y.shape[0]) - 1
    if ti < 0:
        ti = t_last + 1 + ti
    return max(0, min(ti, t_last))


def gt_gamma_dot_nd(data, time_index: int, device: torch.device) -> torch.Tensor:
    """COMSOL GT shear rate ``gamma_dot`` [1/s] ND from ``y[ti]`` ``u,v`` and graph grads."""
    y = data.y[time_index].to(device=device, dtype=torch.float32)
    u = y[:, 0]
    v = y[:, 1]
    u_col = u.reshape(-1, 1)
    G_x = data.G_x.to(device=device)
    G_y = data.G_y.to(device=device)
    du_dx = torch.sparse.mm(G_x, u_col).squeeze(1)
    du_dy = torch.sparse.mm(G_y, u_col).squeeze(1)
    dv_dx = torch.sparse.mm(G_x, v.reshape(-1, 1)).squeeze(1)
    dv_dy = torch.sparse.mm(G_y, v.reshape(-1, 1)).squeeze(1)
    return compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy).clamp(min=1e-12)


def clot_phi_dgamma_slice_enabled() -> bool:
    return _env_bool("CLOT_PHI_DGAMMA_SLICE", False)


def clot_phi_dgamma_ref_time_index(data) -> int:
    """Reference time for GT ``u,v`` used in ``d(gamma)/dx`` slice (default ``0`` = SS / inlet)."""
    raw = (os.environ.get("CLOT_PHI_DGAMMA_REF_TIME") or "0").strip()
    try:
        ti = int(raw)
    except ValueError:
        ti = 0
    t_last = int(data.y.shape[0]) - 1
    if ti < 0:
        ti = t_last + 1 + ti
    return max(0, min(ti, t_last))


def clot_phi_dgamma_feature_time_index(data, time_index: int) -> int:
    """Time index for ``-dgamma/dx`` in node features (mask slice still uses ``DGAMMA_REF_TIME``)."""
    raw = (os.environ.get("CLOT_PHI_DGAMMA_FEATURE_TIME") or "ref").strip().lower()
    if raw in ("current", "slice", "same", "match"):
        t_last = int(data.y.shape[0]) - 1
        return max(0, min(int(time_index), t_last))
    return clot_phi_dgamma_ref_time_index(data)


def clot_phi_dgamma_wall_min_si() -> float:
    """Wall nodes: keep GT clot seeds or ``-d(gamma)/dx >=`` this [1/(m*s)] (COMSOL adhesion band)."""
    return max(_env_float("CLOT_PHI_DGAMMA_WALL_MIN_SI", 100.0), 0.0)


def clot_phi_dgamma_offwall_percentile() -> float:
    """Off-wall nodes: keep if ``-d(gamma)/dx`` >= this percentile within off-wall base (0 = drop all)."""
    return max(0.0, min(_env_float("CLOT_PHI_DGAMMA_OFFWALL_PCT", 80.0), 100.0))


def gt_neg_dgamma_dx_phys(
    data,
    time_index: int,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> torch.Tensor:
    """COMSOL-aligned ``max(0, -d(gamma)/dx)`` [1/(m*s)] from GT ``u,v`` (matches ``d(spf.sr,x)`` band)."""
    y = data.y[time_index].to(device=device, dtype=torch.float32)
    return pred_neg_dgamma_dx_phys(data, y[:, 0], y[:, 1], bio_cfg, device)


def pred_neg_dgamma_dx_phys(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> torch.Tensor:
    """``max(0, -d(gamma)/dx)`` from predicted or patched ``u,v`` (deploy-safe)."""
    props = {
        "u_ref": data.u_ref.to(device=device),
        "d_bar": data.d_bar.to(device=device),
    }
    u = u_nd.reshape(-1).to(device=device, dtype=torch.float32)
    v = v_nd.reshape(-1).to(device=device, dtype=torch.float32)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    return (-fields.dgamma_dx_phys).clamp(min=0.0)


def dgamma_dx_slice_mask(
    data,
    device: torch.device,
    base: torch.Tensor,
    clot_seed: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """Tighten ``base`` using SS ``-d(gamma)/dx`` at ``CLOT_PHI_DGAMMA_REF_TIME``.

    - **Wall**: GT clot seeds always; other wall nodes need ``-d(gamma)/dx >= wall_min``.
    - **Off-wall**: keep nodes above the off-wall percentile of ``-d(gamma)/dx`` (drops halo specks).
    """
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    base = base.reshape(-1).to(device=device).bool()
    clot = clot_seed.reshape(-1).to(device=device).bool()
    ti = clot_phi_dgamma_ref_time_index(data)
    neg_dx = gt_neg_dgamma_dx_phys(data, ti, bio_cfg, device)

    out = torch.zeros(n, device=device, dtype=torch.bool)
    on_wall = base & wall
    off_wall = base & ~wall
    wall_min = clot_phi_dgamma_wall_min_si()
    out[on_wall] = clot[on_wall] | (neg_dx[on_wall] >= wall_min)
    pct = clot_phi_dgamma_offwall_percentile()
    if bool(off_wall.any().item()) and pct > 0.0:
        thr = torch.quantile(neg_dx[off_wall], pct / 100.0)
        out[off_wall] = neg_dx[off_wall] >= thr
    return out


def shear_activity_mask(
    data,
    device: torch.device,
    wall: torch.Tensor,
    *,
    pool: torch.Tensor | None = None,
) -> torch.Tensor:
    """Nodes above a fraction of the reference max ``gamma_dot`` (GT at ``CLOT_PHI_SHEAR_REF_TIME``)."""
    n = int(data.num_nodes)
    frac = clot_phi_shear_min_frac()
    if frac <= 0.0:
        return torch.ones(n, device=device, dtype=torch.bool)
    ti = clot_phi_shear_ref_time_index(data)
    gamma = gt_gamma_dot_nd(data, ti, device)
    if pool is None:
        gamma_max = gamma.max()
    else:
        pool = pool.reshape(-1).to(device=device).bool()
        if not bool(pool.any().item()):
            return torch.ones(n, device=device, dtype=torch.bool)
        gamma_max = gamma[pool].max()
    thr = frac * gamma_max
    active = gamma >= thr
    if clot_phi_shear_wall_exempt():
        active = active | wall
    return active


def _lumen_supervision_eligible(
    data,
    device: torch.device,
    wall: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Wall always eligible; off-wall nodes exclude the most interior SDF fraction (centerline)."""
    frac = clot_phi_center_exclude_frac()
    eligible = wall.clone()
    if frac <= 0.0:
        return torch.ones(n, device=device, dtype=torch.bool)
    sdf = sdf_nd_from_data(data, device, n)
    lumen = (~wall) & (sdf > 1e-8)
    if not bool(lumen.any().item()):
        return eligible
    lumen_sdf = sdf[lumen]
    thr = torch.quantile(lumen_sdf, 1.0 - frac)
    eligible |= lumen & (sdf <= thr)
    return eligible


def _graph_dilate(active: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """One undirected hop on the biochem mesh (``edge_index`` is typically bidirectional)."""
    if not bool(active.any().item()):
        return active
    edge_index = edge_index.to(device=active.device)
    row, col = edge_index[0], edge_index[1]
    out = active.clone()
    out[row[active[col]]] = True
    out[col[active[row]]] = True
    return out


def sdf_supervision_mask(data, device: torch.device) -> torch.Tensor:
    """Legacy SDF shell (``sdf_nd <= CLOT_PHI_SDF_MAX_ND``)."""
    n = int(data.num_nodes)
    sdf = sdf_nd_from_data(data, device, n)
    sdf_cap = _env_float("CLOT_PHI_SDF_MAX_ND", 0.04)
    if sdf_cap <= 0.0:
        return torch.ones(n, device=device, dtype=torch.bool)
    return (sdf <= sdf_cap).to(device=device)


def neighbor_supervision_mask(
    data,
    device: torch.device,
    clot_seed: torch.Tensor,
) -> torch.Tensor:
    """Supervision = all wall nodes (nucleation candidates) + off-wall clot neighbors only.

    - **Wall**: every ``mask_wall`` node is included (always a candidate site).
    - **Off-wall**: GT clot seeds + ``CLOT_PHI_CLOT_TOUCH_HOPS`` mesh neighbors (default 1).
    - **Centerline gap**: drop the inner ``CLOT_PHI_CENTER_EXCLUDE_FRAC`` (default 0.10)
      of lumen nodes by SDF (farthest from wall) so the core flow channel is not supervised
      (applies to off-wall clot seeds and their 1-hop band; wall nodes are always kept).
    """
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    h_touch = max(int(_env_float("CLOT_PHI_CLOT_TOUCH_HOPS", 1)), 0)
    ei = data.edge_index.to(device=device)
    seed = clot_seed.reshape(-1).to(device=device).bool()
    eligible = _lumen_supervision_eligible(data, device, wall, n)
    active_seed = seed & eligible

    # Dilate only from eligible seeds so inner (excluded) clots do not leak a 1-hop halo.
    near = active_seed.clone()
    for _ in range(h_touch):
        near = _graph_dilate(near, ei)

    # Wall always; off-wall seeds and 1-hop band respect the centerline fluid gap.
    lumen_band = near & ~wall & ~seed
    return wall | active_seed | (lumen_band & eligible)


def clot_phi_mask_mode() -> str:
    raw = (os.environ.get("CLOT_PHI_MASK_MODE") or "neighbor").strip().lower()
    if raw in ("sdf", "shell", "legacy"):
        return "sdf"
    return "neighbor"


def clot_phi_clot_seed_source() -> str:
    """How neighbor-band clot seeds are chosen (``supervision_region_mask`` only).

    - ``wall`` (default): geometry wall nodes + mesh hops (deploy nucleation shell).
    - ``gt_mu``: COMSOL capped ``mu_eff`` at the current step (**oracle debug only**).
    - ``none``: wall nodes only (no off-wall dilation from seeds).
    """
    raw = (os.environ.get("CLOT_PHI_CLOT_SEED_SOURCE") or "wall").strip().lower()
    if raw in ("wall", "geometry", "mask_wall", "deploy", "nucleation"):
        return "wall"
    if raw in ("none", "wall_only", "off"):
        return "none"
    if raw in ("gt_mu", "oracle", "legacy"):
        return "gt_mu"
    return "wall"


def clot_phi_loss_scope() -> str:
    """Where clot loss / eval F1 are computed.

    - ``ceiling`` (deploy default): fixed wall + ``CLOT_PHI_CEILING_HOPS`` band (no GT expansion).
    - ``support``: time-varying B_t (wall @ t0 + 1-hop growth; pred or oracle seed).
    - ``full_mesh``: all nodes (debug; includes Carreau bulk artifacts).
    - ``nucleation``: static wall+hops band (geometry only).
    - ``oracle``: legacy GT-mu seeds + GT dgamma slice (**debug only**).
    """
    raw = (os.environ.get("CLOT_PHI_LOSS_SCOPE") or "support").strip().lower()
    if raw in ("oracle", "oracle_band", "legacy", "gt_mu", "band"):
        return "oracle"
    if raw in ("nucleation", "deploy_band", "wall"):
        return "nucleation"
    if raw in ("ceiling", "deploy"):
        return "ceiling"
    if raw in ("support", "growth", "ceiling_growth", "final_support", "b_t"):
        return "support"
    return "full_mesh"


def clot_phi_forward_apply_region() -> bool:
    """Whether physics/hybrid phi is zeroed outside the loss/support band."""
    return clot_phi_loss_scope() != "full_mesh"


def resolve_clot_loss_mask(
    data,
    device: torch.device,
    mu_cap_si: torch.Tensor,
    phys_cfg: PhysicsConfig,
    *,
    time_index: int = 0,
    bio_cfg: BiochemConfig | None = None,
    phi_pred_by_time: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Boolean mask for loss / val F1 (scope-dependent)."""
    n = int(data.num_nodes)
    scope = clot_phi_loss_scope()
    if scope == "full_mesh":
        return torch.ones(n, device=device, dtype=torch.bool)
    if scope == "ceiling":
        from src.core_physics.clot_growth_masks import resolve_ceiling_mask

        bio = bio_cfg or BiochemConfig(phase="biochem")
        return resolve_ceiling_mask(data, device, bio).reshape(-1).to(device=device).bool()
    if scope == "support":
        from src.core_physics.clot_growth_masks import resolve_growth_support_at_time

        bio = bio_cfg or BiochemConfig(phase="biochem")
        return resolve_growth_support_at_time(
            data,
            int(time_index),
            device,
            phys_cfg,
            bio,
            phi_pred_by_time=phi_pred_by_time,
        ).reshape(-1).to(device=device).bool()
    region = supervision_region_mask(data, device, mu_cap_si, phys_cfg)
    return region.reshape(-1).to(device=device).bool()


def resolve_clot_seed_mask(
    data,
    device: torch.device,
    mu_cap_si: torch.Tensor,
    phys_cfg: PhysicsConfig,
) -> torch.Tensor:
    """Boolean clot-seed nodes for neighbor-band dilation (before dgamma slice)."""
    n = int(data.num_nodes)
    src = clot_phi_clot_seed_source()
    if src == "wall":
        return _wall_mask_from_data(data, device, n)
    if src == "none":
        return torch.zeros(n, device=device, dtype=torch.bool)
    thr = clot_phi_thresh_si(phys_cfg)
    return mu_cap_si.reshape(-1).to(device=device) >= thr


def supervision_region_mask(
    data,
    device: torch.device,
    mu_cap_si: torch.Tensor,
    phys_cfg: PhysicsConfig,
) -> torch.Tensor:
    """Nodes where φ / μ-blend loss is applied."""
    if clot_phi_mask_mode() == "sdf":
        region = sdf_supervision_mask(data, device)
    else:
        clot_seed = resolve_clot_seed_mask(data, device, mu_cap_si, phys_cfg)
        region = neighbor_supervision_mask(data, device, clot_seed)
    if clot_phi_dgamma_slice_enabled():
        bio_cfg = BiochemConfig(phase="biochem")
        clot_seed = resolve_clot_seed_mask(data, device, mu_cap_si, phys_cfg)
        region = dgamma_dx_slice_mask(data, device, region, clot_seed, bio_cfg)
    elif clot_phi_shear_min_frac() > 0.0:
        n = int(data.num_nodes)
        wall = _wall_mask_from_data(data, device, n)
        scope = (os.environ.get("CLOT_PHI_SHEAR_MAX_SCOPE") or "global").strip().lower()
        pool = region if scope in ("region", "mask", "base") else None
        region = region & shear_activity_mask(data, device, wall, pool=pool)
    return region


def wall_supervision_mask(data, device: torch.device) -> torch.Tensor:
    """Backward-compatible alias: neighbor mode without clot seeds uses wall + 1-hop only."""
    if clot_phi_mask_mode() == "sdf":
        return sdf_supervision_mask(data, device)
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    return neighbor_supervision_mask(data, device, wall)


def wall_adjacent_mask(data, device: torch.device) -> torch.Tensor:
    """Gaussian off-wall band (viz / optional ablation); training uses ``wall_supervision_mask``."""
    n = int(data.num_nodes)
    sdf = sdf_nd_from_data(data, device, n)
    wall = None
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        wall = data.mask_wall.view(-1).to(device=device)
    peak = _env_float("CLOT_PHI_D_PEAK_ND", 0.008)
    sigma = _env_float("CLOT_PHI_SIGMA_ND", 0.008)
    return adjacent_band_mask(sdf, wall, peak_nd=peak, sigma_nd=sigma).to(device=device)


def carreau_mu_si_from_uv(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    phys_cfg: PhysicsConfig,
) -> torch.Tensor:
    """Shear-thinning viscosity [Pa*s] from ND velocity components."""
    device = u_nd.device
    dtype = torch.float32
    u = u_nd.reshape(-1, 1).to(device=device, dtype=dtype)
    v = v_nd.reshape(-1, 1).to(device=device, dtype=dtype)
    G_x = data.G_x.to(device=device)
    G_y = data.G_y.to(device=device)
    du_dx = torch.sparse.mm(G_x, u)
    du_dy = torch.sparse.mm(G_y, u)
    dv_dx = torch.sparse.mm(G_x, v)
    dv_dy = torch.sparse.mm(G_y, v)
    gamma_dot_nd = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy)
    mu_inf_nd = phys_cfg.mu_inf / phys_cfg.mu_viscosity_nd_scale
    mu_0_nd = phys_cfg.mu_0 / phys_cfg.mu_viscosity_nd_scale
    lam_nd = phys_cfg.lam * float(data.u_ref.view(-1)[0].item()) / float(data.d_bar.view(-1)[0].item())
    mu_nd = carreau_yasuda_viscosity(
        gamma_dot_nd,
        torch.full_like(gamma_dot_nd, mu_inf_nd),
        torch.full_like(gamma_dot_nd, mu_0_nd),
        torch.full_like(gamma_dot_nd, lam_nd),
        float(phys_cfg.n),
        float(phys_cfg.a),
    )
    return phys_cfg.viscosity_nd_to_si(mu_nd).reshape(-1)


def gamma_dot_nd_graph_from_uv(data, u_nd: torch.Tensor, v_nd: torch.Tensor) -> torch.Tensor:
    """ND shear-rate invariant from sparse WLS gradients on ``u_nd``, ``v_nd``."""
    device = u_nd.device
    u = u_nd.reshape(-1, 1).to(device=device, dtype=torch.float32)
    v = v_nd.reshape(-1, 1).to(device=device, dtype=torch.float32)
    G_x = data.G_x.to(device=device)
    G_y = data.G_y.to(device=device)
    du_dx = torch.sparse.mm(G_x, u)
    du_dy = torch.sparse.mm(G_y, u)
    dv_dx = torch.sparse.mm(G_x, v)
    dv_dy = torch.sparse.mm(G_y, v)
    return compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy).reshape(-1)


def gamma_dot_nd_kinematic_from_uv(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Kinematic ND shear proxy ``|u| / width`` (matches COMSOL bulk ``spf.mu`` at M=1)."""
    n = int(data.num_nodes)
    width = width_nd_from_data(data, device, n)
    speed_nd = torch.sqrt(u_nd.reshape(-1).float() ** 2 + v_nd.reshape(-1).float() ** 2).clamp(min=1e-8)
    return speed_nd / width


def gamma_dot_nd_poiseuille_from_uv(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Parabolic-channel ND shear proxy (matches mesh biochem prior)."""
    from src.data_gen.lib.graph_velocity_priors import mass_conserving_umax_nd, width_nd_to_radius_nd

    n = int(data.num_nodes)
    sdf = sdf_nd_from_data(data, device, n).float()
    width = width_nd_from_data(data, device, n)
    r_nd = width_nd_to_radius_nd(width.reshape(-1, 1)).reshape(-1).clamp(min=1e-6)
    u_max_nd = mass_conserving_umax_nd(r_nd.reshape(-1, 1)).reshape(-1)
    r_lane = (r_nd - sdf).clamp(min=0.0)
    return torch.abs(-2.0 * u_max_nd * r_lane / (r_nd ** 2 + 1e-12))


def clot_phi_physics_gamma_scale() -> float:
    """Optional multiplier on resolved ND shear (1 = default)."""
    return max(_env_float("CLOT_PHI_PHYSICS_GAMMA_SCALE", 1.0), 1e-12)


def clot_phi_physics_poiseuille_scale() -> float:
    """Scale Poiseuille leg of ``max`` gamma (COMSOL ``spf.sr`` calib on patient007 ~0.85)."""
    return max(_env_float("CLOT_PHI_PHYSICS_POISEUILLE_SCALE", 0.85), 1e-12)


def _comsol_sr_sidecar_path(anchor: str):
    from pathlib import Path

    from src.utils.paths import get_project_root

    root = get_project_root()
    for rel in (
        f"data/processed/cfd_results_biochem_diag/{anchor}_sr.pt",
        f"outputs/biochem/diagnostics/{anchor}_sr.pt",
    ):
        p = root / rel
        if p.is_file():
            return p
    return None


def gamma_dot_nd_comsol_sr_from_sidecar(
    data,
    *,
    device: torch.device,
    time_index: int = 0,
    anchor: str | None = None,
) -> torch.Tensor | None:
    """COMSOL ``spf.sr`` [1/s] -> ND ``gamma_dot`` when sidecar exists (oracle/diag only)."""
    stem = (anchor or (os.environ.get("CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR") or "")).strip()
    if not stem:
        return None
    path = _comsol_sr_sidecar_path(stem)
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not torch.is_tensor(payload.get("gamma_si")):
        return None
    gamma_si = payload["gamma_si"].float()
    if gamma_si.dim() == 1:
        g_si = gamma_si.reshape(-1)
    else:
        ti = max(0, min(int(time_index), int(gamma_si.shape[0]) - 1))
        g_si = gamma_si[ti].reshape(-1)
    if int(g_si.numel()) != int(data.num_nodes):
        return None
    u_ref = float(data.u_ref.view(-1)[0].item())
    d_bar = float(data.d_bar.view(-1)[0].item())
    return (g_si.to(device=device) * (d_bar / max(u_ref, 1e-8))).clamp(min=1e-12)


def resolve_gamma_dot_nd_for_carreau(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    *,
    device: torch.device,
    mode: str | None = None,
    time_index: int | None = None,
) -> torch.Tensor:
    """Resolve ND shear rate per ``CLOT_PHI_PHYSICS_GAMMA_MODE``."""
    mode = (mode or clot_phi_physics_gamma_mode()).strip().lower()
    scale = clot_phi_physics_gamma_scale()
    if mode in ("comsol_sr", "spf_sr", "spf.sr"):
        g_sr = gamma_dot_nd_comsol_sr_from_sidecar(
            data, device=device, time_index=int(time_index or 0)
        )
        if g_sr is not None:
            return g_sr * scale
        mode = "max"
    g_graph = gamma_dot_nd_graph_from_uv(data, u_nd, v_nd)
    if mode == "graph":
        return g_graph * scale
    g_kin = gamma_dot_nd_kinematic_from_uv(data, u_nd, v_nd, device=device)
    if mode in ("kinematic", "speed_width", "speed/width", "u_over_width"):
        return g_kin * scale
    poi_scale = clot_phi_physics_poiseuille_scale()
    g_poi = gamma_dot_nd_poiseuille_from_uv(data, u_nd, v_nd, device=device) * poi_scale
    if mode == "poiseuille":
        return g_poi * scale
    if mode in ("max_kinematic", "max_graph_kinematic"):
        return torch.maximum(g_graph, g_kin) * scale
    if mode in ("max", "max_graph_poiseuille", "blend_max", "comsol"):
        return torch.maximum(torch.maximum(g_graph, g_poi), g_kin) * scale
    return torch.maximum(torch.maximum(g_graph, g_poi), g_kin) * scale


def carreau_mu_si_from_gamma_nd(
    gamma_dot_nd: torch.Tensor,
    mu_0_si: torch.Tensor,
    mu_inf_si: torch.Tensor,
    phys_cfg: PhysicsConfig,
    *,
    data,
) -> torch.Tensor:
    """Carreau-Yasuda in SI with per-node ``mu_0`` / ``mu_inf`` and ND ``gamma_dot``."""
    device = gamma_dot_nd.device
    dtype = torch.float32
    g = gamma_dot_nd.reshape(-1, 1).to(device=device, dtype=dtype)
    scale = float(phys_cfg.mu_viscosity_nd_scale)
    mu_0_nd = (mu_0_si.reshape(-1, 1).to(device=device, dtype=dtype) / scale).clamp(min=1e-8)
    mu_inf_nd = (mu_inf_si.reshape(-1, 1).to(device=device, dtype=dtype) / scale).clamp(min=1e-8)
    lam_nd = float(phys_cfg.lam) * float(data.u_ref.view(-1)[0].item()) / float(
        data.d_bar.view(-1)[0].item()
    )
    mu_nd = carreau_yasuda_viscosity(
        g,
        mu_inf_nd,
        mu_0_nd,
        torch.full_like(g, lam_nd),
        float(phys_cfg.n),
        float(phys_cfg.a),
    )
    return phys_cfg.viscosity_nd_to_si(mu_nd).reshape(-1).clamp(min=1e-8)


def comsol_carreau_mu_si_from_uv(
    data,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    gel_factor: torch.Tensor,
    phys_cfg: PhysicsConfig,
    *,
    device: torch.device,
    gamma_mode: str | None = None,
    time_index: int | None = None,
) -> torch.Tensor:
    """COMSOL ``spf.mu``: Carreau with gel-scaled limits ``mu0=mu_0*M``, ``mu_inf=mu_b*M``."""
    gf = gel_factor.reshape(-1).to(device=device, dtype=torch.float32).clamp(min=1e-8)
    mu_0_si = float(phys_cfg.mu_0) * gf
    mu_inf_si = clot_phi_physics_mu_blood_si(phys_cfg) * gf
    gamma_nd = resolve_gamma_dot_nd_for_carreau(
        data,
        u_nd,
        v_nd,
        device=device,
        mode=gamma_mode,
        time_index=time_index,
    )
    return carreau_mu_si_from_gamma_nd(
        gamma_nd, mu_0_si, mu_inf_si, phys_cfg, data=data
    )


def _resolve_gelation_legs(
    species_log1p: torch.Tensor,
    bio_cfg: BiochemConfig,
    *,
    device: torch.device,
    data=None,
    u_nd: torch.Tensor | None = None,
    v_nd: torch.Tensor | None = None,
    time_index: int | None = None,
    base_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mu1, mu2, gel)`` with COMSOL or soft gelation legs."""
    mu_ratio_max = clot_phi_physics_mu_ratio_max(bio_cfg)
    sp_si = species_log1p_nd_to_si(species_log1p.to(device=device), bio_cfg)
    fi_si = sp_si[:, 8]
    mat_si = mat_si_for_gelation_from_log1p(species_log1p[:, 11].to(device=device), bio_cfg)
    if base_mode in ("comsol", "comsol_carreau"):
        mu1 = mu1_comsol_from_mat_si(mat_si, bio_cfg, mu_ratio_max)
    else:
        mu1 = mu1_gelation_from_mat_si(mat_si, bio_cfg, mu_ratio_max)
    # Fibrin leg: COMSOL validation (docs/COMSOL_PHYSICS_VALIDATION.md) shows mu2(FI)
    # is identically 0 on patient007 -- FI peaks ~0.013 uM, far below the 0.6 uM
    # viscosity_fi_crit gelation threshold, so fibrin never gels and contributes
    # nothing to mu_eff. The clot trigger is platelet-matrix (Mat) driven. Fibrin is
    # dropped by default to remove a spurious-trigger pathway under FI decode drift;
    # re-enable with CLOT_PHI_PHYSICS_USE_FIBRIN=1 for ablations.
    if clot_phi_physics_use_fibrin():
        if base_mode in ("comsol", "comsol_carreau"):
            mu2 = mu2_comsol_from_fi_si(fi_si, bio_cfg, mu_ratio_max)
        else:
            mu2 = mu2_gelation_from_fi_si(fi_si, bio_cfg, mu_ratio_max)
        cap = clot_phi_physics_mu2_cap()
        if cap is not None:
            mu2 = mu2.clamp(max=cap)
    else:
        mu2 = torch.zeros_like(mu1)
    if clot_phi_physics_wall_mat_only() and data is not None:
        n = int(data.num_nodes)
        wall = _wall_mask_from_data(data, device, n)
        if base_mode in ("comsol", "comsol_carreau"):
            mu1 = torch.where(wall, mu1, torch.ones_like(mu1))
        else:
            mu1 = torch.where(wall, mu1, torch.zeros_like(mu1))
    onset = clot_phi_physics_gelation_onset_frac()
    if onset > 0.0 and data is not None and time_index is not None:
        from src.core_physics.clot_continuous_time import growth_time_frac

        t_frac = growth_time_frac(data, int(time_index), bio_cfg=bio_cfg)
        if t_frac < onset:
            if base_mode in ("comsol", "comsol_carreau"):
                mu1 = torch.ones_like(mu1)
            else:
                mu1 = torch.zeros_like(mu1)
            mu2 = torch.zeros_like(mu2)
    gel = mu1 + mu2
    if (
        clot_phi_physics_gelation_gate_enabled()
        and data is not None
        and u_nd is not None
        and v_nd is not None
    ):
        props = {
            "u_ref": data.u_ref.view(-1)[:1].to(device=device, dtype=gel.dtype),
            "d_bar": data.d_bar.view(-1)[:1].to(device=device, dtype=gel.dtype),
        }
        gate = clot_prior_score_flat(
            data,
            u_nd.reshape(-1).to(device=device, dtype=gel.dtype),
            v_nd.reshape(-1).to(device=device, dtype=gel.dtype),
            bio_cfg,
            props,
        ).clamp(0.0, 1.0)
        gel = gel * gate
    return mu1, mu2, gel


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def clot_phi_oracle_mu_enabled() -> bool:
    return _env_bool("CLOT_PHI_ORACLE_MU", False)


def clot_phi_physics_oracle_enabled() -> bool:
    """Analytical clot: Carreau(GT u,v) x (1 + mu1(Mat) + mu2(FI)), no learned weights."""
    return _env_bool("CLOT_PHI_PHYSICS_ORACLE", False)


def clot_phi_species_features_enabled() -> bool:
    """Append log1p(FI), log1p(Mat) from GT ``y`` to node features."""
    return _env_bool("CLOT_PHI_SPECIES_FEATURES", False)


def clot_phi_joint_bio_enabled() -> bool:
    """Train a small species head with ``L_Data_Bio`` (MSE on ``y[:,4:16]``) alongside clot head."""
    return _env_bool("CLOT_PHI_JOINT_BIO", False)


def clot_phi_physics_mu_ratio_max(bio_cfg: BiochemConfig) -> float:
    raw = (os.environ.get("CLOT_PHI_PHYSICS_MU_RATIO_MAX") or "").strip()
    if raw:
        return max(float(raw), 1.0)
    return max(float(getattr(bio_cfg, "mu_ratio_max", 80.0)), 1.0)


def clot_phi_physics_gelation_gate_enabled() -> bool:
    return _env_bool("CLOT_PHI_PHYSICS_GELATION_GATE", False)


def clot_phi_physics_mu_base_mode() -> str:
    """``carreau``, ``blood``, ``comsol`` (mu_b*gel), or ``comsol_carreau`` (spf.mu-faithful)."""
    raw = (os.environ.get("CLOT_PHI_PHYSICS_MU_BASE") or "carreau").strip().lower()
    if raw in ("blood", "mu_b", "newtonian", "mu_inf"):
        return "blood"
    if raw in ("comsol_carreau", "comsol-carreau", "spf_mu", "spf.mu"):
        return "comsol_carreau"
    if raw in ("comsol", "comsol_blood", "mu_b_gel"):
        return "comsol"
    return "carreau"


def clot_phi_physics_gamma_mode() -> str:
    """Shear rate for COMSOL Carreau: ``graph``, ``kinematic``, ``poiseuille``, or ``max``."""
    raw = (os.environ.get("CLOT_PHI_PHYSICS_GAMMA_MODE") or "").strip().lower()
    if raw in ("graph", "wls", "gradient"):
        return "graph"
    if raw in ("kinematic", "speed_width", "speed/width", "u_over_width"):
        return "kinematic"
    if raw in ("poiseuille", "poi", "analytic"):
        return "poiseuille"
    if raw in ("comsol_sr", "spf_sr", "spf.sr"):
        return "comsol_sr"
    if raw in ("max", "max_graph_poiseuille", "blend_max", "comsol", "max_graph_kinematic"):
        return "max"
    base = clot_phi_physics_mu_base_mode()
    return "max" if base == "comsol_carreau" else "graph"


def clot_phi_physics_hard_step() -> bool:
    return _env_bool("CLOT_PHI_PHYSICS_HARD_STEP", False)


def clot_phi_physics_wall_mat_only() -> bool:
    return _env_bool("CLOT_PHI_PHYSICS_WALL_MAT_ONLY", False)


def clot_phi_physics_gelation_onset_frac() -> float:
    try:
        return max(float(os.environ.get("CLOT_PHI_PHYSICS_GELATION_ONSET_FRAC", "0") or "0"), 0.0)
    except ValueError:
        return 0.0


def clot_phi_physics_mu_blood_si(phys_cfg: PhysicsConfig) -> float:
    """COMSOL mu_b infinite-shear scale [Pa*s] (= 0.035 Poise)."""
    raw = (os.environ.get("CLOT_PHI_PHYSICS_MU_BLOOD_SI") or "").strip()
    if raw:
        return max(float(raw), 1e-8)
    return float(phys_cfg.mu_inf)


def gt_mu_anchor_cap_si(data, phys_cfg: PhysicsConfig, device: torch.device) -> torch.Tensor:
    """Per-node capped COMSOL mu_eff at macro t=0 (growth baseline for GT labels)."""
    y0 = data.y[0].to(device)
    mu0 = phys_cfg.viscosity_nd_to_si(y0[:, STATE_CHANNEL_MU_EFF_ND])
    return cap_mu_eff_si(mu0)


def snapshot_clot_physics_trigger_config() -> dict[str, object]:
    return {
        "mu_base": clot_phi_physics_mu_base_mode(),
        "mu_ratio_max": os.environ.get("CLOT_PHI_PHYSICS_MU_RATIO_MAX", ""),
        "hard_step": clot_phi_physics_hard_step(),
        "gelation_gate": clot_phi_physics_gelation_gate_enabled(),
        "wall_mat_only": clot_phi_physics_wall_mat_only(),
        "use_fibrin": clot_phi_physics_use_fibrin(),
        "gelation_onset_frac": clot_phi_physics_gelation_onset_frac(),
        "subtract_t0_mu": True,
        "gt_subtract_t0_mu": True,
        "mu2_cap": os.environ.get("CLOT_PHI_PHYSICS_MU2_CAP", ""),
        "thresh_si": os.environ.get("CLOT_PHI_THRESH_SI", ""),
        "mu_blood_si": os.environ.get("CLOT_PHI_PHYSICS_MU_BLOOD_SI", ""),
        "gamma_mode": clot_phi_physics_gamma_mode(),
    }


def mat_si_for_gelation_from_log1p(
    mat_log1p: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """COMSOL ``Mat`` argument for ``mu1(Mat)`` (model units, not plt/m^2).

    COMSOL exports ``Mat`` as a concentration-like field compared at
    ``viscosity_mat_crit`` (~2e7). Extract encodes surface channels as
    ``log1p(raw/Minf)`` because ``surface_scale`` cancels in ND; gelation must
    decode with ``expm1 * Minf`` only. ``species_log1p_nd_to_si`` multiplies an
    extra ``surface_scale`` (~1e4) and breaks ``mu1`` step matching.
    """
    sp = mat_log1p.clamp(min=-10.0, max=8.0)
    return torch.expm1(sp) * float(bio_cfg.Minf)


def fi_si_for_gelation_from_log1p(
    fi_log1p: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """COMSOL ``FI`` argument for ``mu2(FI)`` in uM (matches ``viscosity_fi_crit`` = 0.6 uM).

    ``get_species_scales`` decodes FI to *working* units (SI [mol/m^3] * ``bulk_scale``).
    COMSOL ``mu2`` steps at 0.6 **uM**, so convert working -> uM:
    ``uM = (working / bulk_scale) * 1e3``  (1 mol/m^3 = 1e3 uM). Comparing FI in working
    units against 0.6 was ~1e3x too lenient and spuriously gelled fibrin that never reaches
    the physical 0.6 uM threshold (max GT FI ~0.013 uM). Parallels ``mat_si_for_gelation``.
    """
    sp = fi_log1p.clamp(min=-10.0, max=8.0)
    scale_fi = float(bio_cfg.get_species_scales(device=fi_log1p.device)[8])
    working = torch.expm1(sp) * scale_fi
    return working * (1e3 / float(bio_cfg.bulk_scale))


def species_log1p_nd_to_si(species_log1p: torch.Tensor, bio_cfg: BiochemConfig) -> torch.Tensor:
    """``y`` species channels (log1p ND) -> gelation-ready concentrations (12 bulk).

    Bulk solutes are returned in working units, EXCEPT the two channels that feed the
    COMSOL gelation steps: Mat (idx 11) -> ``mu1`` model units, FI (idx 8) -> uM for
    ``mu2``. Both crits (``viscosity_mat_crit``, ``viscosity_fi_crit``) live in those units.
    """
    scales = bio_cfg.get_species_scales(device=species_log1p.device)[:12].to(
        device=species_log1p.device, dtype=species_log1p.dtype
    )
    sp = species_log1p.reshape(-1, 12).clamp(min=-10.0, max=8.0)
    out = torch.expm1(sp) * scales.view(1, -1)
    out[:, 11] = mat_si_for_gelation_from_log1p(sp[:, 11], bio_cfg)
    out[:, 8] = fi_si_for_gelation_from_log1p(sp[:, 8], bio_cfg)
    return out


def mu1_gelation_from_mat_si(mat_si: torch.Tensor, bio_cfg: BiochemConfig, mu_ratio_max: float) -> torch.Tensor:
    t_scale = max(float(bio_cfg.soft_step_T_scale), 1e-5)
    temp = max(float(bio_cfg.viscosity_gnode_temp_mat) * t_scale, 1e-8)
    crit = float(bio_cfg.viscosity_mat_crit)
    z = torch.clamp((mat_si.reshape(-1) - crit) / temp, min=-50.0, max=50.0)
    return (float(mu_ratio_max) - 1.0) * torch.sigmoid(z)


def mu1_comsol_from_mat_si(mat_si: torch.Tensor, bio_cfg: BiochemConfig, mu_ratio_max: float) -> torch.Tensor:
    """COMSOL ``mu1(Mat)``: step from 1 -> mu_ratio_max at Mat crit."""
    crit = float(bio_cfg.viscosity_mat_crit)
    mat = mat_si.reshape(-1)
    if clot_phi_physics_hard_step():
        return torch.where(
            mat >= crit,
            torch.full_like(mat, float(mu_ratio_max)),
            torch.ones_like(mat),
        )
    t_scale = max(float(bio_cfg.soft_step_T_scale), 1e-5)
    temp = max(float(bio_cfg.viscosity_gnode_temp_mat) * t_scale, 1e-8)
    z = torch.clamp((mat - crit) / temp, min=-50.0, max=50.0)
    return 1.0 + (float(mu_ratio_max) - 1.0) * torch.sigmoid(z)


def mu2_gelation_from_fi_si(fi_si: torch.Tensor, bio_cfg: BiochemConfig, mu_ratio_max: float) -> torch.Tensor:
    t_scale = max(float(bio_cfg.soft_step_T_scale), 1e-5)
    temp = max(float(bio_cfg.viscosity_gnode_temp_fi) * t_scale, 1e-8)
    crit = float(bio_cfg.viscosity_fi_crit)
    z = torch.clamp((fi_si.reshape(-1) - crit) / temp, min=-50.0, max=50.0)
    return float(mu_ratio_max) * torch.sigmoid(z)


def mu2_comsol_from_fi_si(fi_si: torch.Tensor, bio_cfg: BiochemConfig, mu_ratio_max: float) -> torch.Tensor:
    """COMSOL ``mu2(FI)``: step from 0 -> mu_ratio_max at FI crit."""
    crit = float(bio_cfg.viscosity_fi_crit)
    fi = fi_si.reshape(-1)
    if clot_phi_physics_hard_step():
        return torch.where(
            fi >= crit,
            torch.full_like(fi, float(mu_ratio_max)),
            torch.zeros_like(fi),
        )
    return mu2_gelation_from_fi_si(fi, bio_cfg, mu_ratio_max)


def clot_phi_physics_mu2_cap() -> float | None:
    raw = (os.environ.get("CLOT_PHI_PHYSICS_MU2_CAP") or "").strip()
    if not raw:
        return None
    return max(float(raw), 0.0)


def clot_phi_physics_use_fibrin() -> bool:
    """Include the fibrin gelation leg ``mu2(FI)`` in the clot trigger.

    Default False: COMSOL validation shows FI never reaches ``viscosity_fi_crit``
    (0.6 uM) on patient007, so ``mu2(FI)`` is identically 0 and the clot is
    platelet-matrix (Mat) driven. Enable for fibrin-pathway ablations.
    """
    return _env_bool("CLOT_PHI_PHYSICS_USE_FIBRIN", False)


def physics_mu_eff_si(
    mu_c_si: torch.Tensor,
    species_log1p: torch.Tensor,
    bio_cfg: BiochemConfig,
    *,
    device: torch.device,
    data=None,
    u_nd: torch.Tensor | None = None,
    v_nd: torch.Tensor | None = None,
    phys_cfg: PhysicsConfig | None = None,
    time_index: int | None = None,
) -> torch.Tensor:
    """Gelation trigger: legacy Carreau*gel, COMSOL export, or COMSOL Carreau (spf.mu)."""
    from src.config import PhysicsConfig as _PhysicsConfig

    phys = phys_cfg or _PhysicsConfig(phase="biochem")
    base_mode = clot_phi_physics_mu_base_mode()
    _mu1, _mu2, gel = _resolve_gelation_legs(
        species_log1p,
        bio_cfg,
        device=device,
        data=data,
        u_nd=u_nd,
        v_nd=v_nd,
        time_index=time_index,
        base_mode=base_mode,
    )
    if base_mode == "comsol_carreau":
        if data is None or u_nd is None or v_nd is None:
            raise ValueError("comsol_carreau requires data, u_nd, and v_nd")
        return comsol_carreau_mu_si_from_uv(
            data,
            u_nd,
            v_nd,
            gel,
            phys,
            device=device,
            time_index=time_index,
        )
    mc = mu_c_si.reshape(-1)
    mu_b = clot_phi_physics_mu_blood_si(phys)
    if base_mode == "comsol":
        mu_out = mu_b * gel
    elif base_mode == "blood":
        mu_out = mu_b * (1.0 + gel)
    else:
        mu_out = mc * (1.0 + gel)
    return mu_out.clamp(min=1e-8)


def physics_phi_from_mu(
    mu_phys_si: torch.Tensor,
    mu_c_si: torch.Tensor,
    region: torch.Tensor | None,
    phys_cfg: PhysicsConfig,
    *,
    soft: bool,
    mu_anchor_si: torch.Tensor | None = None,
) -> torch.Tensor:
    mu_cap = cap_mu_eff_si(mu_phys_si)
    mu_ref = mu_c_si.reshape(-1)
    if mu_anchor_si is not None:
        mu_ref = mu_anchor_si.reshape(-1).to(device=mu_cap.device, dtype=mu_cap.dtype).clamp(min=1e-8)
        delta = (mu_cap.reshape(-1) - mu_ref).clamp(min=0.0)
        mu_cap = (mu_ref + delta).clamp(min=1e-8)
        if soft:
            return phi_gt_soft(mu_cap, mu_ref, region)
        growth = (mu_cap.reshape(-1) - mu_ref).clamp(min=0.0)
        thr = clot_phi_thresh_si(phys_cfg)
        phi = (growth >= thr).to(dtype=torch.float32)
        if region is None:
            return phi
        return phi * region.reshape(-1).to(dtype=torch.float32)
    if soft:
        return phi_gt_soft(mu_cap, mu_ref, region)
    raise ValueError("physics_phi_from_mu requires mu_anchor_si for growth-only GT clot labels")

def clot_phi_use_prior_features() -> bool:
    return _env_bool("CLOT_PHI_USE_PRIOR_FEATURES", False)

def clot_phi_prior_feature_count() -> int:
    try:
        return max(0, min(4, int(os.environ.get("CLOT_PHI_PRIOR_N", "2"))))
    except ValueError:
        return 2


def clot_phi_minimal_features_enabled() -> bool:
    """``[sdf, log10(gamma_dot), log1p(-dgamma/dx)_ref]`` only (3 channels)."""
    return _env_bool("CLOT_PHI_MINIMAL_FEATURES", False)


def clot_phi_hybrid_enabled() -> bool:
    """Dual head: BCE on phi + log-mu regression ``log(mu_c)+softplus(delta)``."""
    return _env_bool("CLOT_PHI_HYBRID", False)


def clot_phi_fixed_mu_from_phi_enabled() -> bool:
    """Rollout/deploy: ``mu = log_blend(mu_c, phi, mu_solid)``; no mu carry or delta head."""
    return _env_bool("CLOT_PHI_FIXED_MU_FROM_PHI", False)


def clot_phi_model_kind() -> str:
    raw = (os.environ.get("CLOT_PHI_MODEL") or "mlp").strip().lower()
    if raw in ("linear", "logistic", "lr"):
        return "linear"
    if raw in ("mpnn", "gnn", "conv"):
        return "mpnn"
    return "mlp"


def clot_phi_dropout() -> float:
    """Dropout on MLP trunk only (0 disables)."""
    return max(0.0, min(_env_float("CLOT_PHI_DROPOUT", 0.0), 0.5))


def clot_phi_mlp_depth() -> int:
    """Hidden SiLU blocks in the MLP trunk (1 = single hidden layer)."""
    return max(1, min(int(_env_float("CLOT_PHI_MLP_DEPTH", 1.0)), 3))


def clot_phi_feature_dim() -> int:
    from src.core_physics.clot_forecast import clot_forecast_extra_feature_dim
    from src.core_physics.clot_phi_rollout import clot_phi_rollout_extra_feature_dim

    extra_sp = 2 if clot_phi_species_features_enabled() else 0
    if clot_phi_minimal_features_enabled():
        base = 3 + extra_sp
    else:
        base = 7 if clot_phi_oracle_mu_enabled() else 6
        if clot_phi_use_prior_features():
            base += clot_phi_prior_feature_count()
    return base + clot_phi_rollout_extra_feature_dim() + clot_forecast_extra_feature_dim()


def rule_phi_from_mu_cap(
    mu_cap_si: torch.Tensor,
    region: torch.Tensor,
    phys_cfg: PhysicsConfig,
    *,
    mu_anchor_si: torch.Tensor,
) -> torch.Tensor:
    """Sanity floor: phi=1 where growth relu(mu - anchor) >= threshold inside the shell."""
    return phi_gt_binary(mu_cap_si, region, phys_cfg, mu_anchor_si=mu_anchor_si)


def clot_prior_rule_p_quantile() -> float:
    """Prior top-(1-p) fraction inside ceiling (default p80). Env: ``CLOT_PHI_PRIOR_RULE_P``."""
    raw = (os.environ.get("CLOT_PHI_PRIOR_RULE_P") or "0.80").strip()
    try:
        p = float(raw)
    except ValueError:
        p = 0.95
    return max(min(p, 0.99), 0.50)


def _prior_rule_env_bool(key: str, default: bool) -> bool:
    raw = (os.environ.get(key) or ("1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _prior_rule_env_frac(key: str) -> float | None:
    raw = (os.environ.get(key) or "").strip()
    if not raw or raw.lower() in ("0", "none", "off", "false"):
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return max(min(v, 0.95), 0.01)


def sweep_winner_prior_rule_config() -> ClotPriorRuleConfig:
    """Best mean band-F1 from ``sweep_clot_prior_rules`` (2026-06, post topk fix): prior_p0.80."""
    return ClotPriorRuleConfig(
        name="prior_p0.80",
        prior_p=0.80,
        use_t0_strip=False,
    )


def _prior_rule_post_gate_from_env() -> str | None:
    raw = (os.environ.get("CLOT_PHI_PRIOR_RULE_POST_GATE") or "").strip().lower()
    if not raw or raw in ("0", "none", "off", "false"):
        return None
    return raw


def prior_rule_config_from_env() -> ClotPriorRuleConfig:
    """Build deploy rule from env (defaults = sweep winner)."""
    base = sweep_winner_prior_rule_config()
    combine = (os.environ.get("CLOT_PHI_PRIOR_RULE_COMBINE") or base.combine_legs).strip().lower()
    if combine not in ("or", "and"):
        combine = base.combine_legs
    return ClotPriorRuleConfig(
        name=base.name,
        prior_p=clot_prior_rule_p_quantile() if _prior_rule_env_bool("CLOT_PHI_PRIOR_RULE_USE_PRIOR", True) else None,
        use_t0_strip=_prior_rule_env_bool("CLOT_PHI_PRIOR_RULE_T0_STRIP", base.use_t0_strip),
        flux_stream_top_frac=_prior_rule_env_frac("CLOT_PHI_PRIOR_RULE_FLUX_STREAM_TOP")
        or base.flux_stream_top_frac,
        flux_stag_top_frac=_prior_rule_env_frac("CLOT_PHI_PRIOR_RULE_FLUX_STAG_TOP")
        or base.flux_stag_top_frac,
        neg_dgamma_top_frac=_prior_rule_env_frac("CLOT_PHI_PRIOR_RULE_NEG_DGAMMA_TOP")
        or base.neg_dgamma_top_frac,
        require_on_wall=_prior_rule_env_bool("CLOT_PHI_PRIOR_RULE_ON_WALL", False),
        max_hop_from_wall=(
            int(os.environ["CLOT_PHI_PRIOR_RULE_MAX_HOP_WALL"])
            if (os.environ.get("CLOT_PHI_PRIOR_RULE_MAX_HOP_WALL") or "").strip()
            else None
        ),
        combine_legs=combine,
        post_gate=_prior_rule_post_gate_from_env() or base.post_gate,
        stag_off_wall_adjacent=_prior_rule_env_bool(
            "CLOT_PHI_PRIOR_RULE_STAG_OFF_WALL_ADJ", base.stag_off_wall_adjacent
        ),
        rank_tie_break=_prior_rule_env_bool("CLOT_PHI_PRIOR_RULE_TIE_BREAK", base.rank_tie_break),
        rank_dgamma_slice=_prior_rule_env_bool(
            "CLOT_PHI_PRIOR_RULE_RANK_DGAMMA_SLICE", base.rank_dgamma_slice
        ),
        rank_sdf_max_nd=(
            float(os.environ["CLOT_PHI_PRIOR_RULE_RANK_SDF_MAX"])
            if (os.environ.get("CLOT_PHI_PRIOR_RULE_RANK_SDF_MAX") or "").strip()
            else base.rank_sdf_max_nd
        ),
        flux_dx_raw_top_frac=_prior_rule_env_frac("CLOT_PHI_PRIOR_RULE_FLUX_DX_RAW_TOP")
        or base.flux_dx_raw_top_frac,
        skip_inlet_quantile=(
            float(os.environ["CLOT_PHI_PRIOR_RULE_SKIP_INLET_Q"])
            if (os.environ.get("CLOT_PHI_PRIOR_RULE_SKIP_INLET_Q") or "").strip()
            else base.skip_inlet_quantile
        ),
    )


@dataclass(frozen=True)
class ClotPriorRuleConfig:
    """Composable deploy rule legs (all evaluated @ t_in flow, union OR)."""

    name: str = "prior_p85_or_t0"
    prior_p: float | None = 0.85
    use_t0_strip: bool = True
    flux_stream_top_frac: float | None = None
    flux_stag_top_frac: float | None = None
    neg_dgamma_top_frac: float | None = None
    require_on_wall: bool = False
    max_hop_from_wall: int | None = None
    combine_legs: str = "or"  # ``or`` = union; ``and`` = intersection of legs
    post_gate: str | None = None  # optional final mask: t0_strip, t0_d1, t0_d2
    stag_off_wall_adjacent: bool = False  # stag leg: adjacent band, exclude no-slip wall
    rank_tie_break: bool = False  # top-k tie-break: secondary dx raw, tertiary hop from wall
    rank_dgamma_slice: bool = False  # top-k pool = ceiling intersect dgamma adhesion slice @ t_in
    rank_sdf_max_nd: float | None = None  # further restrict rank pool: sdf_nd <= cap inside ceiling
    skip_inlet_quantile: float | None = None  # drop inlet-adjacent pool: keep hop_from_inlet >= q within ceiling
    flux_dx_raw_top_frac: float | None = None  # top fraction by unclamped flux_path_dx_raw

    def describe(self) -> str:
        parts: list[str] = []
        if self.prior_p is not None:
            parts.append(f"prior_p{self.prior_p:.2f}")
        if self.use_t0_strip:
            parts.append("t0_strip")
        if self.flux_stream_top_frac is not None:
            parts.append(f"flux_stream_top{int(100*self.flux_stream_top_frac)}")
        if self.flux_stag_top_frac is not None:
            tag = f"flux_stag_top{int(100*self.flux_stag_top_frac)}"
            if self.stag_off_wall_adjacent:
                tag += "_offwall"
            parts.append(tag)
        if self.flux_dx_raw_top_frac is not None:
            parts.append(f"dx_raw_top{int(100*self.flux_dx_raw_top_frac)}")
        if self.neg_dgamma_top_frac is not None:
            parts.append(f"neg_dx_top{int(100*self.neg_dgamma_top_frac)}")
        if self.require_on_wall:
            parts.append("on_wall")
        if self.max_hop_from_wall is not None:
            parts.append(f"hop_wall<={self.max_hop_from_wall}")
        if self.combine_legs.strip().lower() == "and" and len(parts) > 1:
            parts.append("AND")
        if self.post_gate:
            parts.append(f"gate_{self.post_gate}")
        if self.rank_tie_break:
            parts.append("tie_dx_hop")
        if self.rank_dgamma_slice:
            parts.append("dgamma_rank")
        if self.rank_sdf_max_nd is not None:
            parts.append(f"sdf<={self.rank_sdf_max_nd:.3f}")
        if self.skip_inlet_quantile is not None:
            parts.append(f"skip_inlet_q{int(100 * self.skip_inlet_quantile)}")
        if parts:
            return "|".join(parts)
        return self.name if self.name else "empty"


def default_prior_rule_config() -> ClotPriorRuleConfig:
    return prior_rule_config_from_env()


def _noslip_wall_mask(wall: torch.Tensor, u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Wall nodes with zero assigned velocity (no-slip BC patch)."""
    speed = u.reshape(-1).abs() + v.reshape(-1).abs()
    return wall.reshape(-1).bool() & (speed <= eps)


def _rank01(x: torch.Tensor) -> torch.Tensor:
    xmin = x.min()
    span = (x.max() - xmin).clamp(min=1e-12)
    return (x - xmin) / span


def _top_frac_mask(
    values: torch.Tensor,
    mask: torch.Tensor,
    top_frac: float,
    *,
    tie_dx: torch.Tensor | None = None,
    tie_hop: torch.Tensor | None = None,
) -> torch.Tensor:
    """Top ``ceil(frac * n)`` nodes inside ``mask`` (exact count, optional lex tie-break)."""
    frac = max(min(float(top_frac), 0.95), 0.01)
    n = int(mask.sum())
    if n <= 0:
        return torch.zeros_like(mask.reshape(-1), dtype=torch.bool)
    k = max(int(math.ceil(frac * n)), 1)
    k = min(k, n)
    idx_all = torch.where(mask.reshape(-1))[0]
    vals = values.reshape(-1)[idx_all].to(dtype=torch.float32)
    if tie_dx is not None and tie_hop is not None:
        dx = tie_dx.reshape(-1)[idx_all].to(dtype=torch.float32)
        hop = tie_hop.reshape(-1)[idx_all].to(dtype=torch.float32)
        score = (
            _rank01(vals) * 1_000_000.0
            + _rank01(dx) * 1_000.0
            + _rank01(-hop) * 1.0
        )
        pick = torch.argsort(score, descending=True, stable=True)[:k]
    else:
        pick = torch.topk(vals, k).indices
    out = torch.zeros(int(values.numel()), device=values.device, dtype=torch.bool)
    out[idx_all[pick]] = True
    return out


def predict_phi_prior_rule(
    data,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    rule: ClotPriorRuleConfig | None = None,
    t_in: int = 0,
    ceiling_hops: int | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Deploy rule: phi=1 on union of configured legs inside ceiling (optional filters)."""
    from src.core_physics.clot_growth_masks import resolve_ceiling_mask, resolve_t0_dgamma_wall_mask

    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")

    cfg = rule or default_prior_rule_config()
    y_in = data.y[int(t_in)].to(device=device, dtype=torch.float32)
    u = y_in[:, 0]
    v = y_in[:, 1]
    props = _anchor_flow_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior = clot_prior_score_flat(data, u, v, bio_cfg, props).reshape(-1)

    ceiling = resolve_ceiling_mask(data, device, bio_cfg, ceiling_hops=ceiling_hops)
    t0_strip = resolve_t0_dgamma_wall_mask(data, device, bio_cfg)
    wall = _wall_mask_from_data(data, device, int(data.num_nodes))
    noslip = _noslip_wall_mask(wall, u, v)
    rank_mask = ceiling
    n_rank_mask = int(ceiling.sum().item())
    if cfg.rank_dgamma_slice:
        clot_seed = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)
        rank_mask = dgamma_dx_slice_mask(data, device, ceiling, clot_seed, bio_cfg)
        n_rank_mask = int(rank_mask.sum().item())
    if cfg.rank_sdf_max_nd is not None:
        sdf = sdf_nd_from_data(data, device, int(data.num_nodes))
        rank_mask = rank_mask & (sdf <= float(cfg.rank_sdf_max_nd))
        n_rank_mask = int(rank_mask.sum().item())
    if cfg.skip_inlet_quantile is not None and hasattr(data, "mask_inlet") and data.mask_inlet is not None:
        n_nodes = int(data.num_nodes)
        inlet = data.mask_inlet.view(-1).to(device=device).bool()
        if int(inlet.numel()) == n_nodes and bool(inlet.any().item()):
            hop_in = _hop_distance_from_seed(inlet, data.edge_index.to(device=device)).float()
            eligible = rank_mask & (hop_in > 0)
            if bool(eligible.any().item()):
                q = max(min(float(cfg.skip_inlet_quantile), 0.95), 0.0)
                thr = torch.quantile(hop_in[eligible], q)
                rank_mask = rank_mask & (hop_in >= thr)
                n_rank_mask = int(rank_mask.sum().item())
    dx_raw = fields.flux_path_dx_raw.reshape(-1)
    neg_dx_flow = (-fields.dgamma_dx_phys).clamp(min=0.0).reshape(-1)
    hop_wall = _hop_distance_from_seed(wall, data.edge_index.to(device=device)).float()
    tie_dx = dx_raw if cfg.rank_tie_break else None
    tie_hop = hop_wall if cfg.rank_tie_break else None
    stag_rank_mask = rank_mask
    if cfg.stag_off_wall_adjacent:
        stag_rank_mask = rank_mask & fields.adjacent_band.reshape(-1).bool() & ~noslip

    legs: list[torch.Tensor] = []
    prior_thr = float("nan")
    n_prior_leg = 0
    n_stag_leg = 0
    n_dx_raw_leg = 0
    if cfg.prior_p is not None:
        p = max(min(float(cfg.prior_p), 0.99), 0.50)
        top_frac = max(1.0 - p, 0.01)
        leg_prior = _top_frac_mask(prior, rank_mask, top_frac, tie_dx=tie_dx, tie_hop=tie_hop)
        if bool(leg_prior.any().item()):
            prior_thr = float(prior.reshape(-1)[leg_prior].min().item())
        legs.append(leg_prior)
        n_prior_leg = int(leg_prior.sum().item())
    if cfg.use_t0_strip:
        legs.append(t0_strip)
    if cfg.flux_stream_top_frac is not None:
        legs.append(
            _top_frac_mask(
                fields.flux_path_stream.reshape(-1),
                rank_mask,
                cfg.flux_stream_top_frac,
                tie_dx=tie_dx,
                tie_hop=tie_hop,
            )
        )
    if cfg.flux_stag_top_frac is not None:
        leg_stag = _top_frac_mask(
            fields.flux_stag.reshape(-1),
            stag_rank_mask,
            cfg.flux_stag_top_frac,
            tie_dx=tie_dx,
            tie_hop=tie_hop,
        )
        legs.append(leg_stag)
        n_stag_leg = int(leg_stag.sum().item())
    if cfg.flux_dx_raw_top_frac is not None:
        leg_dx = _top_frac_mask(
            dx_raw,
            rank_mask,
            cfg.flux_dx_raw_top_frac,
            tie_dx=neg_dx_flow,
            tie_hop=tie_hop,
        )
        legs.append(leg_dx)
        n_dx_raw_leg = int(leg_dx.sum().item())
    if cfg.neg_dgamma_top_frac is not None:
        legs.append(
            _top_frac_mask(
                neg_dx_flow,
                rank_mask,
                cfg.neg_dgamma_top_frac,
                tie_dx=dx_raw,
                tie_hop=tie_hop,
            )
        )

    if not legs:
        flag = torch.zeros(int(data.num_nodes), device=device, dtype=torch.bool)
    elif str(cfg.combine_legs).strip().lower() == "and" and len(legs) > 1:
        flag = legs[0]
        for leg in legs[1:]:
            flag = flag & leg
    else:
        flag = legs[0]
        for leg in legs[1:]:
            flag = flag | leg

    if cfg.require_on_wall:
        flag = flag & wall
    if cfg.max_hop_from_wall is not None and hop_wall is not None:
        flag = flag & (hop_wall <= float(cfg.max_hop_from_wall))

    post_gate = (str(cfg.post_gate).strip().lower() if cfg.post_gate else "") or ""
    n_post_gate = int(ceiling.sum().item())
    if post_gate in ("t0_strip", "t0"):
        flag = flag & t0_strip
        n_post_gate = int(t0_strip.sum().item())
    elif post_gate in ("t0_d1", "t0_strip_d1"):
        from src.core_physics.clot_growth_masks import graph_dilate_hops

        gate = graph_dilate_hops(t0_strip, data.edge_index.to(device=device), 1) & ceiling
        flag = flag & gate
        n_post_gate = int(gate.sum().item())
    elif post_gate in ("t0_d2", "t0_strip_d2"):
        from src.core_physics.clot_growth_masks import graph_dilate_hops

        gate = graph_dilate_hops(t0_strip, data.edge_index.to(device=device), 2) & ceiling
        flag = flag & gate
        n_post_gate = int(gate.sum().item())

    phi = flag.to(dtype=torch.float32)
    meta: dict[str, float | int] = {
        "rule": cfg.describe(),
        "prior_p": float(cfg.prior_p) if cfg.prior_p is not None else -1.0,
        "prior_thr": prior_thr,
        "n_ceiling": int(ceiling.sum().item()),
        "n_rank_mask": n_rank_mask,
        "rank_dgamma_slice": int(cfg.rank_dgamma_slice),
        "rank_sdf_max_nd": float(cfg.rank_sdf_max_nd) if cfg.rank_sdf_max_nd is not None else -1.0,
        "n_t0_strip": int(t0_strip.sum().item()),
        "n_prior_leg": n_prior_leg,
        "n_stag_leg": n_stag_leg,
        "n_dx_raw_leg": n_dx_raw_leg,
        "stag_off_wall_adjacent": int(cfg.stag_off_wall_adjacent),
        "rank_tie_break": int(cfg.rank_tie_break),
        "n_prior_hit": n_prior_leg,
        "n_post_gate": n_post_gate,
        "post_gate": post_gate,
        "n_flag": int(flag.sum().item()),
    }
    return phi, meta


def _hop_distance_from_seed(seed: torch.Tensor, edge_index: torch.Tensor, max_hops: int = 64) -> torch.Tensor:
    from src.core_physics.clot_growth_masks import graph_dilate_hops

    n = int(seed.numel())
    dist = torch.full((n,), max_hops + 1, dtype=torch.long, device=seed.device)
    if not bool(seed.any().item()):
        return dist
    dist[seed] = 0
    active = seed.clone()
    for h in range(max_hops):
        nxt = graph_dilate_hops(active, edge_index, 1) & ~active
        if not bool(nxt.any().item()):
            break
        dist[nxt] = h + 1
        active = active | nxt
    return dist


def _anchor_flow_props(data, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == data.num_nodes:
        u_ref = data.u_ref.to(device=device, dtype=torch.float32).reshape(-1)[:1]
        d_bar = data.d_bar.to(device=device, dtype=torch.float32).reshape(-1)[:1]
    else:
        u_ref = torch.as_tensor(data.u_ref, device=device, dtype=torch.float32).reshape(1)
        d_bar = torch.as_tensor(data.d_bar, device=device, dtype=torch.float32).reshape(1)
    return {"u_ref": u_ref, "d_bar": d_bar}


def predict_phi_prior_rule_baseline(
    data,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    t_in: int = 0,
    ceiling_hops: int | None = None,
    rule: ClotPriorRuleConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Deploy rule: phi=1 where (prior >= p85 inside ceiling) OR t0 dgamma strip (default)."""
    return predict_phi_prior_rule(
        data,
        device,
        bio_cfg,
        rule=rule or default_prior_rule_config(),
        t_in=t_in,
        ceiling_hops=ceiling_hops,
    )


def predict_prior_rule_deploy(
    data,
    t_out: int,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    t_in: int = 0,
    rule: ClotPriorRuleConfig | None = None,
) -> tuple[object, torch.Tensor, torch.Tensor, dict[str, float | int]]:
    """Static-final deploy: t_in flow -> phi rule -> fixed-mu blend -> support projection."""
    from src.core_physics.clot_forecast import build_clot_forecast_pair_step

    step = build_clot_forecast_pair_step(
        data,
        int(t_in),
        int(t_out),
        phys_cfg,
        bio_cfg,
        device,
    )
    phi, meta = predict_phi_prior_rule_baseline(
        data, device, bio_cfg, t_in=int(t_in), rule=rule
    )
    mu = log_blend_mu_eff_si(step.mu_c_si, phi)
    mu = project_deploy_mu_with_support(
        data=data,
        step=step,
        mu_pred=mu,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        forecast_one_step=True,
        time_index=int(t_out),
        bulk_time_index=int(t_out),
    )
    return step, phi, mu, meta


def mu_growth_clot_binary_mask(
    mu_si: torch.Tensor,
    mu_anchor_si: torch.Tensor,
    thresh_si: float,
) -> torch.Tensor:
    """Growth-only clot mask: ``relu(mu - anchor) >= thresh``."""
    growth = (mu_si.reshape(-1) - mu_anchor_si.reshape(-1)).clamp(min=0.0)
    return growth >= float(thresh_si)


def phi_gt_binary(
    mu_cap_si: torch.Tensor,
    region: torch.Tensor | None,
    phys_cfg: PhysicsConfig,
    *,
    mu_anchor_si: torch.Tensor,
) -> torch.Tensor:
    """Binary GT clot: growth ``relu(mu - anchor) >= thresh``. Region masks band when set."""
    thr = clot_phi_thresh_si(phys_cfg)
    phi = mu_growth_clot_binary_mask(mu_cap_si, mu_anchor_si, thr).to(dtype=torch.float32)
    if region is None:
        return phi
    return phi * region.reshape(-1).to(dtype=torch.float32)


def phi_gt_soft(
    mu_cap_si: torch.Tensor,
    mu_c_si: torch.Tensor,
    region: torch.Tensor | None,
) -> torch.Tensor:
    """Soft phi* from log-blend inversion with fixed mu_solid (stable labels when mu > mu_c)."""
    solid = clot_phi_mu_solid_si()
    mc = mu_c_si.reshape(-1).clamp(min=1e-8)
    mu = mu_cap_si.reshape(-1).clamp(min=1e-8)
    denom = torch.log(torch.tensor(solid, device=mu.device, dtype=mu.dtype)) - torch.log(mc)
    denom = denom.clamp(min=1e-6)
    phi = ((torch.log(mu) - torch.log(mc)) / denom).clamp(0.0, 1.0)
    if region is None:
        return phi
    return phi * region.reshape(-1).to(dtype=torch.float32)


def resolve_phi_gt_labels(
    mu_cap_si: torch.Tensor,
    mu_c_si: torch.Tensor,
    region: torch.Tensor | None,
    phys_cfg: PhysicsConfig,
    *,
    soft: bool,
    mu_anchor_si: torch.Tensor | None = None,
) -> torch.Tensor:
    """GT clot labels: growth above per-node COMSOL mu at macro t=0."""
    anchor = (
        mu_anchor_si.reshape(-1)
        if mu_anchor_si is not None
        else mu_c_si.reshape(-1)
    ).to(device=mu_cap_si.device, dtype=mu_cap_si.dtype)
    if soft:
        return phi_gt_soft(mu_cap_si, anchor, region)
    growth = (mu_cap_si.reshape(-1) - anchor).clamp(min=0.0)
    thr = clot_phi_thresh_si(phys_cfg)
    phi = (growth >= thr).to(dtype=torch.float32)
    if region is None:
        return phi
    return phi * region.reshape(-1).to(dtype=torch.float32)


def mu_eff_from_delta_log_si(mu_c_si: torch.Tensor, delta_log_mu: torch.Tensor) -> torch.Tensor:
    """``mu_eff = mu_c * exp(delta_log)`` (K10d-style additive log, SI)."""
    mc = mu_c_si.reshape(-1).clamp(min=1e-8)
    d = delta_log_mu.reshape(-1)
    return (mc * torch.exp(d)).clamp(min=1e-8)


def log_blend_mu_eff_si(mu_c_si: torch.Tensor, phi: torch.Tensor, mu_solid_si: float | None = None) -> torch.Tensor:
    """log mu_eff = (1-phi) log mu_c + phi log mu_solid."""
    solid = float(mu_solid_si if mu_solid_si is not None else clot_phi_mu_solid_si())
    mc = mu_c_si.reshape(-1).clamp(min=1e-8)
    ph = phi.reshape(-1).clamp(0.0, 1.0)
    log_mu = (1.0 - ph) * torch.log(mc) + ph * torch.log(torch.tensor(solid, device=mc.device, dtype=mc.dtype))
    return torch.exp(log_mu)


def mu_eff_from_carried_phi(
    mu_c_si: torch.Tensor,
    phi_prev: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Deploy/rollout mu seeds from carried phi (bulk Carreau when phi_prev is None)."""
    mc = mu_c_si.reshape(-1).to(device=device, dtype=torch.float32).clamp(min=1e-8)
    if phi_prev is None:
        return mc
    return log_blend_mu_eff_si(mu_c_si, phi_prev).reshape(-1).to(device=device)


def snapshot_phi_only_rollout_config() -> dict[str, object]:
    return {
        "fixed_mu_from_phi": clot_phi_fixed_mu_from_phi_enabled(),
        "mu_solid_si": clot_phi_mu_solid_si(),
    }


def clot_phi_mesh_aux_lambda() -> float:
    """Auxiliary BCE on full eligible lumen at forecast target time (helps clot_shape)."""
    return max(float(os.environ.get("CLOT_PHI_MESH_AUX_LAMBDA", "0") or "0"), 0.0)


def clot_phi_mesh_bulk_lambda() -> float:
    """Penalize phi>0 on bulk nodes (mu_gt below clot threshold) at target time."""
    return max(float(os.environ.get("CLOT_PHI_MESH_BULK_LAMBDA", "0") or "0"), 0.0)


def clot_phi_shape_use_t_out_mu() -> bool:
    """One-step clot_shape: blend phi with Carreau mu_c @ t_out (not t_in)."""
    return _env_bool("CLOT_PHI_SHAPE_USE_T_OUT", True)


def lumen_eligible_mask(
    data,
    device: torch.device,
    *,
    n_nodes: int | None = None,
) -> torch.Tensor:
    """Eligible lumen nodes for mesh-wide aux loss / shape (wall + near-wall band)."""
    n = int(n_nodes or data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    return _lumen_supervision_eligible(data, device, wall, n).bool()


def snapshot_mesh_aux_config() -> dict[str, float | bool]:
    return {
        "mesh_aux_lambda": clot_phi_mesh_aux_lambda(),
        "mesh_bulk_lambda": clot_phi_mesh_bulk_lambda(),
        "shape_use_t_out_mu": clot_phi_shape_use_t_out_mu(),
    }


def clot_phi_hard_support_projection_enabled() -> bool:
    """When true, deployed mu = Carreau bulk off support band B_t (CAVO step 3-4)."""
    raw = (os.environ.get("CLOT_PHI_HARD_SUPPORT_PROJECTION") or "").strip().lower()
    if not raw:
        try:
            from src.core_physics.clot_forecast import clot_forecast_one_step_enabled

            return clot_forecast_one_step_enabled()
        except ImportError:
            return False
    return raw in ("1", "true", "yes", "on")


def clot_support_band_mode() -> str:
    """Physics support B_t for hard mu projection (independent of loss_mask).

    - ``physics`` (default): dgamma wall @ ref + 1-hop from clot seeds @ mu_in / current step.
    - ``frozen_t0``: seeds + dgamma band fixed from COMSOL mu @ t=0 only (deploy_band).
    - ``ceiling_growth``: t0 dgamma growth seed + hop support capped by wall+K ceiling.
    - ``loss_mask``: use step loss_mask (legacy / debug only).
    """
    raw = (os.environ.get("CLOT_PHI_SUPPORT_BAND") or "physics").strip().lower()
    if raw in ("frozen_t0", "t0", "deploy_band", "b0"):
        return "frozen_t0"
    if raw in ("ceiling_growth", "ceiling", "hop_growth", "growth"):
        return "ceiling_growth"
    if raw in ("loss_mask", "loss", "legacy"):
        return "loss_mask"
    return "physics"


def resolve_clot_support_band(
    data,
    device: torch.device,
    mu_seed_cap_si: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig | None = None,
    *,
    frozen_t0: bool = False,
) -> torch.Tensor:
    """Deploy physics support B_t: dgamma-sliced neighbor shell from clot seeds.

    Seeds default to ``mu_seed_cap_si`` (mu @ t_in for forecast). ``frozen_t0`` seeds from
    COMSOL mu @ graph index 0 only (band envelope fixed for the trajectory).
    """
    bio = bio_cfg or BiochemConfig(phase="biochem")
    if frozen_t0 or clot_support_band_mode() == "frozen_t0":
        if hasattr(data, "y") and torch.is_tensor(data.y) and data.y.dim() == 3:
            y0 = data.y[0].to(device)
            mu_seed_cap_si = cap_mu_eff_si(
                phys_cfg.viscosity_nd_to_si(y0[:, STATE_CHANNEL_MU_EFF_ND])
            )
    mu_seed = mu_seed_cap_si.reshape(-1).to(device=device)
    thr = clot_phi_thresh_si(phys_cfg)
    clot_seed = mu_seed >= thr
    region = neighbor_supervision_mask(data, device, clot_seed)
    if clot_phi_dgamma_slice_enabled():
        region = dgamma_dx_slice_mask(data, device, region, clot_seed, bio)
    return region.reshape(-1).bool()


def apply_clot_support_projection(
    mu_c_si: torch.Tensor,
    mu_eff_si: torch.Tensor,
    support_band: torch.Tensor,
) -> torch.Tensor:
    """Inside B_t keep model mu; outside B_t force Carreau bulk (no clot commit)."""
    mc = mu_c_si.reshape(-1)
    mu = mu_eff_si.reshape(-1)
    band = support_band.reshape(-1).to(device=mu.device, dtype=torch.bool)
    out = mc.clone()
    if bool(band.any().item()):
        out[band] = mu[band]
    return out.reshape_as(mu)


def project_mu_from_phi_with_support(
    mu_c_si: torch.Tensor,
    phi: torch.Tensor,
    support_band: torch.Tensor,
    *,
    mu_solid_si: float | None = None,
) -> torch.Tensor:
    """log-blend phi -> mu, then hard-project onto support band."""
    mu_blend = log_blend_mu_eff_si(mu_c_si, phi, mu_solid_si=mu_solid_si)
    return apply_clot_support_projection(mu_c_si, mu_blend, support_band)


def resolve_clot_support_band_for_step(
    data,
    device: torch.device,
    step: "ClotPhiStepBatch",
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig | None = None,
    *,
    forecast_one_step: bool = False,
    time_index: int | None = None,
    phi_pred_by_time: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Resolve B_t for a train/eval step (physics, frozen t=0, ceiling growth, or loss_mask)."""
    mode = clot_support_band_mode()
    if mode == "ceiling_growth":
        from src.core_physics.clot_growth_masks import resolve_growth_support_at_time

        bio = bio_cfg or BiochemConfig(phase="biochem")
        t = 0 if time_index is None else int(time_index)
        return resolve_growth_support_at_time(
            data,
            t,
            device,
            phys_cfg,
            bio,
            phi_pred_by_time=phi_pred_by_time,
        )
    if mode == "loss_mask":
        return step.loss_mask.reshape(-1).to(device=device).bool()
    frozen = mode == "frozen_t0"
    if not frozen and forecast_one_step:
        try:
            from src.core_physics.clot_forecast import clot_forecast_mask_mode

            frozen = clot_forecast_mask_mode() == "deploy_band"
        except ImportError:
            pass
    mu_seed = step.mu_gt_cap.reshape(-1)
    if (
        forecast_one_step
        and not frozen
        and getattr(step, "mu_in_cap", None) is not None
    ):
        mu_seed = step.mu_in_cap.reshape(-1)
    return resolve_clot_support_band(
        data,
        device,
        mu_seed,
        phys_cfg,
        bio_cfg,
        frozen_t0=frozen,
    )


def project_deploy_mu_with_support(
    *,
    data,
    step: "ClotPhiStepBatch",
    mu_pred: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    forecast_one_step: bool = False,
    time_index: int | None = None,
    bulk_time_index: int | None = None,
    phi_pred_by_time: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Hard projection: model mu on support B_t; Carreau bulk elsewhere."""
    if not clot_phi_hard_support_projection_enabled():
        return mu_pred.reshape(-1)
    band = resolve_clot_support_band_for_step(
        data,
        device,
        step,
        phys_cfg,
        bio_cfg,
        forecast_one_step=forecast_one_step,
        time_index=time_index,
        phi_pred_by_time=phi_pred_by_time,
    )
    if bulk_time_index is not None:
        from src.core_physics.clot_growth_masks import resolve_bulk_carreau_mu_si

        mu_bulk = resolve_bulk_carreau_mu_si(data, int(bulk_time_index), phys_cfg, device)
    else:
        mu_bulk = step.mu_c_si
    return apply_clot_support_projection(mu_bulk, mu_pred, band)


def snapshot_clot_support_config() -> dict[str, object]:
    from src.core_physics.clot_growth_masks import snapshot_clot_growth_config

    return {
        "hard_support_projection": clot_phi_hard_support_projection_enabled(),
        "support_band": clot_support_band_mode(),
        **snapshot_clot_growth_config(),
    }


def node_features_from_gt(
    data,
    y_slice: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    *,
    device: torch.device,
    mu_cap_si: torch.Tensor | None = None,
    time_index: int | None = None,
    u_nd_override: torch.Tensor | None = None,
    v_nd_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-node features for phi head.

  Minimal (``CLOT_PHI_MINIMAL_FEATURES=1``): ``[sdf, log10(gamma_dot), log1p(-dgamma/dx)_ref]``
    Base: [sdf, u, v, FI, Mat, log10(gamma_dot)]
    Optional:
    - oracle log(mu_cap): enabled by ``CLOT_PHI_ORACLE_MU=1`` (debug only; leaks GT μ)
    - kinematic clot-risk prior columns: enabled by ``CLOT_PHI_USE_PRIOR_FEATURES=1`` (no μ leak)
    """
    n = int(data.num_nodes)
    sdf = sdf_nd_from_data(data, device, n)
    u = (
        u_nd_override.to(device=device, dtype=torch.float32)
        if u_nd_override is not None
        else y_slice[:, 0].to(device=device, dtype=torch.float32)
    )
    v = (
        v_nd_override.to(device=device, dtype=torch.float32)
        if v_nd_override is not None
        else y_slice[:, 1].to(device=device, dtype=torch.float32)
    )
  # BIO_Y: FI=12, Mat=15
    fi = y_slice[:, 12].to(device=device, dtype=torch.float32)
    mat = y_slice[:, 15].to(device=device, dtype=torch.float32)
    u_col = u.reshape(-1, 1)
    G_x = data.G_x.to(device=device)
    G_y = data.G_y.to(device=device)
    du_dx = torch.sparse.mm(G_x, u_col).squeeze(1)
    du_dy = torch.sparse.mm(G_y, u_col).squeeze(1)
    dv_dx = torch.sparse.mm(G_x, v.reshape(-1, 1)).squeeze(1)
    dv_dy = torch.sparse.mm(G_y, v.reshape(-1, 1)).squeeze(1)
    gamma = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy).clamp(min=1e-8)
    log_g = torch.log10(gamma)
    if clot_phi_minimal_features_enabled():
        ti_dg = (
            clot_phi_dgamma_feature_time_index(data, int(time_index))
            if time_index is not None
            else clot_phi_dgamma_ref_time_index(data)
        )
        neg_dx = gt_neg_dgamma_dx_phys(data, ti_dg, bio_cfg, device)
        cols = [sdf, log_g, torch.log1p(neg_dx)]
        if clot_phi_species_features_enabled():
            cols.extend([fi, mat])
        return torch.stack(cols, dim=1)
    cols = [sdf, u, v, fi, mat, log_g]
    if clot_phi_oracle_mu_enabled() and mu_cap_si is not None:
        log_mu = torch.log(mu_cap_si.reshape(-1).to(device=device, dtype=torch.float32).clamp(min=1e-8))
        cols.append(log_mu)
    if clot_phi_use_prior_features():
        props = {"u_ref": data.u_ref.view(-1)[:1], "d_bar": data.d_bar.view(-1)[:1]}
        pf = clot_prior_features(data, u, v, bio_cfg, props, n_features=clot_phi_prior_feature_count()).to(
            device=device, dtype=torch.float32
        )
        for j in range(int(pf.shape[1])):
            cols.append(pf[:, j])
    return torch.stack(cols, dim=1)


class ClotPhiMLP(nn.Module):
    """Tiny per-node classifier for wall-adjacent clot phase."""

    def __init__(self, in_dim: int = 6, hidden: int = 64):
        super().__init__()
        h = max(int(hidden), 16)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class ClotPhiSpeciesHead(nn.Module):
    """Predict COMSOL log1p species (12 ch) from kinematic / optional species features."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 8)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, 12),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClotPhiMPNNHybrid(nn.Module):
    """One-hop message passing + hybrid phi / delta_log_mu heads (forecast prong C)."""

    uses_mpnn = True

    def __init__(self, in_dim: int = 3, hidden: int = 32):
        super().__init__()
        from torch_geometric.nn import MessagePassing

        h = max(int(hidden), 8)

        class _Conv(MessagePassing):
            def __init__(self) -> None:
                super().__init__(aggr="add")
                self.lin_nei = nn.Linear(in_dim, h)
                self.lin_self = nn.Linear(in_dim, h)

            def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
                return self.propagate(edge_index, x=x)

            def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
                return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))

        self.conv = _Conv()
        drop = clot_phi_dropout()
        self.post = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Dropout(p=drop) if drop > 0.0 else nn.Identity(),
        )
        self.phi_fc = nn.Linear(h, 1)
        self.dlog_fc = nn.Linear(h, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _hidden(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.post(self.conv(x, edge_index))

    def forward_logits(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("ClotPhiMPNNHybrid requires edge_index")
        return self.phi_fc(self._hidden(x, edge_index)).squeeze(-1)

    def forward_delta_log_mu(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        if edge_index is None:
            raise ValueError("ClotPhiMPNNHybrid requires edge_index")
        return F.softplus(self.dlog_fc(self._hidden(x, edge_index)).squeeze(-1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x, edge_index))


class ClotPhiHybrid(nn.Module):
    """Minimal hybrid: phi logits + ``delta_log_mu`` (``mu = mu_c * exp(delta)``)."""

    def __init__(self, in_dim: int = 3, hidden: int = 16, *, linear: bool = False):
        super().__init__()
        self.linear = bool(linear)
        if self.linear:
            self.phi_fc = nn.Linear(in_dim, 1)
            self.dlog_fc = nn.Linear(in_dim, 1)
        else:
            h = max(int(hidden), 4)
            depth = clot_phi_mlp_depth()
            drop = clot_phi_dropout()
            layers: list[nn.Module] = []
            d_in = in_dim
            for i in range(depth):
                layers.extend([nn.Linear(d_in, h), nn.SiLU()])
                if drop > 0.0 and i + 1 < depth:
                    layers.append(nn.Dropout(p=drop))
                d_in = h
            if drop > 0.0 and depth == 1:
                layers.append(nn.Dropout(p=drop))
            self.trunk = nn.Sequential(*layers)
            self.phi_fc = nn.Linear(h, 1)
            self.dlog_fc = nn.Linear(h, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _hidden(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear:
            return x
        return self.trunk(x)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi_fc(self._hidden(x)).squeeze(-1)

    def forward_delta_log_mu(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.dlog_fc(self._hidden(x)).squeeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


def build_clot_phi_model(in_dim: int, hidden: int) -> nn.Module:
    """Factory: ``CLOT_PHI_HYBRID`` + ``CLOT_PHI_MODEL=linear|mlp|mpnn``."""
    kind = clot_phi_model_kind()
    if clot_phi_hybrid_enabled():
        if kind == "mpnn":
            return ClotPhiMPNNHybrid(in_dim=in_dim, hidden=hidden)
        return ClotPhiHybrid(in_dim=in_dim, hidden=hidden, linear=(kind == "linear"))
    return ClotPhiMLP(in_dim=in_dim, hidden=hidden)


def clot_phi_model_uses_mpnn(model: nn.Module) -> bool:
    return bool(getattr(model, "uses_mpnn", False))


@dataclass
class ClotPhiStepBatch:
    features: torch.Tensor
    phi_gt: torch.Tensor
    mu_c_si: torch.Tensor
    mu_gt_cap: torch.Tensor
    region: torch.Tensor
    loss_mask: torch.Tensor
    species_log_gt: torch.Tensor
    u_flow_nd: torch.Tensor
    v_flow_nd: torch.Tensor
    mu_in_cap: torch.Tensor | None = None
    phi_in_gt: torch.Tensor | None = None


def build_clot_phi_step(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    u_nd_override: torch.Tensor | None = None,
    v_nd_override: torch.Tensor | None = None,
    y_slice_override: torch.Tensor | None = None,
    species_log_override: torch.Tensor | None = None,
    rollout_state: "ClotPhiRolloutState | None" = None,
    train_epoch: int | None = None,
) -> ClotPhiStepBatch:
    """One time slice: labels and features on wall-adjacent nodes."""
    from src.core_physics.clot_phi_rollout import (
        ClotPhiRolloutState,
        append_rollout_carry_features,
        clot_phi_rollout_enabled,
        clot_phi_vel_source,
        resolve_uv_for_rollout_step,
    )

    y_gt = data.y[time_index].to(device)
    y = (
        y_slice_override.to(device)
        if y_slice_override is not None
        else y_gt
    )
    u_gt = y_gt[:, 0]
    v_gt = y_gt[:, 1]
    if clot_phi_rollout_enabled():
        mu_for_kine = None
        if clot_phi_fixed_mu_from_phi_enabled():
            if rollout_state is not None and rollout_state.phi_prev is not None:
                y_k = data.y[time_index].to(device)
                mu_c_k = carreau_mu_si_from_uv(data, y_k[:, 0], y_k[:, 1], phys_cfg)
                mu_for_kine = mu_eff_from_carried_phi(
                    mu_c_k, rollout_state.phi_prev, device=device
                )
        elif rollout_state is not None and rollout_state.log_mu_prev is not None:
            mu_for_kine = torch.exp(rollout_state.log_mu_prev.clamp(max=20.0))
        u, v, _, _ = resolve_uv_for_rollout_step(
            data, time_index, mu_for_kine, device
        )
    elif u_nd_override is not None and v_nd_override is not None:
        u, v = u_nd_override, v_nd_override
    elif clot_phi_vel_source() == "kinematics":
        from src.core_physics.clot_temporal_growth_rules import _resolve_uv_for_temporal_risk

        u, v = _resolve_uv_for_temporal_risk(data, time_index, device)
    else:
        u, v = u_gt, v_gt
    mu_gt = phys_cfg.viscosity_nd_to_si(y_gt[:, STATE_CHANNEL_MU_EFF_ND])
    mu_cap = cap_mu_eff_si(mu_gt)
    mu_c = carreau_mu_si_from_uv(data, u, v, phys_cfg)
    use_soft = (os.environ.get("CLOT_PHI_SOFT_LABELS") or "0").strip().lower() in ("1", "true", "yes", "on")
    loss_mask = resolve_clot_loss_mask(
        data,
        device,
        mu_cap,
        phys_cfg,
        time_index=int(time_index),
        bio_cfg=bio_cfg,
    )
    scope = clot_phi_loss_scope()
    if scope in ("full_mesh", "support", "ceiling"):
        region = loss_mask
    else:
        region = supervision_region_mask(data, device, mu_cap, phys_cfg)
    # GT labels: growth above t=0 COMSOL mu (not baseline high-vis Carreau pockets).
    mu_anchor = gt_mu_anchor_cap_si(data, phys_cfg, device)
    phi_gt = resolve_phi_gt_labels(
        mu_cap, mu_c, None, phys_cfg, soft=use_soft, mu_anchor_si=mu_anchor
    )
    y_feat = y_gt.clone()
    if species_log_override is not None:
        y_feat[:, sc.SPECIES_BLOCK] = species_log_override.to(device=device, dtype=torch.float32)
    elif y_slice_override is not None:
        y_feat = y.to(device=device, dtype=torch.float32)
    feats = node_features_from_gt(
        data,
        y_feat,
        phys_cfg,
        bio_cfg,
        device=device,
        mu_cap_si=mu_cap,
        time_index=time_index,
        u_nd_override=u,
        v_nd_override=v,
    )
    if clot_phi_rollout_enabled() and rollout_state is not None:
        from src.core_physics.clot_phi_rollout import resolve_carry_log_mu_feature

        log_mu_carry = resolve_carry_log_mu_feature(
            time_index=int(time_index),
            train_epoch=train_epoch,
            gt_mu_cap_si=mu_cap,
            rollout_state=rollout_state,
            device=device,
        )
        feats = append_rollout_carry_features(
            feats,
            phi_prev=rollout_state.phi_prev,
            log_mu_prev=rollout_state.log_mu_prev,
            log_mu_override=log_mu_carry,
            n_nodes=int(data.num_nodes),
            device=device,
            dtype=feats.dtype,
        )
    species_log = y_feat[:, sc.SPECIES_BLOCK].to(device=device, dtype=torch.float32)
    return ClotPhiStepBatch(
        features=feats,
        phi_gt=phi_gt,
        mu_c_si=mu_c,
        mu_gt_cap=mu_cap,
        region=region,
        loss_mask=loss_mask,
        species_log_gt=species_log,
        u_flow_nd=u,
        v_flow_nd=v,
    )
