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

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
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
        return data.x[:, 2].to(device=device, dtype=torch.float32).clamp(min=0.0)
    return torch.zeros(n, device=device, dtype=torch.float32)


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
    du_dx = torch.sparse.mm(data.G_x, u_col).squeeze(1)
    du_dy = torch.sparse.mm(data.G_y, u_col).squeeze(1)
    dv_dx = torch.sparse.mm(data.G_x, v.reshape(-1, 1)).squeeze(1)
    dv_dy = torch.sparse.mm(data.G_y, v.reshape(-1, 1)).squeeze(1)
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
        thr = clot_phi_thresh_si(phys_cfg)
        clot_seed = mu_cap_si.reshape(-1) >= thr
        region = neighbor_supervision_mask(data, device, clot_seed)
    if clot_phi_dgamma_slice_enabled():
        bio_cfg = BiochemConfig(phase="biochem")
        thr = clot_phi_thresh_si(phys_cfg)
        clot_seed = mu_cap_si.reshape(-1) >= thr
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
    du_dx = torch.sparse.mm(data.G_x, u)
    du_dy = torch.sparse.mm(data.G_y, u)
    dv_dx = torch.sparse.mm(data.G_x, v)
    dv_dy = torch.sparse.mm(data.G_y, v)
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


def species_log1p_nd_to_si(species_log1p: torch.Tensor, bio_cfg: BiochemConfig) -> torch.Tensor:
    """``y`` species channels (log1p ND) -> SI concentrations (12 bulk)."""
    scales = bio_cfg.get_species_scales(device=species_log1p.device)[:12].to(
        device=species_log1p.device, dtype=species_log1p.dtype
    )
    sp = species_log1p.reshape(-1, 12).clamp(min=-10.0, max=8.0)
    return torch.expm1(sp) * scales.view(1, -1)


def mu1_gelation_from_mat_si(mat_si: torch.Tensor, bio_cfg: BiochemConfig, mu_ratio_max: float) -> torch.Tensor:
    t_scale = max(float(bio_cfg.soft_step_T_scale), 1e-5)
    temp = max(float(bio_cfg.viscosity_gnode_temp_mat) * t_scale, 1e-8)
    crit = float(bio_cfg.viscosity_mat_crit)
    z = torch.clamp((mat_si.reshape(-1) - crit) / temp, min=-50.0, max=50.0)
    return (float(mu_ratio_max) - 1.0) * torch.sigmoid(z)


def mu2_gelation_from_fi_si(fi_si: torch.Tensor, bio_cfg: BiochemConfig, mu_ratio_max: float) -> torch.Tensor:
    t_scale = max(float(bio_cfg.soft_step_T_scale), 1e-5)
    temp = max(float(bio_cfg.viscosity_gnode_temp_fi) * t_scale, 1e-8)
    crit = float(bio_cfg.viscosity_fi_crit)
    z = torch.clamp((fi_si.reshape(-1) - crit) / temp, min=-50.0, max=50.0)
    return float(mu_ratio_max) * torch.sigmoid(z)


def clot_phi_physics_mu2_cap() -> float | None:
    raw = (os.environ.get("CLOT_PHI_PHYSICS_MU2_CAP") or "").strip()
    if not raw:
        return None
    return max(float(raw), 0.0)


def physics_mu_eff_si(
    mu_c_si: torch.Tensor,
    species_log1p: torch.Tensor,
    bio_cfg: BiochemConfig,
    *,
    device: torch.device,
    data=None,
    u_nd: torch.Tensor | None = None,
    v_nd: torch.Tensor | None = None,
) -> torch.Tensor:
    """``mu = mu_c * (1 + mu1(Mat) + mu2(FI))`` with optional prior gate / mu2 cap."""
    mu_ratio_max = clot_phi_physics_mu_ratio_max(bio_cfg)
    sp_si = species_log1p_nd_to_si(species_log1p.to(device=device), bio_cfg)
    fi_si = sp_si[:, 8]
    mat_si = sp_si[:, 11]
    mu1 = mu1_gelation_from_mat_si(mat_si, bio_cfg, mu_ratio_max)
    mu2 = mu2_gelation_from_fi_si(fi_si, bio_cfg, mu_ratio_max)
    cap = clot_phi_physics_mu2_cap()
    if cap is not None:
        mu2 = mu2.clamp(max=cap)
    gel = mu1 + mu2
    if (
        clot_phi_physics_gelation_gate_enabled()
        and data is not None
        and u_nd is not None
        and v_nd is not None
    ):
        props = {
            "u_ref": data.u_ref.view(-1)[:1].to(device=device, dtype=mu_c_si.dtype),
            "d_bar": data.d_bar.view(-1)[:1].to(device=device, dtype=mu_c_si.dtype),
        }
        gate = clot_prior_score_flat(
            data,
            u_nd.reshape(-1).to(device=device, dtype=mu_c_si.dtype),
            v_nd.reshape(-1).to(device=device, dtype=mu_c_si.dtype),
            bio_cfg,
            props,
        ).clamp(0.0, 1.0)
        gel = gel * gate
    return (mu_c_si.reshape(-1) * (1.0 + gel)).clamp(min=1e-8)


def physics_phi_from_mu(
    mu_phys_si: torch.Tensor,
    mu_c_si: torch.Tensor,
    region: torch.Tensor,
    phys_cfg: PhysicsConfig,
    *,
    soft: bool,
) -> torch.Tensor:
    mu_cap = cap_mu_eff_si(mu_phys_si)
    if soft:
        return phi_gt_soft(mu_cap, mu_c_si, region)
    return phi_gt_binary(mu_cap, region, phys_cfg)

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
) -> torch.Tensor:
    """Sanity floor: φ=1 where capped μ_gt ≥ threshold inside the supervision shell."""
    return phi_gt_binary(mu_cap_si, region, phys_cfg)


def phi_gt_binary(
    mu_cap_si: torch.Tensor,
    region: torch.Tensor,
    phys_cfg: PhysicsConfig,
) -> torch.Tensor:
    """Binary clot label (threshold on capped mu) inside the supervision region."""
    thr = clot_phi_thresh_si(phys_cfg)
    phi = (mu_cap_si.reshape(-1) >= thr).to(dtype=torch.float32)
    return phi * region.reshape(-1).to(dtype=torch.float32)


def phi_gt_soft(
    mu_cap_si: torch.Tensor,
    mu_c_si: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    """Soft phi* from log-blend inversion with fixed mu_solid (stable labels when mu > mu_c)."""
    solid = clot_phi_mu_solid_si()
    mc = mu_c_si.reshape(-1).clamp(min=1e-8)
    mu = mu_cap_si.reshape(-1).clamp(min=1e-8)
    denom = torch.log(torch.tensor(solid, device=mu.device, dtype=mu.dtype)) - torch.log(mc)
    denom = denom.clamp(min=1e-6)
    phi = ((torch.log(mu) - torch.log(mc)) / denom).clamp(0.0, 1.0)
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
    du_dx = torch.sparse.mm(data.G_x, u_col).squeeze(1)
    du_dy = torch.sparse.mm(data.G_y, u_col).squeeze(1)
    dv_dx = torch.sparse.mm(data.G_x, v.reshape(-1, 1)).squeeze(1)
    dv_dy = torch.sparse.mm(data.G_y, v.reshape(-1, 1)).squeeze(1)
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
    rollout_state: "ClotPhiRolloutState | None" = None,
) -> ClotPhiStepBatch:
    """One time slice: labels and features on wall-adjacent nodes."""
    from src.core_physics.clot_phi_rollout import (
        ClotPhiRolloutState,
        append_rollout_carry_features,
        clot_phi_rollout_enabled,
        resolve_uv_for_rollout_step,
    )

    y = (
        y_slice_override.to(device)
        if y_slice_override is not None
        else data.y[time_index].to(device)
    )
    u_gt = y[:, 0]
    v_gt = y[:, 1]
    if clot_phi_rollout_enabled():
        mu_for_kine = None
        if rollout_state is not None and rollout_state.log_mu_prev is not None:
            mu_for_kine = torch.exp(rollout_state.log_mu_prev.clamp(max=20.0))
        u, v, _, _ = resolve_uv_for_rollout_step(
            data, time_index, mu_for_kine, device
        )
    elif u_nd_override is not None and v_nd_override is not None:
        u, v = u_nd_override, v_nd_override
    else:
        u, v = u_gt, v_gt
    mu_gt = phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])
    mu_cap = cap_mu_eff_si(mu_gt)
    region = supervision_region_mask(data, device, mu_cap, phys_cfg)
    mu_c = carreau_mu_si_from_uv(data, u, v, phys_cfg)
    use_soft = (os.environ.get("CLOT_PHI_SOFT_LABELS") or "0").strip().lower() in ("1", "true", "yes", "on")
    if use_soft:
        phi_gt = phi_gt_soft(mu_cap, mu_c, region)
    else:
        phi_gt = phi_gt_binary(mu_cap, region, phys_cfg)
    feats = node_features_from_gt(
        data,
        y,
        phys_cfg,
        bio_cfg,
        device=device,
        mu_cap_si=mu_cap,
        time_index=time_index,
        u_nd_override=u,
        v_nd_override=v,
    )
    if clot_phi_rollout_enabled() and rollout_state is not None:
        feats = append_rollout_carry_features(
            feats,
            phi_prev=rollout_state.phi_prev,
            log_mu_prev=rollout_state.log_mu_prev,
            n_nodes=int(data.num_nodes),
            device=device,
            dtype=feats.dtype,
        )
    loss_mask = region.bool()
    species_log = y[:, 4:16].to(device=device, dtype=torch.float32)
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
