"""Shared kinematics / biochem node-feature builders for graph pipelines.

Anchor patient graphs store:
  - ``data.x`` (18ch, ``KINE_X_SCHEMA``) for Stage-A GINO-DEQ
  - ``data.x_biochem`` (15ch, ``BIO_X_SCHEMA``) for biochem encoder / physics BC layout
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.config import NodeFeat, PhysicsConfig
from src.data_gen.lib.graph_velocity_priors import (
    mass_conserving_umax_nd,
    smooth_width_nd_on_edges,
    width_nd_to_radius_nd,
)


def compute_hydraulic_width_nd(
    *,
    pos_nd: torch.Tensor,
    sdf_nd: torch.Tensor,
    wall_normal: torch.Tensor,
    d_bar_si: float,
    wall_tree: cKDTree,
    edge_index: torch.Tensor,
    G_x: Optional[torch.Tensor] = None,
    G_y: Optional[torch.Tensor] = None,
    flow_dir_x: Optional[torch.Tensor] = None,
    flow_dir_y: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sphere-tracing width ``D(x)`` (non-dimensional) plus optional flow-aligned derivatives."""
    n = int(pos_nd.shape[0])
    width_nd = torch.zeros(n, 1, dtype=torch.float32, device=pos_nd.device)
    t_march = sdf_nd.clone() + 0.05
    active = torch.ones(n, dtype=torch.bool, device=pos_nd.device)
    d_bar_f = float(d_bar_si)

    for _ in range(30):
        if not active.any():
            break
        idx = torch.nonzero(active, as_tuple=False).view(-1)
        if idx.numel() == 0:
            break
        probe = pos_nd[idx] + t_march[idx] * wall_normal[idx]
        probe_np = (probe * d_bar_f).detach().cpu().numpy()
        dist, _ = wall_tree.query(probe_np)
        dist_nd = torch.tensor(dist / d_bar_f, dtype=torch.float32, device=pos_nd.device).view(-1, 1)
        hit_mask = (dist_nd < 0.02).squeeze(-1)
        hit_idx = idx[hit_mask]
        width_nd[hit_idx] = sdf_nd[hit_idx] + t_march[hit_idx]
        active[hit_idx] = False
        still_idx = idx[~hit_mask]
        if still_idx.numel() == 0:
            break
        dist_still = dist_nd[~hit_mask]
        t_march[still_idx] = t_march[still_idx] + torch.clamp(dist_still, min=0.01)

    width_nd[width_nd.squeeze(-1) < 1e-6] = 1.0
    width_nd = smooth_width_nd_on_edges(width_nd, edge_index, n)
    width_nd = torch.clamp(width_nd, min=1e-6)

    width_d1 = torch.zeros(n, 1, dtype=torch.float32, device=pos_nd.device)
    width_d2 = torch.zeros(n, 1, dtype=torch.float32, device=pos_nd.device)
    if G_x is not None and G_y is not None and flow_dir_x is not None and flow_dir_y is not None:
        grad_w_x = torch.sparse.mm(G_x, width_nd)
        grad_w_y = torch.sparse.mm(G_y, width_nd)
        width_d1 = grad_w_x * flow_dir_x.unsqueeze(1) + grad_w_y * flow_dir_y.unsqueeze(1)
        grad2_w_x = torch.sparse.mm(G_x, width_d1)
        grad2_w_y = torch.sparse.mm(G_y, width_d1)
        width_d2 = grad2_w_x * flow_dir_x.unsqueeze(1) + grad2_w_y * flow_dir_y.unsqueeze(1)

    return width_nd, width_d1, width_d2


def flow_direction_from_wall_normals(
    wall_normal: torch.Tensor,
    pos_nd: torch.Tensor,
    *,
    centerline_pts_nd: Optional[np.ndarray] = None,
    centerline_tangents_nd: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unit flow tangent (non-dimensional) from wall normals, optionally oriented with centerline."""
    t_x = -wall_normal[:, 1]
    t_y = wall_normal[:, 0]

    if (
        centerline_pts_nd is not None
        and centerline_tangents_nd is not None
        and len(centerline_pts_nd) > 0
    ):
        spine_tree = cKDTree(np.asarray(centerline_pts_nd, dtype=np.float64))
        _, nearest = spine_tree.query(pos_nd.detach().cpu().numpy())
        tangents = np.asarray(centerline_tangents_nd, dtype=np.float64)[nearest]
        t_x = torch.tensor(tangents[:, 0], dtype=torch.float32, device=pos_nd.device)
        t_y = torch.tensor(tangents[:, 1], dtype=torch.float32, device=pos_nd.device)
        flow_norm = torch.sqrt(t_x ** 2 + t_y ** 2).clamp_min(1e-8)
        return t_x / flow_norm, t_y / flow_norm
    else:
        center_pt = pos_nd.mean(dim=0)
        inward = center_pt - pos_nd
        dot = t_x * inward[:, 0] + t_y * inward[:, 1]
        sign = torch.sign(dot)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        t_x = t_x * sign
        t_y = t_y * sign

    flow_norm = torch.sqrt(t_x ** 2 + t_y ** 2).clamp_min(1e-8)
    return t_x / flow_norm, t_y / flow_norm


def build_poiseuille_priors(
    *,
    pos_nd: torch.Tensor,
    sdf_nd: torch.Tensor,
    wall_normal: torch.Tensor,
    mask_wall: torch.Tensor,
    width_nd: torch.Tensor,
    flow_dir_x: torch.Tensor,
    flow_dir_y: torch.Tensor,
    d_bar_si: float,
    u_ref: float,
    phys_cfg: PhysicsConfig,
    mu_nd_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``u_prior``, ``v_prior``, ``mu_prior`` (1d), ``wss_prior`` (1d)."""
    ref_mu = float(mu_nd_scale if mu_nd_scale is not None else phys_cfg.mu_viscosity_nd_scale)
    r_nd = width_nd_to_radius_nd(width_nd).reshape(-1)
    u_max_nd = mass_conserving_umax_nd(r_nd).reshape(-1)
    sdf_1d = sdf_nd.reshape(-1).clamp_min(0.0)
    safe_sdf = torch.minimum(sdf_1d, r_nd)
    r_lane = (r_nd - safe_sdf).clamp_min(0.0)
    u_prior_mag = torch.clamp(
        u_max_nd * (1.0 - (r_lane ** 2 / (r_nd ** 2 + 1e-12))),
        min=0.0,
    )
    u_prior = u_prior_mag * flow_dir_x
    v_prior = u_prior_mag * flow_dir_y

    gamma_dot = torch.abs(-2.0 * u_max_nd * r_lane / (r_nd ** 2 + 1e-12))
    if phys_cfg.viscosity_model == "newtonian":
        mu_prior = torch.ones_like(u_prior)
    else:
        lambda_nd = phys_cfg.lam * (float(u_ref) / float(d_bar_si))
        mu_prior = (phys_cfg.mu_inf / ref_mu) + (
            (phys_cfg.mu_0 / ref_mu) - (phys_cfg.mu_inf / ref_mu)
        ) * (1.0 + (lambda_nd * gamma_dot) ** phys_cfg.a) ** ((phys_cfg.n - 1.0) / phys_cfg.a)
    wss_prior = (mu_prior * gamma_dot) * mask_wall.float()
    return u_prior, v_prior, mu_prior, wss_prior


def build_biochem_bc_x_tensor(
    *,
    pos_nd: torch.Tensor,
    sdf_nd: torch.Tensor,
    wall_normal: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    mask_wall: torch.Tensor,
    u_bc: torch.Tensor,
    v_bc: torch.Tensor,
    p_bc: torch.Tensor,
    mu_bc_nd: torch.Tensor,
) -> torch.Tensor:
    """15-channel biochem BC / mask layout (``BIO_X_SCHEMA``)."""
    m_in = mask_inlet.float().unsqueeze(1)
    m_out = mask_outlet.float().unsqueeze(1)
    m_wall = mask_wall.float().unsqueeze(1)
    uv_mask = (mask_inlet | mask_wall).float().unsqueeze(1)
    p_mask = mask_outlet.float().unsqueeze(1)
    mu_mask = torch.ones((pos_nd.shape[0], 1), dtype=torch.float32, device=pos_nd.device)
    return torch.cat(
        [
            pos_nd,
            sdf_nd,
            wall_normal,
            m_in,
            m_out,
            m_wall,
            u_bc,
            v_bc,
            p_bc,
            uv_mask,
            p_mask,
            mu_bc_nd.unsqueeze(1) if mu_bc_nd.dim() == 1 else mu_bc_nd,
            mu_mask,
        ],
        dim=1,
    )


def build_kinematics_node_x_tensor(
    *,
    pos_nd: torch.Tensor,
    sdf_nd: torch.Tensor,
    wall_normal: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    mask_wall: torch.Tensor,
    d_bar_si: float,
    u_ref: float,
    phys_cfg: PhysicsConfig,
    wall_tree: cKDTree,
    edge_index: torch.Tensor,
    G_x: Optional[torch.Tensor] = None,
    G_y: Optional[torch.Tensor] = None,
    centerline_pts_nd: Optional[np.ndarray] = None,
    centerline_tangents_nd: Optional[np.ndarray] = None,
    inlet_uv_nd: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    mu_nd_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """18-channel kinematics layout (``KINE_X_SCHEMA``) + ``u_prior``, ``mu_prior`` vectors."""
    flow_dir_x, flow_dir_y = flow_direction_from_wall_normals(
        wall_normal,
        pos_nd,
        centerline_pts_nd=centerline_pts_nd,
        centerline_tangents_nd=centerline_tangents_nd,
    )
    width_nd, width_d1, width_d2 = compute_hydraulic_width_nd(
        pos_nd=pos_nd,
        sdf_nd=sdf_nd,
        wall_normal=wall_normal,
        d_bar_si=d_bar_si,
        wall_tree=wall_tree,
        edge_index=edge_index,
        G_x=G_x,
        G_y=G_y,
        flow_dir_x=flow_dir_x,
        flow_dir_y=flow_dir_y,
    )
    u_prior, v_prior, mu_prior, wss_prior = build_poiseuille_priors(
        pos_nd=pos_nd,
        sdf_nd=sdf_nd,
        wall_normal=wall_normal,
        mask_wall=mask_wall,
        width_nd=width_nd,
        flow_dir_x=flow_dir_x,
        flow_dir_y=flow_dir_y,
        d_bar_si=d_bar_si,
        u_ref=u_ref,
        phys_cfg=phys_cfg,
        mu_nd_scale=mu_nd_scale,
    )
    shear_pot = torch.abs(1.0 - 2.0 * sdf_nd)
    rheo_flag = torch.full(
        (pos_nd.shape[0], 1),
        1.0 if phys_cfg.viscosity_model == "carreau" else 0.0,
        dtype=torch.float32,
        device=pos_nd.device,
    )
    x_kine = torch.cat(
        [
            pos_nd,
            sdf_nd,
            shear_pot,
            wall_normal,
            torch.zeros((pos_nd.shape[0], 4), dtype=torch.float32, device=pos_nd.device),
            rheo_flag,
            u_prior.unsqueeze(1),
            v_prior.unsqueeze(1),
            mu_prior.unsqueeze(1),
            wss_prior.unsqueeze(1),
            width_nd,
            width_d1,
            width_d2,
        ],
        dim=1,
    )
    if inlet_uv_nd is not None:
        u_in, v_in = inlet_uv_nd
        u_prior = u_prior.clone()
        v_prior = v_prior.clone()
        if hasattr(mask_inlet, "bool"):
            m_in = mask_inlet.view(-1).bool()
            u_prior[m_in] = u_in.reshape(-1)[m_in]
            v_prior[m_in] = v_in.reshape(-1)[m_in]

    if inlet_uv_nd is not None:
        x_kine = x_kine.clone()
        x_kine[:, NodeFeat.UV_PRIOR] = torch.cat(
            [u_prior.unsqueeze(1), v_prior.unsqueeze(1)],
            dim=1,
        )

    if x_kine.shape[1] != NodeFeat.WIDTH_D2.stop:
        raise ValueError(f"kinematics x width {x_kine.shape[1]} != {NodeFeat.WIDTH_D2.stop}")
    return x_kine, u_prior, mu_prior


def _resolve_graph_stem(data, stem: Optional[str] = None) -> str:
    if stem:
        return str(stem).strip()
    for attr in ("stem", "patient_id", "case_id", "name"):
        val = getattr(data, attr, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def kinematics_uv_prior_max(x_kine: torch.Tensor) -> float:
    if x_kine.shape[1] < NodeFeat.UV_PRIOR.stop:
        return 0.0
    return float(x_kine[:, NodeFeat.UV_PRIOR].abs().max().item())


def apply_gt_flow_priors_to_kine_x(
    x_kine: torch.Tensor,
    *,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    mu_nd: torch.Tensor,
    mask_wall: torch.Tensor,
    wall_normal: torch.Tensor,
    edge_index: torch.Tensor,
    M_inv: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """Overwrite ``uv_prior`` / ``mu_prior`` / ``wss_prior`` from COMSOL t=0 fields (anchor cheat)."""
    from src.data_gen.lib.kinematics_graph_builder import comsol_fields_to_kinematics_y

    x = x_kine.clone()
    u1 = u_nd.reshape(-1).clone()
    v1 = v_nd.reshape(-1).clone()
    u1 = u1.clone()
    v1 = v1.clone()
    u1[mask_wall.view(-1).bool()] = 0.0
    v1[mask_wall.view(-1).bool()] = 0.0
    x[:, NodeFeat.UV_PRIOR] = torch.stack([u1, v1], dim=1)
    x[:, NodeFeat.MU_PRIOR] = mu_nd.reshape(-1, 1)
    y5 = comsol_fields_to_kinematics_y(
        u_nd=u1,
        v_nd=v1,
        p_nd=torch.zeros_like(u1),
        mu_nd=mu_nd.reshape(-1),
        wall_normal_vec=wall_normal,
        mask_wall=mask_wall.view(-1).bool(),
        edge_index=edge_index,
        M_inv=M_inv,
        V=V,
        W=W,
        ref_mu=1.0,
    )
    x[:, NodeFeat.WSS_PRIOR] = y5[:, 4:5]
    x[:, 10:11] = 1.0
    return x


def resolve_anchor_kine_phys_cfg() -> PhysicsConfig:
    """Carreau Stage-A config for biochem COMSOL anchors (matches COMSOL rheology)."""
    return PhysicsConfig(phase="kinematics", rheology="carreau")


def patch_kinematics_priors_on_graph(
    data,
    *,
    phys_cfg: Optional[PhysicsConfig] = None,
    y_time_index: int = 0,
    use_inlet_bc: bool = True,
) -> bool:
    """Update only ``uv_prior`` / ``wss_prior`` channels in-place (keep width bands as stored)."""
    if not hasattr(data, "x") or int(data.x.shape[1]) < NodeFeat.WIDTH_D2.stop:
        return False
    device = data.x.device
    dtype = data.x.dtype
    pos_nd = data.x[:, NodeFeat.XY]
    sdf_nd = data.x[:, NodeFeat.SDF]
    wall_normal = data.x[:, NodeFeat.WALL_NORMAL]
    width_nd = data.x[:, NodeFeat.WIDTH_ND]
    mask_wall = data.mask_wall.to(device=device)
    mask_inlet = data.mask_inlet.to(device=device)

    d_bar_si = float(data.d_bar.reshape(-1)[0].item()) if hasattr(data, "d_bar") else 0.015
    u_ref = float(data.u_ref.reshape(-1)[0].item()) if hasattr(data, "u_ref") else 0.1
    if phys_cfg is None:
        phys_cfg = PhysicsConfig(phase="kinematics", rheology="newtonian")

    from src.data_gen.lib.centerline_utils import resolve_centerline_nd

    stem_resolved = _resolve_graph_stem(data)
    cl_pts, cl_tan, _cl_src = resolve_centerline_nd(
        pos_nd,
        mask_inlet,
        data.mask_outlet.to(device=device),
        edge_index=data.edge_index,
        mask_wall=mask_wall,
        stem=stem_resolved,
    )
    flow_x, flow_y = flow_direction_from_wall_normals(
        wall_normal,
        pos_nd,
        centerline_pts_nd=cl_pts,
        centerline_tangents_nd=cl_tan,
    )
    u_prior, v_prior, mu_prior, wss_prior = build_poiseuille_priors(
        pos_nd=pos_nd,
        sdf_nd=sdf_nd,
        wall_normal=wall_normal,
        mask_wall=mask_wall,
        width_nd=width_nd,
        flow_dir_x=flow_x,
        flow_dir_y=flow_y,
        d_bar_si=d_bar_si,
        u_ref=u_ref,
        phys_cfg=phys_cfg,
    )
    if use_inlet_bc and hasattr(data, "x_biochem") and int(data.x_biochem.shape[1]) >= 10:
        xb = data.x_biochem
        m_in = mask_inlet.view(-1).bool()
        u_prior = u_prior.clone()
        v_prior = v_prior.clone()
        u_prior[m_in] = xb[m_in, 8].reshape(-1)
        v_prior[m_in] = xb[m_in, 9].reshape(-1)

    data.x = data.x.clone()
    data.x[:, NodeFeat.UV_PRIOR] = torch.cat(
        [u_prior.unsqueeze(1), v_prior.unsqueeze(1)],
        dim=1,
    )
    data.x[:, NodeFeat.WSS_PRIOR] = wss_prior.unsqueeze(1)
    data.x[:, NodeFeat.MU_PRIOR] = mu_prior.unsqueeze(1)
    data.x[:, 10:11] = 1.0 if phys_cfg.viscosity_model == "carreau" else 0.0
    return True


def refresh_kinematics_node_x_on_graph(
    data,
    *,
    phys_cfg: Optional[PhysicsConfig] = None,
    stem: Optional[str] = None,
    raw_sidecar_dir: Optional[Path] = None,
    y_time_index: int = 0,
    force: bool = False,
    prior_floor: float = 1e-3,
    preserve_width: bool = True,
    width_nd_max_sane: float = 4.0,
) -> bool:
    """Rebuild ``data.x`` (18ch kine layout) with Poiseuille priors + patient FD inlet BCs.

    Returns True when ``data.x`` was updated. Skips when existing priors look healthy unless
    ``force=True``.
    """
    if not hasattr(data, "x") or data.x is None:
        return False
    if int(data.x.shape[1]) < NodeFeat.WIDTH_D2.stop:
        return False
    if not force and kinematics_uv_prior_max(data.x) > float(prior_floor):
        return False
    for req in ("mask_inlet", "mask_outlet", "mask_wall", "edge_index"):
        if not hasattr(data, req) or getattr(data, req) is None:
            return False

    device = data.x.device
    dtype = data.x.dtype
    pos_nd = data.x[:, NodeFeat.XY]
    sdf_nd = data.x[:, NodeFeat.SDF]
    wall_normal = data.x[:, NodeFeat.WALL_NORMAL]
    mask_inlet = data.mask_inlet.to(device=device)
    mask_outlet = data.mask_outlet.to(device=device)
    mask_wall = data.mask_wall.to(device=device)

    d_bar_si = 0.015
    u_ref = 0.1
    if hasattr(data, "d_bar") and data.d_bar is not None:
        d_bar_si = float(data.d_bar.reshape(-1)[0].item())
    if hasattr(data, "u_ref") and data.u_ref is not None:
        u_ref = float(data.u_ref.reshape(-1)[0].item())

    wall_coords = pos_nd[mask_wall].detach().cpu().numpy()
    if wall_coords.size == 0:
        wall_coords = pos_nd.detach().cpu().numpy()
    wall_tree = cKDTree(wall_coords)

    if phys_cfg is None:
        phys_cfg = PhysicsConfig(phase="kinematics", rheology="newtonian")

    stem_resolved = _resolve_graph_stem(data, stem=stem)
    from src.data_gen.lib.centerline_utils import resolve_centerline_nd

    cl_pts, cl_tan, cl_src = resolve_centerline_nd(
        pos_nd,
        mask_inlet,
        mask_outlet,
        edge_index=data.edge_index,
        mask_wall=mask_wall,
        stem=stem_resolved,
        raw_sidecar_dir=raw_sidecar_dir,
    )

    inlet_uv = None
    if hasattr(data, "x_biochem") and data.x_biochem is not None and int(data.x_biochem.shape[1]) >= 10:
        xb = data.x_biochem
        inlet_uv = (xb[:, 8].reshape(-1), xb[:, 9].reshape(-1))

    width_max = float(data.x[:, NodeFeat.WIDTH_ND].max().item())
    if (
        preserve_width
        and width_max > 1e-3
        and width_max <= float(width_nd_max_sane)
    ):
        return patch_kinematics_priors_on_graph(
            data,
            phys_cfg=phys_cfg,
            y_time_index=y_time_index,
            use_inlet_bc=True,
        )

    G_x = getattr(data, "G_x", None)
    G_y = getattr(data, "G_y", None)

    x_kine, _, _ = build_kinematics_node_x_tensor(
        pos_nd=pos_nd,
        sdf_nd=sdf_nd,
        wall_normal=wall_normal,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        mask_wall=mask_wall,
        d_bar_si=d_bar_si,
        u_ref=u_ref,
        phys_cfg=phys_cfg,
        wall_tree=wall_tree,
        edge_index=data.edge_index,
        G_x=G_x,
        G_y=G_y,
        centerline_pts_nd=cl_pts,
        centerline_tangents_nd=cl_tan,
        inlet_uv_nd=inlet_uv,
        mu_nd_scale=phys_cfg.mu_viscosity_nd_scale,
    )
    data.x = x_kine.to(device=device, dtype=dtype)
    if hasattr(data, "centerline_source"):
        data.centerline_source = cl_src
    return True
