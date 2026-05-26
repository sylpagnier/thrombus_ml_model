"""Shared kinematics / biochem node-feature builders for graph pipelines.

Anchor patient graphs store:
  - ``data.x`` (18ch, ``KINE_X_SCHEMA``) for Stage-A GINO-DEQ
  - ``data.x_biochem`` (15ch, ``BIO_X_SCHEMA``) for biochem encoder / physics BC layout
"""

from __future__ import annotations

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
        local_flow = torch.tensor(
            np.asarray(centerline_tangents_nd, dtype=np.float64)[nearest],
            dtype=torch.float32,
            device=pos_nd.device,
        )
        dot = t_x * local_flow[:, 0] + t_y * local_flow[:, 1]
        sign = torch.sign(dot)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        t_x = t_x * sign
        t_y = t_y * sign
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
    if x_kine.shape[1] != NodeFeat.WIDTH_D2.stop:
        raise ValueError(f"kinematics x width {x_kine.shape[1]} != {NodeFeat.WIDTH_D2.stop}")
    return x_kine, u_prior, mu_prior
