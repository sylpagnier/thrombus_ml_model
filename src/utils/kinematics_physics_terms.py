"""
Physics loss **terms** for Stage A predictor training (Tier 1 Newtonian / Tier 2 Carreau).

``train_t1_predictor.compute_step_loss`` and ``train_t2_predictor.compute_step_loss`` call this so
validation tests exercise the **exact** same code path as training (no duplicated derivative stacks).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.config import PredChannels
from src.core_physics.physics_kernels import PhysicsKernels, scatter_add
from src.utils.anchor_mask import anchor_node_mask
from src.utils.rheology import compute_shear_rate


def _sdf_edge_gradient_proxy(data) -> torch.Tensor:
    """Mean |ΔSDF| over outgoing edges per node (geometry-variation proxy). Shape ``[N]``."""
    sdf = data.x[:, 2]
    row, col = data.edge_index
    n = int(data.num_nodes)
    diff = (sdf[row] - sdf[col]).abs()
    sum_d = scatter_add(diff, row, dim_size=n)
    cnt = scatter_add(torch.ones_like(diff), row, dim_size=n)
    return sum_d / (cnt + 1e-8)


def compute_anchor_kinematic_importance(
    data,
    node_is_anchor: torch.Tensor,
    *,
    mode: str = "uniform",
    sdf_wall_beta: float = 2.0,
    sdf_wall_tau: float = 0.12,
    sdf_grad_beta: float = 1.0,
    shear_true_alpha: float = 1.0,
    kernels: Optional[PhysicsKernels] = None,
    props=None,
) -> Optional[torch.Tensor]:
    """Return per-anchor-node positive weights, or ``None`` for uniform weighting."""
    mode = (mode or "uniform").strip().lower()
    if mode == "uniform" or node_is_anchor is None or int(node_is_anchor.sum().item()) == 0:
        return None
    if mode == "sdf_wall":
        sdf_a = data.x[node_is_anchor, 2].abs()
        return 1.0 + float(sdf_wall_beta) * torch.exp(-sdf_a / float(sdf_wall_tau))
    if mode == "sdf_grad":
        g = _sdf_edge_gradient_proxy(data)
        ga = g[node_is_anchor]
        med = torch.median(ga) if ga.numel() else ga.new_tensor(1.0)
        med = torch.clamp(med, min=1e-6)
        return 1.0 + float(sdf_grad_beta) * (ga / med)
    if mode == "shear_true":
        if kernels is None:
            return None
        # shear_true requires dense CFD labels across the graph. If this graph has sparse
        # anchor labeling, avoid derivative-based weighting to prevent boundary label leakage.
        if hasattr(data, "is_anchor") and torch.is_tensor(data.is_anchor):
            labeled = data.is_anchor.view(-1).bool()
            if int(labeled.sum().item()) < int(labeled.numel()):
                return None
        if props is None:
            props = kernels._get_geometric_props(data)
        # Ground-truth kinematic gradients from labels (u, v) to emphasize accelerated/sheared zones.
        c_u_true = kernels._compute_derivatives(data.y[:, PredChannels.U:PredChannels.U + 1], props)
        c_v_true = kernels._compute_derivatives(data.y[:, PredChannels.V:PredChannels.V + 1], props)
        u_x_true = c_u_true[:, 0, 0]
        u_y_true = c_u_true[:, 1, 0]
        v_x_true = c_v_true[:, 0, 0]
        v_y_true = c_v_true[:, 1, 0]
        gamma_dot_true = compute_shear_rate(u_x_true, u_y_true, v_x_true, v_y_true, eps=1e-6)
        gamma_anchor = gamma_dot_true[node_is_anchor]
        gamma_mean = torch.clamp(gamma_anchor.mean(), min=1e-6)
        return 1.0 + float(shear_true_alpha) * (gamma_anchor / gamma_mean)
    return None


def boundary_weighted_mse(
    pred_uvp: torch.Tensor,
    true_uvp: torch.Tensor,
    node_is_anchor: torch.Tensor,
    wall_mask: Optional[torch.Tensor] = None,
    wall_weight: float = 2.0,
    p_weight: float = 1.0,
    anchor_importance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Supervised kinematic loss on anchor nodes (Tier 1 training).

    ``anchor_importance`` — optional positive weights per anchor node (e.g. SDF-based explorer).
    """
    if node_is_anchor is None or int(node_is_anchor.sum().item()) == 0:
        return pred_uvp.sum() * 0.0
    p = pred_uvp[node_is_anchor, PredChannels.KINEMATICS]
    y = true_uvp[node_is_anchor, PredChannels.KINEMATICS]
    e = (p - y) ** 2
    wp = float(p_weight)
    channel_weights = e.new_tensor([1.0, 1.0, wp]).view(1, 3)
    active_channels = torch.clamp(channel_weights.sum(), min=1e-12)
    if wp != 1.0:
        e = e * channel_weights
    if anchor_importance is None:
        if wall_mask is None:
            return e.sum() / (e.shape[0] * active_channels + 1e-12)
        wm = wall_mask[node_is_anchor].view(-1, 1).float()
        w = 1.0 + (max(float(wall_weight), 1.0) - 1.0) * wm
        return (e * w).sum() / (w.sum() * active_channels + 1e-12)
    if wall_mask is None:
        w = torch.ones_like(e[:, :1])
    else:
        wm = wall_mask[node_is_anchor].view(-1, 1).float()
        w = 1.0 + (max(float(wall_weight), 1.0) - 1.0) * wm
    imp = anchor_importance.view(-1, 1).to(device=e.device, dtype=e.dtype)
    w = w * imp
    return (e * w).sum() / (w.sum() * active_channels + 1e-12)


def compute_kinematics_physics_terms(
    pred: torch.Tensor,
    data,
    kernels: PhysicsKernels,
    *,
    tier: str,
    boundary_data_weight: float = 2.0,
    carreau_n: Optional[float] = None,
    tier2_distillation: bool = False,
    tier1_kine_p_weight: float = 1.0,
    tier2_kine_p_weight: float = 1.0,
    anchor_kine_importance: Optional[torch.Tensor] = None,
    re_ref: Optional[float] = None,
    re_scale: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute every scalar physics term used in Tier 1 or Tier 2 training (except DEQ ``jac_loss``).

    Parameters
    ----------
    tier:
        ``"tier1"`` (Newtonian) or ``"tier2"`` (Carreau).
    tier2_distillation:
        If True (Tier 2 only), skip momentum / continuity (distillation phase) and still compute
        rheology + BC + IO + WSS + anchor kinematic loss.
    tier2_kine_p_weight:
        Tier 2 only: multiplies the pressure channel in anchor ``l_data_kine`` (u,v use weight 1).
        ``1.0`` recovers uniform MSE over ``(u,v,p)``.
    anchor_kine_importance:
        Tier 1 only: optional per-anchor-node weights for ``l_data_kine`` (see
        :func:`compute_anchor_kinematic_importance`).
    """
    if tier not in ("tier1", "tier2"):
        raise ValueError(f"tier must be 'tier1' or 'tier2', got {tier!r}")

    props = kernels._get_geometric_props(data)
    l_wss = kernels.wall_shear_stress_loss(pred, data, props=props)

    z = pred.sum() * 0.0
    node_is_anchor = anchor_node_mask(data)

    l_data_kine = z.clone()
    l_data_mu = z.clone()
    if node_is_anchor is not None and int(node_is_anchor.sum().item()) > 0:
        if tier == "tier1":
            wall_mask = getattr(data, "mask_wall", None)
            l_data_kine = boundary_weighted_mse(
                pred,
                data.y,
                node_is_anchor,
                wall_mask=wall_mask,
                wall_weight=boundary_data_weight,
                p_weight=tier1_kine_p_weight,
                anchor_importance=anchor_kine_importance,
            )
        else:
            pa = pred[node_is_anchor, PredChannels.KINEMATICS]
            ya = data.y[node_is_anchor, PredChannels.KINEMATICS]
            wp = float(tier2_kine_p_weight)
            if wp != 1.0:
                d = pa - ya
                w = d.new_tensor([1.0, 1.0, wp]).view(1, 3)
                l_data_kine = ((d * d) * w).mean()
            else:
                l_data_kine = F.mse_loss(pa, ya)
            if not tier2_distillation:
                l_data_mu = F.mse_loss(
                    pred[node_is_anchor, PredChannels.MU_EFF_ND],
                    data.y[node_is_anchor, PredChannels.MU_EFF_ND],
                )

    l_bc = kernels.boundary_condition_loss(pred, data)
    l_io = kernels.inlet_outlet_loss(pred, data)

    if tier == "tier1":
        l_mom = kernels.navier_stokes_residual(pred, data, props=props, re_ref=re_ref, re_scale=re_scale)
        c_u = kernels._compute_derivatives(pred[:, PredChannels.U:PredChannels.U + 1], props)
        c_v = kernels._compute_derivatives(pred[:, PredChannels.V:PredChannels.V + 1], props)
        du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)
        l_cont = kernels.continuity_loss(du_ij, data=data)
        l_rheo = z
    elif tier2_distillation:
        l_mom = z
        l_cont = z
        l_rheo = kernels.rheology_loss(pred, data, props=props, carreau_n=carreau_n)
    else:
        l_mom = kernels.navier_stokes_residual(pred, data, props=props, re_ref=re_ref, re_scale=re_scale)
        c_u = kernels._compute_derivatives(pred[:, PredChannels.U:PredChannels.U + 1], props)
        c_v = kernels._compute_derivatives(pred[:, PredChannels.V:PredChannels.V + 1], props)
        du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)
        l_cont = kernels.continuity_loss(du_ij, data=data)
        l_rheo = kernels.rheology_loss(pred, data, props=props, carreau_n=carreau_n)

    return {
        "l_wss": l_wss,
        "l_data_kine": l_data_kine,
        "l_data_mu": l_data_mu,
        "l_mom": l_mom,
        "l_cont": l_cont,
        "l_bc": l_bc,
        "l_io": l_io,
        "l_rheo": l_rheo,
    }


__all__ = [
    "boundary_weighted_mse",
    "compute_anchor_kinematic_importance",
    "compute_kinematics_physics_terms",
]
