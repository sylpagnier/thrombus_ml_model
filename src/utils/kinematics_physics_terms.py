"""
Physics loss **terms** for Stage A predictor training (Tier 1 Newtonian / Tier 2 Carreau).

``train_t1_predictor.compute_step_loss`` and ``train_t2_predictor.compute_step_loss`` call this so
validation tests exercise the **exact** same code path as training (no duplicated derivative stacks).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.core_physics.physics_kernels import PhysicsKernels
from src.utils.anchor_mask import anchor_node_mask


def boundary_weighted_mse(
    pred_uvp: torch.Tensor,
    true_uvp: torch.Tensor,
    node_is_anchor: torch.Tensor,
    wall_mask: Optional[torch.Tensor] = None,
    wall_weight: float = 2.0,
) -> torch.Tensor:
    """Supervised kinematic loss on anchor nodes (Tier 1 training)."""
    if node_is_anchor is None or int(node_is_anchor.sum().item()) == 0:
        return pred_uvp.sum() * 0.0
    p = pred_uvp[node_is_anchor, :3]
    y = true_uvp[node_is_anchor, :3]
    e = (p - y) ** 2
    if wall_mask is None:
        return e.mean()
    wm = wall_mask[node_is_anchor].view(-1, 1).float()
    w = 1.0 + (max(float(wall_weight), 1.0) - 1.0) * wm
    return (e * w).mean()


def compute_kinematics_physics_terms(
    pred: torch.Tensor,
    data,
    kernels: PhysicsKernels,
    *,
    tier: str,
    boundary_data_weight: float = 2.0,
    carreau_n: Optional[float] = None,
    tier2_distillation: bool = False,
    tier2_kine_p_weight: float = 1.0,
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
            )
        else:
            pa = pred[node_is_anchor, :3]
            ya = data.y[node_is_anchor, :3]
            wp = float(tier2_kine_p_weight)
            if wp != 1.0:
                d = pa - ya
                w = d.new_tensor([1.0, 1.0, wp]).view(1, 3)
                l_data_kine = ((d * d) * w).mean()
            else:
                l_data_kine = F.mse_loss(pa, ya)
            if not tier2_distillation:
                l_data_mu = F.mse_loss(pred[node_is_anchor, 3], data.y[node_is_anchor, 3])

    l_bc = kernels.boundary_condition_loss(pred, data)
    l_io = kernels.inlet_outlet_loss(pred, data)

    if tier == "tier1":
        l_mom = kernels.navier_stokes_residual(pred, data, props=props)
        c_u = kernels._compute_derivatives(pred[:, 0:1], props)
        c_v = kernels._compute_derivatives(pred[:, 1:2], props)
        du_dx, du_dy = c_u[:, 0, 0], c_u[:, 1, 0]
        dv_dx, dv_dy = c_v[:, 0, 0], c_v[:, 1, 0]
        du_ij = torch.stack([du_dx, du_dy, dv_dx, dv_dy], dim=1)
        l_cont = kernels.continuity_loss(du_ij, data=data)
        l_rheo = z
    elif tier2_distillation:
        l_mom = z
        l_cont = z
        l_rheo = kernels.rheology_loss(pred, data, props=props, carreau_n=carreau_n)
    else:
        l_mom = kernels.navier_stokes_residual(pred, data, props=props)
        c_u = kernels._compute_derivatives(pred[:, 0:1], props)
        c_v = kernels._compute_derivatives(pred[:, 1:2], props)
        du_dx, du_dy = c_u[:, 0, 0], c_u[:, 1, 0]
        dv_dx, dv_dy = c_v[:, 0, 0], c_v[:, 1, 0]
        du_ij = torch.stack([du_dx, du_dy, dv_dx, dv_dy], dim=1)
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
    "compute_kinematics_physics_terms",
]
