"""Discrete boundary volume-flux helpers for 2D planar graphs (ND units).

Used to compare predicted inlet flow against the fully-developed (FD) inlet prescription
(Re = ``PhysicsConfig.re_target``, COMSOL ``Uav`` = ``get_u_ref(d_bar)``) and to flag
collapsed / trivial velocity fields during biochem training.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from src.config import BiochemNodeFeat, PhysicsConfig
from src.utils.channel_schema import biochem_encoder_x
from src.utils.metrics import poiseuille_2d_planar_volume_flux_nd


def _as_scalar_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _velocity_xy_from_bc_or_vel(
    velocity: torch.Tensor,
    u_bc: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return [N, 2] velocity; prefer boundary BC tensor on ``mask`` when provided."""
    vel = velocity.view(-1, 2).float()
    if u_bc is None:
        return vel
    bc = u_bc.float()
    out = vel.clone()
    m = mask.view(-1).bool()
    if bc.dim() == 1:
        out[m, 0] = bc[m]
        return out
    if bc.shape[1] == 1:
        out[m, 0] = bc[m, 0]
        return out
    out[m, 0] = bc[m, 0]
    out[m, 1] = bc[m, 1]
    return out


def _domain_centroid(pos: torch.Tensor) -> torch.Tensor:
    return pos.float().mean(dim=0)


def _inward_face_normals_and_lengths(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Undirected boundary faces: row/col node ids, unit inward normal, segment length."""
    pos = pos.float()
    row, col = edge_index[0], edge_index[1]
    b_faces = boundary_mask.view(-1)[row] & boundary_mask.view(-1)[col] & (row < col)
    r = row[b_faces]
    c = col[b_faces]
    if r.numel() == 0:
        z = pos.new_zeros(0)
        return r, c, z, z

    edge_vecs = pos[c] - pos[r]
    lengths = torch.linalg.norm(edge_vecs, dim=1).clamp_min(1e-12)
    tangents = edge_vecs / lengths.unsqueeze(1)
    # Left-hand normal to edge direction; flip so it points into the domain.
    normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=1)
    center = _domain_centroid(pos)
    mid = 0.5 * (pos[r] + pos[c])
    inward = center - mid
    flip = (normals * inward).sum(dim=1, keepdim=True) < 0
    normals = torch.where(flip, -normals, normals)
    return r, c, normals, lengths


def discrete_boundary_volume_flux_nd(
    velocity: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> float:
    """
    Trapezoidal ∫ (u·n_inward) ds on a 1D boundary selection (2D planar, ND units).

    ``velocity`` is [N, 2]. Returns 0.0 when no boundary faces exist.
    """
    vel = velocity.view(-1, 2).float()
    r, c, normals, lengths = _inward_face_normals_and_lengths(pos, edge_index, boundary_mask)
    if r.numel() == 0:
        return 0.0
    un_r = (vel[r] * normals).sum(dim=1)
    un_c = (vel[c] * normals).sum(dim=1)
    flux = 0.5 * (un_r + un_c) * lengths
    return float(flux.sum().item())


def inlet_effective_width_nd(
    pos: torch.Tensor,
    mask_inlet: torch.Tensor,
    flow_hint: Optional[torch.Tensor] = None,
) -> float:
    """
    Geometric span of inlet nodes perpendicular to mean flow (ND).

    Used with COMSOL ``Uav`` (= ``u_ref``) for ``Q_ref ≈ u_ref * width``.
    """
    m = mask_inlet.view(-1).bool()
    if not bool(m.any().item()):
        return float("nan")
    pts = pos.float()[m]
    if flow_hint is not None and flow_hint.numel() >= 2:
        fh = flow_hint.float().reshape(-1, 2)
        n_in = int(m.sum().item())
        # ``flow_hint`` may already be inlet-only (e.g. ``bc_vel[m_in]``) or full-node.
        if fh.shape[0] == n_in:
            fd = fh.mean(dim=0)
        elif fh.shape[0] == m.shape[0]:
            fd = fh[m].mean(dim=0)
        else:
            fd = fh.mean(dim=0)
    else:
        fd = pts[-1] - pts[0]
    mag = torch.linalg.norm(fd)
    if mag < 1e-9:
        fd = torch.tensor([1.0, 0.0], device=pos.device, dtype=pos.dtype)
    else:
        fd = fd / mag
    perp = torch.stack([-fd[1], fd[0]])
    proj = (pts * perp.unsqueeze(0)).sum(dim=1)
    return float((proj.max() - proj.min()).clamp_min(1e-8).item())


def fd_inlet_flux_ref_from_re_nd(
    *,
    u_ref_nd: float,
    width_nd: float,
) -> float:
    """Volume flux for FD parabolic inlet with COMSOL ``Uav`` = ``u_ref`` (2D planar)."""
    u_ref = max(float(u_ref_nd), 0.0)
    w = max(float(width_nd), 1e-8)
    return u_ref * w


def fd_inlet_flux_ref_from_poiseuille_nd(
    *,
    u_ref_nd: float,
    width_nd: float,
) -> float:
    """
    Same FD flow when ``u_ref`` is cross-sectional mean velocity: ``Q = U_av * width``.

    Equivalent to ``poiseuille_2d_planar_volume_flux_nd(u_max, width)`` with ``u_max = 1.5 * u_ref``.
    """
    u_ref = max(float(u_ref_nd), 0.0)
    w = max(float(width_nd), 1e-8)
    u_max = 1.5 * u_ref
    q = poiseuille_2d_planar_volume_flux_nd(
        torch.tensor(u_max, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
    )
    return float(q.item())


def compute_inlet_outlet_flux_debug(
    *,
    velocity: torch.Tensor,
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    u_inlet_bc: Optional[torch.Tensor] = None,
    u_ref_nd: Optional[float] = None,
    phys_cfg: Optional[PhysicsConfig] = None,
) -> Dict[str, float]:
    """
    Inlet/outlet volume flux vs FD reference (ND).

    Returns NaN fields when masks or geometry are missing. Primary reference is the
    edge-integrated prescribed inlet BC (``u_inlet_bc``); analytic Re prescription uses
    ``u_ref`` × effective inlet width (COMSOL ``Uav`` at Re = ``re_target``).
    """
    nan = float("nan")
    out: Dict[str, float] = {
        "Q_pred_inlet_nd": nan,
        "Q_pred_outlet_nd": nan,
        "Q_ref_bc_nd": nan,
        "Q_ref_re_nd": nan,
        "Q_inlet_rel_err": nan,
        "Q_flow_ratio": nan,
        "Q_inlet_outlet_imbalance": nan,
        "Q_ref_bc_re_mismatch": nan,
        "inlet_width_nd": nan,
        "speed_inlet_mean": nan,
        "speed_outlet_mean": nan,
        "flow_collapse": nan,
        "flow_trivial_score": nan,
    }

    m_in = mask_inlet.view(-1).bool()
    m_out = mask_outlet.view(-1).bool()
    if not bool(m_in.any().item()) or not bool(m_out.any().item()):
        return out

    vel = velocity.view(-1, 2).float()
    pos = pos.float()
    speed = torch.linalg.norm(vel, dim=1)
    out["speed_inlet_mean"] = float(speed[m_in].mean().item())
    out["speed_outlet_mean"] = float(speed[m_out].mean().item())

    q_pred_in = discrete_boundary_volume_flux_nd(vel, pos, edge_index, m_in)
    q_pred_out = discrete_boundary_volume_flux_nd(vel, pos, edge_index, m_out)
    out["Q_pred_inlet_nd"] = q_pred_in
    out["Q_pred_outlet_nd"] = q_pred_out

    bc_vel = _velocity_xy_from_bc_or_vel(vel, u_inlet_bc, m_in)
    q_ref_bc = discrete_boundary_volume_flux_nd(bc_vel, pos, edge_index, m_in)
    out["Q_ref_bc_nd"] = q_ref_bc

    flow_hint = bc_vel[m_in] if bool(m_in.any().item()) else None
    width_nd = inlet_effective_width_nd(pos, m_in, flow_hint=flow_hint)
    out["inlet_width_nd"] = width_nd

    if u_ref_nd is not None and math.isfinite(width_nd):
        q_ref_re = fd_inlet_flux_ref_from_re_nd(u_ref_nd=u_ref_nd, width_nd=width_nd)
        out["Q_ref_re_nd"] = q_ref_re
    else:
        q_ref_re = nan

    eps = 1e-8
    # Primary reference: analytic Re-target Uav (``u_ref`` × inlet width). Fallback: BC-integrated flux.
    if math.isfinite(q_ref_re) and abs(q_ref_re) > eps:
        q_ref_primary = q_ref_re
    elif math.isfinite(q_ref_bc) and abs(q_ref_bc) > eps:
        q_ref_primary = q_ref_bc
    else:
        q_ref_primary = nan

    if math.isfinite(q_ref_bc) and math.isfinite(q_ref_re):
        out["Q_ref_bc_re_mismatch"] = abs(q_ref_bc - q_ref_re) / (abs(q_ref_bc) + eps)

    if math.isfinite(q_pred_in) and math.isfinite(q_ref_primary) and abs(q_ref_primary) > eps:
        out["Q_inlet_rel_err"] = abs(q_pred_in - q_ref_primary) / abs(q_ref_primary)
        out["Q_flow_ratio"] = q_pred_in / q_ref_primary
        trivial = 1.0 - max(0.0, min(1.0, q_pred_in / q_ref_primary))
        out["flow_trivial_score"] = trivial

    if math.isfinite(q_pred_in) and math.isfinite(q_pred_out):
        out["Q_inlet_outlet_imbalance"] = abs(q_pred_in - abs(q_pred_out)) / (abs(q_pred_in) + eps)

    sp_in = out["speed_inlet_mean"]
    sp_out = out["speed_outlet_mean"]
    if math.isfinite(sp_in) and math.isfinite(sp_out):
        flow_ref = max(sp_in, sp_out, eps)
        out["flow_collapse"] = 1.0 - ((sp_in + sp_out) / (2.0 * flow_ref + eps))

    return out


def flux_debug_to_training_metrics(flux: Dict[str, float]) -> Dict[str, float]:
    """Map ``compute_inlet_outlet_flux_debug`` keys to ``DBG_*`` biochem training metrics."""
    nan = float("nan")
    return {
        "DBG_Q_pred_inlet": flux.get("Q_pred_inlet_nd", nan),
        "DBG_Q_pred_outlet": flux.get("Q_pred_outlet_nd", nan),
        "DBG_Q_ref_bc": flux.get("Q_ref_bc_nd", nan),
        "DBG_Q_ref_re": flux.get("Q_ref_re_nd", nan),
        "DBG_Q_inlet_rel_err": flux.get("Q_inlet_rel_err", nan),
        "DBG_Q_flow_ratio": flux.get("Q_flow_ratio", nan),
        "DBG_flow_trivial_score": flux.get("flow_trivial_score", nan),
        "DBG_Q_inlet_outlet_imbalance": flux.get("Q_inlet_outlet_imbalance", nan),
        "DBG_inlet_width_nd": flux.get("inlet_width_nd", nan),
        "DBG_Q_ref_bc_re_mismatch": flux.get("Q_ref_bc_re_mismatch", nan),
        # Legacy names: mean speed on boundary (not volume flux).
        "DBG_flux_inlet": flux.get("speed_inlet_mean", nan),
        "DBG_flux_outlet": flux.get("speed_outlet_mean", nan),
        "DBG_flux_imbalance": flux.get("Q_inlet_outlet_imbalance", nan),
        "DBG_flow_collapse": flux.get("flow_collapse", nan),
    }


def flux_debug_from_graph_data(
    data,
    velocity: torch.Tensor,
    *,
    phys_cfg: Optional[PhysicsConfig] = None,
) -> Dict[str, float]:
    """Convenience wrapper using PyG ``Data`` fields (``x``, masks, ``u_inlet_bc``, ``u_ref``)."""
    if not (
        hasattr(data, "mask_inlet")
        and hasattr(data, "mask_outlet")
        and hasattr(data, "edge_index")
        and data.mask_inlet is not None
        and data.mask_outlet is not None
        and data.edge_index is not None
    ):
        return compute_inlet_outlet_flux_debug(
            velocity=velocity,
            pos=velocity.new_zeros(0, 2),
            edge_index=velocity.new_zeros(2, 0, dtype=torch.long),
            mask_inlet=torch.zeros(0, dtype=torch.bool),
            mask_outlet=torch.zeros(0, dtype=torch.bool),
        )

    pos = biochem_encoder_x(data)[:, BiochemNodeFeat.XY].to(device=velocity.device)
    u_ref_nd = None
    if hasattr(data, "u_ref") and data.u_ref is not None:
        u_ref_nd = _as_scalar_float(data.u_ref)
    elif phys_cfg is not None and hasattr(data, "d_bar") and data.d_bar is not None:
        u_ref_nd = phys_cfg.get_u_ref(_as_scalar_float(data.d_bar))

    u_inlet_bc = getattr(data, "u_inlet_bc", None)

    return compute_inlet_outlet_flux_debug(
        velocity=velocity,
        pos=pos,
        edge_index=data.edge_index.to(device=velocity.device),
        mask_inlet=data.mask_inlet.to(device=velocity.device),
        mask_outlet=data.mask_outlet.to(device=velocity.device),
        u_inlet_bc=u_inlet_bc.to(device=velocity.device) if u_inlet_bc is not None else None,
        u_ref_nd=u_ref_nd,
        phys_cfg=phys_cfg,
    )
