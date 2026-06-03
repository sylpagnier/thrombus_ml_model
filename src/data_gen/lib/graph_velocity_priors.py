"""2D planar mesh→graph velocity priors (width-adaptive Poiseuille scaling)."""

import torch

from src.config import VesselConfig

R_REF_ND = 0.5
U_MAX_BASE_ND = 1.5
# Cap Poiseuille peak ND; training graphs stay ~<=2.0. Uncapped 1/R blow-up in tight stenosis
# (patient anchors) reached ~8 ND and broke GINO-DEQ (pred spikes, misleading viz scales).
U_PRIOR_PEAK_CAP_ND = 2.0
EPSILON = 1e-5

# Calculate a safe physical floor based on the generator config.
# Theoretical min radius = 0.5 * 0.2 = 0.1.
# We multiply by 0.5 as a tolerance buffer for ray-marching underestimations.
_v_cfg = VesselConfig()
MIN_PHYSICAL_R_ND = (_v_cfg.nominal_radius * _v_cfg.min_radius_factor) * 0.5

# Jacobi diffusion on mesh edges before Poiseuille priors (reduces ray-march / KD-tree speckle).
WIDTH_PRIOR_SMOOTH_ALPHA = 0.45
WIDTH_PRIOR_SMOOTH_ITERS = 3


def smooth_width_nd_on_edges(
    width_nd: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    alpha: float = WIDTH_PRIOR_SMOOTH_ALPHA,
    iters: int = WIDTH_PRIOR_SMOOTH_ITERS,
) -> torch.Tensor:
    """
    Smooth a per-node scalar (hydraulic width) on the mesh graph.

    Each iteration applies ``w <- (1 - alpha) * w + alpha * mean(neighbors(w))`` using
    directed edges in ``edge_index`` (typically symmetric triangle edges). Nodes with no
    incident edges keep their value.
    """
    w = width_nd.view(num_nodes, 1).clone()
    row, col = edge_index[0], edge_index[1]
    device, dtype = w.device, w.dtype
    e = row.numel()
    ones_e = torch.ones(e, device=device, dtype=dtype)
    deg = torch.zeros(num_nodes, device=device, dtype=dtype).scatter_add_(0, row, ones_e)

    for _ in range(iters):
        neigh_sum = torch.zeros(num_nodes, 1, device=device, dtype=dtype)
        neigh_sum.scatter_add_(0, row.unsqueeze(1).expand(-1, 1), w[col])
        neigh_mean = torch.where(
            deg.unsqueeze(1) > 0.0,
            neigh_sum / deg.unsqueeze(1).clamp_min(1e-12),
            w,
        )
        w = (1.0 - alpha) * w + alpha * neigh_mean
    return w


def width_nd_to_radius_nd(width_nd: torch.Tensor) -> torch.Tensor:
    """
    Hydraulic width from sphere tracing is the full lumen width along the inward normal ray.
    Poiseuille half-width (vessel radius) is ``R = width / 2``.
    """
    return (width_nd.view(-1) * 0.5).clamp_min(MIN_PHYSICAL_R_ND).clamp_max(2.0)


def mass_conserving_umax_nd(R_nd: torch.Tensor, u_max_base: float = U_MAX_BASE_ND, r_ref: float = R_REF_ND) -> torch.Tensor:
    """Scale peak velocity ~ 1/R relative to reference radius ``r_ref`` (2D mass conservation)."""
    # We still use EPSILON here just in case, but MIN_PHYSICAL_R_ND above guarantees safety
    peak = u_max_base * (r_ref / R_nd.clamp_min(EPSILON))
    return peak.clamp(max=float(U_PRIOR_PEAK_CAP_ND))


__all__ = [
    "EPSILON",
    "MIN_PHYSICAL_R_ND",
    "R_REF_ND",
    "U_MAX_BASE_ND",
    "WIDTH_PRIOR_SMOOTH_ALPHA",
    "WIDTH_PRIOR_SMOOTH_ITERS",
    "U_PRIOR_PEAK_CAP_ND",
    "mass_conserving_umax_nd",
    "smooth_width_nd_on_edges",
    "width_nd_to_radius_nd",
]
