"""
Graph-based velocity priors: width-adaptive Poiseuille scaling, Laplace stream function,
Kirchhoff outlet splitting, and Dean-type radial offset.

Used by ``mesh_to_graph.py`` (Tier 1/2) and ``mesh_to_graph_tier3.py`` (Tier 3).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix, identity
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve


# Legacy nominal half-width (non-dim) and Poiseuille peak scale used before width adaptation.
R_REF_ND = 0.5
U_MAX_BASE_ND = 1.5


def torch_sparse_coo_to_csr(L: torch.Tensor) -> csr_matrix:
    """Convert a coalesced torch sparse COO to ``scipy.sparse.csr_matrix`` (CPU, float64)."""
    L = L.coalesce()
    idx = L.indices().cpu().numpy()
    val = L.values().cpu().numpy().astype(np.float64)
    n = L.size(0)
    return coo_matrix((val, (idx[0], idx[1])), shape=(n, n)).tocsr()


def outlet_connected_components(mask_outlet: torch.Tensor, edge_index: torch.Tensor) -> List[np.ndarray]:
    """Connected components of the outlet boundary subgraph (edges with both ends on outlet)."""
    out_np = mask_outlet.numpy()
    oidx = np.where(out_np)[0]
    if len(oidx) == 0:
        return []
    pos = {int(i): k for k, i in enumerate(oidx)}
    row, col = edge_index.numpy()
    edges_o: List[Tuple[int, int]] = []
    for r, c in zip(row, col):
        if r == c or (not out_np[r]) or (not out_np[c]):
            continue
        a, b = pos[r], pos[c]
        if a != b:
            edges_o.append((a, b))
    nloc = len(oidx)
    if not edges_o:
        return [oidx]
    ei = np.array(edges_o, dtype=np.int64)
    data = np.ones(len(ei), dtype=np.float64)
    adj = coo_matrix((data, (ei[:, 0], ei[:, 1])), shape=(nloc, nloc))
    adj = adj + adj.T
    ncomp, labels = connected_components(adj, directed=False)
    comps: List[np.ndarray] = []
    for k in range(ncomp):
        comps.append(oidx[np.where(labels == k)[0]])
    return comps


def kirchhoff_outlet_psi_values(
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    edge_index: torch.Tensor,
    edge_length_nd: torch.Tensor,
    width_nd: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Poiseuille-style branch resistances R ~ L / D^4; conductance split gives outlet ψ in (0, 1].

    Returns a tensor ``psi_o`` of shape ``(N,)`` with nonzero entries only on ``mask_outlet``.
    """
    n = mask_inlet.numel()
    device = mask_inlet.device
    psi_o = torch.zeros(n, dtype=torch.float32, device=device)

    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    w = width_nd.view(-1).detach().cpu().numpy()
    le = edge_length_nd.view(-1).detach().cpu().numpy()
    d_edge = 0.5 * (w[row] + w[col])
    d_edge = np.maximum(d_edge, eps)
    weight = le / (d_edge**4)

    adj = coo_matrix((weight, (row, col)), shape=(n, n))
    inlet_idx = np.where(mask_inlet.numpy())[0]
    if len(inlet_idx) == 0:
        psi_o[mask_outlet] = 1.0
        return psi_o

    dist = dijkstra(adj, directed=False, indices=inlet_idx)
    dist = np.min(dist, axis=0)

    comps = outlet_connected_components(mask_outlet, edge_index)
    if not comps:
        return psi_o

    inv_r: List[float] = []
    for comp in comps:
        sub = dist[comp]
        finite = sub[np.isfinite(sub)]
        R_k = float(np.min(finite)) if finite.size else 1e6
        inv_r.append(1.0 / max(R_k, eps))

    s = sum(inv_r)
    if s <= 0:
        psi_o[mask_outlet] = 1.0
        return psi_o

    for comp, g in zip(comps, inv_r):
        psi_o[torch.tensor(comp, device=device, dtype=torch.long)] = float(g / s)

    return psi_o


def solve_laplace_stream_function(
    laplacian: torch.Tensor,
    psi_bc: torch.Tensor,
    fixed: torch.Tensor,
) -> torch.Tensor:
    """
    Solve ``L @ psi = 0`` on free nodes with Dirichlet data on ``fixed`` (inlet/outlet).

    ``psi_bc`` holds prescribed ψ; ``fixed[i]`` True means ``psi[i]`` is fixed.
    """
    n = psi_bc.numel()
    L = torch_sparse_coo_to_csr(laplacian)
    fix = fixed.cpu().numpy().astype(bool)
    free = ~fix
    psi = psi_bc.detach().cpu().numpy().astype(np.float64).copy()

    if free.sum() == 0:
        return torch.tensor(psi, dtype=torch.float32)

    L_ff = L[free][:, free]
    L_fd = L[free][:, fix]
    b = -L_fd @ psi[fix]
    nf = int(free.sum())
    L_ff = L_ff.tocsr() + 1e-10 * identity(nf, dtype=np.float64, format="csr")
    x = spsolve(L_ff, b)
    psi[free] = x
    return torch.tensor(psi, dtype=torch.float32)


def stream_function_to_velocity(
    G_x: torch.Tensor,
    G_y: torch.Tensor,
    psi: torch.Tensor,
    u_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discrete 2D stream function: ``u = u_scale * G_y @ psi``, ``v = -u_scale * G_x @ psi``.

    With WLS operators, ``G_y`` / ``G_x`` approximate ``∂/∂y`` and ``∂/∂x``.
    """
    psi_c = psi.unsqueeze(1)
    u_raw = torch.sparse.mm(G_y, psi_c).squeeze(1)
    v_raw = -torch.sparse.mm(G_x, psi_c).squeeze(1)
    return u_raw * u_scale, v_raw * u_scale


def scale_stream_velocity_to_umax(
    u: torch.Tensor,
    v: torch.Tensor,
    u_max_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Uniform scale so ``max(sqrt(u^2+v^2))`` matches ``max(u_max_target)`` (peak Poiseuille scale)."""
    mag = torch.sqrt(u**2 + v**2)
    peak = float(mag.max().clamp(min=1e-8).item())
    target = float(u_max_target.max().clamp(min=1e-8).item())
    s = target / peak
    return u * s, v * s, s


def width_nd_to_radius_nd(width_nd: torch.Tensor) -> torch.Tensor:
    """
    Hydraulic width from sphere tracing is the full lumen width along the inward normal ray.
    Poiseuille half-width (vessel radius) is ``R = width / 2``.
    """
    return (width_nd.view(-1) * 0.5).clamp(min=1e-5, max=2.0)


def mass_conserving_umax_nd(R_nd: torch.Tensor, u_max_base: float = U_MAX_BASE_ND, r_ref: float = R_REF_ND) -> torch.Tensor:
    """Scale peak velocity ~ 1/R^2 relative to reference radius ``r_ref`` (fixed global Re)."""
    return u_max_base * (r_ref / R_nd.clamp(min=1e-5)) ** 2


def dean_r_nd_effective(
    r_nd: torch.Tensor,
    R_nd: torch.Tensor,
    flow_dir_x: torch.Tensor,
    flow_dir_y: torch.Tensor,
    wall_normal_x: torch.Tensor,
    wall_normal_y: torch.Tensor,
    G_x: torch.Tensor,
    G_y: torch.Tensor,
    strength: float = 0.12,
) -> torch.Tensor:
    """
    Shift effective radial coordinate toward the outer bend using local streamline curvature
    and wall normal (Dean-type heuristic).
    """
    if strength <= 0:
        return r_nd

    fd_x = flow_dir_x.unsqueeze(1)
    fd_y = flow_dir_y.unsqueeze(1)
    curl_proxy = torch.sparse.mm(G_x, fd_y).squeeze(1) - torch.sparse.mm(G_y, fd_x).squeeze(1)
    kappa = torch.abs(curl_proxy)

    # Cross-stream axis (right-hand from flow): (-v, u); align with wall normal for "outer" side.
    cx = -flow_dir_y
    cy = flow_dir_x
    align = cx * wall_normal_x + cy * wall_normal_y
    shift = strength * R_nd * kappa * torch.sign(align)
    r_eff = r_nd - shift
    # clamp requires min/max both scalar or both tensor-broadcastable (not float + tensor).
    return torch.clamp(r_eff, min=torch.zeros_like(r_eff), max=R_nd * 1.5)
