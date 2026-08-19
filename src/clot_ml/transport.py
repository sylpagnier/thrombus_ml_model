"""Advective transport of the wall-deposited species -- COMSOL's own operator, discretised.

WHY THIS EXISTS.  `docs/PHASE7_FINDINGS.md` 1.1 read the production `.mph` node tree and
found that `Mat` is **not** a surface coverage.  `tds2` is Transport of Diluted Species with

    D_M = D_Mas = D_Mat = 0            zero diffusion
    Reacting Flow, Diluted Species     convection ON, coupled to `spf`
    wall_surface_reactions_3spec       J0_Mat as an INWARD FLUX boundary condition

so the governing equation is

    dMat/dt + u . grad(Mat) = 0        with a wall flux source

i.e. a **pure hyperbolic transport problem with a boundary source and no diffusion**.

Everything the project currently uses off-wall is a surrogate for that: the 0.16
attenuation of PHASE7 3.2 (`Mat_off ~ 0.16 * Mat_owner`), the topological shell, the
`grow_into_lumen` speed rule, and the `owner` feature channels.  All of them are *nearest
wall node* rules -- they transport information along the mesh **normal**, which is the one
direction the physics does **not** transport along.  PHASE7 12.5 says the residual off-wall
error is the attenuation's *variance* (0.12-0.19 within a vessel against a 0.16 median) and
that the variance should be computable from the mesh.  This module computes it from the
flow instead, which is where the equation says it comes from.

WHAT IS SOLVED.  The steady form of the same operator,

    u . grad(C) = S                    C = 0 at the inflow

whose solution is the integral of the source along the **backward characteristic** through
each node -- exactly "how much wall flux has the fluid arriving here already picked up".
Discretised as vertex-centred first-order upwind finite volume, which is the same upwind
family COMSOL uses (its `tds2` carries Do Carmo & Galeao crosswind stabilisation).

    C_i * sum_j F_ij(out)  =  sum_j F_ji(in) * C_j  +  S_i * V_i

with `F_ij = max(0, ubar_ij . dhat_ij) * a_ij`.  Divergence-free flow makes in- and outflow
balance, so this is a diagonally-dominant M-matrix and Jacobi iteration converges.

THE STAGNATION TERM IS NOT A NUMERICAL FUDGE.  At the wall `u -> 0` (no-slip), so the
outflow sum vanishes and `C` is unbounded -- which is correct for a *steady* problem and
wrong for ours, because the real run has a finite horizon `T`.  Adding `V_i / T` to the
denominator caps the residence time at the horizon, so a fully stagnant node reads
`C = S * T`, the well-mixed accumulation over the run.  That single term is why the field
reproduces the boundary-layer attenuation without anybody writing 0.16 down: near-wall
parcels are slow, dwell long, and accumulate; a node one row further out is swept.

Also returned is the **residence time** `tau`, the solution of `u . grad(tau) = 1` under the
same discretisation -- how long the fluid arriving at a node has been in the vessel.  It is
the transport-side explanation of the low-shear gate: recirculation zones are exactly where
`tau` is large.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

__all__ = ["upwind_operator", "advect_source", "residence_time", "transport_fields"]


def upwind_operator(pos: np.ndarray, ei: np.ndarray, u: np.ndarray, v: np.ndarray,
                    ) -> tuple[sp.csr_matrix, np.ndarray]:
    """Directed upwind flux matrix ``F`` and the per-node outflow ``F.sum(0)``.

    ``F[j, i]`` is the volumetric flux carried from ``j`` into ``i``.  Only the downwind
    half of each undirected edge carries flux, which is what makes the scheme upwind.
    """
    src, dst = ei[0], ei[1]
    d = pos[dst] - pos[src]
    ln = np.linalg.norm(d, axis=1)
    keep = ln > 1e-12
    src, dst, d, ln = src[keep], dst[keep], d[keep], ln[keep]
    dhat = d / ln[:, None]

    # face-normal velocity, averaged over the edge (a two-point approximation of the flux
    # through the face separating the two control volumes)
    ubar = 0.5 * np.stack([u[src] + u[dst], v[src] + v[dst]], axis=1)
    un = (ubar * dhat).sum(axis=1)

    # face "area" in 2D is a length; the edge length is the only local scale available and
    # it is the standard choice for a vertex-centred median-dual mesh
    area = ln
    f_out = np.clip(un, 0.0, None) * area          # src -> dst
    f_in = np.clip(-un, 0.0, None) * area          # dst -> src

    n = len(u)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    vals = np.concatenate([f_out, f_in])
    F = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    F.eliminate_zeros()
    out = np.asarray(F.sum(axis=1)).reshape(-1)    # total flux leaving each node
    return F, out


def _solve_upwind(F: sp.csr_matrix, out: np.ndarray, rhs: np.ndarray, vol: np.ndarray,
                  horizon: float, iters: int = 600, tol: float = 1e-7) -> np.ndarray:
    """Solve ``C_i * (out_i + V_i/T) - sum_j F_ji C_j = rhs_i``.

    ``V_i / T`` is the finite-horizon cap described in the module docstring: without it a
    stagnation point has no outflow and the steady problem is singular there.

    Direct sparse solve, because Jacobi does **not** always converge here: strong
    recirculation makes the iteration matrix's spectral radius approach 1, and on
    ``patient035`` 400 sweeps still leave a 23% residual.  Jacobi is kept as a fallback for
    the (rare) singular factorisation.
    """
    Fin = F.T.tocsr()                              # Fin[i, j] = flux j -> i
    # Flux leaving through the DOMAIN BOUNDARY has no edge to travel along, so `out` misses
    # it and outlet nodes look like stagnation points that accumulate without limit.  For
    # divergence-free flow an interior node balances, so any excess of inflow over edge
    # outflow is exactly what crosses the boundary; charge it as additional outflow.
    # Without this the outlet row of every vessel reads spuriously high.
    inflow = np.asarray(Fin.sum(axis=1)).reshape(-1)
    out = out + np.clip(inflow - out, 0.0, None)
    den = np.maximum(out + vol / max(horizon, 1e-12), 1e-30)
    A = (sp.diags(den) - Fin).tocsc()
    try:
        C = spl.spsolve(A, rhs)
        if np.all(np.isfinite(C)):
            return C
    except Exception:                              # noqa: BLE001 - singular factorisation
        pass
    C = rhs / den
    for _ in range(iters):
        Cn = (Fin @ C + rhs) / den
        if np.max(np.abs(Cn - C)) <= tol * max(np.max(np.abs(Cn)), 1e-30):
            return Cn
        C = Cn
    return C


def _node_volume(pos: np.ndarray, ei: np.ndarray) -> np.ndarray:
    """Median-dual control volume proxy: (mean incident edge length)^2, in 2D."""
    n = len(pos)
    ln = np.linalg.norm(pos[ei[1]] - pos[ei[0]], axis=1)
    s = np.zeros(n)
    c = np.zeros(n)
    np.add.at(s, ei[0], ln)
    np.add.at(c, ei[0], 1.0)
    np.add.at(s, ei[1], ln)
    np.add.at(c, ei[1], 1.0)
    h = s / np.maximum(c, 1.0)
    return np.maximum(h, 1e-12) ** 2


def advect_source(pos, ei, u, v, source, *, horizon: float = 1.0, iters: int = 400
                  ) -> np.ndarray:
    """Solve ``u . grad(C) = source`` upwind; ``C`` is the source integrated upstream."""
    F, out = upwind_operator(pos, ei, u, v)
    vol = _node_volume(pos, ei)
    return _solve_upwind(F, out, np.asarray(source, float) * vol, vol, horizon, iters)


def residence_time(pos, ei, u, v, *, horizon: float = 1.0, iters: int = 400) -> np.ndarray:
    """Solve ``u . grad(tau) = 1``: age of the fluid arriving at each node."""
    return advect_source(pos, ei, u, v, np.ones(len(u)), horizon=horizon, iters=iters)


def transport_fields(pos, ei, u, v, wall, wall_source, *, horizon: float = 1.0,
                     iters: int = 400) -> dict:
    """The advective channels for one vessel.

    ``wall_source`` is the wall production rate per node (the backbone's own ``Mat``
    divided by the horizon is proportional to it, since the backbone is a pure
    accumulator of ``J0/h`` -- PHASE7 12.1).

    Returns
    -------
    mat_adv    the wall source transported downstream, the physics' own off-wall field
    tau        residence time
    mat_adv_n  ``mat_adv`` normalised by ``tau`` -- concentration rather than dose, which
               separates "sat in a slow region" from "swept past a strong source"
    src_reach  the same transport with a UNIT source on every gated wall node, i.e. the
               purely geometric question "does wall flux reach here at all"
    """
    F, out = upwind_operator(pos, ei, u, v)
    vol = _node_volume(pos, ei)
    ws = np.zeros(len(u))
    ws[wall] = np.asarray(wall_source, float)[wall]
    mat_adv = _solve_upwind(F, out, ws * vol, vol, horizon, iters)
    tau = _solve_upwind(F, out, np.ones(len(u)) * vol, vol, horizon, iters)
    reach = _solve_upwind(F, out, (ws > 0).astype(float) * vol, vol, horizon, iters)
    return dict(mat_adv=mat_adv, tau=tau,
                mat_adv_n=mat_adv / np.maximum(tau, 1e-30),
                src_reach=reach)
