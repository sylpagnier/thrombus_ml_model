"""Thrombin exposure as a screened-Poisson field -- physics for the ad-hoc growth term.

WHY.  The shipped model spreads clot with a fitted graph dilation: 6 hops along the wall,
admitting neighbours with ``sr < 2*lss``.  Two fitted scalars and no mechanism.  COMSOL's
actual spreading mechanism is chemical: a committed node generates thrombin
(``J0_th = beta*phi_at*Mat*PT``), thrombin activates platelets
(``Omega = APS/APScrit + APR/APRcrit + T/Tcrit``), and activated platelets adhere at
``k_as = 4.5e-4`` against ``k_rs = 3.7e-5`` -- **12x faster**.  So a committed node makes
its neighbours deposit faster, over the range thrombin can reach.

That range is not fitted, it is set by two constants already in ``BiochemConfig``:

    diffusion length  sqrt(2 * D_T * t_final) = sqrt(2 * 4.16e-11 * 30000) = 1.58 mm
    vessel scale      d_bar ~ 15 mm
    -> ~0.10 d_bar, i.e. 3-5 mesh hops on these packs

against a fitted ``grow_hops`` of 6, and 26.13.2's independently measured "every late
commit within 2 hops of existing clot".  The agreement is the reason to think the
dilation is a thrombin surrogate.

Bulk blood is wildly advection-dominated (Pe ~ 1e7), so thrombin would normally be washed
away -- but clot forms exactly where the low-shear gate is open, i.e. where the local
speed goes to zero and diffusion wins.  Hence a **screened** Poisson equation, with the
screening term being local washout:

    (lambda(x) - D grad^2) T = S(x)      lambda = |u|/d_bar,  S = committed Mat

Solved on the graph with one sparse factorisation reused across timesteps.  The absolute
scale of ``T`` folds into a single gain (the same 145x Damkohler ambiguity as 2.1), so the
field is normalised and only its SHAPE -- diffusion length against washout -- is used.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def graph_laplacian(pos: np.ndarray, edge_index: np.ndarray) -> sp.csr_matrix:
    """Sparse ``L`` approximating ``-grad^2`` in the pack's non-dimensional length unit.

    Edge weights ``1/h_ij^2`` with degree normalisation, so on a uniform 4-neighbour grid
    of spacing ``h`` this reduces to the standard 5-point stencil.
    """
    n = pos.shape[0]
    src, dst = edge_index[0], edge_index[1]
    h = np.linalg.norm(pos[src] - pos[dst], axis=1)
    w = 1.0 / np.maximum(h, 1e-12) ** 2
    W = sp.coo_matrix((w, (src, dst)), shape=(n, n))
    W = ((W + W.T) * 0.5).tocsr()
    cnt = np.asarray((W != 0).sum(axis=1)).reshape(-1).astype(np.float64)
    deg = np.asarray(W.sum(axis=1)).reshape(-1)
    scale = 4.0 / np.maximum(cnt, 1.0)          # normalise to a 4-neighbour equivalent
    return sp.diags(scale) @ (sp.diags(deg) - W)


def make_thrombin_solver(
    data, bio_cfg, pos: np.ndarray, shear_si: np.ndarray, *,
    wash_coef: float = 0.01, wall: np.ndarray | None = None, t_final: float | None = None,
):
    """Factorise ``(lambda + D L)`` once; return ``solve(source) -> normalised field``.

    ``D = D_T / d_bar^2`` [1/s on the nd-length Laplacian] and

        lambda = wash_coef * shear_si  +  1/t_final                     [1/s]

    **Why shear and not speed.** No-slip pins ``|u|`` to 0 at every wall node, so a
    speed-based washout collapses to its floor exactly where the coupling has to act, and
    the field goes global (measured: 3236 mesh hops, a uniform multiplier, i.e. a no-op).
    The local shear rate is already a rate in 1/s, is large where flow flushes the wall and
    small in stagnation, and is the field this project validated against COMSOL at
    spearman 0.998.

    **Why the 1/t_final floor.** In a stagnation pocket advective washout is essentially
    absent, so what limits thrombin's reach is simply how long the run lasts. With
    ``lambda -> 1/t_final`` the screened length is ``sqrt(D * t_final)``, which for
    ``D_T = 4.16e-11`` and a 30000 s horizon is 0.098 in nd units, about **3 mesh hops** --
    matching both the transient estimate ``sqrt(2*D_T*t) = 1.58 mm ~ 0.14 d_bar`` and
    26.13.2's independently measured "late commits within ~2 hops of existing clot".
    Without it the steady-state length is sub-mesh and the coupling is a no-op at the
    other extreme.

    Diagnostics report the range at the WALL, where the coupling acts, not the domain
    median, which the fast bulk dominates.
    """
    d_bar = float(data.d_bar.reshape(-1)[0])          # m
    D = float(bio_cfg.D_T) / (d_bar ** 2)             # 1/s
    if t_final is None:
        t_final = float(data.t.reshape(-1)[-1]) or float(bio_cfg.t_final)
    lam = float(wash_coef) * np.abs(shear_si) + 1.0 / max(t_final, 1e-9)
    L = graph_laplacian(pos, data.edge_index.detach().cpu().numpy())
    A = (sp.diags(lam) + D * L).tocsc()
    try:
        lu = spla.factorized(A)
    except Exception:                                  # pragma: no cover - singular fallback
        lu = None

    def solve(source: np.ndarray) -> np.ndarray:
        s = np.asarray(source, dtype=np.float64)
        if not np.any(s):
            return np.zeros_like(s)
        t = lu(s) if lu is not None else spla.spsolve(A, s)
        t = np.asarray(t).reshape(-1)
        t = np.clip(t, 0.0, None)
        m = t.max()
        return t / m if m > 0 else t

    sel = wall.astype(bool) if wall is not None else np.ones(len(lam), dtype=bool)
    lam_w = float(np.median(lam[sel])) if sel.any() else float(np.median(lam))
    edge = data.edge_index.detach().cpu().numpy()
    h = float(np.median(np.linalg.norm(pos[edge[0]] - pos[edge[1]], axis=1)))
    return solve, {
        "D_nd": D,
        "lam_median_wall": lam_w,
        "screened_length_wall_nd": float(np.sqrt(D / max(lam_w, 1e-30))),
        "mesh_edge_nd": h,
        # < 1 means the field cannot even reach the next node: the coupling is a no-op.
        "range_in_hops": float(np.sqrt(D / max(lam_w, 1e-30)) / max(h, 1e-12)),
    }


def make_ap_boost(solver, bio_cfg, *, gain: float = 4.0, every: int = 5, cap: float = 8.0):
    """``ap_boost(mat, step) -> multiplier`` from the thrombin field sourced by committed Mat.

    ``ap_eff = ap0 * (1 + gain * T_norm)``, a stand-in for COMSOL's activation chain
    ``T -> Omega -> k_pa -> AP``.  One scalar, replacing the two of the graph-dilation term.
    """
    crit = float(bio_cfg.viscosity_mat_crit)
    state = {"mult": None, "last": -10 ** 9}

    def ap_boost(mat, step):
        if state["mult"] is not None and (step - state["last"]) < every:
            return state["mult"]
        src = (mat >= crit).astype(np.float64)
        field = solver(src)
        state["mult"] = np.clip(1.0 + float(gain) * field, 1.0, cap)
        state["last"] = step
        return state["mult"]

    return ap_boost
