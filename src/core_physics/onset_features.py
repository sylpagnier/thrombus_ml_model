"""Deploy-legal per-node features for predicting ONSET TIME directly.

WHY DIRECT ONSET, AND NOT A FIELD FED THROUGH THE ODE.  ``scripts/diag_lever_panel.py``
measured every physical lever on the metric of record.  Every field ORACLE -- perfect wall
AP, perfect time-varying flow, both together -- scored **below** the frozen-``ap`` baseline
(-0.007 / -0.009 / -0.034), while the onset oracle scored +0.099.  The information is not
missing from the inputs; the readout destroys it:

    gate*ap_early ranks true onset at |rho| = 0.877  as a raw FEATURE
    the same field pushed through the ODE yields rho = 0.649  as an ONSET

So the stiff ODE plus its first-crossing threshold is the lossy step.  These features exist
to be regressed straight onto onset time, with the physics keeping the parts it is good at:
the committed SET (gate + graph growth, already at the flow-oracle ceiling on the mask) and
the structure of the features themselves.

EVERYTHING HERE IS DEPLOY-LEGAL: geometry, mesh connectivity, and the t=0 velocity field
(the Phase-3 bandaid, or ``u0_pred`` on arm B).  No GT species, no GT flow evolution, no
``data.x`` prior channels -- see PHASE6_HANDOFF 7 for why that last one is a trap.

The hop-distance features carry the one lever that actually worked: the shipped mask grows
``GROW`` hops out from the igniting seeds, but every grown node currently inherits the
ODE's *median* onset -- a constant, on 15-26% of the mask.  ``hop`` is that geometry made
explicit so a model can time the front instead of flattening it.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

M_TO_CM = 100.0
RELAX, GROW = 2.0, 6

#: order is fixed: it is the column order of :func:`build_features` and of every fitted
#: coefficient vector written alongside a model.
FEATURE_NAMES = (
    "log_sr", "sr_rank", "log_absdsrx", "dsrx_rank",
    "gate", "log_gate", "gate_low", "gate_sep",
    "hop", "hop_frac", "is_seed",
    "nbr1_log_sr", "nbr2_log_sr", "nbr3_log_sr", "nbr1_gate", "nbr2_gate",
    "ap_closure", "log_ap_closure",
    "seed_dist_xy", "arc_frac", "degree",
)


def _adj(edges: np.ndarray, n: int) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def _rank01(v: np.ndarray) -> np.ndarray:
    """Within-vessel rank in [0, 1].  Vessels differ in absolute shear by ~20x, so raw
    magnitudes do not transfer; the ORDER does, and order is what onset needs."""
    if len(v) < 2:
        return np.zeros_like(v, dtype=np.float64)
    r = np.argsort(np.argsort(v)).astype(np.float64)
    return r / (len(v) - 1)


def _khop_mean(A: sp.csr_matrix, v: np.ndarray, k: int) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64)
    P = A.astype(np.float64)
    deg = np.asarray(P.sum(1)).reshape(-1)
    deg[deg == 0] = 1.0
    for _ in range(k):
        out = np.asarray(P @ out).reshape(-1) / deg
    return out


def committed_set(gate: np.ndarray, sr0: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The shipped mask, on the wall subgraph.

    ``scripts/predict_wall_clot.py`` grows on the full graph but admits only wall nodes, and
    a non-wall node can never be admitted -- so the growth is exactly the wall subgraph's,
    and this reproduces ``S`` from the cache without touching a 300 MB pack.
    """
    n = len(gate)
    A = _adj(edges, n)
    cur = gate > 0
    adm = sr0 < 25.0 * RELAX          # bio.lss * RELAX
    for _ in range(GROW):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur


def hop_distance(seed: np.ndarray, edges: np.ndarray, max_hops: int = GROW) -> np.ndarray:
    """Mesh hops from the nearest igniting seed; ``max_hops + 1`` if unreachable."""
    n = len(seed)
    A = _adj(edges, n)
    d = np.where(seed, 0, max_hops + 1).astype(np.float64)
    cur = seed.copy()
    for h in range(1, max_hops + 1):
        nxt = ((A @ cur.astype(np.int8)) > 0) & ~cur
        d[nxt & (d > h)] = h
        cur = cur | nxt
    return d


def build_features(z, bio, *, C: float, q: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(X [Nw, F], S [Nw])`` from one cached vessel.  Column order is ``FEATURE_NAMES``."""
    sr0, dsrx0, pos, edges = z["sr0"], z["dsrx0"], z["pos"], z["wall_edges"]
    n = len(sr0)
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    k_as = float(bio.k_as) * M_TO_CM

    gate_low = (sr0 < lss).astype(np.float64)
    gate_sep = (dsrx0 < sgt).astype(np.float64)
    gate = gate_sep * coef * np.abs(dsrx0) + gate_low
    S = committed_set(gate, sr0, edges)
    seed = gate > 0
    A = _adj(edges, n)

    log_sr = np.log1p(np.maximum(sr0, 0.0))
    log_dsx = np.log1p(np.abs(dsrx0))
    ap_mult = 1.0 / (1.0 + C * gate * k_as / np.power(np.maximum(sr0, 1e-3), q))
    hop = hop_distance(seed, edges)

    # euclidean distance to the nearest seed, and position along the vessel axis
    if seed.any():
        from scipy.spatial import cKDTree
        seed_dist = cKDTree(pos[seed]).query(pos)[0]
    else:
        seed_dist = np.zeros(n)
    arc = pos[:, 0]
    arc = (arc - arc.min()) / max(np.ptp(arc), 1e-12)

    cols = [
        log_sr, _rank01(sr0), log_dsx, _rank01(dsrx0),
        gate, np.log1p(gate), gate_low, gate_sep,
        hop, hop / (GROW + 1.0), seed.astype(np.float64),
        _khop_mean(A, log_sr, 1), _khop_mean(A, log_sr, 2), _khop_mean(A, log_sr, 3),
        _khop_mean(A, gate, 1), _khop_mean(A, gate, 2),
        ap_mult, np.log1p(ap_mult),
        seed_dist / max(np.ptp(pos[:, 0]), 1e-12), arc,
        np.asarray(A.sum(1)).reshape(-1).astype(np.float64),
    ]
    X = np.stack(cols, axis=1)
    assert X.shape[1] == len(FEATURE_NAMES), (X.shape, len(FEATURE_NAMES))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), S


def onset_target(z) -> tuple[np.ndarray, np.ndarray]:
    """``(y, valid)`` where ``y`` is GT onset as a fraction of the horizon.

    Vessels run to different horizons and commit at wildly different absolute times, so the
    target is normalised per vessel; the metric is a per-vessel rank correlation anyway.
    """
    on = z["gt_onset"]
    nt = len(z["t"])
    valid = on >= 0
    return np.where(valid, on / max(nt - 1, 1), 0.0).astype(np.float64), valid
