"""ARM 2 -- algebraic shear redistribution as the clot occludes its own lumen.

WHY.  The wall model freezes the gates at t=0, so growth never sees the clot narrowing
the vessel.  That is why it ignites in a flash: nothing ever shuts a gate.  COMSOL does
not remove geometry when a node commits -- ``mu1(Mat)`` steps the local viscosity 1 -> 80
(``BiochemConfig.mu_ratio_max``), which excludes flow from that tissue just as
effectively.  Conservation of flow rate then does the rest: if a fraction ``phi`` of the
local cross-section is occluded, the remaining lumen carries the same Q, so

    speed  ~  1 / (1 - phi)          and       shear ~ 1 / (1 - phi)**p

with ``p = 2`` for a 2D channel (gamma_wall ~ Q/r^2 at fixed Q) and ``p = 3`` for a tube.
Rising shear CLOSES the low-shear gate, so growth self-limits -- which is the missing
negative feedback.

This is deliberately algebraic, not a flow solve: no ML, no network, one sparse matvec per
update.  It IS non-local in the way that matters -- the cross-section ball spans the lumen,
so clot on the far wall raises the shear here, which is the effect a purely local rule
cannot express.

Deploy-legal: node positions, connectivity and ``sdf_nd`` (geometry only).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from src.utils.channel_schema import KINE_X_SCHEMA, X_SCHEMAS


def sdf_nd(data) -> np.ndarray:
    """Signed distance to the wall in the pack's non-dimensional length unit."""
    ch = list(X_SCHEMAS[KINE_X_SCHEMA].channels)
    return data.x[:, ch.index("sdf_nd")].detach().cpu().numpy().astype(np.float64)


def local_half_width(pos: np.ndarray, sdf: np.ndarray, wall: np.ndarray,
                     *, probe: float | None = None) -> np.ndarray:
    """Per-wall-node lumen half-width: the deepest interior point within a probe radius."""
    if probe is None:
        probe = 2.0 * float(np.percentile(sdf, 99))
    tree = cKDTree(pos)
    out = np.zeros(len(pos))
    for i in np.where(wall)[0]:
        idx = tree.query_ball_point(pos[i], probe)
        out[i] = float(sdf[idx].max()) if idx else 0.0
    return np.maximum(out, 1e-6)


def build_crosssection_operator(
    pos: np.ndarray, sdf: np.ndarray, wall: np.ndarray, *,
    radius_mult: float = 1.0, probe: float | None = None,
) -> sp.csr_matrix:
    """Row-normalised ``B`` [N,N]: ``B @ occluded`` is the occluded AREA FRACTION of the
    local cross-section at each wall node.

    The ball radius is the local lumen half-width times ``radius_mult``, so the stencil
    reaches across the lumen to the opposite wall and no further.  Node areas are taken as
    uniform (the meshes are near-uniform: patient007 edge-length CV is small).
    """
    r0 = local_half_width(pos, sdf, wall, probe=probe)
    tree = cKDTree(pos)
    rows, cols = [], []
    for i in np.where(wall)[0]:
        idx = tree.query_ball_point(pos[i], max(r0[i] * radius_mult, 1e-6))
        if not idx:
            idx = [i]
        rows.append(np.full(len(idx), i))
        cols.append(np.asarray(idx))
    if not rows:
        return sp.csr_matrix((len(pos), len(pos)))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    B = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pos), len(pos))).tocsr()
    deg = np.asarray(B.sum(axis=1)).reshape(-1)
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ B


def make_blockage(
    fields, bio_cfg, B: sp.csr_matrix, wall: np.ndarray, *,
    exponent: float = 2.0, phi_max: float = 0.85, every: int = 5,
    graded_mode: str = "hard", tau_low: float = 0.0, tau_sep: float = 0.0,
    feedback: str = "occlude", wake: float = 1.0, thrombin_solve=None,
):
    """Return a ``blockage(mat, gate0, step) -> gate`` callable for the ODE integrator.

    Recomputes the occluded fraction, rescales ``sr``/``dsrx``, and re-evaluates the gate
    every ``every`` steps.  ``dsrx`` is rescaled by the same local factor -- exact for a
    uniform rescale, first-order otherwise, and it keeps the two gate branches consistent.

    ``feedback='occlude'``  shear RISES as the lumen narrows, ``1/(1-phi)**p``.  Measured
        against GT this is the wrong sign: ``scripts/diag_gt_shear_evolution.py`` finds the
        low-shear gate open fraction *rising* over the run (patient007 0.153 -> 0.298),
        and the occluded fraction never exceeds a few percent anyway.
    ``feedback='wake'``  shear FALLS next to committed tissue, ``1 - wake*phi``.  Committed
        tissue is a no-slip obstacle at 80x viscosity, so it sheds a stagnation wake rather
        than accelerating the bulk -- which is what the GT gate-opening actually shows.
    """
    from src.core_physics.physics_wall_model import T0Fields, graded_gate

    crit = float(bio_cfg.viscosity_mat_crit)
    state = {"gate": None, "last": -10 ** 9}

    def blockage(mat, gate0, step):
        if state["gate"] is not None and (step - state["last"]) < every:
            return state["gate"]
        occ = (mat >= crit).astype(np.float64)
        if feedback == "thrombin":
            # Same mechanism as 'wake' -- committed tissue lowers the shear its neighbours
            # see, flipping their gates -- but the RANGE comes from D_T and the horizon
            # (src/core_physics/thrombin_field.py) instead of a fitted ball radius. This is
            # the fitted-scalar-for-derived-constant trade the whole rung exists to test.
            phi = np.clip(thrombin_solve(occ), 0.0, phi_max)
        else:
            phi = np.clip(np.asarray(B @ occ).reshape(-1), 0.0, phi_max)
        if feedback in ("wake", "thrombin"):
            amp = np.clip(1.0 - float(wake) * phi, 0.02, 1.0)
        else:
            amp = (1.0 - phi) ** (-float(exponent))
        f2 = T0Fields(sr=fields.sr * amp, dsrx=fields.dsrx * amp,
                      gate_low=None, gate_sep=None, gate=None)
        f2.gate_low = (f2.sr < float(bio_cfg.lss)).astype(np.float64)
        f2.gate_sep = (f2.dsrx < float(bio_cfg.sgt) / 100.0).astype(np.float64)
        g = graded_gate(f2, bio_cfg, mode=graded_mode, tau_low=tau_low, tau_sep=tau_sep) * wall
        # A node already committed keeps depositing: mu1 has fired, it is clot now.
        g = np.where(occ > 0, np.maximum(g, gate0), g)
        state["gate"] = g
        state["last"] = step
        return g

    return blockage
