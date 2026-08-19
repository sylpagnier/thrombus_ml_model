"""Time-resolved clot masks: the shipped SET, scheduled by the physics ODE's TIMING.

Measured motivation (`scripts/eda_timing_prize.py`, mean-over-time severity deploy score):

    frozen_model  (ships today)            wall 0.7921   off 0.5015
    frozen_oracle (perfect SET, no time)   wall 0.8190   off 0.5744
    physics_onset (zero-param ODE timing)  wall 0.8247   off 0.5015
    oracle_onset  (perfect timing)         wall 0.9897   off 1.0000

Two things follow.  Crude timing beats a *perfect* frozen set on the wall -- committing
everything at t=0 is worse than committing the wrong things at roughly the right times.
And the off-wall timing prize (+0.43) is completely untouched, because the ODE is a wall
object.

This module supplies both halves without training anything:

  wall     onset = the ODE's own first crossing of ``viscosity_mat_crit``
  off-wall onset = the time the node's OWNER wall trajectory crosses ``crit / attenuation``

The off-wall rule is the time-domain form of the measured 0.16 attenuation: if
``Mat_off(t) ~= att * Mat_owner(t)`` with ``att`` stable in time -- which
`scripts/eda_extrapolate.py` confirmed (0.004 -> 0.003 across the horizon) -- then an
off-wall node crosses ``crit`` exactly when its owner crosses ``crit / att``.  Same
trajectory, later threshold; no second model.

The SET is never taken from the ODE.  It stays whatever the caller supplies (the locked
GNN mask), because the GNN's set is materially better -- the ODE only supplies *when*.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_ATTENUATION = 0.16


def _first_crossing(traj: np.ndarray, thresh: float) -> np.ndarray:
    hot = traj >= thresh
    return np.where(hot.any(axis=0), hot.argmax(axis=0), -1)


def ode_trajectory(data, bio_cfg, *, flow: str = "gt", ap_closure: bool = True):
    """The zero-parameter surface ODE's ``Mat`` trajectory ``[T, N]`` plus the time grid."""
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.physics_wall_model import integrate_mat_trajectory, t0_flow_fields

    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    f = t0_flow_fields(data, bio_cfg, hops={"gt": 3, "pred": 4}[flow], flow_source=flow)
    hook = make_rollout_hook(SHIPPED, bio_cfg, f.sr) if ap_closure else None
    traj, t = integrate_mat_trajectory(data, bio_cfg, f.gate * wall,
                                       da_scale=SHIPPED_DA_SCALE, ap_closure=hook)
    return traj, np.asarray(t).reshape(-1)


def onset_from_ode(traj, mask, wall, pos, crit, *, attenuation=DEFAULT_ATTENUATION):
    """Per-node onset INDEX for the nodes in ``mask``; -1 elsewhere.

    Wall nodes take the ODE's own crossing of ``crit``.  Off-wall nodes take their owner's
    crossing of ``crit / attenuation``.  Masked nodes the ODE never ignites (the graph-grown
    ones) fall back to the median onset of those it does, which is the convention
    ``predict_wall_onset`` already uses.
    """
    T = traj.shape[0]
    on_w = _first_crossing(traj, crit)
    on_hi = _first_crossing(traj, crit / max(attenuation, 1e-9))

    widx = np.flatnonzero(wall)
    owner = widx[cKDTree(pos[wall]).query(pos)[1]] if len(widx) else np.zeros(len(wall), int)

    onset = np.full(len(wall), -1, dtype=int)
    onset[wall & mask] = on_w[wall & mask]
    off = (~wall) & mask
    onset[off] = on_hi[owner][off]

    # Fallback is taken from WALL-ignited nodes only.  Pooling wall and off-wall makes the
    # fallback depend on the off-wall rule, which silently changes the WALL score between
    # arms that should differ only off-wall.
    ignited = on_w[wall & mask & (on_w >= 0)]
    fallback = int(np.median(ignited)) if ignited.size else T - 1
    onset[mask & (onset < 0)] = fallback
    return onset


def mask_series(onset: np.ndarray, mask: np.ndarray, times) -> dict:
    """``time_index -> boolean mask``.  Nested by construction: clot never un-clots."""
    return {int(ti): mask & (onset >= 0) & (onset <= ti) for ti in times}
