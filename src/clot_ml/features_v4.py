"""The v4 feature block: advective transport + the indicator-gate physics variant.

Factored out of `scripts/build_clot_ml_cache_v4.py` so the SHIPPED model can rebuild exactly
the same channels at deploy time -- `src/clot_ml/locked.py`'s `build_sample` produces the 55
v3 channels, and a `clot_gnn_v4` member expects 68.  Single source of truth: the cache
builder imports from here.

See `docs/PHASE10_V4.md` 5 for what these are and why the boundary-outflow term in
`src/clot_ml/transport.py` is what makes them work.
"""
from __future__ import annotations

import numpy as np

from src.clot_ml.transport import transport_fields

M_TO_CM = 100.0

__all__ = ["indicator_physics", "horizon_for", "new_channels", "augment_sample",
           "V4_CHANNELS"]


def indicator_physics(data, bio, wall, hops=3):
    """The backbone rerun with the separation branch as an INDICATOR (see module docs)."""
    from src.core_physics.ap_closure import SHIPPED, SHIPPED_DA_SCALE, make_rollout_hook
    from src.core_physics.physics_wall_model import integrate_mat_trajectory, t0_flow_fields

    f0 = t0_flow_fields(data, bio, hops=hops, flow_source="gt")
    sgt = float(bio.sgt) / M_TO_CM
    gate_ind = (f0.dsrx < sgt).astype(np.float64) + (f0.sr < float(bio.lss)).astype(np.float64)
    hook = make_rollout_hook(SHIPPED, bio, f0.sr)
    traj, _ = integrate_mat_trajectory(data, bio, gate_ind * wall,
                                       da_scale=SHIPPED_DA_SCALE, ap_closure=hook)
    crit = float(bio.viscosity_mat_crit)
    hot = traj >= crit
    onset = np.where(hot.any(0), hot.argmax(0), traj.shape[0]).astype(np.float32) / traj.shape[0]
    return traj[-1], onset, gate_ind


def horizon_for(pos, u, v, wall):
    """Domain-crossing time at the bulk speed -- the natural time unit for the transport."""
    L = float(np.ptp(pos[:, 0]) + np.ptp(pos[:, 1]))
    spd = float(np.median(np.hypot(u, v)[~wall])) + 1e-12
    return L / spd


def new_channels(S, mat_ind, onset_ind, gate_ind, crit) -> dict:
    wall, ei, owner = S["wall"], S["edge_index"], S["owner"]
    pos = S["pos"].astype(np.float64)
    u, v = S["u"].astype(np.float64), S["v"].astype(np.float64)
    mat_phys = S["mat_phys"].astype(np.float64)
    H = horizon_for(pos, u, v, wall)

    T = transport_fields(pos, ei, u, v, wall, mat_phys, horizon=H)
    Ti = transport_fields(pos, ei, u, v, wall, np.asarray(mat_ind, float), horizon=H)

    def rel(a):
        """Value relative to the owner wall node's -- the attenuation, made dimensionless.

        This is the quantity PHASE7 12.5 asks for: `Mat_off/Mat_owner` has median 0.16 on
        every vessel but spans 0.12-0.19 *within* one, and near a threshold that spread is
        the whole off-wall gap.  Here it is computed from the flow rather than assumed.
        """
        return np.log1p(np.maximum(a, 0) / np.maximum(a[owner], 1e-30))

    tau = np.maximum(T["tau"], 0.0)
    return {
        # --- (A) advective transport -------------------------------------------------
        "log_mat_adv": np.log1p(np.maximum(T["mat_adv"], 0) / crit).astype(np.float32),
        "log_mat_adv_ind": np.log1p(np.maximum(Ti["mat_adv"], 0) / crit).astype(np.float32),
        "log_tau": np.log1p(tau / max(H, 1e-30)).astype(np.float32),
        "log_mat_adv_n": np.log1p(np.maximum(T["mat_adv_n"], 0) / crit).astype(np.float32),
        "log_src_reach": np.log1p(np.maximum(T["src_reach"], 0) / max(H, 1e-30)).astype(np.float32),
        # the flow-computed attenuation, and the same for pure wall-contact dose
        "att_adv": rel(T["mat_adv"]).astype(np.float32),
        "att_reach": rel(T["src_reach"]).astype(np.float32),
        "tau_rel_owner": rel(tau).astype(np.float32),
        # an absolute off-wall Mat estimate: owner's backbone Mat times the computed
        # attenuation, which is the shipped 0.16 rule with the constant made per-node
        "log_mat_off_est": np.log1p(
            np.maximum(mat_phys[owner] * np.minimum(
                np.maximum(T["mat_adv"], 0) / np.maximum(T["mat_adv"][owner], 1e-30), 4.0),
                0) / crit).astype(np.float32),
        # --- (B) separation branch as an indicator ------------------------------------
        "log_mat_phys_ind": np.log1p(np.maximum(mat_ind, 0) / crit).astype(np.float32),
        "onset_phys_ind": np.asarray(onset_ind, np.float32),
        "log_mat_ind_owner": np.log1p(
            np.maximum(np.asarray(mat_ind)[owner], 0) / crit).astype(np.float32),
        "gate_ind": np.asarray(gate_ind, np.float32),
    }


def augment_sample(data, S: dict, bio) -> tuple[np.ndarray, list[str]]:
    """Extend a v3 sample's ``X``/``cols`` with the v4 channels, in the cache's own order."""
    crit = float(bio.viscosity_mat_crit)
    wall = S["wall"]
    mat_ind, onset_ind, gate_ind = indicator_physics(data, bio, wall)
    NC = new_channels(S, mat_ind, onset_ind, gate_ind, crit)
    order = sorted(NC)
    X = np.concatenate([S["X"]] + [NC[k].reshape(-1, 1) for k in order], axis=1)
    return X.astype(np.float32), [str(c) for c in S["cols"]] + order


#: the 13 added channel names, sorted -- the order the cache and the models use
V4_CHANNELS = sorted([
    "att_adv", "att_reach", "gate_ind", "log_mat_adv", "log_mat_adv_ind", "log_mat_adv_n",
    "log_mat_ind_owner", "log_mat_off_est", "log_mat_phys_ind", "log_src_reach", "log_tau",
    "onset_phys_ind", "tau_rel_owner"])
