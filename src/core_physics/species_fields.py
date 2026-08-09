"""Bulk platelet fields at the wall, as constants or as GT trajectories.

The wall model holds ``rp``/``ap`` at their t=0 values on the strength of
PHASE3_HANDOFF 1.3 / 26.16 ("spatial CV 0.003 / 0.095 -- there is almost no chemistry to
learn").  COMSOL's own patient007 wall export disagrees:

    ap   min 5.14e5   max 1.25e7   mean 8.52e6      [plt/cm^3]

``c_AP0`` is 1.25e7, i.e. the maximum -- so **AP is depleted to 4% of its inlet value
where deposition is heavy**, a 24x range rather than 10%.  That is a negative feedback the
model does not have: clot consumes activated platelets and starves its own growth, which
is one of the few mechanisms that could stretch onset over the 70%-of-horizon spread GT
shows.  Running the opposite way, committed nodes generate thrombin
(``J0_th = beta*phi_at*Mat*PT``) which activates platelets, and ``k_as`` is 12x ``k_rs``.

This module supplies both the frozen constants and the GT trajectories so the two can be
swapped in a controlled ablation (``scripts/opt_ladder.py chem``).
"""
from __future__ import annotations

import numpy as np
import torch

PER_M3_TO_PER_CM3 = 1.0e-6


def _channel(data, name: str) -> int:
    return data.y_channel_names.split(",").index(name)


def gt_species_trajectory(data, bio_cfg) -> tuple[np.ndarray, np.ndarray]:
    """``(rp, ap)`` in CGS [plt/cm^3], shape ``[T, N]`` -- the GT bulk fields at every step.

    Oracle input: legal for an ablation that bounds what the chemistry is worth, not for
    a deployable model.
    """
    scales = bio_cfg.get_species_scales(device="cpu")
    out = []
    for name, idx in (("RP_log1p_nd", 0), ("AP_log1p_nd", 1)):
        nd = torch.expm1(data.y[:, :, _channel(data, name)].clamp(-10, 8)).numpy()
        out.append(nd.astype(np.float64) * float(scales[idx]) * PER_M3_TO_PER_CM3)
    return out[0], out[1]


def constant_species(data, bio_cfg) -> tuple[np.ndarray, np.ndarray]:
    """``(rp, ap)`` in CGS [plt/cm^3], shape ``[N]`` -- the t=0 initial condition."""
    from src.core_physics.physics_wall_model import wall_platelet_constants

    return wall_platelet_constants(data, bio_cfg)


def depletion_report(data, bio_cfg, wall: np.ndarray) -> dict:
    """How far from constant are the GT bulk fields at the wall, in time and space?"""
    rp, ap = gt_species_trajectory(data, bio_cfg)
    w = wall.astype(bool)
    a0 = ap[0][w]
    af = ap[-1][w]
    amin = ap[:, w].min()
    return {
        "ap_t0_median": float(np.median(a0)),
        "ap_tfinal_median": float(np.median(af)),
        "ap_min_over_run": float(amin),
        "ap_depletion_ratio": float(np.median(af) / max(np.median(a0), 1e-30)),
        "ap_min_frac_of_inlet": float(amin / max(ap[0].max(), 1e-30)),
        "ap_spatial_cv_t0": float(np.std(a0) / max(np.mean(a0), 1e-30)),
        "ap_spatial_cv_tfinal": float(np.std(af) / max(np.mean(af), 1e-30)),
        "rp_depletion_ratio": float(np.median(rp[-1][w]) / max(np.median(rp[0][w]), 1e-30)),
    }
