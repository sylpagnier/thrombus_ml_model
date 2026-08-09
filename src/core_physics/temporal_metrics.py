"""Onset-time metrics: does the model's growth CURVE match GT, not just its final mask?

The wall model reproduces the final committed set at deploy score 0.79/0.91 with zero
learned parameters, but ``scripts/diag_ignition_timing.py`` showed the trajectory behind
that mask is wrong: on patient043 all 84 nodes cross the threshold in the SAME step
(onset spread 0.000 of the horizon against GT's 0.725), because its gate is 100%
low-shear, ``gate == 1`` uniformly, and every node then has an identical ODE.

Any time-resolved or longer-horizon claim needs the curve, so these are the metrics the
temporal arms are scored on.  All are computed on the nodes that BOTH model and GT
commit, so they measure timing and not the mask (which the deploy score already covers).
"""
from __future__ import annotations

import numpy as np
import torch


def gt_onset_index(data, phys_cfg, wall: np.ndarray) -> np.ndarray:
    """Per-node index of the first timestep at which the GT growth label goes hot."""
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time

    nt = int(data.y.shape[0])
    hot = np.zeros((nt, len(wall)), dtype=bool)
    for i in range(nt):
        hot[i] = gt_clot_phi_at_time(data, i, phys_cfg, device=torch.device("cpu")).numpy() > 0.5
    any_hot = hot.any(axis=0)
    return np.where(any_hot, hot.argmax(axis=0), -1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def onset_metrics(model_idx, gt_idx, t, wall) -> dict:
    """Timing agreement on the nodes both sides commit.

    ``rho``       rank correlation of onset times -- does the model get the ORDER right?
    ``bias``      median (model - GT) onset, as a fraction of the horizon (signed)
    ``mae``       median |model - GT| onset, as a fraction of the horizon
    ``spread_*``  interquartile onset range as a fraction of the horizon
    ``spread_ratio`` model spread / GT spread; 1.0 means the growth curve is as gradual
    """
    horizon = float(t[-1]) if t[-1] > 0 else 1.0
    m_ok = (model_idx >= 0) & wall
    g_ok = (gt_idx >= 0) & wall
    both = m_ok & g_ok
    out = {"n_model": int(m_ok.sum()), "n_gt": int(g_ok.sum()), "n_both": int(both.sum())}
    mt_all = t[model_idx[m_ok]] if m_ok.any() else np.array([])
    gt_all = t[gt_idx[g_ok]] if g_ok.any() else np.array([])

    def iqr(v):
        return float(np.percentile(v, 75) - np.percentile(v, 25)) / horizon if len(v) else np.nan

    out["spread_model"] = iqr(mt_all)
    out["spread_gt"] = iqr(gt_all)
    out["spread_ratio"] = (out["spread_model"] / out["spread_gt"]
                           if out["spread_gt"] and out["spread_gt"] > 1e-9 else np.nan)
    out["median_model"] = float(np.median(mt_all)) / horizon if len(mt_all) else np.nan
    out["median_gt"] = float(np.median(gt_all)) / horizon if len(gt_all) else np.nan
    if both.sum() >= 3:
        mt, gtt = t[model_idx[both]], t[gt_idx[both]]
        out["rho"] = spearman(mt, gtt)
        out["bias"] = float(np.median(mt - gtt)) / horizon
        out["mae"] = float(np.median(np.abs(mt - gtt))) / horizon
    else:
        out["rho"] = out["bias"] = out["mae"] = np.nan
    return out


def curve_l1(model_idx, gt_idx, t, wall) -> float:
    """L1 distance between the two cumulative committed-fraction curves, normalised.

    A single number for "is the growth curve the right shape", insensitive to which
    individual nodes commit.  0 = identical curves, 1 = maximally different.
    """
    nt = len(t)
    n_gt = int(((gt_idx >= 0) & wall).sum())
    if n_gt == 0:
        return float("nan")
    mc = np.zeros(nt)
    gc = np.zeros(nt)
    for arr, dst in ((model_idx, mc), (gt_idx, gc)):
        sel = (arr >= 0) & wall
        for i in arr[sel]:
            dst[i] += 1
    mc = np.cumsum(mc) / max(mc.sum(), 1)
    gc = np.cumsum(gc) / max(gc.sum(), 1)
    w = np.diff(t, prepend=t[0])
    return float((np.abs(mc - gc) * w).sum() / max(float(t[-1] - t[0]), 1e-9))
