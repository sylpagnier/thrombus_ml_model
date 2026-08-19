"""Growth-curve error by NODE COUNT: ``|n_pred(t) - n_gt(t)|``, integrated over time.

WHY A COUNT METRIC REPLACES THE OVERLAP SCORE FOR THIS QUESTION.  The wall-masked
``deploy_clot_score`` is a set-overlap score evaluated per timestep, and PHASE6_RESULTS 15.3
measured what that does to a time-resolved comparison:

        predict nothing -> 1.0000 while GT is empty, 0.0000 the instant GT is not
        predict full S  -> 0.05-0.24 while GT is empty, partial after

It is a **cliff**, discontinuous in the model's commit time, and it rewards committing early
and completely rather than committing at the right time.  Every physically-realistic arm --
every field oracle, every direct-onset model -- lost on it, and the arm with the best growth
curve scored worst.  A metric that ranks the best answer last is not measuring the thing.

This one is continuous in every onset time, blind to WHICH nodes commit, and therefore
measures exactly the stated objective: does the predicted growth curve track the real one.

    growth_l1 = mean_t |n_pred(t) - n_gt(t)| / N_gt_final

Normalised by the vessel's final GT count so vessels with 20 and 200 committed nodes are
comparable; ``growth_l1_abs`` keeps the raw node count for when the absolute scale matters.

WHAT IT DELIBERATELY DOES NOT DO.  It cannot see node identity, so a model that commits the
right NUMBER of wrong nodes scores perfectly.  That is acceptable here only because the
committed set is a separately solved and separately scored problem (the final mask sits at
the flow-oracle ceiling, PHASE6_HANDOFF 0).  Never report this without the mask score
beside it.
"""
from __future__ import annotations

import numpy as np


def count_curve(onset: np.ndarray, nt: int, sel: np.ndarray | None = None) -> np.ndarray:
    """``n(t)`` for t = 0..nt-1: how many nodes have committed by each step."""
    idx = np.asarray(onset)
    if sel is not None:
        idx = np.where(sel, idx, -1)
    hist = np.bincount(idx[idx >= 0], minlength=nt)[:nt]
    return np.cumsum(hist).astype(np.float64)


def growth_error(model_onset: np.ndarray, gt_onset: np.ndarray, nt: int,
                 wall: np.ndarray | None = None) -> dict:
    """``|n_pred - n_gt|`` summarised over the whole horizon.

    ``growth_l1``      mean over time, normalised by the final GT count -- the headline
    ``growth_linf``    worst single timestep, same normalisation
    ``final_err``      signed final-count error; the part no timing model can fix
    ``timing_l1``      the same L1 with both curves rescaled to their own final count, i.e.
                       the error that remains after the mask-size mismatch is removed
    """
    m = count_curve(model_onset, nt, wall)
    g = count_curve(gt_onset, nt, wall)
    n_final = float(g[-1])
    if n_final <= 0:
        return {k: float("nan") for k in
                ("growth_l1", "growth_linf", "growth_l1_abs", "final_err", "timing_l1",
                 "n_pred_final", "n_gt_final")}
    d = np.abs(m - g)
    mm = m / max(m[-1], 1e-9)
    gg = g / n_final
    return dict(
        growth_l1=float(d.mean() / n_final),
        growth_linf=float(d.max() / n_final),
        growth_l1_abs=float(d.mean()),
        final_err=float((m[-1] - n_final) / n_final),
        timing_l1=float(np.abs(mm - gg).mean()),
        n_pred_final=float(m[-1]),
        n_gt_final=n_final,
    )


def count_optimal_onset(S: np.ndarray, gt_onset: np.ndarray, nt: int,
                        wall: np.ndarray | None = None) -> np.ndarray:
    """The best onset assignment achievable ON THIS MASK under the count metric.

    The metric only sees counts, so the optimum is to make the k-th committing node appear
    exactly when GT's k-th node does.  Whatever error remains is purely the mask-size
    mismatch -- it is the FLOOR, and the gap between the shipped model and this floor is the
    entire prize available to any timing model.
    """
    g = count_curve(gt_onset, nt, wall)
    idx = np.where(S)[0]
    out = -np.ones(len(S), dtype=int)
    # step at which GT's count first reaches k+1
    reach = np.searchsorted(g, np.arange(1, len(idx) + 1), side="left")
    for k, node in enumerate(idx):
        out[node] = int(min(reach[k], nt - 1)) if k < len(reach) else nt - 1
    return out
