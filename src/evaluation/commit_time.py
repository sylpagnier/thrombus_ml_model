"""Commitment-order signal for pocket selection (docs/WALL_MODEL_PLAN.md s2.10 / s4 Step 1b).

s2.9 established that flow depth (``h2min``, src/evaluation/pocket_gate.py) cannot rank
patient037's pockets: the model commits a 40-node TRUE and a 40-node FALSE component at
statistically identical stagnation (0.048 vs 0.047). s2.10's hypothesis is that *timing* is
orthogonal to depth -- ``mat_seed_prec = 1.000`` on every checkpoint examined says the first
commitment is correct, so an earlier-committing component may outrank a later one even when
both sit equally deep in stagnant flow.

This module holds only the measurement primitives. It deliberately does NOT gate anything:
Step 1b's decision rule is to measure first, and fold commit-time into the gate only if it
separates the tie. Keeping it inert means a negative result costs a script, not a rollback.
"""

from __future__ import annotations

import numpy as np
import torch


def first_commit_step(
    phi_series: torch.Tensor,
    *,
    phi_thresh: float = 0.5,
    never: int | None = None,
) -> np.ndarray:
    """Per-node index of the first step at which ``phi`` crosses ``phi_thresh``.

    ``phi_series`` is ``[T, N]`` as returned by ``deploy_clot_phi_trajectory`` -- the same
    field the score thresholds, so "committed" here means exactly what it means in the metric.
    Nodes that never cross get ``never`` (default ``T``), which sorts after every real commit
    so it can be ranked without a special case.
    """
    if phi_series.dim() != 2:
        raise ValueError(f"phi_series must be [T, N], got shape {tuple(phi_series.shape)}")
    n_times = int(phi_series.shape[0])
    sentinel = n_times if never is None else int(never)
    committed = (phi_series > phi_thresh).cpu().numpy()
    ever = committed.any(axis=0)
    first = committed.argmax(axis=0).astype(np.int64)
    first[~ever] = sentinel
    return first


def rank_auc(
    pos: np.ndarray | list[float],
    neg: np.ndarray | list[float],
    *,
    w_pos: np.ndarray | list[float] | None = None,
    w_neg: np.ndarray | list[float] | None = None,
    pair_mask: np.ndarray | None = None,
) -> float:
    """``P(pos < neg) + 0.5 * P(pos == neg)`` -- 1.0 means the signal ranks perfectly.

    Lower values are the "true pocket" direction (early commit, deep stagnation), matching
    the ``AUC(TP h2min < FP h2min)`` convention used in s2.7/s2.9. ``pair_mask`` is a
    ``[len(pos), len(neg)]`` boolean restricting which pairs count -- used to score the
    tiebreak question ("among pairs flow calls a tie, does timing order them?").
    Returns ``nan`` when no pair is available to score.
    """
    a = np.asarray(pos, dtype=np.float64).reshape(-1, 1)
    b = np.asarray(neg, dtype=np.float64).reshape(1, -1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    wa = np.ones(a.shape[0]) if w_pos is None else np.asarray(w_pos, dtype=np.float64).reshape(-1)
    wb = np.ones(b.shape[1]) if w_neg is None else np.asarray(w_neg, dtype=np.float64).reshape(-1)
    w = wa.reshape(-1, 1) * wb.reshape(1, -1)
    if pair_mask is not None:
        w = w * np.asarray(pair_mask, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0:
        return float("nan")
    score = (a < b).astype(np.float64) + 0.5 * (a == b).astype(np.float64)
    return float((score * w).sum() / total)


def flow_tie_pairs(
    pos_flow: np.ndarray | list[float],
    neg_flow: np.ndarray | list[float],
    *,
    rel_tol: float = 0.05,
) -> np.ndarray:
    """``[len(pos), len(neg)]`` mask of (TP, FP) pairs whose ``h2min`` flow is a tie.

    A tie is ``|a - b| <= rel_tol * mean(a, b)``. These are exactly the pairs s2.9 says the
    flow gate cannot order (patient037's 0.048 vs 0.047 is a 2.1% gap), so they are the
    pairs a second signal has to earn its keep on.
    """
    a = np.asarray(pos_flow, dtype=np.float64).reshape(-1, 1)
    b = np.asarray(neg_flow, dtype=np.float64).reshape(1, -1)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[1]), dtype=bool)
    scale = 0.5 * (np.abs(a) + np.abs(b))
    return np.abs(a - b) <= (rel_tol * scale)


def predicted_components(pred_mask: np.ndarray, edge_index: np.ndarray) -> list[np.ndarray]:
    """Connected components of a predicted node mask, matching ``apply_pocket_gate``.

    Same construction as src/evaluation/pocket_gate.py so a component measured here is the
    same object the gate would keep or drop -- a probe that split them differently would
    not be measuring the gate's decision.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    idx = np.nonzero(np.asarray(pred_mask).reshape(-1))[0]
    if idx.size == 0:
        return []
    ei = np.asarray(edge_index)
    keep = np.isin(ei[0], idx) & np.isin(ei[1], idx)
    if not keep.any():
        return [np.array([i]) for i in idx]
    remap = {int(val): i for i, val in enumerate(idx)}
    rr = np.fromiter((remap[int(x)] for x in ei[0][keep]), dtype=int, count=int(keep.sum()))
    cc = np.fromiter((remap[int(x)] for x in ei[1][keep]), dtype=int, count=int(keep.sum()))
    adj = coo_matrix((np.ones(rr.size), (rr, cc)), shape=(idx.size, idx.size))
    ncomp, lab = connected_components(adj, directed=False)
    return [idx[lab == k] for k in range(ncomp)]
