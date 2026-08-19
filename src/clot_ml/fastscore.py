"""A fast, exactly-equivalent copy of the domain-restricted deploy score.

The reference path (`compute_clot_relaxed_metrics`) re-does a 2-hop graph dilation on a
15k-node mesh for every call.  Threshold selection needs O(10^4) calls per arm, so the
dilation operator is precomputed once per vessel and the score becomes two sparse matvecs.

`assert_matches_reference` pins it against the real implementation; run it in the tests.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

RELAX_HOPS = 2
F_BETA = 0.5
IOU_W = F05_W = 0.5
EMPTY_GT_FP_TOL = 8.0


def _dilator(ei: np.ndarray, n: int, hops: int) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(ei.shape[1], np.int8), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    D = (A + sp.eye(n, format="csr", dtype=np.int8)).astype(np.int8)
    out = D
    for _ in range(hops - 1):
        out = ((out @ D) > 0).astype(np.int8)
    return out


class VesselScorer:
    """Precomputed scorer for one vessel.  ``score(pred, domain)`` matches the reference."""

    def __init__(self, ei: np.ndarray, gt: np.ndarray, n: int):
        self.D = _dilator(ei, n, RELAX_HOPS)
        self.gt = gt.astype(bool)
        self.n = n
        self._cache: dict[int, tuple] = {}

    def _gt_ctx(self, domain: np.ndarray):
        key = id(domain) if domain is not None else 0
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        g = self.gt & domain if domain is not None else self.gt
        gd = (self.D @ g.astype(np.int8)) > 0
        self._cache[key] = (g, gd, int(g.sum()))
        return self._cache[key]

    def score(self, pred: np.ndarray, domain: np.ndarray | None = None) -> float:
        g, gd, n_gt = self._gt_ctx(domain)
        p = pred & domain if domain is not None else pred.astype(bool)
        n_pred = int(p.sum())
        if n_gt == 0:
            return float("nan")
        if n_pred == 0:
            return 0.0
        pd = (self.D @ p.astype(np.int8)) > 0
        rel_p = float((p & gd).sum()) / n_pred
        rel_r = float((g & pd).sum()) / n_gt
        b2 = F_BETA ** 2
        den = b2 * rel_p + rel_r
        f_beta = 0.0 if den <= 0 else (1 + b2) * rel_p * rel_r / den
        inter = int((pd & gd).sum())
        union = int((pd | gd).sum())
        iou = 0.0 if union == 0 else inter / union
        return (IOU_W * iou + F05_W * f_beta) / (IOU_W + F05_W)


def assert_matches_reference(ei, gt, pred, wall, n, *, atol=1e-9):
    import torch

    from src.clot_ml.evaluate import domain_score

    vs = VesselScorer(ei, gt, n)
    for domain in (wall, ~wall):
        ref = domain_score(pred, gt, torch.tensor(ei), domain, wall)
        got = vs.score(pred, domain)
        if ref != ref and got != got:
            continue
        assert abs(ref - got) < atol, (ref, got)
    return True
