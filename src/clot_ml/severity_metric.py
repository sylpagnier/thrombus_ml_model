"""`clot_severity_score` -- a burden-aware replacement for the flat relaxed deploy score.

WHY A NEW METRIC.  The shipped `deploy_clot_score` is
``0.5*dilation_IoU2 + 0.5*relaxed_F0.5_2``, computed per vessel and averaged.  It gets the
shape half right -- the 2-hop relaxation is exactly the "off by a couple of nodes is fine"
intuition -- but it has one property that does not match how a miss is actually judged:

    missing 5 of 15 off-wall nodes and missing 50 of 150 both read recall 0.667.

Clinically those are not the same failure.  The first under-reports a small clot slightly;
the second under-reports a large clot by a third.  And because the flat score is a *rate*,
low-burden vessels are the ones it punishes hardest: on a 4-node vessel a single false
positive costs more score than 30 false positives cost on a 120-node vessel.  That is why
`docs/PHASE9_ML.md` 5 found the off-wall mean dominated by the vessels with the least clot
in them.

THE FIX, in one sentence: **an absolute grace of a few nodes, capped at a fraction of the
true burden.**

    tau_eff = min(tau_abs, rho * n_gt)
    recall_eff = min(1, TP_rel / max(n_gt - tau_eff, 1))

`tau_abs` says "a handful of nodes either way is not a real failure"; `rho` stops that
grace from swallowing a small vessel whole (without it, predicting *nothing* on a 4-node
vessel would score well, which is the empty-prediction hole in reverse).  Worked example at
the defaults, `tau_abs = 5`, `rho = 0.25`:

    n_gt =  15, found 10  ->  tau_eff = 3.75  ->  recall_eff = 10/11.25 = 0.889
    n_gt = 150, found 100 ->  tau_eff = 5.00  ->  recall_eff = 100/145  = 0.690
    n_gt =   4, found  1  ->  tau_eff = 1.00  ->  recall_eff =  1/3     = 0.333

which is the ordering asked for, and cannot be gamed by predicting nothing.

Precision gets its own, much smaller grace (`tau_fp_abs`, `rho_fp`) relative to what was
*predicted*, so spraying is still punished in proportion to the spray.

PROPERTIES (all pinned by tests in `src/tests/test_severity_metric.py`):
  * exact agreement with the old score when `tau_abs = rho = tau_fp_abs = rho_fp = 0`;
  * monotone -- adding a true positive never lowers it, adding a false positive never
    raises it;
  * predicting nothing on a clot-carrying vessel scores 0, at every burden;
  * continuous in the counts (no cliff of the kind PHASE6_RESULTS 15.3 diagnosed);
  * a differentiable soft form for training (`soft_severity`).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class SeverityConfig:
    """All tolerances in NODES, except the two `rho` caps which are fractions."""

    relax_hops: int = 2
    beta: float = 0.5          # precision weight, as in the shipped score
    shape_w: float = 0.5       # weight on dilation IoU vs detection
    tau_abs: float = 5.0       # absolute miss grace
    rho: float = 0.25          # ... capped at this fraction of the true burden
    tau_fp_abs: float = 2.0    # absolute false-positive grace
    rho_fp: float = 0.15       # ... capped at this fraction of the prediction
    empty_gt_fp_tol: float = 8.0   # unchanged from the shipped score

    def as_dict(self) -> dict:
        return asdict(self)


LEGACY = SeverityConfig(tau_abs=0.0, rho=0.0, tau_fp_abs=0.0, rho_fp=0.0)
DEFAULT = SeverityConfig()


def dilation_operator(ei: np.ndarray, n: int, hops: int = 2) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(ei.shape[1], np.int8), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    D = (A + sp.eye(n, format="csr", dtype=np.int8)).astype(np.int8)
    out = D
    for _ in range(max(hops, 1) - 1):
        out = ((out @ D) > 0).astype(np.int8)
    return out


def _f_beta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    den = b2 * p + r
    return 0.0 if den <= 0 else (1 + b2) * p * r / den


def severity_components(pred: np.ndarray, gt: np.ndarray, D: sp.csr_matrix,
                        domain: np.ndarray | None = None,
                        cfg: SeverityConfig = DEFAULT) -> dict:
    """All the numbers, so a run can be diagnosed instead of just ranked."""
    g = gt if domain is None else (gt & domain)
    p = pred if domain is None else (pred & domain)
    n_g, n_p = int(g.sum()), int(p.sum())
    if n_g == 0:
        # No clot to find: grade the false-positive volume, as the shipped score does.
        return dict(score=1.0 / (1.0 + n_p / max(cfg.empty_gt_fp_tol, 1e-6)),
                    shape=float("nan"), detect=float("nan"), n_gt=0, n_pred=n_p,
                    tp_rel=0, fn=0, fp=n_p, recall_eff=float("nan"),
                    prec_eff=float("nan"), empty_gt=True)
    gd = (D @ g.astype(np.int8)) > 0
    if domain is not None:
        gd = gd & domain
    if n_p == 0:
        return dict(score=0.0, shape=0.0, detect=0.0, n_gt=n_g, n_pred=0, tp_rel=0,
                    fn=n_g, fp=0, recall_eff=0.0, prec_eff=0.0, empty_gt=False)
    pd_ = (D @ p.astype(np.int8)) > 0
    if domain is not None:
        pd_ = pd_ & domain
    tp_r = int((g & pd_).sum())          # GT nodes with a prediction within relax_hops
    tp_p = int((p & gd).sum())           # predictions with GT within relax_hops
    fn, fp = n_g - tp_r, n_p - tp_p

    tau = min(cfg.tau_abs, cfg.rho * n_g)
    rec = min(1.0, tp_r / max(n_g - tau, 1.0))
    tau_p = min(cfg.tau_fp_abs, cfg.rho_fp * n_p)
    prec = min(1.0, tp_p / max(n_p - tau_p, 1.0))
    detect = _f_beta(prec, rec, cfg.beta)

    inter = int((pd_ & gd).sum())
    union = int((pd_ | gd).sum())
    shape = 0.0 if union == 0 else inter / union
    score = cfg.shape_w * shape + (1.0 - cfg.shape_w) * detect
    return dict(score=score, shape=shape, detect=detect, n_gt=n_g, n_pred=n_p,
                tp_rel=tp_r, fn=fn, fp=fp, recall_eff=rec, prec_eff=prec, empty_gt=False)


def clot_severity_score(pred, gt, D, domain=None, cfg: SeverityConfig = DEFAULT) -> float:
    return severity_components(pred, gt, D, domain, cfg)["score"]


class SeverityScorer:
    """Precomputed per-vessel scorer (the dilation is the expensive part)."""

    def __init__(self, ei: np.ndarray, gt: np.ndarray, n: int,
                 cfg: SeverityConfig = DEFAULT):
        self.D = dilation_operator(ei, n, cfg.relax_hops)
        self.gt = gt.astype(bool)
        self.cfg = cfg

    def score(self, pred: np.ndarray, domain: np.ndarray | None = None) -> float:
        r = severity_components(pred, self.gt, self.D, domain, self.cfg)
        return float("nan") if r.get("empty_gt") else r["score"]

    def components(self, pred: np.ndarray, domain: np.ndarray | None = None) -> dict:
        return severity_components(pred, self.gt, self.D, domain, self.cfg)


# ---------------------------------------------------------------------------
# differentiable form -- the loss must be the metric (docs/PHASE9_ML.md 2a)
# ---------------------------------------------------------------------------
def soft_severity(p, gt, D_t, domain, gt_dil, cfg: SeverityConfig = DEFAULT):
    """Torch version of :func:`severity_components`'s ``score``.

    ``p`` is a probability field; the counts become expectations, the graces become the
    same capped quantities, and the noisy-OR dilation is exact on hard masks.
    """
    import torch

    eps = 1e-6
    p = p * domain
    g = gt * domain
    n_g = g.sum()
    if float(n_g) <= 0:
        return None
    n_p = p.sum()
    s = torch.sparse.mm(D_t, torch.log1p(-p.clamp(0, 1 - 1e-5)).reshape(-1, 1)).reshape(-1)
    p_dil = (1.0 - torch.exp(s)) * domain
    gd = gt_dil * domain
    tp_p = (p * gd).sum()
    tp_r = (g * p_dil).sum()

    tau = torch.clamp(cfg.rho * n_g, max=cfg.tau_abs)
    rec = torch.clamp(tp_r / torch.clamp(n_g - tau, min=1.0), max=1.0)
    tau_p = torch.clamp(cfg.rho_fp * n_p, max=cfg.tau_fp_abs)
    prec = torch.clamp(tp_p / torch.clamp(n_p - tau_p, min=1.0), max=1.0)
    b2 = cfg.beta ** 2
    detect = (1 + b2) * prec * rec / (b2 * prec + rec + eps)

    inter = torch.minimum(p_dil, gd).sum()
    union = torch.maximum(p_dil, gd).sum()
    shape = inter / (union + eps)
    return cfg.shape_w * shape + (1.0 - cfg.shape_w) * detect
