"""A differentiable copy of the deploy score, so the network can train on the metric.

`deploy_clot_score = 0.5*dilation_IoU + 0.5*relaxed_F0.5`, both defined on 2-hop dilated
masks.  BCE optimises none of that: it is per-node, unweighted by neighbourhood, and blind
to the fact that the score forgives a 2-hop miss entirely and punishes precision at
beta=0.5.  Training on a soft copy of the real thing removes that mismatch.

Dilation is a **noisy-OR** over the 2-hop neighbourhood -- ``1 - prod(1 - p_j)`` -- which is
the smooth analogue of the boolean OR the metric uses and is exact at p in {0,1}.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

EPS = 1e-6


def dilation_operator(ei: np.ndarray, n: int, hops: int = 2) -> sp.csr_matrix:
    A = sp.coo_matrix((np.ones(ei.shape[1], np.int8), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    D = (A + sp.eye(n, format="csr", dtype=np.int8)).astype(np.int8)
    out = D
    for _ in range(hops - 1):
        out = ((out @ D) > 0).astype(np.int8)
    return out


def to_torch_sparse(M: sp.csr_matrix, device) -> torch.Tensor:
    C = M.tocoo()
    idx = torch.tensor(np.stack([C.row, C.col]), dtype=torch.long, device=device)
    val = torch.ones(idx.shape[1], dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, M.shape).coalesce()


def soft_dilate(p: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """noisy-OR over the 2-hop ball; exact for hard masks."""
    s = torch.sparse.mm(D, torch.log1p(-p.clamp(0, 1 - 1e-5)).reshape(-1, 1)).reshape(-1)
    return 1.0 - torch.exp(s)


def soft_score(p: torch.Tensor, gt: torch.Tensor, D: torch.Tensor,
               domain: torch.Tensor, gt_dil: torch.Tensor,
               shape_w: float = 0.5) -> torch.Tensor:
    """Differentiable `shape_w*dilation_IoU + (1-shape_w)*relaxed_F0.5` over ``domain``.

    ``shape_w`` exists because the two halves are NOT equally binding.  Measured per vessel
    on the shipped off-wall mask (docs/PHASE10_V4.md 13.1), recall is already 1.000 on nine
    of thirteen vessels and `patient012` still scores 0.845 with precision 1.000 and recall
    0.988 -- because its dilation IoU is 0.691.  Above ~0.85 the off-wall score is a SHAPE
    problem, and the training loss has always weighted shape at exactly the 0.5 the
    evaluation metric uses.  The EVALUATION weight never changes; this is the loss only.
    """
    p = p * domain
    g = gt * domain
    n_gt = g.sum()
    if float(n_gt) <= 0:
        return None
    p_dil = soft_dilate(p, D) * domain
    gd = gt_dil * domain
    rel_p = (p * gd).sum() / (p.sum() + EPS)
    rel_r = (g * p_dil).sum() / (n_gt + EPS)
    b2 = 0.25
    f05 = (1 + b2) * rel_p * rel_r / (b2 * rel_p + rel_r + EPS)
    inter = torch.minimum(p_dil, gd).sum()
    union = torch.maximum(p_dil, gd).sum()
    iou = inter / (union + EPS)
    return shape_w * iou + (1.0 - shape_w) * f05
