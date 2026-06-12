"""Principled soft clot growth loss for V3+ GNN rollouts.

Design (V3.1):
  - **Soft Tversky** on deploy band: symmetric FN/FP via probabilities (not hard snap).
  - **Hop-weighted false-clot penalty**: FP cost = pred^2 * dist_w^2 where dist_w grows
    with graph hops from nearest GT clot. Forgives 1-2 hop "almost right" errors;
    punishes distant wall-ring / lumen paint (better than pred_frac^2 alone).
  - **Volume hinge**: quadratic penalty only when pred_frac exceeds GT by a margin
    (under-paint is handled by Tversky/FN, not this term).
  - **Temporal weight**: sqrt(t_frac) so mid-trajectory is supervised, not just t_final.

Training uses step.loss_mask (deploy band). Eval reports band F1 + clot_shape on phi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.evaluation.clot_shape_score import (
    clot_shape_proximity_max_hops,
    graph_hop_distance_from_seeds,
    proximity_weight_from_gt_hops,
)


@dataclass
class ClotGrowthLossConfig:
    tversky_weight: float = 1.0
    tversky_alpha: float = 0.5  # FN weight
    tversky_beta: float = 0.55  # FP weight (slightly > alpha)
    tversky_smooth: float = 1e-6
    fp_hop_weight: float = 0.6
    fp_hop_power: float = 2.0
    vol_hinge_weight: float = 0.35
    vol_margin_frac: float = 0.12
    vol_power: float = 2.0
    bce_weight: float = 0.25
    onset_weight: float = 1.0
    onset_gt_thresh: float = 0.08
    onset_pred_thresh: float = 0.04
    ring_weight: float = 0.4
    ring_wall_hop_thresh: int = 3
    temporal_equal: bool = False


def hop_penalty_from_gt_clot(
    edge_index: torch.Tensor,
    n_nodes: int,
    gt_binary: np.ndarray,
    *,
    max_hop: int | None = None,
) -> np.ndarray:
    """Per-node FP distance penalty in [0, 1]: 0 on GT clot, ~1 far from GT."""
    mh = clot_shape_proximity_max_hops() if max_hop is None else max(int(max_hop), 0)
    hop = graph_hop_distance_from_seeds(
        edge_index, n_nodes, gt_binary.astype(bool), max_hops=mh + 2
    )
    prox = np.array([proximity_weight_from_gt_hops(int(h), max_hop=mh) for h in hop], dtype=np.float32)
    return np.clip(1.0 - prox, 0.0, 1.0)


def soft_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Differentiable Tversky on masked nodes (pred/target in [0,1])."""
    m = mask.reshape(-1).bool()
    p = pred.reshape(-1)[m].clamp(0.0, 1.0)
    t = target.reshape(-1)[m].clamp(0.0, 1.0)
    if p.numel() == 0:
        return pred.sum() * 0.0
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    fn = ((1.0 - p) * t).sum()
    num = tp + smooth
    den = tp + float(alpha) * fn + float(beta) * fp + smooth
    return 1.0 - num / den


def hop_weighted_fp_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    hop_penalty: torch.Tensor,
    *,
    power: float = 2.0,
) -> torch.Tensor:
    """Penalize false clots: pred^power on GT-negative nodes, scaled by hop^2."""
    m = mask.reshape(-1).bool()
    p = pred.reshape(-1)[m].clamp(0.0, 1.0)
    t = target.reshape(-1)[m].clamp(0.0, 1.0)
    hp = hop_penalty.reshape(-1)[m].clamp(0.0, 1.0)
    neg = (1.0 - t).clamp(0.0, 1.0)
    if p.numel() == 0:
        return pred.sum() * 0.0
    fp_w = neg * (p ** float(power)) * (hp ** 2.0)
    return fp_w.mean()


def volume_hinge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    margin_frac: float = 0.12,
    power: float = 2.0,
) -> torch.Tensor:
    """Quadratic penalty when masked pred volume exceeds GT + margin (over-paint only)."""
    m = mask.reshape(-1).bool()
    if not bool(m.any().item()):
        return pred.sum() * 0.0
    p = pred.reshape(-1)[m].clamp(0.0, 1.0)
    t = target.reshape(-1)[m].clamp(0.0, 1.0)
    pred_frac = p.mean()
    gt_frac = t.mean()
    cap = gt_frac + float(margin_frac)
    over = (pred_frac - cap).clamp(min=0.0)
    return over ** float(power)


def temporal_frame_weight(t_frac: float, *, equal: bool = False) -> float:
    """Supervise full timeline; equal weight or sqrt ramp."""
    if equal:
        return 1.0
    return float(max(t_frac, 0.0) ** 0.5) + 0.35


def onset_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    gt_thresh: float = 0.08,
    pred_thresh: float = 0.04,
) -> torch.Tensor:
    """Penalize missing growth when GT already has clot in band."""
    m = mask.reshape(-1).bool()
    p = pred.reshape(-1)[m].clamp(0.0, 1.0)
    t = target.reshape(-1)[m].clamp(0.0, 1.0)
    if p.numel() == 0:
        return pred.sum() * 0.0
    gt_frac = t.mean()
    pred_frac = p.mean()
    if float(gt_frac.item()) < gt_thresh:
        return pred.sum() * 0.0
    miss = (gt_frac - pred_frac).clamp(min=0.0)
    return miss ** 2


def wall_ring_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    wall_mask: torch.Tensor,
    hop_penalty: torch.Tensor,
    *,
    commit_thresh: float = 0.5,
    wall_hop_thresh: float = 3.0,
) -> torch.Tensor:
    """Soft penalty for pred+ on wall far from GT clot."""
    p = pred.reshape(-1).clamp(0.0, 1.0)
    t = target.reshape(-1).clamp(0.0, 1.0)
    wall = wall_mask.reshape(-1).bool()
    hp = hop_penalty.reshape(-1).clamp(0.0, 1.0)
    far = hp > (wall_hop_thresh / 8.0)
    false_wall = wall & far & (t < 0.5)
    if not bool(false_wall.any().item()):
        return pred.sum() * 0.0
    return (p[false_wall] ** 2).mean()


def clot_growth_frame_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    hop_penalty: torch.Tensor,
    *,
    cfg: ClotGrowthLossConfig | None = None,
    wall_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Single-frame composite loss (differentiable)."""
    c = cfg or ClotGrowthLossConfig()
    tversky = soft_tversky_loss(
        pred,
        target,
        mask,
        alpha=c.tversky_alpha,
        beta=c.tversky_beta,
        smooth=c.tversky_smooth,
    )
    fp_hop = hop_weighted_fp_loss(
        pred, target, mask, hop_penalty, power=c.fp_hop_power
    )
    vol = volume_hinge_loss(
        pred, target, mask, margin_frac=c.vol_margin_frac, power=c.vol_power
    )
    onset = onset_alignment_loss(
        pred,
        target,
        mask,
        gt_thresh=c.onset_gt_thresh,
        pred_thresh=c.onset_pred_thresh,
    )
    ring = pred.sum() * 0.0
    if wall_mask is not None and c.ring_weight > 0:
        ring = wall_ring_loss(
            pred,
            target,
            wall_mask,
            hop_penalty,
            wall_hop_thresh=float(c.ring_wall_hop_thresh),
        )
    m = mask.reshape(-1).bool()
    bce = (
        F.binary_cross_entropy(pred.reshape(-1)[m], target.reshape(-1)[m], reduction="mean")
        if bool(m.any().item())
        else pred.sum() * 0.0
    )
    total = (
        c.tversky_weight * tversky
        + c.fp_hop_weight * fp_hop
        + c.vol_hinge_weight * vol
        + c.bce_weight * bce
        + c.onset_weight * onset
        + c.ring_weight * ring
    )
    return {
        "loss": total,
        "tversky": tversky.detach(),
        "fp_hop": fp_hop.detach(),
        "vol_hinge": vol.detach(),
        "bce": bce.detach(),
        "onset": onset.detach(),
        "ring": ring.detach(),
    }
