"""Tests for relaxed clot guiding metrics."""

from __future__ import annotations

import torch

from src.evaluation.clot_relaxed_metrics import (
    clot_guiding_score,
    compute_clot_relaxed_metrics,
    empty_gt_match_score,
    f_beta_score,
)


def _chain_graph(n: int) -> torch.Tensor:
    edges = []
    for i in range(n - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges, dtype=torch.long).t()


def test_perfect_overlap_guiding_one():
    n = 7
    ei = _chain_graph(n)
    gt = torch.zeros(n)
    gt[3] = 1.0
    pred = gt.clone()
    m = compute_clot_relaxed_metrics(pred, gt, ei, relax_hops=2)
    assert m["clot_guiding"] == 1.0
    assert m["clot_dilation_iou"] == 1.0
    assert m["clot_relaxed_f05"] == 1.0
    assert m["clot_f1"] == 1.0


def test_distant_fp_hurts_relaxed_precision_and_iou():
    n = 9
    ei = _chain_graph(n)
    gt = torch.zeros(n)
    gt[4] = 1.0
    pred = gt.clone()
    pred[0] = 1.0  # far from GT on chain
    m = compute_clot_relaxed_metrics(pred, gt, ei, relax_hops=1)
    assert m["clot_fp"] == 1.0
    assert m["clot_relaxed_prec"] < 1.0
    assert m["clot_dilation_iou"] < 1.0
    assert m["clot_guiding"] < 1.0


def test_near_miss_within_hops_counts_as_tp():
    n = 7
    ei = _chain_graph(n)
    gt = torch.zeros(n)
    gt[3] = 1.0
    pred = torch.zeros(n)
    pred[4] = 1.0  # 1 hop off
    m = compute_clot_relaxed_metrics(pred, gt, ei, relax_hops=1)
    assert m["clot_relaxed_rec"] == 1.0
    assert m["clot_relaxed_prec"] == 1.0
    assert m["clot_f1"] == 0.0  # strict miss


def test_spam_predictions_tank_f05():
    n = 11
    ei = _chain_graph(n)
    gt = torch.zeros(n)
    gt[5] = 1.0
    pred = torch.ones(n)  # predict clot everywhere
    m = compute_clot_relaxed_metrics(pred, gt, ei, relax_hops=1, f_beta=0.5)
    assert m["clot_relaxed_rec"] == 1.0
    assert m["clot_relaxed_prec"] < 0.5
    assert m["clot_relaxed_f05"] < m["clot_relaxed_rec"]


def test_f_beta_weights_precision():
    p, r = 0.8, 0.4
    f1 = f_beta_score(p, r, beta=1.0)
    f05 = f_beta_score(p, r, beta=0.5)
    assert f05 > f1  # precision-heavy case: F0.5 > F1 when P > R


def test_guiding_score_blend():
    g = clot_guiding_score(0.6, 0.8)
    assert abs(g - 0.7) < 1e-6


def test_vacuous_empty_match_scores_one():
    n = 7
    ei = _chain_graph(n)
    pred = torch.zeros(n)
    gt = torch.zeros(n)
    m = compute_clot_relaxed_metrics(pred, gt, ei, relax_hops=2)
    assert m["clot_guiding"] == 1.0
    assert m["clot_dilation_iou"] == 1.0
    assert m["clot_relaxed_f05"] == 1.0
    assert m["clot_f1"] == 1.0
    assert m["clot_vacuous_match"] == 1.0
    assert m["clot_empty_gt"] == 1.0


def test_empty_gt_grades_by_false_positive_count():
    """Clot-free GT: a small blip must outrank a spray instead of both scoring 0."""
    n = 40
    ei = _chain_graph(n)
    gt = torch.zeros(n)

    blip = torch.zeros(n)
    blip[5] = 1.0
    spray = torch.zeros(n)
    spray[::2] = 1.0  # 20 nodes

    m_blip = compute_clot_relaxed_metrics(blip, gt, ei, relax_hops=2)
    m_spray = compute_clot_relaxed_metrics(spray, gt, ei, relax_hops=2)

    # Ordering is what matters: nothing > blip > spray, and none of them collapse to 0.
    assert m_blip["clot_guiding"] > m_spray["clot_guiding"]
    assert 0.0 < m_spray["clot_guiding"] < m_blip["clot_guiding"] < 1.0
    assert m_blip["clot_empty_gt"] == 1.0
    # Raw counts stay truthful for diagnostics.
    assert m_blip["clot_fp"] == 1.0
    assert m_blip["clot_gt_pos"] == 0.0
    assert m_spray["clot_fp"] == 20.0


def test_empty_gt_match_score_monotonic():
    assert empty_gt_match_score(0) == 1.0
    scores = [empty_gt_match_score(k) for k in (0, 1, 5, 20, 200)]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] < 0.1
    # tol is the half-way point by construction
    assert abs(empty_gt_match_score(8, tol=8.0) - 0.5) < 1e-9


def test_offwall_empty_gt_graded_penalizes_spray():
    """Wall-only GT clot: off-wall spray must score below a clean off-wall prediction."""
    n = 30
    ei = _chain_graph(n)
    wall = torch.zeros(n, dtype=torch.bool)
    wall[:10] = True

    gt = torch.zeros(n)
    gt[2] = 1.0  # GT clot on the wall only -> no off-wall GT

    clean = gt.clone()  # matches GT, predicts nothing off-wall
    sprayer = gt.clone()
    sprayer[15:25] = 1.0  # 10 off-wall false positives

    m_clean = compute_clot_relaxed_metrics(clean, gt, ei, relax_hops=2, wall_mask=wall)
    m_spray = compute_clot_relaxed_metrics(sprayer, gt, ei, relax_hops=2, wall_mask=wall)

    assert m_clean["offwall_empty_gt"] == 1.0
    assert m_clean["offwall_n_gt"] == 0.0
    assert m_clean["offwall_relaxed_f1"] == 1.0
    assert m_spray["offwall_relaxed_f1"] < m_clean["offwall_relaxed_f1"]
    assert m_spray["offwall_n_pred"] == 10.0
