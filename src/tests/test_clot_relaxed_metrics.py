"""Tests for relaxed clot guiding metrics."""

from __future__ import annotations

import torch

from src.evaluation.clot_relaxed_metrics import (
    clot_guiding_score,
    compute_clot_relaxed_metrics,
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
