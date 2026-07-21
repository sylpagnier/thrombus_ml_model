"""Tests for growth-specialist loss / checkpoint helpers (Arm C recipe)."""

from __future__ import annotations

import torch

from src.training.train_offwall_growth import (
    compute_shape_loss,
    growth_specialist_ckpt_score,
)


def test_ckpt_score_offwall_balanced_prefers_volume_match():
    low_vol = growth_specialist_ckpt_score(
        ckpt_metric="offwall_balanced",
        clot_score=0.9,
        offwall_relaxed_f1=0.8,
        offwall_n_pred=1.0,
        offwall_n_gt=20.0,
    )
    good_vol = growth_specialist_ckpt_score(
        ckpt_metric="offwall_balanced",
        clot_score=0.5,
        offwall_relaxed_f1=0.8,
        offwall_n_pred=18.0,
        offwall_n_gt=20.0,
    )
    assert good_vol > low_vol


def test_ckpt_score_relaxed_ignores_volume():
    a = growth_specialist_ckpt_score(
        ckpt_metric="offwall_relaxed",
        clot_score=0.1,
        offwall_relaxed_f1=0.55,
        offwall_n_pred=1.0,
        offwall_n_gt=99.0,
    )
    assert abs(a - 0.55) < 1e-6


def test_ckpt_score_legacy_clot():
    s = growth_specialist_ckpt_score(
        ckpt_metric="clot_score",
        clot_score=0.77,
        offwall_relaxed_f1=0.9,
        offwall_n_pred=50.0,
        offwall_n_gt=20.0,
    )
    assert abs(s - 0.77) < 1e-6


def test_ckpt_score_hop_ge2_balanced_prefers_localization():
    low = growth_specialist_ckpt_score(
        ckpt_metric="hop_ge2_balanced",
        clot_score=0.9,
        offwall_relaxed_f1=0.9,
        offwall_n_pred=20.0,
        offwall_n_gt=20.0,
        hop_ge2_strict_f1=0.05,
        hop_ge2_n_pred=50.0,
        hop_ge2_n_gt=20.0,
    )
    high = growth_specialist_ckpt_score(
        ckpt_metric="hop_ge2_balanced",
        clot_score=0.5,
        offwall_relaxed_f1=0.5,
        offwall_n_pred=5.0,
        offwall_n_gt=20.0,
        hop_ge2_strict_f1=0.40,
        hop_ge2_n_pred=18.0,
        hop_ge2_n_gt=20.0,
    )
    assert high > low


def test_loss_blurring_prec_penalizes_far_fp(monkeypatch):
    monkeypatch.setenv("SPECIES_CONTINUOUS_DELTA_VALUE_SCALE", "1.0")
    monkeypatch.setenv("SPECIES_CONTINUOUS_HUBER_BETA_GROWTH", "1.0")
    monkeypatch.setenv("SPECIES_CONTINUOUS_DELTA_THRESHOLD", "0.1")
    # Line graph 0-1-2-3-4
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    n = 5
    mask = torch.ones(n, dtype=torch.bool)
    # GT growth only at node 0; pred also fires far away at node 4
    tgt = torch.zeros(n, 1)
    tgt[0, 0] = 1.0
    pred_far = torch.zeros(n, 1)
    pred_far[0, 0] = 1.0
    pred_far[4, 0] = 1.0
    pred_near = torch.zeros(n, 1)
    pred_near[0, 0] = 1.0

    loss_far = compute_shape_loss(pred_far, tgt, mask, edge_index, n, "loss_blurring_prec")
    loss_near = compute_shape_loss(pred_near, tgt, mask, edge_index, n, "loss_blurring_prec")
    loss_blur_far = compute_shape_loss(pred_far, tgt, mask, edge_index, n, "loss_blurring")
    assert float(loss_far) > float(loss_near)
    assert float(loss_far) >= float(loss_blur_far) - 1e-5
