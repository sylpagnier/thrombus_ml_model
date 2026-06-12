"""Clot growth loss unit tests."""

from __future__ import annotations

import numpy as np
import torch

from src.training.clot_growth_loss import (
    ClotGrowthLossConfig,
    clot_growth_frame_loss,
    hop_penalty_from_gt_clot,
    soft_tversky_loss,
    volume_hinge_loss,
)


def test_soft_tversky_perfect_match_beats_mismatch():
    pred = torch.tensor([0.9, 0.1, 0.8, 0.0])
    target = pred.clone()
    mask = torch.ones(4, dtype=torch.bool)
    good = soft_tversky_loss(pred, target, mask)
    bad = soft_tversky_loss(pred, 1.0 - target, mask)
    assert float(good.item()) < float(bad.item())


def test_hop_fp_penalizes_distant_false_clot_more():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    gt = np.array([True, False, False, False])
    hop = hop_penalty_from_gt_clot(edge_index, 4, gt)
    assert hop[0] < 0.1
    assert hop[3] > hop[1]


def test_volume_hinge_only_on_overpaint():
    pred = torch.tensor([0.8, 0.7, 0.1, 0.1])
    target = torch.tensor([0.2, 0.2, 0.0, 0.0])
    mask = torch.ones(4, dtype=torch.bool)
    over = volume_hinge_loss(pred, target, mask, margin_frac=0.05)
    under_pred = torch.tensor([0.05, 0.05, 0.0, 0.0])
    under = volume_hinge_loss(under_pred, target, mask, margin_frac=0.05)
    assert float(over.item()) > float(under.item())


def test_composite_loss_finite():
    pred = torch.rand(8)
    target = (torch.rand(8) > 0.7).float()
    mask = torch.ones(8, dtype=torch.bool)
    hop = torch.linspace(0, 1, 8)
    out = clot_growth_frame_loss(pred, target, mask, hop, cfg=ClotGrowthLossConfig())
    assert float(out["loss"].item()) == float(out["loss"].item())
    assert float(out["loss"].item()) > 0.0
