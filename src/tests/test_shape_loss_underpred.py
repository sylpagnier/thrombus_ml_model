"""Shape-loss under-prediction tilt for growth specialist."""

from __future__ import annotations

import torch

from src.training.train_offwall_growth import compute_shape_loss


def test_loss_lumen_shape_underpred_increases_when_missed_mass(monkeypatch):
    monkeypatch.setenv("SPECIES_CONTINUOUS_UNDERPRED_WEIGHT", "4.0")
    monkeypatch.setenv("SPECIES_LUMEN_SHAPE_FN_W", "5")
    monkeypatch.setenv("SPECIES_LUMEN_SHAPE_FP_W", "2.5")
    monkeypatch.setenv("SPECIES_CONTINUOUS_DELTA_VALUE_SCALE", "1")
    monkeypatch.setenv("SPECIES_CONTINUOUS_DELTA_THRESH", "0.1")
    monkeypatch.setenv("SPECIES_CONTINUOUS_HUBER_BETA", "1.0")

    # 4 nodes; edge chain 0-1-2-3
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    mask = torch.ones(4, dtype=torch.bool)
    hop = torch.tensor([0, 1, 2, 2], dtype=torch.float32)
    # Strong GT growth on lumen nodes; weak pred -> underpred
    tgt = torch.tensor([0.0, 0.0, 1.0, 1.0])
    pred_weak = torch.tensor([0.0, 0.0, 0.0, 0.0])
    pred_match = torch.tensor([0.0, 0.0, 1.0, 1.0])

    loss_weak = compute_shape_loss(
        pred_weak,
        tgt,
        mask,
        edge_index,
        4,
        "loss_lumen_shape",
        hop_dist=hop,
        lumen_shape_weight=4.0,
    )
    loss_match = compute_shape_loss(
        pred_match,
        tgt,
        mask,
        edge_index,
        4,
        "loss_lumen_shape",
        hop_dist=hop,
        lumen_shape_weight=4.0,
    )
    assert float(loss_weak.item()) > float(loss_match.item())
