"""Unit test: freeze_growth_backbone keeps only dual heads trainable."""

from __future__ import annotations

import torch

from src.core_physics.species_pushforward_continuous import SpeciesDualHeadContinuousGNN
from src.training.train_offwall_growth import freeze_growth_backbone


def test_freeze_growth_backbone_trains_heads_only():
    model = SpeciesDualHeadContinuousGNN(in_dim=64, hidden=32, out_dim=1)
    n_fr, n_tr = freeze_growth_backbone(model)
    assert n_fr > 0
    assert n_tr > 0
    for name, p in model.named_parameters():
        is_head = name.startswith("spatial_head") or name.startswith("magnitude_head")
        assert p.requires_grad is is_head, name
    # Optimizer would see only head params
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) == n_tr
    loss = sum(p.sum() for p in trainable)
    loss.backward()
    assert any(p.grad is not None for p in trainable)
