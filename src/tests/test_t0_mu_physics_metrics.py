"""Tests for viscosity calibration helpers in t0_mu_physics."""

from __future__ import annotations

import torch

from src.core_physics.t0_mu_physics import _mu_log_mae, _pearson, _region_masks


def test_mu_log_mae_identity_is_zero():
    mu = torch.tensor([0.01, 0.02, 0.05], dtype=torch.float32)
    assert _mu_log_mae(mu, mu) == 0.0


def test_pearson_perfect_correlation():
    mu = torch.tensor([0.01, 0.02, 0.04, 0.08], dtype=torch.float32)
    assert _pearson(mu, 2.0 * mu) > 0.999


def test_region_masks_keys():
    from src.config import PhysicsConfig
    from pathlib import Path
    import torch

    root = Path(__file__).resolve().parents[2]
    p = root / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not p.is_file():
        return
    data = torch.load(p, map_location="cpu", weights_only=False)
    phys = PhysicsConfig()
    dev = torch.device("cpu")
    t = int(data.y.shape[0]) - 1
    mu_gt = torch.full((int(data.num_nodes),), 0.01, dtype=torch.float32)
    masks = _region_masks(data, t, phys, dev, mu_gt)
    assert set(masks.keys()) == {"growth", "wall"}
    assert masks["growth"].shape == (int(data.num_nodes),)
