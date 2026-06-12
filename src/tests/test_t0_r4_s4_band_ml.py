"""Rung 4 s4 band GNN wiring."""

from __future__ import annotations

import torch

from src.core_physics.t0_r4_s4_band_ml import (
    T0R4S4BandGNN,
    s4_feature_dim,
    save_s4_checkpoint,
    load_s4_bundle,
)
from src.core_physics.t0_rung4_ladder import rung4_step_from_env


def test_s4_alias():
    import os

    os.environ["T0_RUNG4_STEP"] = "s4"
    assert rung4_step_from_env() == "s4_band_ml"


def test_s4_gnn_zero_init():
    m = T0R4S4BandGNN(in_dim=s4_feature_dim(), hidden=16)
    n = 20
    x = torch.randn(n, s4_feature_dim())
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    logit = m(x, ei)
    assert logit.shape == (n,)
    assert torch.allclose(logit, torch.zeros_like(logit), atol=1e-6)


def test_s4_ckpt_roundtrip(tmp_path):
    m = T0R4S4BandGNN(in_dim=s4_feature_dim(), hidden=16)
    ckpt = tmp_path / "s4.pth"
    save_s4_checkpoint(ckpt, m, loc_scale=0.75, hidden=16, meta={"epoch": 1})
    bundle = load_s4_bundle(ckpt, device=torch.device("cpu"), quiet=True)
    assert bundle is not None
    assert bundle.loc_scale == 0.75
