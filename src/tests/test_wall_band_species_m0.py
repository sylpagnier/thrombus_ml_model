"""Wall-band species M0 smoke tests."""

from __future__ import annotations

import torch

from src.core_physics.wall_band_species_m0 import (
    CHANNEL_SETS,
    build_m0_bundle,
    feature_dim,
    resolve_channel_set,
)


def test_channel_sets():
    fi, names, w_fi = resolve_channel_set("fimat")
    assert fi == [8, 11]
    assert names == ["FI", "Mat"]
    assert len(w_fi) == 2
    c4, n4, w4 = resolve_channel_set("cascade4")
    assert c4 == [2, 3, 8, 11]
    assert w4[2] > w4[0]
    assert len(CHANNEL_SETS) >= 2


def test_m0_gnn_zero_init():
    bundle = build_m0_bundle("fimat", torch.device("cpu"), hidden=16)
    assert bundle.in_dim == feature_dim(2)
    n = 20
    x = torch.randn(n, bundle.in_dim)
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    y = bundle.model(x, ei)
    assert y.shape == (n, 2)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)
