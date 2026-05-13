from types import SimpleNamespace

import torch

from src.config import BiochemConfig
from src.core_physics.kinematics_clot_prior import clot_prior_features, clot_prior_score_flat


def _zero_sparse(n: int) -> torch.Tensor:
    idx = torch.empty((2, 0), dtype=torch.long)
    vals = torch.empty((0,), dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()


def _toy_data(sdf_values, wall_flags):
    sdf = torch.tensor(sdf_values, dtype=torch.float32)
    wall = torch.tensor(wall_flags, dtype=torch.float32)
    n = int(sdf.numel())
    x = torch.zeros(n, 15, dtype=torch.float32)
    x[:, 2] = sdf
    x[:, 7] = wall
    return SimpleNamespace(
        x=x,
        mask_wall=wall.bool(),
        G_x=_zero_sparse(n),
        G_y=_zero_sparse(n),
    )


def _props(n: int):
    return {
        "u_ref": torch.ones(n, 1, dtype=torch.float32),
        "d_bar": torch.ones(n, 1, dtype=torch.float32),
    }


def test_prior_uses_near_wall_decay_instead_of_uniform_bulk_floor(monkeypatch):
    monkeypatch.setenv("BIOCHEM_PRIOR_BULK_SCALE", "0.10")
    monkeypatch.setenv("BIOCHEM_PRIOR_MIN_FLOOR", "1e-4")
    monkeypatch.setenv("BIOCHEM_PRIOR_WALL_DECAY_ND", "0.18")

    data = _toy_data(sdf_values=[0.0, 0.10, 2.0], wall_flags=[1, 0, 0])
    u = torch.zeros(3)
    v = torch.zeros(3)

    prior = clot_prior_score_flat(data, u, v, BiochemConfig(phase="biochem"), _props(3))

    assert torch.all((prior >= 0.0) & (prior <= 1.0))
    assert prior[0] > prior[1] > prior[2]
    assert prior[2] < 1e-3


def test_prior_feature_channels_are_bounded_and_interpretable(monkeypatch):
    monkeypatch.setenv("BIOCHEM_PRIOR_BULK_SCALE", "0.10")
    monkeypatch.setenv("BIOCHEM_PRIOR_MIN_FLOOR", "1e-4")

    data = _toy_data(sdf_values=[0.0, 0.05, 1.0, 2.0], wall_flags=[1, 0, 0, 0])
    u = torch.zeros(4)
    v = torch.zeros(4)

    feats = clot_prior_features(data, u, v, BiochemConfig(phase="biochem"), _props(4), n_features=5)

    assert feats.shape == (4, 5)
    assert torch.all((feats >= 0.0) & (feats <= 1.0))
    # Channel 0 is the full prior; channel 1 is the gated adhesion-flux proxy.
    assert torch.all(feats[:, 0] >= feats[:, 1])
    # All physics channels should inherit the wall/near-wall localization gate.
    assert torch.all(feats[0] > feats[-1])
