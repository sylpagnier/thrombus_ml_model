"""Tests for neighbor-band species scope and trigger eval helpers."""

from __future__ import annotations

import os

import pytest
import torch

from src.training.biochem_species_scope import (
    data_bio_species_channel_indices,
    slice_bio_species_channels,
)


def test_fi_mat_scope_channels() -> None:
    os.environ["BIOCHEM_DATA_BIO_SPECIES_SCOPE"] = "fi_mat"
    assert data_bio_species_channel_indices() == [8, 11]


def test_all_scope_channels() -> None:
    os.environ["BIOCHEM_DATA_BIO_SPECIES_SCOPE"] = "all"
    assert data_bio_species_channel_indices() == list(range(12))


def test_slice_bio_species_channels() -> None:
    os.environ["BIOCHEM_DATA_BIO_SPECIES_SCOPE"] = "fi_mat"
    x = torch.randn(2, 10, 12)
    y = slice_bio_species_channels(x)
    assert y.shape == (2, 10, 2)


def test_neighbor_mask_mode_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.biochem_supervision_masks import resolve_data_bio_supervision_mask

    monkeypatch.setenv("BIOCHEM_DATA_BIO_MASK_MODE", "neighbor")
    monkeypatch.setenv("CLOT_PHI_MASK_MODE", "neighbor")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "0")

    n = 5
    device = torch.device("cpu")
    from types import SimpleNamespace

    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x_feat = torch.zeros(n, 4)
    x_feat[:, 2] = torch.linspace(0.0, 0.04, n)
    wall = torch.zeros(n, dtype=torch.bool)
    wall[0] = True
    data = SimpleNamespace(num_nodes=n, edge_index=ei, x=x_feat, mask_wall=wall)
    truth = torch.ones(n, dtype=torch.bool)
    # High mu at wall node -> clot seed
    target = torch.zeros(1, n, 16)
    target[0, 0, 3] = 100.0  # mu_eff_nd placeholder (mask uses SI cap path)

    class _Kernels:
        class core:
            cfg = __import__("src.config", fromlist=["PhysicsConfig"]).PhysicsConfig()

    bio_cfg = __import__("src.config", fromlist=["BiochemConfig"]).BiochemConfig(phase="biochem")
    m = resolve_data_bio_supervision_mask(
        data=data,
        device=device,
        truth_mask=truth,
        target_series=target,
        bio_cfg=bio_cfg,
        kernels=_Kernels(),
    )
    assert m[0].item() is True
    assert int(m.sum().item()) >= 1


def test_physics_trigger_mu_shape() -> None:
    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.clot_phi_simple import physics_mu_eff_si

    os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = "4"
    n = 8
    device = torch.device("cpu")
    bio_cfg = BiochemConfig(phase="biochem")
    mu_c = torch.full((n,), 0.004, dtype=torch.float32)
    species = torch.zeros(n, 12)
    species[:, 8] = 2.0
    species[:, 11] = 1.5
    mu = physics_mu_eff_si(mu_c, species, bio_cfg, device=device)
    assert mu.shape == (n,)
    assert float(mu.max()) > float(mu_c.max())
