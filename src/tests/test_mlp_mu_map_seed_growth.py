"""Seed-growth deploy commit mask (GT t=0 vision + pred-clot 1-hop expansion)."""

from __future__ import annotations

import pytest
import torch

from src.config import PhysicsConfig
from src.core_physics.clot_phi_mu_inject import (
    expand_seed_growth_allowed_mask,
    init_seed_growth_allowed_mask,
    mlp_mu_map_mask_mode,
    normalize_mlp_mu_map_mask_mode,
    resolve_deploy_seed_growth_commit_mask,
)


class _ChainGraph:
    num_nodes = 4
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    mask_wall = torch.tensor([True, True, False, False])

    y = torch.zeros(2, 4, 16)
    x = torch.zeros(4, 2)


def test_seed_growth_mask_mode_normalized():
    assert normalize_mlp_mu_map_mask_mode("seed_growth") == "seed_growth"


def test_seed_growth_commit_inside_allowed_only(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "seed_growth")
    monkeypatch.setenv("BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI", "0")
    device = torch.device("cpu")
    allowed = torch.tensor([True, True, False, False])
    phi = torch.tensor([0.1, 0.9, 0.9, 0.1])
    gate = resolve_deploy_seed_growth_commit_mask(allowed, phi, device=device)
    assert gate.tolist() == [True, True, False, False]


def test_seed_growth_expands_one_hop_after_pred_clot(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_SEED_GROWTH_HOPS", "1")
    monkeypatch.setenv("CLOT_SHAPE_MU_THRESH_SI", "0.055")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    data = _ChainGraph()
    allowed = torch.tensor([True, False, False, False])
    mu = torch.tensor([0.08, 0.04, 0.04, 0.04])
    out = expand_seed_growth_allowed_mask(allowed, mu, data, device, phys_cfg=phys)
    assert bool(out[0])
    assert bool(out[1])
    assert not bool(out[2])
