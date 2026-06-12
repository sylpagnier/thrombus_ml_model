"""Hop-growing clot support masks (t0 wall, ceiling, time growth)."""

from __future__ import annotations

import os

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import (
    graph_dilate_hops,
    resolve_ceiling_mask,
    resolve_growth_support_at_time,
    resolve_t0_dgamma_wall_mask,
)
from src.core_physics.clot_phi_simple import apply_clot_support_projection


def _chain_graph(n: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    cols = []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    x = torch.zeros(n, 2)
    x[:, 0] = torch.linspace(0.0, 1.0, n)
    y = torch.zeros(n, 20, 17)
    y[:, 0] = 1.0
    mask_wall = torch.zeros(n, dtype=torch.bool)
    mask_wall[0] = True
    data = type(
        "G",
        (),
        {
            "num_nodes": n,
            "edge_index": edge_index,
            "x": x,
            "y": y.unsqueeze(0),
            "mask_wall": mask_wall,
            "u_ref": torch.tensor([1.0]),
            "d_bar": torch.tensor([1.0]),
            "G_x": torch.sparse_coo_tensor(torch.zeros(2, 0), [], (n, 1)),
            "G_y": torch.sparse_coo_tensor(torch.zeros(2, 0), [], (n, 1)),
        },
    )()
    return data, edge_index


def test_graph_dilate_hops_expands_chain():
    _, ei = _chain_graph(5)
    seed = torch.tensor([True, False, False, False, False])
    out = graph_dilate_hops(seed, ei, 2)
    assert out[0].item() and out[1].item() and out[2].item()
    assert not out[3].item() and not out[4].item()


def test_t0_mask_is_subset_of_ceiling(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "0")
    data, _ = _chain_graph(6)
    device = torch.device("cpu")
    bio = BiochemConfig(phase="biochem")
    t0 = resolve_t0_dgamma_wall_mask(data, device, bio)
    ceiling = resolve_ceiling_mask(data, device, bio, ceiling_hops=3)
    assert bool((t0 & ~ceiling).any().item()) is False
    assert int(ceiling.sum()) >= int(t0.sum())


def test_growth_support_monotone_and_capped(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
    monkeypatch.setenv("CLOT_PHI_CEILING_HOPS", "5")
    monkeypatch.setenv("CLOT_PHI_GROWTH_SEED", "gt")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "1")
    from src.utils.paths import get_project_root

    graph_path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not graph_path.is_file():
        pytest.skip("patient007 anchor missing")
    data = torch.load(graph_path, weights_only=False)
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    ceiling = resolve_ceiling_mask(data, device, bio, ceiling_hops=5)
    prev = None
    for t in (0, 1, 5, 20, 53):
        support = resolve_growth_support_at_time(data, t, device, phys, bio, growth_seed="gt")
        assert bool((support & ~ceiling).any().item()) is False
        if prev is not None:
            assert bool((prev & ~support).any().item()) is False
        prev = support


def test_projection_uses_bulk_at_target_time_not_t_in(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
    monkeypatch.setenv("CLOT_PHI_SUPPORT_BAND", "physics")
    # Off-band bulk: t0 Carreau wrongly high (old deploy bug) vs t_out Carreau low.
    mu_c_t0 = torch.tensor([0.04, 0.04, 0.10, 0.10], dtype=torch.float32)
    mu_c_tout = torch.tensor([0.04, 0.04, 0.04, 0.04], dtype=torch.float32)
    mu_model = torch.tensor([0.10, 0.10, 0.10, 0.10], dtype=torch.float32)
    band = torch.tensor([True, True, False, False])
    out_old = apply_clot_support_projection(mu_c_t0, mu_model, band)
    out_new = apply_clot_support_projection(mu_c_tout, mu_model, band)
    assert out_old[2].item() == pytest.approx(0.10)
    assert out_new[2].item() == pytest.approx(0.04)
