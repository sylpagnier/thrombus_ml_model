"""Deploy Leg B neighbor mask (no GT clot seeds in commit gate)."""

from __future__ import annotations

import pytest
import torch

from src.config import PhysicsConfig
from src.core_physics.clot_phi_mu_inject import (
    mlp_deploy_no_commit_at_t0,
    mlp_mu_map_uses_gt_labels,
    resolve_clot_trigger_gate,
    resolve_deploy_neighbor_commit_mask,
)


class _ChainGraph:
    num_nodes = 4
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    mask_wall = torch.tensor([True, True, False, False])


def test_neighbor_mask_no_gt_labels(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "neighbor")
    assert not mlp_mu_map_uses_gt_labels()


def test_deploy_neighbor_wall_and_pred_clot_hop(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_NEIGHBOR_SEED", "pred_clot")
    monkeypatch.setenv("BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI", "0")
    monkeypatch.setenv("CLOT_SHAPE_MU_THRESH_SI", "0.055")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    data = _ChainGraph()
    phi = torch.tensor([0.1, 0.1, 0.1, 0.1])
    prev = torch.tensor([0.04, 0.06, 0.04, 0.04])
    gate = resolve_deploy_neighbor_commit_mask(data, device, phi=phi, prev_mu_eff_si=prev, phys_cfg=phys)
    assert bool(gate[1])
    assert int(gate.sum().item()) >= 1


def test_deploy_no_commit_at_macro_t0(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP", "1")
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "neighbor")
    monkeypatch.setenv("BIOCHEM_MLP_DEPLOY_NO_COMMIT_T0", "1")
    assert mlp_deploy_no_commit_at_t0()


def test_neighbor_commit_intersects_supervision_vision(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "neighbor")
    monkeypatch.setenv("BIOCHEM_MLP_NEIGHBOR_SEED", "pred_clot")
    monkeypatch.setenv("BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI", "0")
    monkeypatch.setenv("CLOT_SHAPE_MU_THRESH_SI", "0.055")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    data = _ChainGraph()
    phi = torch.tensor([0.1, 0.1, 0.1, 0.1])
    mu_c = torch.full((4, 1), 0.04)
    mu_mlp = torch.full((4, 1), 0.10)
    prev = torch.tensor([0.04, 0.08, 0.04, 0.04])
    allowed = torch.tensor([True, True, False, False])
    gate = resolve_clot_trigger_gate(
        phi,
        mu_c,
        mu_mlp,
        phys_cfg=phys,
        graph_data=data,
        prev_mu_eff_si=prev,
        allowed_commit_mask=allowed,
    )
    assert bool(gate[1])
    assert not bool(gate[2])


def test_resolve_clot_trigger_gate_neighbor_uses_prev_mu(monkeypatch):
    monkeypatch.setenv("BIOCHEM_MLP_MU_MAP_MASK", "neighbor")
    monkeypatch.setenv("BIOCHEM_MLP_NEIGHBOR_SEED", "pred_clot")
    monkeypatch.setenv("BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI", "0")
    monkeypatch.setenv("CLOT_SHAPE_MU_THRESH_SI", "0.055")
    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    data = _ChainGraph()
    phi = torch.zeros(4)
    mu_c = torch.full((4, 1), 0.04)
    mu_mlp = torch.full((4, 1), 0.10)
    prev = torch.tensor([0.04, 0.08, 0.04, 0.04])
    gate = resolve_clot_trigger_gate(
        phi,
        mu_c,
        mu_mlp,
        phys_cfg=phys,
        graph_data=data,
        prev_mu_eff_si=prev,
    )
    assert int(gate.sum().item()) >= 1
