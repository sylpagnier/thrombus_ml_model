"""Mesh aux config + lumen eligible mask smoke tests."""

from __future__ import annotations

import torch

from src.core_physics.clot_phi_simple import (
    clot_phi_mesh_aux_lambda,
    clot_phi_shape_use_t_out_mu,
    lumen_eligible_mask,
)


def test_mesh_aux_env(monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MESH_AUX_LAMBDA", "0.5")
    monkeypatch.setenv("CLOT_PHI_MESH_BULK_LAMBDA", "0.2")
    assert clot_phi_mesh_aux_lambda() == 0.5
    monkeypatch.setenv("CLOT_PHI_SHAPE_USE_T_OUT", "0")
    assert clot_phi_shape_use_t_out_mu() is False


def test_lumen_eligible_mask_smoke():
    class _G:
        num_nodes = 4
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        mask_wall = torch.tensor([True, True, False, False])

    m = lumen_eligible_mask(_G(), torch.device("cpu"))
    assert m.shape == (4,)
    assert bool(m[:2].all())
