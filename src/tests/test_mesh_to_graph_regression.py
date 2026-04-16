"""Regression checks for mesh-to-graph data object contracts."""

from __future__ import annotations

import torch

from src.data_gen.lib.mesh_to_graph import MeshToGraphComplete
from src.data_gen.lib.mesh_to_graph_tier3 import MeshToGraphTier3


def _base_context(num_nodes: int = 3):
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long)
    return {
        "edge_index": edge_index,
        "edge_attr": torch.zeros((edge_index.shape[1], 3), dtype=torch.float32),
        "mask_inlet": torch.tensor([True, False, False], dtype=torch.bool),
        "mask_outlet": torch.tensor([False, False, True], dtype=torch.bool),
        "mask_wall": torch.tensor([False, True, False], dtype=torch.bool),
        "d_bar": 1.0,
        "u_ref": 1.0,
        "V": torch.zeros((edge_index.shape[1], 5), dtype=torch.float32),
        "W": torch.ones(edge_index.shape[1], dtype=torch.float32),
        "M_inv": torch.eye(5, dtype=torch.float32).unsqueeze(0).repeat(num_nodes, 1, 1),
        "outlet_normal": torch.zeros((num_nodes, 2), dtype=torch.float32),
        "num_nodes": num_nodes,
    }


def test_tier12_data_object_drops_sparse_gradient_matrices(tmp_path):
    builder = MeshToGraphComplete(tier="tier1", raw_dir=tmp_path, label_dir=tmp_path, proc_dir=tmp_path)
    context = _base_context(num_nodes=3)
    priors = {
        "u_prior": torch.zeros(3, dtype=torch.float32),
        "v_prior": torch.zeros(3, dtype=torch.float32),
        "mu_prior": torch.ones(3, dtype=torch.float32),
        "wss_prior": torch.zeros(3, dtype=torch.float32),
    }
    x = torch.zeros((3, 15), dtype=torch.float32)
    y = torch.zeros((3, 5), dtype=torch.float32)

    data = builder._build_data_object(context, priors, x, y, is_anchor=False)

    assert hasattr(data, "V")
    assert hasattr(data, "W")
    assert hasattr(data, "M_inv")
    assert not hasattr(data, "G_x")
    assert not hasattr(data, "G_y")
    assert not hasattr(data, "Laplacian")


def test_tier3_non_anchor_data_object_has_transient_targets_without_sparse_derivatives(tmp_path):
    builder = MeshToGraphTier3(raw_dir=tmp_path, label_dir=tmp_path, proc_dir=tmp_path)
    context = _base_context(num_nodes=3)
    priors = {
        "u_prior": torch.tensor([1.0, 0.2, 0.0], dtype=torch.float32),
        "v_prior": torch.tensor([0.0, 0.1, 0.0], dtype=torch.float32),
        "mu_prior": torch.ones(3, dtype=torch.float32),
        "wss_prior": torch.zeros(3, dtype=torch.float32),
    }
    x = torch.zeros((3, 15), dtype=torch.float32)
    y = torch.zeros((3, 5), dtype=torch.float32)

    data = builder._build_data_object(context, priors, x, y, is_anchor=False)

    assert hasattr(data, "t")
    assert data.y.dim() == 3  # [T, N, C]
    assert data.y.shape[1] == 3
    assert data.u_inlet_bc.shape[1] == 2
    assert hasattr(data, "bio_inlet_bc")
    assert not hasattr(data, "G_x")
    assert not hasattr(data, "G_y")
    assert not hasattr(data, "Laplacian")


def test_tier3_anchor_data_object_is_steady_state_shape(tmp_path):
    builder = MeshToGraphTier3(raw_dir=tmp_path, label_dir=tmp_path, proc_dir=tmp_path)
    context = _base_context(num_nodes=3)
    priors = {
        "u_prior": torch.zeros(3, dtype=torch.float32),
        "v_prior": torch.zeros(3, dtype=torch.float32),
        "mu_prior": torch.ones(3, dtype=torch.float32),
        "wss_prior": torch.zeros(3, dtype=torch.float32),
    }
    x = torch.zeros((3, 15), dtype=torch.float32)
    y = torch.zeros((3, 5), dtype=torch.float32)

    data = builder._build_data_object(context, priors, x, y, is_anchor=True)

    assert data.y.shape == (3, 5)
    assert data.is_anchor.shape == (1,)
