"""Rung 3 flow_source wiring (pred kine + GT species)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import torch

from src.core_physics.t0_mu_physics import resolve_t0_flow_uv_nd
from src.core_physics.t0_rung_config import t0_rung3_env
from src.utils.paths import get_project_root


def _tiny_graph() -> mock.MagicMock:
    n = 4
    y = torch.zeros(2, n, 16)
    y[:, :, 0] = torch.tensor([1.0, 0.5, -0.2, 0.1])
    y[:, :, 1] = torch.tensor([0.0, 0.3, 0.4, -0.1])
    data = mock.MagicMock()
    data.y = y
    data.num_nodes = n
    data.edge_index = torch.zeros(2, 2, dtype=torch.long)
    data.x = torch.randn(n, 3)
    return data


def test_resolve_t0_flow_gt_matches_y_channels():
    data = _tiny_graph()
    device = torch.device("cpu")
    u, v = resolve_t0_flow_uv_nd(data, 0, device, flow_source="gt")
    assert torch.allclose(u, data.y[0, :, 0].reshape(-1))
    assert torch.allclose(v, data.y[0, :, 1].reshape(-1))


def test_t0_rung3_env_sets_kinematics_flags():
    with t0_rung3_env(kine_ckpt="outputs/kinematics/kinematics_best.pth") as cfg:
        assert os.environ.get("CLOT_PHI_VEL_SOURCE") == "kinematics"
        assert os.environ.get("CLOT_TEMPORAL_VEL_SOURCE") == "kinematics"
        assert cfg["flow_source"] == "kinematics"
    assert os.environ.get("CLOT_PHI_VEL_SOURCE") is None or os.environ.get("CLOT_PHI_VEL_SOURCE") != "kinematics"


@pytest.mark.skipif(
    not (get_project_root() / "outputs/kinematics/kinematics_best.pth").is_file(),
    reason="kinematics_best.pth not present",
)
def test_rung3_pred_flow_differs_from_gt_on_patient007():
    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not graph.is_file():
        pytest.skip("patient007 graph missing")
    data = torch.load(graph, map_location="cpu", weights_only=False)
    device = torch.device("cpu")
    kine = str(root / "outputs/kinematics/kinematics_best.pth")
    with t0_rung3_env(kine_ckpt=kine):
        u_pred, v_pred = resolve_t0_flow_uv_nd(data, 0, device, flow_source="kinematics")
    u_gt, v_gt = resolve_t0_flow_uv_nd(data, 0, device, flow_source="gt")
    rel = float((u_pred - u_gt).pow(2).add((v_pred - v_gt).pow(2)).mean().sqrt().item())
    assert rel > 1e-4
