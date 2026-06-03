"""Anchor kinematics prior refresh (centerline + Poiseuille, same as synthetic)."""
from __future__ import annotations

import torch

from src.data_gen.lib.node_feature_assembly import (
    kinematics_uv_prior_max,
    refresh_kinematics_node_x_on_graph,
)
from src.tests.test_anchor_dual_x_schema import _minimal_anchor_graph


def test_refresh_raises_uv_prior_with_inlet_outlet_centerline():
    data = _minimal_anchor_graph(n=24)
    data.y = torch.zeros(2, 24, 16)
    data.x[:, 11:13] = 0.0
    assert kinematics_uv_prior_max(data.x) < 1e-6
    ok = refresh_kinematics_node_x_on_graph(data, force=True)
    assert ok
    assert kinematics_uv_prior_max(data.x) > 0.1
