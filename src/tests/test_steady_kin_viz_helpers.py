"""Helpers for steady kinematics visualization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.evaluation.visualize_pipeline import (
    _graph_has_comsol_trajectory,
    _rel_l2_uvp,
    _steady_kine_target_tensor,
)


def test_steady_kine_target_3d_and_2d():
    n, t = 8, 3
    y3 = torch.randn(t, n, 16)
    out = _steady_kine_target_tensor(Data(y=y3), time_index=-1)
    assert out is not None and tuple(out.shape) == (n, 5)
    y2 = torch.randn(n, 5)
    out2 = _steady_kine_target_tensor(Data(y=y2), time_index=0)
    assert out2 is not None and tuple(out2.shape) == (n, 5)


def test_graph_has_comsol_trajectory():
    assert _graph_has_comsol_trajectory(Data(y=torch.ones(10, 5)))
    assert not _graph_has_comsol_trajectory(Data(y=torch.zeros(10, 5)))


def test_rel_l2_uvp():
    pred = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    tgt = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert _rel_l2_uvp(pred, tgt) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-3)
