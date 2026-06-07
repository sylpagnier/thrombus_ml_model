"""Tests for shared kinematics rel-L2 helper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.utils.metrics import rel_l2_uvp


def test_rel_l2_uvp_matches_kinematics_convention():
    pred = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    true = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    assert rel_l2_uvp(pred, true) == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-5)


def test_rel_l2_uvp_respects_node_mask():
    pred = torch.tensor([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=torch.float32)
    true = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([True, False])
    assert rel_l2_uvp(pred, true, node_mask=mask) == pytest.approx(0.0, abs=1e-6)
