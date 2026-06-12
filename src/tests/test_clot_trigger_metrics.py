"""Unit tests for clot trigger F1 metrics."""

from __future__ import annotations

import torch

from src.evaluation.clot_trigger_metrics import (
    clot_trigger_metric_bundle,
    f1_scaled_above_baseline,
    f1_zero_prediction_baseline,
)


def test_f1_zero_baseline_is_zero_when_gt_has_positives() -> None:
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])
    mask = torch.ones(4, dtype=torch.bool)
    assert f1_zero_prediction_baseline(target, mask) == 0.0


def test_f1_scaled_maps_baseline_to_zero() -> None:
    assert f1_scaled_above_baseline(0.0, 0.0) == 0.0
    assert abs(f1_scaled_above_baseline(1.0, 0.0) - 1.0) < 1e-6
    assert abs(f1_scaled_above_baseline(0.5, 0.25) - (1.0 / 3.0)) < 1e-5


def test_metric_bundle_includes_scaled_full_mesh() -> None:
    pred = torch.tensor([0.0, 1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])
    m = clot_trigger_metric_bundle(pred, target)
    assert m["full_mesh_f1_baseline_zero"] == 0.0
    assert m["full_mesh_f1_scaled"] == m["full_mesh_f1"]
    assert m["full_mesh_f1"] > 0.0
