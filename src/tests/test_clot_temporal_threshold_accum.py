"""Smoke tests for threshold-accumulation temporal clot rule."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    eval_temporal_rule_on_anchor,
    rollout_temporal_phi,
    threshold_accum_rule_grid,
)
from src.utils.paths import get_project_root


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 anchor missing")
    return torch.load(path, map_location="cpu", weights_only=False)


def test_threshold_accum_grid_nonempty():
    rules = threshold_accum_rule_grid()
    assert len(rules) >= 6
    assert all(r.kind == "threshold_accum" for r in rules)


def test_threshold_accum_rollout_monotone(patient007, monkeypatch):
    monkeypatch.setenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    monkeypatch.setenv("CLOT_SHAPE_EVAL_MASK", "ceiling")
    rule = threshold_accum_rule_grid()[0]
    device = torch.device("cpu")
    phi_by_t = rollout_temporal_phi(
        patient007,
        rule,
        device=device,
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    counts = [int((phi > 0.5).sum().item()) for phi in phi_by_t.values()]
    assert counts == sorted(counts)


def test_offset_incubation_zeros_early_flags(patient007, monkeypatch):
    monkeypatch.setenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    monkeypatch.setenv("CLOT_SHAPE_EVAL_MASK", "ceiling")
    from dataclasses import replace

    from src.core_physics.clot_temporal_growth_rules import _localized_prog_template

    base = _localized_prog_template()
    rule = replace(base, name="test_inc40", global_onset_frac=0.40)
    device = torch.device("cpu")
    phi_by_t = rollout_temporal_phi(
        patient007,
        rule,
        device=device,
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    early = [int((phi_by_t[i] > 0.5).sum().item()) for i in range(0, 8)]
    assert max(early) == 0


def test_eval_reports_timeline_and_tfinal_shape(patient007, monkeypatch):
    monkeypatch.setenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    monkeypatch.setenv("CLOT_SHAPE_EVAL_MASK", "ceiling")
    rule = threshold_accum_rule_grid()[0]
    device = torch.device("cpu")
    row = eval_temporal_rule_on_anchor(
        patient007,
        rule,
        stem="patient007",
        device=device,
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert row["n_pairs"] > 0
    assert row["n_timeline_shape"] == row["n_pairs"]
    assert row["mean_clot_shape"] == row["mean_clot_shape"]
    assert row["tfinal_clot_shape"] == row["tfinal_clot_shape"]
