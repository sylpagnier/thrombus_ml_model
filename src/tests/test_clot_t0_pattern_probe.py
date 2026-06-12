"""Smoke test for t=0 clot pattern probe."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.core_physics.clot_t0_pattern_probe import (
    aggregate_feature_rankings,
    probe_all_anchors,
    probe_anchor_patterns,
)
from src.utils.paths import get_project_root


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 anchor missing")
    return torch.load(path, map_location="cpu", weights_only=False)


def test_probe_patient007_has_clots_in_ceiling(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CEILING_HOPS", "2")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "1")
    monkeypatch.setenv("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    rep = probe_anchor_patterns(patient007, stem="patient007", ceiling_hops=2)
    assert rep.n_clot_ceiling >= 100
    assert rep.clot_recall_in_ceiling >= 0.90
    assert len(rep.feature_rows) >= 10
    assert len(rep.rule_rows) >= 3


def test_aggregate_rankings_nonempty(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CEILING_HOPS", "2")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "1")
    reports = [probe_anchor_patterns(patient007, stem="patient007", ceiling_hops=2)]
    agg = aggregate_feature_rankings(reports)
    assert agg
    assert agg[0]["mean_auc"] >= 0.5


def test_probe_all_anchors_runs():
    anchor_dir = get_project_root() / "data/processed/graphs_biochem_anchors"
    if not list(anchor_dir.glob("patient*.pt")):
        pytest.skip("no anchors")
    reports = probe_all_anchors(anchor_dir, ceiling_hops=2)
    assert len(reports) >= 1
