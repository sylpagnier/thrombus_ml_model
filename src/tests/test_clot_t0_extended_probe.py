"""Smoke test for extended clot t0 probe."""

from __future__ import annotations

import pytest
import torch

from src.core_physics.clot_t0_extended_probe import (
    aggregate_extended_rows,
    build_feature_table_at_time,
    probe_anchor_extended,
)
from src.config import BiochemConfig, PhysicsConfig
from src.utils.paths import get_project_root


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return torch.load(path, map_location="cpu", weights_only=False)


def test_extended_table_has_bio_x_and_topology(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CEILING_HOPS", "2")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "1")
    device = torch.device("cpu")
    feats = build_feature_table_at_time(
        patient007, 0, device=device, phys_cfg=PhysicsConfig(phase="biochem"), bio_cfg=BiochemConfig(phase="biochem")
    )
    assert "prior_score" in feats
    assert "bio_x_mask_wall" in feats
    assert "graph_degree" in feats
    assert "hop_from_inlet" in feats


def test_extended_probe_runs(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_CEILING_HOPS", "2")
    monkeypatch.setenv("CLOT_PHI_DGAMMA_SLICE", "1")
    rep = probe_anchor_extended(patient007, stem="patient007", ceiling_hops=2)
    assert rep.n_clot >= 100
    assert len(rep.rows) >= 40
    agg = aggregate_extended_rows([rep])
    assert agg[0]["feature"] == "prior_score"
