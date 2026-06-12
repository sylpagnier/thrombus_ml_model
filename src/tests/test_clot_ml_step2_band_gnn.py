"""Step 2 band GNN ranker smoke tests."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step0_coef import Step0RuleCoefs
from src.training.clot_ml_step2_band_gnn import (
    ClotBandRiskGNN,
    combine_gnn_risk,
    eval_step2_on_anchor,
    rollout_step2_phi,
    step2_feature_dim,
)
from src.utils.paths import get_project_root


def test_combine_gnn_risk_zero_delta_matches_hand():
    from dataclasses import replace

    hand = torch.tensor([0.2, 0.8, 0.1])
    pool = torch.tensor([1.0, 1.0, 0.0]).bool()
    logits = torch.zeros(3)
    base = Step0RuleCoefs.inc40_baseline().to_rule_config()
    loc = base.localized
    assert loc is not None
    rule = replace(base, localized=replace(loc, normalize_risk_per_half=False))
    out = combine_gnn_risk(
        hand,
        logits,
        pool,
        data=None,  # type: ignore[arg-type]
        rule_cfg=rule,
        device=torch.device("cpu"),
        delta_scale=0.30,
    )
    assert out[0].item() == pytest.approx(0.2, abs=0.01)
    assert out[1].item() == pytest.approx(0.8, abs=0.01)
    assert out[2].item() == pytest.approx(0.0, abs=1e-6)


def test_step2_feature_dim():
    assert step2_feature_dim() == 3


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return path


def test_rollout_step2_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotBandRiskGNN(in_dim=step2_feature_dim(), hidden=16)
    phi_by_t = rollout_step2_phi(
        data,
        rule,
        model,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
        delta_scale=0.30,
    )
    assert len(phi_by_t) > 0


def test_eval_step2_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotBandRiskGNN(in_dim=step2_feature_dim(), hidden=16)
    row = eval_step2_on_anchor(
        model,
        rule,
        graph_path=patient007,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
        delta_scale=0.30,
        pair_stride=8,
    )
    assert row["n_pairs"] > 0
    assert 0.0 <= row["deploy_score"] <= 1.0
