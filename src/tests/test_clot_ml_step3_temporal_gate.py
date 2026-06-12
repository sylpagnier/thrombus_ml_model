"""Step 3 temporal gate smoke tests."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step0_coef import Step0RuleCoefs
from src.training.clot_ml_step3_temporal_gate import (
    ClotTemporalGateModel,
    compute_gt_onset_frac,
    eval_step3_on_anchor,
    map_onset_logit,
    rollout_step3_phi,
    step2_feature_dim,
)
from src.utils.paths import get_project_root


def test_map_onset_logit_midrange():
    logit = torch.tensor(0.0)
    onset = map_onset_logit(logit, 0.25, 0.50)
    assert float(onset.item()) == pytest.approx(0.375, abs=0.02)


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return path


def test_compute_gt_onset_frac(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    onset = compute_gt_onset_frac(
        data,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert 0.0 <= onset <= 1.0


def test_rollout_step3_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotTemporalGateModel(in_dim=step2_feature_dim(), hidden=16)
    phi_by_t = rollout_step3_phi(
        data,
        rule,
        model,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert len(phi_by_t) > 0


def test_eval_step3_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotTemporalGateModel(in_dim=step2_feature_dim(), hidden=16)
    row = eval_step3_on_anchor(
        model,
        rule,
        graph_path=patient007,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
        pair_stride=8,
    )
    assert row["n_pairs"] > 0
    assert 0.0 <= row["deploy_score"] <= 1.0
    assert 0.25 <= row["onset_pred"] <= 0.50
