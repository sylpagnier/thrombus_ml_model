"""Step 1 residual corrector smoke tests."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step0_coef import Step0RuleCoefs
from src.training.clot_ml_step1_residual import (
    ClotRuleResidualMLP,
    combine_rule_residual,
    eval_step1_on_anchor,
    rollout_step1_phi,
    step1_feature_dim,
)
from src.utils.paths import get_project_root


def test_combine_rule_residual():
    phi_rule = torch.tensor([0.0, 0.5, 1.0])
    ceiling = torch.tensor([1.0, 1.0, 0.0])
    delta = torch.zeros(3)
    out = combine_rule_residual(phi_rule, delta, ceiling, alpha=0.35)
    assert out[0].item() == pytest.approx(0.175, abs=0.01)
    assert out[2].item() == pytest.approx(0.0)


def test_step1_feature_dim():
    assert step1_feature_dim() == 4


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return path


def test_rollout_step1_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotRuleResidualMLP(in_dim=step1_feature_dim(), hidden=16)
    phi_by_t = rollout_step1_phi(
        data,
        rule,
        model,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
        alpha=0.35,
    )
    assert len(phi_by_t) > 0


def test_eval_step1_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotRuleResidualMLP(in_dim=step1_feature_dim(), hidden=16)
    row = eval_step1_on_anchor(
        model,
        rule,
        graph_path=patient007,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
        alpha=0.35,
        pair_stride=8,
    )
    assert row["n_pairs"] > 0
    assert 0.0 <= row["deploy_score"] <= 1.0
