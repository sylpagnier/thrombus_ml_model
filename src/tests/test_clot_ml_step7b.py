"""Step 7b hybrid smoke test."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_pivot_rule_mixture import ClotRuleMixtureModel, n_rule_experts
from src.training.clot_ml_step0_coef import Step0RuleCoefs
from src.training.clot_ml_step1_residual import ClotRuleResidualMLP
from src.training.clot_ml_step7b_hybrid import (
    rollout_step7b_phi,
    step7b_feature_dim,
)
from src.utils.paths import get_project_root


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return path


def test_step7b_feature_dim():
    assert step7b_feature_dim() == 4


def test_step7b_rollout_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    mixture = ClotRuleMixtureModel(hidden=16, n_experts=n_rule_experts())
    residual = ClotRuleResidualMLP(in_dim=step7b_feature_dim(), hidden=16)
    phi = rollout_step7b_phi(
        data,
        rule,
        mixture,
        residual,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
        alpha=0.35,
    )
    assert len(phi) > 0
    t53 = phi[max(phi.keys())]
    assert float(t53.max()) > 0.05
