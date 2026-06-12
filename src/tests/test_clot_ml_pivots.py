"""Side pivot smoke tests."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_pivot_data_driven import (
    ClotDataDrivenPhiGNN,
    data_driven_feature_dim,
    rollout_data_driven_phi,
)
from src.training.clot_ml_pivot_rule_mixture import (
    ClotRuleMixtureModel,
    n_rule_experts,
    rollout_rule_mixture_phi,
)
from src.training.clot_ml_pivot_soft_commit import (
    ClotSoftCommitModel,
    rollout_soft_commit_phi,
)
from src.training.clot_ml_step0_coef import Step0RuleCoefs
from src.utils.paths import get_project_root


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return path


def test_n_rule_experts():
    assert n_rule_experts() == 8


def test_data_driven_feature_dim():
    assert data_driven_feature_dim() == 5


def test_soft_commit_rollout_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotSoftCommitModel(hidden=16)
    phi = rollout_soft_commit_phi(
        data,
        rule,
        model,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert len(phi) > 0


def test_rule_mixture_rollout_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotRuleMixtureModel(hidden=16)
    phi = rollout_rule_mixture_phi(
        data,
        rule,
        model,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert len(phi) > 0


def test_data_driven_rollout_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    model = ClotDataDrivenPhiGNN(hidden=16)
    phi = rollout_data_driven_phi(
        data,
        model,
        device=torch.device("cpu"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert len(phi) > 0
