"""Step 7 band GNN phi smoke test."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step0_coef import Step0RuleCoefs
from src.training.clot_ml_step7_band_phi import (
    ClotBandPhiGNN,
    rollout_step7_phi,
    step7_feature_dim,
)
from src.utils.paths import get_project_root


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return path


def test_step7_feature_dim():
    assert step7_feature_dim() == 5


def test_step7_rollout_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    rule = Step0RuleCoefs.inc40_baseline().to_rule_config()
    model = ClotBandPhiGNN(in_dim=step7_feature_dim(), hidden=16)
    phi = rollout_step7_phi(
        data,
        rule,
        model,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert len(phi) > 0
    t0 = phi[0]
    assert float(t0.max()) < 0.01  # inc40 onset gate zeros early times
