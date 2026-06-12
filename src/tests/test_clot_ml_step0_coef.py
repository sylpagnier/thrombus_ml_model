"""Step 0 learned rule coefficients."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step0_coef import (
    PARAM_NAMES,
    Step0RuleCoefs,
    eval_coef_on_anchor,
)
from src.utils.paths import get_project_root


def test_coef_vector_roundtrip():
    c0 = Step0RuleCoefs.inc40_baseline()
    c1 = Step0RuleCoefs.from_vector(c0.to_vector())
    assert c1.onset == c0.onset
    assert c1.neg_dx == c0.neg_dx
    assert len(PARAM_NAMES) == 10


def test_to_rule_config_smoke():
    cfg = Step0RuleCoefs.inc40_baseline().to_rule_config()
    assert cfg.kind == "progressive_topk"
    assert cfg.global_onset_frac == pytest.approx(0.40)
    assert cfg.localized is not None
    assert cfg.localized.neg_dx_risk_weight == pytest.approx(0.25)


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return path


def test_eval_inc40_pred_kine_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    reset_temporal_kinematics_cache()
    row = eval_coef_on_anchor(
        Step0RuleCoefs.inc40_baseline(),
        graph_path=patient007,
        device=torch.device("cpu"),
        phys_cfg=PhysicsConfig(phase="biochem"),
        bio_cfg=BiochemConfig(phase="biochem"),
    )
    assert row["n_pairs"] > 0
    assert row["deploy_score"] == row["deploy_score"]
    assert 0.0 <= row["deploy_score"] <= 1.0
