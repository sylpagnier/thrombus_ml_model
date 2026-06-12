"""Deploy v1 smoke tests."""

from __future__ import annotations

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.evaluation.clot_extrap_plausibility import compute_extrap_plausibility
from src.inference.clot_ml_deploy_v1 import (
    audit_y_invariance,
    default_deploy_v1_recipe,
    eval_deploy_v1_on_graph,
    load_deploy_v1_recipe,
)
from src.utils.paths import get_project_root


@pytest.fixture
def patient007():
    path = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not path.is_file():
        pytest.skip("patient007 missing")
    return path


def test_load_deploy_v1_recipe():
    r = load_deploy_v1_recipe()
    assert r.phi_shell == "step1"
    assert r.step1_ckpt


def test_extrap_plausibility_keys():
    phi = {0: torch.zeros(10), 5: torch.ones(10) * 0.6}
    class _D:
        y = torch.zeros(6, 20)
        num_nodes = 10
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    out = compute_extrap_plausibility(_D(), phi, sim_end_scale=1.5)
    assert "pred_frac_delta" in out
    assert "phi_monotone" in out


def test_eval_deploy_v1_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    ckpt = get_project_root() / "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth"
    if not ckpt.is_file():
        pytest.skip("step1 ckpt missing")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    recipe = default_deploy_v1_recipe()
    recipe.step1_ckpt = str(ckpt.relative_to(get_project_root()))
    row = eval_deploy_v1_on_graph(data, recipe, device=torch.device("cpu"), anchor="patient007")
    assert row["n_phi_steps"] > 0
    assert 0.0 <= row["deploy_score"] <= 1.0


def test_audit_y_invariance_smoke(patient007, monkeypatch):
    monkeypatch.setenv("CLOT_PHI_MINIMAL_FEATURES", "1")
    monkeypatch.setenv("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    ckpt = get_project_root() / "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth"
    if not ckpt.is_file():
        pytest.skip("step1 ckpt missing")
    reset_temporal_kinematics_cache()
    data = torch.load(patient007, map_location="cpu", weights_only=False)
    recipe = default_deploy_v1_recipe()
    recipe.step1_ckpt = str(ckpt.relative_to(get_project_root()))
    out = audit_y_invariance(data, recipe, device=torch.device("cpu"))
    assert "pass_y_invariance" in out
