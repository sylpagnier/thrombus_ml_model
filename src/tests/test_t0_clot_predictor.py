"""T0 clot predictor sweep (no spf.mu / spf.sr input)."""

from pathlib import Path

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_clot_predictor import (
    predict_clot_phi_species_hard,
    sweep_t0_clot_predictors,
    t0_gt_baseline_env,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, predict_clot_phi_at_time


@pytest.fixture
def patient007_graph():
    p = Path("data/processed/graphs_biochem_anchors/patient007.pt")
    if not p.is_file():
        pytest.skip("patient007 graph missing")
    return torch.load(p, map_location="cpu", weights_only=False)


def test_species_hard_f1_beats_random(patient007_graph):
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    device = torch.device("cpu")
    t = min(53, int(patient007_graph.y.shape[0]) - 1)
    phi_gt = gt_clot_phi_at_time(patient007_graph, t, phys, device)
    phi_sp = predict_clot_phi_species_hard(patient007_graph, t, bio, device)
    tp = ((phi_sp > 0.5) & (phi_gt > 0.5)).float().sum()
    assert float(tp.item()) > 0


def test_sweep_finds_mu_growth_config(patient007_graph):
    times = [0, min(53, int(patient007_graph.y.shape[0]) - 1)]
    report = sweep_t0_clot_predictors(
        patient007_graph, anchor="patient007", times=times
    )
    assert report.best.get("species_hard") is not None
    assert report.best.get("mu_growth") is not None
    assert report.best["species_hard"]["mean_f1"] > 0.3


def test_gt_baseline_env_blocks_comsol_sr_anchor():
    with t0_gt_baseline_env(gamma_mode="kinematic") as cfg:
        assert cfg["gamma_mode"] == "kinematic"
        assert "CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR" not in __import__("os").environ
