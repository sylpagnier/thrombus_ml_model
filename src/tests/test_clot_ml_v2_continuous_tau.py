"""V2 continuous tau rollout smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import extrapolated_t_out_max, rollout_time_indices
from src.training.clot_ml_v2_continuous_tau import (
    apply_step2_v2_env,
    extrap_rollout_metrics,
    rollout_step1_v2_continuous_tau,
    v2_continuous_tau_enabled,
)


def test_apply_step2_v2_env_enables_continuous_extrap(monkeypatch):
    monkeypatch.delenv("CLOT_ML_CONTINUOUS_EXTRAP", raising=False)
    apply_step2_v2_env(sim_end_scale=2.5)
    assert v2_continuous_tau_enabled()
    assert os.environ.get("CLOT_ML_SIM_END_SCALE") == "2.5"


def test_rollout_indices_extend_with_sim_end_scale():
    class _G:
        y = torch.zeros(10, 4, 16)
        t = torch.linspace(0.0, 90.0, 10)

    g = _G()
    n1 = extrapolated_t_out_max(g, sim_end_scale=1.0)
    n2 = extrapolated_t_out_max(g, sim_end_scale=2.0)
    assert n2 > n1
    assert len(rollout_time_indices(g, sim_end_scale=2.0)) > len(rollout_time_indices(g, sim_end_scale=1.0))


def test_extrap_metrics_monotone_on_synthetic_phi():
    class _G:
        y = torch.zeros(6, 4, 8)
        t = torch.linspace(0.0, 50.0, 6)
        num_nodes = 8

    g = _G()
    bio = BiochemConfig(phase="biochem")
    phi_by_t = {}
    frac = 0.0
    for t in rollout_time_indices(g, sim_end_scale=2.0):
        frac = min(frac + 0.02, 0.5)
        phi_by_t[int(t)] = torch.full((8,), frac)
    m = extrap_rollout_metrics(phi_by_t, g, bio_cfg=bio, sim_end_scale=2.0)
    assert m["monotone_commit_frac"] is True
    assert m["n_extrap_steps"] > 0
    assert m["commit_frac_delta_extrap"] >= 0.0


@pytest.mark.skipif(
    not Path("data/processed/graphs_biochem_anchors/patient007.pt").exists(),
    reason="patient007 anchor missing",
)
@pytest.mark.skipif(
    not Path("outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth").exists(),
    reason="step1_a35 ckpt missing",
)
def test_v2_rollout_smoke_patient007():
    from src.training.clot_ml_device import resolve_clot_ml_eval_device
    from src.training.clot_ml_step1_residual import load_step1_checkpoint, resolve_step1_rule_cfg
    from src.utils.paths import get_project_root

    root = get_project_root()
    device = resolve_clot_ml_eval_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors/patient007.pt",
        map_location=device,
        weights_only=False,
    )
    rule_cfg = resolve_step1_rule_cfg(root / "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    model, meta = load_step1_checkpoint(
        root / "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
        device=device,
    )
    phi = rollout_step1_v2_continuous_tau(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys,
        bio_cfg=bio,
        alpha=float(meta.get("alpha", 0.35)),
        sim_end_scale=2.0,
    )
    t_max = extrapolated_t_out_max(data, sim_end_scale=2.0)
    assert int(t_max) in phi
    assert len(phi) > int(data.y.shape[0])
