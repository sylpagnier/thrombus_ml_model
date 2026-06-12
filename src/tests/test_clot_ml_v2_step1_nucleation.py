"""V1 step1 nucleation rollout smoke tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step1_residual import load_step1_checkpoint, resolve_step1_rule_cfg, rollout_step1_phi
from src.training.clot_ml_v2_step1_nucleation import rollout_step1_v1_nucleation


@pytest.mark.skipif(
    not (Path("data/processed/graphs_biochem_anchors/patient007.pt").is_file()),
    reason="patient007 anchor missing",
)
def test_v1_nucleation_rollout_differs_from_ceiling():
    root = Path(__file__).resolve().parents[2]
    ckpt = root / "outputs/biochem/sweep_clot_ml_physics_6h/clot_ml_step1_best.pth"
    step0 = root / "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    graph = root / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not ckpt.is_file():
        pytest.skip("step1 ckpt missing")

    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
    device = torch.device("cpu")
    data = torch.load(graph, map_location=device, weights_only=False)
    rule_cfg = load_step0_coef_json(step0).to_rule_config()
    model, meta = load_step1_checkpoint(ckpt, device=device)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    alpha = float(meta.get("alpha", 0.35))

    reset_temporal_kinematics_cache()
    phi_c = rollout_step1_phi(
        data, rule_cfg, model, device=device, phys_cfg=phys, bio_cfg=bio, alpha=alpha
    )
    reset_temporal_kinematics_cache()
    phi_n = rollout_step1_v1_nucleation(
        data, rule_cfg, model, device=device, phys_cfg=phys, bio_cfg=bio, alpha=alpha
    )

    t_final = int(data.y.shape[0]) - 1
    fc = float((phi_c[t_final] > 0.5).float().mean().item())
    fn = float((phi_n[t_final] > 0.5).float().mean().item())
    assert phi_c.keys() == phi_n.keys()
    assert abs(fc - fn) > 1e-6 or not torch.allclose(phi_c[t_final], phi_n[t_final], atol=1e-5)
