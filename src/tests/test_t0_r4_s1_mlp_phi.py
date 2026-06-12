"""Rung 4 s1 residual phi MLP wiring."""

from __future__ import annotations

import pytest
import torch

from src.core_physics.t0_r4_s1_mlp_phi import T0R4S1PhiMLP, feature_dim, save_s1_checkpoint, load_s1_bundle
from src.core_physics.t0_rung4_ladder import rung4_step_from_env, _species_step_for


def test_s1_alias_and_species_step():
    import os

    os.environ["T0_RUNG4_STEP"] = "s1_mlp"
    assert rung4_step_from_env() == "s1_mlp_phi"
    assert _species_step_for("s1_mlp_phi") == "s0"


def test_s1_mlp_forward_shape():
    m = T0R4S1PhiMLP(in_dim=feature_dim(), hidden=16)
    x = torch.randn(50, feature_dim())
    y = m(x)
    assert y.shape == (50,)
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_s1_ckpt_roundtrip(tmp_path):
    m = T0R4S1PhiMLP(in_dim=feature_dim(), hidden=16)
    ckpt = tmp_path / "s1.pth"
    save_s1_checkpoint(ckpt, m, alpha=0.75, hidden=16, meta={"test": 1})
    bundle = load_s1_bundle(ckpt, device=torch.device("cpu"), quiet=True)
    assert bundle is not None
    assert bundle.alpha == 0.75
    assert bundle.in_dim == feature_dim()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_s1_rollout_requires_ckpt():
    from src.utils.paths import get_project_root
    from src.core_physics.t0_rung4_ladder import rollout_rung4_phi_trajectory
    from src.config import BiochemConfig, PhysicsConfig

    graph = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not graph.is_file():
        pytest.skip("patient007 missing")
    ckpt = get_project_root() / "outputs/biochem/t0_r4_s1_mlp_phi/best.pth"
    if not ckpt.is_file():
        pytest.skip("s1 checkpoint missing; run train_t0_r4_s1_mlp_phi")
    device = torch.device("cuda")
    data = torch.load(graph, map_location=device, weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    traj = rollout_rung4_phi_trajectory(data, phys, bio, device, step="s1_mlp_phi")
    assert len(traj) == int(data.y.shape[0])
