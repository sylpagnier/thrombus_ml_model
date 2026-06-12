"""Rung 4 s3 temporal GRU wiring."""

from __future__ import annotations

import pytest
import torch

from src.core_physics.t0_r4_s2_species import _apply_loc_gate_residual, feature_dim
from src.core_physics.t0_r4_s3_temporal import (
    T0R4S3TemporalModel,
    s3_feature_dim,
    save_s3_checkpoint,
    load_s3_bundle,
)
from src.core_physics.t0_rung4_ladder import rung4_step_from_env


def test_s3_alias():
    import os

    os.environ["T0_RUNG4_STEP"] = "s3"
    assert rung4_step_from_env() == "s3_temporal"


def test_s3_gru_zero_init():
    m = T0R4S3TemporalModel(in_dim=s3_feature_dim(), gru_hidden=16, res_hidden=16, use_residual=False)
    n = 20
    feats = torch.randn(n, s3_feature_dim())
    h0 = m.init_hidden(n, torch.device("cpu"), torch.float32)
    logit, h1 = m.forward_step(feats, h0, res_scale=1.0)
    assert logit.shape == (n,)
    assert h1.shape == (n, 16)
    assert torch.allclose(logit, torch.zeros_like(logit), atol=1e-6)


def test_gate_actuator_moves_species():
    """Gate residual must change FI/Mat on eligible nodes (unlike risk rank barrier)."""
    from src.config import BiochemConfig

    bio = BiochemConfig(phase="biochem")
    n = 8
    logit = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    elig = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
    s0_gate = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.0, 0.0, 0.0, 0.0])
    g = _apply_loc_gate_residual(s0_gate, logit, elig, onset=1.0, loc_scale=0.5)
    assert float(g[0]) > float(s0_gate[0])
    assert float(g[1]) < float(s0_gate[1])


def test_s3_ckpt_roundtrip(tmp_path):
    m = T0R4S3TemporalModel(in_dim=s3_feature_dim(), gru_hidden=16, res_hidden=16, use_residual=False)
    ckpt = tmp_path / "s3.pth"
    save_s3_checkpoint(
        ckpt, m, loc_scale=1.5, res_scale=1.0, gru_hidden=16, res_hidden=16,
        use_residual=False, actuator="gate",
    )
    bundle = load_s3_bundle(ckpt, device=torch.device("cpu"), quiet=True)
    assert bundle is not None
    assert bundle.loc_scale == 1.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_s3_rollout_requires_ckpt():
    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.t0_rung4_ladder import rollout_rung4_phi_trajectory
    from src.utils.paths import get_project_root

    graph = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    ckpt = get_project_root() / "outputs/biochem/t0_r4_s3_temporal/best.pth"
    if not graph.is_file():
        pytest.skip("patient007 missing")
    if not ckpt.is_file():
        pytest.skip("s3 checkpoint missing")
    device = torch.device("cuda")
    data = torch.load(graph, map_location=device, weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    traj = rollout_rung4_phi_trajectory(data, phys, bio, device, step="s3_temporal")
    assert len(traj) == int(data.y.shape[0])
