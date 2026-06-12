"""Rung 4 s2 species corrector wiring."""

from __future__ import annotations

import pytest
import torch

from src.core_physics.t0_r4_s2_species import (
    S2_MODE_LOC,
    T0R4S2LocMLP,
    T0R4S2SpeciesMLP,
    apply_s2_species_delta,
    feature_dim,
    load_s2_bundle,
    save_s2_checkpoint,
)
from src.core_physics.t0_rung4_ladder import _species_step_for, rung4_step_from_env


def test_s2_alias():
    import os

    os.environ["T0_RUNG4_STEP"] = "s2"
    assert rung4_step_from_env() == "s2_species"
    assert _species_step_for("s2_species") == "s2_species"


def test_s2_mlp_zero_init():
    m = T0R4S2SpeciesMLP(in_dim=feature_dim(), hidden=16)
    x = torch.randn(30, feature_dim())
    d = m(x)
    assert d.shape == (30, 2)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


def test_s2_loc_mlp_zero_init():
    m = T0R4S2LocMLP(in_dim=feature_dim(), hidden=16)
    x = torch.randn(30, feature_dim())
    d = m(x)
    assert d.shape == (30,)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-6)


def test_apply_delta_masked():
    sp = torch.zeros(10, 12)
    delta = torch.ones(10, 2)
    elig = torch.zeros(10, dtype=torch.bool)
    elig[3:6] = True
    out = apply_s2_species_delta(sp, delta, elig, delta_scale=0.5)
    assert out[2, 8] == 0.0
    assert out[4, 8] == 0.5
    assert out[4, 11] == 0.5


def test_s2_ckpt_roundtrip(tmp_path):
    m = T0R4S2LocMLP(in_dim=feature_dim(), hidden=16)
    ckpt = tmp_path / "s2.pth"
    save_s2_checkpoint(
        ckpt, m, mode=S2_MODE_LOC, hidden=16, delta_scale=1.0, loc_scale=1.5, meta={"t": 1}
    )
    bundle = load_s2_bundle(ckpt, device=torch.device("cpu"), quiet=True)
    assert bundle is not None
    assert bundle.mode == S2_MODE_LOC
    assert bundle.loc_scale == 1.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_s2_rollout_requires_ckpt():
    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.t0_rung4_ladder import rollout_rung4_phi_trajectory
    from src.utils.paths import get_project_root

    graph = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    ckpt = get_project_root() / "outputs/biochem/t0_r4_s2_loc/best.pth"
    if not ckpt.is_file():
        ckpt = get_project_root() / "outputs/biochem/t0_r4_s2_species/best.pth"
    if not graph.is_file():
        pytest.skip("patient007 missing")
    if not ckpt.is_file():
        pytest.skip("s2 checkpoint missing")
    device = torch.device("cuda")
    data = torch.load(graph, map_location=device, weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    traj = rollout_rung4_phi_trajectory(data, phys, bio, device, step="s2_species")
    assert len(traj) == int(data.y.shape[0])
