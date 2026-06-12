"""Rung 4 mini-ladder (legacy R4.1 alias) species wiring."""

from __future__ import annotations

import pytest
import torch

from src.core_physics.t0_rules_species import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    build_rules_species_log_nd_at_time,
    resting_species_log_nd,
)
from src.config import BiochemConfig, PhysicsConfig


def _tiny_graph():
    n, t_steps = 20, 4
    y = torch.randn(t_steps, n, 16)
    data = type("G", (), {})()
    data.y = y
    data.num_nodes = n
    data.edge_index = torch.stack(
        [torch.arange(n - 1), torch.arange(1, n)], dim=0
    )
    data.x = torch.randn(n, 3)
    data.mask_wall = torch.zeros(n, dtype=torch.bool)
    data.mask_wall[:3] = True
    data.u_ref = torch.tensor(1.0)
    data.d_bar = torch.tensor(0.01)
    return data


def test_s0_deploy_flag_vs_oracle():
    from src.core_physics.t0_rung4_ladder import rung4_step_is_deploy, rung4_step_uses_gt_species

    assert rung4_step_is_deploy("s0")
    assert not rung4_step_uses_gt_species("s0")
    assert rung4_step_uses_gt_species("s0_oracle_fi_mat")
    assert not rung4_step_is_deploy("s0_oracle_fi_mat")


def test_s0_oracle_fi_mat_uses_gt_in_elig():
    device = torch.device("cpu")
    data = _tiny_graph()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    sp, elig = build_rules_species_log_nd_at_time(
        data, 0, device, phys, bio, commits_prev=None, mode="s0_oracle_fi_mat"
    )
    sp_rest = resting_species_log_nd(data, device)
    sp_gt = data.y[0, :, 4:16]
    if bool(elig.any()):
        assert torch.allclose(sp[elig, FI_SLICE_IDX], sp_gt[elig, FI_SLICE_IDX])
        assert torch.allclose(sp[elig, MAT_SLICE_IDX], sp_gt[elig, MAT_SLICE_IDX])
    off = ~elig
    if bool(off.any()):
        assert torch.allclose(sp[off], sp_rest[off])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rollout_rules_species_shape_cuda():
    from src.utils.paths import get_project_root
    from src.core_physics.t0_rules_species import rollout_t0_rules_species_series

    graph = get_project_root() / "data/processed/graphs_biochem_anchors/patient007.pt"
    if not graph.is_file():
        pytest.skip("patient007 missing")
    device = torch.device("cuda")
    data = torch.load(graph, map_location=device, weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    out = rollout_t0_rules_species_series(data, phys, bio, device, mode="s0")
    assert out.shape == data.y.shape
