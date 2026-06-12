"""V0 nucleation mask unit tests."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_nucleation_mask import (
    catalytic_rate_multiplier,
    project_phi_with_nucleation,
    resolve_catalytic_hood,
    resolve_nucleation_eligibility,
    resolve_wall_mask,
)


def _chain_graph() -> Data:
    # 0 -- 1 -- 2 ; wall at node 0 only
    x = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.05],
            [2.0, 0.0, 0.10],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1], [1, 2], [1, 0], [2, 1]], dtype=torch.long)
    y = torch.zeros(3, 4, 8)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.mask_wall = torch.tensor([1, 0, 0], dtype=torch.uint8)
    data.num_nodes = 3
    return data


def test_t0_eligibility_wall_only():
    data = _chain_graph()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    elig = resolve_nucleation_eligibility(
        data, 0, device, phys, bio, use_dgamma_wall_seed=False
    )
    wall = resolve_wall_mask(data, device)
    assert torch.equal(elig, wall)
    assert int(elig.sum()) == 1


def test_eligibility_includes_neighbor_of_commit():
    data = _chain_graph()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    commits = torch.tensor([False, True, False])
    elig = resolve_nucleation_eligibility(
        data,
        2,
        device,
        phys,
        bio,
        commits_prev=commits,
        use_dgamma_wall_seed=False,
    )
    assert bool(elig[0].item())  # wall
    assert bool(elig[2].item())  # 1-hop from commit at node 1


def test_project_phi_monotone_and_eligibility():
    elig = torch.tensor([1.0, 1.0, 0.0])
    prev = torch.tensor([0.0, 1.0, 0.0])
    raw = torch.tensor([0.2, 0.3, 0.9])
    out = project_phi_with_nucleation(raw, prev, elig, commit_thresh=0.5)
    assert out[1] >= 1.0 - 1e-6
    assert out[2] <= 0.0 + 1e-6
    assert out[0] >= 0.2 - 1e-6


def test_catalytic_multiplier():
    hood = torch.tensor([0.0, 1.0, 0.0])
    mult = catalytic_rate_multiplier(hood, beta=0.5)
    assert abs(float(mult[1].item()) - 1.5) < 1e-6


def test_catalytic_hood_dilate():
    data = _chain_graph()
    commits = torch.tensor([False, True, False])
    hood = resolve_catalytic_hood(commits, data.edge_index, catalytic_hops=1)
    assert bool(hood[0].item())
    assert bool(hood[1].item())
    assert bool(hood[2].item())
