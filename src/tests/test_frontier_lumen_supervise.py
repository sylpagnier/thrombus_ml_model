"""frontier_lumen / frontier_ge2 supervise mask geometry."""

from __future__ import annotations

import torch

from src.core_physics.clot_growth_masks import graph_dilate_hops
from src.core_physics.species_pushforward_continuous import compute_hop_distances


def test_frontier_lumen_mask_excludes_wall():
    # Line: 0(wall)-1-2-3 ; clot at 0 and 1
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    wall = torch.tensor([True, False, False, False])
    clot = torch.tensor([True, True, False, False])
    frontier = graph_dilate_hops(clot, edge_index, hops=1)
    # dilate1 from {0,1} -> {0,1,2}
    assert frontier.tolist() == [True, True, True, False]
    lumen = frontier & (~wall)
    assert lumen.tolist() == [False, True, True, False]
    assert not bool(lumen[wall].any().item())


def test_frontier_ge2_mask_is_dilate_and_hop_ge2():
    # Line: 0(wall)-1-2-3-4 ; clot at wall+hop1
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    wall = torch.tensor([True, False, False, False, False])
    clot = torch.tensor([True, True, False, False, False])
    frontier = graph_dilate_hops(clot, edge_index, hops=2)
    hop = compute_hop_distances(edge_index, wall, num_nodes=5)
    ge2 = frontier & (hop >= 2)
    # hop: 0,1,2,3,4 -> 0,1,2,3,4
    assert hop.tolist() == [0, 1, 2, 3, 4]
    # dilate2 from {0,1} reaches through node 3; node 4 may or may not depending on hops
    assert bool(ge2[2].item())  # hop2 inside dilate
    assert not bool(ge2[0].item())  # wall / hop0 excluded
    assert not bool(ge2[1].item())  # hop1 excluded
