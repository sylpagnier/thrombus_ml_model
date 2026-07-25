"""Connected-component clot tiling helpers."""

from __future__ import annotations

import torch

from src.core_physics.clot_growth_masks import bool_mask_connected_components


def test_bool_mask_connected_components_two_islands():
    # 0-1-2   and   4-5  (3 isolated False)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]],
        dtype=torch.long,
    )
    mask = torch.tensor([True, True, True, False, True, True])
    comps = bool_mask_connected_components(mask, edge_index)
    assert len(comps) == 2
    sizes = sorted(int(c.sum().item()) for c in comps)
    assert sizes == [2, 3]
    # Largest first
    assert int(comps[0].sum().item()) == 3
    assert bool(comps[0][0].item()) and bool(comps[0][2].item())
    assert not bool(comps[0][4].item())


def test_bool_mask_connected_components_empty():
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mask = torch.zeros(2, dtype=torch.bool)
    assert bool_mask_connected_components(mask, edge_index) == []
