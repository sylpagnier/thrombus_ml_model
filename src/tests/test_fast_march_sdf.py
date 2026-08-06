import torch
import pytest
from src.utils.fast_march_sdf import fast_march_sdf

def test_fast_march_sdf_boundary_zero():
    pos_nd = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    wall_mask = torch.tensor([True, False, False])
    clot_mask = torch.tensor([False, False, False])
    original_sdf = torch.tensor([0.0, 5.0, 5.0])
    
    updated = fast_march_sdf(pos_nd, edge_index, wall_mask, clot_mask, original_sdf, max_hops=10)
    assert updated[0].item() == 0.0

def test_fast_march_sdf_1_hop_distance():
    pos_nd = torch.tensor([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    wall_mask = torch.tensor([True, False, False])
    clot_mask = torch.tensor([False, False, False])
    original_sdf = torch.tensor([0.0, 5.0, 5.0])
    
    updated = fast_march_sdf(pos_nd, edge_index, wall_mask, clot_mask, original_sdf, max_hops=10)
    assert torch.isclose(updated[1], torch.tensor(1.5))
    assert torch.isclose(updated[2], torch.tensor(3.0))

def test_fast_march_sdf_monotonically_increasing():
    # A chain of 5 nodes
    pos_nd = torch.tensor([[float(i), 0.0] for i in range(5)])
    src = []
    dst = []
    for i in range(4):
        src.extend([i, i+1])
        dst.extend([i+1, i])
    edge_index = torch.tensor([src, dst])
    
    wall_mask = torch.tensor([True, False, False, False, False])
    clot_mask = torch.tensor([False, False, False, False, False])
    original_sdf = torch.full((5,), 10.0)
    
    updated = fast_march_sdf(pos_nd, edge_index, wall_mask, clot_mask, original_sdf, max_hops=10)
    
    for i in range(4):
        assert updated[i] < updated[i+1]

def test_fast_march_sdf_max_hops():
    # A chain of 5 nodes, length between each is 1.0
    pos_nd = torch.tensor([[float(i), 0.0] for i in range(5)])
    src = []
    dst = []
    for i in range(4):
        src.extend([i, i+1])
        dst.extend([i+1, i])
    edge_index = torch.tensor([src, dst])
    
    wall_mask = torch.tensor([True, False, False, False, False])
    clot_mask = torch.tensor([False, False, False, False, False])
    original_sdf = torch.tensor([0.0, 10.0, 10.0, 10.0, 10.0])
    
    # Restrict to 2 hops
    updated = fast_march_sdf(pos_nd, edge_index, wall_mask, clot_mask, original_sdf, max_hops=2)
    
    assert updated[0].item() == 0.0
    assert updated[1].item() == 1.0
    assert updated[2].item() == 2.0
    # Node 3 and 4 are beyond 2 hops, should keep original SDF (10.0)
    assert updated[3].item() == 10.0
    assert updated[4].item() == 10.0
