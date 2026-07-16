import torch
from src.core_physics.species_snapshot_gnn import (
    build_even_hop_edges,
    find_odd_to_even_neighbors,
    SpeciesSnapshotGNN,
)

def test_skip_hop_routing():
    # Simple line graph: 0 (wall) - 1 - 2 - 3 - 4
    # Wall is at node 0.
    # Even-hop nodes (even distances from wall): 0, 2, 4
    # Odd-hop nodes (odd distances from wall): 1, 3
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    
    even_mask = torch.tensor([True, False, True, False, True])
    odd_mask = torch.tensor([False, True, False, True, False])
    
    even_edges = build_even_hop_edges(edge_index, even_mask)
    assert even_edges.shape[1] > 0
    
    neighbor_1, neighbor_2 = find_odd_to_even_neighbors(edge_index, even_mask, odd_mask)
    assert neighbor_1[1].item() in (0, 2)
    assert neighbor_2[1].item() in (0, 2)
    assert neighbor_1[1].item() != neighbor_2[1].item()
    
    assert neighbor_1[3].item() in (2, 4)
    assert neighbor_2[3].item() in (2, 4)
    assert neighbor_1[3].item() != neighbor_2[3].item()

def test_reconstruct_odd_nodes():
    gnn = SpeciesSnapshotGNN(in_dim=4, hidden=16, out_dim=2)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    wall_mask_band = torch.tensor([True, False, False, False, False])
    
    pos_band = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0]
    ])
    
    gnn.set_band_geometry(pos_band, edge_index, wall_mask_band)
    
    # Values at even nodes: node 0: 1.0, node 2: 3.0, node 4: 5.0
    # Values at odd nodes (to be overwritten): node 1: 99.0, node 3: 99.0
    values = torch.tensor([
        [1.0, 10.0],
        [99.0, 99.0],
        [3.0, 30.0],
        [99.0, 99.0],
        [5.0, 50.0]
    ], dtype=torch.float32)
    
    reconstructed = gnn._reconstruct_odd_nodes(values, edge_index)
    
    # Check that odd nodes are averages of even neighbors
    # Node 1 = average of Node 0 and Node 2: [2.0, 20.0]
    assert torch.allclose(reconstructed[1], torch.tensor([2.0, 20.0]))
    # Node 3 = average of Node 2 and Node 4: [4.0, 40.0]
    assert torch.allclose(reconstructed[3], torch.tensor([4.0, 40.0]))
    # Even nodes should be unchanged
    assert torch.allclose(reconstructed[0], torch.tensor([1.0, 10.0]))
    assert torch.allclose(reconstructed[2], torch.tensor([3.0, 30.0]))
    assert torch.allclose(reconstructed[4], torch.tensor([5.0, 50.0]))
