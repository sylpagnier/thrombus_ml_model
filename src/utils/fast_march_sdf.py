import torch

def fast_march_sdf(
    pos_nd: torch.Tensor,
    edge_index: torch.Tensor,
    wall_mask: torch.Tensor,
    clot_mask: torch.Tensor,
    original_sdf_nd: torch.Tensor,
    max_hops: int = 10,
) -> torch.Tensor:
    """
    Computes a smooth SDF field from the combined wall + clot boundary using graph fast-marching.
    
    Args:
        pos_nd: [N, 2] tensor of node positions
        edge_index: [2, E] tensor of graph edges
        wall_mask: [N] boolean tensor indicating original wall nodes
        clot_mask: [N] boolean tensor indicating clot nodes
        original_sdf_nd: [N] tensor of original SDF values
        max_hops: int, maximum number of hops for propagation
        
    Returns:
        sdf_updated: [N] tensor of updated SDF values
    """
    device = pos_nd.device
    num_nodes = pos_nd.size(0)
    
    sdf_updated = original_sdf_nd.clone()
    
    # Combined boundary
    boundary_mask = wall_mask | clot_mask
    boundary_indices = torch.nonzero(boundary_mask, as_tuple=False).view(-1)
    
    if boundary_indices.numel() == 0:
        return sdf_updated
        
    sdf_updated[boundary_mask] = 0.0
    
    # Edge lengths
    src, dst = edge_index
    edge_lengths = torch.norm(pos_nd[src] - pos_nd[dst], dim=1)
    
    # Simple BFS / Dijkstra approach
    # Since we need to run in <10ms for 15k nodes, we can use a dense array for distances
    # initialized to infinity, and iterative relaxation up to max_hops.
    # Actually, Bellman-Ford style relaxation is very fast on GPU and simple to implement.
    
    dist = torch.full((num_nodes,), float('inf'), device=device, dtype=pos_nd.dtype)
    dist[boundary_mask] = 0.0
    
    from torch_geometric.utils import scatter

    for _ in range(max_hops):
        # Propagate from src to dst
        new_dist = dist[src] + edge_lengths
        # Scatter min
        scattered_dist = scatter(new_dist, dst, dim=0, dim_size=num_nodes, reduce="min")
        
        # Update distances
        improved = scattered_dist < dist
        if not improved.any():
            break
        dist = torch.minimum(dist, scattered_dist)
        
    # Find nodes that were updated within max_hops (and aren't boundary)
    updated_mask = (dist < float('inf')) & (~boundary_mask)
    
    # We need to scale dist to match original SDF scale?
    # "d. Normalize the resulting distances to match the scale of the original SDF field"
    # The original SDF field is already a distance field (signed distance to wall).
    # Euclidean edge lengths in `pos_nd` are in the same non-dimensional space as `original_sdf_nd`.
    # Wait, SDF might be normalized by d_bar. Are `pos_nd` normalized by d_bar too?
    # "Positions by the geometric length scale (data.x[:, 0:2] are already ND on patient graphs)"
    # Yes, pos_nd is ND, meaning distance in pos_nd is identical to SDF.
    # No extra scaling factor needed! "Normalize the resulting distances to match the scale" might just mean ensuring they are in the same units, which they are if we use ND pos.
    
    sdf_updated[updated_mask] = dist[updated_mask]
    
    return sdf_updated
