"""Small graph-data masks for biochem labels (GNODE-independent helpers)."""

from __future__ import annotations

import torch


def biochem_truth_node_mask(batch, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Nodes whose entries in ``y`` are trusted COMSOL labels.

    Resolves a per-node trust mask from either a per-node ``is_anchor`` tensor or a
    graph-level ``is_anchor`` flag (broadcast via ``batch.batch`` when batched).
    """
    if not hasattr(batch, "is_anchor"):
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    m = batch.is_anchor
    if not torch.is_tensor(m):
        m = torch.tensor(m, dtype=torch.bool)
    m = m.reshape(-1)
    if m.numel() == 1:
        if bool(m.item()):
            return torch.ones(num_nodes, dtype=torch.bool, device=device)
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    batch_idx = getattr(batch, "batch", None)
    if batch_idx is not None:
        return m[batch_idx].to(device)
    if m.shape[0] != num_nodes:
        return torch.zeros(num_nodes, dtype=torch.bool, device=device)
    return m.to(device)
