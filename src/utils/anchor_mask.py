"""Resolve per-node anchor masks from graph-level or per-node ``is_anchor`` flags."""

from __future__ import annotations

import torch


def graph_has_anchor(data) -> bool:
    ia = getattr(data, "is_anchor", None)
    if ia is None:
        return False
    if torch.is_tensor(ia):
        return bool(ia.any().item())
    return bool(ia)


def anchor_node_mask(data):
    """Return ``[N]`` bool mask, or ``None`` if no anchor supervision.

    ``mesh_to_graph`` uses graph-level ``is_anchor`` of shape ``[1]``.
    Batched graphs use ``is_anchor[batch]`` (one flag per graph).
    Tier-3 style graphs may use per-node ``is_anchor`` of shape ``[N]``.
    """
    if not hasattr(data, "is_anchor"):
        return None
    ia = data.is_anchor
    if not torch.is_tensor(ia):
        ia = torch.tensor(bool(ia), dtype=torch.bool)
    else:
        ia = ia.view(-1)
    n_nodes = int(data.x.shape[0])
    dev = data.x.device

    if ia.numel() == n_nodes:
        return ia.to(device=dev)

    if hasattr(data, "batch") and data.batch is not None:
        return ia[data.batch].to(device=dev)

    if ia.numel() == 1:
        return torch.full((n_nodes,), bool(ia.item()), dtype=torch.bool, device=dev)

    return None
