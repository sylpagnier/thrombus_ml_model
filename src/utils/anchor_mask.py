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


def wall_wss_supervision_mask(data) -> torch.Tensor:
    """Wall nodes used for WSS supervision: exclude wall vertices adjacent to inlet/outlet (junction artifacts)."""
    if not hasattr(data, "mask_wall"):
        return torch.zeros(int(data.num_nodes), dtype=torch.bool, device=data.x.device)
    mw = data.mask_wall.view(-1).bool()
    mi = getattr(data, "mask_inlet", None)
    mo = getattr(data, "mask_outlet", None)
    if mi is None or mo is None:
        return mw
    mi = mi.view(-1).bool()
    mo = mo.view(-1).bool()
    if not mi.any() and not mo.any():
        return mw
    row, col = data.edge_index
    io = mi | mo
    # Neighbor of an IO node along the mesh edge: that wall vertex is a cap/junction
    excl_nodes = torch.zeros_like(mw)
    excl_nodes[col[io[row] & mw[col]]] = True
    excl_nodes[row[io[col] & mw[row]]] = True
    return mw & ~excl_nodes
