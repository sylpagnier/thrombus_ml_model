"""Unit tests for off-wall relaxed clot metrics.

Regression: `offwall_relaxed_f1` must not reward off-wall recall using wall
predictions via relaxed dilation. Only off-wall pred/GT masks should generate
the relaxed neighborhoods.
"""

from __future__ import annotations

import torch

from src.evaluation.clot_relaxed_metrics import compute_clot_relaxed_metrics


def _line_edge_index(n: int) -> torch.Tensor:
    """Undirected line graph edge_index for `n` nodes."""
    src: list[int] = []
    dst: list[int] = []
    for i in range(n - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
    return torch.tensor([src, dst], dtype=torch.long)


def test_offwall_relaxed_ignores_wall_predictions():
    # Nodes: 0-1-2-3-4-5-6 (line)
    edge_index = _line_edge_index(7)

    # Wall mask: node 3 only. All other nodes are "off-wall".
    wall_mask = torch.zeros(7, dtype=torch.bool)
    wall_mask[3] = True

    # GT clots on off-wall nodes: 1 and 6.
    phi_gt = torch.zeros(7, dtype=torch.float32)
    phi_gt[1] = 1.0
    phi_gt[6] = 1.0

    # Predictions:
    # - pred off-wall node 4 (can relax-match gt node 6 within 2 hops)
    # - pred wall node 3 (can relax-match gt node 1 within 2 hops)
    phi_pred = torch.zeros(7, dtype=torch.float32)
    phi_pred[4] = 1.0
    phi_pred[3] = 1.0

    # Use relax_hops=2 and beta=0.5 (the default in the metric code).
    res = compute_clot_relaxed_metrics(
        phi_pred,
        phi_gt,
        edge_index,
        relax_hops=2,
        f_beta=0.5,
        wall_mask=wall_mask,
    )

    # With off-wall-only dilation:
    # - off-wall precision: pred_off={4} overlaps gt_dil_off -> 1.0
    # - off-wall recall: pred_dil_off from {4} covers gt_off={6} but not {1} -> 0.5
    # beta=0.5 => F_beta = (1+0.25)*P*R/(0.25*P + R) = 1.25*1*0.5/(0.25+0.5) = 0.8333...
    assert abs(res["offwall_relaxed_prec"] - 1.0) < 1e-6
    assert abs(res["offwall_relaxed_rec"] - 0.5) < 1e-6
    assert abs(res["offwall_relaxed_f1"] - (5.0 / 6.0)) < 1e-4
    # Hop-stratified: pred offwall at node 4 is hop1 from wall node 3.
    assert res["offwall_n_pred_hop1"] == 1.0
    assert res["offwall_n_pred_hop_ge2"] == 0.0


def test_hop_ge2_strict_counts():
    edge_index = _line_edge_index(7)
    wall_mask = torch.zeros(7, dtype=torch.bool)
    wall_mask[0] = True
    phi_gt = torch.zeros(7)
    phi_gt[2] = 1.0  # hop 2
    phi_gt[3] = 1.0  # hop 3
    phi_pred = torch.zeros(7)
    phi_pred[2] = 1.0  # hit hop2
    res = compute_clot_relaxed_metrics(
        phi_pred, phi_gt, edge_index, relax_hops=2, f_beta=1.0, wall_mask=wall_mask
    )
    assert res["offwall_n_gt_hop_ge2"] == 2.0
    assert res["offwall_n_pred_hop_ge2"] == 1.0
    assert res["offwall_strict_f1_hop2"] > 0.0

