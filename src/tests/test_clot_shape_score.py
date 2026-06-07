"""Tests for location-weighted clot shape score."""

from __future__ import annotations

import numpy as np
import torch

from src.config import PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.evaluation.clot_shape_score import (
    compute_clot_shape_metrics,
    graph_hop_distance_from_seeds,
    mu_clot_binary_mask,
)


def _chain_graph(n: int) -> torch.Tensor:
    edges = []
    for i in range(n - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges, dtype=torch.long).t()


def _state_with_mu(n: int, mu_si: np.ndarray, u: float = 1.0) -> torch.Tensor:
    phys = PhysicsConfig(phase="biochem")
    y = torch.zeros(n, 4, dtype=torch.float32)
    y[:, 0] = u
    y[:, 1] = 0.0
    y[:, 2] = 0.0
    y[:, STATE_CHANNEL_MU_EFF_ND] = torch.tensor(
        [phys.viscosity_si_to_nd(float(m)) for m in mu_si],
        dtype=torch.float32,
    )
    return y


def test_perfect_overlap_scores_one():
    phys = PhysicsConfig(phase="biochem")
    n = 5
    mu = np.array([0.04, 0.04, 0.08, 0.08, 0.04], dtype=np.float32)
    gt = _state_with_mu(n, mu)
    pred = gt.clone()
    ei = _chain_graph(n)
    m = compute_clot_shape_metrics(
        pred_state=pred,
        gt_state=gt,
        edge_index=ei,
        phys_cfg=phys,
        mu_thresh_si=0.055,
    )
    assert m["clot_dice"] == 1.0
    assert m["clot_shape"] == 1.0
    assert m["clot_fp"] == 0
    assert m["clot_fn"] == 0


def test_distant_fp_hurts_more_than_adjacent_fp():
    phys = PhysicsConfig(phase="biochem")
    n = 7
    gt_mu = np.full(n, 0.04, dtype=np.float32)
    gt_mu[3] = 0.08
    gt = _state_with_mu(n, gt_mu)

    pred_adj = _state_with_mu(n, gt_mu.copy())
    pred_adj_mu = gt_mu.copy()
    pred_adj_mu[4] = 0.08
    pred_adj[:, STATE_CHANNEL_MU_EFF_ND] = torch.tensor(
        [phys.viscosity_si_to_nd(float(m)) for m in pred_adj_mu],
        dtype=torch.float32,
    )

    pred_dist = _state_with_mu(n, gt_mu.copy())
    pred_dist_mu = gt_mu.copy()
    pred_dist_mu[0] = 0.08
    pred_dist[:, STATE_CHANNEL_MU_EFF_ND] = torch.tensor(
        [phys.viscosity_si_to_nd(float(m)) for m in pred_dist_mu],
        dtype=torch.float32,
    )

    ei = _chain_graph(n)
    m_adj = compute_clot_shape_metrics(
        pred_state=pred_adj, gt_state=gt, edge_index=ei, phys_cfg=phys, mu_thresh_si=0.055
    )
    m_dist = compute_clot_shape_metrics(
        pred_state=pred_dist, gt_state=gt, edge_index=ei, phys_cfg=phys, mu_thresh_si=0.055
    )
    assert m_adj["clot_fp_adjacent"] == 1
    assert m_dist["clot_fp_distant"] >= 1
    assert m_adj["clot_shape"] > m_dist["clot_shape"]


def test_hop_distance_bfs():
    n = 5
    ei = _chain_graph(n)
    seed = np.array([False, False, True, False, False])
    dist = graph_hop_distance_from_seeds(ei, n, seed)
    assert dist[2] == 0
    assert dist[1] == 1
    assert dist[3] == 1
    assert dist[0] == 2
    assert dist[4] == 2


def test_mu_clot_binary_mask():
    mu = torch.tensor([0.04, 0.06, 0.08])
    mask = mu_clot_binary_mask(mu, 0.055)
    assert mask.tolist() == [False, True, True]
