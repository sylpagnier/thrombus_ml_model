"""Commit-order probe primitives (docs/WALL_MODEL_PLAN.md s4 Step 1b)."""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation.commit_time import (
    first_commit_step,
    flow_tie_pairs,
    predicted_components,
    rank_auc,
)


def test_first_commit_step_picks_first_crossing():
    # 3 nodes x 4 steps: node0 commits at t=1, node1 at t=3, node2 never.
    phi = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.1],
            [0.9, 0.2, 0.4],
            [0.9, 0.8, 0.49],
        ]
    )
    got = first_commit_step(phi)
    assert got.tolist() == [1, 3, 4]  # never -> T, sorts after every real commit


def test_first_commit_step_ignores_later_dropouts():
    """A node that crosses then falls back still counts as committed at the first crossing."""
    phi = torch.tensor([[0.0], [0.7], [0.1], [0.9]])
    assert first_commit_step(phi).tolist() == [1]


def test_rank_auc_direction_and_ties():
    assert rank_auc([1.0], [5.0]) == 1.0  # lower = "true" direction
    assert rank_auc([5.0], [1.0]) == 0.0
    assert rank_auc([2.0], [2.0]) == 0.5
    assert np.isnan(rank_auc([], [1.0]))


def test_rank_auc_size_weighting_follows_the_big_components():
    # One small TP ranks right, one big TP ranks wrong -> unweighted 0.5, weighted worse.
    pos, neg = [1.0, 9.0], [5.0]
    assert rank_auc(pos, neg) == 0.5
    assert rank_auc(pos, neg, w_pos=[1.0, 99.0], w_neg=[1.0]) < 0.05


def test_rank_auc_pair_mask_scores_only_masked_pairs():
    pos, neg = [1.0, 9.0], [5.0, 5.0]
    mask = np.array([[True, False], [False, False]])  # only the correctly-ranked pair
    assert rank_auc(pos, neg, pair_mask=mask) == 1.0
    assert np.isnan(rank_auc(pos, neg, pair_mask=np.zeros((2, 2), dtype=bool)))


def test_flow_tie_pairs_matches_the_s29_gap():
    """patient037's 0.048 vs 0.047 is a tie at the 5% default; 021's clean gap is not."""
    ties = flow_tie_pairs([0.048], [0.047], rel_tol=0.05)
    assert bool(ties[0, 0]) is True
    assert bool(flow_tie_pairs([0.065], [0.120], rel_tol=0.05)[0, 0]) is False


def test_predicted_components_splits_disconnected_pockets():
    # 0-1 connected, 3 isolated, 2 not predicted.
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])
    mask = np.array([True, True, False, True])
    comps = sorted(predicted_components(mask, edge_index), key=len, reverse=True)
    assert [c.tolist() for c in comps] == [[0, 1], [3]]


def test_predicted_components_empty_mask():
    edge_index = np.array([[0, 1], [1, 0]])
    assert predicted_components(np.zeros(2, dtype=bool), edge_index) == []
