"""Trajectory eval metrics for clot growth rollouts."""

from __future__ import annotations

import numpy as np
import torch

from src.training.clot_growth_eval import trajectory_score_from_row


def test_trajectory_score_weights_ring_penalty() -> None:
    good = {
        "mean_band_f1": 0.5,
        "mean_clot_shape": 0.5,
        "early_recall_cov": 0.5,
        "tfinal_wall_ring_frac": 0.0,
    }
    bad_ring = {**good, "tfinal_wall_ring_frac": 1.0}
    assert trajectory_score_from_row(good) > trajectory_score_from_row(bad_ring)


def test_trajectory_score_nan_safe() -> None:
    row = {
        "mean_band_f1": float("nan"),
        "mean_clot_shape": float("nan"),
        "early_recall_cov": float("nan"),
        "tfinal_wall_ring_frac": 0.2,
    }
    s = trajectory_score_from_row(row)
    assert 0.0 <= s <= 1.0
