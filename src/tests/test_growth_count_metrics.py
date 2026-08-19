"""Guards for the growth-count metric.

This metric exists because the overlap score was discontinuous in commit time
(PHASE6_RESULTS 15.3).  The guards that matter are therefore: it must be exactly zero for a
perfect prediction, it must move CONTINUOUSLY as a prediction slides in time (the property
the old metric lacked), and the count floor must genuinely be a floor.
"""

from __future__ import annotations

import numpy as np

from src.core_physics.growth_count_metrics import (
    count_curve,
    count_optimal_onset,
    growth_error,
)

NT = 50


def test_perfect_prediction_scores_zero():
    gt = np.array([10, 20, 30, -1, 40])
    assert growth_error(gt, gt, NT)["growth_l1"] == 0.0


def test_metric_is_continuous_in_commit_time():
    """One node slid one step at a time must change the error smoothly.

    The retired overlap score jumped 1.0 -> 0.0 on a single step of the same slide; that
    discontinuity is what made every realistic timing arm score worse than the flash.
    """
    gt = np.array([10, 12, 14, 16])
    errs = []
    for shift in range(0, 20):
        errs.append(growth_error(gt + shift, gt, NT)["growth_l1"])
    d = np.diff(errs)
    assert np.all(d >= -1e-12)                      # monotone as it drifts away
    assert d.max() < 0.12                           # and no cliff


def test_count_curve_is_cumulative_and_ignores_never_committing_nodes():
    on = np.array([2, 2, 5, -1])
    c = count_curve(on, 8)
    assert list(c) == [0, 0, 2, 2, 2, 3, 3, 3]


def test_final_error_is_signed_and_reports_over_prediction():
    gt = np.array([5, 6, 7])
    over = np.array([5, 6, 7, 8, 9])
    under = np.array([5, 6])
    assert growth_error(over, gt, NT)["final_err"] > 0
    assert growth_error(under, gt, NT)["final_err"] < 0


def test_count_floor_is_a_floor_on_its_own_mask():
    """No onset assignment on the same committed set may beat the count-optimal one."""
    rng = np.random.default_rng(0)
    gt = np.sort(rng.integers(5, 40, 30))
    gt_full = -np.ones(60, dtype=int)
    gt_full[:30] = gt
    S = np.zeros(60, dtype=bool)
    S[10:40] = True                                  # 30 committed nodes, same count as GT
    floor = count_optimal_onset(S, gt_full, NT)
    f_err = growth_error(floor, gt_full, NT)["growth_l1"]
    for seed in range(20):
        r = np.random.default_rng(seed)
        cand = -np.ones(60, dtype=int)
        cand[S] = r.integers(0, NT, int(S.sum()))
        assert growth_error(cand, gt_full, NT)["growth_l1"] >= f_err - 1e-12


def test_count_floor_reaches_zero_when_the_mask_size_matches():
    gt_full = -np.ones(40, dtype=int)
    gt_full[:12] = np.arange(2, 26, 2)
    S = np.zeros(40, dtype=bool)
    S[20:32] = True                                  # 12 nodes, exactly N_gt
    floor = count_optimal_onset(S, gt_full, NT)
    assert growth_error(floor, gt_full, NT)["growth_l1"] < 1e-9


def test_empty_gt_returns_nan_rather_than_a_free_score():
    """6.2: an empty-GT vessel must not be scorable at all."""
    gt = -np.ones(10, dtype=int)
    assert np.isnan(growth_error(np.array([1, 2]), gt, NT)["growth_l1"])
