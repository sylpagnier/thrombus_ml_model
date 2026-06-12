"""Deploy score: symmetric coverage + band F1 for checkpoint selection."""

from __future__ import annotations

from src.core_physics.clot_temporal_growth_rules import (
    compute_deploy_score,
    deploy_score_from_eval_row,
    symmetric_coverage_match,
)


def test_symmetric_coverage_penalizes_under_and_over_equally():
    gt = 0.20
    on_target = symmetric_coverage_match(0.20, gt)
    under = symmetric_coverage_match(0.0, gt)
    over = symmetric_coverage_match(1.0, gt)
    assert on_target == 1.0
    assert abs(under - over) < 1e-6
    assert under < 0.05
    assert over < 0.05


def test_deploy_score_prefers_inc40_over_accum_wall_paint():
    """inc40: matched pred+; accum: over-paints vs GT."""
    gt = 0.18
    accum = compute_deploy_score(
        p007_tfinal_shape=0.593,
        p007_early_pred=0.23,
        p007_tfinal_bal=0.10,
        p007_pred=0.53,
        tfinal_band_f1=0.35,
        tfinal_gt_pos_frac=gt,
        early_gt_pos_frac=0.0,
    )
    inc40 = compute_deploy_score(
        p007_tfinal_shape=0.572,
        p007_early_pred=0.0,
        p007_tfinal_bal=0.27,
        p007_pred=0.22,
        tfinal_band_f1=0.50,
        tfinal_gt_pos_frac=gt,
        early_gt_pos_frac=0.0,
    )
    assert inc40 > accum


def test_deploy_score_zero_commit_not_inflated():
    """Step-7 pattern: high legacy shape terms but no band commits."""
    zero_commit = compute_deploy_score(
        p007_tfinal_shape=0.374,
        p007_early_pred=0.0,
        p007_tfinal_bal=0.22,
        p007_pred=0.0,
        tfinal_band_f1=0.0,
        tfinal_gt_pos_frac=0.15,
        early_gt_pos_frac=0.0,
    )
    good = compute_deploy_score(
        p007_tfinal_shape=0.45,
        p007_early_pred=0.0,
        p007_tfinal_bal=0.25,
        p007_pred=0.12,
        tfinal_band_f1=0.50,
        tfinal_gt_pos_frac=0.15,
        early_gt_pos_frac=0.0,
    )
    assert zero_commit < 0.15
    assert good > zero_commit


def test_deploy_score_from_eval_row():
    row = {
        "tfinal_clot_shape": 0.45,
        "early_mean_pred_frac": 0.0,
        "tfinal_clot_shape_bal": 0.25,
        "tfinal_band_pred_frac": 0.12,
        "tfinal_band_f1": 0.50,
        "tfinal_gt_pos_frac": 0.15,
        "early_mean_gt_pos_frac": 0.0,
    }
    score = deploy_score_from_eval_row(row)
    assert 0.0 <= score <= 1.0
    assert score > 0.35
