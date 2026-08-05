"""Checkpoint selection must not be steered by GT-teacher-forced diagnostics.

`eval_continuous_window` is handed GT velocity and GT species blocks, so every `val_*` metric
is teacher-forced. Selecting on them picks the model that best replays GT rather than the one
that rolls out best unaided -- which is how a run whose canonical clot F1 was 0.234 came to look
like 0.844 (see docs/GENERALIZATION_PLAN.md s2b-quater).
"""

from __future__ import annotations

import pytest

from src.training.train_species_pushforward_continuous import (
    GT_FED_DIAGNOSTIC_KEYS,
    select_checkpoint_score,
)


def _row(**over: float) -> dict:
    row = {k: 0.5 for k in GT_FED_DIAGNOSTIC_KEYS}
    row["deploy_eval_t"] = 1.0
    row["loss"] = 1.0
    row.update(over)
    return row


def _score(row: dict, **kw) -> float:
    base = dict(
        held_out_val=True,
        physics_on=True,
        mat_precision_select=False,
        deploy_clot_score=0.3,
        deploy_mat_f1=0.4,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.75,
    )
    base.update(kw)
    return select_checkpoint_score(row, **base)[0]


def test_held_out_without_deploy_falls_back_to_loss() -> None:
    score, mode = select_checkpoint_score(
        _row(deploy_eval_t=-1.0, loss=2.5),
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.0,
        deploy_mat_f1=0.0,
        deploy_clot_pred_pos_frac=0.0,
        clot_weight=0.75,
    )
    assert mode == "lowest_loss"
    assert score == pytest.approx(-2.5)


@pytest.mark.parametrize("key", GT_FED_DIAGNOSTIC_KEYS)
def test_gt_fed_metrics_cannot_move_held_out_score(key: str) -> None:
    """Perfect GT-fed diagnostics must not raise the score by even a hair."""
    baseline = _score(_row(**{key: 0.0}))
    perfect = _score(_row(**{key: 1.0}))
    assert baseline == pytest.approx(perfect)


def test_held_out_score_tracks_deploy_clot() -> None:
    weak = _score(_row(), deploy_clot_score=0.20)
    strong = _score(_row(), deploy_clot_score=0.80)
    assert strong > weak


def test_held_out_soft_mass_penalty_prefers_lower_mass() -> None:
    """Soft mass gate: same clot score, lower mass_ratio must win."""
    base = dict(
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.36,
        deploy_mat_f1=0.38,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.75,
        select_clot_score_weight=0.90,
        select_mat_f1_weight=0.10,
        select_mass_soft_lambda=0.15,
        select_mass_soft_target=1.2,
        select_mass_hard_max=3.5,
        select_overpaint_lambda=0.0,
    )
    tight, mode_t = select_checkpoint_score(
        _row(), deploy_clot_mass_ratio=1.3, **base
    )
    spray, mode_s = select_checkpoint_score(
        _row(), deploy_clot_mass_ratio=2.4, **base
    )
    assert mode_t == "deploy_only"
    assert mode_s == "deploy_only"
    assert tight > spray


def test_held_out_hard_mass_reject_catastrophe() -> None:
    score, mode = select_checkpoint_score(
        _row(),
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.50,
        deploy_mat_f1=0.50,
        deploy_clot_pred_pos_frac=0.05,
        clot_weight=0.75,
        deploy_clot_mass_ratio=4.7,
        select_mass_hard_max=3.5,
        select_mass_soft_lambda=0.15,
    )
    assert mode == "deploy_only_mass_reject"
    assert score < -1.0e11


def test_held_out_legacy_formula_when_mass_gate_off() -> None:
    """Defaults preserve 0.70*score + 0.30*mat_f1 with no mass term."""
    score, mode = select_checkpoint_score(
        _row(),
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.40,
        deploy_mat_f1=0.20,
        deploy_clot_pred_pos_frac=0.20,
        clot_weight=0.75,
        deploy_clot_mass_ratio=4.0,
    )
    assert mode == "deploy_only"
    assert score == pytest.approx(0.70 * 0.40 + 0.30 * 0.20)


def test_held_out_downweights_mat_f1_vs_spray_comove() -> None:
    """Lower mat_f1 weight: spray-inflated mat_f1 cannot dominate clot score."""
    base = dict(
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.75,
        deploy_clot_mass_ratio=1.1,
        select_mass_soft_lambda=0.0,
        select_mass_hard_max=0.0,
    )
    # High mat_f1, low clot score vs low mat_f1, high clot score.
    sprayish, _ = select_checkpoint_score(
        _row(),
        deploy_clot_score=0.25,
        deploy_mat_f1=0.90,
        select_clot_score_weight=0.90,
        select_mat_f1_weight=0.10,
        **base,
    )
    precise, _ = select_checkpoint_score(
        _row(),
        deploy_clot_score=0.40,
        deploy_mat_f1=0.30,
        select_clot_score_weight=0.90,
        select_mat_f1_weight=0.10,
        **base,
    )
    assert precise > sprayish


def test_f1_min_hard_floor_rejects_a_checkpoint_that_only_looks_good_at_the_final_point() -> None:
    """s9.8: sliding-window grading exists to catch this -- t_final can hide a mid-rollout collapse."""
    score, mode = select_checkpoint_score(
        _row(deploy_clot_f1=0.60, deploy_clot_f1_min=0.20),
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.6,
        deploy_mat_f1=0.6,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.0,
        select_clot_f1_weight=1.0,
        select_f1_min_hard_floor=0.35,
    )
    assert mode == "deploy_only_f1_min_reject"
    assert score == -1.0e12


def test_f1_min_hard_floor_passes_a_checkpoint_that_holds_up_across_the_horizon() -> None:
    score, mode = select_checkpoint_score(
        _row(deploy_clot_f1=0.60, deploy_clot_f1_min=0.55),
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.6,
        deploy_mat_f1=0.6,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.0,
        select_clot_f1_weight=1.0,
        select_f1_min_hard_floor=0.35,
    )
    assert mode == "deploy_only"
    assert score == pytest.approx(0.6)


def test_f1_min_hard_floor_disabled_is_a_no_op_even_with_a_terrible_min() -> None:
    """0.0 (default) must reproduce every existing leg's behaviour exactly -- no new rejects."""
    score, mode = select_checkpoint_score(
        _row(deploy_clot_f1=0.60, deploy_clot_f1_min=0.01),
        held_out_val=True,
        physics_on=False,
        mat_precision_select=False,
        deploy_clot_score=0.6,
        deploy_mat_f1=0.6,
        deploy_clot_pred_pos_frac=0.01,
        clot_weight=0.0,
        select_clot_f1_weight=1.0,
        select_f1_min_hard_floor=0.0,
    )
    assert mode == "deploy_only"
    assert score == pytest.approx(0.6)


def test_front_speed_target_lambda_penalizes_overshoot_not_just_undershoot() -> None:
    """s9.10: select_front_speed_lambda (old, unchanged) rewards MORE speed, capped at 1.5 --
    dead once front_speed exceeds that (WG_stenosis_subcohort_ft_v2 saw 2.5-5.06 every epoch,
    a flat +0.30 with zero discrimination) and actively backwards past 1.0. The new
    select_front_speed_target_lambda penalizes distance from 1.0 in either direction instead."""
    base = dict(
        held_out_val=True, physics_on=False, mat_precision_select=False,
        deploy_clot_score=0.5, deploy_mat_f1=0.5, deploy_clot_pred_pos_frac=0.1,
        clot_weight=0.0, deploy_clot_mass_ratio=1.0, select_clot_f1_weight=1.0,
        select_front_speed_target_lambda=0.2,
    )
    at_target, _ = select_checkpoint_score(
        _row(deploy_clot_f1=0.5, mat_front_speed_ratio=1.0), **base
    )
    overshoot, _ = select_checkpoint_score(
        _row(deploy_clot_f1=0.5, mat_front_speed_ratio=5.0), **base
    )
    undershoot, _ = select_checkpoint_score(
        _row(deploy_clot_f1=0.5, mat_front_speed_ratio=0.3), **base
    )
    assert at_target == pytest.approx(0.5)  # no penalty exactly at target
    assert overshoot < at_target
    assert undershoot < at_target


def test_front_speed_target_lambda_disabled_is_a_no_op() -> None:
    score, mode = select_checkpoint_score(
        _row(deploy_clot_f1=0.5, mat_front_speed_ratio=99.0),
        held_out_val=True, physics_on=False, mat_precision_select=False,
        deploy_clot_score=0.5, deploy_mat_f1=0.5, deploy_clot_pred_pos_frac=0.1,
        clot_weight=0.0, deploy_clot_mass_ratio=1.0, select_clot_f1_weight=1.0,
        select_front_speed_target_lambda=0.0,
    )
    assert score == pytest.approx(0.5)


def test_fp_fn_imbalance_lambda_is_symmetric_unlike_the_old_fn_fp_lambda() -> None:
    """s9.10: select_fn_fp_lambda (old, unchanged) only fires FN-heavy -- zero signal in the
    FP-heavy/overspray regime (fn=110-294 fp seen on v2, term was 0.000 every epoch). The new
    select_fp_fn_imbalance_lambda penalizes |fn-fp| imbalance in either direction equally."""
    base = dict(
        held_out_val=True, physics_on=False, mat_precision_select=False,
        deploy_clot_score=0.5, deploy_mat_f1=0.5, deploy_clot_pred_pos_frac=0.1,
        clot_weight=0.0, deploy_clot_mass_ratio=1.0, select_clot_f1_weight=1.0,
        select_fp_fn_imbalance_lambda=0.2,
    )
    balanced, _ = select_checkpoint_score(
        _row(deploy_clot_f1=0.5, deploy_clot_fn=50, deploy_clot_fp=50), **base
    )
    fp_heavy, _ = select_checkpoint_score(
        _row(deploy_clot_f1=0.5, deploy_clot_fn=5, deploy_clot_fp=200), **base
    )
    fn_heavy, _ = select_checkpoint_score(
        _row(deploy_clot_f1=0.5, deploy_clot_fn=200, deploy_clot_fp=5), **base
    )
    assert balanced == pytest.approx(0.5)  # no penalty when balanced
    assert fp_heavy < balanced
    assert fn_heavy < balanced
    assert fp_heavy == pytest.approx(fn_heavy)  # symmetric


def test_fp_fn_imbalance_lambda_disabled_is_a_no_op() -> None:
    score, mode = select_checkpoint_score(
        _row(deploy_clot_f1=0.5, deploy_clot_fn=1000, deploy_clot_fp=1),
        held_out_val=True, physics_on=False, mat_precision_select=False,
        deploy_clot_score=0.5, deploy_mat_f1=0.5, deploy_clot_pred_pos_frac=0.1,
        clot_weight=0.0, deploy_clot_mass_ratio=1.0, select_clot_f1_weight=1.0,
        select_fp_fn_imbalance_lambda=0.0,
    )
    assert score == pytest.approx(0.5)


def test_dead_phi_term_cannot_be_masked_when_val_in_train() -> None:
    """Regression guard: the physics branch is still GT-fed, so it stays opt-in only."""
    score, mode = select_checkpoint_score(
        _row(val_clot_phi_f1=0.0, val_growth_f1=0.2666666, val_state_f1=0.9130434, val_growth_mat_f1=0.2666666),
        held_out_val=False,
        physics_on=True,
        mat_precision_select=False,
        deploy_clot_score=0.877,
        deploy_mat_f1=0.5825,
        deploy_clot_pred_pos_frac=0.006,
        clot_weight=0.75,
    )
    assert mode == "physics_gt_fed"
    # Reproduces the 0.2303 best_score recorded for wall_family7_cold ep10.
    assert score == pytest.approx(0.2303, abs=1e-4)
    # And it is blind to the deploy clot score entirely.
    assert score < 0.877
