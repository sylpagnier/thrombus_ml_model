"""Scoring-parity guards (WALL_MODEL_PLAN.md 20.1).

Two tools scored the SAME predictions differently for the whole of sections 9-19, because
`deploy_clot_score` depends on ambient state (typed runtime binding + deploy protocol env) that
one tool established and the other did not. Strict `deploy_clot_f1` was bit-identical
throughout, which is exactly why it went unnoticed.

These tests pin the two invariants that would have caught it.
"""
from __future__ import annotations

import inspect

import pytest

from src.evaluation.clot_relaxed_metrics import (
    relaxed_prec_floor_score,
    scoring_fingerprint,
)


def test_scoring_fingerprint_reports_binding_state():
    """The fingerprint must expose whether a runtime is bound -- that is the failure mode."""
    fp = scoring_fingerprint()
    for key in ("clout_score_mode", "clout_prec_rec_floor", "guide_relax_hops",
                "guide_f_beta", "empty_gt_fp_tol", "runtime_bound"):
        assert key in fp, f"fingerprint missing {key}"
    assert isinstance(fp["runtime_bound"], bool)


def test_fingerprint_tracks_the_active_runtime():
    """Binding a runtime must be visible in the fingerprint, so tools can compare before scores."""
    from src.architecture.runtime_config import use_biochem_runtime
    from src.biochem_gnn.config import build_train_recipe_configs

    _pf, rt = build_train_recipe_configs()
    unbound = scoring_fingerprint()
    with use_biochem_runtime(rt):
        bound = scoring_fingerprint()
    assert unbound["runtime_bound"] is False
    assert bound["runtime_bound"] is True


def test_both_canonical_entry_points_apply_the_deploy_protocol():
    """`canonical_grade_series` must bind the same protocol as `canonical_deploy_clot_metrics`.

    The regression was that the gate sweep called `grade_deploy_clot_series` directly, skipping
    `bind_canonical_deploy_protocol`. Assert both canonical wrappers reference it.
    """
    from src.evaluation import canonical_clot_eval as cce

    for fn_name in ("canonical_deploy_clot_metrics", "canonical_grade_series"):
        fn = getattr(cce, fn_name)
        src = inspect.getsource(fn)
        assert "bind_canonical_deploy_protocol" in src, (
            f"{fn_name} does not bind the canonical deploy protocol; scores it produces are "
            "not comparable with the other canonical entry point"
        )


def test_gate_sweep_uses_the_canonical_grader():
    """The sweep must not call the raw grader again."""
    from pathlib import Path

    src = Path("scripts/diag_regime_gate_sweep.py").read_text(encoding="utf-8")
    assert "canonical_grade_series(" in src
    # the raw grader may still be imported transitively, but must not be *called* here
    assert "grade_deploy_clot_series(" not in src, (
        "diag_regime_gate_sweep.py calls grade_deploy_clot_series directly, bypassing the "
        "canonical deploy protocol (WALL_MODEL_PLAN.md 20.1)"
    )


@pytest.mark.parametrize(
    "prec,rec,floor_expected",
    [(0.8, 1.0, 0.8), (0.8, 0.0, 0.0)],
)
def test_relaxed_prec_floor_is_inert_at_full_recall(prec, rec, floor_expected):
    """At rec >= floor the score is just precision -- so the floor CANNOT explain a score gap.

    Recorded because 13.4a wrongly attributed a cross-tool discrepancy to this constant while
    every run had relaxed recall ~1.000, where the floor provably does nothing.
    """
    assert relaxed_prec_floor_score(prec, rec) == pytest.approx(floor_expected)
