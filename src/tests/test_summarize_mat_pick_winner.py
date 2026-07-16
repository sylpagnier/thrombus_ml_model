"""Tests for FP-aware winner selection in summarize_mat_only_full."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.summarize_mat_only_full import _pick_winner  # noqa: E402


def _row(leg: str, **metrics: float) -> dict:
    return {"leg": leg, "cohort_mean": dict(metrics)}


def test_pick_winner_minimize_fp_breaks_tie_on_f1():
    rows = [
        _row(
            "W",
            deploy_clot_f1=0.80,
            deploy_clot_score=0.98,
            clot_fp_p90=70,
            clot_fp_median=35,
            mat_overpaint_per_gt=0.01,
        ),
        _row(
            "WC",
            deploy_clot_f1=0.80,
            deploy_clot_score=0.95,
            clot_fp_p90=34,
            clot_fp_median=14,
            mat_overpaint_per_gt=0.01,
        ),
    ]
    winner = _pick_winner(
        rows,
        rank_by=["deploy_clot_f1", "clot_fp_p90", "clot_fp_median", "deploy_clot_score"],
        minimize_by={"clot_fp_p90", "clot_fp_median"},
        max_overpaint_per_gt=0.02,
        max_clot_fp_p90=None,
        max_clot_fp_early_mean=None,
    )
    assert winner is not None
    assert winner["leg"] == "WC"


def test_prefer_wc_within_f1_eps():
    rows = [
        _row("W_mat_flow_stagnation", deploy_clot_f1=0.805, deploy_clot_score=0.98, clot_fp_p90=70),
        _row("WC_mat_flow_dynamic", deploy_clot_f1=0.800, deploy_clot_score=0.95, clot_fp_p90=34),
    ]
    winner = _pick_winner(
        rows,
        rank_by=["deploy_clot_f1", "clot_fp_p90"],
        minimize_by={"clot_fp_p90"},
        max_overpaint_per_gt=None,
        max_clot_fp_p90=None,
        max_clot_fp_early_mean=None,
        prefer_leg="WC_mat_flow_dynamic",
        tie_eps=0.008,
    )
    assert winner is not None
    assert winner["leg"] == "WC_mat_flow_dynamic"


def test_overpaint_filter_excludes_leg():
    rows = [
        _row("W", deploy_clot_f1=0.90, mat_overpaint_per_gt=0.05),
        _row("WC", deploy_clot_f1=0.70, mat_overpaint_per_gt=0.01),
    ]
    winner = _pick_winner(
        rows,
        rank_by=["deploy_clot_f1"],
        minimize_by=set(),
        max_overpaint_per_gt=0.02,
        max_clot_fp_p90=None,
        max_clot_fp_early_mean=None,
    )
    assert winner is not None
    assert winner["leg"] == "WC"
