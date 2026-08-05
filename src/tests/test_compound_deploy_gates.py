"""Compound deploy gate logic (precision-first spray)."""

from __future__ import annotations

from src.evaluation.compound_deploy_gates import gate_compound_eval, idle_spray_ok


def _row(**kwargs: float) -> dict:
    base = {
        "deploy_clot_f1": 0.8,
        "deploy_clot_score": 0.82,
        "deploy_clot_relaxed_prec": 1.0,
        "deploy_clot_offwall_relaxed_f1": 0.5,
        "deploy_clot_offwall_strict_f1_hop_ge2": 0.2,
        "deploy_clot_offwall_n_pred_hop_ge2": 10.0,
        "deploy_clot_offwall_n_gt_hop_ge2": 20.0,
    }
    base.update(kwargs)
    return base


def test_idle_spray_allows_paint_with_good_precision():
    per = {"patient002": _row(deploy_clot_offwall_n_gt_hop_ge2=0.0, deploy_clot_offwall_n_pred_hop_ge2=5.0)}
    ok, reason = idle_spray_ok(per, "patient002")
    assert ok
    assert reason == "paint_ok_precision"


def test_idle_spray_fails_on_precision_collapse():
    per = {
        "patient002": _row(
            deploy_clot_offwall_n_gt_hop_ge2=0.0,
            deploy_clot_offwall_n_pred_hop_ge2=50.0,
            deploy_clot_f1=0.5,
            deploy_clot_relaxed_prec=0.6,
        )
    }
    ok, _ = idle_spray_ok(per, "patient002")
    assert not ok


def test_gate_compound_target_hit():
    per = {
        "patient001": _row(deploy_clot_offwall_n_pred_hop_ge2=30.0, deploy_clot_offwall_n_gt_hop_ge2=50.0),
        "patient007": _row(deploy_clot_offwall_n_pred_hop_ge2=20.0, deploy_clot_offwall_n_gt_hop_ge2=40.0),
        "patient002": _row(deploy_clot_offwall_n_gt_hop_ge2=0.0, deploy_clot_offwall_n_pred_hop_ge2=0.0),
        "patient004": _row(deploy_clot_offwall_n_gt_hop_ge2=0.0, deploy_clot_offwall_n_pred_hop_ge2=0.0),
        "patient008": _row(deploy_clot_offwall_n_gt_hop_ge2=0.0, deploy_clot_offwall_n_pred_hop_ge2=0.0),
    }
    gate = gate_compound_eval(per, wall_floor_f1=0.78)
    assert gate["gates"]["lumen_teachers_open"]
    assert gate["gates"]["idle_precision_ok"]
    assert gate["target_hit"]
