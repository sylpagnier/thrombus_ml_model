"""Compound deploy evaluation gates (precision-first, wall guardrail).

Primary:
  deploy_clot_score, deploy_clot_offwall_relaxed_f1

Lumen quality:
  deploy_clot_offwall_strict_f1_hop_ge2, ge2 recall (pred/gt)

Spray (reframed):
  idle vessels (no GT lumen): fail only if relaxed prec / F1 collapse under paint

Guardrail:
  mean deploy_clot_f1 >= wall_floor - delta (do not destroy wall)
"""

from __future__ import annotations

from typing import Any

# Orig10 focus + standard idle spray negatives
DEFAULT_TEACHER_ANCHORS = ("patient001", "patient007")
DEFAULT_IDLE_ANCHORS = ("patient002", "patient004", "patient008")
DEFAULT_FOCUS_ANCHORS = (
    "patient001",
    "patient002",
    "patient007",
    "patient008",
    "patient010",
)

# Idle: allow some hop_ge2 paint if precision stays healthy
IDLE_MIN_F1 = 0.70
IDLE_MIN_RELAXED_PREC = 0.85

# Wall guardrail vs wall-alone deploy
DEFAULT_WALL_FLOOR_DELTA = 0.04
DEFAULT_MIN_CLOUT_SCORE = 0.78
DEFAULT_MIN_OFFWALL_RELAXED_F1 = 0.40


def _f(x: Any, key: str, default: float = 0.0) -> float:
    return float((x or {}).get(key, default) or default)


def anchor_metrics(per_anchor: dict, anc: str) -> dict[str, float]:
    r = per_anchor.get(anc) or {}
    ge2_pred = _f(r, "deploy_clot_offwall_n_pred_hop_ge2")
    ge2_gt = _f(r, "deploy_clot_offwall_n_gt_hop_ge2")
    return {
        "f1": _f(r, "deploy_clot_f1"),
        "score": _f(r, "deploy_clot_score"),
        "relaxed_prec": _f(r, "deploy_clot_relaxed_prec"),
        "offwall_relaxed_f1": _f(r, "deploy_clot_offwall_relaxed_f1"),
        "hop_ge2_strict": _f(r, "deploy_clot_offwall_strict_f1_hop_ge2"),
        "ge2_pred": ge2_pred,
        "ge2_gt": ge2_gt,
        "ge2_recall": ge2_pred / ge2_gt if ge2_gt > 0.5 else None,
    }


def idle_spray_ok(per_anchor: dict, anc: str) -> tuple[bool, str]:
    """True if idle vessel passes reframed spray check."""
    m = anchor_metrics(per_anchor, anc)
    if m["ge2_gt"] > 0.5:
        return True, "has_gt_lumen"
    if m["ge2_pred"] <= 0.5:
        return True, "clean"
    if m["f1"] >= IDLE_MIN_F1 and m["relaxed_prec"] >= IDLE_MIN_RELAXED_PREC:
        return True, "paint_ok_precision"
    return False, f"f1={m['f1']:.3f} rprec={m['relaxed_prec']:.3f}"


def gate_compound_eval(
    per_anchor: dict,
    mean: dict | None = None,
    *,
    wall_floor_f1: float | None = None,
    wall_floor_delta: float = DEFAULT_WALL_FLOOR_DELTA,
    teacher_anchors: tuple[str, ...] = DEFAULT_TEACHER_ANCHORS,
    idle_anchors: tuple[str, ...] = DEFAULT_IDLE_ANCHORS,
    focus_anchors: tuple[str, ...] = DEFAULT_FOCUS_ANCHORS,
    min_clot_score: float = DEFAULT_MIN_CLOUT_SCORE,
    min_offwall_relaxed_f1: float = DEFAULT_MIN_OFFWALL_RELAXED_F1,
) -> dict[str, Any]:
    """Score a compound eval report (per_anchor + optional mean dict)."""
    if mean is None:
        keys = (
            "deploy_clot_f1",
            "deploy_clot_score",
            "deploy_clot_offwall_relaxed_f1",
            "deploy_clot_offwall_strict_f1_hop_ge2",
            "deploy_clot_offwall_n_pred_hop_ge2",
            "deploy_clot_offwall_n_gt_hop_ge2",
        )
        mean = {k: sum(_f(per_anchor.get(a), k) for a in per_anchor) / max(len(per_anchor), 1) for k in keys}

    mean_f1 = _f(mean, "deploy_clot_f1")
    mean_score = _f(mean, "deploy_clot_score")
    mean_off_rel = _f(mean, "deploy_clot_offwall_relaxed_f1")
    mean_ge2_strict = _f(mean, "deploy_clot_offwall_strict_f1_hop_ge2")
    sum_ge2_pred = sum(_f(per_anchor.get(a), "deploy_clot_offwall_n_pred_hop_ge2") for a in per_anchor)
    sum_ge2_gt = sum(_f(per_anchor.get(a), "deploy_clot_offwall_n_gt_hop_ge2") for a in per_anchor)
    ge2_recall = sum_ge2_pred / sum_ge2_gt if sum_ge2_gt > 0.5 else 0.0

    focus = {a: anchor_metrics(per_anchor, a) for a in focus_anchors if a in per_anchor}

    opened_teachers = all(focus.get(a, {}).get("ge2_pred", 0) > 0.5 for a in teacher_anchors if a in per_anchor)

    idle_detail: dict[str, dict] = {}
    idle_ok = True
    for anc in idle_anchors:
        if anc not in per_anchor:
            continue
        ok, reason = idle_spray_ok(per_anchor, anc)
        idle_detail[anc] = {"ok": ok, "reason": reason, **anchor_metrics(per_anchor, anc)}
        if not ok:
            idle_ok = False

    wall_ok = True
    wall_floor_used = None
    if wall_floor_f1 is not None:
        wall_floor_used = float(wall_floor_f1) - float(wall_floor_delta)
        wall_ok = mean_f1 >= wall_floor_used

    gates = {
        "primary_score_ok": mean_score >= min_clot_score,
        "primary_offwall_relaxed_ok": mean_off_rel >= min_offwall_relaxed_f1,
        "lumen_teachers_open": opened_teachers,
        "lumen_ge2_strict_up": mean_ge2_strict >= 0.12,
        "idle_precision_ok": idle_ok,
        "wall_guardrail_ok": wall_ok,
    }
    target_hit = all(gates.values())

    return {
        "mean_f1": mean_f1,
        "mean_score": mean_score,
        "mean_offwall_relaxed_f1": mean_off_rel,
        "mean_hop_ge2_strict": mean_ge2_strict,
        "sum_ge2_pred": sum_ge2_pred,
        "sum_ge2_gt": sum_ge2_gt,
        "ge2_recall": ge2_recall,
        "focus": focus,
        "idle_detail": idle_detail,
        "gates": gates,
        "spray_clean": idle_ok,
        "target_hit": target_hit,
        "wall_floor_f1": wall_floor_f1,
        "wall_floor_threshold": wall_floor_used,
        "thresholds": {
            "min_clot_score": min_clot_score,
            "min_offwall_relaxed_f1": min_offwall_relaxed_f1,
            "wall_floor_delta": wall_floor_delta,
            "idle_min_f1": IDLE_MIN_F1,
            "idle_min_relaxed_prec": IDLE_MIN_RELAXED_PREC,
        },
    }


def gate_compound_eval_report(
    report: dict,
    *,
    wall_floor_f1: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Gate from eval_mat_growth_simple JSON (uses report['simple'])."""
    simple = report.get("simple") or report
    per = simple.get("per_anchor") or {}
    mean = simple.get("mean") or {}
    out = gate_compound_eval(per, mean, wall_floor_f1=wall_floor_f1, **kwargs)
    out["anchors"] = list(per.keys())
    return out


def format_gate_summary(gate: dict[str, Any]) -> str:
    g = gate.get("gates") or {}
    parts = [
        f"score={gate.get('mean_score', 0):.3f}",
        f"off_rel={gate.get('mean_offwall_relaxed_f1', 0):.3f}",
        f"ge2_rec={gate.get('ge2_recall', 0):.2f}",
        f"ge2_strict={gate.get('mean_hop_ge2_strict', 0):.3f}",
        f"f1={gate.get('mean_f1', 0):.3f}",
        f"target={gate.get('target_hit')}",
    ]
    failed = [k for k, v in g.items() if not v]
    if failed:
        parts.append(f"fail={','.join(failed)}")
    return " | ".join(parts)
