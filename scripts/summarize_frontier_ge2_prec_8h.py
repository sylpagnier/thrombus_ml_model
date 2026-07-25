"""Summarize Frontier-ge2 precision 8h: Arm A (canonical) vs Arm S (compound).

Includes hop-stratified off-wall metrics and a wall clot F1 floor gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "deploy_mat_f1",
    "deploy_clot_f1",
    "deploy_clot_score",
    "deploy_clot_offwall_relaxed_f1",
    "deploy_clot_offwall_strict_f1",
    "deploy_clot_offwall_n_pred",
    "deploy_clot_offwall_n_gt",
    "deploy_clot_offwall_n_pred_hop_ge2",
    "deploy_clot_offwall_n_gt_hop_ge2",
    "deploy_clot_offwall_strict_f1_hop_ge2",
)


def _mean(report: dict) -> dict:
    simple = report.get("simple") or {}
    return dict(simple.get("mean") or {})


def _load(path: str, label: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    report = json.loads(p.read_text(encoding="utf-8"))
    m = _mean(report)
    return {
        "label": label,
        "path": str(p),
        "two_model": report.get("two_model"),
        "mean": {k: float(m.get(k, 0.0) or 0.0) for k in METRICS},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-s", required=True)
    ap.add_argument("--wall-clot-floor-delta", type=float, default=0.02)
    ap.add_argument("--hop-ge2-strict-floor", type=float, default=0.017)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    a = _load(args.arm_a, "A_canonical_WC_v7")
    s = _load(args.arm_s, "S_compound_frontier_ge2_prec")
    delta = {
        k: float(s["mean"][k]) - float(a["mean"][k]) for k in METRICS
    }

    wall_ok = float(s["mean"]["deploy_clot_f1"]) >= (
        float(a["mean"]["deploy_clot_f1"]) - float(args.wall_clot_floor_delta)
    )
    hop_vol_up = float(s["mean"]["deploy_clot_offwall_n_pred_hop_ge2"]) > (
        float(a["mean"]["deploy_clot_offwall_n_pred_hop_ge2"]) + 0.5
    )
    hop_strict_up = float(s["mean"]["deploy_clot_offwall_strict_f1_hop_ge2"]) > (
        float(a["mean"]["deploy_clot_offwall_strict_f1_hop_ge2"]) + 0.01
    )
    hop_strict_vs_6h = float(s["mean"]["deploy_clot_offwall_strict_f1_hop_ge2"]) > (
        float(args.hop_ge2_strict_floor) + 1e-6
    )

    if wall_ok and hop_vol_up and hop_strict_up and hop_strict_vs_6h:
        verdict = "pass_precision_signal"
    elif wall_ok and hop_vol_up and hop_strict_up:
        verdict = "weak_beat_6h_miss"
    elif wall_ok and hop_vol_up:
        verdict = "weak_volume_wall_ok"
    elif hop_vol_up and not wall_ok:
        verdict = "lumen_up_wall_regress"
    else:
        verdict = "null_or_regress"

    report = {
        "arm_a": a,
        "arm_s": s,
        "delta_S_minus_A": delta,
        "gates": {
            "wall_clot_floor_delta": float(args.wall_clot_floor_delta),
            "hop_ge2_strict_floor_vs_6h": float(args.hop_ge2_strict_floor),
            "wall_ok": wall_ok,
            "hop_ge2_volume_up": hop_vol_up,
            "hop_ge2_strict_up": hop_strict_up,
            "hop_ge2_strict_vs_6h": hop_strict_vs_6h,
        },
        "verdict": verdict,
    }

    print("=" * 80, flush=True)
    print("WC_v7 FRONTIER_GE2_PREC 8H  A vs S", flush=True)
    print("=" * 80, flush=True)
    print(
        f"{'metric':<40} {'A canon':>10} {'S ge2':>10} {'dS-A':>10}",
        flush=True,
    )
    for k in METRICS:
        print(
            f"{k:<40} {a['mean'][k]:10.4f} {s['mean'][k]:10.4f} {delta[k]:+10.4f}",
            flush=True,
        )
    print(
        f"[i] gates wall_ok={wall_ok} hop_vol_up={hop_vol_up} "
        f"hop_strict_up={hop_strict_up} hop_strict_vs_6h={hop_strict_vs_6h} "
        f"-> verdict={verdict}",
        flush=True,
    )

    if args.out.strip():
        out = Path(args.out)
        if not out.is_absolute():
            out = Path.cwd() / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
