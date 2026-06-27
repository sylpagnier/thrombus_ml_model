"""Summarize the full-budget Mat-only legs (G / N / O) vs the locked fi_mat baseline.

Reads each leg's ``<LADDER_ROOT>/<leg>/compare.json`` (written by go_mat_growth_simple.ps1's eval,
whose baseline defaults to the locked full-budget fi_mat ckpt). Prints a table ranked by deploy
clot F1 with the delta vs the locked baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import LADDER_ROOT, mat_growth_leg_spec  # noqa: E402

METRICS = (
    "deploy_mat_f1",
    "deploy_clot_f1",
    "deploy_clot_score",
    "mat_seed_prec",
    "mat_seed_count",
    "mat_front_prec",
    "mat_front_speed_ratio",
    "mat_overpaint_frac",
    "mat_overpaint_per_gt",
)
DEFAULT_LEGS = (
    "P_mat_plain",
    "G_dual_mat_neighbor_gate",
    "N_mat_geom_rich",
    "O_mat_neighbor_geom_rich",
    "U_mat_frontier_only",
    "V_mat_frontier_geom",
    "S_mat_frontier_nuc",
    "T_mat_frontier_sharp",
    "Q_mat_gate_sharp_fp",
    "R_mat_geom_gate_sharp_fp",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize full-budget Mat-only legs vs locked baseline")
    ap.add_argument("--legs", default=",".join(DEFAULT_LEGS), help="comma list of leg codes")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/mat_only_full/mat_only_full_summary.json")
    args = ap.parse_args()

    legs = [s.strip() for s in args.legs.split(",") if s.strip()]
    rows: list[dict] = []
    baseline_mean: dict[str, float] | None = None
    for leg in legs:
        p = Path(LADDER_ROOT) / leg / "compare.json"
        if not p.is_absolute():
            p = (REPO / p).resolve()
        if not p.is_file():
            print(f"[skip] missing {p}", flush=True)
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        simple = payload.get("simple") or {}
        mean = simple.get("mean") or {}
        delta = payload.get("delta_simple_minus_baseline") or {}
        if baseline_mean is None:
            baseline_mean = dict((payload.get("baseline") or {}).get("mean") or {})
        rows.append(
            {
                "leg": leg,
                "label": mat_growth_leg_spec(leg).label,
                "cohort_mean": {k: float(mean.get(k, 0.0)) for k in METRICS},
                "delta_vs_locked": {k: float(delta.get(k, 0.0)) for k in METRICS},
            }
        )

    if not rows:
        print(f"[ERR] no compare.json found for legs={legs} under {LADDER_ROOT}", flush=True)
        return 1

    rows.sort(key=lambda r: r["cohort_mean"]["deploy_clot_f1"], reverse=True)

    print("\n==================== MAT-ONLY FULL BUDGET (deploy, pred kine) ====================", flush=True)
    if baseline_mean:
        print(
            f"  [locked fi_mat] mat={baseline_mean.get('deploy_mat_f1', 0):.3f} "
            f"clot_f1={baseline_mean.get('deploy_clot_f1', 0):.3f} "
            f"clot_score={baseline_mean.get('deploy_clot_score', 0):.3f}",
            flush=True,
        )
    hdr = (
        f"  {'leg':<28}{'mat':>7}{'clot_f1':>9}{'d_clot':>8}{'score':>8}"
        f"{'seedP':>8}{'frontP':>8}{'speed':>8}{'over/gt':>9}"
    )
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for row in rows:
        cm, dv = row["cohort_mean"], row["delta_vs_locked"]
        print(
            f"  {row['leg']:<28}{cm['deploy_mat_f1']:>7.3f}{cm['deploy_clot_f1']:>9.3f}"
            f"{dv['deploy_clot_f1']:>+8.3f}{cm['deploy_clot_score']:>8.3f}"
            f"{cm.get('mat_seed_prec', 0.0):>8.3f}{cm.get('mat_front_prec', 0.0):>8.3f}"
            f"{cm.get('mat_front_speed_ratio', 0.0):>8.2f}{cm.get('mat_overpaint_per_gt', 0.0):>9.2f}",
            flush=True,
        )

    summary = {"baseline_locked_mean": baseline_mean, "legs": rows}
    out = Path(args.out)
    if not out.is_absolute():
        out = (REPO / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
