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
    "clot_fp_median",
    "clot_fp_p90",
    "clot_fp_max",
    "clot_fn_median",
    "clot_err_median",
    "clot_err_p90",
    "clot_fp_early_mean",
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


def _pick_winner(
    rows: list[dict],
    *,
    rank_by: list[str],
    minimize_by: set[str] | None = None,
    max_overpaint_per_gt: float | None,
    max_clot_fp_p90: float | None,
    max_clot_fp_early_mean: float | None,
    prefer_leg: str | None = None,
    tie_eps: float = 0.0,
) -> dict | None:
    eligible = rows
    if max_overpaint_per_gt is not None:
        eligible = [
            r
            for r in rows
            if r["cohort_mean"].get("mat_overpaint_per_gt", 1.0) <= max_overpaint_per_gt
        ]
    if max_clot_fp_p90 is not None:
        eligible = [r for r in eligible if r["cohort_mean"].get("clot_fp_p90", 1e9) <= max_clot_fp_p90]
    if max_clot_fp_early_mean is not None:
        eligible = [
            r
            for r in eligible
            if r["cohort_mean"].get("clot_fp_early_mean", 1e9) <= max_clot_fp_early_mean
        ]
    if not eligible:
        return None

    min_set = minimize_by or set()

    def sort_key(row: dict) -> tuple[float, ...]:
        cm = row["cohort_mean"]
        out: list[float] = []
        for k in rank_by:
            v = float(cm.get(k, 0.0))
            out.append(-v if k in min_set else v)
        return tuple(out)

    best = max(eligible, key=sort_key)
    pref = (prefer_leg or "").strip()
    if pref and pref != best["leg"]:
        cand = next((r for r in eligible if r["leg"] == pref), None)
        if cand is not None and tie_eps > 0.0 and rank_by:
            k0 = rank_by[0]
            b0 = float(best["cohort_mean"].get(k0, 0.0))
            c0 = float(cand["cohort_mean"].get(k0, 0.0))
            if k0 in min_set:
                if c0 <= b0 + tie_eps:
                    return cand
            elif c0 >= b0 - tie_eps:
                return cand
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize full-budget Mat-only legs vs locked baseline")
    ap.add_argument("--legs", default=",".join(DEFAULT_LEGS), help="comma list of leg codes")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/mat_only_full/mat_only_full_summary.json")
    ap.add_argument(
        "--pick-winner",
        action="store_true",
        help="print and embed optimal leg (requires --rank-by; optional --max-overpaint-per-gt filter)",
    )
    ap.add_argument(
        "--max-overpaint-per-gt",
        type=float,
        default=None,
        help="exclude legs with mat_overpaint_per_gt above this threshold",
    )
    ap.add_argument(
        "--rank-by",
        default="deploy_clot_f1",
        help="comma list of metrics; first metric breaks ties (descending)",
    )
    ap.add_argument(
        "--max-clot-fp-p90",
        type=float,
        default=None,
        help="exclude legs with clot_fp_p90 above this threshold",
    )
    ap.add_argument(
        "--max-clot-fp-early-mean",
        type=float,
        default=None,
        help="exclude legs with clot_fp_early_mean above this threshold",
    )
    ap.add_argument(
        "--minimize-metrics",
        default="",
        help="comma metrics where lower is better (negated in rank sort)",
    )
    ap.add_argument(
        "--prefer-leg",
        default="",
        help="when within --tie-eps on the first rank metric, prefer this leg",
    )
    ap.add_argument(
        "--tie-eps",
        type=float,
        default=0.0,
        help="tolerance on first rank metric for --prefer-leg override",
    )
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
        f"{'medFP':>7}{'p90FP':>7}{'medFN':>7}{'over/gt':>9}"
    )
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for row in rows:
        cm, dv = row["cohort_mean"], row["delta_vs_locked"]
        print(
            f"  {row['leg']:<28}{cm['deploy_mat_f1']:>7.3f}{cm['deploy_clot_f1']:>9.3f}"
            f"{dv['deploy_clot_f1']:>+8.3f}{cm['deploy_clot_score']:>8.3f}"
            f"{cm.get('clot_fp_median', 0.0):>7.0f}{cm.get('clot_fp_p90', 0.0):>7.0f}"
            f"{cm.get('clot_fn_median', 0.0):>7.0f}{cm.get('mat_overpaint_per_gt', 0.0):>9.2f}",
            flush=True,
        )

    rank_by = [s.strip() for s in args.rank_by.split(",") if s.strip()]
    minimize_by = {s.strip() for s in args.minimize_metrics.split(",") if s.strip()}
    winner = None
    if args.pick_winner:
        winner = _pick_winner(
            rows,
            rank_by=rank_by,
            minimize_by=minimize_by,
            max_overpaint_per_gt=args.max_overpaint_per_gt,
            max_clot_fp_p90=args.max_clot_fp_p90,
            max_clot_fp_early_mean=args.max_clot_fp_early_mean,
            prefer_leg=args.prefer_leg.strip() or None,
            tie_eps=float(args.tie_eps),
        )
        if winner is None:
            print(
                "\n[WARN] no winner: no leg passed overpaint filter "
                f"(max_overpaint_per_gt={args.max_overpaint_per_gt}, "
                f"max_clot_fp_p90={args.max_clot_fp_p90}, "
                f"max_clot_fp_early_mean={args.max_clot_fp_early_mean})",
                flush=True,
            )
        else:
            cm = winner["cohort_mean"]
            print(
                f"\n[WINNER] {winner['leg']} "
                f"clot_score={cm['deploy_clot_score']:.3f} "
                f"clot_f1={cm['deploy_clot_f1']:.3f} "
                f"over/gt={cm.get('mat_overpaint_per_gt', 0.0):.2f} "
                f"(rank: {', '.join(rank_by)})",
                flush=True,
            )

    summary: dict = {"baseline_locked_mean": baseline_mean, "legs": rows}
    if args.pick_winner:
        summary["pick_config"] = {
            "rank_by": rank_by,
            "minimize_metrics": sorted(minimize_by),
            "max_overpaint_per_gt": args.max_overpaint_per_gt,
            "max_clot_fp_p90": args.max_clot_fp_p90,
            "max_clot_fp_early_mean": args.max_clot_fp_early_mean,
            "prefer_leg": args.prefer_leg.strip() or None,
            "tie_eps": float(args.tie_eps),
        }
        summary["winner"] = winner
    out = Path(args.out)
    if not out.is_absolute():
        out = (REPO / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
