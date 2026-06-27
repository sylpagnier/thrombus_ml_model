"""Summarize the precision-sweep legs against the fast dual fi_mat baseline.

Reads each ``compare_<leg>_vs_baseline_fast.json`` (written by eval_mat_growth_simple.py with
``--baseline-ckpt``) and prints a table ranked by deploy clot F1, with the delta vs baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import mat_growth_leg_spec  # noqa: E402

METRICS = ("deploy_mat_f1", "deploy_clot_f1", "deploy_clot_score")
SWEEP_LEGS = (
    "K_fimat_neighbor_gate",
    "L_fimat_geom_rich",
    "M_fimat_neighbor_geom_rich",
    "N_mat_geom_rich",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize precision sweep")
    ap.add_argument("--run-root", default="outputs/biochem/biochem_gnn/precision_sweep")
    ap.add_argument("--val-anchor", default="patient007")
    args = ap.parse_args()

    root = Path(args.run_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    va = args.val_anchor.strip()

    rows: list[dict] = []
    baseline_mean: dict[str, float] | None = None
    for leg in SWEEP_LEGS:
        p = root / f"compare_{leg}_vs_baseline_fast.json"
        if not p.is_file():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        simple = payload.get("simple") or {}
        mean = simple.get("mean") or {}
        per = simple.get("per_anchor") or {}
        p007 = per.get(va) or {}
        delta = payload.get("delta_simple_minus_baseline") or {}
        if baseline_mean is None:
            baseline_mean = dict((payload.get("baseline") or {}).get("mean") or {})
        rows.append(
            {
                "leg": leg,
                "label": mat_growth_leg_spec(leg).label,
                "cohort_mean": {k: float(mean.get(k, 0.0)) for k in METRICS},
                "p007": {k: float(p007.get(k, 0.0)) for k in METRICS},
                "delta_vs_baseline": {k: float(delta.get(k, 0.0)) for k in METRICS},
            }
        )

    if not rows:
        print(f"[ERR] no compare_*_vs_baseline_fast.json under {root}", flush=True)
        return 1

    rows.sort(key=lambda r: r["cohort_mean"]["deploy_clot_f1"], reverse=True)

    print("\n==================== PRECISION SWEEP (deploy, pred kine) ====================", flush=True)
    if baseline_mean:
        print(
            f"  [baseline_fast] mat={baseline_mean.get('deploy_mat_f1', 0):.3f} "
            f"clot_f1={baseline_mean.get('deploy_clot_f1', 0):.3f} "
            f"clot_score={baseline_mean.get('deploy_clot_score', 0):.3f}",
            flush=True,
        )
    hdr = f"  {'leg':<28}{'mat':>7}{'clot_f1':>9}{'d_clot':>8}{'score':>8}{'d_score':>9}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    for row in rows:
        cm, dv = row["cohort_mean"], row["delta_vs_baseline"]
        print(
            f"  {row['leg']:<28}{cm['deploy_mat_f1']:>7.3f}{cm['deploy_clot_f1']:>9.3f}"
            f"{dv['deploy_clot_f1']:>+8.3f}{cm['deploy_clot_score']:>8.3f}{dv['deploy_clot_score']:>+9.3f}",
            flush=True,
        )

    summary = {"val_anchor": va, "baseline_fast_mean": baseline_mean, "legs": rows}
    out_json = root / "precision_sweep_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[save] {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
