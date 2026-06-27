"""Summarize mat-growth ladder legs vs triangle6 fi_mat baseline.

Reads each leg's ``<leg>/compare.json`` (from eval_mat_growth_simple.py).
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path

# Ensure repo root is on sys.path so `import src.*` works when launched from wrappers.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import LADDER_LEG_ORDER, mat_growth_leg_spec

METRICS = ("deploy_mat_f1", "deploy_clot_f1", "deploy_clot_score")


def _load_leg_compare(leg_dir: Path) -> dict | None:
    p = leg_dir / "compare.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize mat-growth ladder")
    ap.add_argument("--run-root", default="outputs/biochem/biochem_gnn/mat_growth_ladder")
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    root = Path(args.run_root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    va = args.val_anchor.strip()

    rows: list[dict] = []
    baseline_mean: dict[str, float] | None = None
    for leg in LADDER_LEG_ORDER:
        payload = _load_leg_compare(root / leg)
        if payload is None:
            continue
        simple = payload.get("simple") or {}
        per = simple.get("per_anchor") or {}
        mean = simple.get("mean") or {}
        p007 = per.get(va) or {}
        rest = [v for k, v in per.items() if k != va]
        rest_mean = {
            k: sum(float(r.get(k, 0.0)) for r in rest) / max(len(rest), 1) for k in METRICS
        }
        if baseline_mean is None:
            baseline_mean = dict((payload.get("baseline") or {}).get("mean") or {})
        spec = mat_growth_leg_spec(leg)
        rows.append(
            {
                "leg": leg,
                "label": spec.label,
                "init_mode": spec.init_mode,
                "no_init": spec.no_init,
                "env_overrides": spec.env_overrides,
                "p007": {k: float(p007.get(k, 0.0)) for k in METRICS},
                "rest_mean": rest_mean,
                "cohort_mean": {k: float(mean.get(k, 0.0)) for k in METRICS},
                "meta": simple.get("meta") or {},
            }
        )

    if not rows:
        print(f"[ERR] no leg compare.json under {root}", flush=True)
        return 1

    print("\n==================== MAT GROWTH LADDER (10-anchor deploy, pred kine) ====================", flush=True)
    hdr = f"  {'leg':<14}{'mat rest':>9}{'mat p007':>9}{'clot rest':>10}{'clot p007':>10}{'score rest':>11}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    base_rest = base_p007 = None
    for row in rows:
        leg = row["leg"]
        rm = row["rest_mean"]
        p7 = row["p007"]
        print(
            f"  {leg:<14}{rm['deploy_mat_f1']:>9.3f}{p7['deploy_mat_f1']:>9.3f}"
            f"{rm['deploy_clot_f1']:>10.3f}{p7['deploy_clot_f1']:>10.3f}"
            f"{rm['deploy_clot_score']:>11.3f}",
            flush=True,
        )
        if leg == "A_random":
            base_rest = rm
            base_p007 = p7

    if baseline_mean:
        print(f"\n  [ref] triangle6 species/best.pth cohort mean: mat={baseline_mean.get('deploy_mat_f1', 0):.3f} "
              f"clot={baseline_mean.get('deploy_clot_f1', 0):.3f}", flush=True)

    if base_rest is not None:
        print("\n  delta vs A_random (rest cohort):", flush=True)
        for row in rows:
            if row["leg"] == "A_random":
                continue
            d_mat = row["rest_mean"]["deploy_mat_f1"] - base_rest["deploy_mat_f1"]
            d_clot = row["rest_mean"]["deploy_clot_f1"] - base_rest["deploy_clot_f1"]
            print(f"    {row['leg']:<14} d_mat={d_mat:+.3f}  d_clot={d_clot:+.3f}", flush=True)

    summary = {
        "val_anchor": va,
        "baseline_cohort_mean": baseline_mean,
        "legs": rows,
    }
    out_json = Path(args.out_json) if args.out_json.strip() else root / "ladder_summary.json"
    if not out_json.is_absolute():
        out_json = (Path.cwd() / out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[save] {out_json}", flush=True)

    out_md = Path(args.out_md) if args.out_md.strip() else root / "ladder_summary.md"
    if not out_md.is_absolute():
        out_md = (Path.cwd() / out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mat growth ladder summary",
        "",
        f"val anchor: `{va}` | metrics: deploy Mat + analytical gelation clot (pred kine)",
        "",
        "| leg | mat rest | mat p007 | clot rest | clot p007 | score rest |",
        "|-----|---------:|---------:|----------:|----------:|-----------:|",
    ]
    for row in rows:
        rm, p7 = row["rest_mean"], row["p007"]
        lines.append(
            f"| {row['leg']} | {rm['deploy_mat_f1']:.3f} | {p7['deploy_mat_f1']:.3f} | "
            f"{rm['deploy_clot_f1']:.3f} | {p7['deploy_clot_f1']:.3f} | {rm['deploy_clot_score']:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
