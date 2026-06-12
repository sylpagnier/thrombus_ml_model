"""Three patient007 timeline PNGs for visual metric calibration.

Slot A: anti-pattern (full timeline mean -> ceiling paint)
Slot B: prog baseline (acceptable)
Slot C: deploy_score winner (visual best)

Usage:
  python scripts/viz_clot_metric_triplet.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.clot_temporal_growth_rules import (  # noqa: E402
    aggregate_architecture_sweep,
    compute_deploy_score,
)
from scripts.promote_clot_architecture_winner import _parse_rule_name  # noqa: E402
from scripts.viz_clot_sweep_dual_winners import _apply_base_env, _load_prior_winner_env  # noqa: E402


def _p007_deploy_rows(per_anchor: list[dict], anchor: str) -> list[dict]:
    out: list[dict] = []
    for row in per_anchor:
        if row.get("anchor") != anchor or row.get("n_pairs", 0) < 1:
            continue
        ds = compute_deploy_score(
            p007_tfinal_shape=float(row.get("tfinal_clot_shape", float("nan"))),
            p007_early_pred=float(row.get("early_mean_pred_frac", float("nan"))),
            p007_tfinal_bal=float(row.get("tfinal_clot_shape_bal", float("nan"))),
            p007_pred=float(row.get("tfinal_band_pred_frac", float("nan"))),
        )
        out.append(
            {
                "rule": row["rule"],
                "deploy_score": ds,
                "p007_tfinal_clot_shape": row.get("tfinal_clot_shape"),
                "p007_early_pred_frac": row.get("early_mean_pred_frac"),
                "p007_timeline_clot_shape": row.get("mean_clot_shape"),
            }
        )
    return out


def _pick_triplet(agg: list[dict], *, per_anchor: list[dict], anchor: str) -> list[dict]:
    by_rule = {r["rule"]: r for r in agg}
    p007 = _p007_deploy_rows(per_anchor, anchor)

    def _max_tl() -> dict:
        if not p007:
            return max(agg, key=lambda x: float(x.get("p007_clot_shape", 0.0)))
        return max(p007, key=lambda x: float(x.get("p007_timeline_clot_shape", 0.0)))

    baseline = by_rule.get("loc_prog_both_t20_s0_ndx25")
    if baseline is None:
        baseline = agg[0]

    deploy_row = max(p007, key=lambda x: x["deploy_score"]) if p007 else baseline
    if "loc_prog_both_t20_s0_ndx25_inc40" in {r["rule"] for r in p007}:
        inc = next(r for r in p007 if r["rule"] == "loc_prog_both_t20_s0_ndx25_inc40")
        top = float(deploy_row["deploy_score"])
        if float(inc["deploy_score"]) >= top - 1e-6:
            deploy_row = inc

    anti_rule = _max_tl()["rule"]
    anti = by_rule.get(anti_rule, _max_tl())

    return [
        {
            "slug": "metric_antipattern_timeline",
            "label": "AVOID: full timeline_mean clot_shape",
            "row": anti,
            "detail": (
                f"p007 timeline={anti.get('p007_timeline_clot_shape', anti.get('p007_clot_shape', float('nan'))):.3f} "
                f"rewards ceiling-wide paint | tfinal={anti.get('p007_tfinal_clot_shape', float('nan')):.3f}"
            ),
        },
        {
            "slug": "metric_baseline_prog",
            "label": "baseline: loc_prog (acceptable)",
            "row": baseline,
            "detail": (
                f"deploy={baseline.get('deploy_score', float('nan')):.3f} "
                f"tfinal={baseline.get('p007_tfinal_clot_shape', float('nan')):.3f} "
                f"early+={baseline.get('p007_early_pred_frac', float('nan')):.3f}"
            ),
        },
        {
            "slug": "metric_deploy_best",
            "label": "deploy_score winner (visual best)",
            "row": deploy_row or baseline,
            "detail": (
                f"deploy={deploy_row.get('deploy_score', float('nan')):.3f} "
                f"tfinal={deploy_row.get('p007_tfinal_clot_shape', float('nan')):.3f} "
                f"early+={deploy_row.get('p007_early_pred_frac', float('nan')):.3f}"
            ),
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Metric triplet timeline viz")
    ap.add_argument(
        "--json",
        default="outputs/biochem/diagnostics/clot_rule_ideas_sweep.json",
    )
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument("--out-dir", default="outputs/biochem/viz/clot_deploy")
    args = ap.parse_args()

    json_path = REPO / args.json
    if not json_path.is_file():
        print(f"[ERR] missing {json_path}", file=sys.stderr)
        return 1

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    per_anchor = payload.get("per_anchor") or []
    agg = payload.get("aggregated") or aggregate_architecture_sweep(per_anchor)
    if not agg:
        print("[ERR] no aggregated rows", file=sys.stderr)
        return 1

    _apply_base_env()
    _load_prior_winner_env()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for item in _pick_triplet(agg, per_anchor=per_anchor, anchor=args.anchor):
        row = item["row"]
        rule = str(row["rule"])
        env = _parse_rule_name(rule)
        for key, val in env.items():
            os.environ[key] = val

        out_png = out_dir / f"temporal_rule_{args.anchor}_{item['slug']}.png"
        title = f"{args.anchor} | {item['label']}\n{rule} | {item['detail']}"
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "viz_clot_temporal_rule_timeline.py"),
            "--anchor",
            args.anchor,
            "--anchor-dir",
            args.anchor_dir,
            "--keyframes",
            str(args.keyframes),
            "--out",
            str(out_png),
            "--title",
            title,
        ]
        print(f"[i] {item['slug']}: {rule}", flush=True)
        subprocess.run(cmd, cwd=str(REPO), check=True)
        manifest.append({"slug": item["slug"], "rule": rule, "png": str(out_png.relative_to(REPO))})

    manifest_path = out_dir / f"temporal_rule_{args.anchor}_metric_triplet.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
