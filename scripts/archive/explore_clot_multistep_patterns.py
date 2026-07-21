"""5-snapshot clot vs non-clot pattern analysis + rule sweep (all anchors).

Usage:
  python scripts/explore_clot_multistep_patterns.py
  python scripts/explore_clot_multistep_patterns.py --anchor patient007 --detail
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.clot_multistep_pattern_probe import (  # noqa: E402
    aggregate_rules,
    aggregate_static_features,
    aggregate_time_auc,
    format_multistep_report,
    probe_all_multistep,
    probe_anchor_multistep,
    write_multistep_json,
)
from src.utils.paths import get_project_root


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-timestep clot pattern probe (5 snapshots)")
    p.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    p.add_argument("--anchor", default="", help="Single stem e.g. patient007")
    p.add_argument("--ceiling-hops", type=int, default=None)
    p.add_argument("--detail", action="store_true")
    p.add_argument(
        "--out-json",
        default="outputs/biochem/diagnostics/clot_multistep_pattern_probe.json",
    )
    p.add_argument(
        "--out-report",
        default="outputs/biochem/diagnostics/clot_multistep_pattern_probe.txt",
    )
    args = p.parse_args()

    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")

    root = get_project_root()
    anchor_dir = root / args.anchor_dir

    if args.anchor.strip():
        import torch

        path = anchor_dir / f"{args.anchor.strip()}.pt"
        if not path.is_file():
            print(f"[ERR] missing {path}")
            sys.exit(1)
        data = torch.load(path, map_location="cpu", weights_only=False)
        reports = [probe_anchor_multistep(data, stem=path.stem, ceiling_hops=args.ceiling_hops)]
    else:
        reports = probe_all_multistep(anchor_dir, ceiling_hops=args.ceiling_hops)

    if not reports:
        print(f"[ERR] no patient*.pt under {anchor_dir}")
        sys.exit(1)

    report_text = format_multistep_report(reports)
    print(report_text)

    if args.detail:
        static = aggregate_static_features(reports)
        print()
        print("--- All static/trend/neighbor features (mean AUC desc) ---")
        print(f"{'feature':<28} {'AUC':>6} {'d10rec':>7} {'delta':>10}  group")
        for row in static:
            print(
                f"{row['feature']:<28} {row['mean_auc']:>6.3f} {row['mean_decile_rec']:>6.1%} "
                f"{row['mean_delta']:>10.3g}  {row['group']}"
            )
        print()
        print("--- All rules (pred+ <= 40%) ---")
        rules = aggregate_rules(reports, max_pred_frac=0.40)
        print(f"{'rule':<52} {'F1':>6} {'prec':>6} {'rec':>6} {'pred+':>6}")
        for row in rules:
            print(
                f"{row['rule']:<52} {row['mean_band_f1']:>6.3f} {row['mean_band_prec']:>6.3f} "
                f"{row['mean_band_rec']:>6.3f} {row['mean_band_pred_frac']:>6.3f}"
            )
        print()
        print("--- AUC trajectory (5 snapshots) ---")
        time_auc = aggregate_time_auc(reports)
        for feat in sorted({r["feature"] for r in time_auc}):
            sub = sorted([r for r in time_auc if r["feature"] == feat], key=lambda x: x["time_index"])
            traj = " ".join(f"t{r['time_index']}={r['mean_auc']:.2f}" for r in sub)
            print(f"  {feat:<20} {traj}")

    out_json = root / args.out_json
    out_report = root / args.out_report
    write_multistep_json(reports, out_json)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(report_text + "\n", encoding="utf-8")
    print()
    print(f"[save] {out_json}")
    print(f"[save] {out_report}")


if __name__ == "__main__":
    main()
