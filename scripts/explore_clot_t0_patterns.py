"""Explore t=0 kinematic patterns vs GT clot @ t_final inside deploy ceiling mask.

Usage:
  python scripts/explore_clot_t0_patterns.py
  python scripts/explore_clot_t0_patterns.py --anchor patient007 --detail
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.clot_t0_pattern_probe import (  # noqa: E402
    format_pattern_report,
    probe_all_anchors,
    probe_anchor_patterns,
    write_probe_json,
)
from src.utils.paths import get_project_root


def main() -> None:
    p = argparse.ArgumentParser(description="t=0 vs t_final clot pattern probe")
    p.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    p.add_argument("--anchor", default="", help="Single patient stem e.g. patient007")
    p.add_argument("--ceiling-hops", type=int, default=None)
    p.add_argument("--detail", action="store_true", help="Per-anchor feature table")
    p.add_argument(
        "--out-json",
        default="outputs/biochem/diagnostics/clot_t0_pattern_probe.json",
    )
    args = p.parse_args()

    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    if args.ceiling_hops is not None:
        os.environ["CLOT_PHI_CEILING_HOPS"] = str(args.ceiling_hops)
    else:
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
        reports = [probe_anchor_patterns(data, stem=path.stem, ceiling_hops=args.ceiling_hops)]
    else:
        reports = probe_all_anchors(anchor_dir, ceiling_hops=args.ceiling_hops)

    if not reports:
        print(f"[ERR] no patient*.pt under {anchor_dir}")
        sys.exit(1)

    print(format_pattern_report(reports))

    if args.detail:
        for rep in reports:
            if rep.n_clot_ceiling < 1:
                continue
            print()
            print(f"--- {rep.anchor} feature detail (sorted by AUC) ---")
            rows = sorted(rep.feature_rows, key=lambda r: -r.auc)
            print(f"{'feature':<18} {'AUC':>6} {'d10_rec':>8} {'clot_m':>10} {'non_m':>10} {'delta':>10}")
            for row in rows:
                print(
                    f"{row.feature:<18} {row.auc:>6.3f} {row.decile_rec:>7.1%} "
                    f"{row.clot_mean:>10.3g} {row.non_mean:>10.3g} {row.delta_mean:>10.3g}"
                )
            print(f"{'rule':<36} {'F1':>6} {'prec':>6} {'rec':>6} {'n':>6}")
            for rr in sorted(rep.rule_rows, key=lambda r: -r.f1):
                print(f"{rr.rule:<36} {rr.f1:>6.3f} {rr.prec:>6.3f} {rr.rec:>6.3f} {rr.n_flag:>6}")

    out_path = root / args.out_json
    write_probe_json(reports, out_path)
    print()
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
