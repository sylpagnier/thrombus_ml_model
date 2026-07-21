"""Extended feature sweep: graph x, bio_x, topology, t0 vs tfinal AUC gaps.

Usage:
  python scripts/explore_clot_t0_extended.py
  python scripts/explore_clot_t0_extended.py --anchor patient007 --detail
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.clot_t0_extended_probe import (  # noqa: E402
    format_extended_report,
    probe_all_extended,
    probe_anchor_extended,
    write_extended_json,
)
from src.utils.paths import get_project_root


def main() -> None:
    p = argparse.ArgumentParser(description="Extended t=0 clot feature diagnostic")
    p.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    p.add_argument("--anchor", default="")
    p.add_argument("--ceiling-hops", type=int, default=None)
    p.add_argument("--detail", action="store_true")
    p.add_argument(
        "--out-json",
        default="outputs/biochem/diagnostics/clot_t0_extended_probe.json",
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
        reports = [probe_anchor_extended(data, stem=path.stem, ceiling_hops=args.ceiling_hops)]
    else:
        reports = probe_all_extended(anchor_dir, ceiling_hops=args.ceiling_hops)

    if not reports:
        print("[ERR] no anchors")
        sys.exit(1)

    print(format_extended_report(reports))

    if args.detail and reports:
        rep = reports[0]
        print()
        print(f"--- {rep.anchor} all features (AUC_t0 desc) ---")
        rows = sorted(rep.rows, key=lambda r: -r.auc_t0)
        print(f"{'feature':<28} {'AUC_t0':>7} {'AUC_tf':>7} {'dAUC':>7} {'grp':<10} source")
        for row in rows:
            print(
                f"{row.feature:<28} {row.auc_t0:>7.3f} {row.auc_tfinal:>7.3f} {row.delta_auc:>7.3f} "
                f"{row.group:<10} {row.source}"
            )

    out_path = root / args.out_json
    write_extended_json(reports, out_path)
    print()
    print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
