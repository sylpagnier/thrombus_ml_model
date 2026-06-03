"""Backfill ``data.x`` Poiseuille / FD inlet priors on biochem anchor graphs.

Older ``patient*.pt`` files may have ``uv_prior == 0`` (wall-normal flow direction vanishes
in the lumen). This script rebuilds 18ch kine ``x`` using GT ``u,v`` at t=0 for flow
direction and ``x_biochem`` inlet BCs.

Example:
    python scripts/backfill_anchor_kine_priors.py
    python scripts/backfill_anchor_kine_priors.py --stem patient007 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import PhysicsConfig, VesselConfig
from src.data_gen.lib.node_feature_assembly import (
    kinematics_uv_prior_max,
    refresh_kinematics_node_x_on_graph,
)
from src.utils.channel_schema import attach_patient_anchor_graph_metadata, infer_missing_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill kinematics priors on anchor graphs.")
    parser.add_argument("--stem", type=str, default="", help="Only this patient stem (default: all).")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write .pt files.")
    parser.add_argument("--force", action="store_true", help="Rewrite even when priors look nonzero.")
    args = parser.parse_args()

    anchor_dir = Path(VesselConfig(phase="biochem_anchors").graph_output_dir)
    if not anchor_dir.is_dir():
        raise SystemExit(f"[ERR] anchor dir missing: {anchor_dir}")

    paths = sorted(anchor_dir.glob("*.pt"))
    if args.stem:
        paths = [anchor_dir / f"{args.stem.strip()}.pt"]

    phys = PhysicsConfig(phase="kinematics", rheology="newtonian")
    n_ok = 0
    for path in paths:
        if not path.is_file():
            print(f"[WARN] skip missing {path.name}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        data = infer_missing_schema(data, phase_hint="biochem")
        before = kinematics_uv_prior_max(data.x)
        refreshed = refresh_kinematics_node_x_on_graph(
            data,
            phys_cfg=phys,
            stem=path.stem,
            force=bool(args.force),
        )
        after = kinematics_uv_prior_max(data.x)
        tag = "refresh" if refreshed else "skip"
        print(f"[{tag}] {path.name}: uv_prior max {before:.4f} -> {after:.4f}")
        if refreshed and not args.dry_run:
            if hasattr(data, "x_biochem"):
                data = attach_patient_anchor_graph_metadata(data, mask_wall=getattr(data, "mask_wall", None))
            torch.save(data, path)
            n_ok += 1

    if args.dry_run:
        print(f"[i] dry-run complete ({len(paths)} graphs inspected)")
    else:
        print(f"[OK] wrote {n_ok} graph(s) under {anchor_dir}")


if __name__ == "__main__":
    main()
