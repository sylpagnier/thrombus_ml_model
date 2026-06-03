"""Rebuild 18ch ``data.x`` priors on anchor graphs after Poiseuille cap fixes (no COMSOL re-extract).

Example:
    python scripts/refresh_anchor_kine_x_priors.py
    python scripts/refresh_anchor_kine_x_priors.py --stem patient007
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import VesselConfig
from src.data_gen.lib.node_feature_assembly import (
    kinematics_uv_prior_max,
    refresh_kinematics_node_x_on_graph,
)
from src.utils.channel_schema import attach_patient_anchor_graph_metadata, infer_missing_schema


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stem", type=str, default="", help="Single stem (default: all).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    anchor_dir = Path(VesselConfig(phase="biochem_anchors").graph_output_dir)
    kine_dir = _REPO / "data/processed/graphs_kinematics_anchors/newtonian"
    stems = [args.stem.strip()] if args.stem.strip() else sorted(s.stem for s in anchor_dir.glob("*.pt"))

    from src.config import PhysicsConfig

    phys = PhysicsConfig(phase="kinematics", rheology="newtonian")
    n = 0
    for stem in stems:
        for path in (anchor_dir / f"{stem}.pt", kine_dir / f"{stem}.pt"):
            if not path.is_file():
                continue
            data = torch.load(path, map_location="cpu", weights_only=False)
            data = infer_missing_schema(data, phase_hint="biochem" if "biochem_anchors" in str(path) else "kinematics")
            before = kinematics_uv_prior_max(data.x)
            ok = refresh_kinematics_node_x_on_graph(data, phys_cfg=phys, stem=stem, force=True)
            after = kinematics_uv_prior_max(data.x)
            tag = "refresh" if ok else "skip"
            print(f"[{tag}] {path.name}: uv_prior max {before:.3f} -> {after:.3f}")
            if ok and not args.dry_run:
                if hasattr(data, "x_biochem"):
                    data = attach_patient_anchor_graph_metadata(
                        data, mask_wall=getattr(data, "mask_wall", None)
                    )
                torch.save(data, path)
                n += 1
    print(f"[OK] wrote {n} file(s)")


if __name__ == "__main__":
    main()
