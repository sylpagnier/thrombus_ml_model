"""
Backfill ``geometry_level`` / ``config_id`` on existing kinematics ``vessel_*.pt`` graphs.

Reads ``level`` from ``data/raw/kinematics/meshes/vessel_<id>.json`` (no COMSOL regen).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import VesselConfig
from src.utils.kinematics_geometry import attach_geometry_metadata, vessel_index_from_stem


def backfill_graph_dir(
    graph_dir: Path,
    mesh_input_dir: Path,
    *,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    paths = sorted(graph_dir.glob("vessel_*.pt"))
    updated = 0
    missing_json = 0
    already = 0
    for pt_path in tqdm(paths, desc=str(graph_dir.name)):
        stem = pt_path.stem
        data = torch.load(pt_path, weights_only=False)
        had_level = hasattr(data, "geometry_level") and int(data.geometry_level.view(-1)[0].item()) >= 0
        data.graph_stem = stem
        attach_geometry_metadata(data, mesh_input_dir=mesh_input_dir, stem=stem)
        lvl = int(data.geometry_level.view(-1)[0].item())
        if lvl < 0:
            missing_json += 1
            continue
        if had_level and int(getattr(data, "config_id", vessel_index_from_stem(stem) or -1)) >= 0:
            already += 1
            continue
        if not dry_run:
            torch.save(data, pt_path)
        updated += 1
    return updated, missing_json, already


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill geometry_level on kinematics graphs from mesh JSON.")
    p.add_argument(
        "--rheology",
        choices=("newtonian", "carreau", "both"),
        default="both",
        help="Which graph subfolder(s) to update.",
    )
    p.add_argument("--dry-run", action="store_true", help="Report counts only; do not write .pt files.")
    args = p.parse_args()

    cfg = VesselConfig(phase="kinematics")
    mesh_dir = cfg.mesh_input_dir
    base = cfg.graph_output_dir
    subdirs = ["newtonian", "carreau"] if args.rheology == "both" else [args.rheology]

    print(f"Mesh JSON source: {mesh_dir}")
    for sub in subdirs:
        gdir = base / sub
        if not gdir.is_dir():
            print(f"Skip missing: {gdir}")
            continue
        u, miss, skip = backfill_graph_dir(gdir, mesh_dir, dry_run=args.dry_run)
        print(f"{sub}: updated={u}, missing_json={miss}, unchanged={skip}")


if __name__ == "__main__":
    main()
