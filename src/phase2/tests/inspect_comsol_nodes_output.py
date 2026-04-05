import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import VesselConfig
from src.utils.paths import get_project_root


def get_boundary_mask(boundary_file, tree, num_nodes, tolerance=1e-6):
    mask = np.zeros(num_nodes, dtype=bool)
    if not boundary_file.exists():
        print(f"Warning: Boundary file missing: {boundary_file}")
        return mask

    bnd_df = pd.read_csv(boundary_file, comment="%", sep=r"\s+", header=None)
    bnd_coords = np.unique(bnd_df.iloc[:, -2:].values, axis=0)
    distances, indices = tree.query(bnd_coords)
    valid_matches = indices[distances < tolerance]
    mask[valid_matches] = True
    return mask


def _available_stems(data_dir):
    stems = []
    for p in sorted(Path(data_dir).glob("*.txt")):
        if p.stem.endswith("_inlet") or p.stem.endswith("_outlet") or p.stem.endswith("_wall"):
            continue
        stems.append(p.stem)
    return stems


def _pick_stem_interactively(data_dir):
    stems = _available_stems(data_dir)
    if len(stems) == 0:
        print(f"No domain .txt files found in {data_dir}")
        return None

    print("\nAvailable patient stems:")
    for idx, stem in enumerate(stems):
        print(f"  [ {idx} ] {stem}")

    while True:
        user_input = input(f"\nSelect index [0-{len(stems) - 1}] or q to quit: ").strip()
        if user_input.lower() in [ "q", "quit", "exit" ]:
            return None
        try:
            idx = int(user_input)
            if 0 <= idx < len(stems):
                return stems[idx]
            print(f"Invalid selection. Enter a value in [ 0, {len(stems) - 1} ].")
        except ValueError:
            print("Invalid input. Enter an integer index.")


def inspect_patient(stem, data_dir):
    domain_file = Path(data_dir) / f"{stem}.txt"
    inlet_file = Path(data_dir) / f"{stem}_inlet.txt"
    outlet_file = Path(data_dir) / f"{stem}_outlet.txt"
    wall_file = Path(data_dir) / f"{stem}_wall.txt"

    if not domain_file.exists():
        print(f"CRITICAL ERROR: Main file not found at {domain_file}")
        return

    col_names = [
        "x_orig", "y_orig", "x", "y", "u", "v", "p", "mu_eff",
        "rp", "ap", "apr", "aps", "PT", "th", "at", "fg", "fi",
        "M", "Mas", "Mat",
    ]
    df = pd.read_csv(domain_file, comment="%", sep=r"\s+", header=None, names=col_names)
    domain_coords = df[[ "x", "y" ]].values
    tree = cKDTree(domain_coords)

    mask_inlet = get_boundary_mask(inlet_file, tree, len(df))
    mask_outlet = get_boundary_mask(outlet_file, tree, len(df))
    mask_wall = get_boundary_mask(wall_file, tree, len(df))
    mask_fluid = ~(mask_inlet | mask_outlet | mask_wall)

    print("\n" + "=" * 45)
    print(f"   GROUND-TRUTH SELECTION: {stem.upper()}")
    print("=" * 45)
    print(f"Total Unique Nodes: {len(df)}")
    print("-" * 45)
    print(f"Inlet Nodes:       {mask_inlet.sum()}")
    print(f"Outlet Nodes:      {mask_outlet.sum()}")
    print(f"Wall Nodes:        {mask_wall.sum()}")
    print(f"Interior Fluid:    {mask_fluid.sum()}")
    print("=" * 45)

    if mask_inlet.sum() == 0:
        print("ERROR: Still 0 inlet nodes. Check your COMSOL edge exports.")
    else:
        print("SUCCESS: Boundary nodes extracted from spatial mapping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect COMSOL node boundary mapping for Tier 3 patients")
    parser.add_argument("--stem", type=str, default=None, help="Patient stem (for example: patient001)")
    parser.add_argument("--tier", type=str, default="tier3_patients", help="Tier used to resolve default data directory")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional directory containing COMSOL .txt exports")
    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
    else:
        cfg = VesselConfig(tier=args.tier)
        data_dir = get_project_root() / cfg.output_dir

    stem = args.stem if args.stem is not None else _pick_stem_interactively(data_dir)
    if stem is None:
        print("Exiting without action.")
    else:
        inspect_patient(stem, data_dir)