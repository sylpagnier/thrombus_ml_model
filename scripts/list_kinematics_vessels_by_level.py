"""List ``vessel_*`` stems grouped by mesh geometry level (0/1/2).

Example:
    python scripts/list_kinematics_vessels_by_level.py
    python scripts/list_kinematics_vessels_by_level.py --rheology carreau --level 2 --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils.kinematics_paths import kinematics_graph_rheology_dir


def _level_for_stem(stem: str, mesh_dir: Path, graph_path: Path) -> int:
    import torch

    if graph_path.is_file():
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
        if hasattr(data, "geometry_level"):
            return int(data.geometry_level.view(-1)[0].item())
    mj = mesh_dir / f"{stem}.json"
    if mj.is_file():
        return int(json.loads(mj.read_text(encoding="utf-8")).get("level", -1))
    return -1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rheology", default="newtonian", choices=("newtonian", "carreau"))
    p.add_argument("--level", type=int, default=None, help="Only print this level (0/1/2).")
    p.add_argument("--limit", type=int, default=20, help="Max stems printed per level.")
    p.add_argument(
        "--require-labels",
        action="store_true",
        help="Only list stems whose graph y has nonzero |u| (skip empty CFD label graphs).",
    )
    args = p.parse_args()

    kdir = kinematics_graph_rheology_dir(args.rheology)
    mesh_dir = _REPO / "data/raw/kinematics/meshes"
    if not kdir.is_dir():
        raise SystemExit(f"[ERR] missing {kdir}")

    by_level: dict[int, list[str]] = defaultdict(list)
    for path in sorted(kdir.glob("vessel_*.pt")):
        lvl = _level_for_stem(path.stem, mesh_dir, path)
        if args.require_labels:
            import torch

            data = torch.load(path, map_location="cpu", weights_only=False)
            mag = float((data.y[:, 0] ** 2 + data.y[:, 1] ** 2).max().item())
            if mag < 1e-6:
                continue
        by_level[lvl].append(path.stem)

    print(f"[i] graph dir: {kdir}")
    print(f"[i] mesh json: {mesh_dir}")
    for lvl in sorted(by_level.keys()):
        if args.level is not None and lvl != int(args.level):
            continue
        stems = by_level[lvl]
        print(f"\nlevel {lvl}: {len(stems)} vessel(s)")
        for s in stems[: max(0, int(args.limit))]:
            print(f"  {s}")
        if args.limit > 0 and len(stems) > args.limit:
            print(f"  ... +{len(stems) - args.limit} more")


if __name__ == "__main__":
    main()
