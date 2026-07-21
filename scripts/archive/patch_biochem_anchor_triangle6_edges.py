"""One-off: replace corner-only ``edge_index`` in biochem anchor graphs with full P2 triangle6.

Reads ``patient*.pt`` under ``graphs_biochem_anchors`` and matching ``.nas`` / ``.msh``
from ``raw/biochem_anchors`` (via ``resolve_anchor_mesh_path``).

Run:
  python scripts/patch_biochem_anchor_triangle6_edges.py
  python scripts/patch_biochem_anchor_triangle6_edges.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from src.config import VesselConfig
from src.data_gen.lib.centerline_utils import resolve_anchor_mesh_path
from src.data_gen.lib.mesh_triangle6_edges import edge_index_from_mesh_path_checked


def _backup(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"{path.stem}_corner_edges_{stamp}.pt"
    shutil.copy2(path, dest)
    return dest


def patch_anchor(
    graph_path: Path,
    raw_dir: Path,
    *,
    dry_run: bool,
    backup_dir: Path,
) -> str:
    stem = graph_path.stem
    mesh_path = resolve_anchor_mesh_path(raw_dir, stem)
    if mesh_path is None:
        return f"[skip] {stem}: no mesh (.nas/.msh)"

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    n = int(data.num_nodes)
    old_e = int(data.edge_index.shape[1]) // 2

    try:
        new_ei = edge_index_from_mesh_path_checked(mesh_path, num_nodes=n, stem=stem)
    except Exception as exc:
        return f"[skip] {stem}: {exc}"

    new_e = int(new_ei.shape[1]) // 2
    if dry_run:
        return f"[dry] {stem}: edges {old_e} -> {new_e}  mesh={mesh_path.name}"

    row, col = new_ei
    pos_nd = data.x[:, :2].to(dtype=torch.float32)
    edge_vec = pos_nd[row] - pos_nd[col]
    edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
    edge_attr = torch.cat([edge_vec, edge_len], dim=1)

    bak = _backup(graph_path, backup_dir)
    data.edge_index = new_ei
    data.edge_attr = edge_attr
    torch.save(data, graph_path)
    return (
        f"[OK] {stem}: edges {old_e} -> {new_e}  "
        f"edge_attr={tuple(edge_attr.shape)}  backup={bak.name}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch biochem anchor graphs to triangle6 edge_index")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--graph-dir", default="", help="Override graphs_biochem_anchors dir")
    ap.add_argument("--raw-dir", default="", help="Override raw/biochem_anchors dir")
    args = ap.parse_args()

    cfg = VesselConfig(phase="biochem_anchors")
    graph_dir = Path(args.graph_dir) if args.graph_dir.strip() else REPO / cfg.graph_output_dir
    raw_dir = Path(args.raw_dir) if args.raw_dir.strip() else REPO / cfg.mesh_input_dir
    backup_dir = graph_dir / "_backup_corner_edges"

    paths = sorted(graph_dir.glob("patient*.pt"))
    if not paths:
        print(f"[ERR] no patient*.pt under {graph_dir}", flush=True)
        return 1

    print(f"[i] graph_dir={graph_dir}", flush=True)
    print(f"[i] raw_dir={raw_dir}  dry_run={bool(args.dry_run)}", flush=True)
    ok = skip = 0
    for p in paths:
        line = patch_anchor(p, raw_dir, dry_run=bool(args.dry_run), backup_dir=backup_dir)
        print(line, flush=True)
        if line.startswith("[OK]") or line.startswith("[dry]"):
            ok += 1
        else:
            skip += 1
    print(f"[i] done: patched={ok} skipped={skip}", flush=True)
    return 0 if skip == 0 or ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
