"""Measure graph-hop distance of GT clot nodes from the wall (band/recall probe).

For each biochem anchor graph at t_final:
  - GT clot node := growth-only ``relu(mu_eff(t) - mu_eff(t=0)) >= thresh``
  - BFS hop distance from nearest wall node on the (undirected) biochem mesh
  - Histogram of clot nodes by hop (0=wall, 1, 2, 3, 4+)
  - Recall ceiling if supervision is restricted to wall+Hhops band.

Run:
  python scripts/_probe_clot_hop_distance.py
  python scripts/_probe_clot_hop_distance.py --triangle6-edges
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from src.config import PhysicsConfig, VesselConfig
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time
from src.core_physics.clot_phi_simple import clot_phi_thresh_si
from src.data_gen.lib.centerline_utils import resolve_anchor_mesh_path
from src.data_gen.lib.mesh_triangle6_edges import edge_index_from_mesh_path


def _resolve_mesh_path(anchor: str) -> Path | None:
    raw = REPO / VesselConfig(phase="biochem_anchors").mesh_input_dir
    return resolve_anchor_mesh_path(raw, anchor)


def bfs_hop_from_wall(edge_index: torch.Tensor, wall: torch.Tensor, n: int) -> torch.Tensor:
    """Undirected BFS hop distance from any wall node; -1 if unreachable."""
    dist = torch.full((n,), -1, dtype=torch.long)
    dist[wall] = 0
    row, col = edge_index[0], edge_index[1]
    frontier = wall.clone()
    h = 0
    while bool(frontier.any()):
        h += 1
        nxt = torch.zeros(n, dtype=torch.bool)
        nxt[col[frontier[row]]] = True
        nxt[row[frontier[col]]] = True
        nxt &= dist < 0
        dist[nxt] = h
        frontier = nxt
        if h > 10000:
            break
    return dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--triangle6-edges",
        action="store_true",
        help="Rebuild edge_index from anchor .nas/.msh (full P2 connectivity)",
    )
    args = ap.parse_args()

    root = REPO / "data" / "processed" / "graphs_biochem_anchors"
    paths = sorted(root.glob("patient*.pt"))
    phys = PhysicsConfig()
    thresh = clot_phi_thresh_si(phys)
    dev = torch.device("cpu")
    edge_mode = "triangle6" if args.triangle6_edges else "corner"

    print(
        f"GT clot = growth-only relu(mu_eff(t) - mu_eff(t=0)) >= {thresh:.4f} Pa*s, at t_final. "
        f"Hop = mesh distance from wall.  edges={edge_mode}"
    )
    print("-" * 104)
    hdr = (
        f"{'patient':<11}{'Nnode':>7}{'Nclot':>7}{'h0(wall)':>9}{'h1':>7}{'h2':>7}"
        f"{'h3':>7}{'h4+':>7}{'maxhop':>7}{'rec@w+1h':>9}{'rec@w+2h':>9}"
    )
    print(hdr)

    agg = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    tot_clot = 0
    for p in paths:
        data = torch.load(p, map_location="cpu", weights_only=False)
        n = int(data.num_nodes)
        y = data.y.to(dtype=torch.float32)
        t_final = int(y.shape[0]) - 1
        clot = gt_growth_commit_mask_at_time(data, t_final, phys, dev)

        if data.mask_wall is None:
            wall = torch.zeros(n, dtype=torch.bool)
        else:
            wall = data.mask_wall.view(-1).bool()

        edge_index = data.edge_index
        if args.triangle6_edges:
            mesh_path = _resolve_mesh_path(p.stem)
            if mesh_path is None:
                print(f"[WARN] skip {p.stem}: no mesh for triangle6 edges", flush=True)
                continue
            edge_index = edge_index_from_mesh_path(mesh_path)

        dist = bfs_hop_from_wall(edge_index, wall, n)
        nclot = int(clot.sum())
        if nclot == 0:
            print(f"{p.stem:<11}{n:>7}{0:>7}{'-':>9}{'-':>7}{'-':>7}{'-':>7}{'-':>7}{'-':>7}{'-':>9}{'-':>9}")
            continue

        cd = dist[clot]
        h0 = int((cd == 0).sum())
        h1 = int((cd == 1).sum())
        h2 = int((cd == 2).sum())
        h3 = int((cd == 3).sum())
        h4 = int((cd >= 4).sum())
        maxhop = int(cd.max())
        rec_w1 = (h0 + h1) / nclot
        rec_w2 = (h0 + h1 + h2) / nclot
        agg[0] += h0
        agg[1] += h1
        agg[2] += h2
        agg[3] += h3
        agg[4] += h4
        tot_clot += nclot
        print(
            f"{p.stem:<11}{n:>7}{nclot:>7}{h0:>9}{h1:>7}{h2:>7}{h3:>7}{h4:>7}{maxhop:>7}"
            f"{rec_w1:>9.3f}{rec_w2:>9.3f}"
        )

    print("-" * 104)
    if tot_clot:
        tot = tot_clot
        print(
            f"POOLED clot nodes={tot}: "
            f"h0={agg[0]/tot:.3f} h1={agg[1]/tot:.3f} h2={agg[2]/tot:.3f} "
            f"h3={agg[3]/tot:.3f} h4+={agg[4]/tot:.3f}"
        )
        print(
            f"POOLED recall ceiling  wall+1hop={ (agg[0]+agg[1])/tot:.3f}  "
            f"wall+2hop={ (agg[0]+agg[1]+agg[2])/tot:.3f}  "
            f"wall+3hop={ (agg[0]+agg[1]+agg[2]+agg[3])/tot:.3f}"
        )


if __name__ == "__main__":
    main()
