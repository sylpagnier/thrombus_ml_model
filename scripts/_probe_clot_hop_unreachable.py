"""Break down growth-only GT clots by wall-hop, including unreachable (hop=-1)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from src.config import PhysicsConfig
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time


def bfs_hop_from_wall(edge_index: torch.Tensor, wall: torch.Tensor, n: int) -> torch.Tensor:
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
    root = REPO / "data" / "processed" / "graphs_biochem_anchors"
    phys = PhysicsConfig()
    dev = torch.device("cpu")

    hdr = f"{'patient':<12}{'Nclot':>7}{'unreach':>9}{'pct_unr':>9}{'hop2':>7}{'hop3+':>7}{'rec_w3':>9}"
    print(hdr)
    print("-" * len(hdr))

    agg = {"clot": 0, "unr": 0, "h2": 0, "h3p": 0, "in_w3": 0}
    for p in sorted(root.glob("patient*.pt")):
        data = torch.load(p, map_location="cpu", weights_only=False)
        n = int(data.num_nodes)
        tf = int(data.y.shape[0]) - 1
        clot = gt_growth_commit_mask_at_time(data, tf, phys, dev)
        wall = data.mask_wall.view(-1).bool() if data.mask_wall is not None else torch.zeros(n, dtype=torch.bool)
        dist = bfs_hop_from_wall(data.edge_index, wall, n)
        cd = dist[clot]
        nc = int(clot.sum())
        unr = int((cd < 0).sum())
        h2 = int((cd == 2).sum())
        h3p = int((cd >= 3).sum())
        in_w3 = int(((cd >= 0) & (cd <= 3)).sum())
        agg["clot"] += nc
        agg["unr"] += unr
        agg["h2"] += h2
        agg["h3p"] += h3p
        agg["in_w3"] += in_w3
        rec_w3 = in_w3 / max(nc, 1)
        print(f"{p.stem:<12}{nc:>7}{unr:>9}{100*unr/max(nc,1):>8.1f}%{h2:>7}{h3p:>7}{rec_w3:>9.3f}")

    print("-" * len(hdr))
    tot = agg["clot"]
    print(
        f"{'POOLED':<12}{tot:>7}{agg['unr']:>9}{100*agg['unr']/max(tot,1):>8.1f}%"
        f"{agg['h2']:>7}{agg['h3p']:>7}{agg['in_w3']/max(tot,1):>9.3f}"
    )
    print()
    print("[i] unreach = growth clot nodes with hop=-1 (no mesh path to any wall node)")
    print("[i] rec_w3 = fraction with hop 0..3 (matches wall+3hop band ceiling)")
    print(f"[i] missing from band = {agg['unr']} unreachable + {agg['h3p']} at hop>=3 "
          f"= {agg['unr']+agg['h3p']} ({100*(agg['unr']+agg['h3p'])/max(tot,1):.1f}%)")


if __name__ == "__main__":
    main()
