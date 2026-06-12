"""GT clot coverage vs wall-hop masks (patient007 @ t=53)."""
from __future__ import annotations

import os

import numpy as np
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import (
    graph_dilate_hops,
    gt_clot_mask_at_time,
    resolve_ceiling_mask,
    resolve_growth_support_at_time,
    resolve_t0_dgamma_wall_mask,
)
from src.core_physics.clot_phi_simple import (
    _wall_mask_from_data,
    clot_phi_thresh_si,
    neighbor_supervision_mask,
    supervision_region_mask,
)
from src.evaluation.clot_shape_score import graph_hop_distance_from_seeds
from src.utils.paths import get_project_root


def _coverage(name: str, region: torch.Tensor, gt: torch.Tensor) -> dict:
    r = region.reshape(-1).cpu().bool()
    g = gt.reshape(-1).cpu().bool()
    gt_n = int(g.sum())
    in_r = int((g & r).sum())
    frac = float(in_r / gt_n) if gt_n else 0.0
    return {"name": name, "n": int(r.sum()), "gt_in": in_r, "gt_n": gt_n, "recall": frac}


def _t0_dilate(data, device, bio, hops: int) -> torch.Tensor:
    t0 = resolve_t0_dgamma_wall_mask(data, device, bio)
    if hops <= 0:
        return t0
    return graph_dilate_hops(t0, data.edge_index.to(device), hops)


def main() -> None:
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("CLOT_PHI_DGAMMA_REF_TIME", "0")
    os.environ.setdefault("CLOT_PHI_MASK_MODE", "neighbor")
    os.environ.setdefault("CLOT_PHI_CLOT_TOUCH_HOPS", "1")

    root = get_project_root()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(root / "data/processed/graphs_biochem_anchors/patient007.pt", weights_only=False)
    ti = 53
    n = int(data.num_nodes)
    thr = clot_phi_thresh_si(phys)
    gt = gt_clot_mask_at_time(data, ti, phys, device)
    gt_np = gt.cpu().numpy()

    wall = _wall_mask_from_data(data, device, n)
    t0 = resolve_t0_dgamma_wall_mask(data, device, bio)
    y = data.y[ti]
    mu_cap = phys.viscosity_nd_to_si(y[:, 3]).clamp(max=0.10)

    # Oracle neighbor @ t=53 (viz_clot_phi_masks style: seeds from GT mu @ label time)
    neighbor_t53 = supervision_region_mask(data, device, mu_cap, phys_cfg=phys)

    # Neighbor from empty seeds (wall + 1-hop only)
    neighbor_wall = neighbor_supervision_mask(data, device, torch.zeros(n, dtype=torch.bool, device=device))

    rows = [
        _coverage("wall", wall, gt),
        _coverage("t0 dgamma wall", t0, gt),
        _coverage("neighbor @ GT t=53 (oracle)", neighbor_t53, gt),
        _coverage("neighbor wall-only (no seeds)", neighbor_wall, gt),
        _coverage("support ceiling_growth @53", resolve_growth_support_at_time(data, ti, device, phys, bio), gt),
        _coverage("ceiling wall+2 hops", resolve_ceiling_mask(data, device, bio, ceiling_hops=2), gt),
    ]
    for h in (1, 2, 3, 5, 8, 10, 12):
        rows.append(_coverage(f"t0 dilate +{h} hops", _t0_dilate(data, device, bio, h), gt))

    print(f"patient007 t={ti}  GT clot n={int(gt.sum())}  thr={thr:.3f} Pa*s\n")
    print(f"{'mask':<32} {'n':>6} {'gt_in':>6} {'recall':>8}")
    print("-" * 56)
    for r in rows:
        print(f"{r['name']:<32} {r['n']:>6} {r['gt_in']:>6} {r['recall']:>8.3f}")

    # Hop distance from t0 seed to GT clot nodes
    t0_np = t0.cpu().numpy()
    hop = graph_hop_distance_from_seeds(data.edge_index, n, t0_np, max_hops=32)
    gt_hops = hop[gt_np]
    if gt_hops.size:
        print("\nHop distance from t0_mask to GT clot nodes:")
        for h in range(0, 13):
            c = int((gt_hops == h).sum())
            if c:
                print(f"  hop {h:2d}: {c:4d} nodes")
        print(f"  hop >12: {int((gt_hops > 12).sum())} nodes")
        print(f"  median hop: {float(np.median(gt_hops)):.1f}  max: {int(gt_hops.max())}")

    # Hop from wall to GT clot
    wall_np = wall.cpu().numpy()
    hop_w = graph_hop_distance_from_seeds(data.edge_index, n, wall_np, max_hops=32)
    gt_hops_w = hop_w[gt_np]
    if gt_hops_w.size:
        print("\nHop distance from wall to GT clot nodes:")
        for h in range(0, 8):
            c = int((gt_hops_w == h).sum())
            if c:
                print(f"  hop {h}: {c} nodes")
        print(f"  median: {float(np.median(gt_hops_w)):.1f}  max: {int(gt_hops_w.max())}")


if __name__ == "__main__":
    main()
