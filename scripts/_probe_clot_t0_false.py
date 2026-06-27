"""Compare absolute mu threshold vs growth-only GT clot (subtract t=0 baseline).

Run: python scripts/_probe_clot_t0_false.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from src.config import PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    cap_mu_eff_si,
    clot_phi_thresh_si,
    gt_mu_anchor_cap_si,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time


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
    thresh = clot_phi_thresh_si(phys)
    dev = torch.device("cpu")

    print(f"Canonical GT clot = relu(mu_eff(t) - mu_eff(t=0)) >= {thresh:.4f} Pa*s")
    print(f"Hop probe uses absolute mu_eff(t_final) >= 0.055 (NO t=0 subtraction)")
    print("-" * 100)
    hdr = (
        f"{'patient':<12}{'abs@t0':>8}{'abs@tf':>8}{'grow@tf':>9}{'false@tf':>10}"
        f"{'pct_false':>11}{'h4+_abs':>9}{'h4+_grow':>10}{'rec_w1_abs':>11}{'rec_w1_grow':>12}"
    )
    print(hdr)

    agg = {"abs_tf": 0, "grow_tf": 0, "false": 0, "h4_abs": 0, "h4_grow": 0, "w1_abs": 0, "w1_grow": 0}

    for p in sorted(root.glob("patient*.pt")):
        data = torch.load(p, map_location="cpu", weights_only=False)
        n = int(data.num_nodes)
        y = data.y.to(dtype=torch.float32)
        tf = int(y.shape[0]) - 1
        mu_tf = cap_mu_eff_si(phys.viscosity_nd_to_si(y[tf, :, STATE_CHANNEL_MU_EFF_ND])).reshape(-1)
        mu_t0 = cap_mu_eff_si(phys.viscosity_nd_to_si(y[0, :, STATE_CHANNEL_MU_EFF_ND])).reshape(-1)
        anchor = gt_mu_anchor_cap_si(data, phys, dev).reshape(-1)
        abs_t0 = mu_t0 >= thresh
        abs_tf = mu_tf >= thresh
        grow_tf = (mu_tf - anchor).clamp(min=0.0) >= thresh
        false_tf = abs_tf & ~grow_tf

        wall = data.mask_wall.view(-1).bool() if data.mask_wall is not None else torch.zeros(n, dtype=torch.bool)
        dist = bfs_hop_from_wall(data.edge_index, wall, n)

        n_abs = int(abs_tf.sum())
        n_grow = int(grow_tf.sum())
        n_false = int(false_tf.sum())
        h4_abs = int((abs_tf & (dist >= 4)).sum())
        h4_grow = int((grow_tf & (dist >= 4)).sum())
        w1_abs = int((abs_tf & (dist <= 1)).sum()) if n_abs else 0
        w1_grow = int((grow_tf & (dist <= 1)).sum()) if n_grow else 0
        pct = 100.0 * n_false / max(n_abs, 1)
        rec_w1_abs = w1_abs / max(n_abs, 1)
        rec_w1_grow = w1_grow / max(n_grow, 1)

        agg["abs_tf"] += n_abs
        agg["grow_tf"] += n_grow
        agg["false"] += n_false
        agg["h4_abs"] += h4_abs
        agg["h4_grow"] += h4_grow
        agg["w1_abs"] += w1_abs
        agg["w1_grow"] += w1_grow

        print(
            f"{p.stem:<12}{int(abs_t0.sum()):>8}{n_abs:>8}{n_grow:>9}{n_false:>10}"
            f"{pct:>10.1f}%{h4_abs:>9}{h4_grow:>10}{rec_w1_abs:>11.3f}{rec_w1_grow:>12.3f}"
        )

    print("-" * 100)
    tot_abs = agg["abs_tf"]
    tot_grow = agg["grow_tf"]
    tot_false = agg["false"]
    print(
        f"POOLED: abs@tf={tot_abs} grow@tf={tot_grow} false={tot_false} "
        f"({100*tot_false/max(tot_abs,1):.1f}% of abs clots are t=0 baseline artifacts)"
    )
    print(
        f"        h4+ abs={agg['h4_abs']} grow={agg['h4_grow']} | "
        f"rec@w+1hop abs={agg['w1_abs']/max(tot_abs,1):.3f} grow={agg['w1_grow']/max(tot_grow,1):.3f}"
    )

    # sanity: canonical API matches growth mask
    p007 = root / "patient007.pt"
    data = torch.load(p007, map_location="cpu", weights_only=False)
    tf = int(data.y.shape[0]) - 1
    phi = gt_clot_phi_at_time(data, tf, phys, dev).reshape(-1).bool()
    mu_tf = cap_mu_eff_si(phys.viscosity_nd_to_si(data.y[tf, :, STATE_CHANNEL_MU_EFF_ND])).reshape(-1)
    grow = (mu_tf - gt_mu_anchor_cap_si(data, phys, dev).reshape(-1)).clamp(min=0.0) >= thresh
    print(f"[OK] gt_clot_phi_at_time matches growth mask on p007: {(phi == grow).all().item()}")


if __name__ == "__main__":
    main()
