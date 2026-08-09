"""What IS off-wall clot, structurally? The measurement the lumen specialist should encode.

The wall model is settled (docs/PHASE3_RESULTS.md).  Before building a lumen arm, measure
the target:

  * how much GT clot sits off-wall, by hop distance from the wall
  * is an off-wall clot node always "behind" a committed wall node (i.e. is the lumen clot
    a THICKNESS on the wall clot, or does it nucleate independently)?
  * how well does the wall GT, dilated k hops into the lumen, reproduce the off-wall GT?
    -- this is the ceiling for any pure-thickness lumen model
  * does penetration depth track a local flow quantity (speed / shear) at t=0?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")


def adj_of(data, n):
    ei = data.edge_index.numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def hops_from(seed, A, max_h=8):
    """BFS hop distance from a boolean seed set; unreached = max_h+1."""
    d = np.full(len(seed), max_h + 1, dtype=np.int16)
    cur = seed.copy()
    d[cur] = 0
    for h in range(1, max_h + 1):
        nxt = ((A @ cur.astype(np.int8)) > 0) & ~cur
        if not nxt.any():
            break
        d[nxt] = h
        cur = cur | nxt
    return d


def prf(pred, gt):
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
    print("%12s %5s %6s %6s %7s | %5s %5s %5s %5s %5s | %6s"
          % ("vessel", "T", "nGTw", "nGTof", "off/all", "h1", "h2", "h3", "h4", "h5+", "orphan"))
    tot = dict(w=0, off=0, hop={h: 0 for h in range(1, 6)}, orphan=0)
    dil_rows, depth_rows = [], []
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        n = int(d.num_nodes)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        A = adj_of(d, n)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = (gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu"))
              .reshape(-1).numpy() > 0.5)
        hw = hops_from(wall, A)
        gt_w, gt_off = gt & wall, gt & ~wall
        hc = hops_from(gt_w, A)
        orphan = int((gt_off & (hc > 6)).sum())
        row = [int((gt_off & (hw == h)).sum()) for h in range(1, 6)]
        tot["w"] += int(gt_w.sum()); tot["off"] += int(gt_off.sum()); tot["orphan"] += orphan
        for i, h in enumerate(range(1, 6)):
            tot["hop"][h] += row[i]
        print("%12s %5d %6d %6d %7.3f | %5d %5d %5d %5d %5d | %6d"
              % (a, int(d.y.shape[0]), int(gt_w.sum()), int(gt_off.sum()),
                 gt_off.sum() / max(gt.sum(), 1), row[0], row[1], row[2], row[3],
                 int((gt_off & (hw >= 5)).sum()), orphan))

        best = (0.0, 0)
        cur = gt_w.copy()
        for k in range(1, 7):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & ~wall)
            f1 = prf(cur & ~wall, gt_off)[2]
            if f1 > best[0]:
                best = (f1, k)
        dil_rows.append(best)
        f = t0_flow_fields(d, bio, hops=3)
        u = d.y[0, :, 0].numpy(); v = d.y[0, :, 1].numpy()
        depth_rows.append((a, int(gt_off.sum()),
                           float(np.median(np.hypot(u, v)[gt_off])) if gt_off.any() else np.nan,
                           float(np.median(np.hypot(u, v)[~wall & ~gt])),
                           float(np.median(f.sr[gt_off])) if gt_off.any() else np.nan,
                           float(np.median(f.sr[~wall & ~gt]))))

    print("\ncohort totals: wall clot %d, off-wall clot %d (%.1f%% of all clot)"
          % (tot["w"], tot["off"], 100 * tot["off"] / max(tot["w"] + tot["off"], 1)))
    print("  off-wall by hop from wall: %s" % {h: tot["hop"][h] for h in range(1, 6)})
    print("  off-wall nodes NOT within 6 hops of committed wall tissue: %d (%.2f%%)"
          % (tot["orphan"], 100 * tot["orphan"] / max(tot["off"], 1)))
    f1s = [x[0] for x in dil_rows]
    ks = [x[1] for x in dil_rows]
    print("\n[CEILING] GT wall clot dilated k hops into the lumen, vs off-wall GT:")
    print("  best-k F1 mean %.3f  median %.3f  (per-vessel best k: median %d, range %d-%d)"
          % (float(np.mean(f1s)), float(np.median(f1s)), int(np.median(ks)), min(ks), max(ks)))
    print("  -> a pure THICKNESS model (dilate the wall clot) cannot beat this")
    print("\n[flow contrast at off-wall clot vs clear lumen]")
    sp_c = [r[2] for r in depth_rows if r[2] == r[2]]
    sp_n = [r[3] for r in depth_rows if r[2] == r[2]]
    sr_c = [r[4] for r in depth_rows if r[4] == r[4]]
    sr_n = [r[5] for r in depth_rows if r[4] == r[4]]
    print("  median speed_nd : clot %.4f  clear %.4f  (ratio %.2f)"
          % (np.mean(sp_c), np.mean(sp_n), np.mean(sp_c) / max(np.mean(sp_n), 1e-9)))
    print("  median shear 1/s: clot %.2f  clear %.2f  (ratio %.2f)"
          % (np.mean(sr_c), np.mean(sr_n), np.mean(sr_c) / max(np.mean(sr_n), 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
