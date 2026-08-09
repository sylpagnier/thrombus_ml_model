"""Does the GT flow field actually evolve as the clot grows?

This decides arms 2 AND 3.  Both assume the frozen t=0 gate is wrong because the clot
narrows the lumen and changes the shear; arm 2 approximates that algebraically, arm 3
would spend a network on it.  But ``diag_blockage_magnitude.py`` measured the occluded
cross-section fraction at a median of 0.000 -- using the GT clot, not just the model's.

The packs carry the GT velocity at all 201 timesteps, so the premise is directly testable:
recompute spf.sr from GT (u,v) at every sampled time and measure how much the field, and
the gates built on it, actually move.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import build_mls_gradient, node_positions, shear_rate_2d  # noqa: E402
from src.core_physics.temporal_metrics import gt_onset_index, spearman  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = [a for a in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))]
    print("%12s %6s | %8s %8s %8s | %9s %9s | %8s"
          % ("vessel", "nWall", "rho(t0,tF)", "medRatio", "p90Ratio",
             "gateOpen0", "gateOpenF", "flipFrac"))
    rows = []
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        pos = node_positions(d)
        Dx, Dy = build_mls_gradient(pos, d.edge_index.numpy(), hops=3)
        u_ref = float(d.u_ref.reshape(-1)[0])
        d_bar = float(d.d_bar.reshape(-1)[0])
        nt = int(d.y.shape[0])

        def sr_at(ti):
            u = d.y[ti, :, 0].numpy().astype(np.float64)
            v = d.y[ti, :, 1].numpy().astype(np.float64)
            return shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v) * (u_ref / d_bar)

        s0, sf = sr_at(0), sr_at(nt - 1)
        g0 = s0[wall] < float(bio.lss)
        gf = sf[wall] < float(bio.lss)
        ratio = sf[wall] / np.maximum(s0[wall], 1e-9)
        r = dict(anchor=a, rho=spearman(s0[wall], sf[wall]),
                 med=float(np.median(ratio)), p90=float(np.percentile(ratio, 90)),
                 open0=float(g0.mean()), openf=float(gf.mean()),
                 flip=float((g0 != gf).mean()))
        rows.append(r)
        print("%12s %6d | %8.3f %8.3f %8.3f | %9.3f %9.3f | %8.3f"
              % (a, wall.sum(), r["rho"], r["med"], r["p90"], r["open0"], r["openf"], r["flip"]))

    print("\nmean over %d full-horizon vessels:" % len(rows))
    for k, lbl in (("rho", "spearman(sr @t=0, sr @t_final) at the wall"),
                   ("med", "median sr(t_final)/sr(t=0)"),
                   ("p90", "p90 sr(t_final)/sr(t=0)"),
                   ("flip", "fraction of wall nodes whose low-shear gate FLIPS")):
        print("   %-45s %.4f" % (lbl, float(np.mean([r[k] for r in rows]))))
    print("\n   If rho ~ 1 and flip ~ 0, the frozen t=0 gate is not the reason the growth")
    print("   curve is wrong, and arms 2 and 3 are both attacking a premise that does not hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
