"""Where did the 8 mask nodes actually go, 81.5 -> 73.3?

``diag_corrector_rollout.py`` attributed the shrink to the corrector closing gates.  The
sign test (``diag_corrector_sign.py``) shows the corrector in fact LOWERS wake shear like
GT does and RAISES the low-shear open fraction, so that cause cannot be right.

The remaining suspect is a mask-definition mismatch the rollout could not see:

  * ``static Track A``      mask = t=0 gate ignition  UNION 6-hop shear-admitted GRAPH GROWTH
  * ``corrector rollout``   mask = ODE ignition only, NO growth term

``diag_rollout_trackA.py`` knew about this -- it carries a separate "+ front admission" arm
described as "the time-resolved analogue of 6-hop growth" -- but the corrector script has
no such arm, so the two arms it compares are not the same estimator.

This counts the split directly, and separately measures how far the corrector's influence
reaches (it has a 5-hop receptive field) against how far GT's does.
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

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import SHIPPED, make_rollout_hook  # noqa: E402
from src.core_physics.mls_gradient import (  # noqa: E402
    build_mls_gradient, node_positions, shear_rate_2d,
)
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, integrate_mat_trajectory, t0_flow_fields,
)

DIR = Path("data/processed/graphs_biochem_anchors")
RELAX, GROW, HOPS = 2.0, 6, 3


def main() -> int:
    import json
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    crit = float(bio.viscosity_mat_crit)
    lss = float(bio.lss)

    print("STATIC MASK COMPOSITION (the arm the corrector was compared against)\n")
    print("%-12s %8s %9s %9s %8s" % ("vessel", "mask", "ignited", "by growth", "growth%"))
    tot_m = tot_i = 0
    rows = []
    for n in names:
        p = DIR / f"{n}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        w = d.mask_wall.reshape(-1).bool().numpy()
        nn_ = len(w)
        ei = d.edge_index.numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(nn_, nn_)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        f0 = t0_flow_fields(d, bio, hops=HOPS, flow_source="gt")
        cur = (f0.gate > 0) & w
        adm = (f0.sr < lss * RELAX) & w
        for _ in range(GROW):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        hook = make_rollout_hook(SHIPPED, bio, f0.sr)
        traj, _ = integrate_mat_trajectory(d, bio, f0.gate * w, da_scale=40.0, ap_closure=hook)
        ign = (first_crossing(traj, crit) >= 0) & w
        nm, ni = int(cur.sum()), int((cur & ign).sum())
        tot_m += nm
        tot_i += ni
        rows.append((n, nm, ni))
        print("%-12s %8d %9d %9d %7.1f%%" % (n, nm, ni, nm - ni, 100 * (nm - ni) / max(nm, 1)))

    k = len(rows)
    print("\n%-12s %8.1f %9.1f %9.1f %7.1f%%"
          % ("MEAN", tot_m / k, tot_i / k, (tot_m - tot_i) / k,
             100 * (tot_m - tot_i) / max(tot_m, 1)))
    print("\n  static mask (with growth)      : %.1f" % (tot_m / k))
    print("  ignition-only subset           : %.1f   <- the estimator the corrector arm used"
          % (tot_i / k))
    print("  corrector rollout reported     : 73.3")
    print("\n  So the like-for-like baseline for the corrector arm is %.1f, not 81.5."
          % (tot_i / k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
