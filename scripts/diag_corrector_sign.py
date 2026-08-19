"""Does the corrector move near-clot wall shear the SAME WAY GT does?

``scripts/diag_corrector_rollout.py`` reported a clean negative and attributed it to a
sign error: GT opens low-shear gates as the clot grows, the corrector closes them.  The
proposed cause was that the patch factory trains on "an isolated obstruction in a free
channel", so the corrector learned diversion instead of wake.

That cause is checkable and this script checks it, isolating two confounds the rollout
test could not separate:

  1. **The Delta-mu the corrector was driven at.**  The rollout clamped to 3.0 Pa.s.  The
     patch factory's actual training range is ``_CLOT_MU_RANGE = (0.1, 10.0)`` Pa.s and
     the real GT Delta-mu at committed wall nodes is ~0.35-2.31 (median 0.68).  So 3.0 is
     ~4x the median real value -- inside the trained range, but over-applied.  If the sign
     is a magnitude artefact it will flip somewhere on this sweep.

  2. **Mask definition.**  The rollout compared an ODE-ignition-only corrector mask against
     a static mask that also carries 6-hop graph growth.  Here nothing is masked: the
     measurement is the shear field itself.

MEASURED, on GT's own committed set (so both arms see an identical clot):
    d_sr at wall nodes 1-2 hops OUTSIDE the committed set  -- the wake region
    versus GT's own sr change over the same interval at the same nodes.
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
from src.core_physics.mls_gradient import (  # noqa: E402
    build_mls_gradient, node_positions, shear_rate_2d,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
CKPT = Path("outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth")
VESSELS = ("patient044", "patient041", "patient012", "patient025", "patient018", "patient005")
DMUS = (0.1, 0.35, 0.68, 1.5, 3.0)
HOPS = 3


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.core_physics.coupled_shear_gnn import load_local_corrector
    from src.inference.corrector_coupling import couple_flow_with_corrector
    corr = load_local_corrector(CKPT, device)
    lss = float(bio.lss)

    print("Wake-region wall shear: median relative change vs the t=0 field.")
    print("Negative = shear DROPS near the clot (wake, opens the low-shear gate).")
    print("Positive = shear RISES (diversion, closes it).\n")
    print("%-12s %9s | %s" % ("vessel", "GT", "  ".join("dmu=%.2f" % m for m in DMUS)))

    agg = {m: [] for m in DMUS}
    agg_gt, agg_open = [], {m: [] for m in DMUS}
    gt_open = []
    for n in VESSELS:
        p = DIR / f"{n}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        w = d.mask_wall.reshape(-1).bool().numpy()
        nn_ = len(w)
        ei = d.edge_index.numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(nn_, nn_)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        Dx, Dy = build_mls_gradient(node_positions(d), ei, hops=HOPS)
        u_ref = float(d.u_ref.reshape(-1)[0])
        d_bar = float(d.d_bar.reshape(-1)[0])
        scale = u_ref / d_bar

        occ = gt_clot_phi_at_time(d, int(d.y.shape[0]) - 1, phys,
                                  device=torch.device("cpu")).reshape(-1).numpy() > 0.5
        occ = occ & w
        if occ.sum() < 5:
            continue
        # wake region: wall nodes within 2 hops of clot but NOT clot themselves
        near = occ.copy()
        for _ in range(2):
            near = near | ((A @ near.astype(np.int8)) > 0)
        wake = near & ~occ & w
        if wake.sum() < 5:
            continue

        u0 = d.y[0, :, 0].numpy().astype(np.float64)
        v0 = d.y[0, :, 1].numpy().astype(np.float64)
        sr0 = shear_rate_2d(Dx @ u0, Dy @ u0, Dx @ v0, Dy @ v0) * scale

        # GT's own answer: the real converged flow with the real clot present
        uT = d.y[-1, :, 0].numpy().astype(np.float64)
        vT = d.y[-1, :, 1].numpy().astype(np.float64)
        srT = shear_rate_2d(Dx @ uT, Dy @ uT, Dx @ vT, Dy @ vT) * scale
        g_rel = float(np.median((srT[wake] - sr0[wake]) / np.maximum(sr0[wake], 1e-9)))
        agg_gt.append(g_rel)
        gt_open.append(float((srT[wake] < lss).mean() - (sr0[wake] < lss).mean()))

        u0_t = torch.tensor(u0, dtype=torch.float32, device=device)
        v0_t = torch.tensor(v0, dtype=torch.float32, device=device)

        class _V:
            def __init__(s, dd):
                s.x = dd.x.to(device)
                s.edge_index = dd.edge_index.to(device)
                s.num_nodes = int(dd.num_nodes)
        dv = _V(d)

        cells = []
        for m in DMUS:
            delta = torch.tensor(occ.astype(np.float32) * m, device=device)
            with torch.no_grad():
                uu, vv, _ = couple_flow_with_corrector(
                    dv, u0_t, v0_t, delta, corrector=corr, phys_cfg=phys,
                    device=device, num_hops=5)
            un = uu.detach().cpu().numpy().astype(np.float64)
            vn = vv.detach().cpu().numpy().astype(np.float64)
            src = shear_rate_2d(Dx @ un, Dy @ un, Dx @ vn, Dy @ vn) * scale
            r = float(np.median((src[wake] - sr0[wake]) / np.maximum(sr0[wake], 1e-9)))
            agg[m].append(r)
            agg_open[m].append(float((src[wake] < lss).mean() - (sr0[wake] < lss).mean()))
            cells.append("%+8.3f" % r)
        print("%-12s %+8.3f | %s" % (n, g_rel, "  ".join(cells)))

    print("\n%-12s %+8.3f | %s" % ("MEAN", float(np.mean(agg_gt)),
                                   "  ".join("%+8.3f" % np.mean(agg[m]) for m in DMUS)))
    print("\nChange in low-shear-gate OPEN fraction over the wake region:")
    print("%-12s %+8.3f | %s" % ("MEAN", float(np.mean(gt_open)),
                                 "  ".join("%+8.3f" % np.mean(agg_open[m]) for m in DMUS)))
    print("\n(GT column = the real converged flow with the real clot: the target sign.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
