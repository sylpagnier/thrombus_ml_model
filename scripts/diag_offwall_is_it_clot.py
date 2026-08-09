"""Is off-wall "clot" actually clot, or is it shear-thinning of the evolving flow?

``gt_clot_phi_at_time`` labels a node as clot when ``relu(mu_eff(t) - mu_eff(0))`` clears a
threshold.  In COMSOL, ``spf.mu = mu_blood(shear) * (1 + mu1(Mat) + mu2(fi))``:

  * ``mu1(Mat)`` is the platelet gelation step -- Mat is a WALL species
  * ``mu2(fi)``  is fibrin -- PHASE3_HANDOFF 1.2 says it is provably inert
  * ``mu_blood(shear)`` is shear-thinning -- and the shear field CHANGES as the clot grows

So a lumen node can gain viscosity with no clot in it at all, purely because the clot
upstream slowed the flow past it.  If that is what the off-wall label mostly is, then a
"lumen specialist" is a flow-change predictor, not a clot model, and no local rule built
on the t=0 field can supply it.

Measured on the patient007 domain export (51240 nodes, t = 0 / 6000 / 18000 / 30000).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402


def main() -> int:
    dm = np.load("outputs/comsol_p007_domain2.npz")
    data = torch.load("data/processed/graphs_biochem_anchors/patient007.pt",
                      map_location="cpu", weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    pos = data.siren_pos.detach().numpy().astype(np.float64)
    wall = data.mask_wall.reshape(-1).bool().numpy()
    n = int(data.num_nodes)

    cx, cy = dm["x"][0], dm["y"][0]
    L0 = (cx.max() - cx.min()) / (pos[:, 0].max() - pos[:, 0].min())
    P = np.stack([(pos[:, 0] - pos[:, 0].min()) * L0 + cx.min(),
                  (pos[:, 1] - pos[:, 1].min()) * L0 + cy.min()], 1)
    nn = cKDTree(np.stack([cx, cy], 1)).query(P)[1]

    mu0, muF = dm["mu"][0][nn], dm["mu"][-1][nn]
    mu1_0, mu1_F = dm["mu1"][0][nn], dm["mu1"][-1][nn]
    mu2_F = dm["mu2"][-1][nn]
    sr0, srF = dm["sr"][0][nn], dm["sr"][-1][nn]

    # pack's own GT clot label at the eval time
    from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index
    t_eval = resolve_deploy_eval_time_index(int(data.y.shape[0]))
    gt = (gt_clot_phi_at_time(data, t_eval, phys, device=torch.device("cpu"))
          .reshape(-1).numpy() > 0.5)

    ei = data.edge_index.numpy()
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    hw = np.full(n, 99, dtype=np.int16)
    cur = wall.copy()
    hw[cur] = 0
    for h in range(1, 9):
        nxt = ((A @ cur.astype(np.int8)) > 0) & ~cur
        if not nxt.any():
            break
        hw[nxt] = h
        cur = cur | nxt

    off = gt & ~wall
    print("patient007: GT clot %d (wall %d, off-wall %d)"
          % (gt.sum(), (gt & wall).sum(), off.sum()))
    print("\n[does mu1 -- the platelet gelation step -- fire off-wall?]")
    print("  mu1 at t_final: max %.4g   nodes with mu1 > 1e-6: %d  (of %d)"
          % (mu1_F.max(), int((mu1_F > 1e-6).sum()), n))
    print("  of the %d off-wall GT clot nodes, mu1 > 1e-6 on %d (%.1f%%)"
          % (off.sum(), int((mu1_F[off] > 1e-6).sum()),
             100 * (mu1_F[off] > 1e-6).mean() if off.any() else 0.0))
    print("  mu2 (fibrin) at t_final: max %.4g  nodes > 1e-6: %d"
          % (mu2_F.max(), int((mu2_F > 1e-6).sum())))

    print("\n[what actually changed mu at the off-wall clot nodes?]")
    dmu = muF - mu0
    struct = mu1_F - mu1_0 + mu2_F          # the gelation contribution (relative units)
    print("  off-wall clot: median d(spf.mu) %.5g,  median d(mu1+mu2) %.5g"
          % (float(np.median(dmu[off])) if off.any() else np.nan,
             float(np.median(struct[off])) if off.any() else np.nan))
    print("  off-wall clot: median shear %.2f -> %.2f  1/s  (ratio %.3f)"
          % (float(np.median(sr0[off])), float(np.median(srF[off])),
             float(np.median(srF[off] / np.maximum(sr0[off], 1e-9)))))
    print("  wall clot    : median shear %.2f -> %.2f  1/s"
          % (float(np.median(sr0[gt & wall])), float(np.median(srF[gt & wall]))))

    # Carreau-style check: is d(mu) at off-wall clot explained by shear change alone?
    clear = (~gt) & (~wall) & (hw >= 2)
    print("\n[control: lumen nodes with NO clot label]")
    print("  median d(spf.mu) %.5g   median shear %.2f -> %.2f"
          % (float(np.median(dmu[clear])), float(np.median(sr0[clear])),
             float(np.median(srF[clear]))))
    print("\n  ratio of median d(mu): off-wall-clot / clear-lumen = %.2f"
          % (float(np.median(dmu[off])) / max(float(np.median(dmu[clear])), 1e-12)
             if off.any() else np.nan))

    print("\n[hop profile of mu1 > 1e-6 at t_final]")
    for h in range(0, 5):
        m = hw == h
        print("   hop %d: %5d nodes, mu1>1e-6 on %4d, GT clot %4d"
              % (h, int(m.sum()), int((mu1_F[m] > 1e-6).sum()), int(gt[m].sum())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
