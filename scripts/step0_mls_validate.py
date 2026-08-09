"""Validate the MLS gradient operators against COMSOL's own spf.sr / d(spf.sr,x).

patient007 only (the vessel with a raw COMSOL export).  Reports:
  * linear/quadratic consistency of Dx, Dy
  * reconstructed ``spf.sr`` vs COMSOL, interior and wall
  * reconstructed ``d(spf.sr,x)`` vs COMSOL
  * both deposition gates as classifiers of the final committed wall set
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.mls_gradient import build_mls_gradient, shear_rate_2d  # noqa: E402

LSS, SGT = 25.0, -750.0
MAT_CRIT = 2.0e7


def spr(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def prf(pred, gt):
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    dm = np.load("outputs/comsol_p007_domain.npz")
    wl = np.load("outputs/comsol_p007_wall.npz")
    data = torch.load("data/processed/graphs_biochem_anchors/patient007.pt",
                      map_location="cpu", weights_only=False)
    pos = data.siren_pos.detach().numpy().astype(np.float64)
    wall = data.mask_wall.reshape(-1).bool().numpy()
    u_ref = float(data.u_ref.reshape(-1)[0])
    ei = data.edge_index.numpy()

    cx, cy = dm["x"][0], dm["y"][0]
    L0 = (cx.max() - cx.min()) / (pos[:, 0].max() - pos[:, 0].min())     # cm per pack unit
    P = np.stack([(pos[:, 0] - pos[:, 0].min()) * L0 + cx.min(),
                  (pos[:, 1] - pos[:, 1].min()) * L0 + cy.min()], 1)     # pack nodes in cm
    nn = cKDTree(np.stack([cx, cy], 1)).query(P)[1]
    csr, cdx = dm["sr"][0][nn], dm["dsrx"][0][nn]

    for hops in (2, 3):
        t0 = time.time()
        Dx, Dy = build_mls_gradient(P, ei, hops=hops)
        el = time.time() - t0
        ones = np.ones(len(P))
        print("\n=== hops=%d  (%.1fs, mean stencil %.1f) ===" % (hops, el, Dx.nnz / len(P)))
        print("  consistency: Dx(x) med %.4f  Dx(y) med %.2e  Dx(1) med %.2e  Dy(y) med %.4f"
              % (np.median(Dx @ P[:, 0]), np.median(np.abs(Dx @ P[:, 1])),
                 np.median(np.abs(Dx @ ones)), np.median(Dy @ P[:, 1])))

        u = data.y[0, :, 0].numpy().astype(np.float64) * u_ref * 100.0   # cm/s
        v = data.y[0, :, 1].numpy().astype(np.float64) * u_ref * 100.0
        sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v)               # 1/s
        for nm, m in (("interior", ~wall), ("wall", wall)):
            print("  sr %-9s spearman %.3f pearson %.3f  med ratio %.3f"
                  % (nm, spr(sr[m], csr[m]), np.corrcoef(sr[m], csr[m])[0, 1],
                     np.median(sr[m] / np.maximum(csr[m], 1e-9))))
        dsrx = Dx @ sr                                                    # 1/(s*cm)
        print("  dsrx wall  spearman %.3f pearson %.3f  med|ratio| %.3f"
              % (spr(dsrx[wall], cdx[wall]), np.corrcoef(dsrx[wall], cdx[wall])[0, 1],
                 np.median(np.abs(dsrx[wall]) / np.maximum(np.abs(cdx[wall]), 1e-9))))

        # gates vs the final committed wall set
        wnn = cKDTree(np.stack([wl["x"][0], wl["y"][0]], 1)).query(P[wall])[1]
        gt = wl["Mat"][-1][wnn] >= MAT_CRIT
        print("  [gates on wall, base rate %.3f]" % gt.mean())
        for nm, pred in (("COMSOL ref either", (csr[wall] < LSS) | (cdx[wall] < SGT)),
                         ("MLS sr<lss", sr[wall] < LSS),
                         ("MLS dsrx<sgt", dsrx[wall] < SGT),
                         ("MLS either", (sr[wall] < LSS) | (dsrx[wall] < SGT))):
            p, r, f1 = prf(pred, gt)
            print("     %-18s open %.3f prec %.3f rec %.3f F1 %.3f" % (nm, pred.mean(), p, r, f1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
