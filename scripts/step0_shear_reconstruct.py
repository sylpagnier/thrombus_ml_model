"""Can COMSOL's spf.sr and d(spf.sr,x) be reconstructed from the pack's t=0 GT flow?

The repo's ``gamma_si`` rank-correlates 0.19 with COMSOL ``spf.sr`` and its
``dshear_ds`` is identically zero.  The gates are therefore never actually evaluated,
which explains PHASE3_HANDOFF 1.4/1.5b.  Under the Phase-3 bandaid the GT flow at t=0
IS available (``data.y[0,:,0:2]``), so the question is purely one of discretisation.

Tries several operators/scalings on patient007 and scores each against COMSOL truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402

LSS, SGT_CGS = 25.0, -750.0
MAT_CRIT = 2.0e7


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def prf(pred, gt):
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    c = np.load("outputs/comsol_p007_wall.npz")
    cx, cy = c["x"][0], c["y"][0]
    data = torch.load("data/processed/graphs_biochem_anchors/patient007.pt",
                      map_location="cpu", weights_only=False)
    bio = BiochemConfig(phase="biochem")
    wall = data.mask_wall.reshape(-1).bool().numpy()
    pos = data.siren_pos.detach().numpy()

    # nearest-neighbour match wall nodes (bbox-normalised; validated exact earlier)
    def nrm(a, b):
        return ((a - a.min()) / (a.max() - a.min() + 1e-12),
                (b - b.min()) / (b.max() - b.min() + 1e-12))
    pw = pos[wall]
    px, qx = nrm(pw[:, 0], cx)
    py, qy = nrm(pw[:, 1], cy)
    D = ((np.stack([px, py], 1)[:, None] - np.stack([qx, qy], 1)[None]) ** 2).sum(-1)
    nn = D.argmin(1)

    print("u_ref=%s d_bar=%s re_actual=%s"
          % (data.u_ref.reshape(-1).tolist(), data.d_bar.reshape(-1).tolist(),
             data.re_actual.reshape(-1).tolist()))
    # pack length unit -> cm
    L0_cm = (cx.max() - cx.min()) / (pos[:, 0].max() - pos[:, 0].min())
    print("pack length unit = %.5f cm  (y check %.5f)"
          % (L0_cm, (cy.max() - cy.min()) / (pos[:, 1].max() - pos[:, 1].min())))

    u = data.y[0, :, 0].float()
    v = data.y[0, :, 1].float()
    Gx, Gy = data.G_x, data.G_y

    def mm(G, f):
        return torch.sparse.mm(G, f.reshape(-1, 1)).reshape(-1)

    du_dx, du_dy = mm(Gx, u), mm(Gy, u)
    dv_dx, dv_dy = mm(Gx, v), mm(Gy, v)
    # COMSOL spf.sr for incompressible 2D = sqrt(2*(exx^2+eyy^2) + (exy+eyx)^2)... use
    # the repo's compute_shear_rate as well as the standard second invariant.
    from src.utils.rheology import compute_shear_rate
    g_repo = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy).numpy()
    exx, eyy = du_dx.numpy(), dv_dy.numpy()
    exy = 0.5 * (du_dy.numpy() + dv_dx.numpy())
    g_inv2 = np.sqrt(2.0 * (2 * exx ** 2 + 2 * eyy ** 2 + 4 * exy ** 2) / 2.0)
    g_simple = np.sqrt(2 * exx ** 2 + 2 * eyy ** 2 + (du_dy.numpy() + dv_dx.numpy()) ** 2)

    cs = c["sr"][0][nn]
    cd = c["dsrx"][0][nn]
    print("\n[shear operator]  (wall nodes, vs COMSOL spf.sr)")
    for nm, g in (("repo compute_shear_rate", g_repo), ("2nd invariant", g_inv2),
                  ("sqrt(2exx^2+2eyy^2+(uy+vx)^2)", g_simple)):
        gw = g[wall]
        k = float(np.median(cs / np.maximum(gw, 1e-12)))
        print("   %-30s spearman %.3f  pearson %.3f  median scale %.4g"
              % (nm, spearman(gw, cs), np.corrcoef(gw, cs)[0, 1], k))

    # ---- what scale converts nd shear to 1/s? -------------------------------
    gw = g_simple[wall]
    lsq = float((gw @ cs) / (gw @ gw))
    print("   lsq scale = %.4f   (u_ref/d_bar = %.4f)"
          % (lsq, float(data.u_ref.reshape(-1)[0] / data.d_bar.reshape(-1)[0])))

    # ---- gradient of shear in x ---------------------------------------------
    g_full = torch.tensor(g_simple * lsq, dtype=torch.float32)   # 1/s on all nodes
    dsr_dx_nd = mm(Gx, g_full).numpy()                            # 1/s per nd-length
    dsr_dx_cm = dsr_dx_nd / L0_cm                                 # 1/(s*cm)
    print("\n[shear-gradient operator]  vs COMSOL d(spf.sr,x)")
    print("   G_x(sr) spearman %.3f pearson %.3f"
          % (spearman(dsr_dx_cm[wall], cd), np.corrcoef(dsr_dx_cm[wall], cd)[0, 1]))
    print("   COMSOL dsrx pct[5,50,95] %s" % np.round(np.percentile(cd, [5, 50, 95]), 1))
    print("   recon  dsrx pct[5,50,95] %s" % np.round(np.percentile(dsr_dx_cm[wall], [5, 50, 95]), 1))
    lsq_d = float((dsr_dx_cm[wall] @ cd) / (dsr_dx_cm[wall] @ dsr_dx_cm[wall]))
    print("   lsq rescale %.4f" % lsq_d)

    # ---- do the reconstructed gates classify? -------------------------------
    gt = c["Mat"][-1][nn] >= MAT_CRIT
    sr_r = (g_simple * lsq)[wall]
    dx_r = dsr_dx_cm[wall] * lsq_d
    print("\n[gates on reconstructed fields]  base rate %.3f" % gt.mean())
    for nm, pred in (
        ("COMSOL either (ref)", (cs < LSS) | (cd < SGT_CGS)),
        ("recon sr<lss", sr_r < LSS),
        ("recon dsrx<sgt", dx_r < SGT_CGS),
        ("recon either", (sr_r < LSS) | (dx_r < SGT_CGS)),
    ):
        p, r, f1 = prf(pred, gt)
        print("   %-22s open %.3f prec %.3f rec %.3f F1 %.3f" % (nm, pred.mean(), p, r, f1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
