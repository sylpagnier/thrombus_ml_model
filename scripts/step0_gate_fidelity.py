"""Do the REPO's reconstructed shear fields reproduce COMSOL's spf.sr / d(spf.sr,x)?

COMSOL's own t=0 gate classifies patient007's final committed wall set at
precision 0.981 / recall 0.760.  The repo's reconstructed gate scores AUC 0.51-0.66
(PHASE3_HANDOFF 1.4).  Either the graph is a different mesh, or the reconstruction of
``sr`` / ``dsrx`` from G_x/G_y loses the signal.  This matches the two node sets by
nearest neighbour and measures the reconstruction directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_t0_extended_probe import build_feature_table_at_time  # noqa: E402

MAT_CRIT = 2.0e7
LSS, SGT_CGS, SGT_SI = 25.0, -750.0, -7.5e4


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.concatenate([pos, neg])
    order = a.argsort()
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if cnt.max() > 1:
        s = np.zeros(len(u))
        np.add.at(s, inv, ranks)
        ranks = (s / cnt)[inv]
    n1 = len(pos)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))


def prf(pred, gt):
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    c = np.load("outputs/comsol_p007_wall.npz")
    cx, cy = c["x"][0], c["y"][0]           # cm, static
    csr0, cdsrx0 = c["sr"][0], c["dsrx"][0]
    cmat_f = c["Mat"][-1]

    data = torch.load("data/processed/graphs_biochem_anchors/patient007.pt",
                      map_location="cpu", weights_only=False)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    wall = data.mask_wall.reshape(-1).bool().numpy()
    f = build_feature_table_at_time(data, 0, device=torch.device("cpu"),
                                    phys_cfg=phys, bio_cfg=bio)
    gam = f["gamma_si"][0].numpy()
    dsh = f["dshear_ds"][0].numpy()
    ndg = f["neg_dgamma_dx"][0].numpy() if "neg_dgamma_dx" in f else None

    names = data.y_channel_names.split(",")
    mat_nd = torch.expm1(data.y[-1, :, names.index("Mat_log1p_nd")].clamp(-10, 8)).numpy()
    pmat_f = mat_nd * float(bio.Minf)     # COMSOL model units (plt/cm^2 convention)

    pos = data.siren_pos.detach().numpy() if hasattr(data, "siren_pos") else None
    print("pack nodes=%d wall=%d | comsol wall nodes=%d" % (len(wall), wall.sum(), len(cx)))
    print("pack pos range x[%.3f,%.3f] y[%.3f,%.3f]"
          % (pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max()))
    print("comsol   range x[%.3f,%.3f] y[%.3f,%.3f]" % (cx.min(), cx.max(), cy.min(), cy.max()))

    # --- affine match: rescale comsol coords onto pack coords by bbox --------
    def norm(a, b):
        return (a - a.min()) / (a.max() - a.min() + 1e-12), (b - b.min()) / (b.max() - b.min() + 1e-12)

    pw = pos[wall]
    px, qx = norm(pw[:, 0], cx)
    py, qy = norm(pw[:, 1], cy)
    P = np.stack([px, py], 1)
    Q = np.stack([qx, qy], 1)
    dmat = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
    nn = dmat.argmin(1)
    dist = np.sqrt(dmat[np.arange(len(P)), nn])
    good = dist < 0.02
    print("matched %d/%d pack wall nodes within 0.02 normalised (median dist %.4f)"
          % (good.sum(), len(P), np.median(dist)))

    gm, dm = gam[wall][good], dsh[wall][good]
    cs, cd = csr0[nn][good], cdsrx0[nn][good]
    cmf, pmf = cmat_f[nn][good], pmat_f[wall][good]

    def sp(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    print("\n[shear]  repo gamma_si vs comsol spf.sr : pearson %.3f  spearman %.3f"
          % (np.corrcoef(gm, cs)[0, 1], sp(gm, cs)))
    print("   comsol sr  pct[5,50,95] = %s" % np.round(np.percentile(cs, [5, 50, 95]), 3))
    print("   repo gamma pct[5,50,95] = %s" % np.round(np.percentile(gm, [5, 50, 95]), 3))
    print("[grad]   repo dshear_ds [1/(s*m)] vs comsol dsrx [1/(s*cm)]: pearson %.3f  spearman %.3f"
          % (np.corrcoef(dm, cd)[0, 1], sp(dm, cd)))
    print("   comsol dsrx pct[5,50,95] = %s" % np.round(np.percentile(cd, [5, 50, 95]), 2))
    print("   repo dshear pct[5,50,95] = %s" % np.round(np.percentile(dm, [5, 50, 95]), 2))

    print("\n[labels] comsol Mat_final>=2e7 rate %.3f | pack Mat_final>=2e7 rate %.3f | agree %.3f"
          % ((cmf >= MAT_CRIT).mean(), (pmf >= MAT_CRIT).mean(),
             ((cmf >= MAT_CRIT) == (pmf >= MAT_CRIT)).mean()))

    gt = cmf >= MAT_CRIT
    print("\n[gate as classifier on the MATCHED set] base rate %.3f" % gt.mean())
    for nm, pred in (
        ("COMSOL sr<lss", cs < LSS),
        ("COMSOL dsrx<sgt", cd < SGT_CGS),
        ("COMSOL either", (cs < LSS) | (cd < SGT_CGS)),
        ("repo gamma<lss", gm < LSS),
        ("repo dshear<sgt_si", dm < SGT_SI),
        ("repo either", (gm < LSS) | (dm < SGT_SI)),
    ):
        p, r, f1 = prf(pred, gt)
        print("   %-20s open %.3f  prec %.3f  rec %.3f  F1 %.3f" % (nm, pred.mean(), p, r, f1))
    print("\n[AUC vs same GT] comsol -sr %.3f  comsol -dsrx %.3f | repo -gamma %.3f  repo -dshear %.3f"
          % (auc(-cs[gt], -cs[~gt]), auc(-cd[gt], -cd[~gt]),
             auc(-gm[gt], -gm[~gt]), auc(-dm[gt], -dm[~gt])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
