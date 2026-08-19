"""Re-derive the surface ODE from COMSOL's own export. Trust nothing in the docs.

Everything downstream rests on an ODE nobody has actually recovered: 2.1 recorded that
``d(Mat,t)/J0_Mat`` has median 146 and ``d(Mas,t)/J0_Mas`` median 25.2 -- two different
constants -- and then declared it harmless because the mask saturates.  For TIMING it is
not harmless: the rate is exactly what sets onset.

The export carries every state variable, both exported fluxes, and all three time
derivatives at 876 wall nodes x 201 timesteps, so the ODE is recoverable rather than
guessable.  The sharp test is the DIFFERENCE:

    R_Mat - R_Mas = gate * (Mas/Minf) * k_aa * ap        (the autocatalytic term alone)

because the fresh-deposition terms cancel exactly.  If that fits with a single clean
constant, the law is confirmed and the constant is the real Damkohler.

Sections:
  A  identity checks (is M == Mas? is Sat what the docs claim?)
  B  the autocatalytic term, isolated by differencing
  C  what actually sets ONSET ORDER among identically-gated nodes
  D  locality: is the surface ODE local, or does it need neighbours?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MINF, K_RS, K_AS, K_AA = 7.0e6, 3.7e-3, 4.5e-2, 4.5e-2
L, GM, LSS, SGT, DA, CRIT = 7.5e-2, 150.0, 25.0, -750.0, 1.0e-4, 2.0e7


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    p = Path("outputs/comsol_p007_wall.npz")
    if not p.exists():
        print("[!] run scripts/parse_comsol_wall_export.py first")
        return 1
    d = np.load(p)
    t = d["t"]
    M, Mas, Mat, Sat = d["M"], d["Mas"], d["Mat"], d["Sat"]
    ap, rp, sr, dsrx = d["ap"], d["rp"], d["sr"], d["dsrx"]
    dMt, dMast, dMatt = d["dMt"], d["dMast"], d["dMatt"]
    J0_M, J0_Mas, J0_Mat = d["J0_M"], d["J0_Mas"], d["J0_Mat"]
    gate = ((dsrx < SGT) * (L / GM) * np.abs(dsrx) + (sr < LSS)).astype(float)
    T, N = Mat.shape

    print("=" * 78)
    print("A. IDENTITY CHECKS")
    print("=" * 78)
    print("  M == Mas ?                       max|M-Mas|/max(Mas) = %.3e"
          % (np.abs(M - Mas).max() / max(Mas.max(), 1e-30)))
    print("  d(M,t) == d(Mas,t) ?             max rel diff        = %.3e"
          % (np.abs(dMt - dMast).max() / max(np.abs(dMast).max(), 1e-30)))
    print("  J0_M == J0_Mas ?                 max rel diff        = %.3e"
          % (np.abs(J0_M - J0_Mas).max() / max(np.abs(J0_Mas).max(), 1e-30)))
    print("  Sat == 1 - Mas/Minf ?            max abs err         = %.3e"
          % np.abs(Sat - (1 - Mas / MINF)).max())
    print("  Sat == 1 - (M+Mas+Mat)/Minf ?    max abs err         = %.3e"
          % np.abs(Sat - (1 - (M + Mas + Mat) / MINF)).max())
    print("  Mat >= Mas everywhere ?          frac violating      = %.4f"
          % float((Mat < Mas - 1e-6).mean()))

    print("\n" + "=" * 78)
    print("B. THE AUTOCATALYTIC TERM, ISOLATED BY DIFFERENCING")
    print("=" * 78)
    lhs = dMatt - dMast                       # fresh-deposition terms cancel exactly
    auto = gate * (Mas / MINF) * K_AA * ap
    m = np.isfinite(lhs) & np.isfinite(auto)
    c = float((auto[m] * lhs[m]).sum() / max((auto[m] * auto[m]).sum(), 1e-30))
    print("  d(Mat,t) - d(Mas,t)  vs  gate*(Mas/Minf)*k_aa*ap")
    print("     coefficient %.6g   ( / Da = %.4f )   R2 = %.4f" % (c, c / DA, r2(lhs[m], c * auto[m])))
    # and the fresh-deposition equation on its own
    dep = gate * Sat * (K_RS * rp + K_AS * ap)
    c2 = float((dep[m] * dMast[m]).sum() / max((dep[m] * dep[m]).sum(), 1e-30))
    print("  d(Mas,t)             vs  gate*Sat*(k_rs*rp + k_as*ap)")
    print("     coefficient %.6g   ( / Da = %.4f )   R2 = %.4f" % (c2, c2 / DA, r2(dMast[m], c2 * dep[m])))
    print("  ratio of the two recovered constants: %.4f" % (c / max(c2, 1e-30))
          + "   (1.0 would mean ONE Damkohler governs both)")
    # exported flux vs derivative, for the record
    q = J0_Mat > 0
    print("  exported J0_Mat / d(Mat,t) median ratio: %.4f"
          % float(np.median(J0_Mat[q] / np.maximum(dMatt[q], 1e-30))))

    print("\n" + "=" * 78)
    print("C. WHAT SETS ONSET ORDER AMONG IDENTICALLY-GATED NODES")
    print("=" * 78)
    hot = Mat >= CRIT
    onset = np.where(hot.any(0), hot.argmax(0), -1)
    low_only = (sr[0] < LSS) & ~(dsrx[0] < SGT)          # gate == 1 exactly, at t=0
    sep_any = (dsrx[0] < SGT)
    for nm, sel in (("low-shear ONLY (gate==1)", low_only), ("separation-gated", sep_any)):
        s = sel & (onset >= 0)
        if s.sum() < 5:
            print("  %-26s n=%d  (too few)" % (nm, int(s.sum())))
            continue
        ot = t[onset[s]]
        print("  %-26s n=%3d   onset pct[10,50,90] = %s   spread %.3f of horizon"
              % (nm, int(s.sum()), np.round(np.percentile(ot, [10, 50, 90]), 0),
                 (ot.max() - ot.min()) / t[-1]))
    s = low_only & (onset >= 0)
    if s.sum() > 5:
        print("\n  Among gate==1 nodes the law is IDENTICAL, so any spread must come from")
        print("  a state variable. Rank-correlation of onset with each candidate:")
        cands = {
            "ap at t=0": ap[0][s],
            "ap at onset": np.array([ap[onset[i], i] for i in np.where(s)[0]]),
            "ap at t_final": ap[-1][s],
            "rp at t=0": rp[0][s],
            "sr at t=0": sr[0][s],
            "|dsrx| at t=0": np.abs(dsrx[0])[s],
            "Mas at t=25%": Mas[T // 4][s],
            "gate": gate[0][s],
        }
        ot = t[onset[s]]
        for k, v in sorted(cands.items(), key=lambda z: -abs(spear(z[1], ot))):
            print("     %-18s spearman %+.3f   (CV %.4f)"
                  % (k, spear(v, ot), float(np.std(v) / max(abs(np.mean(v)), 1e-30))))

    print("\n" + "=" * 78)
    print("D. IS THE SURFACE ODE LOCAL?")
    print("=" * 78)
    from scipy.spatial import cKDTree
    xy = np.stack([d["x"][0], d["y"][0]], 1)
    _, nbr = cKDTree(xy).query(xy, k=7)
    resid = lhs - c * auto                     # residual of the best local fit
    k_mid = T // 2
    rn = resid[k_mid][nbr[:, 1:]].mean(1)
    good = np.isfinite(resid[k_mid]) & np.isfinite(rn)
    print("  residual of the local autocat fit vs its 6-neighbour mean:")
    print("     pearson %.4f  spearman %.4f   (near 0 => the ODE is LOCAL)"
          % (float(np.corrcoef(resid[k_mid][good], rn[good])[0, 1]), spear(resid[k_mid][good], rn[good])))
    apn = ap[k_mid][nbr[:, 1:]].mean(1)
    print("  ap vs its 6-neighbour mean: pearson %.4f  (near 1 => ap is a smooth FIELD,"
          % float(np.corrcoef(ap[k_mid], apn)[0, 1]))
    print("     i.e. set by transport, not by each node independently)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
