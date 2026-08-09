"""Find the ACTUAL RHS of COMSOL's Mat surface ODE from the p007 wall export.

Step 0 established the exported ``J0_Mat`` is NOT the RHS (median d(Mat,t)/J0_Mat = 146).
This searches a dictionary of physically-motivated candidate terms and reports which
combination reproduces ``d(Mat,t)``.
"""
from __future__ import annotations

import numpy as np

MINF = 7.0e6
K_RS, K_AS, K_AA = 3.7e-3, 4.5e-2, 4.5e-2
L, GAMMA_M, LSS, SGT, DA = 7.5e-2, 150.0, 25.0, -750.0, 1.0e-4
MAT_CRIT = 2.0e7


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main() -> int:
    d = np.load("outputs/comsol_p007_wall.npz")
    t = d["t"]
    Mat, Mas, M, Sat = d["Mat"], d["Mas"], d["M"], d["Sat"]
    sr, dsrx, rp, ap, s2t = d["sr"], d["dsrx"], d["rp"], d["ap"], d["step2t"]
    PT, th, mu1 = d["PT"], d["th"], d["mu1"]
    dMatt, dMast = d["dMatt"], d["dMast"]
    T, N = Mat.shape

    gate = ((dsrx < SGT) * (L / GAMMA_M) * np.abs(dsrx) + (sr < LSS)).astype(np.float64)

    # ---- 1. effective per-node growth rate lambda = (dMat/dt)/Mat ------------
    big = Mat > 1e5
    lam = np.full_like(Mat, np.nan)
    lam[big] = dMatt[big] / Mat[big]
    print("[lambda] (dMat/dt)/Mat where Mat>1e5   n=%d" % big.sum())
    print("   pct[5,25,50,75,95] = %s  [1/s]"
          % np.format_float_scientific(0) if False else np.round(
              np.nanpercentile(lam[big], [5, 25, 50, 75, 95]), 8))
    print("   1/median = %.1f s   (horizon %.0f s)" % (1.0 / np.nanmedian(lam[big]), t[-1]))
    print("   candidate Da*k_aa*ap/Minf (ungated) median = %.3e"
          % np.median(DA * K_AA * ap / MINF))
    print("   candidate Da*k_aa*ap/Minf * gate  median(gate>0) = %.3e"
          % np.median((DA * K_AA * ap / MINF * gate)[gate > 0]))

    # ---- 2. dictionary regression -------------------------------------------
    terms = {
        "gate*Sat*(krs*rp+kas*ap)": gate * Sat * (K_RS * rp + K_AS * ap),
        "gate*(Mas/Minf)*kaa*ap": gate * (Mas / MINF) * K_AA * ap,
        "gate*(Mat/Minf)*kaa*ap": gate * (Mat / MINF) * K_AA * ap,
        "(Mat/Minf)*kaa*ap": (Mat / MINF) * K_AA * ap,
        "(Mas/Minf)*kaa*ap": (Mas / MINF) * K_AA * ap,
        "Sat*(krs*rp+kas*ap)": Sat * (K_RS * rp + K_AS * ap),
        "Mat*PT": Mat * PT,
        "Mat*th": Mat * th,
        "Mat": Mat,
        "Mas": Mas,
    }
    y = dMatt.reshape(-1)
    keep = np.isfinite(y)
    print("\n[single-term R2 for d(Mat,t)]")
    for k, v in terms.items():
        x = (v * s2t).reshape(-1)[keep]
        c = (x @ y[keep]) / max((x @ x), 1e-30)
        print("   %-26s c=%-12.5g R2=%7.4f" % (k, c, r2(y[keep], c * x)))

    # ---- 3. best pair --------------------------------------------------------
    keys = list(terms)
    best = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            A = np.stack([(terms[keys[i]] * s2t).reshape(-1)[keep],
                          (terms[keys[j]] * s2t).reshape(-1)[keep]], axis=1)
            c, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
            best.append((r2(y[keep], A @ c), keys[i], keys[j], c))
    best.sort(reverse=True, key=lambda z: z[0])
    print("\n[best 5 pairs]")
    for s, a, b, c in best[:5]:
        print("   R2=%.4f  %.5g*%s + %.5g*%s" % (s, c[0], a, c[1], b))

    # ---- 4. same for Mas -----------------------------------------------------
    ym = dMast.reshape(-1)
    print("\n[single-term R2 for d(Mas,t)]")
    for k in ("gate*Sat*(krs*rp+kas*ap)", "Sat*(krs*rp+kas*ap)",
              "gate*(Mas/Minf)*kaa*ap", "Mas"):
        x = (terms[k] * s2t).reshape(-1)[keep]
        c = (x @ ym[keep]) / max((x @ x), 1e-30)
        print("   %-26s c=%-12.5g R2=%7.4f" % (k, c, r2(ym[keep], c * x)))

    # ---- 5. where does growth happen relative to the gates? -----------------
    grow = dMatt > 0
    print("\n[gate audit]  cells with d(Mat,t)>0: %d" % grow.sum())
    print("   low-shear gate open   : %.3f (all cells %.3f)"
          % ((sr < LSS)[grow].mean(), (sr < LSS).mean()))
    print("   separation gate open  : %.3f (all cells %.3f)"
          % ((dsrx < SGT)[grow].mean(), (dsrx < SGT).mean()))
    print("   either open           : %.3f (all cells %.3f)"
          % ((gate > 0)[grow].mean(), (gate > 0).mean()))
    # weighted by growth magnitude
    w = dMatt[grow]
    print("   MASS-weighted either-open: %.3f" % ((gate[grow] > 0) * w).sum() / 1 if False else
          float(((gate[grow] > 0) * w).sum() / w.sum()))

    # ---- 6. does the t=0 gate predict the final committed set? --------------
    fin = Mat[-1] >= MAT_CRIT
    g0 = gate[0] > 0
    inter = (fin & g0).sum()
    print("\n[t=0 gate as a classifier of final commit]  base rate %.3f" % fin.mean())
    print("   t=0 gate open frac %.3f  prec %.3f  rec %.3f"
          % (g0.mean(), inter / max(g0.sum(), 1), inter / max(fin.sum(), 1)))
    gt0_ever = (gate > 0).any(axis=0)
    print("   gate EVER open frac %.3f  rec %.3f"
          % (gt0_ever.mean(), (fin & gt0_ever).sum() / max(fin.sum(), 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
