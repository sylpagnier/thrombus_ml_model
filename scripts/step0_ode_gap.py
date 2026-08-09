"""What separates the exported J0_* fluxes from the exported d(*,t) time derivatives?

Step 0 found: recomputing ``J0_Mat`` from state is exact (1e-16), integrating the
exported ``d(Mat,t)`` reproduces ``Mat`` (0.992) -- but ``d(Mat,t)/J0_Mat`` has median
146 and ``d(Mas,t)/J0_Mas`` median 25.2.  Two different constants means it is not a
single unit slip.  Hypotheses tested here:

  H1  per-NODE geometric factor (mesh/surface-to-volume): ratio constant in time per node
  H2  global constant
  H3  the Mat ODE autocatalyses on **Mat**, not Mas: dMat/dt = c*gate*(dep + (Mat/Minf)*k_aa*ap)
"""
from __future__ import annotations

import numpy as np

MINF = 7.0e6
K_RS, K_AS, K_AA = 3.7e-3, 4.5e-2, 4.5e-2
L, GAMMA_M, LSS, SGT, DA = 7.5e-2, 150.0, 25.0, -750.0, 1.0e-4


def main() -> int:
    d = np.load("outputs/comsol_p007_wall.npz")
    Mat, Mas, Sat = d["Mat"], d["Mas"], d["Sat"]
    sr, dsrx, rp, ap, s2t = d["sr"], d["dsrx"], d["rp"], d["ap"], d["step2t"]
    dMatt, dMast, J0_Mat, J0_Mas = d["dMatt"], d["dMast"], d["J0_Mat"], d["J0_Mas"]
    T, N = Mat.shape

    gate = ((dsrx < SGT) * (L / GAMMA_M) * np.abs(dsrx) + (sr < LSS)).astype(np.float64)
    dep = Sat * (K_RS * rp + K_AS * ap)
    auto_mas = (Mas / MINF) * K_AA * ap
    auto_mat = (Mat / MINF) * K_AA * ap

    # ---- H1/H2 on Mas (no autocat ambiguity there) --------------------------
    m = J0_Mas > 0
    r = np.full_like(J0_Mas, np.nan)
    r[m] = dMast[m] / J0_Mas[m]
    per_node = np.nanmedian(r, axis=0)
    ok = np.isfinite(per_node)
    print("[Mas] ratio d(Mas,t)/J0_Mas")
    print("   global median %.4f" % np.nanmedian(r))
    print("   per-node median: pct[1,25,50,75,99] = %s"
          % np.round(np.percentile(per_node[ok], [1, 25, 50, 75, 99]), 3))
    # within-node time CV of the ratio
    cv = np.nanstd(r, axis=0) / np.abs(np.nanmedian(r, axis=0) + 1e-30)
    print("   within-node CV over time: median %.4f  (H1 wants ~0)" % np.nanmedian(cv[ok]))
    print("   across-node CV of medians: %.4f  (H2 wants ~0)"
          % (np.nanstd(per_node[ok]) / np.abs(np.nanmean(per_node[ok]))))

    # ---- H3 on Mat -----------------------------------------------------------
    c_mas = float(np.nanmedian(r))
    print("\n[Mat] with the SAME constant c=%.3f from Mas:" % c_mas)
    for name, rhs in (("dep+auto(Mas)", DA * gate * (dep + auto_mas) * s2t),
                      ("dep+auto(Mat)", DA * gate * (dep + auto_mat) * s2t)):
        pred = c_mas * rhs
        mm = dMatt > 0
        print("   %-14s  sum(pred)/sum(GT) = %.4f   corr = %.4f   medratio %.3f"
              % (name, pred[mm].sum() / dMatt[mm].sum(),
                 np.corrcoef(pred[mm], dMatt[mm])[0, 1],
                 np.median(dMatt[mm] / np.maximum(pred[mm], 1e-30))))

    # ---- least squares for the Mat ODE coefficients --------------------------
    mm = (dMatt != 0) & np.isfinite(dMatt)
    A = np.stack([(gate * dep * s2t)[mm], (gate * auto_mas * s2t)[mm],
                  (gate * auto_mat * s2t)[mm]], axis=1)
    coef, *_ = np.linalg.lstsq(A, dMatt[mm], rcond=None)
    pred = A @ coef
    print("\n[LSQ] dMat/dt ~ a*gate*dep + b*gate*auto_Mas + c*gate*auto_Mat")
    print("   a=%.4g  b=%.4g  c=%.4g   R2=%.4f"
          % (coef[0], coef[1], coef[2],
             1 - ((dMatt[mm] - pred) ** 2).sum() / ((dMatt[mm] - dMatt[mm].mean()) ** 2).sum()))
    # same for Mas
    Am = np.stack([(gate * dep * s2t)[mm], (gate * auto_mas * s2t)[mm]], axis=1)
    cm, *_ = np.linalg.lstsq(Am, dMast[mm], rcond=None)
    pm = Am @ cm
    print("   Mas: a=%.4g b=%.4g  R2=%.4f"
          % (cm[0], cm[1],
             1 - ((dMast[mm] - pm) ** 2).sum() / ((dMast[mm] - dMast[mm].mean()) ** 2).sum()))
    print("   (Da=%.1e ; a/Da = %.4g , c/Da = %.4g)" % (DA, coef[0] / DA, coef[2] / DA))

    # ---- is the gate itself right at late times? -----------------------------
    growing = dMatt > 0
    print("\n[gate coverage] of cells with d(Mat,t)>0, frac with gate>0: %.4f"
          % (gate[growing] > 0).mean())
    print("   frac of ALL cells with gate>0: %.4f" % (gate > 0).mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
