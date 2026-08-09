"""STEP 0 (PHASE3_HANDOFF 1.5) on COMSOL's OWN export -- no repo plumbing in the way.

Answers, on patient007's 876 wall nodes x 201 timesteps, in COMSOL-native CGS:

  A. Is the exported ``d(Mat,t)`` equal to the exported ``J0_Mat``?  (is the law THE ODE?)
  B. Does integrating ``J0_Mat`` reproduce ``Mat(t)``?               (integration check)
  C. Does the law recomputed from state reproduce ``J0_Mat``?        (already pinned by tests)
  D. **The real test**: freeze the gates at t=0, integrate the law forward with the
     Mas/Mat feedback, and compare the resulting committed-node count and total mass
     against GT.
"""
from __future__ import annotations

import numpy as np

MINF = 7.0e6        # plt/cm^2   (BiochemConfig.Minf 7e10 plt/m^2 -> CGS)
K_RS = 3.7e-3       # cm/s
K_AS = 4.5e-2       # cm/s
K_AA = 4.5e-2       # cm/s
L = 7.5e-2          # cm
GAMMA_M = 150.0
LSS = 25.0
SGT = -750.0        # 1/(s*cm)
DA = 1.0e-4
MAT_CRIT = 2.0e7    # plt/cm^2  (viscosity_mat_crit)


def law(sat, sr, dsrx, rp, ap, mas, step2t):
    common = sat * (K_RS * rp + K_AS * ap) + (mas / MINF) * K_AA * ap
    g_sep = (dsrx < SGT).astype(np.float64)
    g_low = (sr < LSS).astype(np.float64)
    return DA * (g_sep * (L / GAMMA_M) * np.abs(dsrx) * common + g_low * common) * step2t


def rel(a, b):
    return float(np.abs(a - b).sum() / (np.abs(b).sum() + 1e-30))


def main() -> int:
    d = np.load("outputs/comsol_p007_wall.npz")
    t = d["t"]
    Mat, Mas, M, Sat = d["Mat"], d["Mas"], d["M"], d["Sat"]
    sr, dsrx, rp, ap, s2t = d["sr"], d["dsrx"], d["rp"], d["ap"], d["step2t"]
    J0 = d["J0_Mat"]
    dMatt = d["dMatt"]
    T, N = Mat.shape
    print(f"grid: T={T} N={N}  t=[{t[0]},{t[-1]}]  dt={t[1]-t[0]}\n")

    # ---- Sat definition ------------------------------------------------------
    print("[Sat] 1-Mas/Minf  rel-err %.3e | 1-(M+Mas+Mat)/Minf rel %.3e"
          % (rel(Sat, 1 - Mas / MINF), rel(Sat, 1 - (M + Mas + Mat) / MINF)))

    # ---- C: law reproduces exported J0_Mat ----------------------------------
    J0_rec = law(Sat, sr, dsrx, rp, ap, Mas, s2t)
    print("[C] recomputed J0_Mat vs export : rel-err %.3e" % rel(J0_rec, J0))

    # ---- A: is d(Mat,t) == J0_Mat? ------------------------------------------
    m = J0 > 0
    ratio = dMatt[m] / J0[m]
    print("[A] d(Mat,t)/J0_Mat  n=%d  pct[1,25,50,75,99]=%s"
          % (m.sum(), np.round(np.percentile(ratio, [1, 25, 50, 75, 99]), 2)))
    print("    corr(d(Mat,t), J0_Mat) = %.4f" % np.corrcoef(dMatt[m], J0[m])[0, 1])
    # also: is d(Mas,t) == J0_Mas?
    mm = d["J0_Mas"] > 0
    print("    d(Mas,t)/J0_Mas median = %.3f   d(M,t)/J0_M median = %.3f"
          % (np.median(d["dMast"][mm] / d["J0_Mas"][mm]),
             np.median(d["dMt"][d["J0_M"] > 0] / d["J0_M"][d["J0_M"] > 0])))

    # ---- B: integrate the EXPORTED J0_Mat ------------------------------------
    dt = np.diff(t)[:, None]
    cum = np.concatenate([np.zeros((1, N)), np.cumsum(0.5 * (J0[1:] + J0[:-1]) * dt, axis=0)])
    print("\n[B] integral of exported J0_Mat vs GT Mat, at t_final:")
    print("    pred total mass %.4e   GT %.4e   ratio %.4f"
          % (cum[-1].sum(), Mat[-1].sum(), cum[-1].sum() / Mat[-1].sum()))
    cum2 = np.concatenate([np.zeros((1, N)), np.cumsum(0.5 * (dMatt[1:] + dMatt[:-1]) * dt, axis=0)])
    print("    integral of exported d(Mat,t): %.4e  ratio %.4f"
          % (cum2[-1].sum(), cum2[-1].sum() / Mat[-1].sum()))

    # ---- D: forward integration, gates frozen at t=0 -------------------------
    for label, freeze in (("GT gates every step", False), ("gates FROZEN at t=0", True)):
        for scale in (1.0,):
            mas = Mas[0].copy()
            mat = Mat[0].copy()
            mm_ = M[0].copy()
            for i in range(T - 1):
                k = 0 if freeze else i
                sat = 1.0 - mas / MINF
                # J0_M / J0_Mas: fresh deposition only (no autocat term)
                common_dep = sat * (K_RS * rp[k] + K_AS * ap[k])
                autocat = (mas / MINF) * K_AA * ap[k]
                g_sep = (dsrx[k] < SGT).astype(np.float64)
                g_low = (sr[k] < LSS).astype(np.float64)
                gate = g_sep * (L / GAMMA_M) * np.abs(dsrx[k]) + g_low
                j_mas = DA * gate * common_dep * s2t[k] * scale
                j_mat = DA * gate * (common_dep + autocat) * s2t[k] * scale
                h = t[i + 1] - t[i]
                mas = mas + h * j_mas
                mm_ = mm_ + h * j_mas
                mat = mat + h * j_mat
            gt_c = (Mat[-1] >= MAT_CRIT).sum()
            pr_c = (mat >= MAT_CRIT).sum()
            print("\n[D] %-22s mass pred %.4e  GT %.4e  ratio %.4f"
                  % (label, mat.sum(), Mat[-1].sum(), mat.sum() / Mat[-1].sum()))
            print("    committed nodes pred %d  GT %d   (crit=%.1e)" % (pr_c, gt_c, MAT_CRIT))
            inter = ((mat >= MAT_CRIT) & (Mat[-1] >= MAT_CRIT)).sum()
            prec = inter / max(pr_c, 1)
            recl = inter / max(gt_c, 1)
            f1 = 2 * prec * recl / max(prec + recl, 1e-9)
            print("    prec %.3f  rec %.3f  F1 %.3f   corr(log1p) %.3f"
                  % (prec, recl, f1,
                     np.corrcoef(np.log1p(np.clip(mat, 0, None)),
                                 np.log1p(np.clip(Mat[-1], 0, None)))[0, 1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
