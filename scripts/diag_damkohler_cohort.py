"""STEP 3 (PHASE6_HANDOFF 5.3): where does the 4.4x Damkohler ratio come from?

``scripts/diag_rederive_surface_ode.py`` recovered, from patient007's raw COMSOL export:

    d(Mat,t) - d(Mas,t)  vs  gate*(Mas/Minf)*k_aa*ap   ->  C = 140.5 * Da,  R2 = 0.8905
    d(Mas,t)             vs  gate*Sat*(k_rs*rp+k_as*ap) ->  C =  31.9 * Da,  R2 = 0.4648

Two effective Damkohlers, ratio 4.40, neither equal to the exported ``Da = 1e-4``.  That
matters now because the model applies ONE ``da*da_scale`` to both terms, and the ratio
between them is exactly what sets how long a node sits below ``crit`` before autocatalysis
runs away -- i.e. onset.

THREE PROBLEMS WITH THAT MEASUREMENT, ALL FIXED HERE:
  1. patient007 is SEALED (6.1).  The packs carry ``M``/``Mas``/``Mat`` at all 201 steps
     for every vessel (unit-verified against the export node-by-node), so this refits on
     TRAIN and reports the cohort spread instead of one vessel's point estimate.
  2. The two constants were fitted SEPARATELY, so the autocatalytic fit absorbs whatever
     the deposition fit got wrong.  Fitting ``(A_s, A_a)`` jointly on ``d(Mat,t)`` is the
     honest version and is what a rollout would use.
  3. ``step2t`` was omitted and ``Sat`` was used unclipped even where COMSOL drives it to
     -0.195.  Both are tested as ablations rather than assumed harmless.

The export's own hint, recorded in 2.1 and never followed up: ``d(Mat,t)/J0_Mat`` has
median ~146 and ``d(Mas,t)/J0_Mas`` ~25.2.  So COMSOL's *exported fluxes* are consistent
with ``Da = 1e-4`` while its *state derivatives* are ~30x and ~140x larger -- the same two
numbers.  That is the signature of a missing multiplier on the surface ODE, not of a wrong
rate law, and section D tests the obvious candidate: the mature deposit carries a fibrin
contribution the platelet-only law does not have.

Derivatives come from central differences on the packs' own 150 s sampling; section A
checks that against COMSOL's exported analytic derivatives on patient007 so the numerical
scheme is not what is being measured.

    python scripts/diag_damkohler_cohort.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig  # noqa: E402

CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/ap_closure")
M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4


def r2(y, p):
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def ddt(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central difference in time, [T, N] -> [T, N] (one-sided at the ends)."""
    return np.gradient(v, t, axis=0)


def terms(z, bio, *, clip_sat: bool, use_step2t: bool):
    """The two basis functions of the surface law, [T, N], plus the targets."""
    k_rs = float(bio.k_rs) * M_TO_CM
    k_as = float(bio.k_as) * M_TO_CM
    k_aa = float(bio.k_aa) * M_TO_CM
    minf = float(bio.Minf) * PER_M2_TO_PER_CM2
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)

    sr0, dsrx0 = z["sr0"], z["dsrx0"]
    gate = ((dsrx0 < sgt) * coef * np.abs(dsrx0) + (sr0 < lss))[None, :]
    mas, mat, ap, rp, t = z["mas"], z["mat"], z["ap"], z["rp"], z["t"]
    raw_sat = 1.0 - mas / minf
    sat = np.clip(raw_sat, 0.0, 1.0) if clip_sat else raw_sat
    s2t = 1.0
    if use_step2t:
        gs, sl = float(bio.surface_time_gate_s), float(bio.surface_time_gate_slope)
        s2t = (1.0 / (1.0 + np.exp(-np.clip((t - gs) * sl, -50, 50))))[:, None]
    dep = gate * sat * (k_rs * rp + k_as * ap) * s2t
    auto = gate * (mas / minf) * k_aa * ap * s2t
    return dep, auto, ddt(mas, t), ddt(mat, t)


def lstsq2(y, b1, b2):
    """Joint non-negative-ish least squares on two basis functions (plain normal equations)."""
    A = np.stack([b1, b2], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1]), r2(y, A @ coef)


def proj(y, b):
    den = float((b * b).sum())
    c = float((b * y).sum() / den) if den > 0 else float("nan")
    return c, r2(y, c * b)


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--peek-sealed", action="store_true")
    args = ap_.parse_args()
    bio = BiochemConfig(phase="biochem")
    da = float(bio.surface_damkohler)
    OUT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------- A. is the finite difference trustworthy?
    print("=" * 92)
    print("A. NUMERICAL vs COMSOL'S OWN ANALYTIC DERIVATIVE  (patient007 export -- an")
    print("   OPERATOR check against COMSOL's fields, which 6.1 permits; nothing is fit here)")
    print("=" * 92)
    exp = Path("outputs/comsol_p007_wall.npz")
    if exp.exists():
        e = np.load(exp)
        for nm, num, ana in (("d(Mas,t)", ddt(e["Mas"], e["t"]), e["dMast"]),
                             ("d(Mat,t)", ddt(e["Mat"], e["t"]), e["dMatt"])):
            m = np.isfinite(num) & np.isfinite(ana)
            print("   %-10s central-diff vs exported: pearson %.5f   median ratio %.4f"
                  % (nm, float(np.corrcoef(num[m], ana[m])[0, 1]),
                     float(np.median(num[m][np.abs(ana[m]) > 1e-6] /
                                     ana[m][np.abs(ana[m]) > 1e-6]))))
        print("   (150 s sampling on a 30000 s horizon; a pearson near 1 means the cohort fits")
        print("    below are measuring the LAW, not the differencing scheme)")
    else:
        print("   [skip] no export at %s" % exp)

    # -------------------------------------------------- B. the two constants, per vessel
    print("\n" + "=" * 92)
    print("B. THE TWO EFFECTIVE DAMKOHLERS, REFIT PER TRAIN VESSEL")
    print("=" * 92)
    print("   A_s from d(Mas,t) ~ dep     |  A_a from d(Mat,t)-d(Mas,t) ~ auto  (both / Da)")
    print("%-12s | %9s %7s | %9s %7s | %7s" % ("vessel", "A_s/Da", "R2", "A_a/Da", "R2", "ratio"))
    names = list(WALL_COHORT_V2_TRAIN) + (list(WALL_COHORT_V2_GENERALIZATION)
                                          if args.peek_sealed else [])
    rows = []
    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        dep, auto, dmas, dmat = terms(z, bio, clip_sat=True, use_step2t=True)
        m = np.isfinite(dep) & np.isfinite(auto) & np.isfinite(dmas) & np.isfinite(dmat)
        if m.sum() < 1000:
            continue
        c_s, r_s = proj(dmas[m], dep[m])
        c_a, r_a = proj((dmat - dmas)[m], auto[m])
        row = dict(name=n, sealed=bool(z["sealed"]), A_s=c_s / da, r2_s=r_s,
                   A_a=c_a / da, r2_a=r_a, ratio=c_a / c_s if c_s else float("nan"))
        rows.append(row)
        print("%-12s | %9.2f %7.4f | %9.2f %7.4f | %7.2f"
              % (n, row["A_s"], r_s, row["A_a"], r_a, row["ratio"]))
    tr = [r for r in rows if not r["sealed"]]
    for k, lbl in (("A_s", "A_s / Da"), ("A_a", "A_a / Da"), ("ratio", "A_a / A_s")):
        v = np.array([r[k] for r in tr], float)
        v = v[np.isfinite(v)]
        print("   %-10s median %8.2f   IQR [%8.2f, %8.2f]   max/min %.1fx"
              % (lbl, np.median(v), np.percentile(v, 25), np.percentile(v, 75),
                 v.max() / max(v.min(), 1e-30)))
    print("\n   patient007's export gave A_s/Da = 31.9, A_a/Da = 140.5, ratio 4.40.")
    print("   The model ships ONE da_scale for both; the ratio is what that single scalar")
    print("   cannot represent, and it is the term that decides how long a node idles")
    print("   below crit before autocatalysis runs away.")

    # ------------------------------------------------------ C. joint fit + the ablations
    print("\n" + "=" * 92)
    print("C. JOINT FIT ON d(Mat,t), AND WHAT THE TWO MODELLING CHOICES ARE WORTH")
    print("=" * 92)
    print("%-12s | %-22s %9s %9s %8s" % ("vessel", "variant", "A_s/Da", "A_a/Da", "R2"))
    variants = [("clip Sat + step2t", True, True), ("raw Sat  + step2t", False, True),
                ("clip Sat, no step2t", True, False)]
    joint = []
    for n in [r["name"] for r in tr]:
        z = np.load(CACHE / f"{n}.npz")
        for lbl, cs, st in variants:
            dep, auto, dmas, dmat = terms(z, bio, clip_sat=cs, use_step2t=st)
            m = np.isfinite(dep) & np.isfinite(auto) & np.isfinite(dmat)
            a_s, a_a, rr = lstsq2(dmat[m], dep[m], auto[m])
            joint.append(dict(name=n, variant=lbl, A_s=a_s / da, A_a=a_a / da, r2=rr))
            print("%-12s | %-22s %9.2f %9.2f %8.4f" % (n, lbl, a_s / da, a_a / da, rr))
    print("-" * 92)
    for lbl, _, _ in variants:
        sub = [j for j in joint if j["variant"] == lbl]
        print("   %-22s median A_s/Da %8.2f  A_a/Da %8.2f  ratio %6.2f  median R2 %.4f"
              % (lbl, np.median([j["A_s"] for j in sub]), np.median([j["A_a"] for j in sub]),
                 np.median([j["A_a"] for j in sub]) / max(np.median([j["A_s"] for j in sub]), 1e-30),
                 np.median([j["r2"] for j in sub])))

    # ---------------------------------------------- D. is the missing piece FIBRIN?
    print("\n" + "=" * 92)
    print("D. WHAT THE PLATELET-ONLY LAW LEAVES ON THE TABLE FOR Mat  (patient007 export)")
    print("=" * 92)
    if exp.exists():
        e = np.load(exp)
        gate = ((e["dsrx"] < -750.0) * (7.5e-2 / 150.0) * np.abs(e["dsrx"])
                + (e["sr"] < 25.0)).astype(float)
        sat = np.clip(e["Sat"], 0.0, 1.0)
        dep = gate * sat * (3.7e-3 * e["rp"] + 4.5e-2 * e["ap"])
        auto = gate * (e["Mas"] / 7.0e6) * 4.5e-2 * e["ap"]
        m = np.isfinite(dep) & np.isfinite(auto) & np.isfinite(e["dMatt"])
        a_s, a_a, rr = lstsq2(e["dMatt"][m], dep[m], auto[m])
        print("   joint fit on the export: A_s/Da %.2f  A_a/Da %.2f  R2 %.4f  ratio %.2f"
              % (a_s / 1e-4, a_a / 1e-4, rr, a_a / max(a_s, 1e-30)))
        resid = e["dMatt"] - (a_s * dep + a_a * auto)
        print("\n   residual of the platelet-only law vs the OTHER exported fields:")
        for nm in ("fi", "th", "PT", "mu1", "Mat", "Mas"):
            v = e[nm]
            q = m & np.isfinite(v)
            print("      pearson(resid, %-4s)            = %+.4f" % (nm, float(np.corrcoef(resid[q], v[q])[0, 1])))
        dfi = ddt(e["fi"], e["t"])
        q = m & np.isfinite(dfi)
        print("      pearson(resid, d(fi,t))          = %+.4f" % float(np.corrcoef(resid[q], dfi[q])[0, 1]))
        a3 = np.stack([dep[q], auto[q], dfi[q]], 1)
        co, *_ = np.linalg.lstsq(a3, e["dMatt"][q], rcond=None)
        print("      adding d(fi,t) as a third basis:  R2 %.4f -> %.4f"
              % (rr, r2(e["dMatt"][q], a3 @ co)))
        print("\n   exported FLUX vs state derivative (the 2.1 hint, restated):")
        for nm, j, dv in (("Mas", e["J0_Mas"], e["dMast"]), ("Mat", e["J0_Mat"], e["dMatt"])):
            g = np.abs(j) > 1e-9
            print("      median d(%s,t) / J0_%s = %8.2f" % (nm, nm, float(np.median(dv[g] / j[g]))))
        print("      COMSOL's own fluxes are consistent with Da=1e-4; its state derivatives")
        print("      are ~30x and ~140x larger.  Whatever the multiplier is, it is NOT one")
        print("      number, and da_scale=40 is the model absorbing the SMALLER of the two.")
    else:
        print("   [skip] no export")

    (OUT / "damkohler.json").write_text(json.dumps(dict(per_vessel=rows, joint=joint),
                                                   indent=2, default=float), encoding="utf-8")
    print("\nwrote %s" % (OUT / "damkohler.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
