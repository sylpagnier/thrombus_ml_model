"""STEP 1 (PHASE6_HANDOFF 5.1): recalibrate the wall-AP closure on TRAIN vessels only.

The closure under test (2.1):

    ap_i(t) / ap0_i  =  1 / (1 + C * consumption_i(t) / sr_i^q)

    consumption = gate * (Sat + Mas/Minf) * k_as          [cm/s]

``C = 68, q = 1, R2 = 0.9041`` came from ``outputs/comsol_p007_wall.npz`` -- **patient007
is SEALED**, so that constant is not usable.  This refits on TRAIN only, using the packs'
own ``Mas``/``Mat``/``AP`` channels (unit-verified against the p007 export node-by-node in
``scripts/build_wall_species_cache.py``).

READ THE ALGEBRA BEFORE THE NUMBERS.  ``Sat = 1 - Mas/Minf`` and ``k_aa == k_as`` in the
config, so ``Sat + Mas/Minf == 1`` identically until ``Mas`` overshoots ``Minf``.  The
handoff's ``consumption`` is therefore **almost exactly static** -- with the gate and shear
frozen at t=0 the whole closure is a fixed spatial multiplier, not a depletion feedback.
That is not a defect (it is the quasi-steady wall Damkohler balance, and a static
shear-graded multiplier is exactly what breaks the flash: identically-gated nodes get
different rates), but it must be *stated*, and it means the honest questions are:

  * how much of the fitted R2 is spatial and how much is temporal?
  * does any genuinely time-varying kernel beat the static one?
  * is the ``sr`` exponent really 1 (a stirred-renewal balance) and not 1/3 (Leveque)?

Sections:
  A  kernel / exponent scan, pooled over TRAIN, fitted TWO ways (linearised and direct)
  B  per-vessel C -- the 9 kill criterion ("C varies wildly -> a global constant is wrong")
  C  variance decomposition: spatial vs temporal R2
  D  does C track any vessel descriptor (i.e. can it be conditioned rather than global)?
  E  residual locality -- the measurement that decides whether a GNN has a job (4)

    python scripts/fit_ap_closure.py
    python scripts/fit_ap_closure.py --arm oracle_sr   # sr(t) instead of frozen sr(0)
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
SR_FLOOR = 1e-3          # 1/s; below this the renewal term is meaningless anyway


# --------------------------------------------------------------------------- helpers

def r2(y: np.ndarray, p: np.ndarray) -> float:
    """R2, but NaN when the target is essentially constant.

    Several vessels deplete ``ap`` by <1% (patient024/025/028/036), so at a fixed time
    ``SS_tot`` is ~1e-9 and a raw R2 reads -1e10 -- a division artefact, not a model
    failure.  Those cases need RMSE, and are reported as such.
    """
    ss = float(((y - y.mean()) ** 2).sum())
    scale = float(np.mean(np.abs(y))) or 1.0
    if ss <= 0 or ss / max(len(y), 1) < (1e-4 * scale) ** 2:
        return float("nan")
    return float(1.0 - ((y - p) ** 2).sum() / ss)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def fit_C_direct(ratio: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """``ap_closure.fit_C`` plus its R2, so the two estimators can be printed side by side."""
    from src.core_physics.ap_closure import fit_C

    c = fit_C(ratio, x)
    return c, r2(ratio, 1.0 / (1.0 + c * x))


def fit_C_linear(ratio: np.ndarray, x: np.ndarray) -> float:
    y = 1.0 / np.clip(ratio, 1e-9, None) - 1.0
    den = float((x * x).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def kernels(v: dict, bio) -> dict[str, np.ndarray]:
    """Candidate wall-AP consumption kernels, all in [cm/s], shape [T, N].

    ``handoff`` and ``static`` are algebraically the same thing (see the module docstring
    of ``src.core_physics.ap_closure``); both are kept so the identity is *shown* rather
    than asserted.  ``unit_gate``/``clip1_gate`` drop the separation branch's
    rate-amplification factor out of the SINK while leaving it in the reaction.
    """
    k_as = float(bio.k_as) * M_TO_CM
    k_aa = float(bio.k_aa) * M_TO_CM
    minf = float(bio.Minf) * PER_M2_TO_PER_CM2
    gate = v["gate"][None, :]
    mas_f = v["mas"] / minf
    mat_f = v["mat"] / minf
    sat = np.clip(1.0 - mas_f, 0.0, 1.0)
    one = np.ones_like(mas_f)
    return {
        # the handoff's kernel.  Sat + Mas/Minf == 1 until Mas overshoots Minf.
        "handoff": gate * (sat + mas_f) * k_as,
        # its algebraic limit: a purely static, gate-weighted sink
        "static": gate * one * k_as,
        # sink independent of the gate's magnitude
        "unit_gate": (gate > 0) * one * k_as,
        "clip1_gate": np.minimum(gate, 1.0) * one * k_as,
        # fresh deposition only -- switches OFF as the surface saturates
        "sat_only": gate * sat * k_as,
        # autocatalysis carried by the MATURE deposit, which grows to ~56x Minf
        "sat_plus_mat": gate * (sat * k_as + mat_f * k_aa),
        # what the ODE literally spends: both surface sinks, mature-deposit-free
        "sat_plus_mas": gate * (sat * k_as + mas_f * k_aa),
    }


def load(names: list[str], bio, *, arm: str) -> dict[str, dict]:
    out = {}
    lss = float(bio.lss)
    sgt = float(bio.sgt) / M_TO_CM
    L_cm = float(bio.L_char) * M_TO_CM
    coef = L_cm / float(bio.gamma_m)
    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        sr0, dsrx0 = d["sr0"], d["dsrx0"]
        d["gate"] = (dsrx0 < sgt) * coef * np.abs(dsrx0) + (sr0 < lss)
        if arm == "oracle_sr":
            if "sr_t" not in d:
                continue
            d["sr_use"] = np.maximum(d["sr_t"], SR_FLOOR)
        else:
            d["sr_use"] = np.maximum(sr0, SR_FLOOR)[None, :]
        out[n] = d
    return out


def q_star_guess(scan: list[dict], kernel: str) -> float:
    """The exponent that won for this kernel in the pooled scan."""
    rows = [s for s in scan if s["kernel"] == kernel and np.isfinite(s["r2_direct"])]
    return max(rows, key=lambda s: s["r2_direct"])["q"] if rows else 1.0


def samples(d: dict, ker: np.ndarray, q: float, *, gated_only: bool,
            smoother=None, window: tuple[float, float] | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """(ratio, x) over all (t>0, wall node) samples, optionally within a horizon window."""
    ap0 = d["ap"][0]
    ratio = d["ap"] / np.maximum(ap0, 1e-30)[None, :]
    x = ker / np.power(d["sr_use"], q)
    if smoother is not None:
        x = np.stack([smoother(row) for row in np.atleast_2d(x)])
    if x.shape[0] == 1:
        x = np.broadcast_to(x, ratio.shape)
    sel = np.zeros_like(ratio, dtype=bool)
    if window is None:
        sel[1:] = True
    else:
        nt = ratio.shape[0]
        sel[max(1, int(window[0] * (nt - 1))): int(window[1] * (nt - 1)) + 1] = True
    sel &= np.isfinite(ratio) & np.isfinite(x) & (ratio > 0)
    if gated_only:
        sel &= (d["gate"] > 0)[None, :]
    return ratio[sel], x[sel]


# ------------------------------------------------------------------------------ main

def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--arm", default="frozen_sr", choices=["frozen_sr", "oracle_sr"])
    ap_.add_argument("--gated-only", action="store_true", default=True)
    ap_.add_argument("--all-nodes", dest="gated_only", action="store_false")
    ap_.add_argument("--peek-sealed", action="store_true",
                     help="also print SEALED numbers. Selection must NEVER read these.")
    args = ap_.parse_args()
    bio = BiochemConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)

    train = [n for n in WALL_COHORT_V2_TRAIN]
    cache = load(train, bio, arm=args.arm)
    if not cache:
        print("[!] no cache -- run scripts/build_wall_species_cache.py first")
        return 1
    print("arm=%s  gated_only=%s  TRAIN vessels with cache: %d" % (args.arm, args.gated_only, len(cache)))
    print("   %s" % " ".join(sorted(n[-3:] for n in cache)))
    assert not (set(cache) & set(WALL_COHORT_V2_GENERALIZATION)), "SEALED vessel leaked into the fit"

    # ------------------------------------------------------------ A. kernel/exponent scan
    print("\n" + "=" * 90)
    print("A. KERNEL x EXPONENT SCAN, pooled over TRAIN  (n_samples in the last column)")
    print("=" * 90)
    print("%-14s %5s | %10s %8s | %10s %8s | %9s"
          % ("kernel", "q", "C_direct", "R2", "C_linear", "R2", "n"))
    ker_cache = {n: kernels(v, bio) for n, v in cache.items()}
    q_grid = [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0, 1.5, 2.0]
    scan = []
    for kname in ker_cache[next(iter(ker_cache))]:
        for q in q_grid:
            rs, xs = [], []
            for n, v in cache.items():
                r_, x_ = samples(v, ker_cache[n][kname], q, gated_only=args.gated_only)
                rs.append(r_)
                xs.append(x_)
            ratio, x = np.concatenate(rs), np.concatenate(xs)
            c_d, r_d = fit_C_direct(ratio, x)
            c_l = fit_C_linear(ratio, x)
            r_l = r2(ratio, 1.0 / (1.0 + c_l * x))
            scan.append(dict(kernel=kname, q=q, C_direct=c_d, r2_direct=r_d,
                             C_linear=c_l, r2_linear=r_l, n=int(len(ratio))))
            print("%-14s %5.3f | %10.4g %8.4f | %10.4g %8.4f | %9d"
                  % (kname, q, c_d, r_d, c_l, r_l, len(ratio)))
    best = max(scan, key=lambda s: (s["r2_direct"] if np.isfinite(s["r2_direct"]) else -np.inf))
    print("\n  BEST: kernel=%s q=%.3f  C=%.4g  R2=%.4f"
          % (best["kernel"], best["q"], best["C_direct"], best["r2_direct"]))

    # the handoff's own configuration, for a like-for-like comparison with C=68 / R2=0.9041
    ref = [s for s in scan if s["kernel"] == "handoff" and abs(s["q"] - 1.0) < 1e-9][0]
    print("  handoff configuration (kernel=handoff, q=1): C=%.4g R2=%.4f"
          % (ref["C_direct"], ref["r2_direct"]))
    print("  (patient007-fitted value was C=68, R2=0.9041 -- SEALED, shown only as the"
          "\n   number this replaces, never as a target)")

    # a finer exponent scan around the winner, on the winning kernel
    print("\n  fine exponent scan on kernel=%s:" % best["kernel"])
    fine = []
    for q in np.round(np.arange(0.2, 1.81, 0.1), 3):
        rs, xs = [], []
        for n, v in cache.items():
            r_, x_ = samples(v, ker_cache[n][best["kernel"]], float(q), gated_only=args.gated_only)
            rs.append(r_)
            xs.append(x_)
        ratio, x = np.concatenate(rs), np.concatenate(xs)
        c_d, r_d = fit_C_direct(ratio, x)
        fine.append((float(q), c_d, r_d))
        print("     q=%.2f  C=%10.4g  R2=%.4f%s" % (q, c_d, r_d, "  <-- peak" if r_d == max(f[2] for f in fine) else ""))

    # ------------------------------------------------- A2. does SMOOTHING the sink help?
    # ap has neighbour correlation 0.993 and the local residual 0.926: the leftover error
    # is a transport field.  Mesh-averaging the sink is the zero-parameter stand-in, and
    # it is the baseline any GNN residual on C_i has to beat (9).
    print("\n" + "=" * 90)
    print("A2. MESH-SMOOTHED SINK (zero extra parameters) -- the GNN's baseline to beat")
    print("=" * 90)
    from src.core_physics.ap_closure import build_smoother
    smoothers = {}
    for n, v in cache.items():
        smoothers[n] = {h: build_smoother(v["wall_edges"], v["ap"].shape[1], h)
                        for h in (0, 1, 2, 4, 8)}
    print("%-14s %5s %6s | %10s %8s" % ("kernel", "q", "hops", "C", "R2"))
    smooth_scan = []
    for kname in ("static", "unit_gate", best["kernel"]):
        for h in (0, 1, 2, 4, 8):
            rs, xs = [], []
            for n, v in cache.items():
                r_, x_ = samples(v, ker_cache[n][kname], q_star_guess(scan, kname),
                                 gated_only=args.gated_only, smoother=smoothers[n][h])
                rs.append(r_)
                xs.append(x_)
            ratio, x = np.concatenate(rs), np.concatenate(xs)
            c_d, r_d = fit_C_direct(ratio, x)
            smooth_scan.append(dict(kernel=kname, hops=h, C=c_d, r2=r_d))
            print("%-14s %5.2f %6d | %10.4g %8.4f"
                  % (kname, q_star_guess(scan, kname), h, c_d, r_d))
    b2 = max(smooth_scan, key=lambda s: (s["r2"] if np.isfinite(s["r2"]) else -np.inf))
    print("\n  BEST smoothed: kernel=%s hops=%d C=%.4g R2=%.4f  (unsmoothed best was %.4f)"
          % (b2["kernel"], b2["hops"], b2["C"], b2["r2"], best["r2_direct"]))

    # ------------------------------------------- A3. WINDOW STABILITY picks the kernel
    # The selection criterion that matters, and it never looks at R2.  A correctly
    # specified kernel recovers the SAME C whichever slice of the horizon it is fitted on.
    # A misspecified one launders the drift into C -- which is the whole reason this repo
    # has carried C=68 (patient007, early-weighted) and C=250 (pooled TRAIN) and believed
    # them to be in conflict.  They are one measurement under two weightings.
    print("\n" + "=" * 90)
    print("A3. WINDOW STABILITY -- refit C on disjoint slices of the horizon")
    print("=" * 90)
    from src.core_physics.ap_closure import consumption as _cons
    k_as_ = float(bio.k_as) * M_TO_CM
    k_aa_ = float(bio.k_aa) * M_TO_CM
    minf_ = float(bio.Minf) * PER_M2_TO_PER_CM2
    WIN = ((0.0, 0.25), (0.25, 0.60), (0.60, 1.0), (0.0, 1.0))
    print("%-22s | %s | %8s %9s"
          % ("kernel", " ".join("%12s" % ("C[%.0f-%.0f%%]" % (100 * a, 100 * b)) for a, b in WIN),
             "drift", "R2_pooled"))
    stab = []
    for label, kname, mc in (("static  (= handoff)", "static", 0.0),
                             ("mat_linear a=0.03", "mat_linear", 0.03),
                             ("mat_linear a=0.10", "mat_linear", 0.10),
                             ("mat_linear a=0.30", "mat_linear", 0.30),
                             ("mat_linear a=1.00", "mat_linear", 1.00),
                             ("sat_plus_mat", "sat_plus_mat", 0.0)):
        Cs = []
        for lo_, hi_ in WIN:
            rs, xs = [], []
            for n, v in cache.items():
                mas_f = v["mas"] / minf_
                mat_f = v["mat"] / minf_
                sat_ = np.clip(1.0 - mas_f, 0.0, 1.0)
                ker = _cons(kname, v["gate"][None, :], sat_, mas_f, mat_f, k_as_, k_aa_,
                            mat_coef=mc)
                r_, x_ = samples(v, np.broadcast_to(ker, sat_.shape), 1.0,
                                 gated_only=args.gated_only, window=(lo_, hi_))
                rs.append(r_)
                xs.append(x_)
            Cs.append(fit_C_direct(np.concatenate(rs), np.concatenate(xs))[0])
        rs, xs = [], []
        for n, v in cache.items():
            mas_f, mat_f = v["mas"] / minf_, v["mat"] / minf_
            sat_ = np.clip(1.0 - mas_f, 0.0, 1.0)
            ker = _cons(kname, v["gate"][None, :], sat_, mas_f, mat_f, k_as_, k_aa_, mat_coef=mc)
            r_, x_ = samples(v, np.broadcast_to(ker, sat_.shape), 1.0, gated_only=args.gated_only)
            rs.append(r_)
            xs.append(x_)
        r_pool = r2(np.concatenate(rs), 1.0 / (1.0 + Cs[3] * np.concatenate(xs)))
        drift = max(Cs[:3]) / max(min(Cs[:3]), 1e-30)
        stab.append(dict(kernel=label, C=Cs, drift=drift, r2=r_pool))
        print("%-22s | %s | %7.2fx %9.4f"
              % (label, " ".join("%12.4g" % c for c in Cs), drift, r_pool))
    win = min(stab, key=lambda s: s["drift"])
    print("\n  most window-stable: %s  (drift %.2fx, C=%.4g, pooled R2 %.4f)"
          % (win["kernel"], win["drift"], win["C"][3], win["r2"]))
    print("  A kernel whose C depends on which part of the run you fit is MISSPECIFIED in")
    print("  time; the drift is the size of the time dependence it is failing to carry.")

    # -------------------------------------------------------- B. per-vessel C (kill test)
    kern, q_star = best["kernel"], best["q"]
    print("\n" + "=" * 90)
    print("B. PER-VESSEL C  (9: 'C varies wildly across train vessels' kills the global constant)")
    print("=" * 90)
    print("%-12s %10s %8s %8s | %8s %8s %8s"
          % ("vessel", "C", "R2", "n", "med_sr", "gate>0", "ap_min/ap0"))
    per = {}
    for n in sorted(cache):
        v = cache[n]
        ratio, x = samples(v, ker_cache[n][kern], q_star, gated_only=args.gated_only)
        if len(ratio) < 200:
            print("%-12s   (too few samples: %d)" % (n, len(ratio)))
            continue
        c_d, r_d = fit_C_direct(ratio, x)
        g = v["gate"] > 0
        per[n] = dict(C=c_d, r2=r_d, n=int(len(ratio)),
                      med_sr=float(np.median(v["sr0"][g])) if g.any() else float("nan"),
                      frac_gated=float(g.mean()),
                      ap_min_frac=float((v["ap"] / np.maximum(v["ap"][0], 1e-30)[None, :]).min()),
                      u_ref=float(v["u_ref"]), d_bar=float(v["d_bar"]),
                      n_wall=int(v["ap"].shape[1]))
        print("%-12s %10.4g %8.4f %8d | %8.3f %8.3f %8.4f"
              % (n, c_d, r_d, len(ratio), per[n]["med_sr"], per[n]["frac_gated"],
                 per[n]["ap_min_frac"]))
    cs = np.array([p["C"] for p in per.values()])
    print("\n  C across %d train vessels: median %.4g  geo-mean %.4g  IQR [%.4g, %.4g]"
          % (len(cs), np.median(cs), float(np.exp(np.log(cs).mean())),
             np.percentile(cs, 25), np.percentile(cs, 75)))
    print("  spread: max/min %.2fx   log10 sd %.3f   (a global constant is defensible if"
          " the ratio is small)" % (cs.max() / max(cs.min(), 1e-30), float(np.std(np.log10(cs)))))
    # what does the global C cost each vessel, vs its own best?
    glob = best["C_direct"]
    loss = []
    for n in sorted(per):
        ratio, x = samples(cache[n], ker_cache[n][kern], q_star, gated_only=args.gated_only)
        loss.append((n, per[n]["r2"], r2(ratio, 1.0 / (1.0 + glob * x))))
    fin = [(n, a, b) for n, a, b in loss if np.isfinite(a) and np.isfinite(b)]
    worst = min(fin, key=lambda z: z[2] - z[1]) if fin else ("-", np.nan, np.nan)
    print("\n  cost of using the pooled C=%.4g instead of each vessel's own:" % glob)
    print("     median R2 own %.4f -> pooled %.4f   (worst vessel %s: %.4f -> %.4f)"
          % (np.median([a for _, a, _ in fin]), np.median([b for _, _, b in fin]),
             worst[0], worst[1], worst[2]))

    # --------------------------------------------------- C. spatial vs temporal variance
    print("\n" + "=" * 90)
    print("C. WHERE THE R2 COMES FROM  (the closure is near-static: is that enough?)")
    print("=" * 90)
    print("%-12s | %8s %8s %8s | %8s %8s | %8s"
          % ("vessel", "R2_all", "R2_space", "R2_time", "cv_space", "cv_time", "rmse"))
    for n in sorted(per):
        v = cache[n]
        ap0 = v["ap"][0]
        ratio = v["ap"] / np.maximum(ap0, 1e-30)[None, :]
        x = ker_cache[n][kern] / np.power(v["sr_use"], q_star)
        if x.shape[0] == 1:
            x = np.broadcast_to(x, ratio.shape)
        g = v["gate"] > 0
        sel = np.zeros_like(ratio, dtype=bool)
        sel[1:, :] = True
        if args.gated_only:
            sel &= g[None, :]
        sel &= np.isfinite(ratio) & (ratio > 0)
        pred = 1.0 / (1.0 + glob * x)
        # spatial: at the final timestep only.  temporal: vessel-mean over nodes vs time.
        tf = ratio.shape[0] - 1
        sp = sel[tf]
        r_sp = r2(ratio[tf][sp], pred[tf][sp]) if sp.sum() > 3 else float("nan")
        mt_y = np.array([ratio[i][sel[i]].mean() for i in range(1, ratio.shape[0]) if sel[i].any()])
        mt_p = np.array([pred[i][sel[i]].mean() for i in range(1, ratio.shape[0]) if sel[i].any()])
        r_t = r2(mt_y, mt_p) if len(mt_y) > 3 else float("nan")
        print("%-12s | %8.4f %8.4f %8.4f | %8.4f %8.4f | %8.4f"
              % (n, r2(ratio[sel], pred[sel]), r_sp, r_t,
                 float(np.std(ratio[tf][sp]) / max(np.mean(ratio[tf][sp]), 1e-30)) if sp.sum() > 3 else np.nan,
                 float(np.std(mt_y) / max(np.mean(mt_y), 1e-30)) if len(mt_y) > 3 else np.nan,
                 rmse(ratio[sel], pred[sel])))
    print("\n  R2 is NaN wherever the target is flat to <1e-4 -- read the rmse column there.")
    print("  R2_space is the closure doing its actual job (grading identically-gated nodes).")
    print("  R2_time being poor/negative means the closure holds ap fixed while GT depletes it;")
    print("  that is a level error the rollout can absorb through da_scale, not an ordering error.")

    # ------------------------------------------------- D. is C conditionable on descriptors?
    print("\n" + "=" * 90)
    print("D. DOES C TRACK A VESSEL DESCRIPTOR?  (if yes, condition it; if no, keep it global)")
    print("=" * 90)

    def spear(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    keys = ["med_sr", "frac_gated", "u_ref", "d_bar", "n_wall", "ap_min_frac"]
    lc = np.log10(np.array([per[n]["C"] for n in sorted(per)]))
    for k in keys:
        v = np.array([per[n][k] for n in sorted(per)])
        f = np.isfinite(v) & np.isfinite(lc)
        print("   spearman(log10 C, %-12s) = %+.3f" % (k, spear(v[f], lc[f])))

    # ---------------------------------------------------------- E. residual locality (-> GNN)
    print("\n" + "=" * 90)
    print("E. RESIDUAL LOCALITY -- does a graph model have a job left? (4)")
    print("=" * 90)
    from scipy.spatial import cKDTree
    print("%-12s | %10s %10s | %10s" % ("vessel", "corr(res,", "res_nbr)", "corr(ap, ap_nbr)"))
    rr, aa = [], []
    for n in sorted(per):
        v = cache[n]
        xy = v["pos"]
        k = min(7, len(xy))
        _, nbr = cKDTree(xy).query(xy, k=k)
        ap0 = v["ap"][0]
        ratio = v["ap"] / np.maximum(ap0, 1e-30)[None, :]
        x = ker_cache[n][kern] / np.power(v["sr_use"], q_star)
        if x.shape[0] == 1:
            x = np.broadcast_to(x, ratio.shape)
        res = ratio - 1.0 / (1.0 + glob * x)
        mid = ratio.shape[0] // 2
        g = v["gate"] > 0
        rn = res[mid][nbr[:, 1:]].mean(1)
        an = v["ap"][mid][nbr[:, 1:]].mean(1)
        m = g & np.isfinite(res[mid]) & np.isfinite(rn)
        if m.sum() < 10:
            continue
        c_res = float(np.corrcoef(res[mid][m], rn[m])[0, 1])
        c_ap = float(np.corrcoef(v["ap"][mid][m], an[m])[0, 1])
        rr.append(c_res)
        aa.append(c_ap)
        print("%-12s | %10.4f %10s | %10.4f" % (n, c_res, "", c_ap))
    if rr:
        print("\n  median residual neighbour-correlation %.3f  (0 => the closure is complete;"
              "\n  large => the leftover error is a smooth FIELD, i.e. transport a GNN can carry)"
              % float(np.median(rr)))
        print("  median ap neighbour-correlation       %.3f" % float(np.median(aa)))

    payload = dict(arm=args.arm, gated_only=bool(args.gated_only), train=sorted(cache),
                   scan=scan, fine=[dict(q=a, C=b, r2=c) for a, b, c in fine],
                   best=best, per_vessel=per,
                   residual_nbr_corr_median=float(np.median(rr)) if rr else None)
    (OUT / f"fit_{args.arm}.json").write_text(json.dumps(payload, indent=2, default=float),
                                              encoding="utf-8")
    print("\nwrote %s" % (OUT / f"fit_{args.arm}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
