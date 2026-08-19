"""PHASE 8: is the missing term a WASHOUT -- Mat leaving the wall with the flow?

Established by ``diag_local_ode_closure.py`` / ``diag_local_ode_residual.py``: a per-node
surface ODE handed perfect GT ``RP/AP/M/Mas/sr/dsrx`` still cannot order GT ``Mat`` (rank
0.31 on live nodes, and NEGATIVE on 5 of 19 vessels), and the two obvious non-local
mechanisms are dead -- the per-node rate is not the local cell size (rank -0.02 against
1/h_i) and upwind accumulation along the wall buys exactly 0.000, which is what no-slip
predicts since the tangential velocity AT the wall is zero.

What is left is the thing the repo's surface ODE assumes away.  ``Mat`` is not a surface
coverage.  It is a *Transport of Diluted Species* DOMAIN concentration with ``D = 0``, and
the wall reaction is a FLUX boundary condition into that domain field.  Material deposited
at the wall is therefore not pinned there: it sits in the near-wall fluid and is carried off
by the flow.  The repo integrates accumulation with no removal at all, so it has

    d(Mat)/dt = k * J0                     (repo -- monotone, can only ever grow)
    d(Mat)/dt = k * J0 - lambda * sr * Mat (with washout -- a real steady state)

The washout rate uses ``sr`` because the near-wall clearance rate IS the wall velocity
gradient; ``lambda`` is a single global scalar with units of length, swept here.

WHY THIS WOULD PRODUCE THE OBSERVED SIGN ERROR.  The gate has two branches.  The stagnation
branch fires where ``sr < lss``, so those nodes deposit AND retain.  The separation branch
fires on ``d(sr,x) < sgt``, which happens at reattachment points where ``sr`` itself can be
large -- those nodes deposit and are immediately scoured.  A model with no removal ranks the
second group far too high, and the separation branch is exactly the one that carries a
magnitude (``(L/gamma_m)*|dsrx|`` reaches ~1.5), so it dominates the predicted ordering.

    python scripts/diag_mat_washout.py
    python scripts/diag_mat_washout.py --lam-grid 25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig  # noqa: E402
from src.core_physics.physics_wall_model import gate_from_shear, washout_step  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"
CACHE = REPO / "outputs/wall_species_cache"
M_TO_CM = 100.0
PER_M2_TO_PER_CM2 = 1.0e-4


def j0_parts(z, bio):
    """``J0_Mat`` split into its FLOW factor and its CHEMISTRY factor, each ``[T, W]``.

    ``J0_Mat = Da * gate(sr, dsrx) * chem(M, Mas, RP, AP)``.  Freezing one factor at t=0 while
    letting the other evolve is what separates "the flow has to evolve" from "the chemistry has
    to evolve" -- two claims that a single time-varying ``J0`` cannot distinguish, and which
    FINDINGS 7 conflated when it tested a flow oracle only.
    """
    k_rs = float(bio.k_rs) * M_TO_CM
    k_as = float(bio.k_as) * M_TO_CM
    k_aa = float(bio.k_aa) * M_TO_CM
    minf = float(bio.Minf) * PER_M2_TO_PER_CM2
    gate = gate_from_shear(z["sr_t"], z["dsrx_t"], bio)
    sat = np.clip(1.0 - z["m_tot"] / minf, 0.0, 1.0)
    chem = sat * (k_rs * z["rp"] + k_as * z["ap"]) + (z["mas"] / minf) * k_aa * z["ap"]
    return gate, float(bio.surface_damkohler) * chem


def j0_mat(z, bio):
    gate, chem = j0_parts(z, bio)
    return gate * chem


#: The mechanisms being compared.  ``washout`` is the hypothesis; the other two are the
#: cheaper explanations that must be ruled out before it can be called flow-driven.
#:
#:   washout   dMat/dt = J0 - lam*sr*Mat    clearance BY THE FLOW; rate set by the near-wall
#:                                          velocity gradient, so it is spatially structured
#:                                          and differs between the gate's two branches
#:   lifetime  dMat/dt = J0 - lam*Mat       a finite lifetime with no flow dependence at all
#:                                          (fibrinolysis would look like this)
#:   saturate  dMat/dt = J0*(1 - Mat/Msat)  no removal, just a ceiling; lam = 1/Msat
#:
#: ``lifetime`` is the one that matters.  It has the same number of parameters and the same
#: "curve bends over" behaviour, so if it scores the same then the improvement is only
#: relaxation and says nothing about flow.
MECHANISMS = ("washout", "lifetime", "saturate")


def rollout(j0, t, sr, lam, mech="washout"):
    """Integrate one mechanism.  Returns final ``Mat`` up to the global rate scalar.

    The removal arms go through ``physics_wall_model.washout_step`` -- the same backward-Euler
    update the model uses -- so the ``lambda`` fitted here is the ``lambda`` the model wants.
    Integrating this explicitly here and implicitly there would make the fitted scalar mean
    two different things at ``h*lambda*sr ~ 1``, which is where the interesting nodes are.
    """
    mat = np.zeros(j0.shape[1])
    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        if mech == "washout":
            mat = washout_step(mat, j0[i], h, lam * np.abs(sr[i]))
        elif mech == "lifetime":
            mat = washout_step(mat, j0[i], h, np.full(mat.shape, lam))
        elif mech == "saturate":
            mat = mat + h * j0[i] * np.maximum(1.0 - lam * mat, 0.0)
        else:
            raise ValueError(f"unknown mechanism {mech!r}")
        np.maximum(mat, 0.0, out=mat)
    return mat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-grid", type=int, default=17)
    ap.add_argument("--save", default="outputs/phase8_mat_washout.json")
    args = ap.parse_args()
    bio = BiochemConfig(phase="biochem")
    lams = np.concatenate([[0.0], np.logspace(-8, -3, args.lam_grid)])
    lg = lambda a: np.log10(np.maximum(np.asarray(a, float), 1e-30))

    packs = []
    for anchor in WALL_COHORT_V2_TRAIN:
        pk, cf = DIR / f"{anchor}.pt", CACHE / f"{anchor}.npz"
        if not (pk.exists() and cf.exists()):
            continue
        z = np.load(cf)
        if "sr_t" not in z.files:
            continue
        j0 = j0_mat(z, bio)
        gate, chem = j0_parts(z, bio)
        t = z["t"]
        integ = np.zeros_like(j0)
        integ[1:] = np.cumsum(j0[:-1] * np.diff(t)[:, None], axis=0)
        live = (integ[-1] > 0) & (z["mat"][-1] > 0)
        if live.sum() < 25:
            continue
        packs.append((anchor, j0, t, z["sr_t"], z["mat"][-1], live, gate, chem))
    print("[i] %d vessels" % len(packs))

    def score(mech, lam):
        """Per-vessel rank correlation against GT Mat at one ``lam``."""
        return np.array([spearman(rollout(j0, t, sr, lam, mech)[live], gt[live])
                         for _, j0, t, sr, gt, live, _g, _c in packs])

    # Each mechanism needs its own lambda scale: washout multiplies sr [1/s], lifetime is
    # a bare rate [1/s], saturate is an inverse concentration [cm^2/plt].
    grids = {"washout": lams, "lifetime": np.concatenate([[0.0], np.logspace(-8, -3, args.lam_grid)]),
             "saturate": np.concatenate([[0.0], np.logspace(-11, -5, args.lam_grid)])}
    curves, results = {}, {}
    base = score("washout", 0.0)

    for mech in MECHANISMS:
        rows = [(lam, score(mech, lam)) for lam in grids[mech]]
        curves[mech] = [dict(lam=float(l), rho=float(np.nanmean(r)),
                             n_neg=int((r < 0).sum())) for l, r in rows]
        lam_b, r_b = max(rows, key=lambda kv: np.nanmean(kv[1]))
        results[mech] = dict(lam=float(lam_b), rho=float(np.nanmean(r_b)),
                             n_neg=int((r_b < 0).sum()),
                             per_vessel={a: float(v) for (a, *_), v in zip(packs, r_b)})

    # THE CONTROL THAT DECIDES WHETHER ANY OF THIS IS NEW.  As lam*sr*T grows the washout
    # solution goes to its steady state J0/(lam*sr), so its ORDERING tends to 1/sr and the
    # chemistry drops out entirely.  If plain 1/sr already ranks GT Mat as well as the fitted
    # washout does, then the "gain" is only re-discovering that GT clot sits in slow flow --
    # which the stagnation branch of the gate already encodes -- and is not a new term.
    nulls = {}
    for name, pred_fn in (
            ("1/sr", lambda j0, sr: 1.0 / np.maximum(sr.mean(0), 1e-3)),
            ("-sr", lambda j0, sr: -sr.mean(0)),
            ("J0/sr", lambda j0, sr: j0[-1] / np.maximum(sr.mean(0), 1e-3)),
            ("J0 integral", lambda j0, sr: None)):
        vals = []
        for anchor, j0, t, sr, gt, live, _g, _c in packs:
            p = pred_fn(j0, sr)
            if p is None:
                p = rollout(j0, t, sr, 0.0)
            vals.append(spearman(p[live], gt[live]))
        nulls[name] = float(np.nanmean(vals))

    print("\n=== NULLS: what a bare shear correlate already gets ===")
    for name, v in nulls.items():
        print("   %-14s rank vs GT Mat  %.3f" % (name, v))

    print("\n=== MECHANISM COMPARISON (rank vs GT Mat, mean over %d vessels) ===" % len(packs))
    print("   %-10s %12s %10s %10s" % ("mechanism", "best lam", "rho", "n_neg"))
    print("   %-10s %12s %10.3f %10d" % ("none", "-", float(np.nanmean(base)),
                                         int((base < 0).sum())))
    for mech in MECHANISMS:
        r = results[mech]
        print("   %-10s %12.3e %10.3f %10d" % (mech, r["lam"], r["rho"], r["n_neg"]))

    # Leave-one-vessel-out: fit lambda on 18, score the held-out one.  This is what says
    # whether the scalar transfers or is being read off the vessel it is scored on.
    print("\n=== LEAVE-ONE-VESSEL-OUT (lambda fit on the other 18) ===")
    print("   %-12s %9s %9s %12s" % ("held out", "no term", "washout", "lam(fit on 18)"))
    loo_base, loo_wash = [], []
    all_scores = {lam: score("washout", lam) for lam in lams}
    for k, (anchor, *_rest) in enumerate(packs):
        keep = [i for i in range(len(packs)) if i != k]
        lam_k = max(lams, key=lambda l: np.nanmean(all_scores[l][keep]))
        b, w = float(base[k]), float(all_scores[lam_k][k])
        loo_base.append(b)
        loo_wash.append(w)
        print("   %-12s %+9.3f %+9.3f %12.3e" % (anchor, b, w, lam_k))
    print("   %-12s %+9.3f %+9.3f" % ("MEAN", float(np.mean(loo_base)), float(np.mean(loo_wash))))

    # === WHY THE TERM DIES IN THE DEPLOY MODEL ===
    #
    # ``scripts/eval_washout_arm.py`` finds the washout makes the SHIPPED model's ordering
    # worse, not better, and the per-vessel signature is exact sign flips -- the mark of the
    # solution having reached its steady state ``J0/(lam*sr)``, whose ordering is the 1/sr null
    # above.  The shipped model freezes the gate and ``ap``/``rp`` at t=0, so its source is
    # constant in time, and a constant source against linear removal has exactly one attractor.
    # Accumulation is what let the frozen approximation survive: integrating a constant source
    # still yields a growing, informative field.
    #
    # ``J0 = gate(flow) * chem`` though, so "let the inputs evolve" bundles two very different
    # claims.  FINDINGS 7 tested a FLOW oracle only and found nothing, and the model path in
    # ``scripts/eval_flow_washout_2x2.py`` agrees -- a GT-flow gate plus washout is still worse
    # than the shipped model.  Freezing the two factors independently says which one the removal
    # term actually needs, which is the difference between "invest in the corrector" and
    # "invest in the chemistry".
    print("\n=== WHICH INPUT HAS TO EVOLVE?  (rank vs GT Mat) ===")
    lam_b = results["washout"]["lam"]
    hold = lambda a: np.repeat(a[:1], a.shape[0], axis=0)
    axes = (("frozen both", True, True), ("evolving FLOW only", False, True),
            ("evolving CHEMISTRY only", True, False), ("evolving both", False, False))
    cells = {}
    for lbl, fg, fc in axes:
        for lam in (0.0, lam_b):
            vals = []
            for anchor, j0, t, sr, gt, live, gate, chem in packs:
                jj = (hold(gate) if fg else gate) * (hold(chem) if fc else chem)
                ss = hold(sr) if fg else sr
                vals.append(spearman(rollout(jj, t, ss, lam)[live], gt[live]))
            cells[(lbl, lam > 0)] = float(np.nanmean(vals))
    print("   %-26s %16s %14s %10s" % ("inputs", "accumulate-only", "with washout",
                                       "d(removal)"))
    for lbl, _fg, _fc in axes:
        a, b = cells[(lbl, False)], cells[(lbl, True)]
        print("   %-26s %16.3f %14.3f %+10.3f" % (lbl, a, b, b - a))
    # The column above is the removal delta WITHIN a row.  What decides priorities is each
    # change measured against the shipped-like corner, so state that separately rather than
    # leaving two different deltas to be confused for each other.
    ref = cells[("frozen both", False)]
    print("\n   against the frozen accumulate-only corner (%.3f), one change at a time:" % ref)
    for lbl, key in (("removal only", ("frozen both", True)),
                     ("evolving flow only", ("evolving FLOW only", False)),
                     ("evolving chemistry only", ("evolving CHEMISTRY only", False)),
                     ("all three together", ("evolving both", True))):
        print("      %-26s %+.3f" % (lbl, cells[key] - ref))
    print("\n   1/sr null = %.3f." % nulls["1/sr"])

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(curves=curves, results=results, nulls=nulls,
                                   base_rho=float(np.nanmean(base)),
                                   input_axes={"%s|washout=%s" % k: v
                                               for k, v in cells.items()},
                                   loo=dict(base=float(np.mean(loo_base)),
                                            washout=float(np.mean(loo_wash)))), indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
