"""What ONSET ORDERING can the AP closure actually buy?  (guides PHASE6_HANDOFF 5.2)

The closure is fit on ``ap`` and scored with R2, but the rollout is scored on *timing*.
One piece of algebra decides how different those two questions are:

    ap_pred_i = ap0 / (1 + C * gate_i * k_as / sr_i^q)

With the gate and shear frozen at t=0 this is a **deterministic monotone function of
(gate, sr)**.  Among nodes whose gate is exactly 1 -- precisely the set that flashes,
because their ODEs are identical -- ``rank(ap_pred) == rank(sr)``, so the closure's onset
order IS the shear order and can be nothing else.  Its ceiling there is |rho(sr, onset)|,
median 0.470 in 2 -- not the 0.727 that ``ap@t_final`` carries.  The difference is the
non-local part of the AP field, and it is the quantitative case for the graph model in 4.

TWO SETS, AND THEY DISAGREE.  On the gate==1 subset higher ap goes with EARLIER onset
(supply-dominated: rho -0.727).  Over the FULL gated set the sign flips to +0.26, because
there the heterogeneous gate magnitude and the consumption feedback (a node that ignited
early has been eating ap ever since) both cut the other way.  Only the gate==1 set isolates
the mechanism, so it is reported first and is what the budget is computed on.

CONTAMINATION.  ``ap@t_final`` is an outcome-contaminated oracle -- it already knows who
ignited.  ``ap@early`` (a fixed small fraction of the horizon, before much has deposited)
is the transport-set field a graph model could legitimately try to predict, so the budget
is quoted against that and ``t_final`` is shown only for continuity with 2.

Nothing here is fit; it only reads the cache.  SEALED appears only under --peek-sealed.
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
MIN_N = 8


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def med(rows, key):
    v = np.array([r[key] for r in rows], float)
    v = v[np.isfinite(v)]
    return (float(np.median(v)), float(v.mean()), int((v < 0).sum()), len(v)) if len(v) \
        else (float("nan"),) * 2 + (0, 0)


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--C", type=float, default=54.7)
    ap_.add_argument("--q", type=float, default=1.0)
    ap_.add_argument("--early", type=float, default=0.10,
                     help="fraction of the horizon at which the 'transport-set' AP is read")
    ap_.add_argument("--peek-sealed", action="store_true")
    args = ap_.parse_args()
    bio = BiochemConfig(phase="biochem")
    k_as = float(bio.k_as) * M_TO_CM
    lss = float(bio.lss)
    sgt = float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    OUT.mkdir(parents=True, exist_ok=True)

    names = list(WALL_COHORT_V2_TRAIN)
    if args.peek_sealed:
        names += list(WALL_COHORT_V2_GENERALIZATION)

    # predictors of "ignites earlier" -- all expected to rank-correlate NEGATIVELY with onset
    PRED = ["sr", "gate", "gate*ap_pred", "gate*ap_early", "gate*ap_final"]
    rows_g1, rows_all = [], []
    print("C=%.4g q=%.3f  early=%.0f%% of horizon" % (args.C, args.q, 100 * args.early))
    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        sr0, dsrx0, onset, ap = z["sr0"], z["dsrx0"], z["gt_onset"], z["ap"]
        T = ap.shape[0]
        gate = (dsrx0 < sgt) * coef * np.abs(dsrx0) + (sr0 < lss)
        ap_pred = ap[0] / (1.0 + args.C * gate * k_as / np.power(np.maximum(sr0, 1e-3), args.q))
        k_e = max(1, int(round(args.early * (T - 1))))
        cand = {"sr": sr0, "gate": gate, "gate*ap_pred": gate * ap_pred,
                "gate*ap_early": gate * ap[k_e], "gate*ap_final": gate * ap[-1]}

        hot = onset >= 0
        sets = {"g1": (sr0 < lss) & ~(dsrx0 < sgt) & hot, "all": (gate > 0) & hot}
        for tag, sel in sets.items():
            if sel.sum() < MIN_N:
                continue
            r = dict(name=n, sealed=bool(z["sealed"]), n=int(sel.sum()),
                     cv_ap_early=float(np.std(ap[k_e][sel]) / max(np.mean(ap[k_e][sel]), 1e-30)),
                     field_early=spear(ap_pred[sel], ap[k_e][sel]),
                     field_final=spear(ap_pred[sel], ap[-1][sel]))
            for k in PRED:
                r[k] = spear(cand[k][sel], onset[sel])
            (rows_g1 if tag == "g1" else rows_all).append(r)

    for tag, rows in (("gate==1 ONLY  (identical ODEs -- the flash set)", rows_g1),
                      ("ALL GATED     (mixed gate magnitude; sign is confounded)", rows_all)):
        tr = [r for r in rows if not r["sealed"]]
        if not tr:
            continue
        print("\n" + "=" * 92)
        print("%s   TRAIN n=%d" % (tag, len(tr)))
        print("=" * 92)
        print("%-12s %5s | %s | %8s %8s"
              % ("vessel", "n", " ".join("%13s" % k for k in PRED), "fld_ear", "cv_ap"))
        for r in tr:
            print("%-12s %5d | %s | %+8.3f %8.4f"
                  % (r["name"], r["n"], " ".join("%+13.3f" % r[k] for k in PRED),
                     r["field_early"], r["cv_ap_early"]))
        print("-" * 92)
        for k in PRED:
            m, mu, neg, nn = med(tr, k)
            print("   median rho(%-14s, onset) = %+.3f   mean %+.3f   (%d/%d negative)"
                  % (k, m, mu, neg, nn))
        mp = med(tr, "gate*ap_pred")[0]
        me = med(tr, "gate*ap_early")[0]
        mg = med(tr, "gate")[0]
        print("\n   closure over the no-closure baseline : %+.3f -> %+.3f  (delta %+.3f)"
              % (mg, mp, mp - mg))
        print("   closure vs the transport-set AP field: %+.3f -> %+.3f  (BUDGET for a graph"
              " model %+.3f)" % (mp, me, me - mp))
        print("   AP field recovery, spearman(ap_pred, ap@early) median %+.3f"
              % med(tr, "field_early")[0])

    (OUT / "ordering.json").write_text(json.dumps(
        dict(C=args.C, q=args.q, early=args.early, gate1=rows_g1, all_gated=rows_all),
        indent=2, default=float), encoding="utf-8")
    print("\nwrote %s" % (OUT / "ordering.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
