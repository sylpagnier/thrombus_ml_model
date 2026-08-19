"""Two untried readout levers, strictly nested: HEAD FUSION and an ADAPTIVE threshold.

Both attack the gap `scripts/diag_readout_ceiling.py` measured -- a per-vessel oracle cut on
the *same* score field reads wall 0.9447 / off 0.8275 against the cohort cut's 0.9024 /
0.7075 -- from angles `scripts/eval_calibration_rules.py` did not try.

**HEAD FUSION.**  `src/clot_ml/gnn.py` has two heads and only one is ever read out.
`scripts/eval_reg_readout.py` measured the regression head to be a *better off-wall field*
than the classifier on identical weights (0.6006 against 0.5105; priority 0.8296 against
0.6445) and then found its physical anchor `Mat >= crit` unusable because the head is not
magnitude-calibrated.  That kills the anchor, not the field.  Nobody has fused the two.
Both fusions here are rank-based or scale-free, so neither inherits the calibration problem:

    fuse_rank    mean of the two fields' WITHIN-VESSEL, WITHIN-DOMAIN rank transforms
    fuse_logit   logit(cls) + w * (reg - median(reg)), one cohort weight `w`

**ADAPTIVE THRESHOLD.**  `eval_calibration_rules.py` replaced the cohort constant with a
statistic of this vessel's own score distribution (quantile, max-relative, largest gap) and
all of them lost.  Each of those *substitutes* the constant.  This instead **perturbs** it:

    t_v = clip(a + b * (stat_v - median_over_train(stat)), 0.02, 0.98)

so `b = 0` reproduces the cohort cut exactly and the fit can only move away from it if the
statistic pays.  Two parameters, fitted on the selection vessels by directly maximising the
severity score -- not by regressing on oracle thresholds, which is a noisy target.

    python scripts/eval_fusion_calib.py --tags v5a,v5b,v5c --cache v5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
GRID = np.round(np.linspace(0.02, 0.98, 33), 4)
W_GRID = np.round(np.linspace(0.0, 1.5, 13), 3)
B_GRID = np.round(np.linspace(-1.2, 1.2, 13), 3)


def load_both(tags):
    zs = [np.load(REPO / f"outputs/phase9_scores/{t}.npz", allow_pickle=True) for t in tags]
    for t, z in zip(tags, zs):
        if not any(k.startswith("reg|") for k in z.files):
            raise SystemExit("tag %s has no regression field" % t)
    pool = [str(x) for x in zs[0]["pool"]]
    folds = {int(k.split("|")[1]): [str(x) for x in zs[0][k]]
             for k in zs[0].files if k.startswith("held|")}
    cls, reg = {}, {}
    for k in folds:
        for a in pool:
            cls[(k, a)] = np.mean([z["%d|%s" % (k, a)] for z in zs], axis=0)
            reg[(k, a)] = np.mean([z["reg|%d|%s" % (k, a)] for z in zs], axis=0)
    return pool, folds, cls, reg


def rank01(x, d):
    """Rank-transform ``x`` to [0,1] within the domain ``d``; 0 outside."""
    out = np.zeros_like(x, dtype=np.float64)
    v = x[d]
    if v.size == 0:
        return out
    r = np.argsort(np.argsort(v)).astype(np.float64)
    out[d] = r / max(v.size - 1, 1)
    return out


def build_fields(cls, reg, d, w):
    lg = np.log(np.clip(cls, 1e-6, 1 - 1e-6)) - np.log(1 - np.clip(cls, 1e-6, 1 - 1e-6))
    med = float(np.median(reg[d])) if d.any() else 0.0
    fl = 1.0 / (1.0 + np.exp(-(lg + w * (reg - med))))
    return fl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    pool, folds, CLS, REG = load_both(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    fo = {a: k for k, held in folds.items() for a in held}
    cls = {a: CLS[(fo[a], a)] for a in pool}
    reg = {a: REG[(fo[a], a)] for a in pool}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    doms = {"wall": lambda S: S["wall"], "off": lambda S: ~S["wall"]}

    # vessel-level statistics for the adaptive cut, all label-free
    def stats(a, d):
        v = cls[a][d]
        S = cache[a]
        return dict(q90=float(np.quantile(v, 0.90)) if v.size else 0.0,
                    q99=float(np.quantile(v, 0.99)) if v.size else 0.0,
                    mean=float(v.mean()) if v.size else 0.0,
                    physfrac=float((S["phys_mask"] & d).sum() / max(d.sum(), 1)))

    STATS = ["q90", "q99", "mean", "physfrac"]
    ARMS = (["cls", "reg_tuned", "fuse_rank", "fuse_logit"]
            + ["adapt_" + s for s in STATS])
    rows = {r: {a: {} for a in pool} for r in ARMS}
    picks = {}

    def tune_cut(anchors, field, d_of):
        top, pick = -1e9, float(GRID[0])
        for t in GRID:
            vals = []
            for a in anchors:
                d = d_of(cache[a])
                x = vs[a].score(d & (field[a] >= t), d)
                if x == x:
                    vals.append(x)
            if vals and np.mean(vals) > top:
                top, pick = float(np.mean(vals)), float(t)
        return pick, top

    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        for dk, d_of in doms.items():
            # ---- controls ---------------------------------------------------------
            t_cls, _ = tune_cut(sel, cls, d_of)
            t_reg, _ = tune_cut(sel, reg, d_of)
            # ---- rank fusion ------------------------------------------------------
            fr = {a: 0.5 * (rank01(cls[a], d_of(cache[a])) + rank01(reg[a], d_of(cache[a])))
                  for a in pool}
            t_fr, _ = tune_cut(sel, fr, d_of)
            # ---- logit fusion: weight and cut chosen together ---------------------
            best_fl = None
            for w in W_GRID:
                fl = {a: build_fields(cls[a], reg[a], d_of(cache[a]), w) for a in sel}
                t, q = tune_cut(sel, fl, d_of)
                if best_fl is None or q > best_fl[0]:
                    best_fl = (q, float(w), t)
            _, w_fl, t_fl = best_fl
            # ---- adaptive cut: t_v = a + b*(stat_v - median(stat_train)) ----------
            adapt = {}
            for sname in STATS:
                sv = {a: stats(a, d_of(cache[a]))[sname] for a in pool}
                med = float(np.median([sv[a] for a in sel]))
                best = None
                for a0 in GRID:
                    for b in B_GRID:
                        vals = []
                        for a in sel:
                            d = d_of(cache[a])
                            t = float(np.clip(a0 + b * (sv[a] - med), 0.02, 0.98))
                            x = vs[a].score(d & (cls[a] >= t), d)
                            if x == x:
                                vals.append(x)
                        if vals and (best is None or np.mean(vals) > best[0]):
                            best = (float(np.mean(vals)), float(a0), float(b))
                adapt[sname] = (best[1], best[2], med, sv)
            picks.setdefault(k, {})[dk] = dict(
                t_cls=t_cls, t_reg=t_reg, t_fr=t_fr, w_fl=w_fl, t_fl=t_fl,
                adapt={s: (adapt[s][0], adapt[s][1]) for s in STATS})

            for a in held:
                S = cache[a]
                d = d_of(S)
                fl_a = build_fields(cls[a], reg[a], d, w_fl)
                M = {"cls": d & (cls[a] >= t_cls),
                     "reg_tuned": d & (reg[a] >= t_reg),
                     "fuse_rank": d & (fr[a] >= t_fr),
                     "fuse_logit": d & (fl_a >= t_fl)}
                for sname in STATS:
                    a0, b, med, sv = adapt[sname]
                    t = float(np.clip(a0 + b * (sv[a] - med), 0.02, 0.98))
                    M["adapt_" + sname] = d & (cls[a] >= t)
                for r in ARMS:
                    rows[r][a][dk] = vs[a].score(M[r], d)
        print("  fold %d done" % k, flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\nFINAL TIME POINT, strictly nested (tags=%s)\n" % args.tags)
    print("%-14s | %9s %9s | %9s %9s" % ("arm", "wall", "off", "P wall", "P off"))
    for r in ARMS:
        R = rows[r]
        print("%-14s | %9.4f %9.4f | %9.4f %9.4f"
              % (r, np.nanmean([R[a]["wall"] for a in pool]),
                 np.nanmean([R[a]["off"] for a in pool]),
                 np.nanmean([R[a]["wall"] for a in prio]),
                 np.nanmean([R[a]["off"] for a in prio])))
    if args.save:
        Path(args.save).write_text(json.dumps(dict(rows=rows, picks=picks), indent=2,
                                              default=float))
        print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
