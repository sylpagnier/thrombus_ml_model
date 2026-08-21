"""Predict the off-wall BUDGET, which is the dominant recoverable error at the final time.

Two measurements reframe the off-wall readout:

1. `scripts/eval_offwall_final.py` + the oracle-prefix sweep: the best prefix of the current
   ranking scores **0.8185** against the shipped 0.7359.  So ~0.08 is a budget error and the
   ranking itself caps at 0.82.
2. The optimal prefix length is essentially the true burden -- `k* / n_gt` reads
   136/120, 97/90, 81/84, 34/34, 102/122, 8/9, 15/14.  **Choosing the budget IS predicting
   how many off-wall nodes clot.**

`docs/PHASE9_ML.md` 4 killed a confidence-mass budget (`k = a * sum(p)`) on the v3 score
field, and that verdict is worth revisiting because the field has changed: measured on the
v5 field over the 13 vessels with off-wall GT,

    sum(p) over off-wall     spearman +0.820   pearson +0.824   median sum/n_gt = 3.0
    count(mat_off_est>=crit) spearman +0.647   pearson +0.849
    count(wall predicted)    spearman +0.740   pearson +0.766

so the mass is a genuinely informative burden signal now.  Each rule here spends exactly ONE
fitted scalar, chosen in-fold by maximising the severity score directly -- at n=19 the size
of the search space is itself a hyperparameter (docs/PHASE10_V4.md 13.3), so the rules are
kept deliberately thin.

    python scripts/eval_offwall_budget.py --tags v5a,v5b,v5c --cache v5
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

from eval_expected_score_readout import expected_curve  # noqa: E402
from eval_strict import load_scores  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.softmetric import dilation_operator, to_torch_sparse  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
GAMMA = [0.5, 1.0, 2.0]
A_GRID = np.round(np.geomspace(0.05, 4.0, 30), 4)
L2 = float(np.log(2.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--save-masks", default="")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc_all = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    fo = {a: k for k, held in folds.items() for a in held}
    sc = {a: sc_all[(fo[a], a)] for a in pool}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    Dt = {a: to_torch_sparse(dilation_operator(cache[a]["edge_index"],
                                               len(cache[a]["wall"]), 2), dev) for a in pool}
    cols = [str(c) for c in cache[pool[0]]["cols"]]
    i_est = cols.index("log_mat_off_est")

    # --- per-vessel, label-free burden signals ------------------------------------
    sig, order, curves = {}, {}, {}
    for a in pool:
        S = cache[a]
        d = ~S["wall"]
        sig[a] = dict(
            mass=float(sc[a][d].sum()),
            est=float((d & (S["X"][:, i_est] >= L2)).sum()),
            wall=float((S["wall"] & (sc[a] >= 0.5)).sum()))
        order[a] = np.flatnonzero(d)[np.argsort(-sc[a][d])]
        for g in GAMMA:
            curves[(a, g)] = expected_curve(sc[a], d, Dt[a], dev, g)

    def mask_k(a, k):
        m = np.zeros(len(sc[a]), bool)
        m[order[a][:max(1, min(int(k), len(order[a])))]] = True
        return m

    def score_k(a, k):
        return vs[a].score(mask_k(a, k), ~cache[a]["wall"])

    ARMS = ["expected", "mass", "est", "wall", "mass_x_expected", "nested_pick"]
    rows = {r: {} for r in ARMS}
    masks = {a: np.zeros(len(sc[a]), bool) for a in pool}

    for kf, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]

        def fit_scalar(kfun):
            best = None
            for A in A_GRID:
                v = [x for x in (score_k(a, kfun(a, A)) for a in sel) if x == x]
                q = float(np.mean(v)) if v else -1e9
                if best is None or q > best[0]:
                    best = (q, float(A))
            return best

        # control: the expected-score readout, gamma fitted the same way
        bestE = None
        for g in GAMMA:
            kf_ = lambda a, A, _g=g: A * curves[(a, _g)][0][int(np.argmax(curves[(a, _g)][1]))]
            q, A = fit_scalar(kf_)
            if bestE is None or q > bestE[0]:
                bestE = (q, g, A)
        _, gE, AE = bestE

        def k_exp(a):
            ks, vals = curves[(a, gE)]
            return AE * ks[int(np.argmax(vals))]

        cands = {"expected": (bestE[0], k_exp)}
        for name in ("mass", "est", "wall"):
            q, A = fit_scalar(lambda a, A_, _n=name: A_ * sig[a][_n])
            cands[name] = (q, lambda a, _A=A, _n=name: _A * sig[a][_n])
        # geometric mean of the two independent estimates -- one scalar, not two
        q, A = fit_scalar(lambda a, A_: A_ * np.sqrt(max(k_exp(a), 1e-9)
                                                     * max(sig[a]["mass"], 1e-9)))
        cands["mass_x_expected"] = (
            q, lambda a, _A=A: _A * np.sqrt(max(k_exp(a), 1e-9) * max(sig[a]["mass"], 1e-9)))

        pick = max(cands, key=lambda n: cands[n][0])
        for a in held:
            for n in ARMS[:-1]:
                rows[n][a] = score_k(a, cands[n][1](a))
            rows["nested_pick"][a] = score_k(a, cands[pick][1](a))
            masks[a] = mask_k(a, cands[pick][1](a))
        print("  fold %d pick=%-16s sel %s" % (
            kf, pick, " ".join("%s %.3f" % (n, cands[n][0]) for n in ARMS[:-1])), flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    base = [a for a in pool if a not in prio]
    print("\nFINAL-TIME OFF-WALL, strictly nested (tags=%s)\n" % args.tags)
    print("%-18s | %9s %9s %9s" % ("budget rule", "ALL", "baseline", "PRIORITY"))
    for n in ARMS:
        R = rows[n]
        print("%-18s | %9.4f %9.4f %9.4f"
              % (n, np.nanmean([R[a] for a in pool if a in R]),
                 np.nanmean([R[a] for a in base if a in R]),
                 np.nanmean([R[a] for a in prio if a in R])))
    print("\nper vessel: expected -> nested_pick")
    for a in sorted(pool):
        if a not in rows["expected"] or rows["expected"][a] != rows["expected"][a]:
            continue
        print("   %-11s %.4f -> %.4f" % (a, rows["expected"][a], rows["nested_pick"][a]))
    if args.save_masks:
        np.savez_compressed(args.save_masks, **{a: masks[a] for a in pool})
        print("\nwrote %s" % args.save_masks)
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
