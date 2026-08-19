"""Pick WHICH score ensemble to use, per domain, inside the fold.

The deploy metric is domain-restricted -- wall and off-wall are scored separately and
`docs/PHASE9_ML.md` 0 already reports a wall-specialised ensemble as a legitimate reading.
So "which model" is a per-domain choice, and the two ensembles available disagree sharply
about which domain they are good at (strictly nested, final time):

    cv5a,cv5b,cv5c   (v3 features)   wall 0.9024   off 0.7075
    v4a,v4b,v4c      (v4 features)   wall 0.9138   off 0.6358

Reading the best cell of each column off that table is selection on the test set.  This
makes it honest: for each held-out fold, the arm is chosen **per domain** on the out-of-fold
scores of the vessels outside that fold, together with its readout family and thresholds.
The `oracle` row reports what a perfect per-domain arm choice would have given, so the cost
of having to choose is visible rather than assumed away.

    python scripts/eval_multiarm.py --arms cv5a,cv5b,cv5c v4a,v4b,v4c --cache gt
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

from eval_strict import FAMILIES, GRID, load_scores  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="each arm is a comma-separated tag list")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    oofs, pool, folds = {}, None, None
    for arm in args.arms:
        p, f, sc = load_scores(arm.split(","))
        pool = [a for a in p if a in cache] if pool is None else pool
        folds = f if folds is None else folds
        fold_of = {a: k for k, held in f.items() for a in held}
        oofs[arm] = {a: sc[(fold_of[a], a)] for a in pool}
    classes = classes_for(pool, PACKS)
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    doms = {"wall": lambda S: S["wall"], "off": lambda S: ~S["wall"]}

    rows = {a: {} for a in pool}
    per_arm = {arm: {a: {} for a in pool} for arm in args.arms}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        for dk, dom_of in doms.items():
            best = None
            for arm in args.arms:
                sub = {a: oofs[arm][a] for a in sel}
                for fam, (tune, apply_) in FAMILIES.items():
                    th = tune(cache, vs, sel, sub, GRID)
                    vals = []
                    for a in sel:
                        S = cache[a]
                        d = dom_of(S)
                        x = vs[a].score(apply_(S, sub[a], th) & d, d)
                        if x == x:
                            vals.append(x)
                    q = float(np.mean(vals)) if vals else -1e9
                    if best is None or q > best[0]:
                        best = (q, arm, fam, th)
                    for a in held:
                        S = cache[a]
                        d = dom_of(S)
                        cur = per_arm[arm][a].get(dk, -1e9)
                        # per-arm row uses that arm's own best family, chosen on `sel`
                        if q > per_arm[arm][a].get(dk + "_q", -1e9):
                            per_arm[arm][a][dk + "_q"] = q
                            per_arm[arm][a][dk] = vs[a].score(
                                apply_(S, oofs[arm][a], th) & d, d)
                        del cur
            _, arm, fam, th = best
            for a in held:
                S = cache[a]
                d = dom_of(S)
                rows[a][dk] = vs[a].score(FAMILIES[fam][1](S, oofs[arm][a], th) & d, d)
                rows[a][dk + "_arm"] = arm
            print("  fold %d  %-4s <- %-22s %s" % (k, dk, arm, fam), flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]

    def line(name, get):
        print("%-26s | %9.4f %9.4f | %9.4f %9.4f"
              % (name, np.nanmean([get(a, "wall") for a in pool]),
                 np.nanmean([get(a, "off") for a in pool]),
                 np.nanmean([get(a, "wall") for a in prio]),
                 np.nanmean([get(a, "off") for a in prio])))

    print("\nFINAL TIME POINT, strictly nested, per-domain arm choice\n")
    print("%-26s | %9s %9s | %9s %9s" % ("arm", "wall", "off", "P wall", "P off"))
    for arm in args.arms:
        line(arm[:26], lambda a, d, m=per_arm[arm]: m[a][d])
    line("nested per-domain pick", lambda a, d: rows[a][d])
    line("ORACLE per-domain pick",
         lambda a, d: max(per_arm[arm][a][d] for arm in args.arms)
         if all(per_arm[arm][a][d] == per_arm[arm][a][d] for arm in args.arms)
         else float("nan"))
    print("\nchoices: " + ", ".join(
        "%s wall<-%s off<-%s" % (a[-3:], rows[a]["wall_arm"][:12], rows[a]["off_arm"][:12])
        for a in sorted(pool)[:6]) + " ...")
    if args.save:
        Path(args.save).write_text(json.dumps(dict(rows=rows, per_arm=per_arm), indent=2,
                                              default=float))
        print("wrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
