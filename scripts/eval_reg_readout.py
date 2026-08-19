"""Read the mask off the REGRESSION head, whose threshold is fixed by physics.

THE PROBLEM THIS ATTACKS.  `scripts/diag_readout_ceiling.py` measured +0.042 wall / +0.120
off-wall sitting between the cohort-wide cut and a per-vessel oracle cut on the *same*
score field, and `scripts/eval_calibration_rules.py` then showed that no unsupervised rule
built from the score's own distribution recovers it (quantile 0.78/0.53, physics-anchored
0.87/0.54, largest-gap 0.88/0.51, all against the absolute cut's 0.907/0.688).  Those rules
all try to locate a cut from the *shape* of a quantity that has no units.

The regression head has units.  PHASE7 10.1 established that GT clot **is** `{Mat >= crit}`
-- 0.0% of clot below the platelet gelation step, 0.19% of high-`Mat` nodes not clot, wall
and off-wall alike.  `src/clot_ml/gnn.py` regresses `log1p(Mat/crit)` with the physics
backbone as a zero-init additive base.  So the label is a threshold on the thing that head
predicts, and that threshold is

    Mat >= crit   <=>   log1p(Mat/crit) >= log 2 = 0.6931

with **no free parameter and no cohort constant** -- it is per-vessel by construction,
because it is a statement about a physical quantity rather than about a rank.  The head was
trained on every run in this project and never read out (`docs/PHASE9_ML.md` 13.1 measured
it ranking GT `Mat` at 0.619 against the classifier's 0.601 and then used the classifier).

Arms, all strictly nested where they have anything to select:

    cls            classifier, cohort absolute cut                     the control
    reg_phys       `reg >= log 2`                                      ZERO parameters
    reg_tuned      `reg >= t`, t fitted per domain in-fold             is the anchor right?
    reg_budget     classifier cut placed per vessel so that it commits the same COUNT the
                   physical anchor does -- uses the classifier's ranking, which is better,
                   with the regression head's units, which are the part that transfers
    union / inter  the two masks combined

    python scripts/eval_reg_readout.py --tags v4b --cache v4
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
CRIT_LOG = float(np.log(2.0))          # log1p(Mat/crit) at Mat == crit
CLS_GRID = np.round(np.linspace(0.02, 0.98, 33), 4)
REG_GRID = np.round(np.linspace(0.05, 3.0, 40), 4)


def load_both(tags):
    zs = [np.load(REPO / f"outputs/phase9_scores/{t}.npz", allow_pickle=True) for t in tags]
    for t, z in zip(tags, zs):
        if not any(k.startswith("reg|") for k in z.files):
            raise SystemExit("tag %s has no regression field; rerun run_phase9_cv.py" % t)
    pool = [str(x) for x in zs[0]["pool"]]
    folds = {int(k.split("|")[1]): [str(x) for x in zs[0][k]]
             for k in zs[0].files if k.startswith("held|")}
    cls, reg = {}, {}
    for k in folds:
        for a in pool:
            cls[(k, a)] = np.mean([z["%d|%s" % (k, a)] for z in zs], axis=0)
            reg[(k, a)] = np.mean([z["reg|%d|%s" % (k, a)] for z in zs], axis=0)
    return pool, folds, cls, reg


def budget_mask(cls, reg, d):
    """Classifier ranking, count taken from the physical anchor on the regression head."""
    k = int((reg >= CRIT_LOG)[d].sum())
    v = cls[d]
    if k <= 0 or v.size == 0:
        return np.zeros_like(d)
    k = min(k, v.size)
    cut = np.sort(v)[::-1][k - 1]
    return d & (cls >= cut)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="v4")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    pool, folds, CLS, REG = load_both(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    fold_of = {a: k for k, held in folds.items() for a in held}
    cls = {a: CLS[(fold_of[a], a)] for a in pool}
    reg = {a: REG[(fold_of[a], a)] for a in pool}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    doms = {"wall": lambda S: S["wall"], "off": lambda S: ~S["wall"]}
    ARMS = ["cls", "reg_phys", "reg_tuned", "reg_budget", "union", "inter"]
    rows = {r: {} for r in ARMS}

    def tune(anchors, dom_of, grid, field):
        top, pick = -1e9, float(grid[0])
        for t in grid:
            vals = []
            for a in anchors:
                d = dom_of(cache[a])
                x = vs[a].score(d & (field[a] >= t), d)
                if x == x:
                    vals.append(x)
            if vals and np.mean(vals) > top:
                top, pick = float(np.mean(vals)), float(t)
        return pick

    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        for dk, dom_of in doms.items():
            t_cls = tune(sel, dom_of, CLS_GRID, cls)
            t_reg = tune(sel, dom_of, REG_GRID, reg)
            for a in held:
                S = cache[a]
                d = dom_of(S)
                m_cls = d & (cls[a] >= t_cls)
                m_phys = d & (reg[a] >= CRIT_LOG)
                M = {"cls": m_cls, "reg_phys": m_phys,
                     "reg_tuned": d & (reg[a] >= t_reg),
                     "reg_budget": budget_mask(cls[a], reg[a], d),
                     "union": m_cls | m_phys, "inter": m_cls & m_phys}
                for r in ARMS:
                    rows[r].setdefault(a, {})[dk] = vs[a].score(M[r], d)
        print("  fold %d done" % k, flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\nFINAL TIME POINT, strictly nested (tags=%s, cache=%s)\n" % (args.tags, args.cache))
    print("%-12s | %9s %9s | %9s %9s" % ("arm", "wall", "off", "P wall", "P off"))
    for r in ARMS:
        R = rows[r]
        print("%-12s | %9.4f %9.4f | %9.4f %9.4f"
              % (r, np.nanmean([R[a]["wall"] for a in pool]),
                 np.nanmean([R[a]["off"] for a in pool]),
                 np.nanmean([R[a]["wall"] for a in prio]),
                 np.nanmean([R[a]["off"] for a in prio])))
    print("\nper vessel: cls -> reg_budget")
    for a in sorted(pool):
        A, B = rows["cls"][a], rows["reg_budget"][a]
        print("  %-11s wall %.4f -> %.4f   off %s -> %s"
              % (a, A["wall"], B["wall"],
                 ("%.4f" % A["off"]) if A["off"] == A["off"] else "  n/a",
                 ("%.4f" % B["off"]) if B["off"] == B["off"] else "  n/a"))
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2, default=float))
        print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
