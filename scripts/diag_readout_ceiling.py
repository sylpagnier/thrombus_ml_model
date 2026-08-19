"""Is the committed SET limited by the SCORE FIELD or by the THRESHOLD chosen for it?

The final-time score is exactly the quality of the committed set, and under the strict
protocol that set reads wall 0.9024 / off 0.7075.  Two very different things could cap it:

  * the score field does not separate clot from non-clot on this vessel -- nothing a
    readout can do;
  * the field separates fine but the single cohort-wide cut is in the wrong place for this
    vessel -- a calibration problem, and calibration can be attacked with quantities that
    need no labels.

`docs/PHASE9_ML.md` 4 already killed two per-vessel budget rules (physics-mask size,
confidence mass), but neither measured the ceiling first, so nobody knows how much was on
the table.  This measures it: for each vessel, the best score reachable by tuning that
vessel's own threshold against its own answer (an ORACLE, never a model), against the
strictly-nested cohort threshold actually used.

    python scripts/diag_readout_ceiling.py --tags cv5a,cv5b,cv5c --cache gt
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="gt")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    fold_of = {a: k for k, held in folds.items() for a in held}
    oof = {a: sc[(fold_of[a], a)] for a in pool}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}

    # the strictly-nested cohort readout, reproduced exactly as eval_strict picks it
    nested = {}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        best = None
        for fam, (tune, apply_) in FAMILIES.items():
            th = tune(cache, vs, sel, {a: oof[a] for a in sel}, GRID)
            vals = []
            for a in sel:
                S = cache[a]
                for d in (S["wall"], ~S["wall"]):
                    x = vs[a].score(apply_(S, oof[a], th) & d, d)
                    if x == x:
                        vals.append(x)
            q = float(np.mean(vals))
            if best is None or q > best[0]:
                best = (q, fam, th)
        for a in held:
            nested[a] = (best[1], best[2])

    rows = {}
    for a in pool:
        S = cache[a]
        fam, th = nested[a]
        pr = FAMILIES[fam][1](S, oof[a], th)
        r = {}
        for key, d in (("wall", S["wall"]), ("off", ~S["wall"])):
            r[key] = vs[a].score(pr & d, d)
            # ORACLE: this vessel's own best cut.  Plain family only -- the question is
            # "is a better THRESHOLD available", not "is a better readout family".
            top = -1e9
            for t in GRID:
                x = vs[a].score((oof[a] >= t) & d, d)
                if x == x and x > top:
                    top = x
            r[key + "_orc"] = top if top > -1e8 else float("nan")
        r["cls"] = classes.get(a, "?")
        rows[a] = r
        print("  %-11s wall %.4f -> %.4f   off %s -> %s"
              % (a, r["wall"], r["wall_orc"],
                 ("%.4f" % r["off"]) if r["off"] == r["off"] else "  n/a",
                 ("%.4f" % r["off_orc"]) if r["off_orc"] == r["off_orc"] else "  n/a"),
              flush=True)

    groups = [("ALL", pool),
              ("baseline", [a for a in pool if not is_priority(classes.get(a, ""))]),
              ("PRIORITY", [a for a in pool if is_priority(classes.get(a, ""))])]
    print("\nFINAL-TIME SET: nested cohort cut vs PER-VESSEL ORACLE cut\n")
    print("%-10s %3s | %9s %9s | %9s %9s" % ("group", "n", "wall", "wall*", "off", "off*"))
    for name, sub in groups:
        print("%-10s %3d | %9.4f %9.4f | %9.4f %9.4f"
              % (name, len(sub),
                 np.nanmean([rows[a]["wall"] for a in sub]),
                 np.nanmean([rows[a]["wall_orc"] for a in sub]),
                 np.nanmean([rows[a]["off"] for a in sub]),
                 np.nanmean([rows[a]["off_orc"] for a in sub])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
