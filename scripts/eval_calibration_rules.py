"""Which per-vessel calibration rule closes the readout gap? -- strictly nested.

`scripts/diag_readout_ceiling.py` measured +0.042 wall / +0.120 off-wall sitting in the
threshold rather than the model.  `src/clot_ml/calibration.py` proposes five rules for
spending it.  This picks between them under the same nested protocol as
`scripts/eval_strict.py`: the rule AND its parameter are selected per domain on the
out-of-fold scores of the vessels outside the held-out fold, so the choice never sees the
vessel it is scored on.

    python scripts/eval_calibration_rules.py --tags cv5a,cv5b,cv5c --cache gt
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

from eval_strict import load_scores  # noqa: E402
from src.clot_ml.calibration import RULES, apply_rule, rule_grid  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def best_param(cache, vs, anchors, oof, dom_of, rule):
    top, pick = -1e9, float(rule_grid(rule)[0])
    for p in rule_grid(rule):
        vals = []
        for a in anchors:
            S = cache[a]
            d = dom_of(S)
            x = vs[a].score(apply_rule(rule, oof[a], d, S["phys_mask"], p), d)
            if x == x:
                vals.append(x)
        if vals and np.mean(vals) > top:
            top, pick = float(np.mean(vals)), float(p)
    return pick, top


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    fold_of = {a: k for k, held in folds.items() for a in held}
    oof = {a: sc[(fold_of[a], a)] for a in pool}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    doms = {"wall": lambda S: S["wall"], "off": lambda S: ~S["wall"]}

    # per-rule held-out rows, plus the nested "let the fold choose the rule too" arm
    rows = {r: {} for r in RULES}
    rows["nested_pick"] = {}
    picks = {}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        sub = {a: oof[a] for a in sel}
        chosen = {}
        for dk, dom_of in doms.items():
            best = None
            for r in RULES:
                p, q = best_param(cache, vs, sel, sub, dom_of, r)
                if best is None or q > best[1]:
                    best = (r, q, p)
                for a in held:
                    S = cache[a]
                    d = dom_of(S)
                    rows[r].setdefault(a, {})[dk] = vs[a].score(
                        apply_rule(r, oof[a], d, S["phys_mask"], p), d)
            chosen[dk] = (best[0], best[2])
            for a in held:
                S = cache[a]
                d = dom_of(S)
                rows["nested_pick"].setdefault(a, {})[dk] = vs[a].score(
                    apply_rule(best[0], oof[a], d, S["phys_mask"], best[2]), d)
        picks[k] = chosen
        print("  fold %d  wall<-%-13s p=%.4f   off<-%-13s p=%.4f"
              % (k, chosen["wall"][0], chosen["wall"][1],
                 chosen["off"][0], chosen["off"][1]), flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\nFINAL TIME POINT, strictly nested (tags=%s, cache=%s)\n" % (args.tags, args.cache))
    print("%-14s | %9s %9s | %9s %9s" % ("rule", "wall", "off", "P wall", "P off"))
    order = list(RULES) + ["nested_pick"]
    for r in order:
        R = rows[r]
        print("%-14s | %9.4f %9.4f | %9.4f %9.4f"
              % (r,
                 np.nanmean([R[a]["wall"] for a in pool]),
                 np.nanmean([R[a]["off"] for a in pool]),
                 np.nanmean([R[a]["wall"] for a in prio]),
                 np.nanmean([R[a]["off"] for a in prio])))

    best_r = max(RULES, key=lambda r: np.nanmean([rows[r][a]["wall"] for a in pool])
                 + np.nanmean([rows[r][a]["off"] for a in pool]))
    print("\nper vessel: absolute -> %s" % best_r)
    for a in sorted(pool):
        A, B = rows["absolute"][a], rows[best_r][a]
        print("  %-11s wall %.4f -> %.4f   off %s -> %s"
              % (a, A["wall"], B["wall"],
                 ("%.4f" % A["off"]) if A["off"] == A["off"] else "  n/a",
                 ("%.4f" % B["off"]) if B["off"] == B["off"] else "  n/a"))
    if args.save:
        Path(args.save).write_text(json.dumps(
            dict(rows=rows, picks={str(k): v for k, v in picks.items()}), indent=2,
            default=float))
        print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
