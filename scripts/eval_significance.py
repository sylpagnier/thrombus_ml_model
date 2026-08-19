"""How big does a difference have to be, on 19 vessels, before it means anything?

Every table in `docs/PHASE9_ML.md` compares cohort means over 19 vessels, several of which
carry fewer than 15 off-wall GT nodes, and differences of 0.01-0.03 are routinely read as
wins.  Nothing in the project has ever put an interval on one.  This does, two ways:

  * **paired bootstrap over vessels** -- resample the 19 vessels with replacement and
    recompute the paired difference, which is the right unit because both arms are scored
    on the same vessels;
  * **the seed floor** -- the spread between individual configurations of the *same* arm,
    which bounds from below what any feature or readout change has to beat.

    python scripts/eval_significance.py --a cv5a,cv5b,cv5c --b v4a,v4b --cache gt
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
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402


def nested_rows(cache, tags):
    """Per-vessel held-out (wall, off) under the strict protocol of eval_strict.py."""
    pool, folds, sc = load_scores(tags)
    pool = [a for a in pool if a in cache]
    fold_of = {a: k for k, held in folds.items() for a in held}
    oof = {a: sc[(fold_of[a], a)] for a in pool}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    out = {}
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
        _, fam, th = best
        for a in held:
            S = cache[a]
            pr = FAMILIES[fam][1](S, oof[a], th)
            out[a] = (vs[a].score(pr & S["wall"], S["wall"]),
                      vs[a].score(pr & ~S["wall"], ~S["wall"]))
    return pool, out


def boot(pool, A, B, i, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    keep = [a for a in pool if A[a][i] == A[a][i] and B[a][i] == B[a][i]]
    d = np.array([B[a][i] - A[a][i] for a in keep])
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), \
        float((bs <= 0).mean()), len(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline tags, comma separated")
    ap.add_argument("--b", required=True, help="candidate tags")
    ap.add_argument("--cache", default="gt")
    ap.add_argument("--floor", default="cv5a,cv5b,cv5c",
                    help="single tags of the SAME arm, to bound the seed/config floor")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    pool, A = nested_rows(cache, args.a.split(","))
    _, B = nested_rows(cache, args.b.split(","))

    print("\nPAIRED DIFFERENCE  (%s)  ->  (%s),  n=19 vessels\n" % (args.a, args.b))
    print("%-6s %8s %8s %18s %10s %4s" % ("dom", "base", "cand", "diff [95% CI]", "P(<=0)", "n"))
    for i, dom in ((0, "wall"), (1, "off")):
        m, lo, hi, p, n = boot(pool, A, B, i)
        ba = np.nanmean([A[a][i] for a in pool])
        bb = np.nanmean([B[a][i] for a in pool])
        print("%-6s %8.4f %8.4f  %+.4f [%+.4f,%+.4f] %8.3f %4d"
              % (dom, ba, bb, m, lo, hi, p, n))

    floors = args.floor.split(",")
    if len(floors) > 1:
        print("\nSEED / CONFIG FLOOR -- single members of one arm, same protocol\n")
        rs = {}
        for t in floors:
            _, r = nested_rows(cache, [t])
            rs[t] = r
            print("  %-8s wall %.4f  off %.4f"
                  % (t, np.nanmean([r[a][0] for a in pool]),
                     np.nanmean([r[a][1] for a in pool])))
        for i, dom in ((0, "wall"), (1, "off")):
            v = [np.nanmean([rs[t][a][i] for a in pool]) for t in floors]
            print("  %-6s spread %.4f (min %.4f max %.4f)"
                  % (dom, max(v) - min(v), min(v), max(v)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
