"""Off-wall as a WALL-INDEXED companion decision, not a per-node classification.

THE MEASUREMENT THAT FORCES THIS.  `scripts/diag_offwall_structure.py` and the band-count
sweep show that the number of off-wall GT nodes owned by a single wall node is **0 or 1**,
essentially always:

    vessel   n_w=0   n_w=1   n_w=2   n_w>=3
    p032        73     120       0        0
    p012        31      52       7        6
    p020        97      13       0        0
    p044        63      90       4        6

So off-wall clot is not a cloud to be thresholded -- it is **one companion node hanging off
some of the committed wall nodes**, which is exactly PHASE7 3.1's "a thin band of finite
physical thickness that the boundary-layer mesh resolves in ~2 rows".

Three things follow, and they are why this formulation is better posed than the one the
project has used since PHASE9:

1. **The ceiling is higher.**  Painting each wall node's `n_w` nearest owned off-wall nodes
   with ORACLE `n_w` scores **0.9014**, against 0.8185 for the best possible prefix of the
   per-node ranking.  The structure is worth +0.08 of ceiling on its own.
2. **The base rate is workable.**  Among committed wall nodes the label is near-balanced
   (p032: 120 of 193; p012: 52 of 83) instead of the 1.5% of the per-node off-wall problem,
   and the decision count drops from ~5000 nodes to ~150.
3. **It cannot spray.**  At most one companion per wall node is structurally enforced, which
   is the precision failure that dominates the low-burden vessels (`p005` commits 17 nodes
   for 4 GT under the per-node readout).

Nothing here is a new feature.  The same 69 channels are used, read on the wall node and on
its companion; what changes is the object being decided.

    python scripts/eval_offwall_companion.py --tags v5a,v5b,v5c --cache v5
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

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from eval_strict import FAMILIES, GRID, apply_adapt, load_scores, tune_adapt  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
HEAD = dict(max_iter=300, max_depth=4, learning_rate=0.06, l2_regularization=1.0,
            class_weight="balanced")
P_GRID = np.round(np.linspace(0.05, 0.95, 37), 4)


def companions(S, cols):
    """For every wall node, its nearest owned off-wall node (-1 if it owns none).

    Ties are broken by distance to the wall, so the companion is the innermost row -- the
    row PHASE7 3.1 measured off-wall GT to live on.
    """
    n = len(S["wall"])
    dist = S["X"][:, cols.index("dist_wall_edges")]
    comp = np.full(n, -1, dtype=int)
    best = np.full(n, np.inf)
    off = np.flatnonzero(~S["wall"])
    for i in off:
        w = S["owner"][i]
        if dist[i] < best[w]:
            best[w] = dist[i]
            comp[w] = i
    return comp


def build(S, sc, cols, comp):
    """Per-wall-node design row: the wall node's channels, its companion's, and the scores."""
    w = np.flatnonzero(S["wall"] & (comp >= 0))
    c = comp[w]
    X = np.concatenate([S["X"][w], S["X"][c],
                        sc[w][:, None], sc[c][:, None],
                        (sc[c] - sc[w])[:, None]], axis=1)
    return w, c, X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--save-masks", default="")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc_all = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    fo = {a: k for k, held in folds.items() for a in held}
    sc = {a: sc_all[(fo[a], a)] for a in pool}
    cols = [str(c) for c in cache[pool[0]]["cols"]]
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}

    D = {}
    for a in pool:
        S = cache[a]
        comp = companions(S, cols)
        w, c, X = build(S, sc[a], cols, comp)
        gt = S["y"] > 0.5
        # LABEL: does this wall node own ANY off-wall GT clot -- NOT "is my nearest owned
        # node the GT one".  Only 23.6% of GT off-wall nodes are their owner's strict
        # nearest, so the latter label is false even where the wall node does carry a
        # companion, and it caps the construction at 0.5175.  The count is what matters:
        # the severity metric's 2-hop grace forgives painting an adjacent node, which is why
        # the free-thickness construction reaches 0.9014 using plain distance ordering.
        nw = np.zeros(len(S["wall"]), dtype=int)
        np.add.at(nw, S["owner"][gt & ~S["wall"]], 1)
        D[a] = dict(S=S, comp=comp, w=w, c=c, X=X, y=(nw[w] >= 1), gt=gt, nw=nw)

    ARMS = ["companion", "companion_gatedwall", "oracle_label", "per_node_ref"]
    rows = {r: {} for r in ARMS}
    masks = {a: np.zeros(len(sc[a]), bool) for a in pool}

    for kf, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]

        def fit_on(anchors):
            Xtr = np.concatenate([D[a]["X"] for a in anchors])
            ytr = np.concatenate([D[a]["y"] for a in anchors])
            return [HistGradientBoostingClassifier(random_state=s, **HEAD).fit(Xtr, ytr)
                    for s in range(max(args.seeds, 1))]

        def prob_with(models, a):
            return np.mean([m.predict_proba(D[a]["X"])[:, 1] for m in models], axis=0)

        # INNER SPLIT.  Tuning the commit threshold on the same vessels the classifier was
        # fitted on reads 0.897 on the selection set and 0.40 held out: in-sample
        # probabilities are far too confident, the tuner picks ~0.93, and on real
        # out-of-fold probabilities that commits almost nothing.  Every model fitted inside
        # the selection loop needs its own inner split, not just the final one.
        inner = [sel[i::3] for i in range(3)]
        P = {}
        for iv in inner:
            itr = [a for a in sel if a not in iv]
            m_i = fit_on(itr)
            for a in iv:
                P[a] = prob_with(m_i, a)
        ms = fit_on(sel)
        for a in held:
            P[a] = prob_with(ms, a)

        def prob(a):
            return P[a]

        # the shipped WALL set, used to gate which wall nodes may have a companion at all
        sub = {a: sc[a] for a in sel}
        th_w = FAMILIES["resid"][0](cache, vs, sel, sub, GRID)
        b_w, med_w = tune_adapt(cache, vs, sel, sub, "resid", th_w,
                                lambda S_: S_["wall"], )
        wset = {a: apply_adapt(cache[a], sc[a], "resid", th_w, lambda S_: S_["wall"],
                               b_w, med_w) & cache[a]["wall"] for a in pool}

        def mask_of(a, p, t, gate):
            S = D[a]["S"]
            m = np.zeros(len(S["wall"]), bool)
            keep = p >= t
            if gate:
                keep = keep & wset[a][D[a]["w"]]
            m[D[a]["c"][keep]] = True
            return m & ~S["wall"]

        P = {a: prob(a) for a in pool}
        picked = {}
        for gate in (False, True):
            best = None
            for t in P_GRID:
                v = [x for x in (vs[a].score(mask_of(a, P[a], t, gate), ~cache[a]["wall"])
                                 for a in sel) if x == x]
                q = float(np.mean(v)) if v else -1e9
                if best is None or q > best[0]:
                    best = (q, float(t))
            picked[gate] = best
        for a in held:
            d = ~cache[a]["wall"]
            # ceiling of THIS construction: perfect n_w>=1 labels, same painting
            rows["oracle_label"][a] = vs[a].score(
                mask_of(a, D[a]["y"].astype(float), 0.5, False), d)
            rows["companion"][a] = vs[a].score(mask_of(a, P[a], picked[False][1], False), d)
            rows["companion_gatedwall"][a] = vs[a].score(
                mask_of(a, P[a], picked[True][1], True), d)
            masks[a] = mask_of(a, P[a], picked[False][1], False)
        print("  fold %d  t(free)=%.2f sel %.3f | t(gated)=%.2f sel %.3f"
              % (kf, picked[False][1], picked[False][0],
                 picked[True][1], picked[True][0]), flush=True)

    ref = json.loads((REPO / "outputs/offwall_final.json").read_text())["base"] \
        if (REPO / "outputs/offwall_final.json").exists() else {}
    for a in pool:
        if a in ref and ref[a] is not None:
            rows["per_node_ref"][a] = ref[a]

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    base = [a for a in pool if a not in prio]
    print("\nFINAL-TIME OFF-WALL, strictly nested (tags=%s)\n" % args.tags)
    print("%-22s | %9s %9s %9s" % ("arm", "ALL", "baseline", "PRIORITY"))
    for n in ARMS:
        R = {k: v for k, v in rows[n].items() if v == v}
        if not R:
            continue
        print("%-22s | %9.4f %9.4f %9.4f"
              % (n, np.nanmean([R[a] for a in pool if a in R]),
                 np.nanmean([R[a] for a in base if a in R]),
                 np.nanmean([R[a] for a in prio if a in R])))
    print("\nper vessel: per-node -> companion")
    for a in sorted(pool):
        if a not in rows["companion"] or rows["companion"][a] != rows["companion"][a]:
            continue
        print("   %-11s %.4f -> %.4f"
              % (a, rows["per_node_ref"].get(a, float("nan")), rows["companion"][a]))
    if args.save_masks:
        np.savez_compressed(args.save_masks, **{a: masks[a] for a in pool})
        print("\nwrote %s" % args.save_masks)
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
