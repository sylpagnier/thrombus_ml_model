"""A dedicated classifier for the ONE decision off-wall actually turns on.

WHY THIS, AND WHY IT LOOKED IMPOSSIBLE FIRST.  Restricting to off-wall nodes on the first
shell whose owner is committed, the population is 1339 nodes at **38.8% positive** -- not the
0.7% of the full-mesh problem.  Every *single* feature is nearly useless there (best
univariate within-vessel AUC 0.72, and the top three are mesh-structural), which reads like
an information wall.  It is not: a gradient-boosted model on the same channels, fitted
leave-one-vessel-out, reaches

    MESH channels only     (10)   AUC 0.714
    PHYSICS channels only  (59)   AUC 0.896
    ALL                    (69)   AUC 0.902

The decision is jointly determined and it is **physical** -- physics-only is within 0.006 of
everything.  The univariate view was simply the wrong lens.

The GNN never had a chance to learn this: it is trained on the whole mesh against a 0.7%
base rate and its off-wall score is a by-product of a full-mesh objective.  This head is
trained on exactly the population the readout has to discriminate, and nothing else.

The population is defined with the **predicted** wall set, not GT, so it is deployable.
Every threshold is chosen on an inner split of the selection vessels -- fitting a model on
the selection vessels and then tuning its cut on those same vessels reads 0.897 selection
against 0.40 held out, which is how the previous version of this idea was lost.

    python scripts/eval_offwall_shellhead.py --tags v5a,v5b,v5c --cache v5
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
HEAD = dict(max_iter=250, max_depth=4, learning_rate=0.06, l2_regularization=1.0,
            class_weight="balanced")
P_GRID = np.round(np.linspace(0.05, 0.95, 37), 4)


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
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    wall_of = (lambda S_: S_["wall"])

    # POPULATIONS.  `shell` is the topological first row; `near` drops that requirement and
    # keeps only "owner is committed", so the cost of the shell restriction is visible.
    def population(a, wset, use_shell):
        S = cache[a]
        p = (~S["wall"]) & wset[a][S["owner"]]
        return p & S["shell"].astype(bool) if use_shell else p

    def feats(a, p):
        S = cache[a]
        return np.concatenate([S["X"][p], sc[a][p][:, None],
                               sc[a][S["owner"]][p][:, None]], axis=1)

    ARMS = ["shell_head", "near_head", "shell_head_oracle", "per_node_ref"]
    rows = {r: {} for r in ARMS}
    masks = {a: np.zeros(len(sc[a]), bool) for a in pool}

    for kf, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        sub = {a: sc[a] for a in sel}
        th_w = FAMILIES["resid"][0](cache, vs, sel, sub, GRID)
        b_w, med_w = tune_adapt(cache, vs, sel, sub, "resid", th_w, wall_of)
        wset = {a: apply_adapt(cache[a], sc[a], "resid", th_w, wall_of, b_w, med_w)
                & cache[a]["wall"] for a in pool}

        for arm, use_shell in (("shell_head", True), ("near_head", False)):
            P_, POP = {}, {a: population(a, wset, use_shell) for a in pool}

            def fit_on(anchors):
                X = np.concatenate([feats(a, POP[a]) for a in anchors if POP[a].any()])
                y = np.concatenate([(cache[a]["y"] > 0.5)[POP[a]]
                                    for a in anchors if POP[a].any()])
                if y.sum() < 5 or (~y).sum() < 5:
                    return None
                return [HistGradientBoostingClassifier(random_state=s, **HEAD).fit(X, y)
                        for s in range(max(args.seeds, 1))]

            def pred(models, a):
                out = np.zeros(len(sc[a]))
                if models is None or not POP[a].any():
                    return out
                out[POP[a]] = np.mean(
                    [m.predict_proba(feats(a, POP[a]))[:, 1] for m in models], axis=0)
                return out

            inner = [sel[i::3] for i in range(3)]
            for iv in inner:
                m_i = fit_on([a for a in sel if a not in iv])
                for a in iv:
                    P_[a] = pred(m_i, a)
            m_k = fit_on(sel)
            for a in held:
                P_[a] = pred(m_k, a)

            best = None
            for t in P_GRID:
                v = [x for x in (vs[a].score(POP[a] & (P_[a] >= t), ~cache[a]["wall"])
                                 for a in sel) if x == x]
                q = float(np.mean(v)) if v else -1e9
                if best is None or q > best[0]:
                    best = (q, float(t))
            for a in held:
                d = ~cache[a]["wall"]
                m = POP[a] & (P_[a] >= best[1])
                rows[arm][a] = vs[a].score(m, d)
                if arm == "shell_head":
                    masks[a] = m
                    rows["shell_head_oracle"][a] = vs[a].score(
                        POP[a] & (cache[a]["y"] > 0.5), d)
            print("  fold %d %-11s t=%.2f sel %.3f" % (kf, arm, best[1], best[0]), flush=True)

    ref_p = REPO / "outputs/offwall_final.json"
    if ref_p.exists():
        ref = json.loads(ref_p.read_text())["base"]
        for a in pool:
            if a in ref and ref[a] is not None:
                rows["per_node_ref"][a] = ref[a]

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    base = [a for a in pool if a not in prio]
    print("\nFINAL-TIME OFF-WALL, strictly nested (tags=%s)\n" % args.tags)
    print("%-20s | %9s %9s %9s" % ("arm", "ALL", "baseline", "PRIORITY"))
    for n in ARMS:
        R = {k: v for k, v in rows[n].items() if v == v}
        if not R:
            continue
        print("%-20s | %9.4f %9.4f %9.4f"
              % (n, np.nanmean([R[a] for a in pool if a in R]),
                 np.nanmean([R[a] for a in base if a in R]),
                 np.nanmean([R[a] for a in prio if a in R])))
    print("\nper vessel: per-node -> shell_head")
    for a in sorted(pool):
        if a not in rows["shell_head"] or rows["shell_head"][a] != rows["shell_head"][a]:
            continue
        print("   %-11s %.4f -> %.4f"
              % (a, rows["per_node_ref"].get(a, float("nan")), rows["shell_head"][a]))
    if args.save_masks:
        np.savez_compressed(args.save_masks, **{a: masks[a] for a in pool})
        print("\nwrote %s" % args.save_masks)
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
