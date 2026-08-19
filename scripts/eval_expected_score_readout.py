"""Choose the mask that maximises the EXPECTED severity score, instead of thresholding it.

Every readout this project has used answers "which nodes score above a cut".  That is a
per-node question, and the metric is not per-node: it is
`0.5*dilation_IoU + 0.5*F_0.5(precision_eff, recall_eff)`, computed over a whole domain,
with a burden-dependent grace.  Whether the 40th-ranked node is worth committing depends on
how many are already committed and on how confident the rest are -- which a fixed cut cannot
express, and which is exactly why `scripts/diag_readout_ceiling.py` finds +0.042 wall /
+0.120 off-wall sitting between a cohort cut and a per-vessel oracle cut.

This asks the decision-theoretic question instead.  Treat the model's probabilities `p` as a
distribution over the unknown truth; for each prefix of the score-ranked node list, compute
the **expected** severity score of committing that prefix, using `soft_severity` with `p` in
the place of GT; commit the prefix that maximises it.  The stopping point is then a property
of this vessel's own confidence profile, needs no label, and adapts the budget automatically.

`src/clot_ml/calibration.py`'s rules all failed because they locate a cut from the *shape* of
a unitless score.  This does not locate a cut at all -- it evaluates the objective.

Two cohort-level corrections are offered, both fitted in-fold, because `p` is known to be
miscalibrated (`scripts/eval_reg_readout.py`: the regression head's physical anchor lands far
from `crit`, so the classifier's probabilities are not calibrated either):

    gamma   sharpen/flatten the probabilities, ``p -> p**gamma``, before taking expectations
    kscale  multiply the chosen prefix length, ``k -> kscale * k``

    python scripts/eval_expected_score_readout.py --tags v5a,v5b,v5c --cache v5
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

from eval_strict import (  # noqa: E402
    FAMILIES, GRID, apply_adapt, load_scores, tune_adapt,
)
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer, soft_severity  # noqa: E402
from src.clot_ml.softmetric import dilation_operator, soft_dilate, to_torch_sparse  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
GAMMA = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
KSCALE = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
N_PREFIX = 40          # log-spaced prefix lengths evaluated per vessel/domain


def expected_curve(sc, dom, D_t, dev, gamma):
    """-> (ks, expected score at each prefix length) for one vessel/domain."""
    d = torch.tensor(np.asarray(dom, np.float32), device=dev)
    p_raw = np.clip(np.asarray(sc, np.float64), 1e-6, 1 - 1e-6) ** gamma
    p = torch.tensor(p_raw.astype(np.float32), device=dev)
    gt_dil = soft_dilate(p * d, D_t).detach()
    idx = np.argsort(-np.asarray(sc)[np.asarray(dom, bool)])
    order = np.flatnonzero(np.asarray(dom, bool))[idx]
    n = len(order)
    if n < 4:
        return np.array([0]), np.array([0.0])
    ks = np.unique(np.clip(np.geomspace(1, n, N_PREFIX).astype(int), 1, n))
    vals = []
    for k in ks:
        m = np.zeros(len(sc), np.float32)
        m[order[:k]] = 1.0
        v = soft_severity(torch.tensor(m, device=dev), p, D_t, d, gt_dil, DEFAULT)
        vals.append(float(v) if v is not None else -1e9)
    return ks, np.asarray(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--save", default="")
    ap.add_argument("--save-masks", default="",
                    help="npz of the nested-pick committed mask per vessel, for "
                         "scripts/eval_strict_temporal.py --set-masks")
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
    doms = {"wall": lambda S: S["wall"], "off": lambda S: ~S["wall"]}

    # precompute the expected-score curves once per (vessel, domain, gamma)
    print("[i] building expected-score curves ...", flush=True)
    curves = {}
    for a in pool:
        for dk, d_of in doms.items():
            for g in GAMMA:
                curves[(a, dk, g)] = expected_curve(sc[a], d_of(cache[a]), Dt[a], dev, g)
    print("[i] done", flush=True)

    def mask_for(a, dk, d_of, g, ks_scale):
        ks, vals = curves[(a, dk, g)]
        if len(ks) < 2:
            return np.zeros(len(sc[a]), bool)
        k = int(np.clip(round(ks[int(np.argmax(vals))] * ks_scale), 1, ks[-1]))
        d = d_of(cache[a])
        order = np.flatnonzero(d)[np.argsort(-sc[a][d])]
        m = np.zeros(len(sc[a]), bool)
        m[order[:k]] = True
        return m

    ARMS = ["cohort_cut", "expected", "expected_tuned", "resid", "resid_adapt",
            "nested_pick"]
    rows = {r: {a: {} for a in pool} for r in ARMS}
    masks = {a: np.zeros(len(sc[a]), bool) for a in pool}
    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        for dk, d_of in doms.items():
            # control: one cohort cut
            top, t_cut = -1e9, float(GRID[0])
            for t in GRID:
                v = [vs[a].score(d_of(cache[a]) & (sc[a] >= t), d_of(cache[a])) for a in sel]
                v = [x for x in v if x == x]
                if v and np.mean(v) > top:
                    top, t_cut = float(np.mean(v)), float(t)
            # expected-score readout, gamma and kscale fitted on the selection vessels
            best = None
            for g in GAMMA:
                for kscl in KSCALE:
                    v = []
                    for a in sel:
                        x = vs[a].score(mask_for(a, dk, d_of, g, kscl), d_of(cache[a]))
                        if x == x:
                            v.append(x)
                    q = float(np.mean(v)) if v else -1e9
                    if best is None or q > best[0]:
                        best = (q, g, kscl)
            _, g_b, k_b = best
            # the physics-conditioned readout, and its adaptive perturbation
            sub = {a: sc[a] for a in sel}
            th_r = FAMILIES["resid"][0](cache, vs, sel, sub, GRID)
            b_r, med_r = tune_adapt(cache, vs, sel, sub, "resid", th_r, d_of)

            def q_of(fn):
                v = [x for x in (fn(a) for a in sel) if x == x]
                return float(np.mean(v)) if v else -1e9

            cands = {
                "cohort_cut": (q_of(lambda a: vs[a].score(
                    d_of(cache[a]) & (sc[a] >= t_cut), d_of(cache[a]))),
                    lambda a: d_of(cache[a]) & (sc[a] >= t_cut)),
                "expected_tuned": (best[0], lambda a: mask_for(a, dk, d_of, g_b, k_b)),
                "resid": (q_of(lambda a: vs[a].score(
                    FAMILIES["resid"][1](cache[a], sc[a], th_r) & d_of(cache[a]),
                    d_of(cache[a]))),
                    lambda a: FAMILIES["resid"][1](cache[a], sc[a], th_r) & d_of(cache[a])),
                "resid_adapt": (q_of(lambda a: vs[a].score(
                    apply_adapt(cache[a], sc[a], "resid", th_r, d_of, b_r, med_r)
                    & d_of(cache[a]), d_of(cache[a]))),
                    lambda a: apply_adapt(cache[a], sc[a], "resid", th_r, d_of, b_r, med_r)
                    & d_of(cache[a])),
            }
            pick = max(cands, key=lambda r: cands[r][0])
            for a in held:
                d = d_of(cache[a])
                rows["cohort_cut"][a][dk] = vs[a].score(cands["cohort_cut"][1](a), d)
                rows["expected"][a][dk] = vs[a].score(mask_for(a, dk, d_of, 1.0, 1.0), d)
                rows["expected_tuned"][a][dk] = vs[a].score(
                    cands["expected_tuned"][1](a), d)
                rows["resid"][a][dk] = vs[a].score(cands["resid"][1](a), d)
                rows["resid_adapt"][a][dk] = vs[a].score(cands["resid_adapt"][1](a), d)
                m_pick = cands[pick][1](a)
                rows["nested_pick"][a][dk] = vs[a].score(m_pick, d)
                masks[a] |= m_pick
            print("  fold %d %-4s cut=%.2f gamma=%.2f kscale=%.2f  pick=%s"
                  % (k, dk, t_cut, g_b, k_b, pick), flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\nFINAL TIME POINT, strictly nested (tags=%s)\n" % args.tags)
    print("%-16s | %9s %9s | %9s %9s" % ("arm", "wall", "off", "P wall", "P off"))
    for r in ARMS:
        R = rows[r]
        print("%-16s | %9.4f %9.4f | %9.4f %9.4f"
              % (r, np.nanmean([R[a]["wall"] for a in pool]),
                 np.nanmean([R[a]["off"] for a in pool]),
                 np.nanmean([R[a]["wall"] for a in prio]),
                 np.nanmean([R[a]["off"] for a in prio])))
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2, default=float))
        print("\nwrote %s" % args.save)
    if args.save_masks:
        np.savez_compressed(args.save_masks, **{a: masks[a] for a in pool})
        print("wrote %s" % args.save_masks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
