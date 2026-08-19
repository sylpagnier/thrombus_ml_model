"""Sanity check: does clot_gnn_v1 beat the zero-parameter physics backbone?

Uses the SAME canonical scoring convention as every viz in this project
(compute_clot_relaxed_metrics + clot_score_from_deploy_dict, domain-restricted to wall
and off-wall separately) rather than PHASE9_ML.md's own internal metric, so the numbers
are directly comparable to everything already shown to the user.

Only FIT + DEV vessels are touched -- clot_gnn_v1's SEALED set (patient042/043 confirmed
in docs/PHASE9_ML.md 10.1, and by exclusion everything else not in fit_anchors/dev_anchors)
is never opened here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clot_ml.data import attach_physics, load_cache
from src.clot_ml.locked import load_ensemble, predict_scores
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

FIT = ["patient005", "patient006", "patient012", "patient016", "patient018", "patient019",
       "patient020", "patient021", "patient024", "patient025", "patient028", "patient029",
       "patient032", "patient035", "patient036", "patient037"]
DEV = ["patient040", "patient041", "patient044"]
GRID = np.linspace(0.02, 0.98, 25)


def domain_score(pred, gt, edge_index, domain_mask, wall_mask):
    pred_d = torch.tensor((pred & domain_mask).astype(np.float32))
    gt_d = torch.tensor((gt & domain_mask).astype(np.float32))
    m = compute_clot_relaxed_metrics(pred_d, gt_d, edge_index, wall_mask=torch.tensor(wall_mask))
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def sweep_threshold(scores_by_a, gt_by_a, ei_by_a, wall_by_a, domain_by_a, anchors):
    best_t, best_v = 0.5, -1.0
    for t in GRID:
        vals = []
        for a in anchors:
            pred = scores_by_a[a] >= t
            vals.append(domain_score(pred, gt_by_a[a], ei_by_a[a], domain_by_a[a], wall_by_a[a]))
        v = float(np.mean(vals))
        if v > best_v:
            best_v, best_t = v, t
    return best_t, best_v


def main() -> int:
    cache = attach_physics(load_cache("gt"))
    ens = load_ensemble()
    anchors = [a for a in FIT + DEV if a in cache]

    gnn_scores, gt_by_a, ei_by_a, wall_by_a, phys_by_a = {}, {}, {}, {}, {}
    for a in anchors:
        S = cache[a]
        gnn_scores[a] = predict_scores(ens, S)
        gt_by_a[a] = S["y"] > 0.5
        wall_by_a[a] = S["wall"].astype(bool)
        phys_by_a[a] = S["phys_mask"].astype(bool)
        ei_by_a[a] = torch.tensor(S["edge_index"])
        print(f"  scored {a}", flush=True)

    wall_dom = {a: wall_by_a[a] for a in anchors}
    off_dom = {a: ~wall_by_a[a] for a in anchors}

    # threshold tuned on FIT only, exactly as PHASE9_ML.md's "thresh" readout
    t_wall, _ = sweep_threshold(gnn_scores, gt_by_a, ei_by_a, wall_by_a, wall_dom, FIT)
    t_off, _ = sweep_threshold(gnn_scores, gt_by_a, ei_by_a, wall_by_a, off_dom, FIT)
    print(f"\nFIT-tuned thresholds: wall={t_wall:.3f}  off={t_off:.3f}\n")

    def gnn_mask(a):
        s = gnn_scores[a]
        return (s >= t_wall) & wall_by_a[a] | (s >= t_off) & ~wall_by_a[a]

    rows = []
    for a in anchors:
        gnn_pred = gnn_mask(a)
        phys_pred = phys_by_a[a]
        gt = gt_by_a[a]
        r = dict(
            anchor=a, split="FIT" if a in FIT else "DEV",
            n_gt=int(gt.sum()), n_off_gt=int((gt & ~wall_by_a[a]).sum()),
            gnn_wall=domain_score(gnn_pred, gt, ei_by_a[a], wall_dom[a], wall_by_a[a]),
            gnn_off=domain_score(gnn_pred, gt, ei_by_a[a], off_dom[a], wall_by_a[a]),
            phys_wall=domain_score(phys_pred, gt, ei_by_a[a], wall_dom[a], wall_by_a[a]),
            phys_off=domain_score(phys_pred, gt, ei_by_a[a], off_dom[a], wall_by_a[a]),
        )
        rows.append(r)

    print("%12s %5s %6s %6s | %8s %8s | %8s %8s"
          % ("vessel", "split", "nGT", "nOff", "GNN wall", "GNN off", "phys wall", "phys off"))
    for r in rows:
        print("%12s %5s %6d %6d | %8.4f %8.4f | %8.4f %8.4f"
              % (r["anchor"], r["split"], r["n_gt"], r["n_off_gt"],
                 r["gnn_wall"], r["gnn_off"], r["phys_wall"], r["phys_off"]))

    for split in ("FIT", "DEV"):
        sel = [r for r in rows if r["split"] == split]
        print("\n%s (n=%d)  GNN wall %.4f  GNN off %.4f  |  phys wall %.4f  phys off %.4f"
              % (split, len(sel),
                 np.mean([r["gnn_wall"] for r in sel]), np.mean([r["gnn_off"] for r in sel]),
                 np.mean([r["phys_wall"] for r in sel]), np.mean([r["phys_off"] for r in sel])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
