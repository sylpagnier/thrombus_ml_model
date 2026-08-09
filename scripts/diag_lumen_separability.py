"""Is ANY exposure threshold able to separate off-wall clot from clear lumen?

The autocatalytic lumen arm bifurcates: expose_thresh 0.25 floods the lumen, 0.35 is
near-inert, 0.45 is dead.  That is either a badly-chosen grid or an intrinsic property of
the target.  This measures the exposure distribution directly, so the answer does not
depend on the search.

For each candidate lumen node, ``exposure`` = fraction of the radius-r ball that is
committed WALL clot (the state at the first growth step, when the rule must make its
decision).  Reports AUC of exposure for "is off-wall GT clot", and the best achievable
F1 over ALL thresholds -- an oracle sweep, so it upper-bounds any rule of this form.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import median_edge_length, radius_neighbors  # noqa: E402
from src.core_physics.physics_wall_model import node_positions  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")


def auc(score, lab):
    pos, neg = score[lab], score[~lab]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.concatenate([pos, neg])
    o = a.argsort()
    r = np.empty(len(a))
    r[o] = np.arange(1, len(a) + 1)
    u, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if cnt.max() > 1:
        s = np.zeros(len(u))
        np.add.at(s, inv, r)
        r = (s / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_f1(score, lab):
    best = (0.0, np.nan)
    for t in np.unique(np.round(score[score > 0], 3)):
        p = score >= t
        tp = int((p & lab).sum())
        if tp == 0:
            continue
        pr, rc = tp / max(int(p.sum()), 1), tp / max(int(lab.sum()), 1)
        f = 2 * pr * rc / max(pr + rc, 1e-9)
        if f > best[0]:
            best = (f, float(t))
    return best


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
    print("Exposure = committed-wall fraction of the radius ball, at the first growth step.")
    print("Labels: off-wall GT clot vs clear lumen.  Oracle threshold per vessel.\n")
    print("%12s %6s %7s | %7s %7s %7s | %7s %7s"
          % ("vessel", "nOff", "nLumen", "AUC1.5", "AUC2.2", "AUC3.0", "bestF1", "@thr"))
    rows = []
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = (gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu"))
              .reshape(-1).numpy() > 0.5)
        off = gt & ~wall
        if off.sum() == 0 or int(d.y.shape[0]) < 150:
            continue
        pos = node_positions(d)
        ei = d.edge_index.numpy()
        h = median_edge_length(pos, ei)
        seed = (gt & wall).astype(np.float64)           # ORACLE wall seed
        aucs = []
        expo_ref = None
        for r in (1.5, 2.2, 3.0):
            M = radius_neighbors(pos, r * h).tocsr()
            M.data[:] = 1.0
            M.setdiag(0.0)
            M.eliminate_zeros()
            ball = np.maximum(np.asarray(M.sum(axis=1)).reshape(-1), 1.0)
            expo = np.asarray(M @ seed).reshape(-1) / ball
            cand = ~wall
            aucs.append(auc(expo[cand], off[cand]))
            if r == 2.2:
                expo_ref = (expo, cand)
        f1, thr = best_f1(expo_ref[0][expo_ref[1]], off[expo_ref[1]])
        rows.append((a, int(off.sum()), int((~wall).sum()), *aucs, f1, thr))
        print("%12s %6d %7d | %7.3f %7.3f %7.3f | %7.3f %7.3f"
              % (a, int(off.sum()), int((~wall).sum()), aucs[0], aucs[1], aucs[2], f1, thr))

    arr = np.array([[r[3], r[4], r[5], r[6], r[7]] for r in rows], dtype=float)
    print("\n n=%d full-horizon vessels with off-wall clot" % len(rows))
    print("  mean AUC of exposure  r=1.5 %.3f   r=2.2 %.3f   r=3.0 %.3f"
          % (np.nanmean(arr[:, 0]), np.nanmean(arr[:, 1]), np.nanmean(arr[:, 2])))
    print("  mean ORACLE-threshold F1 (r=2.2): %.3f" % np.nanmean(arr[:, 3]))
    print("  oracle threshold varies per vessel: pct[10,50,90] = %s"
          % np.round(np.nanpercentile(arr[:, 4], [10, 50, 90]), 3))
    print("\n  AUC says how well exposure RANKS off-wall clot; the oracle F1 is the ceiling")
    print("  for any single global exposure threshold, with an oracle wall seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
