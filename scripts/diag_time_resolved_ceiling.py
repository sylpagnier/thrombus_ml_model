"""What is a perfect ONSET-TIME model worth, on a time-resolved deploy score?

The physics model already gets the final committed SET right (0.9093 sealed, against a
flow-oracle ceiling of 0.9066).  What it gets wrong is WHEN: patient043 commits all 84
nodes in a single step at t=3000 s while GT climbs from ~7000 s to ~12000 s and creeps on
to 30000 s.

So the deploy score at the FINAL time is already near its ceiling and cannot be the
target.  The quantity with headroom is the deploy score evaluated ACROSS TIME.  This
measures three things on the same committed set, so the mask is held fixed and only the
timing varies:

    flash     physics set, onset from the ODE (what ships today)
    oracle    physics set, onset = each node's GT onset time
    upper     GT set and GT onset (a pure metric ceiling, not attainable)

``oracle - flash`` is the entire prize available to any onset model, learned or not.
If that gap is small there is nothing to build.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, graded_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import gt_onset_index  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
RELAX, GROW = 2.0, 6
N_EVAL = 12          # evaluation times spread over the horizon


def physics_set(d, w, bio):
    f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
    ei = d.edge_index.numpy()
    n = len(w)
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)
    cur = (f.gate > 0) & w
    adm = (f.sr < float(bio.lss) * RELAX) & w
    for _ in range(GROW):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur, f


def score_at(d, w, pred_mask, phys, ti):
    gt = gt_clot_phi_at_time(d, int(ti), phys, device=torch.device("cpu")).reshape(-1)
    wt = torch.tensor(w.astype(np.float32))
    m = compute_clot_relaxed_metrics(torch.tensor(pred_mask.astype(np.float32)) * wt,
                                     gt * wt, d.edge_index, wall_mask=torch.tensor(w))
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def main() -> int:
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = [n for n in sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))]
    rows = []
    print("%-12s %6s | %8s %8s %8s | %8s" % ("vessel", "T", "flash", "oracle", "upper", "gain"))
    for n in names:
        p = DIR / f"{n}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        if T < 150:
            continue
        w = d.mask_wall.reshape(-1).bool().numpy()
        gt_on = gt_onset_index(d, phys, w)
        if not ((gt_on >= 0) & w).any():
            continue
        S, f = physics_set(d, w, bio)
        g = graded_gate(f, bio, mode="hard") * w
        traj, t = integrate_mat_trajectory(d, bio, g, da_scale=40.0)
        model_on = first_crossing(traj, float(bio.viscosity_mat_crit))
        # nodes in S that never cross in the ODE still belong to the shipped mask; give
        # them the ODE's own median onset so the flash arm is not unfairly penalised.
        med_on = int(np.median(model_on[(model_on >= 0) & w])) if ((model_on >= 0) & w).any() else 0
        flash_on = np.where(S, np.where(model_on >= 0, model_on, med_on), -1)
        # oracle: same set, but each node ignites when GT says it does
        orac_on = np.where(S, np.where(gt_on >= 0, gt_on, T - 1), -1)

        eval_ts = np.unique(np.linspace(T // N_EVAL, T - 1, N_EVAL).astype(int))
        sf, so, su = [], [], []
        for ti in eval_ts:
            sf.append(score_at(d, w, (flash_on >= 0) & (flash_on <= ti), phys, ti))
            so.append(score_at(d, w, (orac_on >= 0) & (orac_on <= ti), phys, ti))
            su.append(score_at(d, w, (gt_on >= 0) & (gt_on <= ti) & w, phys, ti))
        r = dict(name=n, flash=float(np.median(sf)), oracle=float(np.median(so)),
                 upper=float(np.median(su)),
                 sealed=n in WALL_COHORT_V2_GENERALIZATION)
        rows.append(r)
        print("%-12s %6d | %8.4f %8.4f %8.4f | %+8.4f"
              % (n, T, r["flash"], r["oracle"], r["upper"], r["oracle"] - r["flash"]))

    for lbl, sel in (("ALL", rows),
                     ("train", [r for r in rows if not r["sealed"]]),
                     ("SEALED", [r for r in rows if r["sealed"]])):
        if not sel:
            continue
        fl = np.array([r["flash"] for r in sel])
        orc = np.array([r["oracle"] for r in sel])
        up = np.array([r["upper"] for r in sel])
        print("\n%-8s n=%2d  median-over-time deploy score" % (lbl, len(sel)))
        print("   flash (ships today) mean %.4f  median %.4f" % (fl.mean(), np.median(fl)))
        print("   perfect onset       mean %.4f  median %.4f" % (orc.mean(), np.median(orc)))
        print("   metric ceiling      mean %.4f  median %.4f" % (up.mean(), np.median(up)))
        print("   >>> PRIZE for a perfect onset model: %+.4f mean, %+.4f median"
              % (orc.mean() - fl.mean(), np.median(orc) - np.median(fl)))
    print("\nThe committed SET is identical in the flash and oracle arms -- only timing")
    print("differs -- so this gap is attributable to onset prediction and nothing else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
