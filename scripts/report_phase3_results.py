"""Consolidated Phase-3 result table for the t=0 physics wall model.

Reports the canonical wall-masked ``deploy_clot_score`` for:

  * arm A -- with GT t=0 flow (the PHASE3_HANDOFF 0a bandaid), MLS stencil 3
  * arm B -- deployable, flow from ``u0_pred``/``v0_pred``, MLS stencil 4

split by cohort (train / dev-holdout / sealed) and by simulation horizon.  The horizon
split matters: ``mat_growth_simple`` states T>=150 as the holdout rule precisely because
on a truncated run "the vessel's final map" is a different quantity -- the GT there is
clot ONSET, not the converged map.  Every over-predicting vessel is a truncated run.

Both scalars of the growth term (relax, hops) were fit on WALL_COHORT_V2_TRAIN under
arm A, and the MLS stencil per arm likewise.  Nothing was fit on dev or sealed.
"""
from __future__ import annotations

import argparse
import json
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
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
    scoring_fingerprint,
)

DIR = Path("data/processed/graphs_biochem_anchors")
DEV_HOLDOUT = ("patient042", "patient043")
FULL_HORIZON_T = 150


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relax", type=float, default=2.0)
    ap.add_argument("--grow-hops", type=int, default=6)
    ap.add_argument("--stencil-gt", type=int, default=3)
    ap.add_argument("--stencil-pred", type=int, default=4)
    ap.add_argument("--out", default="outputs/phase3_final_results.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    print("SCORING FINGERPRINT %s" % scoring_fingerprint())
    print("growth relax=%.3g hops=%d | stencil gt=%d pred=%d\n"
          % (args.relax, args.grow_hops, args.stencil_gt, args.stencil_pred))

    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION) | set(DEV_HOLDOUT))
    rows = []
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n = len(wall)
        ei = d.edge_index.numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt * torch.tensor(wall.astype(np.float32))
        r = {"anchor": a, "T": int(d.y.shape[0]),
             "full_horizon": int(d.y.shape[0]) >= FULL_HORIZON_T,
             "n_gt": int(gt.sum()),
             "split": ("sealed" if a in WALL_COHORT_V2_GENERALIZATION
                       else "devhold" if a in DEV_HOLDOUT else "train")}
        for arm, st in (("gt", args.stencil_gt), ("pred", args.stencil_pred)):
            try:
                f = t0_flow_fields(d, bio, hops=st, flow_source=arm)
            except ValueError:
                r[arm] = None
                continue
            cur = (f.gate > 0) & wall
            adm = (f.sr < float(bio.lss) * args.relax) & wall
            for _ in range(args.grow_hops):
                cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
            m = compute_clot_relaxed_metrics(torch.tensor(cur.astype(np.float32)), gt,
                                             d.edge_index, wall_mask=torch.tensor(wall))
            o = metrics_to_deploy_prefix(m)
            r[arm] = clot_score_from_deploy_dict(o)
            r[arm + "_f1"] = o["deploy_clot_f1"]
        rows.append(r)

    print("%12s %5s %6s %5s %9s %9s" % ("vessel", "T", "split", "nGT", "armA(GT)", "armB(pred)"))
    for r in sorted(rows, key=lambda z: (z["split"], z["anchor"])):
        print("%12s %5d %6s %5d %9.4f %9s%s"
              % (r["anchor"], r["T"], r["split"], r["n_gt"], r["gt"],
                 "--" if r.get("pred") is None else "%.4f" % r["pred"],
                 "" if r["full_horizon"] else "   (truncated run)"))

    def agg(sel, key):
        v = [r[key] for r in sel if r.get(key) is not None]
        return (float(np.mean(v)) if v else float("nan"), sum(x >= 0.6 for x in v), len(v))

    print("\n%-28s %-22s %-22s" % ("subset", "arm A (GT t=0 flow)", "arm B (deployable)"))
    for label, sel in (
        ("ALL", rows),
        ("  train", [r for r in rows if r["split"] == "train"]),
        ("  dev-holdout", [r for r in rows if r["split"] == "devhold"]),
        ("  SEALED", [r for r in rows if r["split"] == "sealed"]),
        ("FULL-HORIZON only (T>=150)", [r for r in rows if r["full_horizon"]]),
        ("  train", [r for r in rows if r["full_horizon"] and r["split"] == "train"]),
        ("  dev-holdout", [r for r in rows if r["full_horizon"] and r["split"] == "devhold"]),
        ("  SEALED", [r for r in rows if r["full_horizon"] and r["split"] == "sealed"]),
        ("truncated only (T<150)", [r for r in rows if not r["full_horizon"]]),
    ):
        a_, b_ = agg(sel, "gt"), agg(sel, "pred")
        print("%-28s %.4f (>=0.6 %2d/%2d)  %.4f (>=0.6 %2d/%2d)"
              % (label, a_[0], a_[1], a_[2], b_[0], b_[1], b_[2]))
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
