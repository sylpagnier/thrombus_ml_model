"""The Phase-3 physics wall model, scored with and WITHOUT the GT-t=0-flow bandaid.

Arm A (Phase 3, "with GT t=0 flow"): shear fields from ``data.y[0,:,0:2]``.
Arm B (Phase 5, deployable):         shear fields from ``u0_pred``/``v0_pred``.

The A-B delta is the deployability gap PHASE3_HANDOFF 4a asks for, measured in the
canonical wall-masked ``deploy_clot_score``.  Both arms use the same two growth scalars,
which were fit on WALL_COHORT_V2_TRAIN under arm A only.
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
    WALL_COHORT_V2_DEV, WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")


def predict(f, wall, A, bio, relax, hops):
    cur = (f.gate > 0) & wall
    adm = (f.sr < float(bio.lss) * relax) & wall
    for _ in range(hops):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relax", type=float, default=2.0)
    ap.add_argument("--hops", type=int, default=6)
    ap.add_argument("--out", default="outputs/phase3_vs_phase5.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_DEV)
                   | set(WALL_COHORT_V2_GENERALIZATION))
    rows = []
    print("growth: relax=%.3g hops=%d  (fit on TRAIN, arm A only)" % (args.relax, args.hops))
    print("%12s %8s %8s %8s %8s | %8s %8s"
          % ("vessel", "split", "gt_flow", "pred", "delta", "sF1_gt", "sF1_pred"))
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
        row = {"anchor": a,
               "split": next((k for k, v in (("train", WALL_COHORT_V2_TRAIN),
                                             ("dev", WALL_COHORT_V2_DEV),
                                             ("sealed", WALL_COHORT_V2_GENERALIZATION))
                              if a in v), "?")}
        for arm, src in (("gt", "gt"), ("pred", "pred")):
            try:
                f = t0_flow_fields(d, bio, hops=3, flow_source=src)
            except ValueError:
                row[arm] = None
                continue
            pr = predict(f, wall, A, bio, args.relax, args.hops)
            m = compute_clot_relaxed_metrics(torch.tensor(pr.astype(np.float32)), gt,
                                             d.edge_index, wall_mask=torch.tensor(wall))
            o = metrics_to_deploy_prefix(m)
            row[arm] = clot_score_from_deploy_dict(o)
            row[arm + "_f1"] = o["deploy_clot_f1"]
            row[arm + "_p"] = o["deploy_clot_relaxed_prec"]
            row[arm + "_r"] = o["deploy_clot_relaxed_rec"]
        rows.append(row)
        dl = "" if row.get("pred") is None else "%8.4f" % (row["pred"] - row["gt"])
        print("%12s %8s %8.4f %8s %8s | %8.3f %8s"
              % (a, row["split"], row["gt"],
                 "--" if row.get("pred") is None else "%.4f" % row["pred"], dl,
                 row["gt_f1"], "--" if row.get("pred") is None else "%.3f" % row["pred_f1"]))

    print("\n%-8s %5s %9s %5s %9s %9s" % ("split", "nA", "arm A(GT)", "nB", "arm B(pred)", "delta"))
    for split in ("train", "dev", "sealed", "ALL"):
        sel = [r for r in rows if split == "ALL" or r["split"] == split]
        a_ = [r["gt"] for r in sel]
        b_ = [r["pred"] for r in sel if r.get("pred") is not None]
        both = [(r["gt"], r["pred"]) for r in sel if r.get("pred") is not None]
        dl = float(np.mean([y - x for x, y in both])) if both else float("nan")
        print("%-8s %5d %9.4f %5d %9.4f %9.4f  (>=0.6  A %d/%d  B %d/%d)"
              % (split, len(a_), float(np.mean(a_)), len(b_),
                 float(np.mean(b_)) if b_ else float("nan"), dl,
                 sum(x >= 0.6 for x in a_), len(a_), sum(x >= 0.6 for x in b_), len(b_)))
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
