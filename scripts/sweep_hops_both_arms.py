"""Stencil width sweep for both flow arms.

The predicted flow costs -0.142 deploy score against GT t=0 flow.  Differentiating a
noisy field amplifies its noise, and the MLS stencil width is the natural regulariser --
a wider graph neighbourhood fits the same quadratic over more points.  Deploy-legal
(geometry only) and fit on TRAIN.
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
)

DIR = Path("data/processed/graphs_biochem_anchors")
DEV_HOLDOUT = ("patient042", "patient043")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relax", type=float, default=2.0)
    ap.add_argument("--hops-grow", type=int, default=6)
    ap.add_argument("--stencils", default="2,3,4,5,6")
    ap.add_argument("--out", default="outputs/hops_sweep.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION) | set(DEV_HOLDOUT))

    base = {}
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
        base[a] = dict(d=d, wall=wall, A=A, gt=gt * torch.tensor(wall.astype(np.float32)))

    results = []
    print("%8s %6s | %-24s | %-20s | %-20s" % ("stencil", "arm", "TRAIN", "DEVHOLD", "SEALED"))
    for st in [int(x) for x in args.stencils.split(",")]:
        for arm in ("gt", "pred"):
            sc = {}
            for a, c in base.items():
                try:
                    f = t0_flow_fields(c["d"], bio, hops=st, flow_source=arm)
                except ValueError:
                    continue
                cur = (f.gate > 0) & c["wall"]
                adm = (f.sr < float(bio.lss) * args.relax) & c["wall"]
                for _ in range(args.hops_grow):
                    cur = cur | (((c["A"] @ cur.astype(np.int8)) > 0) & adm)
                m = compute_clot_relaxed_metrics(torch.tensor(cur.astype(np.float32)), c["gt"],
                                                 c["d"].edge_index,
                                                 wall_mask=torch.tensor(c["wall"]))
                o = metrics_to_deploy_prefix(m)
                sc[a] = (clot_score_from_deploy_dict(o), o["deploy_clot_f1"])

            def agg(mem):
                v = [sc[a] for a in mem if a in sc]
                if not v:
                    return (float("nan"), float("nan"), 0, 0)
                return (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])),
                        sum(x[0] >= 0.6 for x in v), len(v))
            tr, dv, sl = agg(WALL_COHORT_V2_TRAIN), agg(DEV_HOLDOUT), agg(WALL_COHORT_V2_GENERALIZATION)
            results.append(dict(stencil=st, arm=arm, train=tr, dev=dv, sealed=sl,
                                per_vessel={k: v[0] for k, v in sc.items()}))
            print("%8d %6s | %.4f sF1 %.3f (%d/%d) | %.4f (%d/%d) | %.4f (%d/%d)"
                  % (st, arm, tr[0], tr[1], tr[2], tr[3], dv[0], dv[2], dv[3],
                     sl[0], sl[2], sl[3]))

    for arm in ("gt", "pred"):
        cand = [r for r in results if r["arm"] == arm and r["train"][0] == r["train"][0]]
        b = max(cand, key=lambda z: z["train"][0])
        print("\nBEST %-4s on TRAIN: stencil=%d -> train %.4f | devhold %.4f | sealed %.4f"
              % (arm, b["stencil"], b["train"][0], b["dev"][0], b["sealed"][0]))
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
