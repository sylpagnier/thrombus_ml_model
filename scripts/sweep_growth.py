"""Add the clot-front growth term to the t=0 gate model and calibrate it on TRAIN only.

The t=0 gate under-predicts on vessels whose gates are barely open (patient044 0.39x,
042 0.41x, 012 0.43x) and over-predicts on 008/009.  Physically the missing piece is that
the flow evolves as the clot narrows the lumen, so nodes that are not gated at t=0 ignite
later next to nodes that are -- PHASE3_HANDOFF 26.13.2 measured every late commit within
2 hops of existing clot.

Model: seeds = t=0 gate-open wall nodes; then grow along the WALL graph for ``hops``
iterations, admitting a neighbour whose shear sits below a relaxed threshold
``lss * relax``.  Two scalars, both fit on WALL_COHORT_V2_TRAIN and then spent once on
dev / sealed.
"""
from __future__ import annotations

import argparse
import itertools
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


def grow(seeds, adj, admissible, hops):
    cur = seeds.copy()
    for _ in range(hops):
        nxt = (adj @ cur.astype(np.int8)) > 0
        cur = cur | (nxt & admissible)
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/growth_sweep.json")
    ap.add_argument("--arm", default="gt", choices=["gt", "pred"])
    ap.add_argument("--stencil", type=int, default=3)
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_DEV)
                   | set(WALL_COHORT_V2_GENERALIZATION))

    cache = {}
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        wall = d.mask_wall.reshape(-1).bool().numpy()
        try:
            f = t0_flow_fields(d, bio, hops=args.stencil, flow_source=args.arm)
        except ValueError:
            continue
        ei = d.edge_index.numpy()
        n = len(wall)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt * torch.tensor(wall.astype(np.float32))
        cache[a] = dict(data=d, wall=wall, f=f, A=A, gt=gt)
    print("cached %d vessels  arm=%s stencil=%d" % (len(cache), args.arm, args.stencil))

    def evaluate(relax, hops):
        out = {}
        for a, c in cache.items():
            seeds = (c["f"].gate > 0) & c["wall"]
            adm = (c["f"].sr < float(bio.lss) * relax) & c["wall"]
            pred = grow(seeds, c["A"], adm, hops) if hops else seeds
            m = compute_clot_relaxed_metrics(
                torch.tensor(pred.astype(np.float32)), c["gt"],
                c["data"].edge_index, wall_mask=torch.tensor(c["wall"]))
            o = metrics_to_deploy_prefix(m)
            out[a] = (clot_score_from_deploy_dict(o), o["deploy_clot_f1"],
                      o["deploy_clot_relaxed_prec"], o["deploy_clot_relaxed_rec"])
        return out

    grid = list(itertools.product([1.0, 1.5, 2.0, 3.0, 5.0, 1e9], [0, 1, 2, 3, 4, 6]))
    results = []
    print("%7s %5s | %-28s | %-22s | %-22s" % ("relax", "hops", "TRAIN", "DEV", "SEALED"))
    for relax, hops in grid:
        r = evaluate(relax, hops)

        def agg(members):
            v = [r[a] for a in members if a in r]
            return (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])),
                    sum(x[0] >= 0.6 for x in v), len(v))
        tr, dv, sl = agg(WALL_COHORT_V2_TRAIN), agg(WALL_COHORT_V2_DEV), agg(WALL_COHORT_V2_GENERALIZATION)
        results.append(dict(relax=relax, hops=hops, train=tr, dev=dv, sealed=sl,
                            per_vessel={a: v[0] for a, v in r.items()}))
        print("%7.4g %5d | score %.4f sF1 %.3f (%d/%d) | %.4f (%d/%d) | %.4f (%d/%d)"
              % (relax, hops, tr[0], tr[1], tr[2], tr[3], dv[0], dv[2], dv[3], sl[0], sl[2], sl[3]))

    best = max(results, key=lambda z: z["train"][0])
    print("\nBEST on TRAIN: relax=%.4g hops=%d -> train %.4f | dev %.4f | sealed %.4f"
          % (best["relax"], best["hops"], best["train"][0], best["dev"][0], best["sealed"][0]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
