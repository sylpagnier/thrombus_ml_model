"""Which admission rule should the clot front use?

The growth term admits a neighbour of committed wall tissue when its shear is below
``lss * relax``.  The deposition law has a second branch, so a front node might equally
be admitted by a near-threshold shear GRADIENT.  Tested here against the low-shear rule
and against their union / intersection.  Fit on WALL_COHORT_V2_TRAIN only.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gt")
    ap.add_argument("--stencil", type=int, default=3)
    ap.add_argument("--hops", type=int, default=6)
    ap.add_argument("--out", default="outputs/growth_admission.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
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
        n = len(wall)
        ei = d.edge_index.numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        cache[a] = dict(d=d, wall=wall, f=f, A=A, gt=gt * torch.tensor(wall.astype(np.float32)),
                        full=int(d.y.shape[0]) >= 150)
    print("cached %d (arm=%s stencil=%d hops=%d)" % (len(cache), args.arm, args.stencil, args.hops))

    sgt = float(bio.sgt) / 100.0     # 1/(s*cm)
    rules = {}
    for rl in (1.5, 2.0, 3.0):
        rules[f"low<{rl}*lss"] = lambda c, rl=rl: c["f"].sr < float(bio.lss) * rl
    for rs in (0.3, 0.5, 0.7):
        rules[f"sep<{rs}*sgt"] = lambda c, rs=rs: c["f"].dsrx < sgt * rs
    rules["low<2lss OR sep<0.5sgt"] = lambda c: ((c["f"].sr < 2 * float(bio.lss))
                                                 | (c["f"].dsrx < sgt * 0.5))
    rules["low<2lss AND sep<0.5sgt"] = lambda c: ((c["f"].sr < 2 * float(bio.lss))
                                                  & (c["f"].dsrx < sgt * 0.5))
    rules["low<3lss OR sep<0.3sgt"] = lambda c: ((c["f"].sr < 3 * float(bio.lss))
                                                 | (c["f"].dsrx < sgt * 0.3))

    out = []
    print("%-26s %-24s %-16s %-16s" % ("admission rule", "TRAIN", "TRAIN full-hz", "SEALED"))
    for nm, fn in rules.items():
        sc = {}
        for a, c in cache.items():
            cur = (c["f"].gate > 0) & c["wall"]
            adm = fn(c) & c["wall"]
            for _ in range(args.hops):
                cur = cur | (((c["A"] @ cur.astype(np.int8)) > 0) & adm)
            m = compute_clot_relaxed_metrics(torch.tensor(cur.astype(np.float32)), c["gt"],
                                             c["d"].edge_index, wall_mask=torch.tensor(c["wall"]))
            o = metrics_to_deploy_prefix(m)
            sc[a] = (clot_score_from_deploy_dict(o), o["deploy_clot_f1"], c["full"])

        def agg(sel):
            v = [sc[a] for a in sel if a in sc]
            return (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])),
                    sum(x[0] >= 0.6 for x in v), len(v))
        tr = agg(WALL_COHORT_V2_TRAIN)
        trf = agg([a for a in WALL_COHORT_V2_TRAIN if a in sc and sc[a][2]])
        sl = agg(WALL_COHORT_V2_GENERALIZATION)
        out.append(dict(rule=nm, train=tr, train_full=trf, sealed=sl))
        print("%-26s %.4f sF1 %.3f (%2d/%2d) %.4f (%2d/%2d) %.4f (%d/%d)"
              % (nm, tr[0], tr[1], tr[2], tr[3], trf[0], trf[2], trf[3], sl[0], sl[2], sl[3]))
    b = max(out, key=lambda z: z["train"][0])
    print("\nBEST on TRAIN: %s -> train %.4f | train-full %.4f | sealed %.4f"
          % (b["rule"], b["train"][0], b["train_full"][0], b["sealed"][0]))
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
