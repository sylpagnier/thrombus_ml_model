"""Fit the wall-normal thickness lumen arm on TRAIN and report the compound gates.

Compares against the learned lumen specialist recorded in
``data/reference/mat_compound_deploy.json`` (mean_offwall_relaxed_f1 0.4726,
mean_hop_ge2_strict 0.1798, ge2_recall 0.735) and against the graph-dilation lumen rule
(``scripts/sweep_lumen_arm.py``), which caps at ~0.50 off-wall relaxed F1 while lowering
the full-mesh score.

Metrics come from UNMASKED pred/GT with ``wall_mask=`` passed through -- see the protocol
note in ``sweep_lumen_arm.py``.
"""
from __future__ import annotations

import argparse
import itertools
import json
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
from src.core_physics.physics_lumen_model import (  # noqa: E402
    adjacency, lumen_thickness_layer, speed_nd, speed_nd_pred,
)
from src.core_physics.physics_wall_model import node_positions, t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
RELAX, GROW = 2.0, 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gt", choices=["gt", "pred"])
    ap.add_argument("--stencil", type=int, default=3)
    ap.add_argument("--out", default="outputs/lumen_thickness_sweep.json")
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
        A = adjacency(d.edge_index.numpy(), n)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = (gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu"))
              .reshape(-1).numpy() > 0.5)
        cur = (f.gate > 0) & wall
        adm = (f.sr < float(bio.lss) * RELAX) & wall
        for _ in range(GROW):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        cache[a] = dict(d=d, wall=wall, gt=gt, wall_pred=cur,
                        pos=node_positions(d), ei=d.edge_index.numpy(),
                        spd=speed_nd_pred(d) if args.arm == "pred" else speed_nd(d),
                        n_gt_off=int((gt & ~wall).sum()),
                        full=int(d.y.shape[0]) >= 150)
    print("cached %d vessels (%d with off-wall GT)  arm=%s"
          % (len(cache), sum(1 for c in cache.values() if c["n_gt_off"] > 0), args.arm))

    def evaluate(th, sthr, oracle=False):
        out = {}
        for a, c in cache.items():
            seed = (c["gt"] & c["wall"]) if oracle else c["wall_pred"]
            off = lumen_thickness_layer(seed, c["wall"], c["pos"], c["ei"], c["spd"],
                                        thickness_edges=th, speed_thresh=sthr)
            pred = c["wall_pred"] | off
            m = compute_clot_relaxed_metrics(
                torch.tensor(pred.astype(np.float32)),
                torch.tensor(c["gt"].astype(np.float32)),
                c["d"].edge_index, wall_mask=torch.tensor(c["wall"]))
            o = metrics_to_deploy_prefix(m)
            out[a] = dict(score=clot_score_from_deploy_dict(o), f1=o["deploy_clot_f1"],
                          off_rel=o.get("deploy_clot_offwall_relaxed_f1", 0.0),
                          off_strict=o.get("deploy_clot_offwall_strict_f1", 0.0),
                          ge2=o.get("deploy_clot_offwall_strict_f1_hop_ge2", 0.0),
                          ge2p=o.get("deploy_clot_offwall_n_pred_hop_ge2", 0.0),
                          ge2g=o.get("deploy_clot_offwall_n_gt_hop_ge2", 0.0),
                          has_off=c["n_gt_off"] > 0)
        return out

    def agg(res, members):
        v = [res[a] for a in members if a in res]
        vo = [x for x in v if x["has_off"]]
        gp, gg = sum(x["ge2p"] for x in v), sum(x["ge2g"] for x in v)
        return dict(score=float(np.mean([x["score"] for x in v])),
                    f1=float(np.mean([x["f1"] for x in v])),
                    off_rel=float(np.mean([x["off_rel"] for x in vo])) if vo else float("nan"),
                    off_strict=float(np.mean([x["off_strict"] for x in vo])) if vo else float("nan"),
                    ge2=float(np.mean([x["ge2"] for x in v])),
                    ge2_recall=gp / gg if gg > 0.5 else float("nan"))

    results = []
    print("\n%6s %7s | %-50s | %-28s"
          % ("thick", "spdThr", "TRAIN score / f1 / offRel+ / offStrict+ / ge2 / ge2rec", "SEALED score / offRel+"))
    for th, sthr in itertools.product([0.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0], [0.5, 1.0, np.inf]):
        r = evaluate(th, sthr)
        tr, sl = agg(r, WALL_COHORT_V2_TRAIN), agg(r, WALL_COHORT_V2_GENERALIZATION)
        results.append(dict(thickness=th, speed_thresh=float(sthr), train=tr, sealed=sl))
        print("%6.2f %7.3g | %.4f  %.4f  %.4f  %.4f  %.4f  %.4f | %.4f  %.4f"
              % (th, sthr, tr["score"], tr["f1"], tr["off_rel"], tr["off_strict"],
                 tr["ge2"], tr["ge2_recall"], sl["score"], sl["off_rel"]))

    b = max(results, key=lambda z: z["train"]["score"])
    bo = max(results, key=lambda z: (z["train"]["off_rel"]
                                     if z["train"]["off_rel"] == z["train"]["off_rel"] else -1))
    for tag, x in (("full-mesh score", b), ("off-wall relaxed F1", bo)):
        print("\nBEST on TRAIN by %s: thickness=%.2f speed<%.3g" % (tag, x["thickness"], x["speed_thresh"]))
        print("   train  score %.4f  f1 %.4f  offRel+ %.4f  offStrict+ %.4f  ge2 %.4f  ge2recall %.4f"
              % (x["train"]["score"], x["train"]["f1"], x["train"]["off_rel"],
                 x["train"]["off_strict"], x["train"]["ge2"], x["train"]["ge2_recall"]))
        print("   SEALED score %.4f  f1 %.4f  offRel+ %.4f  offStrict+ %.4f  ge2 %.4f  ge2recall %.4f"
              % (x["sealed"]["score"], x["sealed"]["f1"], x["sealed"]["off_rel"],
                 x["sealed"]["off_strict"], x["sealed"]["ge2"], x["sealed"]["ge2_recall"]))

    print("\n[ORACLE] same thickness rule seeded by GT wall clot (ceiling for the rule)")
    for th in (1.5, 1.8, 2.1):
        tr = agg(evaluate(th, np.inf, oracle=True), WALL_COHORT_V2_TRAIN)
        print("   thickness=%.2f  train offRel+ %.4f  offStrict+ %.4f  ge2 %.4f  ge2recall %.4f"
              % (th, tr["off_rel"], tr["off_strict"], tr["ge2"], tr["ge2_recall"]))

    print("\n[REFERENCE] learned lumen specialist (mat_compound_deploy.json, orig10):")
    print("   mean_offwall_relaxed_f1 0.4726   mean_hop_ge2_strict 0.1798   ge2_recall 0.7351")
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
