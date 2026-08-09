"""Fit the lumen arm's three scalars on TRAIN, and report the compound gates.

Reports the metrics the compound reference (``data/reference/mat_compound_deploy.json``)
is gated on: ``offwall_relaxed_f1``, ``offwall_strict_f1_hop_ge2``, ``ge2_recall``, plus
the full-mesh ``deploy_clot_score``.

NOTE ON PROTOCOL.  ``deploy_clot_phi_trajectory`` multiplies BOTH prediction and GT by
``mask_wall`` whenever the pack has one -- unconditionally -- so the canonical eval today
cannot produce a non-zero off-wall metric at all.  The compound reference predates that.
Metrics here are computed from UNMASKED pred/GT with ``wall_mask=`` passed through, which
is what ``compute_clot_relaxed_metrics``'s off-wall block is written for.

An ORACLE row is included: the GT wall clot propagated with the same rule.  That is the
ceiling for any lumen model seeded by a wall arm, and it separates "the rule is wrong"
from "the wall seed is wrong".
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
from src.core_physics.physics_lumen_model import adjacency, grow_into_lumen, speed_nd  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
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
    ap.add_argument("--out", default="outputs/lumen_sweep.json")
    ap.add_argument("--full-horizon-only", action="store_true")
    args = ap.parse_args()
    FULL_ONLY = args.full_horizon_only
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
        # wall arm
        cur = (f.gate > 0) & wall
        adm = (f.sr < float(bio.lss) * RELAX) & wall
        for _ in range(GROW):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        cache[a] = dict(d=d, wall=wall, A=A, gt=gt, wall_pred=cur, sr=f.sr,
                        spd=speed_nd(d), full=int(d.y.shape[0]) >= 150,
                        n_gt_off=int((gt & ~wall).sum()))
    n_off = sum(1 for c in cache.values() if c["n_gt_off"] > 0)
    print("cached %d vessels (%d with off-wall GT clot)  arm=%s" % (len(cache), n_off, args.arm))

    def evaluate(hops, sthr, srmax, oracle=False):
        out = {}
        for a, c in cache.items():
            seed = (c["gt"] & c["wall"]) if oracle else c["wall_pred"]
            off = grow_into_lumen(seed, c["wall"], c["A"], c["spd"], c["sr"],
                                  lumen_hops=hops, speed_thresh=sthr, sr_max=srmax)
            pred = c["wall_pred"] | off
            m = compute_clot_relaxed_metrics(
                torch.tensor(pred.astype(np.float32)),
                torch.tensor(c["gt"].astype(np.float32)),
                c["d"].edge_index, wall_mask=torch.tensor(c["wall"]))
            o = metrics_to_deploy_prefix(m)
            out[a] = dict(score=clot_score_from_deploy_dict(o),
                          f1=o["deploy_clot_f1"],
                          off_rel=o.get("deploy_clot_offwall_relaxed_f1", 0.0),
                          off_strict=o.get("deploy_clot_offwall_strict_f1", 0.0),
                          ge2=o.get("deploy_clot_offwall_strict_f1_hop_ge2", 0.0),
                          ge2p=o.get("deploy_clot_offwall_n_pred_hop_ge2", 0.0),
                          ge2g=o.get("deploy_clot_offwall_n_gt_hop_ge2", 0.0),
                          has_off=c["n_gt_off"] > 0)
        return out

    def agg(res, members, full_only=False):
        v = [res[a] for a in members if a in res and (not full_only or cache[a]["full"])]
        vo = [x for x in v if x["has_off"]]
        gp = sum(x["ge2p"] for x in v)
        gg = sum(x["ge2g"] for x in v)
        return dict(score=float(np.mean([x["score"] for x in v])),
                    f1=float(np.mean([x["f1"] for x in v])),
                    off_rel=float(np.mean([x["off_rel"] for x in v])),
                    off_rel_pos=float(np.mean([x["off_rel"] for x in vo])) if vo else float("nan"),
                    ge2=float(np.mean([x["ge2"] for x in v])),
                    ge2_recall=gp / gg if gg > 0.5 else float("nan"), n=len(v))

    grid = list(itertools.product([0, 1, 2, 3, 4], [0.3, 0.5, 0.7, 1.0, 1e9], [np.inf, 100.0]))
    results = []
    print("\n%5s %7s %7s | %-46s | %-24s"
          % ("hops", "spdThr", "srMax", "TRAIN  score / f1 / offRel(+) / ge2 / ge2rec", "SEALED score / offRel"))
    for hops, sthr, srmax in grid:
        r = evaluate(hops, sthr, srmax)
        tr = agg(r, WALL_COHORT_V2_TRAIN, full_only=FULL_ONLY)
        sl = agg(r, WALL_COHORT_V2_GENERALIZATION, full_only=FULL_ONLY)
        results.append(dict(hops=hops, speed_thresh=sthr, sr_max=float(srmax), train=tr, sealed=sl))
        print("%5d %7.4g %7.4g | %.4f  %.4f  %.4f  %.4f  %.4f | %.4f  %.4f"
              % (hops, sthr, srmax, tr["score"], tr["f1"], tr["off_rel_pos"], tr["ge2"],
                 tr["ge2_recall"], sl["score"], sl["off_rel_pos"]))

    best = max(results, key=lambda z: z["train"]["score"])
    print("\nBEST on TRAIN by full-mesh score: hops=%d speed<%.4g sr<%.4g"
          % (best["hops"], best["speed_thresh"], best["sr_max"]))
    print("   train  score %.4f  f1 %.4f  offRel(+) %.4f  ge2 %.4f  ge2recall %.4f"
          % (best["train"]["score"], best["train"]["f1"], best["train"]["off_rel_pos"],
             best["train"]["ge2"], best["train"]["ge2_recall"]))
    print("   SEALED score %.4f  f1 %.4f  offRel(+) %.4f  ge2 %.4f  ge2recall %.4f"
          % (best["sealed"]["score"], best["sealed"]["f1"], best["sealed"]["off_rel_pos"],
             best["sealed"]["ge2"], best["sealed"]["ge2_recall"]))

    bo = max(results, key=lambda z: (z["train"]["off_rel_pos"]
                                     if z["train"]["off_rel_pos"] == z["train"]["off_rel_pos"] else -1))
    print("\nBEST on TRAIN by off-wall relaxed F1: hops=%d speed<%.4g sr<%.4g -> train %.4f | sealed %.4f"
          % (bo["hops"], bo["speed_thresh"], bo["sr_max"],
             bo["train"]["off_rel_pos"], bo["sealed"]["off_rel_pos"]))

    # ORACLE ceiling: same rule seeded by GT wall clot
    print("\n[ORACLE] same propagation rule seeded by the GT wall clot (ceiling for the rule)")
    for hops, sthr in ((2, 0.5), (3, 0.5), (3, 0.7), (3, 1e9), (4, 0.7)):
        r = evaluate(hops, sthr, np.inf, oracle=True)
        tr = agg(r, WALL_COHORT_V2_TRAIN, full_only=FULL_ONLY)
        print("   hops=%d speed<%-6.4g  train offRel(+) %.4f  ge2 %.4f  ge2recall %.4f"
              % (hops, sthr, tr["off_rel_pos"], tr["ge2"], tr["ge2_recall"]))
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
