"""Autocatalytic lumen arm with physical-radius nucleation -- fit on full-horizon TRAIN.

Three scalars: ``r_nuc`` (nucleation ball radius, in median edge lengths),
``expose_thresh`` (committed fraction of that ball required to ignite), ``n_steps``.

Compared against, on identical vessels and metric:
  * wall arm only
  * graph-dilation lumen arm  (hops=2, speed_nd<0.3)   -- the previous best
  * the learned lumen specialist (mat_compound_deploy.json): offwall_relaxed_f1 0.4726

Selection is on full-horizon TRAIN (T>=150); on a truncated run every off-wall prediction
is a pure false positive (docs/PHASE3_RESULTS.md 6).
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
    adjacency, autocatalytic_lumen, grow_into_lumen, speed_nd, speed_nd_pred,
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
    ap.add_argument("--full-horizon-only", action="store_true", default=True)
    ap.add_argument("--out", default="outputs/lumen_autocat_sweep.json")
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
        A = adjacency(d.edge_index.numpy(), len(wall))
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = (gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu"))
              .reshape(-1).numpy() > 0.5)
        cur = (f.gate > 0) & wall
        adm = (f.sr < float(bio.lss) * RELAX) & wall
        for _ in range(GROW):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        cache[a] = dict(d=d, wall=wall, A=A, gt=gt, wall_pred=cur, sr=f.sr,
                        pos=node_positions(d), ei=d.edge_index.numpy(),
                        spd=speed_nd_pred(d) if args.arm == "pred" else speed_nd(d),
                        full=int(d.y.shape[0]) >= 150,
                        has_off=int((gt & ~wall).sum()) > 0)
    print("cached %d vessels (%d full-horizon, %d with off-wall GT)"
          % (len(cache), sum(c["full"] for c in cache.values()),
             sum(c["has_off"] for c in cache.values())))

    def run(off_fn, oracle=False):
        out = {}
        for a, c in cache.items():
            seed = (c["gt"] & c["wall"]) if oracle else c["wall_pred"]
            off = off_fn(c, seed)
            pred = c["wall_pred"] | off
            m = compute_clot_relaxed_metrics(
                torch.tensor(pred.astype(np.float32)),
                torch.tensor(c["gt"].astype(np.float32)),
                c["d"].edge_index, wall_mask=torch.tensor(c["wall"]))
            o = metrics_to_deploy_prefix(m)
            out[a] = dict(score=clot_score_from_deploy_dict(o), f1=o["deploy_clot_f1"],
                          off_rel=o.get("deploy_clot_offwall_relaxed_f1", 0.0),
                          off_strict=o.get("deploy_clot_offwall_strict_f1", 0.0),
                          ge2p=o.get("deploy_clot_offwall_n_pred_hop_ge2", 0.0),
                          ge2g=o.get("deploy_clot_offwall_n_gt_hop_ge2", 0.0),
                          noff=o.get("deploy_clot_offwall_n_pred", 0.0),
                          has_off=c["has_off"])
        return out

    def agg(res, members):
        v = [res[a] for a in members if a in res and cache[a]["full"]]
        vo = [x for x in v if x["has_off"]]
        gp, gg = sum(x["ge2p"] for x in v), sum(x["ge2g"] for x in v)
        return dict(score=float(np.mean([x["score"] for x in v])),
                    f1=float(np.mean([x["f1"] for x in v])),
                    off_rel=float(np.mean([x["off_rel"] for x in vo])) if vo else float("nan"),
                    off_strict=float(np.mean([x["off_strict"] for x in vo])) if vo else float("nan"),
                    ge2_recall=gp / gg if gg > 0.5 else float("nan"),
                    n_off_pred=float(np.mean([x["noff"] for x in v])))

    def show(tag, res):
        tr, sl = agg(res, WALL_COHORT_V2_TRAIN), agg(res, WALL_COHORT_V2_GENERALIZATION)
        print("%-38s | %.4f %.4f %.4f %.4f %6.1f | %.4f %.4f %.4f"
              % (tag, tr["score"], tr["f1"], tr["off_rel"], tr["off_strict"], tr["n_off_pred"],
                 sl["score"], sl["f1"], sl["off_rel"]))
        return tr, sl

    print("\n%-38s | %-40s | %-22s"
          % ("arm", "TRAIN(full-hz) score/f1/offRel+/offStrict+/nOff", "SEALED score/f1/offRel+"))
    show("wall only", run(lambda c, s: np.zeros_like(c["wall"])))
    show("graph dilation hops=2 spd<0.3",
         run(lambda c, s: grow_into_lumen(s, c["wall"], c["A"], c["spd"], c["sr"],
                                          lumen_hops=2, speed_thresh=0.3)))

    results = []
    print()
    for r_nuc, thr, steps in itertools.product([2.2, 3.0], [0.08, 0.10, 0.12, 0.14, 0.17, 0.20, 0.25], [1, 2, 3, 5]):
        res = run(lambda c, s, r=r_nuc, t=thr, n=steps: autocatalytic_lumen(
            s, c["wall"], c["pos"], c["ei"], r_nuc=r, expose_thresh=t, n_steps=n))
        tr, sl = show("autocat r=%.1f thr=%.2f steps=%d" % (r_nuc, thr, steps), res)
        results.append(dict(r_nuc=r_nuc, expose_thresh=thr, n_steps=steps, train=tr, sealed=sl))

    ok = [x for x in results if x["train"]["off_rel"] == x["train"]["off_rel"]]
    b = max(ok, key=lambda z: z["train"]["score"])
    bo = max(ok, key=lambda z: z["train"]["off_rel"])
    for tag, x in (("full-mesh score", b), ("off-wall relaxed F1", bo)):
        print("\nBEST autocat on TRAIN(full-hz) by %s: r=%.1f thr=%.2f steps=%d"
              % (tag, x["r_nuc"], x["expose_thresh"], x["n_steps"]))
        print("   train  score %.4f  f1 %.4f  offRel+ %.4f  offStrict+ %.4f  ge2recall %.4f"
              % (x["train"]["score"], x["train"]["f1"], x["train"]["off_rel"],
                 x["train"]["off_strict"], x["train"]["ge2_recall"]))
        print("   SEALED score %.4f  f1 %.4f  offRel+ %.4f  offStrict+ %.4f  ge2recall %.4f"
              % (x["sealed"]["score"], x["sealed"]["f1"], x["sealed"]["off_rel"],
                 x["sealed"]["off_strict"], x["sealed"]["ge2_recall"]))

    print("\n[ORACLE] autocat seeded by GT wall clot (ceiling for the rule)")
    for r_nuc, thr in ((2.2, 0.12), (2.2, 0.14), (3.0, 0.14)):
        tr, _ = show("  oracle r=%.1f thr=%.2f steps=2" % (r_nuc, thr),
                     run(lambda c, s, r=r_nuc, t=thr: autocatalytic_lumen(
                         s, c["wall"], c["pos"], c["ei"], r_nuc=r, expose_thresh=t, n_steps=2),
                         oracle=True))
    print("\n[REFERENCE] learned lumen specialist: offwall_relaxed_f1 0.4726, ge2_recall 0.7351")
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
