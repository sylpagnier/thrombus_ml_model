"""ARM 2 sweep -- does letting the clot occlude its own lumen fix the growth curve?

Compares, on full-horizon vessels, onset timing for:
  * the hard t=0 gate (baseline)
  * arm 1 alone (graded gate)
  * arm 2 alone (hard gate + shear redistribution)
  * arm 1 + arm 2

All parameters fit on WALL_COHORT_V2_TRAIN; sealed reported once at the end.
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
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, graded_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.shear_redistribution import (  # noqa: E402
    build_crosssection_operator, make_blockage, sdf_nd,
)
from src.core_physics.mls_gradient import node_positions  # noqa: E402
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index, onset_metrics  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")


def load(names, bio, phys, stencil, arm, radius_mult):
    cache = {}
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        try:
            f = t0_flow_fields(d, bio, hops=stencil, flow_source=arm)
        except ValueError:
            continue
        pos = node_positions(d)
        B = build_crosssection_operator(pos, sdf_nd(d), wall, radius_mult=radius_mult)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        phi_gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        cache[a] = dict(d=d, wall=wall, f=f, B=B,
                        gt_idx=gt_onset_index(d, phys, wall),
                        phi_gt=phi_gt * torch.tensor(wall.astype(np.float32)))
    return cache


def run(c, bio, *, mode, tau_low, tau_sep, da, redistribute, exponent, phi_max, every):
    gate = graded_gate(c["f"], bio, mode=mode, tau_low=tau_low, tau_sep=tau_sep) * c["wall"]
    blk = None
    if redistribute:
        blk = make_blockage(c["f"], bio, c["B"], c["wall"], exponent=exponent,
                            phi_max=phi_max, every=every, graded_mode=mode,
                            tau_low=tau_low, tau_sep=tau_sep)
    traj, t = integrate_mat_trajectory(c["d"], bio, gate, da_scale=da, blockage=blk)
    idx = first_crossing(traj, float(bio.viscosity_mat_crit))
    m = onset_metrics(idx, c["gt_idx"], t, c["wall"])
    m["curve_l1"] = curve_l1(idx, c["gt_idx"], t, c["wall"])
    pred = torch.tensor((((idx >= 0) & c["wall"])).astype(np.float32))
    mm = compute_clot_relaxed_metrics(pred, c["phi_gt"], c["d"].edge_index,
                                      wall_mask=torch.tensor(c["wall"]))
    m["score"] = clot_score_from_deploy_dict(metrics_to_deploy_prefix(mm))
    return m


def agg(rows, members, key):
    v = [rows[a][key] for a in members if a in rows and rows[a][key] == rows[a][key]]
    return float(np.mean(v)) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gt")
    ap.add_argument("--stencil", type=int, default=3)
    ap.add_argument("--radius-mult", type=float, default=1.0)
    ap.add_argument("--out", default="outputs/shear_redistribution_sweep.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
    cache = load(names, bio, phys, args.stencil, args.arm, args.radius_mult)
    tr = [a for a in WALL_COHORT_V2_TRAIN if a in cache]
    sl = [a for a in WALL_COHORT_V2_GENERALIZATION if a in cache]
    print("full-horizon: %d train, %d sealed (arm=%s, radius_mult=%.2f)"
          % (len(tr), len(sl), args.arm, args.radius_mult))

    configs = []
    for da in (40, 100):
        configs.append(dict(tag="baseline hard", mode="hard", tau_low=0.0, tau_sep=0.0,
                            da=da, redistribute=False, exponent=0, phi_max=0, every=0))
    for tl, da in itertools.product((0.05, 0.10, 0.25), (40, 50)):
        configs.append(dict(tag="arm1 graded", mode="sigmoid_low", tau_low=tl, tau_sep=0.0,
                            da=da, redistribute=False, exponent=0, phi_max=0, every=0))
    for p_, da, ev in itertools.product((1.0, 2.0, 3.0), (100, 300, 1000), (5,)):
        configs.append(dict(tag="arm2 redistrib", mode="hard", tau_low=0.0, tau_sep=0.0,
                            da=da, redistribute=True, exponent=p_, phi_max=0.85, every=ev))
    for tl, p_, da in itertools.product((0.05, 0.10), (2.0, 3.0), (100, 300)):
        configs.append(dict(tag="arm1+arm2", mode="sigmoid_low", tau_low=tl, tau_sep=0.0,
                            da=da, redistribute=True, exponent=p_, phi_max=0.85, every=5))

    out = []
    print("\n%-15s %5s %6s %4s | %6s %6s %7s %7s | %7s"
          % ("config", "tauL", "da", "p", "rho", "bias", "sprRat", "curveL1", "score"))
    for cf in configs:
        rows = {a: run(c, bio, **{k: v for k, v in cf.items() if k != "tag"})
                for a, c in cache.items()}
        r = {k: agg(rows, tr, k) for k in ("rho", "bias", "mae", "spread_ratio",
                                           "curve_l1", "score")}
        r.update(cf, sealed_score=agg(rows, sl, "score"),
                 sealed_curve=agg(rows, sl, "curve_l1"), sealed_rho=agg(rows, sl, "rho"),
                 per_vessel={a: rows[a] for a in rows})
        out.append(r)
        print("%-15s %5.2f %6g %4g | %6.3f %6.3f %7.3f %7.4f | %7.4f"
              % (cf["tag"], cf["tau_low"], cf["da"], cf["exponent"],
                 r["rho"], r["bias"], r["spread_ratio"], r["curve_l1"], r["score"]))

    ok = [r for r in out if r["score"] >= 0.75 and 0.6 <= r["spread_ratio"] <= 1.7]
    print("\n-- configs holding train score >= 0.75 and spread_ratio in [0.6,1.7] --")
    for key, lbl in (("curve_l1", "best curve_l1"),):
        b = min(ok, key=lambda z: z[key]) if ok else None
        if b:
            print("%s: %s tau=%.2f da=%g p=%g -> curveL1 %.4f rho %.3f spr %.3f score %.4f"
                  " | SEALED curveL1 %.4f rho %.3f score %.4f"
                  % (lbl, b["tag"], b["tau_low"], b["da"], b["exponent"], b["curve_l1"],
                     b["rho"], b["spread_ratio"], b["score"], b["sealed_curve"],
                     b["sealed_rho"], b["sealed_score"]))
    b = max(ok, key=lambda z: z["rho"]) if ok else None
    if b:
        print("best rho     : %s tau=%.2f da=%g p=%g -> rho %.3f curveL1 %.4f spr %.3f score %.4f"
              " | SEALED rho %.3f score %.4f"
              % (b["tag"], b["tau_low"], b["da"], b["exponent"], b["rho"], b["curve_l1"],
                 b["spread_ratio"], b["score"], b["sealed_rho"], b["sealed_score"]))
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
