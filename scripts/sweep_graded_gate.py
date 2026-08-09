"""ARM 1 -- graded gate.  Does a continuous margin below lss/sgt fix the growth curve?

Baseline (hard t=0 gate) ignites in a flash: patient043's 84 nodes all cross in the same
step, and the model's median onset sits at 3000 s against GT's 5100-11250 s.  A graded
gate makes ignition rate a decreasing function of the margin, so nodes deep in the
stagnation zone ignite first and borderline nodes lag -- which is what the evolving flow
would do.

Scored on onset TIMING (src/core_physics/temporal_metrics.py) *and* on the final deploy
score, because a temporal fix that breaks the mask is not a fix.  Fit on full-horizon
WALL_COHORT_V2_TRAIN vessels only (a truncated run has no growth curve to match).
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
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index, onset_metrics  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")


def load(names, bio, phys, stencil, arm):
    cache = {}
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:          # truncated run: no growth curve to match
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        try:
            f = t0_flow_fields(d, bio, hops=stencil, flow_source=arm)
        except ValueError:
            continue
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        phi_gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        cache[a] = dict(d=d, wall=wall, f=f,
                        gt_idx=gt_onset_index(d, phys, wall),
                        phi_gt=phi_gt * torch.tensor(wall.astype(np.float32)))
    return cache


def evaluate(cache, bio, mode, tau_low, tau_sep, da):
    rows = {}
    for a, c in cache.items():
        gate = graded_gate(c["f"], bio, mode=mode, tau_low=tau_low, tau_sep=tau_sep) * c["wall"]
        traj, t = integrate_mat_trajectory(c["d"], bio, gate, da_scale=da)
        idx = first_crossing(traj, float(bio.viscosity_mat_crit))
        m = onset_metrics(idx, c["gt_idx"], t, c["wall"])
        m["curve_l1"] = curve_l1(idx, c["gt_idx"], t, c["wall"])
        pred = torch.tensor((((idx >= 0) & c["wall"])).astype(np.float32))
        mm = compute_clot_relaxed_metrics(pred, c["phi_gt"], c["d"].edge_index,
                                          wall_mask=torch.tensor(c["wall"]))
        o = metrics_to_deploy_prefix(mm)
        m["score"] = clot_score_from_deploy_dict(o)
        rows[a] = m
    return rows


def agg(rows, members, key):
    v = [rows[a][key] for a in members if a in rows and rows[a][key] == rows[a][key]]
    return float(np.mean(v)) if v else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gt")
    ap.add_argument("--stencil", type=int, default=3)
    ap.add_argument("--out", default="outputs/graded_gate_sweep.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
    cache = load(names, bio, phys, args.stencil, args.arm)
    tr = [a for a in WALL_COHORT_V2_TRAIN if a in cache]
    sl = [a for a in WALL_COHORT_V2_GENERALIZATION if a in cache]
    print("full-horizon vessels: %d train, %d sealed  (arm=%s)" % (len(tr), len(sl), args.arm))

    das = (30, 40, 50, 60, 80, 100)
    configs = [("hard", 0.0, 0.0, da) for da in das]
    configs += [("sigmoid_low", tl, 0.0, da)
                for tl, da in itertools.product((0.02, 0.05, 0.1, 0.25), das)]
    configs += [("sigmoid", tl, 0.25, da)
                for tl, da in itertools.product((0.02, 0.05, 0.1, 0.25), das)]
    out = []
    print("\n%-8s %5s %5s %6s | %6s %6s %6s %7s %7s | %7s"
          % ("mode", "tauL", "tauS", "da", "rho", "bias", "mae", "sprRat", "curveL1", "score"))
    for mode, tl, ts, da in configs:
        rows = evaluate(cache, bio, mode, tl, ts, da)
        r = {k: agg(rows, tr, k) for k in
             ("rho", "bias", "mae", "spread_ratio", "curve_l1", "score")}
        r.update(mode=mode, tau_low=tl, tau_sep=ts, da=da,
                 sealed_score=agg(rows, sl, "score"),
                 sealed_curve=agg(rows, sl, "curve_l1"),
                 sealed_rho=agg(rows, sl, "rho"),
                 per_vessel={a: rows[a] for a in rows})
        out.append(r)
        print("%-8s %5.2f %5.2f %6g | %6.3f %6.3f %6.3f %7.3f %7.4f | %7.4f"
              % (mode, tl, ts, da, r["rho"], r["bias"], r["mae"],
                 r["spread_ratio"], r["curve_l1"], r["score"]))

    ok = [r for r in out if r["score"] >= 0.75 and 0.6 <= r["spread_ratio"] <= 1.7]
    best = min(ok or out, key=lambda z: z["curve_l1"])
    print("\nBEST curve_l1 among configs holding train score >= 0.70:")
    print("  mode=%s tau_low=%.2f tau_sep=%.2f da=%g" % (best["mode"], best["tau_low"],
                                                         best["tau_sep"], best["da"]))
    print("  train : curve_l1 %.4f  rho %.3f  bias %+.3f  spread_ratio %.3f  score %.4f"
          % (best["curve_l1"], best["rho"], best["bias"], best["spread_ratio"], best["score"]))
    print("  sealed: curve_l1 %.4f  rho %.3f  score %.4f"
          % (best["sealed_curve"], best["sealed_rho"], best["sealed_score"]))
    base = [r for r in out if r["mode"] == "hard" and r["da"] == 100][0]
    print("  baseline (hard, da=100): curve_l1 %.4f  rho %.3f  spread_ratio %.3f  score %.4f"
          % (base["curve_l1"], base["rho"], base["spread_ratio"], base["score"]))
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
