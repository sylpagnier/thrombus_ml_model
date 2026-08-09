"""ARM 2, corrected sign -- committed tissue sheds a stagnation WAKE, opening gates.

``diag_gt_shear_evolution.py`` measured the low-shear gate's open fraction RISING through
the run (patient032 0.000 -> 0.202, patient044 0.056 -> 0.256) while the shear magnitude
barely moves (median ratio 0.997).  That is the opposite of lumen narrowing, which would
accelerate the flow and close gates.  Committed tissue is a no-slip obstacle at 80x
viscosity: it sheds a stagnation wake.

So the feedback is ``sr -> sr * (1 - wake*phi)`` with ``phi`` the committed fraction of a
small neighbourhood.  ``diag_timevarying_gate_oracle.py`` bounds what any evolving-flow
model can win at +0.099 train deploy score; this measures how much of that an algebraic
rule recovers, with no network.

Everything fit on WALL_COHORT_V2_TRAIN full-horizon vessels; sealed spent once.
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
from src.core_physics.mls_gradient import node_positions  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, graded_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.shear_redistribution import (  # noqa: E402
    build_crosssection_operator, make_blockage, sdf_nd,
)
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index, onset_metrics  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gt")
    ap.add_argument("--out", default="outputs/wake_feedback_sweep.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
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
            f = t0_flow_fields(d, bio, hops=3, flow_source=args.arm)
        except ValueError:
            continue
        pos = node_positions(d)
        sd = sdf_nd(d)
        Bs = {rm: build_crosssection_operator(pos, sd, wall, radius_mult=rm)
              for rm in (0.2, 0.3, 0.4)}
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        pg = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        cache[a] = dict(d=d, wall=wall, f=f, Bs=Bs, gt_idx=gt_onset_index(d, phys, wall),
                        phi_gt=pg * torch.tensor(wall.astype(np.float32)))
    tr = [a for a in WALL_COHORT_V2_TRAIN if a in cache]
    sl = [a for a in WALL_COHORT_V2_GENERALIZATION if a in cache]
    print("full-horizon: %d train, %d sealed (arm=%s)" % (len(tr), len(sl), args.arm))

    def run(c, mode, tau, da, rm, wake, every):
        gate = graded_gate(c["f"], bio, mode=mode, tau_low=tau, tau_sep=0.0) * c["wall"]
        blk = None
        if wake > 0:
            blk = make_blockage(c["f"], bio, c["Bs"][rm], c["wall"], every=every,
                                graded_mode=mode, tau_low=tau, tau_sep=0.0,
                                feedback="wake", wake=wake)
        traj, t = integrate_mat_trajectory(c["d"], bio, gate, da_scale=da, blockage=blk)
        idx = first_crossing(traj, float(bio.viscosity_mat_crit))
        m = onset_metrics(idx, c["gt_idx"], t, c["wall"])
        m["curve_l1"] = curve_l1(idx, c["gt_idx"], t, c["wall"])
        pred = torch.tensor((((idx >= 0) & c["wall"])).astype(np.float32))
        mm = compute_clot_relaxed_metrics(pred, c["phi_gt"], c["d"].edge_index,
                                          wall_mask=torch.tensor(c["wall"]))
        m["score"] = clot_score_from_deploy_dict(metrics_to_deploy_prefix(mm))
        return m

    def agg(rows, sel, k):
        v = [rows[a][k] for a in sel if a in rows and rows[a][k] == rows[a][k]]
        return float(np.mean(v)) if v else float("nan")

    cfgs = [("hard", 0.0, 40, 0.3, 0.0, 5)]
    cfgs += [("hard", 0.0, da, rm, wk, 5)
             for da, rm, wk in itertools.product((30, 40, 50), (0.2, 0.3, 0.4),
                                                 (6.0, 8.0, 12.0, 16.0, 24.0))]
    cfgs += [("sigmoid_low", tau, da, rm, wk, 5)
             for tau, da, rm, wk in itertools.product((0.05, 0.10), (40,), (0.2, 0.3),
                                                      (8.0, 12.0, 16.0))]
    out = []
    print("\n%-12s %5s %5s %5s %5s | %6s %7s %7s | %7s %7s"
          % ("mode", "tau", "da", "rm", "wake", "rho", "sprRat", "curveL1", "score", "sealed"))
    for mode, tau, da, rm, wk, ev in cfgs:
        rows = {a: run(c, mode, tau, da, rm, wk, ev) for a, c in cache.items()}
        r = {k: agg(rows, tr, k) for k in ("rho", "spread_ratio", "curve_l1", "score")}
        r.update(mode=mode, tau=tau, da=da, rm=rm, wake=wk,
                 sealed_score=agg(rows, sl, "score"), sealed_rho=agg(rows, sl, "rho"),
                 sealed_curve=agg(rows, sl, "curve_l1"),
                 per_vessel={a: rows[a]["score"] for a in rows})
        out.append(r)
        print("%-12s %5.2f %5g %5.2f %5.1f | %6.3f %7.3f %7.4f | %7.4f %7.4f"
              % (mode, tau, da, rm, wk, r["rho"], r["spread_ratio"], r["curve_l1"],
                 r["score"], r["sealed_score"]))

    base = out[0]
    ok = [r for r in out if 0.7 <= r["spread_ratio"] <= 1.4 and r["curve_l1"] <= 0.11]
    best = max(ok or out, key=lambda z: z["score"])
    print("\nbaseline (no wake)   : score %.4f rho %.3f curveL1 %.4f | sealed %.4f"
          % (base["score"], base["rho"], base["curve_l1"], base["sealed_score"]))
    print("best train score s.t. spread in [0.7,1.4] and curveL1 <= 0.11:")
    print("  mode=%s tau=%.2f da=%g rm=%.2f wake=%.1f"
          % (best["mode"], best["tau"], best["da"], best["rm"], best["wake"]))
    print("   train  score %.4f  rho %.3f  curveL1 %.4f  spread %.3f"
          % (best["score"], best["rho"], best["curve_l1"], best["spread_ratio"]))
    print("   SEALED score %.4f  rho %.3f  curveL1 %.4f"
          % (best["sealed_score"], best["sealed_rho"], best["sealed_curve"]))
    print("   oracle ceiling (perfect evolving flow): train 0.8913, sealed 0.9066")
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
