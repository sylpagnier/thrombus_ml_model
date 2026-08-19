"""Score the time-resolved arms: does ODE timing on the GNN set beat the frozen mask?

    frozen        the locked GNN mask, replayed unchanged at every time (ships today)
    ode_set       the ODE's own set AND timing (wall only; off-wall keeps the GNN mask)
    gnn_ode       the GNN set, scheduled by ODE timing, WALL only
    gnn_ode_off   ... and off-wall scheduled by the owner's crit/att crossing
    oracle        perfect timing on the GT set

    python scripts/eval_temporal_arms.py --att 0.16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, eligible_pool, is_priority  # noqa: E402
from src.clot_ml.locked import load_ensemble, predict_scores  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.temporal import _first_crossing, ode_trajectory, onset_from_ode  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
ARMS = ["frozen", "ode_set", "gnn_ode", "gnn_ode_off", "oracle"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", type=float, default=0.16)
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--save", default="outputs/eval_temporal_arms.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    ens = load_ensemble()
    acc = {k: {"wall": [], "off": []} for k in ARMS}
    per_vessel = {}

    for a in pool:
        S = cache[a]
        wall, pos = S["wall"], S["pos"].astype(np.float64)
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
        gt = {ti: (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5) for ti in times}
        gt_final = gt[times[-1]]
        gt_onset = np.full(len(wall), T, dtype=int)
        for ti in reversed(times):
            gt_onset[gt[ti]] = ti

        sc = predict_scores(ens, S)
        gnn_mask = ((sc >= 0.73) & wall) | ((sc >= 0.92) & ~wall)

        traj, _ = ode_trajectory(d, bio, flow="gt")
        on_wall_only = onset_from_ode(traj, gnn_mask, wall, pos, crit, attenuation=1e9)
        on_both = onset_from_ode(traj, gnn_mask, wall, pos, crit, attenuation=args.att)
        ode_mask = (_first_crossing(traj, crit) >= 0) & wall
        on_ode = onset_from_ode(traj, ode_mask, wall, pos, crit, attenuation=1e9)

        rows = {k: {"wall": [], "off": []} for k in ARMS}
        for ti in times:
            scorer = SeverityScorer(S["edge_index"], gt[ti], len(wall), DEFAULT)
            masks = {
                "frozen": gnn_mask,
                "ode_set": (ode_mask & (on_ode >= 0) & (on_ode <= ti)) | (gnn_mask & ~wall),
                # GNN set, ODE timing on the wall; off-wall frozen
                "gnn_ode": (gnn_mask & wall & (on_wall_only <= ti)) | (gnn_mask & ~wall),
                # ... and off-wall scheduled by the owner's crit/att crossing
                "gnn_ode_off": gnn_mask & (on_both >= 0) & (on_both <= ti),
                "oracle": gt_onset <= ti,
            }
            for k, m in masks.items():
                rows[k]["wall"].append(scorer.score(m, wall))
                rows[k]["off"].append(scorer.score(m, ~wall))
        res = {k: dict(wall=float(np.nanmean(rows[k]["wall"])),
                       off=float(np.nanmean(rows[k]["off"]))) for k in ARMS}
        for k in ARMS:
            acc[k]["wall"].append(res[k]["wall"])
            acc[k]["off"].append(res[k]["off"])
        per_vessel[a] = dict(cls=classes.get(a, "?"), **res)

    print("MEAN-OVER-TIME deploy score (severity), %d vessels, att=%.2f\n" % (len(pool), args.att))
    print("%-14s %10s %10s" % ("arm", "wall", "off"))
    for k in ARMS:
        print("%-14s %10.4f %10.4f"
              % (k, np.nanmean(acc[k]["wall"]), np.nanmean(acc[k]["off"])))
    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\npriority-class only (n=%d):" % len(prio))
    for k in ARMS:
        print("%-14s %10.4f %10.4f"
              % (k, np.nanmean([per_vessel[a][k]["wall"] for a in prio]),
                 np.nanmean([per_vessel[a][k]["off"] for a in prio])))

    print("\nper-vessel gnn_ode_off vs frozen:")
    print("%-11s %-9s %16s %16s" % ("vessel", "class", "wall", "off"))
    for a in pool:
        f_, g_ = per_vessel[a]["frozen"], per_vessel[a]["gnn_ode_off"]
        fmt = lambda x, y: "%.3f->%.3f" % (x, y) if x == x else "     n/a   "
        print("%-11s %-9s %16s %16s"
              % (a, classes.get(a, "?")[:9], fmt(f_["wall"], g_["wall"]),
                 fmt(f_["off"], g_["off"])))

    Path(args.save).write_text(json.dumps(per_vessel, indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
