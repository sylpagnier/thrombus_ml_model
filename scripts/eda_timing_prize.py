"""EDA: how much is intermediate-state timing worth, under the CURRENT metric?

Several "timing is hopeless" conclusions in PHASE6 were measured under a deploy score whose
empty-prediction cliff (`predict nothing` = 1.0 while GT is empty) dominated mean-over-time.
`SeverityScorer` now returns NaN on empty GT, so those timesteps are skipped and the cliff
is gone.  Every timing conclusion therefore needs re-measuring before it is trusted.

Arms, all scored as mean-over-time of the domain-restricted deploy score:

  frozen_model    the shipped v2 mask, replayed unchanged at every time  (what ships today)
  frozen_oracle   the GT FINAL mask, replayed unchanged                  (frozen ceiling)
  physics_onset   the zero-parameter ODE's onset: mask(t) = {onset <= t}
  oracle_onset    GT onset on the GT final set                           (= perfect timing)

The gap frozen_oracle -> oracle_onset is the timing prize on a perfect mask.  The gap
frozen_model -> physics_onset says whether the existing ODE already collects any of it.

    python scripts/eda_timing_prize.py
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

from predict_wall_clot import predict_wall_onset  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, eligible_pool, is_priority  # noqa: E402
from src.clot_ml.locked import load_ensemble, predict_scores  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
ARMS = ["frozen_model", "frozen_oracle", "physics_onset", "oracle_onset"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--save", default="outputs/eda_timing_prize.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    ens = load_ensemble()

    acc = {k: {"wall": [], "off": []} for k in ARMS}
    per_vessel = {}
    print("%-11s %7s | %s" % ("vessel", "class",
                              " ".join("%-13s" % a[:13] for a in ARMS)))
    for a in pool:
        S = cache[a]
        wall = S["wall"]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]

        gt = {ti: (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5) for ti in times}
        gt_final = gt[times[-1]]
        # GT onset index per node (first time it is labelled clot on the coarse grid)
        gt_onset = np.full(len(wall), len(times), dtype=int)
        for j, ti in enumerate(times):
            gt_onset = np.minimum(gt_onset, np.where(gt[ti], j, len(times)))

        # shipped model mask, thresholded as in the locked readout
        sc = predict_scores(ens, S)
        thr_w, thr_o = 0.73, 0.92
        model_mask = ((sc >= thr_w) & wall) | ((sc >= thr_o) & ~wall)

        # physics ODE onset (wall only -- the ODE is a wall object)
        try:
            m_ode, onset_ode, _ = predict_wall_onset(d, bio, flow="gt")
        except Exception:
            onset_ode = np.full(len(wall), -1)
            m_ode = np.zeros(len(wall), bool)
        # map ODE onset (index into T) onto the coarse grid
        ode_grid = np.full(len(wall), len(times), dtype=int)
        hot = onset_ode >= 0
        if hot.any():
            ode_grid[hot] = np.searchsorted(np.array(times), onset_ode[hot], side="left")

        rows = {k: {"wall": [], "off": []} for k in ARMS}
        for j, ti in enumerate(times):
            g = gt[ti]
            scorer = SeverityScorer(S["edge_index"], g, len(wall), DEFAULT)
            masks = {
                "frozen_model": model_mask,
                "frozen_oracle": gt_final,
                # ODE supplies WHEN on the wall; off-wall keeps the shipped mask so the
                # arm is not penalised for a domain the ODE does not model
                "physics_onset": ((ode_grid <= j) & wall) | (model_mask & ~wall),
                "oracle_onset": gt_onset <= j,
            }
            for k, m in masks.items():
                rows[k]["wall"].append(scorer.score(m, wall))
                rows[k]["off"].append(scorer.score(m, ~wall))
        res = {}
        for k in ARMS:
            res[k] = dict(wall=float(np.nanmean(rows[k]["wall"])),
                          off=float(np.nanmean(rows[k]["off"])))
            acc[k]["wall"].append(res[k]["wall"])
            acc[k]["off"].append(res[k]["off"])
        per_vessel[a] = dict(cls=classes.get(a, "?"), **res)
        print("%-11s %7s | %s" % (a, classes.get(a, "?")[:7],
              " ".join("%.3f/%-7s" % (res[k]["wall"],
                       ("%.3f" % res[k]["off"]) if res[k]["off"] == res[k]["off"] else "n/a")
                       for k in ARMS)))

    print("\nMEAN-OVER-TIME deploy score (severity), %d vessels" % len(pool))
    print("%-16s %10s %10s" % ("arm", "wall", "off"))
    for k in ARMS:
        print("%-16s %10.4f %10.4f"
              % (k, np.nanmean(acc[k]["wall"]), np.nanmean(acc[k]["off"])))
    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\npriority-class only (n=%d):" % len(prio))
    for k in ARMS:
        print("%-16s %10.4f %10.4f"
              % (k, np.nanmean([per_vessel[a][k]["wall"] for a in prio]),
                 np.nanmean([per_vessel[a][k]["off"] for a in prio])))

    Path(args.save).write_text(json.dumps(per_vessel, indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
