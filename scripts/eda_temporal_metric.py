"""EDA: is "deploy score at each time interval" a usable temporal metric?

PHASE6_RESULTS 15.3 measured an empty-prediction CLIFF in the deploy score -- predicting
nothing is 1.0 while GT is empty and 0.0 the instant GT is not -- and concluded that
mean-over-time is dominated by *when you first commit* rather than by the growth curve.
That was measured on the legacy score.  We now also have `clot_severity_score`, whose
empty-GT branch is graded rather than a step.  This re-measures the question on both.

What it prints, per vessel and pooled, on a coarse time grid:

  * the GT growth timeline -- how much of the horizon is GT-empty (the degenerate region);
  * the score of an EMPTY prediction at each time (the cliff, if it exists);
  * the score of the ORACLE FINAL mask replayed at each time (the ceiling a frozen-mask
    model can reach, and how badly it is punished early);
  * the score of the ORACLE mask AT THAT TIME (a perfect temporal model = 1.0 by
    construction; reported as a sanity check that scoring is wired right).

    python scripts/eda_temporal_metric.py
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
from src.clot_ml.geometry_splits import eligible_pool  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"


def gt_series(anchor, times):
    d = torch.load(PACKS / f"{anchor}.pt", map_location="cpu", weights_only=False)
    phys = PhysicsConfig(phase="biochem")
    out = {}
    for ti in times:
        out[ti] = (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--save", default="outputs/eda_temporal_metric.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    pool = [a for a in eligible_pool() if a in cache]
    T = 201
    times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
    frac = [t / (T - 1) for t in times]

    curves = {"gt_frac": [], "empty_legacy": [], "empty_sev": [],
              "final_legacy": [], "final_sev": []}
    per_vessel = {}
    for a in pool:
        S = cache[a]
        wall = S["wall"]
        g = gt_series(a, times)
        n_fin = max(int(g[times[-1]].sum()), 1)
        final_mask = g[times[-1]]
        rows = {k: [] for k in curves}
        for ti in times:
            gt = g[ti]
            sc_l = SeverityScorer(S["edge_index"], gt, len(wall), LEGACY)
            sc_s = SeverityScorer(S["edge_index"], gt, len(wall), DEFAULT)
            empty = np.zeros(len(wall), bool)
            rows["gt_frac"].append(int(gt.sum()) / n_fin)
            rows["empty_legacy"].append(sc_l.score(empty, wall))
            rows["empty_sev"].append(sc_s.score(empty, wall))
            rows["final_legacy"].append(sc_l.score(final_mask, wall))
            rows["final_sev"].append(sc_s.score(final_mask, wall))
        per_vessel[a] = rows
        for k in curves:
            curves[k].append(rows[k])

    print("Wall-domain scores over time, pooled over %d vessels" % len(pool))
    print("(empty = predict nothing; final = replay the ORACLE FINAL mask at every time)\n")
    print("%6s %8s | %9s %9s | %9s %9s"
          % ("t/T", "GT frac", "empty leg", "empty sev", "final leg", "final sev"))
    M = {k: np.array(v, float) for k, v in curves.items()}
    for i, f in enumerate(frac):
        col = lambda k: np.nanmean(M[k][:, i])
        print("%6.2f %8.3f | %9.4f %9.4f | %9.4f %9.4f"
              % (f, col("gt_frac"), col("empty_legacy"), col("empty_sev"),
                 col("final_legacy"), col("final_sev")))

    print("\nmean-over-time of each curve:")
    for k in ("empty_legacy", "empty_sev", "final_legacy", "final_sev"):
        print("   %-14s %.4f" % (k, np.nanmean(M[k])))

    # How many vessels have a GT-empty region at all, and how long
    n_empty = [(np.array(per_vessel[a]["gt_frac"]) == 0).sum() for a in pool]
    print("\nGT-empty timesteps on the grid (of %d): min %d, median %d, max %d"
          % (len(times), min(n_empty), int(np.median(n_empty)), max(n_empty)))
    print("vessels with ANY GT-empty grid point: %d/%d"
          % (sum(1 for x in n_empty if x > 0), len(pool)))

    Path(args.save).write_text(json.dumps(
        dict(times=times, frac=frac, per_vessel=per_vessel), indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
