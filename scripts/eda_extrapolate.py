"""EDA: is the Mat trajectory extrapolable AT ALL, given perfect history?

This is deliberately an ORACLE test.  It hands each extrapolator the true `Mat` history on
`t <= T_cut` and asks it to predict the clot mask at `T_end`, where `T_end / T_cut = 1.5` --
the same ratio as the real 30000 -> 45000 s ask.  It answers a question that comes strictly
before any modelling:

    if you knew the past exactly, could you get the future right?

If no extrapolator clears the frozen-mask baseline here, then 45000 s is not reachable by
extending a trajectory and the project should not build one.  If some family wins, that
family is the one worth predicting from t=0.

Families compared (all fitted per node on the GT history, no learning):

  freeze       Mat(T_end) = Mat(T_cut)            -- the do-nothing baseline
  linear       + last-window slope * dt           -- constant late rate
  loglinear    log Mat extended at its late slope -- constant relative rate
  sat_exp      Mat_inf * (1 - exp(-(t-t0)/tau))   -- saturating, 2 free params
  oracle       GT at T_end                        -- the ceiling (= 1.0 by construction)

    python scripts/eda_extrapolate.py --cut 0.667
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
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
MAT_S, CRIT = 7e10, 2e7


def fit_window_slope(t, y, lo):
    """Least-squares slope of each column of ``y`` over the window ``t >= lo``."""
    m = t >= lo
    tt, yy = t[m], y[m]
    tc = tt - tt.mean()
    denom = float((tc ** 2).sum()) or 1.0
    return (tc[:, None] * (yy - yy.mean(0))).sum(0) / denom


def extrapolators(t_hist, mat_hist, t_end, win_frac=0.35):
    """dict name -> predicted Mat at ``t_end`` [N]."""
    lo = t_hist[-1] - win_frac * (t_hist[-1] - t_hist[0])
    last = mat_hist[-1]
    dt = t_end - t_hist[-1]
    out = {"freeze": last.copy()}

    sl = fit_window_slope(t_hist, mat_hist, lo)
    out["linear"] = np.maximum(last + sl * dt, 0.0)

    lg = np.log(np.maximum(mat_hist, 1.0))
    slg = fit_window_slope(t_hist, lg, lo)
    out["loglinear"] = np.exp(np.minimum(np.log(np.maximum(last, 1.0)) + slg * dt, 60.0))

    # saturating: dMat/dt = (Mat_inf - Mat)/tau  =>  slope vs level is a straight line.
    # Regress slope on level over the window; slope = a + b*Mat with b<0 gives
    # Mat_inf = -a/b, tau = -1/b.  Falls back to linear where b >= 0 (still accelerating).
    m = t_hist >= lo
    tt, yy = t_hist[m], mat_hist[m]
    d = np.gradient(yy, tt, axis=0)
    ymean, dmean = yy.mean(0), d.mean(0)
    yc = yy - ymean
    b = (yc * (d - dmean)).sum(0) / np.maximum((yc ** 2).sum(0), 1e-9)
    a = dmean - b * ymean
    sat = np.where(b < -1e-12, np.where(b < 0, -a / np.minimum(b, -1e-12), last), np.inf)
    tau = np.where(b < -1e-12, -1.0 / np.minimum(b, -1e-12), np.inf)
    decayed = sat + (last - sat) * np.exp(-dt / np.maximum(tau, 1e-9))
    out["sat_exp"] = np.where(np.isfinite(decayed) & (b < -1e-12),
                              np.maximum(decayed, last), out["linear"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=float, default=0.667)
    ap.add_argument("--save", default="outputs/eda_extrapolate.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    phys = PhysicsConfig(phase="biochem")
    names_order = ["freeze", "linear", "loglinear", "sat_exp"]
    acc = {k: {"wall": [], "off": []} for k in names_order}
    per_vessel = {}

    print("Oracle extrapolation: fit on t <= %.0f%% of horizon, predict the END mask" %
          (100 * args.cut))
    print("%-11s %6s | %s" % ("vessel", "class",
                              "  ".join("%-13s" % n for n in names_order)))
    for a in pool:
        S = cache[a]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        ch = d.y_channel_names.split(",")
        mat = np.expm1(d.y[:, :, ch.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        t = d.t.reshape(-1).numpy().astype(np.float64)
        T = len(t)
        i_cut = int(round(args.cut * (T - 1)))
        gt_end = (gt_clot_phi_at_time(d, T - 1, phys, device=torch.device("cpu"))
                  .reshape(-1).numpy() > 0.5)
        wall = S["wall"]
        sc = SeverityScorer(S["edge_index"], gt_end, len(wall), DEFAULT)
        preds = extrapolators(t[:i_cut + 1], mat[:i_cut + 1], t[-1])
        row = {}
        for k in names_order:
            m = preds[k] >= CRIT
            row[k] = dict(wall=sc.score(m, wall), off=sc.score(m, ~wall),
                          n=int(m.sum()))
            acc[k]["wall"].append(row[k]["wall"])
            acc[k]["off"].append(row[k]["off"])
        per_vessel[a] = dict(cls=classes.get(a, "?"), n_gt=int(gt_end.sum()), **row)
        print("%-11s %6s | %s" % (a, classes.get(a, "?")[:6],
                                  "  ".join("%.3f/%-7s" % (row[k]["wall"],
                                            ("%.3f" % row[k]["off"]) if row[k]["off"] == row[k]["off"] else "n/a")
                                            for k in names_order)))

    print("\n(wall / off-wall deploy score at the END time, severity metric)")
    print("%-12s %10s %10s" % ("family", "wall", "off"))
    for k in names_order:
        print("%-12s %10.4f %10.4f"
              % (k, np.nanmean(acc[k]["wall"]), np.nanmean(acc[k]["off"])))

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\npriority-class only (n=%d):" % len(prio))
    for k in names_order:
        w = [per_vessel[a][k]["wall"] for a in prio]
        o = [per_vessel[a][k]["off"] for a in prio]
        print("%-12s %10.4f %10.4f" % (k, np.nanmean(w), np.nanmean(o)))

    Path(args.save).write_text(json.dumps(
        dict(cut=args.cut, per_vessel=per_vessel), indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
