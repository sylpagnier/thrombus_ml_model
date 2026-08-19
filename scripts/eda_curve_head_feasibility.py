"""EDA: is a monotone curve head worth building, and are its parameters predictable?

The proposal (docs/PHASE9_ML.md 12) is to replace the scalar head with a per-node monotone
trajectory whose EARLY rate is fixed by physics and whose later ramp is learned.  Three
things have to be true, in order, and this measures all three before anything is trained:

  1. CEILING.  What does perfect timing on OUR OWN set score?  If the set caps the
     mean-over-time score near where the ODE already sits, timing work is finished.
  2. PARAMETERISATION.  Does a two-phase curve -- physics rate ``r0`` until ``c``, then a
     constant rate ``r1`` -- reproduce GT onset when FITTED ON GT?  That is the ceiling of
     the family itself, independent of any predictor.  The form is not arbitrary: `r0` is
     exactly computable at t=0 (`Sat = 1`, autocatalysis zero) and the late tail measured
     linear (`lin_r2` 0.936, `scripts/eda_extrapolate.py`).
  3. PREDICTABILITY.  Are ``(c, r1)`` predictable from t=0?  The late RATE alone was not
     (rho 0.04, measured earlier).  If the ramp is not either, the head buys nothing over
     the ODE and the honest answer is to stop.

Arms, all scored mean-over-time on the SAME locked-GNN set so only timing varies:

    frozen        no timing at all
    ode           the zero-parameter ODE's crossing        (ships as of PHASE9 12.2)
    r0_only       onset = crit / r0, pure t=0 physics, no integration
    twophase      the two-phase curve FITTED ON GT         (family ceiling)
    oracle_time   GT onset                                 (timing ceiling on our set)

    python scripts/eda_curve_head_feasibility.py
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
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10
ARMS = ["frozen", "ode", "r0_only", "twophase", "oracle_time"]


def fit_two_phase(t, mat, r0, n_c=12):
    """Per node: pick ``c`` on a grid, solve ``r1`` in closed form, keep the best.

    ``Mat(t) = r0*min(t,c) + r1*max(t-c,0)`` with ``r0`` FIXED from physics, so only the
    late rate and the switch time are free.
    """
    T, N = mat.shape
    best_sse = np.full(N, np.inf)
    best_c = np.zeros(N)
    best_r1 = np.zeros(N)
    for c in np.linspace(t[1], t[-1] * 0.9, n_c):
        base = r0[None, :] * np.minimum(t, c)[:, None]
        x = np.maximum(t - c, 0.0)[:, None]
        resid = mat - base
        denom = float((x[:, 0] ** 2).sum()) or 1.0
        r1 = np.maximum((x * resid).sum(0) / denom, 0.0)          # monotone: r1 >= 0
        sse = ((base + r1[None, :] * x - mat) ** 2).sum(0)
        upd = sse < best_sse
        best_sse[upd], best_c[upd], best_r1[upd] = sse[upd], c, r1[upd]
    return best_c, best_r1, best_sse


def onset_from_two_phase(t, r0, c, r1, crit):
    """First time the two-phase curve reaches ``crit``; ``len(t)`` if never."""
    T = len(t)
    at_c = r0 * c
    idx = np.full(len(r0), T, dtype=int)
    early = crit / np.maximum(r0, 1e-30)
    use_early = (at_c >= crit) & (r0 > 0)
    tt = np.where(use_early, early, c + (crit - at_c) / np.maximum(r1, 1e-30))
    ok = np.isfinite(tt) & (tt <= t[-1]) & ((r0 > 0) | (r1 > 0))
    idx[ok] = np.searchsorted(t, tt[ok], side="left").clip(0, T - 1)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--save", default="outputs/eda_curve_head.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    ens = load_ensemble()

    acc = {k: {"wall": [], "off": []} for k in ARMS}
    fitq, rho_rows, per_vessel = [], [], {}

    for a in pool:
        S = cache[a]
        wall = S["wall"]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
        gt = {ti: (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5) for ti in times}
        gt_onset = np.full(len(wall), T, dtype=int)
        for ti in reversed(times):
            gt_onset[gt[ti]] = ti

        ch = d.y_channel_names.split(",")
        mat_gt = np.expm1(d.y[:, :, ch.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        t = d.t.reshape(-1).numpy().astype(np.float64)

        sc = predict_scores(ens, S)
        gnn_mask = ((sc >= 0.73) & wall) | ((sc >= 0.92) & ~wall)

        traj, _ = ode_trajectory(d, bio, flow="gt")
        r0 = traj[1] / max(t[1] - t[0], 1e-9)          # pure t=0 physics rate
        hot = traj >= crit
        ode_on = np.where(hot.any(0), hot.argmax(0), T)

        c, r1, sse = fit_two_phase(t, mat_gt, r0)
        tp_on = onset_from_two_phase(t, r0, c, r1, crit)

        # family fit quality + parameter/rank diagnostics on nodes that actually clot
        live = gt_onset < T
        if live.sum() >= 10:
            st = ((mat_gt[:, live] - mat_gt[:, live].mean(0)) ** 2).sum()
            fitq.append(1.0 - sse[live].sum() / max(st, 1e-30))
            rho_rows.append(dict(
                r0_vs_onset=spearman(-r0[live], gt_onset[live].astype(float)),
                ode_vs_onset=spearman(ode_on[live].astype(float), gt_onset[live].astype(float)),
                tp_vs_onset=spearman(tp_on[live].astype(float), gt_onset[live].astype(float)),
                r0_vs_r1=spearman(r0[live], r1[live]),
                r0_vs_c=spearman(r0[live], c[live])))

        r0_on = np.searchsorted(t, crit / np.maximum(r0, 1e-30), side="left").clip(0, T)
        r0_on = np.where(r0 > 0, r0_on, T)

        rows = {k: {"wall": [], "off": []} for k in ARMS}
        for ti in times:
            scorer = SeverityScorer(S["edge_index"], gt[ti], len(wall), DEFAULT)
            masks = {
                "frozen": gnn_mask,
                "ode": gnn_mask & (ode_on <= ti),
                "r0_only": gnn_mask & (r0_on <= ti),
                "twophase": gnn_mask & (tp_on <= ti),
                "oracle_time": gnn_mask & (gt_onset <= ti),
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

    print("MEAN-OVER-TIME deploy score, SAME locked-GNN set, only timing varies (n=%d)\n"
          % len(pool))
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

    print("\n[2] two-phase family, FITTED ON GT: R2 of the trajectory  %.4f"
          % float(np.mean(fitq)))
    g = lambda k: np.nanmean([r[k] for r in rho_rows])
    print("\n[3] rank correlations vs GT onset (higher |rho| = better ordering)")
    print("   r0 alone (t=0 physics)      %+.3f" % g("r0_vs_onset"))
    print("   ODE crossing                %+.3f" % g("ode_vs_onset"))
    print("   two-phase fitted on GT      %+.3f" % g("tp_vs_onset"))
    print("\n[4] is the learned part predictable FROM r0? (if |rho| high, it is redundant)")
    print("   rho(r0, r1)                 %+.3f" % g("r0_vs_r1"))
    print("   rho(r0, c)                  %+.3f" % g("r0_vs_c"))

    Path(args.save).write_text(json.dumps(
        dict(per_vessel=per_vessel, fit_r2=float(np.mean(fitq)),
             rho={k: float(g(k)) for k in rho_rows[0]}), indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
