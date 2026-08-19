"""Correct ordering-vs-schedule decomposition, plus the readouts never tested.

SUPERSEDES the decomposition in docs/PHASE9_ML.md 13.6, which was WRONG: its "GT order"
arm passed the ODE's onset as the ordering, not GT's, so it measured the plain ODE arm and
the conclusion "perfect ordering buys nothing" was never actually tested.

Arms (all share the same locked-GNN set; only WHEN each node commits changes):

    frozen           no timing
    ode              ODE order  + ODE schedule
    learn_rank       learned order + ODE schedule            <- current best (13.3)
    learn_abs        learned onset used DIRECTLY, no quantile map   <- never tested
    learn_first      learned order + ODE schedule shifted so the FIRST commit matches GT
                     (PHASE6 15.4 found first-commit alignment worth more than every other
                     mechanism combined, and the metric prize is concentrated at t/T <= 0.4)
    gtorder_ode      GT order + ODE schedule       <- the real "perfect ordering" test
    learn_gtsched    learned order + GT schedule
    oracle           GT onset

    python scripts/eda_onset_decomposition.py
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

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
ARMS = ["frozen", "ode", "learn_rank", "learn_abs", "learn_first",
        "gtorder_ode", "learn_gtsched", "oracle"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="cv5a,cv5b,cv5c")
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--save", default="outputs/onset_decomposition.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    zs = [np.load(REPO / f"outputs/phase9_scores/{t}.npz", allow_pickle=True)
          for t in args.tags.split(",")]
    pool = [str(x) for x in zs[0]["pool"]]
    folds = {int(k.split("|")[1]): [str(x) for x in zs[0][k]]
             for k in zs[0].files if k.startswith("held|")}
    classes = classes_for(pool, PACKS)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)

    print("[i] precomputing ...", flush=True)
    V = {}
    for a in pool:
        S = cache[a]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
        gt = {ti: (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5) for ti in times}
        go = np.full(len(S["wall"]), T, dtype=int)
        for ti in reversed(times):
            go[gt[ti]] = ti
        traj, t = ode_trajectory(d, bio, flow="gt")
        r0 = traj[1] / max(t[1] - t[0], 1e-9)
        hot = traj >= crit
        oon = np.where(hot.any(0), hot.argmax(0), T)
        V[a] = dict(S=S, T=T, times=times, gt=gt, go=go, oon=oon, r0=r0)

    acc = {k: {"wall": [], "off": []} for k in ARMS}
    pv, rho = {}, {"learned": [], "ode": []}
    for k, held in folds.items():
        tr = [a for a in pool if a not in held]

        def F(a, f=k):
            v, S = V[a], V[a]["S"]
            sc = np.mean([z["%d|%s" % (f, a)] for z in zs], 0)
            return np.concatenate([S["X"], np.log1p(np.maximum(v["r0"], 0))[:, None],
                                   (v["oon"] / v["T"])[:, None], sc[:, None]], axis=1)

        Xtr = np.concatenate([F(a)[V[a]["go"] < V[a]["T"]] for a in tr])
        ytr = np.concatenate([(V[a]["go"] / V[a]["T"])[V[a]["go"] < V[a]["T"]] for a in tr])
        m = HistGradientBoostingRegressor(max_iter=250, max_depth=4, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=0).fit(Xtr, ytr)

        for a in held:
            v, S = V[a], V[a]["S"]
            wall, T = S["wall"], v["T"]
            sc = np.mean([z["%d|%s" % (k, a)] for z in zs], 0)
            gm = ((sc >= 0.73) & wall) | ((sc >= 0.92) & ~wall)
            idx = np.flatnonzero(gm)
            pred = m.predict(F(a))
            live = v["go"] < T
            if live.sum() >= 10:
                rho["learned"].append(spearman(pred[live], v["go"][live].astype(float)))
                rho["ode"].append(spearman(v["oon"][live].astype(float),
                                           v["go"][live].astype(float)))

            ref_ode = np.sort(v["oon"][gm & (v["oon"] < T)]).astype(float)
            ref_gt = np.sort(v["go"][gm & (v["go"] < T)]).astype(float)

            def sched(order, ref):
                on = np.full(len(order), T, dtype=int)
                if len(idx) >= 8 and len(ref) >= 8:
                    q = np.argsort(np.argsort(order[idx])) / max(len(idx) - 1, 1)
                    on[idx] = np.clip(np.interp(q, np.linspace(0, 1, len(ref)), ref),
                                      0, T - 1).astype(int)
                return on

            on = {}
            on["learn_rank"] = sched(pred, ref_ode)
            on["gtorder_ode"] = sched(v["go"].astype(float), ref_ode)   # REAL perfect order
            on["learn_gtsched"] = sched(pred, ref_gt)
            # absolute: use the regression output as a time directly
            ab = np.full(len(pred), T, dtype=int)
            ab[idx] = np.clip(pred[idx] * T, 0, T - 1).astype(int)
            on["learn_abs"] = ab
            # first-commit alignment: shift the ODE reference so its earliest commit
            # matches GT's earliest, keeping the shape
            if len(ref_ode) >= 8 and len(ref_gt) >= 8:
                on["learn_first"] = sched(pred, np.clip(ref_ode + (ref_gt[0] - ref_ode[0]),
                                                        0, T - 1))
            else:
                on["learn_first"] = on["learn_rank"]

            rows = {kk: {"wall": [], "off": []} for kk in ARMS}
            for ti in v["times"]:
                s_ = SeverityScorer(S["edge_index"], v["gt"][ti], len(wall), DEFAULT)
                M = {"frozen": gm, "ode": gm & ((v["oon"] <= ti) | ~wall),
                     "oracle": gm & (v["go"] <= ti)}
                for kk, o in on.items():
                    M[kk] = gm & (o <= ti)
                for kk, mm in M.items():
                    rows[kk]["wall"].append(s_.score(mm, wall))
                    rows[kk]["off"].append(s_.score(mm, ~wall))
            pv[a] = {kk: dict(wall=float(np.nanmean(rows[kk]["wall"])),
                              off=float(np.nanmean(rows[kk]["off"]))) for kk in ARMS}
            pv[a]["cls"] = classes.get(a, "?")
            for kk in ARMS:
                acc[kk]["wall"].append(pv[a][kk]["wall"])
                acc[kk]["off"].append(pv[a][kk]["off"])

    print("\nOUT-OF-FOLD, same GNN set (n=%d).  onset rank: learned %+.3f  ODE %+.3f\n"
          % (len(pv), np.nanmean(rho["learned"]), np.nanmean(rho["ode"])))
    print("%-16s %10s %10s" % ("arm", "wall", "off"))
    for kk in ARMS:
        print("%-16s %10.4f %10.4f"
              % (kk, np.nanmean(acc[kk]["wall"]), np.nanmean(acc[kk]["off"])))
    prio = [a for a in pv if is_priority(classes.get(a, ""))]
    print("\npriority (n=%d):" % len(prio))
    for kk in ARMS:
        print("%-16s %10.4f %10.4f"
              % (kk, np.nanmean([pv[a][kk]["wall"] for a in prio]),
                 np.nanmean([pv[a][kk]["off"] for a in prio])))

    Path(args.save).write_text(json.dumps(pv, indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
