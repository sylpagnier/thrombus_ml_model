"""Time-conditioned mask model: predict P(clot at time t), instead of predicting onset.

Every temporal arm so far factorises as ORDER x SCHEDULE, and the corrected decomposition
(`scripts/eda_onset_decomposition.py`) shows that factorisation is now exhausted:

    perfect ordering + ODE schedule   0.8568 / 0.5457   (+0.002 / +0.009 over learned)
    learned ordering + GT schedule    0.8908 / 0.6619   (+0.036 / +0.125)

so ordering is saturated and the schedule is the whole lever -- but the schedule cannot be
carried by 1-2 per-vessel moments (measured: oracle shift and affine both HURT) and a pooled
cohort curve is worse than the ODE's.  The remaining information is a per-vessel onset
DISTRIBUTION SHAPE that no imposed reference supplies.

This drops the factorisation entirely.  Time becomes an INPUT: the model predicts, for each
node and each time, whether that node is clot then.  The schedule is no longer imposed from
the ODE or a cohort curve -- it emerges from the features, so it can differ per vessel
without anyone having to parameterise how.

Physics kept in the formulation rather than in a post-hoc map:
  * the ODE's own crossing time and the t=0 rate ``r0`` enter as features, so the model can
    use the physics schedule where it is right and depart from it where it is not;
  * predictions are made MONOTONE in time by a cumulative maximum -- clot does not un-clot,
    which is a hard property of the production law (no sink) and one the onset formulation
    got for free but a per-time classifier does not;
  * only nodes that are ever plausibly clot are trained on, so the 0.7% base rate does not
    drown the signal.

    python scripts/train_time_conditioned.py
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

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
ARMS = ["frozen", "learn_rank", "timecond_mono", "timecond_tuned",
        "learn_gtsched", "oracle"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="cv5a,cv5b,cv5c")
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--save", default="outputs/time_conditioned.json")
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
    pv = {}
    for k, held in folds.items():
        tr = [a for a in pool if a not in held]

        def node_feats(a, f=k):
            v, S = V[a], V[a]["S"]
            sc = np.mean([z["%d|%s" % (f, a)] for z in zs], 0)
            return np.concatenate([S["X"], np.log1p(np.maximum(v["r0"], 0))[:, None],
                                   (v["oon"] / v["T"])[:, None], sc[:, None]], axis=1), sc

        def candidates(a, sc):
            v, S = V[a], V[a]["S"]
            wall = S["wall"]
            gm = ((sc >= 0.73) & wall) | ((sc >= 0.92) & ~wall)
            return gm | (v["go"] < v["T"])          # union of predicted and true clot

        # ---- build the (node, time) training table ----
        Xs, ys = [], []
        for a in tr:
            v = V[a]
            Xn, sc = node_feats(a)
            cand = candidates(a, sc)
            Xc = Xn[cand]
            for ti in v["times"]:
                tf = np.full((Xc.shape[0], 1), ti / v["T"], dtype=np.float32)
                # the ODE's own answer at this time, so the model can agree or depart
                ode_now = (v["oon"][cand] <= ti).astype(np.float32).reshape(-1, 1)
                Xs.append(np.concatenate([Xc, tf, ode_now], axis=1))
                ys.append(v["gt"][ti][cand])
        Xtr = np.concatenate(Xs)
        ytr = np.concatenate(ys)
        m = HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.07,
                                           l2_regularization=1.0, class_weight="balanced",
                                           random_state=0).fit(Xtr, ytr)

        # ---- per-domain thresholds, tuned on the fold's TRAINING vessels ----
        # Every readout in this project has needed separate wall / off-wall cuts: the
        # metric is domain-restricted and the two domains have very different base rates.
        def probs_for(a, times=None):
            v = V[a]
            Xn, sc = node_feats(a)
            out = np.zeros((len(v["times"]), Xn.shape[0]), dtype=np.float32)
            for j, ti in enumerate(v["times"]):
                tf = np.full((Xn.shape[0], 1), ti / v["T"], dtype=np.float32)
                ode_now = (v["oon"] <= ti).astype(np.float32).reshape(-1, 1)
                out[j] = m.predict_proba(np.concatenate([Xn, tf, ode_now], axis=1))[:, 1]
            return np.maximum.accumulate(out, axis=0), sc

        cachep = {a: probs_for(a) for a in tr}
        grid = np.linspace(0.05, 0.95, 19)

        def tune(domain_of):
            best, bt = -1e9, 0.5
            for th in grid:
                vals = []
                for a in tr:
                    v, S = V[a], V[a]["S"]
                    P_, sc_ = cachep[a]
                    gm_ = ((sc_ >= 0.73) & S["wall"]) | ((sc_ >= 0.92) & ~S["wall"])
                    dom = domain_of(S)
                    for j, ti in enumerate(v["times"]):
                        s2 = SeverityScorer(S["edge_index"], v["gt"][ti], len(S["wall"]),
                                            DEFAULT)
                        x = s2.score(gm_ & (P_[j] >= th), dom)
                        if x == x:
                            vals.append(x)
                if vals and np.mean(vals) > best:
                    best, bt = float(np.mean(vals)), float(th)
            return bt

        th_w = tune(lambda S: S["wall"])
        th_o = tune(lambda S: ~S["wall"])
        print("   fold %d thresholds  wall %.2f  off %.2f" % (k, th_w, th_o), flush=True)

        # ---- the onset-factorised reference arms ----
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xr = np.concatenate([node_feats(a)[0][V[a]["go"] < V[a]["T"]] for a in tr])
        yr = np.concatenate([(V[a]["go"] / V[a]["T"])[V[a]["go"] < V[a]["T"]] for a in tr])
        mr = HistGradientBoostingRegressor(max_iter=250, max_depth=4, learning_rate=0.06,
                                           l2_regularization=1.0, random_state=0).fit(Xr, yr)

        for a in held:
            v, S = V[a], V[a]["S"]
            wall, T = S["wall"], v["T"]
            Xn, sc = node_feats(a)
            gm = ((sc >= 0.73) & wall) | ((sc >= 0.92) & ~wall)
            idx = np.flatnonzero(gm)
            pred = mr.predict(Xn)
            ref_ode = np.sort(v["oon"][gm & (v["oon"] < T)]).astype(float)
            ref_gt = np.sort(v["go"][gm & (v["go"] < T)]).astype(float)

            def sched(order, ref):
                on = np.full(len(order), T, dtype=int)
                if len(idx) >= 8 and len(ref) >= 8:
                    q = np.argsort(np.argsort(order[idx])) / max(len(idx) - 1, 1)
                    on[idx] = np.clip(np.interp(q, np.linspace(0, 1, len(ref)), ref),
                                      0, T - 1).astype(int)
                return on

            on_rank = sched(pred, ref_ode)
            on_gts = sched(pred, ref_gt)

            # time-conditioned probabilities, and their monotone envelope
            P = np.zeros((len(v["times"]), len(wall)), dtype=np.float32)
            for j, ti in enumerate(v["times"]):
                tf = np.full((Xn.shape[0], 1), ti / T, dtype=np.float32)
                ode_now = (v["oon"] <= ti).astype(np.float32).reshape(-1, 1)
                P[j] = m.predict_proba(np.concatenate([Xn, tf, ode_now], axis=1))[:, 1]
            Pmono = np.maximum.accumulate(P, axis=0)

            rows = {kk: {"wall": [], "off": []} for kk in ARMS}
            for j, ti in enumerate(v["times"]):
                s_ = SeverityScorer(S["edge_index"], v["gt"][ti], len(wall), DEFAULT)
                M = {"frozen": gm,
                     "learn_rank": gm & (on_rank <= ti),
                     "learn_gtsched": gm & (on_gts <= ti),
                     "timecond_mono": gm & (Pmono[j] >= 0.5),
                     "timecond_tuned": gm & (Pmono[j] >= np.where(wall, th_w, th_o)),
                     "oracle": gm & (v["go"] <= ti)}
                for kk, mm in M.items():
                    rows[kk]["wall"].append(s_.score(mm, wall))
                    rows[kk]["off"].append(s_.score(mm, ~wall))
            pv[a] = {kk: dict(wall=float(np.nanmean(rows[kk]["wall"])),
                              off=float(np.nanmean(rows[kk]["off"]))) for kk in ARMS}
            pv[a]["cls"] = classes.get(a, "?")
            for kk in ARMS:
                acc[kk]["wall"].append(pv[a][kk]["wall"])
                acc[kk]["off"].append(pv[a][kk]["off"])
            print("   %-11s timecond wall %.3f  off %s" %
                  (a, pv[a]["timecond_mono"]["wall"],
                   ("%.3f" % pv[a]["timecond_mono"]["off"])
                   if pv[a]["timecond_mono"]["off"] == pv[a]["timecond_mono"]["off"] else "n/a"),
                  flush=True)

    print("\nOUT-OF-FOLD, same GNN set (n=%d)\n" % len(pv))
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
