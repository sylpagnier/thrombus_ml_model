"""A learned ONSET head: combine the ~0.6-rank signals that already exist.

Three independent orderings of onset are available and none is used together:

    r0, the t=0 physics rate      rank vs GT onset  +0.640
    the ODE's own crossing        rank vs GT onset  +0.642
    the GNN magnitude field       rho_corner out-of-fold 0.592

Rescaling the ODE trajectory by the GNN magnitude was tried and FAILED
(`scripts/eda_rescaled_ode.py`, wall 0.701 vs the ODE's 0.842) for a structural reason:
where the ODE flashes, the trajectory jumps in a single step, so changing the threshold
cannot move the crossing time.  Timing spread has to come from a per-node ORDERING, not a
per-node threshold.

So this predicts onset directly, as a static per-node regression on the same 56 features
plus the three physics orderings, leave-one-vessel-out.  No recurrence, no backprop through
the stiff ODE -- the two things that killed the earlier attempts.

Censoring is respected: only nodes with a real GT onset train the regression, and the
predicted ordering is mapped onto the ODE's own time distribution so the arm inherits a
physically plausible schedule rather than inventing one.

    python scripts/train_onset_head.py
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
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
ARMS = ["frozen", "ode", "learned", "oracle"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="cv5a,cv5b,cv5c")
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--save", default="outputs/onset_head.json")
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

    print("[i] building onset targets and physics orderings ...", flush=True)
    V = {}
    for a in pool:
        S = cache[a]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, args.n_times)]
        gt = {ti: (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5) for ti in times}
        gt_onset = np.full(len(S["wall"]), T, dtype=int)
        for ti in reversed(times):
            gt_onset[gt[ti]] = ti
        traj, t = ode_trajectory(d, bio, flow="gt")
        r0 = traj[1] / max(t[1] - t[0], 1e-9)
        hot = traj >= crit
        ode_on = np.where(hot.any(0), hot.argmax(0), T)
        V[a] = dict(S=S, T=T, times=times, gt=gt, gt_onset=gt_onset,
                    ode_on=ode_on, r0=r0)

    from sklearn.ensemble import HistGradientBoostingRegressor

    acc = {k: {"wall": [], "off": []} for k in ARMS}
    rho_learn, rho_ode, per_vessel = [], [], {}
    for k, held in folds.items():
        tr = [a for a in pool if a not in held]

        def feats(a, fold=k):
            v, S = V[a], V[a]["S"]
            sc = np.mean([z["%d|%s" % (fold, a)] for z in zs], 0)
            return np.concatenate([
                S["X"],
                np.log1p(np.maximum(v["r0"], 0))[:, None],
                (v["ode_on"] / v["T"])[:, None],
                sc[:, None]], axis=1)

        Xtr = np.concatenate([feats(a)[V[a]["gt_onset"] < V[a]["T"]] for a in tr])
        ytr = np.concatenate([(V[a]["gt_onset"] / V[a]["T"])[V[a]["gt_onset"] < V[a]["T"]]
                              for a in tr])
        m = HistGradientBoostingRegressor(max_iter=250, max_depth=4, learning_rate=0.06,
                                          l2_regularization=1.0,
                                          random_state=0).fit(Xtr, ytr)
        for a in held:
            v, S = V[a], V[a]["S"]
            wall, T = S["wall"], v["T"]
            sc = np.mean([z["%d|%s" % (k, a)] for z in zs], 0)
            gnn_mask = ((sc >= 0.73) & wall) | ((sc >= 0.92) & ~wall)
            pred = m.predict(feats(a))
            # map the predicted ORDERING onto the ODE's own time distribution, so the arm
            # inherits a physically plausible schedule instead of inventing one
            # Rank WITHIN the mask.  Ranking over all ~15k nodes puts every masked node at
            # the bottom of the global order, maps them all to the earliest ODE time, and
            # collapses the arm to `frozen` -- which is exactly what a first run showed
            # (0.7953, identical to frozen, despite rank +0.755).
            ref = np.sort(v["ode_on"][gnn_mask & (v["ode_on"] < T)])
            learned_on = np.full(len(pred), T, dtype=int)
            idx = np.flatnonzero(gnn_mask)
            if len(ref) >= 8 and len(idx) >= 8:
                q = np.argsort(np.argsort(pred[idx])) / max(len(idx) - 1, 1)
                learned_on[idx] = np.interp(q, np.linspace(0, 1, len(ref)), ref).astype(int)
            else:
                learned_on = v["ode_on"].copy()

            live = v["gt_onset"] < T
            if live.sum() >= 10:
                rho_learn.append(spearman(pred[live], v["gt_onset"][live].astype(float)))
                rho_ode.append(spearman(v["ode_on"][live].astype(float),
                                        v["gt_onset"][live].astype(float)))
            rows = {kk: {"wall": [], "off": []} for kk in ARMS}
            for ti in v["times"]:
                scorer = SeverityScorer(S["edge_index"], v["gt"][ti], len(wall), DEFAULT)
                masks = {
                    "frozen": gnn_mask,
                    "ode": gnn_mask & ((v["ode_on"] <= ti) | ~wall),
                    "learned": gnn_mask & (learned_on <= ti),
                    "oracle": gnn_mask & (v["gt_onset"] <= ti),
                }
                for kk, mm in masks.items():
                    rows[kk]["wall"].append(scorer.score(mm, wall))
                    rows[kk]["off"].append(scorer.score(mm, ~wall))
            res = {kk: dict(wall=float(np.nanmean(rows[kk]["wall"])),
                            off=float(np.nanmean(rows[kk]["off"]))) for kk in ARMS}
            for kk in ARMS:
                acc[kk]["wall"].append(res[kk]["wall"])
                acc[kk]["off"].append(res[kk]["off"])
            per_vessel[a] = dict(cls=classes.get(a, "?"), **res)
            print("   %-11s learned wall %.3f off %s" %
                  (a, res["learned"]["wall"],
                   ("%.3f" % res["learned"]["off"]) if res["learned"]["off"] == res["learned"]["off"]
                   else "n/a"), flush=True)

    print("\nMEAN-OVER-TIME, OUT-OF-FOLD, same GNN set (n=%d)\n" % len(per_vessel))
    print("%-10s %10s %10s" % ("arm", "wall", "off"))
    for kk in ARMS:
        print("%-10s %10.4f %10.4f"
              % (kk, np.nanmean(acc[kk]["wall"]), np.nanmean(acc[kk]["off"])))
    prio = [a for a in per_vessel if is_priority(classes.get(a, ""))]
    print("\npriority-class only (n=%d):" % len(prio))
    for kk in ARMS:
        print("%-10s %10.4f %10.4f"
              % (kk, np.nanmean([per_vessel[a][kk]["wall"] for a in prio]),
                 np.nanmean([per_vessel[a][kk]["off"] for a in prio])))
    print("\nonset rank vs GT (out-of-fold):  learned %+.3f   ODE %+.3f"
          % (np.nanmean(rho_learn), np.nanmean(rho_ode)))

    Path(args.save).write_text(json.dumps(per_vessel, indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
