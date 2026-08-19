"""Onset head v2: three physics constraints on top of the learned ordering.

v1 (`scripts/train_onset_head.py`) learns an ordering (rank +0.755) and maps it onto the
ODE's own onset distribution.  Its per-vessel breakdown shows exactly where that breaks:

  * on p020 / p021 / p029 / p032 / p037 the arm is **bit-identical to frozen** -- every
    masked node schedules at t~0.  The schedule reference is the ODE's wall onsets, and on
    a flash vessel that distribution is a single point, so the map is degenerate.
  * off-wall nodes are mapped into the SAME distribution as wall nodes, so an off-wall node
    can be scheduled before the wall node feeding it -- physically impossible.
  * p020 alone carries +0.69 of off-wall headroom.

Three additions, each a physical statement rather than a hyperparameter:

  ``--schedule curve``   Map the ordering onto the COHORT GROWTH CURVE (fraction committed
                         vs t/T, fitted on the fold's training vessels) instead of the ODE's
                         onset histogram.  Clot growth is a smooth S-curve in every vessel;
                         the ODE's flash is a model artefact, and using it as the reference
                         distribution imports that artefact into the schedule.

  ``--follow-owner``     Force ``onset_offwall >= onset_owner``.  Off-wall `Mat` is fed by
                         the wall node it sits behind, so it cannot commit first.  This is
                         the same constraint the shipped entry point already enforces.

  ``--smooth W``         Blend the predicted onset with its 1-hop mean.  Onset is a smooth
                         field -- a clot front, not a speckle -- and 1-hop smoothing of the
                         flux integral was measured worth +0.094 rank (PHASE9 12.3).

    python scripts/train_onset_head_v2.py --schedule curve --follow-owner --smooth 0.3
    python scripts/train_onset_head_v2.py --ablate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

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
NQ = 41                      # quantile grid for the cohort growth curve


def norm_adj(ei, n):
    A = sp.coo_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0.0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg).astype(np.float32) @ A


def cohort_growth_curve(train_vs):
    """Quantile function of GT onset in t/T units, pooled over training vessels.

    Returns the time (in t/T) by which each quantile of committed nodes has committed --
    i.e. the inverse of the growth curve, which is exactly what a rank -> time map needs.
    """
    fr = []
    for v in train_vs:
        on = v["gt_onset"]
        live = on < v["T"]
        if live.sum() >= 8:
            fr.append(np.quantile(on[live] / v["T"], np.linspace(0, 1, NQ)))
    if not fr:
        return np.linspace(0, 1, NQ)
    return np.mean(np.stack(fr), axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="cv5a,cv5b,cv5c")
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--schedule", default="curve", choices=["ode", "curve"])
    ap.add_argument("--follow-owner", action="store_true")
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--save", default="outputs/onset_head_v2.json")
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
        gt_onset = np.full(len(S["wall"]), T, dtype=int)
        for ti in reversed(times):
            gt_onset[gt[ti]] = ti
        traj, t = ode_trajectory(d, bio, flow="gt")
        r0 = traj[1] / max(t[1] - t[0], 1e-9)
        hot = traj >= crit
        ode_on = np.where(hot.any(0), hot.argmax(0), T)
        widx = np.flatnonzero(S["wall"])
        pos = S["pos"].astype(np.float64)
        owner = widx[cKDTree(pos[S["wall"]]).query(pos)[1]] if len(widx) else np.arange(len(pos))
        V[a] = dict(S=S, T=T, times=times, gt=gt, gt_onset=gt_onset, ode_on=ode_on,
                    r0=r0, owner=owner, A=norm_adj(S["edge_index"], len(S["wall"])))

    from sklearn.ensemble import HistGradientBoostingRegressor

    configs = ([dict(schedule="ode", follow=False, smooth=0.0),
                dict(schedule="curve", follow=False, smooth=0.0),
                dict(schedule="curve", follow=True, smooth=0.0),
                dict(schedule="curve", follow=True, smooth=0.3)]
               if args.ablate else
               [dict(schedule=args.schedule, follow=args.follow_owner, smooth=args.smooth)])
    names = ["frozen"] + ["cfg%d" % i for i in range(len(configs))] + ["oracle"]
    acc = {k: {"wall": [], "off": []} for k in names}
    rho_learn, per_vessel = [], {a: {} for a in pool}

    for k, held in folds.items():
        tr = [a for a in pool if a not in held]

        def feats(a, fold=k):
            v, S = V[a], V[a]["S"]
            sc = np.mean([z["%d|%s" % (fold, a)] for z in zs], 0)
            return np.concatenate([S["X"], np.log1p(np.maximum(v["r0"], 0))[:, None],
                                   (v["ode_on"] / v["T"])[:, None], sc[:, None]], axis=1)

        Xtr = np.concatenate([feats(a)[V[a]["gt_onset"] < V[a]["T"]] for a in tr])
        ytr = np.concatenate([(V[a]["gt_onset"] / V[a]["T"])[V[a]["gt_onset"] < V[a]["T"]]
                              for a in tr])
        m = HistGradientBoostingRegressor(max_iter=250, max_depth=4, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
        curve = cohort_growth_curve([V[a] for a in tr])

        for a in held:
            v, S = V[a], V[a]["S"]
            wall, T = S["wall"], v["T"]
            sc = np.mean([z["%d|%s" % (k, a)] for z in zs], 0)
            gnn_mask = ((sc >= 0.73) & wall) | ((sc >= 0.92) & ~wall)
            pred = m.predict(feats(a))
            live = v["gt_onset"] < T
            if live.sum() >= 10:
                rho_learn.append(spearman(pred[live], v["gt_onset"][live].astype(float)))

            idx = np.flatnonzero(gnn_mask)
            onsets = {}
            for ci, cfg in enumerate(configs):
                p = pred.copy()
                if cfg["smooth"] > 0:
                    p = (1 - cfg["smooth"]) * p + cfg["smooth"] * (v["A"] @ p)
                on = np.full(len(p), T, dtype=int)
                if len(idx) >= 8:
                    q = np.argsort(np.argsort(p[idx])) / max(len(idx) - 1, 1)
                    if cfg["schedule"] == "curve":
                        tt = np.interp(q, np.linspace(0, 1, NQ), curve) * T
                    else:
                        ref = np.sort(v["ode_on"][gnn_mask & (v["ode_on"] < T)])
                        tt = (np.interp(q, np.linspace(0, 1, len(ref)), ref)
                              if len(ref) >= 8 else np.full(len(idx), T / 2))
                    on[idx] = np.clip(tt, 0, T - 1).astype(int)
                if cfg["follow"]:
                    off = gnn_mask & ~wall
                    on[off] = np.maximum(on[off], on[v["owner"]][off])
                onsets["cfg%d" % ci] = on

            rows = {kk: {"wall": [], "off": []} for kk in names}
            for ti in v["times"]:
                scorer = SeverityScorer(S["edge_index"], v["gt"][ti], len(wall), DEFAULT)
                masks = {"frozen": gnn_mask, "oracle": gnn_mask & (v["gt_onset"] <= ti)}
                for kk, on in onsets.items():
                    masks[kk] = gnn_mask & (on <= ti)
                for kk, mm in masks.items():
                    rows[kk]["wall"].append(scorer.score(mm, wall))
                    rows[kk]["off"].append(scorer.score(mm, ~wall))
            for kk in names:
                r = dict(wall=float(np.nanmean(rows[kk]["wall"])),
                         off=float(np.nanmean(rows[kk]["off"])))
                per_vessel[a][kk] = r
                acc[kk]["wall"].append(r["wall"])
                acc[kk]["off"].append(r["off"])
            per_vessel[a]["cls"] = classes.get(a, "?")

    print("\nMEAN-OVER-TIME, OUT-OF-FOLD (n=%d).  onset rank +%.3f\n"
          % (len(pool), np.nanmean(rho_learn)))
    label = {"frozen": "frozen", "oracle": "oracle"}
    for i, c in enumerate(configs):
        label["cfg%d" % i] = "%s%s%s" % (c["schedule"],
                                         "+owner" if c["follow"] else "",
                                         "+sm%.1f" % c["smooth"] if c["smooth"] else "")
    print("%-18s %10s %10s" % ("arm", "wall", "off"))
    for kk in names:
        print("%-18s %10.4f %10.4f"
              % (label[kk], np.nanmean(acc[kk]["wall"]), np.nanmean(acc[kk]["off"])))
    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\npriority-class only (n=%d):" % len(prio))
    for kk in names:
        print("%-18s %10.4f %10.4f"
              % (label[kk], np.nanmean([per_vessel[a][kk]["wall"] for a in prio]),
                 np.nanmean([per_vessel[a][kk]["off"] for a in prio])))

    Path(args.save).write_text(json.dumps(per_vessel, indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
