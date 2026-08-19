"""Rescale the physics ODE trajectory to the GNN's magnitude, then take crossings.

WHY.  Out-of-fold `rho_corner` (rank of predicted vs GT wall `Mat` on species-carrying
corner nodes) is **0.592** for the locked GNN against **0.311** for the physics ODE.  The
GNN's magnitude field was never evaluated -- only its mask -- so PHASE9 12.5's claim that
everything terminates at a rho ~ 0.19 magnitude bottleneck is wrong.

That unlocks the synthesis every earlier arm was missing.  The ODE supplies a per-node
trajectory SHAPE that is physically correct but whose LEVEL is biased low (ratio 0.602), and
PHASE7 7.2 measured that pure calibration -- rank untouched -- was 53% of the score gap.  So
rescale each node's ODE trajectory to the magnitude the GNN predicts:

    Mat_i(t)  ->  s_i * traj_i(t),      s_i = Mat_GNN_i / traj_i(T)
    onset_i    =  first t with s_i * traj_i(t) >= crit
               =  first t with traj_i(t) >= crit / s_i

i.e. a per-node THRESHOLD adjustment on the trajectory that already ships.  It should break
the flash (equal-gate nodes now cross at different times) and, applied to an owner, give
off-wall timing.

The GNN score is a probability, not a magnitude, so a monotone quantile map score -> log Mat
is fitted **on each fold's training vessels only** and applied to the held-out one.  All
scores below are out-of-fold.

    python scripts/eda_rescaled_ode.py
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

PACKS = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10
ARMS = ["frozen", "ode", "rescaled", "rescaled_off", "oracle"]


def quantile_map(src_train, tgt_train, src_query):
    """Monotone map fitted on training values: rank in src -> same rank in tgt."""
    s = np.sort(src_train)
    t = np.sort(tgt_train)
    if len(s) < 8 or len(t) < 8:
        return np.full_like(src_query, np.median(t) if len(t) else 0.0, dtype=np.float64)
    q = np.searchsorted(s, src_query, side="left") / max(len(s) - 1, 1)
    return np.interp(np.clip(q, 0, 1), np.linspace(0, 1, len(t)), t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="cv5a,cv5b,cv5c")
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--att", type=float, default=0.16)
    ap.add_argument("--save", default="outputs/eda_rescaled_ode.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    tags = [t.strip() for t in args.tags.split(",")]
    zs = [np.load(REPO / f"outputs/phase9_scores/{t}.npz", allow_pickle=True) for t in tags]
    pool = [str(x) for x in zs[0]["pool"]]
    folds = {int(k.split("|")[1]): [str(x) for x in zs[0][k]]
             for k in zs[0].files if k.startswith("held|")}
    classes = classes_for(pool, PACKS)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)

    print("[i] precomputing trajectories and GT ...", flush=True)
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
        ch = d.y_channel_names.split(",")
        mat_gt = np.expm1(d.y[-1, :, ch.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        traj, _ = ode_trajectory(d, bio, flow="gt")
        V[a] = dict(S=S, T=T, times=times, gt=gt, gt_onset=gt_onset, traj=traj,
                    mat_gt=mat_gt, pos=S["pos"].astype(np.float64))

    from scipy.spatial import cKDTree

    acc = {k: {"wall": [], "off": []} for k in ARMS}
    per_vessel = {}
    for k, held in folds.items():
        tr = [a for a in pool if a not in held]
        src = np.concatenate([np.mean([z["%d|%s" % (k, a)] for z in zs], 0)[cache[a]["wall"]]
                              for a in tr])
        tgt = np.concatenate([np.log1p(np.maximum(V[a]["mat_gt"][cache[a]["wall"]], 0) / crit)
                              for a in tr])
        for a in held:
            v = V[a]
            S, wall, T = v["S"], v["S"]["wall"], v["T"]
            sc = np.mean([z["%d|%s" % (k, a)] for z in zs], 0)
            gnn_mask = ((sc >= 0.73) & wall) | ((sc >= 0.92) & ~wall)
            traj = v["traj"]

            mat_hat = np.expm1(np.clip(quantile_map(src, tgt, sc), 0, 40)) * crit
            final = np.maximum(traj[-1], 1e-30)
            s_i = np.where(final > 0, mat_hat / final, 0.0)
            resc = traj * s_i[None, :]

            def first(field, thr):
                hot = field >= thr
                return np.where(hot.any(0), hot.argmax(0), T)

            ode_on = first(traj, crit)
            res_on = first(resc, crit)
            res_hi = first(resc, crit / max(args.att, 1e-9))
            widx = np.flatnonzero(wall)
            owner = widx[cKDTree(v["pos"][wall]).query(v["pos"])[1]]
            off_on = res_hi[owner]

            rows = {kk: {"wall": [], "off": []} for kk in ARMS}
            for ti in v["times"]:
                scorer = SeverityScorer(S["edge_index"], v["gt"][ti], len(wall), DEFAULT)
                masks = {
                    "frozen": gnn_mask,
                    "ode": gnn_mask & ((ode_on <= ti) | ~wall),
                    "rescaled": gnn_mask & ((res_on <= ti) | ~wall),
                    "rescaled_off": gnn_mask & (np.where(wall, res_on, off_on) <= ti),
                    "oracle": gnn_mask & (v["gt_onset"] <= ti),
                }
                for kk, m in masks.items():
                    rows[kk]["wall"].append(scorer.score(m, wall))
                    rows[kk]["off"].append(scorer.score(m, ~wall))
            res = {kk: dict(wall=float(np.nanmean(rows[kk]["wall"])),
                            off=float(np.nanmean(rows[kk]["off"]))) for kk in ARMS}
            for kk in ARMS:
                acc[kk]["wall"].append(res[kk]["wall"])
                acc[kk]["off"].append(res[kk]["off"])
            per_vessel[a] = dict(cls=classes.get(a, "?"), **res)
            print("   %-11s resc wall %.3f  off %s" %
                  (a, res["rescaled"]["wall"],
                   ("%.3f" % res["rescaled_off"]["off"])
                   if res["rescaled_off"]["off"] == res["rescaled_off"]["off"] else "n/a"),
                  flush=True)

    print("\nMEAN-OVER-TIME, OUT-OF-FOLD, same GNN set (n=%d), att=%.2f\n"
          % (len(per_vessel), args.att))
    print("%-14s %10s %10s" % ("arm", "wall", "off"))
    for kk in ARMS:
        print("%-14s %10.4f %10.4f"
              % (kk, np.nanmean(acc[kk]["wall"]), np.nanmean(acc[kk]["off"])))
    prio = [a for a in per_vessel if is_priority(classes.get(a, ""))]
    print("\npriority-class only (n=%d):" % len(prio))
    for kk in ARMS:
        print("%-14s %10.4f %10.4f"
              % (kk, np.nanmean([per_vessel[a][kk]["wall"] for a in prio]),
                 np.nanmean([per_vessel[a][kk]["off"] for a in prio])))

    Path(args.save).write_text(json.dumps(per_vessel, indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
