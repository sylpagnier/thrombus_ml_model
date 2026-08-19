"""EDA: are the two-phase curve parameters predictable from t=0, out of fold?

`scripts/eda_curve_head_feasibility.py` established the two halves of the case:

  * perfect timing on OUR OWN set scores wall 0.9875 / off 0.9715 -- the set is not the
    limit, timing is;
  * the two-phase curve FITTED ON GT reaches off-wall 0.8257 against frozen's 0.5015, so
    the family is expressive enough;
  * but the learned part is NOT redundant with the physics rate: rho(r0, r1) = +0.105.

So everything now turns on one question: can ``(c, r1)`` be predicted from t=0?  This fits
a predictor on the SAME 56 features the locked GNN uses, leave-one-vessel-out, and scores
the resulting timing end to end against the GT-fitted ceiling.

    predicted      (c, r1) from t=0 features, out of fold      <- the deployable number
    oracle_r1      c predicted, r1 taken from GT               } which half is the blocker
    oracle_c       r1 predicted, c taken from GT               }
    gt_params      both from GT                                <- the family ceiling

    python scripts/eda_curve_param_predictability.py
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

from eda_curve_head_feasibility import fit_two_phase, onset_from_two_phase  # noqa: E402
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
ARMS = ["frozen", "predicted", "oracle_r1", "oracle_c", "gt_params"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-times", type=int, default=11)
    ap.add_argument("--save", default="outputs/eda_curve_param_pred.json")
    args = ap.parse_args()

    cache = attach_physics(load_cache("gt"))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    ens = load_ensemble()

    print("[i] building per-node targets (two-phase fit on GT) ...", flush=True)
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
        mat_gt = np.expm1(d.y[:, :, ch.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        t = d.t.reshape(-1).numpy().astype(np.float64)
        traj, _ = ode_trajectory(d, bio, flow="gt")
        r0 = traj[1] / max(t[1] - t[0], 1e-9)
        c, r1, _ = fit_two_phase(t, mat_gt, r0)
        sc = predict_scores(ens, S)
        gnn_mask = ((sc >= 0.73) & S["wall"]) | ((sc >= 0.92) & ~S["wall"])
        V[a] = dict(S=S, t=t, r0=r0, c=c, r1=r1, gt=gt, gt_onset=gt_onset, T=T,
                    times=times, mask=gnn_mask)

    from sklearn.ensemble import HistGradientBoostingRegressor

    acc = {k: {"wall": [], "off": []} for k in ARMS}
    rho_c, rho_r1, per_vessel = [], [], {}
    for held in pool:
        tr = [a for a in pool if a != held]
        Xtr = np.concatenate([V[a]["S"]["X"] for a in tr])
        ctr = np.concatenate([V[a]["c"] for a in tr])
        r1tr = np.log1p(np.maximum(np.concatenate([V[a]["r1"] for a in tr]), 0.0))
        mc = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.08,
                                           random_state=0).fit(Xtr, ctr)
        mr = HistGradientBoostingRegressor(max_iter=200, max_depth=4, learning_rate=0.08,
                                           random_state=0).fit(Xtr, r1tr)
        v = V[held]
        c_hat = mc.predict(v["S"]["X"])
        r1_hat = np.expm1(np.clip(mr.predict(v["S"]["X"]), 0, 40))
        live = v["gt_onset"] < v["T"]
        if live.sum() >= 10:
            rho_c.append(spearman(c_hat[live], v["c"][live]))
            rho_r1.append(spearman(r1_hat[live], v["r1"][live]))

        variants = {
            "predicted": (c_hat, r1_hat),
            "oracle_r1": (c_hat, v["r1"]),
            "oracle_c": (v["c"], r1_hat),
            "gt_params": (v["c"], v["r1"]),
        }
        onsets = {k: onset_from_two_phase(v["t"], v["r0"], cc, rr, crit)
                  for k, (cc, rr) in variants.items()}
        rows = {k: {"wall": [], "off": []} for k in ARMS}
        for ti in v["times"]:
            scorer = SeverityScorer(v["S"]["edge_index"], v["gt"][ti], len(v["S"]["wall"]),
                                    DEFAULT)
            masks = {"frozen": v["mask"]}
            for k, on in onsets.items():
                masks[k] = v["mask"] & (on <= ti)
            for k, m in masks.items():
                rows[k]["wall"].append(scorer.score(m, v["S"]["wall"]))
                rows[k]["off"].append(scorer.score(m, ~v["S"]["wall"]))
        res = {k: dict(wall=float(np.nanmean(rows[k]["wall"])),
                       off=float(np.nanmean(rows[k]["off"]))) for k in ARMS}
        for k in ARMS:
            acc[k]["wall"].append(res[k]["wall"])
            acc[k]["off"].append(res[k]["off"])
        per_vessel[held] = dict(cls=classes.get(held, "?"), **res)
        print("   %-11s pred wall %.3f off %s" %
              (held, res["predicted"]["wall"],
               ("%.3f" % res["predicted"]["off"]) if res["predicted"]["off"] == res["predicted"]["off"] else "n/a"),
              flush=True)

    print("\nMEAN-OVER-TIME, leave-one-vessel-out, same locked-GNN set (n=%d)\n" % len(pool))
    print("%-12s %10s %10s" % ("arm", "wall", "off"))
    for k in ARMS:
        print("%-12s %10.4f %10.4f"
              % (k, np.nanmean(acc[k]["wall"]), np.nanmean(acc[k]["off"])))
    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\npriority-class only (n=%d):" % len(prio))
    for k in ARMS:
        print("%-12s %10.4f %10.4f"
              % (k, np.nanmean([per_vessel[a][k]["wall"] for a in prio]),
                 np.nanmean([per_vessel[a][k]["off"] for a in prio])))

    print("\nout-of-fold parameter rank correlation:")
    print("   rho(c_hat,  c_gt)   %+.3f" % np.nanmean(rho_c))
    print("   rho(r1_hat, r1_gt)  %+.3f" % np.nanmean(rho_r1))

    Path(args.save).write_text(json.dumps(
        dict(per_vessel=per_vessel, rho_c=float(np.nanmean(rho_c)),
             rho_r1=float(np.nanmean(rho_r1))), indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
