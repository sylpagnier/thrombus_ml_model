"""Train and lock clot_gnn_v3 -- the time-conditioned temporal model (docs/PHASE9_ML.md 13.9).

v1/v2 (`promote_clot_gnn.py` / `promote_clot_gnn_v2.py`) predict a single mask at t_final.
v3 adds the temporal axis: it predicts, for each node and each queried time, whether that
node is clot THEN.  It does this by dropping the order x schedule factorisation that every
earlier temporal arm used (rank the nodes, map the ranks onto a reference time distribution)
and instead treating time as a model INPUT, so the schedule emerges from the features
instead of being imposed from the physics ODE or a cohort curve -- both of which were
measured to cap the achievable score (docs/PHASE9_ML.md 13.5-13.8).

v3 does NOT retrain the GNN.  It reuses the locked v2 ensemble as the SET model (v2's
classifier score is one of the input features) and adds a small gradient-boosted classifier
on top that reads {v2's 56 features, the physics ODE's onset time, the t=0 rate r0, v2's own
score, the query time, whether the ODE itself has fired by that time} -> P(clot now).

Physics kept in the formulation, not bolted on after:
  * the ODE's crossing time and t=0 rate enter as FEATURES, so the model can agree with the
    physics schedule where it is right and depart where it is not;
  * predictions are forced MONOTONE in time by a cumulative maximum -- the production law has
    no sink, so Mat and therefore clot status never regresses;
  * an off-wall node is never predicted clot at a time before its OWNER wall node is, which
    mirrors the physical feed relationship the v2 entry point already enforces.

Measured out-of-fold (`scripts/train_time_conditioned.py`, geometry-stratified 5-fold),
mean-over-time domain-restricted severity deploy score, 19 vessels:

                    wall      off
    frozen mask   0.7953    0.4209
    v2 + ODE      0.8547    0.5369
    v3 (this)     0.8845    0.6110      <- ships
    oracle        0.9705    0.8396

    outputs/clot_ml/locked/clot_gnn_v3/model.pkl
    outputs/clot_ml/locked/clot_gnn_v3/manifest.json
    data/reference/clot_gnn_locked.json          <- repointed to v3; v1/v2 untouched

    python scripts/promote_clot_gnn_v3.py --name clot_gnn_v3
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, eligible_pool, is_priority  # noqa: E402
from src.clot_ml.locked import THRESH_OFF, THRESH_WALL, load_ensemble, predict_scores  # noqa: E402
from src.clot_ml.temporal import ode_trajectory  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
BASE_SET_MODEL = "clot_gnn_v2"
N_TIMES = 11
CLF_KW = dict(max_iter=300, max_depth=5, learning_rate=0.07, l2_regularization=1.0,
             class_weight="balanced", random_state=0)
GRID = np.linspace(0.05, 0.95, 19)

# Out-of-fold scores from the CV run that selected this design (docs/PHASE9_ML.md 13.9).
# There is no held-out vessel left once v3 trains on the whole pool, so this -- not an
# in-sample re-score -- is the number to quote for generalisation.
CV_SCORES_OUT_OF_FOLD = dict(
    all=dict(wall=0.8845, off=0.6110),
    priority=dict(wall=0.9047, off=0.7943),
    baselines=dict(
        frozen=dict(wall=0.7953, off=0.4209),
        v2_plus_ode=dict(wall=0.8547, off=0.5369),
        oracle_schedule=dict(wall=0.8908, off=0.6619),
        oracle=dict(wall=0.9705, off=0.8396)))


def node_feats(S: dict, r0: np.ndarray, oon: np.ndarray, T: int, sc: np.ndarray) -> np.ndarray:
    return np.concatenate([S["X"], np.log1p(np.maximum(r0, 0))[:, None],
                           (oon / T)[:, None], sc[:, None]], axis=1)


def time_row(Xn: np.ndarray, ti: int, T: int, oon: np.ndarray) -> np.ndarray:
    n = Xn.shape[0]
    tf = np.full((n, 1), ti / T, dtype=np.float32)
    ode_now = (oon <= ti).astype(np.float32).reshape(-1, 1)
    return np.concatenate([Xn, tf, ode_now], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clot_gnn_v3")
    ap.add_argument("--flow", default="gt")
    ap.add_argument("--base-set-model", default=BASE_SET_MODEL)
    args = ap.parse_args()

    cache = attach_physics(load_cache(args.flow))
    pool = [a for a in eligible_pool() if a in cache]
    classes = classes_for(pool, PACKS)
    prio = [a for a in pool if is_priority(classes[a])]
    print("[i] training pool n=%d, priority=%d (%s)" % (len(pool), len(prio), ", ".join(prio)),
          flush=True)

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    ens = load_ensemble(name=args.base_set_model)
    t0 = time.time()

    print("[i] building per-vessel targets and physics features ...", flush=True)
    V = {}
    for a in pool:
        S = cache[a]
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        T = int(d.y.shape[0])
        times = [int(round(x)) for x in np.linspace(0, T - 1, N_TIMES)]
        gt = {ti: (gt_clot_phi_at_time(d, ti, phys, device=torch.device("cpu"))
                   .reshape(-1).numpy() > 0.5) for ti in times}
        go = np.full(len(S["wall"]), T, dtype=int)
        for ti in reversed(times):
            go[gt[ti]] = ti
        traj, t = ode_trajectory(d, bio, flow=args.flow)
        r0 = traj[1] / max(t[1] - t[0], 1e-9)
        hot = traj >= crit
        oon = np.where(hot.any(0), hot.argmax(0), T)
        sc = predict_scores(ens, S)
        Xn = node_feats(S, r0, oon, T, sc)
        V[a] = dict(S=S, T=T, times=times, gt=gt, go=go, oon=oon, sc=sc, Xn=Xn)

    print("[i] building the (node, time) training table ...", flush=True)
    Xs, ys = [], []
    for a in pool:
        v = V[a]
        S = v["S"]
        gnn_mask = ((v["sc"] >= THRESH_WALL) & S["wall"]) | ((v["sc"] >= THRESH_OFF) & ~S["wall"])
        cand = gnn_mask | (v["go"] < v["T"])
        Xc = v["Xn"][cand]
        for ti in v["times"]:
            Xs.append(time_row(Xc, ti, v["T"], v["oon"][cand]))
            ys.append(v["gt"][ti][cand])
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys)
    print("[i] fitting the time-conditioned classifier on %d rows ..." % len(ytr), flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(**CLF_KW).fit(Xtr, ytr)
    print("   done (%.0fs)" % (time.time() - t0), flush=True)

    print("[i] tuning per-domain thresholds on the full pool ...", flush=True)

    def probs_for(a):
        v = V[a]
        out = np.zeros((len(v["times"]), v["Xn"].shape[0]), dtype=np.float32)
        for j, ti in enumerate(v["times"]):
            out[j] = clf.predict_proba(time_row(v["Xn"], ti, v["T"], v["oon"]))[:, 1]
        return np.maximum.accumulate(out, axis=0)

    Pcache = {a: probs_for(a) for a in pool}

    from src.clot_ml.severity_metric import DEFAULT, SeverityScorer

    def tune(domain_of):
        best, bt = -1e9, 0.5
        for th in GRID:
            vals = []
            for a in pool:
                v, S = V[a], V[a]["S"]
                gnn_mask = (((v["sc"] >= THRESH_WALL) & S["wall"])
                           | ((v["sc"] >= THRESH_OFF) & ~S["wall"]))
                dom = domain_of(S)
                for j, ti in enumerate(v["times"]):
                    sc_ = SeverityScorer(S["edge_index"], v["gt"][ti], len(S["wall"]), DEFAULT)
                    x = sc_.score(gnn_mask & (Pcache[a][j] >= th), dom)
                    if x == x:
                        vals.append(x)
            if vals and np.mean(vals) > best:
                best, bt = float(np.mean(vals)), float(th)
        return bt

    thresh_wall = tune(lambda S: S["wall"])
    thresh_off = tune(lambda S: ~S["wall"])
    print("   thresh_wall=%.3f  thresh_off=%.3f" % (thresh_wall, thresh_off), flush=True)

    out = REPO / "outputs/clot_ml/locked" / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "model.pkl").open("wb") as fh:
        pickle.dump(clf, fh)

    manifest = dict(
        name=args.name,
        kind="temporal_v3",
        promoted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=(
            "Time-conditioned clot model, PHASE9/10 (docs/PHASE9_ML.md 13.9). Predicts "
            "P(node is clot at time t) directly -- time is a model INPUT -- instead of "
            "factoring into an onset ranking mapped onto a reference schedule, which "
            "measurably capped every earlier temporal arm. Reuses the locked v2 GNN "
            "ensemble as the committed SET and its classifier score as a feature; adds a "
            "gradient-boosted head over {v2 features, ODE onset time, t=0 rate r0, v2 "
            "score, query time, ODE's own answer at that time}. Predictions are forced "
            "monotone in time (cumulative max) and an off-wall node is never predicted "
            "clot before its owner wall node."),
        docs="docs/PHASE9_ML.md",
        supersedes="clot_gnn_v2",
        base_set_model=args.base_set_model,
        flow=args.flow,
        training_pool=list(pool), priority_anchors=list(prio),
        geometry_classes={a: classes[a] for a in pool},
        n_times=N_TIMES,
        clf_params=CLF_KW,
        clf_file="model.pkl",
        thresh_wall=thresh_wall, thresh_off=thresh_off,
        feature_note=("node_feats = [v2 X (56 cols incl phys_mask), log1p(r0), "
                      "ode_onset/T, v2_score]; time_row = [node_feats, t/T, ode_fired_now]"),
        scores_out_of_fold_cv=CV_SCORES_OUT_OF_FOLD)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    ptr = REPO / "data/reference/clot_gnn_locked.json"
    ptr.write_text(json.dumps(dict(
        name=args.name, kind="temporal_v3",
        path=str(out.relative_to(REPO)).replace("\\", "/"),
        manifest=str((out / "manifest.json").relative_to(REPO)).replace("\\", "/"),
        base_set_model=args.base_set_model,
        promoted_at=manifest["promoted_at"], docs="docs/PHASE9_ML.md",
        supersedes="clot_gnn_v2",
        scores_out_of_fold_cv=CV_SCORES_OUT_OF_FOLD), indent=2))
    print("\nlocked -> %s\npointer -> %s (now clot_gnn_v3)  (%.0fs)"
          % (out, ptr, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
