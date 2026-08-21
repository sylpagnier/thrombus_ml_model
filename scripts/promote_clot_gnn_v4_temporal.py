"""Train and lock the `clot_gnn_v4` TEMPORAL head -- the arm actually validated in PHASE10.

`scripts/promote_clot_gnn_v4.py` locks the GNN ensemble only (a `gnn_ensemble`-kind
artifact, final-time mask, no schedule).  Everything this session measured -- the
expected-score off-wall readout (10.2), the adaptive wall cut (10.1), the seed-averaged
time-conditioned head, and the ODE-anchored learned lag (15.2) -- lives in
`scripts/eval_strict_temporal.py` / `scripts/eval_expected_score_readout.py` as CV-evaluation
code, not as a deployable artifact.  This promotes that whole pipeline, fitted on the full
19-vessel eligible pool using the SHIPPED v4 ensemble's own (in-sample) scores, the same
"no held-out vessel left" discipline v2/v3 already use.

Every fitted quantity here is a handful of scalars plus two small GBM ensembles (the
temporal head, 4 seeds; the lag regressor, 3 seeds) -- nothing here needs a GPU.

Reproduces, end to end on the pool, the numbers `docs/PHASE10_V4.md` reports:

                  mean wall   mean off   FIN wall   FIN off
    shipped          0.8750     0.7188     0.9176     0.7366

    outputs/clot_ml/locked/clot_gnn_v4/temporal.pkl
    outputs/clot_ml/locked/clot_gnn_v4/manifest.json     (kind -> temporal_v4)
    data/reference/clot_gnn_locked.json                  (repointed with --repoint)

    python scripts/promote_clot_gnn_v4_temporal.py --repoint
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

import eval_strict_temporal as ET  # noqa: E402
from eval_expected_score_readout import GAMMA, KSCALE, expected_curve  # noqa: E402
from eval_strict import (  # noqa: E402
    FAMILIES, GRID, apply_adapt, readout_resid, tune_adapt, tune_resid,
)
from src.clot_ml.geometry_splits import classes_for, eligible_pool, is_priority  # noqa: E402
from src.clot_ml.locked import build_sample, load_ensemble, predict_scores  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.softmetric import dilation_operator, to_torch_sparse  # noqa: E402
from src.config import BiochemConfig  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
HEAD_SEEDS = 4
LAG_SEEDS = 3
N_TIMES = 11
INNER = 5
# Matches the OUTER fold count of the strict CV study that validated the ODE-anchored lag
# (docs/PHASE10_V4.md 15), not the smaller default used elsewhere for a plain inner split.
# With INNER=3 this promotion's own internal check picked burden_gate=None (i.e. "don't use
# the learned lag") -- a single 3-way split of 19 vessels is inside the project's own noise
# floor (+-0.091 off-wall, 2) and one noisy split should not overrule 5 outer folds' worth
# of evidence that the ODE-anchored lag helps.  INNER=5 gives this check the same amount of
# held-out evidence the design decision was actually made on.


def fit_set_spec(cache, sc, pool, vs):
    """Wall + off-wall committed-set spec, chosen the same way `nested_pick` chooses it in
    every fold of `eval_expected_score_readout.py` -- here on the whole pool, no held-out.
    Returns two dicts: ``{"kind": "resid_adapt"|"resid"|"cohort_cut", ...params}``.
    """
    wall_of, off_of = (lambda S: S["wall"]), (lambda S: ~S["wall"])
    dev = torch.device("cpu")
    Dt = {a: to_torch_sparse(dilation_operator(cache[a]["edge_index"], len(cache[a]["wall"]),
                                               2), dev) for a in pool}

    def q_of(fn, dom_of):
        v = [x for x in (fn(a) for a in pool) if x == x]
        return float(np.mean(v)) if v else -1e9

    def cohort_cut(dom_of):
        top, t_cut = -1e9, float(GRID[0])
        for t in GRID:
            v = [vs[a].score(dom_of(cache[a]) & (sc[a] >= t), dom_of(cache[a])) for a in pool]
            v = [x for x in v if x == x]
            if v and np.mean(v) > top:
                top, t_cut = float(np.mean(v)), float(t)
        return top, t_cut

    th_r = tune_resid(cache, vs, pool, sc, GRID)

    def wall_cands():
        q_cc, t_cc = cohort_cut(wall_of)
        q_r = q_of(lambda a: vs[a].score(FAMILIES["resid"][1](cache[a], sc[a], th_r)
                                         & wall_of(cache[a]), wall_of(cache[a])), wall_of)
        b_w, med_w = tune_adapt(cache, vs, pool, sc, "resid", th_r, wall_of)
        q_ra = q_of(lambda a: vs[a].score(
            apply_adapt(cache[a], sc[a], "resid", th_r, wall_of, b_w, med_w)
            & wall_of(cache[a]), wall_of(cache[a])), wall_of)
        return {"cohort_cut": (q_cc, dict(kind="cohort_cut", t=t_cc)),
                "resid": (q_r, dict(kind="resid", th=list(th_r))),
                "resid_adapt": (q_ra, dict(kind="resid_adapt", th=list(th_r),
                                          b=b_w, med=med_w))}

    curves = {(a, g): expected_curve(sc[a], off_of(cache[a]), Dt[a], dev, g)
              for a in pool for g in GAMMA}

    def mask_for(a, g, kscl):
        ks, vals = curves[(a, g)]
        if len(ks) < 2:
            return np.zeros(len(sc[a]), bool)
        k = int(np.clip(round(ks[int(np.argmax(vals))] * kscl), 1, ks[-1]))
        d = off_of(cache[a])
        order = np.flatnonzero(d)[np.argsort(-sc[a][d])]
        m = np.zeros(len(sc[a]), bool)
        m[order[:k]] = True
        return m

    def off_cands():
        q_cc, t_cc = cohort_cut(off_of)
        q_r = q_of(lambda a: vs[a].score(FAMILIES["resid"][1](cache[a], sc[a], th_r)
                                         & off_of(cache[a]), off_of(cache[a])), off_of)
        b_o, med_o = tune_adapt(cache, vs, pool, sc, "resid", th_r, off_of)
        q_ra = q_of(lambda a: vs[a].score(
            apply_adapt(cache[a], sc[a], "resid", th_r, off_of, b_o, med_o)
            & off_of(cache[a]), off_of(cache[a])), off_of)
        best_e = None
        for g in GAMMA:
            for kscl in KSCALE:
                q = q_of(lambda a, _g=g, _k=kscl: vs[a].score(mask_for(a, _g, _k),
                                                              off_of(cache[a])), off_of)
                if best_e is None or q > best_e[0]:
                    best_e = (q, g, kscl)
        return {"cohort_cut": (q_cc, dict(kind="cohort_cut", t=t_cc)),
                "resid": (q_r, dict(kind="resid", th=list(th_r))),
                "resid_adapt": (q_ra, dict(kind="resid_adapt", th=list(th_r),
                                          b=b_o, med=med_o)),
                "expected_tuned": (best_e[0], dict(kind="expected_tuned",
                                                   gamma=best_e[1], kscale=best_e[2]))}

    wc, oc = wall_cands(), off_cands()
    wall_spec = max(wc.values(), key=lambda x: x[0])[1]
    off_spec = max(oc.values(), key=lambda x: x[0])[1]
    print("[i] wall set  -> %-13s (scores: %s)" % (
        wall_spec["kind"], " ".join("%s=%.3f" % (k, v[0]) for k, v in wc.items())))
    print("[i] off  set  -> %-13s (scores: %s)" % (
        off_spec["kind"], " ".join("%s=%.3f" % (k, v[0]) for k, v in oc.items())))
    return wall_spec, off_spec


def apply_set_spec(S, sc, spec, dom_of, dev="cpu"):
    """Materialise a committed-set spec (from :func:`fit_set_spec`) on one vessel."""
    d = dom_of(S)
    if spec["kind"] == "cohort_cut":
        return d & (sc >= spec["t"])
    if spec["kind"] == "resid":
        return d & readout_resid(S, sc, tuple(spec["th"]))
    if spec["kind"] == "resid_adapt":
        return d & apply_adapt(S, sc, "resid", tuple(spec["th"]), dom_of, spec["b"], spec["med"])
    if spec["kind"] == "expected_tuned":
        Dt = to_torch_sparse(dilation_operator(S["edge_index"], len(S["wall"]), 2),
                             torch.device("cpu"))
        ks, vals = expected_curve(sc, d, Dt, torch.device("cpu"), spec["gamma"])
        if len(ks) < 2:
            return np.zeros(len(sc), bool)
        k = int(np.clip(round(ks[int(np.argmax(vals))] * spec["kscale"]), 1, ks[-1]))
        order = np.flatnonzero(d)[np.argsort(-sc[d])]
        m = np.zeros(len(sc), bool)
        m[order[:k]] = True
        return m
    raise ValueError(spec["kind"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clot_gnn_v4")
    ap.add_argument("--repoint", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ens = load_ensemble(name=args.name)
    bio = BiochemConfig(phase="biochem")
    pool = [a for a in eligible_pool() if (PACKS / f"{a}.pt").exists()]
    classes = classes_for(pool, PACKS)
    pool = [a for a in pool if a in classes]
    prio = [a for a in pool if is_priority(classes[a])]
    print("[i] pool n=%d, priority=%d (%s)" % (len(pool), len(prio), ", ".join(prio)))

    # --- score every pool vessel with the SHIPPED ensemble (in-sample) -----------------
    cache, sc = {}, {}
    for a in pool:
        d = torch.load(PACKS / f"{a}.pt", map_location="cpu", weights_only=False)
        S = build_sample(d, bio, flow="gt", variant="v4")
        cache[a] = S
        sc[a] = predict_scores(ens, S)
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    print("[i] scored pool (%.0fs)" % (time.time() - t0), flush=True)

    # --- committed SET spec (10.1 / 10.2) ----------------------------------------------
    wall_spec, off_spec = fit_set_spec(cache, sc, pool, vs)
    gm = {a: (apply_set_spec(cache[a], sc[a], wall_spec, lambda S: S["wall"])
             | apply_set_spec(cache[a], sc[a], off_spec, lambda S: ~S["wall"]))
          for a in pool}
    fin_wall = np.mean([vs[a].score(gm[a] & cache[a]["wall"], cache[a]["wall"])
                        for a in pool])
    fin_off = np.nanmean([vs[a].score(gm[a] & ~cache[a]["wall"], ~cache[a]["wall"])
                          for a in pool])
    print("[i] FINAL-TIME (in-sample) wall=%.4f off=%.4f" % (fin_wall, fin_off), flush=True)

    # --- reuse eval_strict_temporal's machinery via EXTERNAL_SET -----------------------
    # candidate_mask() returns EXTERNAL_SET[anchor] whenever the anchor is present, so every
    # downstream call (fit_head, tune_time, offwall_burden, ...) sees exactly `gm` above
    # regardless of the `set_th` argument -- the same trick the CV runs use with
    # `--set-masks`, here fed the vessel's OWN shipped-ensemble committed set instead of a
    # frozen npz from an earlier (different) ensemble.
    ET.EXTERNAL_SET.clear()
    ET.EXTERNAL_SET.update(gm)
    ET.LAG_ANCHOR[0] = "ode"
    ET.ANCHOR_LEVEL[0] = 1.0  # docs/PHASE10_V4.md 15.3: sweeping {1,2,4,8} always selects 1

    print("[i] precomputing temporal features ...", flush=True)
    V = ET.precompute(pool, cache, N_TIMES)
    for v in V.values():
        v["clock"] = []                      # measured negative (docs/PHASE10_V4.md 11)
    oofs = {"v4": sc}
    set_th = {}                              # unused: EXTERNAL_SET short-circuits every call

    inner = [pool[i::INNER] for i in range(INNER)]

    def inner_oof(seeds):
        out = {}
        for iv in inner:
            itr = [a for a in pool if a not in iv]
            m_i = ET.fit_head(V, itr, oofs, set_th, seeds)
            for a in iv:
                out[a] = ET.predict_series(V, a, m_i, oofs)
        return out

    Pin = inner_oof(HEAD_SEEDS)
    print("[i] fitting temporal head (%d seeds, full pool) ..." % HEAD_SEEDS, flush=True)
    head = ET.fit_head(V, pool, oofs, set_th, HEAD_SEEDS)
    time_th = ET.tune_time(V, pool, Pin, oofs, set_th)
    print("[i] time cuts: wall=%s off=%s" % (time_th[0], time_th[1]), flush=True)

    print("[i] fitting off-wall lag model (ODE-anchored, %d seeds) ..." % LAG_SEEDS,
          flush=True)
    lag_pred = {}
    for iv in inner:
        itr = [a for a in pool if a not in iv]
        m_l = ET.fit_lag_model(V, itr, oofs, LAG_SEEDS, "ode")
        if m_l is None:
            continue
        for a in iv:
            lag_pred[a] = m_l.predict(ET.lag_features(V, a, oofs))
    lag_model = ET.fit_lag_model(V, pool, oofs, LAG_SEEDS, "ode")
    lag_choice = ET.tune_lag(V, pool, Pin, oofs, set_th, time_th, lag_pred)
    burden_gate = lag_choice[1] if isinstance(lag_choice, tuple) else None
    print("[i] lag burden gate: %s" % burden_gate, flush=True)

    # --- honest in-sample readback, exactly as score_vessel computes it ----------------
    rows = {}
    for a in pool:
        P = ET.predict_series(V, a, head, oofs)
        lag_a = None
        if burden_gate is not None and ET.offwall_burden(V, a, oofs, set_th) >= burden_gate:
            lag_a = ("learned", lag_model.predict(ET.lag_features(V, a, oofs)))
        rows[a] = ET.score_vessel(V, a, P, oofs, set_th, time_th, lag=lag_a)
    mean_w = np.mean([rows[a]["wall"] for a in pool])
    mean_o = np.nanmean([rows[a]["off"] for a in pool])
    print("[i] MEAN-OVER-TIME (in-sample) wall=%.4f off=%.4f  (%.0fs)"
          % (mean_w, mean_o, time.time() - t0), flush=True)

    # --- save ----------------------------------------------------------------------
    out = REPO / "outputs/clot_ml/locked" / args.name
    out.mkdir(parents=True, exist_ok=True)
    # `head` and `lag_model.models` are plain lists of sklearn estimators -- pickle ONLY
    # those, not the `_LagEnsemble` wrapper.  A pickled reference to `eval_strict_temporal`
    # would tie this artifact to that script's exact module path at every future unpickle,
    # unlike every other locked artifact in this project (v2/v3 pickle bare sklearn objects).
    artifact = dict(
        wall_spec=wall_spec, off_spec=off_spec,
        head=head, lag_models=(lag_model.models if lag_model is not None else None),
        lag_anchor="ode", lag_anchor_level=1.0, burden_gate=burden_gate,
        time_th_wall=list(time_th[0]), time_th_off=list(time_th[1]),
        n_times=N_TIMES, head_seeds=HEAD_SEEDS, lag_seeds=LAG_SEEDS)
    with (out / "temporal.pkl").open("wb") as fh:
        pickle.dump(artifact, fh)

    manifest = json.loads((out / "manifest.json").read_text())
    manifest["kind"] = "temporal_v4"
    manifest["temporal_file"] = "temporal.pkl"
    manifest["temporal_promoted_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    manifest["temporal_readout"] = dict(wall=wall_spec["kind"], off=off_spec["kind"],
                                        lag_anchor="ode", burden_gate=burden_gate)
    manifest["temporal_scores_in_sample"] = dict(
        final=dict(wall=float(fin_wall), off=float(fin_off)),
        mean_over_time=dict(wall=float(mean_w), off=float(mean_o)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("locked temporal head -> %s" % (out / "temporal.pkl"))

    if args.repoint:
        ptr = REPO / "data/reference/clot_gnn_locked.json"
        prev = json.loads(ptr.read_text()) if ptr.exists() else {}
        ptr.write_text(json.dumps(dict(
            name=args.name, kind="temporal_v4",
            path=str(out.relative_to(REPO)).replace("\\", "/"),
            manifest=str((out / "manifest.json").relative_to(REPO)).replace("\\", "/"),
            promoted_at=manifest["temporal_promoted_at"], docs="docs/PHASE10_V4.md",
            supersedes=prev.get("name", "clot_gnn_v3"),
            scores_strict_cv=manifest.get("scores_strict_cv", {})), indent=2))
        print("pointer -> %s (now %s, temporal_v4)" % (ptr, args.name))
    else:
        print("[i] pointer NOT moved; rerun with --repoint to ship this generation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
