"""Geometry-stratified K-fold over the whole eligible non-SEALED pool.

Replaces the confounded FIT/DEV cut (see `src/clot_ml/geometry_splits.py`).  Saves, for
every fold, that fold's model's score on every vessel, so readouts and metrics can be
re-evaluated later without retraining.

    python scripts/run_phase9_cv.py --tag cv5 --folds 5 --seeds 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, describe, stratified_folds  # noqa: E402
from train_clot_gnn import train_one  # noqa: E402

OUT = REPO / "outputs/phase9_scores"
PACKS = REPO / "data/processed/graphs_biochem_anchors"
BASE = dict(epochs=80, dim=64, layers=4, drop=0.1, lr=3e-3, wd=1e-4, pos_weight=30.0,
            reg_w=1.0, metric_w=2.0, metric_start=0.3, rounds=3, off_mult=1.0,
            metric="legacy", adv_fb=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in BASE.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    # which feature cache to read: "gt" is the v3 55-channel one, "v4" adds the advective
    # transport + indicator-gate channels (scripts/build_clot_ml_cache_v4.py)
    ap.add_argument("--cache", default="gt")
    args = ap.parse_args()
    cfg = SimpleNamespace(**{k: getattr(args, k) for k in BASE}, seeds=1)
    # advective recurrence (src/clot_ml/recurrent.feedback_channels_advective)
    cfg.adv_fb = bool(cfg.adv_fb)

    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.cache))
    pool = [a for a in cache]
    classes = classes_for(pool, PACKS)
    pool = [a for a in pool if a in classes]
    folds = stratified_folds({a: classes[a] for a in pool}, k=args.folds)
    print("[i] pool n=%d, %d folds\n%s" % (len(pool), len(folds), describe(classes, folds)),
          flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    store, t0 = {}, time.time()
    for k, held in enumerate(folds):
        tr = [a for a in pool if a not in held]
        acc, accr = {}, {}
        for s in range(args.seeds):
            predict = train_one(tr, cache, cfg, dev_t, seed=s)
            for a in pool:
                acc[a] = acc.get(a, 0.0) + predict(a) / args.seeds
                accr[a] = accr.get(a, 0.0) + predict.reg(a) / args.seeds
        for a in pool:
            store["%d|%s" % (k, a)] = acc[a].astype(np.float32)
            # the regression head's log1p(Mat/crit); see train_clot_gnn.predict_reg
            store["reg|%d|%s" % (k, a)] = accr[a].astype(np.float32)
        store["held|%d" % k] = np.array(held)
        print("   fold %d/%d held=%s (%.0fs)"
              % (k + 1, len(folds), ",".join(x[-3:] for x in held), time.time() - t0),
              flush=True)
    store["pool"] = np.array(pool)
    store["classes"] = np.array([classes[a] for a in pool])
    store["cfg"] = np.array([repr(vars(cfg))])
    np.savez_compressed(OUT / f"{args.tag}.npz", **store)
    print("wrote %s (%.0fs)" % (OUT / f"{args.tag}.npz", time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
