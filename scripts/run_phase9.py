"""Train once, save per-fold scores, then evaluate every readout without retraining.

Training dominates the cost, readout choice does not, so they are separated: this script
writes ``outputs/phase9_scores/<tag>.npz`` holding, for every fold, that fold's model's
score on every vessel.  ``scripts/eval_readouts.py`` then tunes readout parameters on each
fold's TRAIN vessels and applies them to the held-out one, which keeps the protocol honest
while making readout experiments free.

    python scripts/run_phase9.py --tag base --epochs 150
    python scripts/run_phase9.py --tag rec3 --epochs 150 --rounds 3
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

from src.clot_ml.data import attach_physics, load_cache, splits  # noqa: E402
from train_clot_gnn import train_one  # noqa: E402

OUT = REPO / "outputs/phase9_scores"

BASE = dict(epochs=150, dim=96, layers=6, drop=0.1, lr=3e-3, wd=1e-4,
            pos_weight=30.0, reg_w=1.0, metric_w=2.0, metric_start=0.3, rounds=1,
            off_mult=1.0, metric='legacy')


def main() -> int:
    ap = argparse.ArgumentParser()
    for k, v in BASE.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--flow", default="gt")
    args = ap.parse_args()
    cfg = SimpleNamespace(**{k: getattr(args, k) for k in BASE}, seeds=args.seeds)

    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.flow))
    fit, dev = splits(cache)
    folds = [list(fit[i::args.folds]) for i in range(args.folds)]
    OUT.mkdir(parents=True, exist_ok=True)
    store, t0 = {}, time.time()

    for k, held in enumerate(folds):
        tr = [a for a in fit if a not in held]
        acc = {}
        for s in range(args.seeds):
            predict = train_one(tr, cache, cfg, dev_t, seed=s)
            for a in fit:
                acc[a] = acc.get(a, 0.0) + predict(a) / args.seeds
        for a in fit:
            store["%d|%s" % (k, a)] = acc[a].astype(np.float32)
        store["held|%d" % k] = np.array(held)
        print("   fold %d/%d trained (%.0fs)" % (k + 1, len(folds), time.time() - t0),
              flush=True)

    acc = {}
    for s in range(args.seeds):
        predict = train_one(fit, cache, cfg, dev_t, seed=s)
        for a in fit + dev:
            acc[a] = acc.get(a, 0.0) + predict(a) / args.seeds
    for a in fit + dev:
        store["final|%s" % a] = acc[a].astype(np.float32)
    store["cfg"] = np.array([repr(vars(cfg))])
    np.savez_compressed(OUT / f"{args.tag}.npz", **store)
    print("wrote %s  (%.0fs)" % (OUT / f"{args.tag}.npz", time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
