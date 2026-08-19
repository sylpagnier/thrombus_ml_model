"""In-process sweep driver for the PHASE9 GNN.  Appends every arm to outputs/phase9_log.jsonl.

Runs configs sequentially, reusing the loaded cache and the precomputed scorers, and
prints one line per config so a long run can be monitored cheaply.

    python scripts/sweep_clot_gnn.py --stage coarse
"""
from __future__ import annotations

import argparse
import itertools
import json
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
from src.clot_ml.protocol import Bench  # noqa: E402
from train_clot_gnn import GRID, apply_readout, pick_readout, train_one  # noqa: E402

LOG = REPO / "outputs/phase9_log.jsonl"

BASE = dict(epochs=250, dim=96, layers=6, drop=0.1, lr=3e-3, wd=1e-4,
            pos_weight=30.0, reg_w=1.0, metric_w=2.0, metric_start=0.3,
            seeds=1, rounds=1, flow="gt")

STAGES = {
    "coarse": [
        dict(metric_w=0.0), dict(metric_w=1.0), dict(metric_w=2.0), dict(metric_w=4.0),
        dict(pos_weight=10.0), dict(pos_weight=100.0),
        dict(dim=128, layers=8), dict(dim=64, layers=4),
        dict(drop=0.2), dict(reg_w=0.0), dict(reg_w=4.0),
        dict(lr=1e-3), dict(lr=6e-3), dict(epochs=450),
    ],
}


def run_cfg(bench, cache, fit, dev, cfg, folds, tag):
    args = SimpleNamespace(**cfg)
    dev_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    t0 = time.time()
    for held in folds:
        tr = [a for a in fit if a not in held]
        sc = {}
        for s in range(int(args.seeds)):
            predict = train_one(tr, cache, args, dev_t, seed=s)
            for a in tr + held:
                sc[a] = sc.get(a, 0.0) + predict(a) / int(args.seeds)
        th = pick_readout(bench, sc, tr, GRID)
        for a in held:
            rows[a] = bench.row(a, apply_readout(cache[a], sc[a], th))
    sc = {}
    for s in range(int(args.seeds)):
        predict = train_one(fit, cache, args, dev_t, seed=s)
        for a in fit + dev:
            sc[a] = sc.get(a, 0.0) + predict(a) / int(args.seeds)
    th = pick_readout(bench, sc, fit, GRID)
    for a in dev:
        rows[a] = bench.row(a, apply_readout(cache[a], sc[a], th))
    summ = bench.summarise(rows)
    rec = dict(tag=tag, t=time.strftime("%m-%d %H:%M:%S"), fit=summ["fit"], dev=summ["dev"],
               cfg={k: v for k, v in cfg.items()}, th=list(th), secs=round(time.time() - t0))
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print("%-42s FIT w %.4f o %.4f | DEV w %.4f o %.4f  (%.0fs)"
          % (tag, summ["fit"]["wall"], summ["fit"]["off"],
             summ["dev"]["wall"], summ["dev"]["off"], time.time() - t0), flush=True)
    return summ, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="coarse")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--configs", default="")
    args = ap.parse_args()
    cache = attach_physics(load_cache("gt"))
    fit, dev = splits(cache)
    bench = Bench(cache, fit, dev)
    folds = [list(fit[i::args.folds]) for i in range(args.folds)]
    cfgs = json.loads(args.configs) if args.configs else STAGES[args.stage]
    print("[i] %d configs, FIT=%d DEV=%d folds=%d" % (len(cfgs), len(fit), len(dev), args.folds),
          flush=True)
    for i, over in enumerate(cfgs):
        cfg = dict(BASE)
        cfg.update(over)
        tag = "sweep/%s/%d/%s" % (args.stage, i,
                                  ",".join("%s=%s" % (k, v) for k, v in over.items()) or "base")
        try:
            run_cfg(bench, cache, fit, dev, cfg, folds, tag)
        except Exception as e:  # noqa: BLE001
            print("[ERR] %s %s" % (tag, e), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
