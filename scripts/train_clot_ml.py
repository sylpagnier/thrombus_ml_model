"""PHASE9: physics-informed ML for the full-mesh clot map.  Targets wall>0.9, off>0.7.

Scored with the domain-restricted deploy score, FIT leave-one-vessel-out, DEV from a
single fit on all of FIT.  SEALED closed.

    python scripts/train_clot_ml.py --arms physics,logreg,gbm
    python scripts/train_clot_ml.py --arms gbm --drop p_nd,width_nd
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import load_cache, splits  # noqa: E402
from src.clot_ml.evaluate import banner  # noqa: E402
from src.clot_ml.protocol import Bench, run_lovo  # noqa: E402

LOG = REPO / "outputs/phase9_log.jsonl"
GRID = np.linspace(0.05, 0.98, 40)


def log_result(tag, summ, extra=None):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(tag=tag, t=time.strftime("%m-%d %H:%M:%S"), fit=summ["fit"], dev=summ["dev"])
    if extra:
        rec.update(extra)
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def cols_of(cache):
    return [str(c) for c in next(iter(cache.values()))["cols"]]


def select(cache, keep_idx):
    return {a: S["X"][:, keep_idx] for a, S in cache.items()}


# ---------------------------------------------------------------------------
def arm_physics(bench):
    cache, fit, dev = bench.cache, bench.fit, bench.dev
    from src.config import BiochemConfig
    from src.core_physics.physics_lumen_model import adjacency, grow_into_lumen
    import predict_wall_clot as P
    bio = BiochemConfig(phase="biochem")
    rows = {}
    for a in list(fit) + list(dev):
        S = cache[a]
        wall, ei = S["wall"], S["edge_index"]
        A = adjacency(ei, len(wall)).astype(np.int8)
        cur = (S["gate"] > 0) & wall
        adm = (S["sr"] < float(bio.lss) * P.RELAX) & wall
        for _ in range(P.GROW_HOPS):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        off = grow_into_lumen(cur, wall, A, S["spd"], S["sr"],
                              lumen_hops=P.LUMEN_HOPS, speed_thresh=P.LUMEN_SPEED)
        rows[a] = bench.row(a, cur | off)
    return bench.summarise(rows), rows, {}


def make_sk_arm(cache, X, kind, **kw):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    def fit_fn(train):
        Xt = np.concatenate([X[a] for a in train])
        yt = np.concatenate([cache[a]["y"] for a in train]) > 0.5
        mu, sd = Xt.mean(0), Xt.std(0)
        sd[sd < 1e-6] = 1.0
        if kind == "logreg":
            m = LogisticRegression(max_iter=3000, C=kw.get("C", 0.1),
                                   class_weight="balanced")
            m.fit((Xt - mu) / sd, yt)
            return lambda a: m.predict_proba((X[a] - mu) / sd)[:, 1]
        m = HistGradientBoostingClassifier(
            max_iter=kw.get("max_iter", 250), max_depth=kw.get("max_depth", 4),
            learning_rate=kw.get("lr", 0.06), l2_regularization=kw.get("l2", 1.0),
            min_samples_leaf=kw.get("leaf", 100),
            max_features=kw.get("max_features", 1.0),
            class_weight="balanced", random_state=kw.get("seed", 0))
        m.fit(Xt, yt)
        return lambda a: m.predict_proba(X[a])[:, 1]

    return fit_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="physics,logreg,gbm")
    ap.add_argument("--flow", default="gt")
    ap.add_argument("--drop", default="")
    ap.add_argument("--keep", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--leaf", type=int, default=100)
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cache = load_cache(args.flow)
    fit, dev = splits(cache)
    cols = cols_of(cache)
    drop = {s.strip() for s in args.drop.split(",") if s.strip()}
    keep = {s.strip() for s in args.keep.split(",") if s.strip()}
    idx = [i for i, c in enumerate(cols)
           if c not in drop and (not keep or c in keep)]
    X = select(cache, idx)
    bench = Bench(cache, fit, dev)
    print("[i] FIT n=%d DEV n=%d  features %d/%d  (SEALED closed)"
          % (len(fit), len(dev), len(idx), len(cols)), flush=True)

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        t0 = time.time()
        if arm == "physics":
            summ, rows, extra = arm_physics(bench)
        elif arm in ("logreg", "gbm"):
            fn = make_sk_arm(cache, X, arm, max_depth=args.depth, leaf=args.leaf,
                             max_iter=args.iters)
            summ, rows, ths = run_lovo(fn, bench, GRID, verbose=args.verbose)
            extra = dict(n_features=len(idx), depth=args.depth, leaf=args.leaf,
                         iters=args.iters, drop=sorted(drop))
        else:
            print("[skip] %s" % arm)
            continue
        tag = f"{arm}{('/' + args.tag) if args.tag else ''}"
        print(banner(tag, summ), " (%.0fs)" % (time.time() - t0), flush=True)
        log_result(tag, summ, extra)
        if args.verbose:
            for a in sorted(rows):
                r = rows[a]
                print("      %-12s wall %.4f off %6s"
                      % (a, r["wall"], ("%.4f" % r["off"]) if r["off"] == r["off"] else " n/a"),
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
