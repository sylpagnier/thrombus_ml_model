"""Evaluate every readout on saved per-fold scores.  No training, no GPU.

    python scripts/eval_readouts.py --tags base,rec3
    python scripts/eval_readouts.py --tags base --ensemble base,rec3
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

from src.clot_ml.data import attach_physics, load_cache, splits  # noqa: E402
from src.clot_ml.protocol import Bench  # noqa: E402
from src.clot_ml.readouts import SEPARABLE, REGISTRY, apply, tune, tune_separable  # noqa: E402

OUT = REPO / "outputs/phase9_scores"
LOG = REPO / "outputs/phase9_log.jsonl"


def load_scores(tags: list[str]) -> dict:
    """Average the score arrays across tags (a simple model ensemble)."""
    zs = [np.load(OUT / f"{t}.npz", allow_pickle=True) for t in tags]
    keys = [k for k in zs[0].files if "|" in k and not k.startswith("held")]
    store = {k: np.mean([z[k] for z in zs], axis=0) for k in keys}
    folds = {}
    for k in zs[0].files:
        if k.startswith("held|"):
            folds[int(k.split("|")[1])] = [str(x) for x in zs[0][k]]
    return store, folds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma list; averaged if more than one")
    ap.add_argument("--tags-off", default="",
                    help="use a different model (or ensemble) for the OFF-WALL domain. "
                         "Legitimate because the metric of record is domain-restricted: the "
                         "final mask is the union of a wall decision and an off-wall one, and "
                         "the two domains want different precision/recall trade-offs.")
    ap.add_argument("--readouts", default="thresh,resid,topk,topk_resid")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    cache = attach_physics(load_cache("gt"))
    fit, dev = splits(cache)
    bench = Bench(cache, fit, dev)
    store, folds = load_scores(tags)
    tags_off = [t.strip() for t in args.tags_off.split(",") if t.strip()]
    store_off, _ = load_scores(tags_off) if tags_off else (store, folds)

    def combine(sc_w, sc_o, S):
        """One array whose wall entries come from the wall model and off entries from the
        off-wall model.  Thresholds are still tuned per domain, so this is exact."""
        out = np.array(sc_w, copy=True)
        out[~S["wall"]] = sc_o[~S["wall"]]
        return out

    for name in [r.strip() for r in args.readouts.split(",") if r.strip()]:
        if name not in REGISTRY:
            continue
        t0 = time.time()
        rows = {}
        for k, held in folds.items():
            tr = [a for a in fit if a not in held]
            sc = {a: combine(store["%d|%s" % (k, a)], store_off["%d|%s" % (k, a)],
                             cache[a]) for a in fit}
            tune_fn = tune_separable if name in SEPARABLE else tune
            p, _ = tune_fn(name, bench, sc, tr)
            for a in held:
                rows[a] = bench.row(a, apply(name, cache[a], sc[a], p))
        sc = {a: combine(store["final|%s" % a], store_off["final|%s" % a], cache[a])
              for a in fit + dev}
        tune_fn = tune_separable if name in SEPARABLE else tune
        p, _ = tune_fn(name, bench, sc, fit)
        for a in dev:
            rows[a] = bench.row(a, apply(name, cache[a], sc[a], p))
        s = bench.summarise(rows)
        tag = "+".join(tags) + ("|off:" + "+".join(tags_off) if tags_off else "") + "/" + name
        print("%-34s FIT wall %.4f off %.4f full %.4f | DEV wall %.4f off %.4f full %.4f  "
              "p=%s (%.0fs)"
              % (tag, s["fit"]["wall"], s["fit"]["off"], s["fit"]["full"],
                 s["dev"]["wall"], s["dev"]["off"], s["dev"]["full"], p, time.time() - t0),
              flush=True)
        with LOG.open("a") as fh:
            fh.write(json.dumps(dict(tag=tag, t=time.strftime("%m-%d %H:%M:%S"),
                                     fit=s["fit"], dev=s["dev"], params=list(map(float, p)))) + "\n")
        if args.verbose:
            for a in sorted(rows):
                r = rows[a]
                print("      %-12s wall %.4f off %s" %
                      (a, r["wall"], ("%.4f" % r["off"]) if r["off"] == r["off"] else " n/a"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
