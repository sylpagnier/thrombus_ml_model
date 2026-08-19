"""Honest evaluation: leave-one-vessel-out on FIT, single fit for DEV.

19 vessels and ~0.7% positive nodes is a regime where an in-sample FIT number means
nothing -- the first GBM read FIT wall 0.90 / off 0.92 in-sample and DEV 0.83 / 0.46.
Every FIT number reported from here is leave-one-vessel-out, and the decision threshold is
chosen inside each fold on that fold's own training vessels.
"""
from __future__ import annotations

import numpy as np

from src.clot_ml.fastscore import VesselScorer


class Bench:
    """Precomputed scorers + the split-level aggregation, so arms are comparable."""

    def __init__(self, cache: dict, fit: list[str], dev: list[str]):
        self.cache, self.fit, self.dev = cache, list(fit), list(dev)
        self.vs = {}
        for a, S in cache.items():
            self.vs[a] = VesselScorer(S["edge_index"], S["y"] > 0.5, len(S["wall"]))

    def row(self, a: str, pred: np.ndarray) -> dict:
        S = self.cache[a]
        wall = S["wall"]
        return dict(wall=self.vs[a].score(pred, wall),
                    off=self.vs[a].score(pred, ~wall),
                    full=self.vs[a].score(pred, None))

    def summarise(self, rows: dict) -> dict:
        out = {}
        for split, anchors in (("fit", self.fit), ("dev", self.dev)):
            acc = {k: [] for k in ("wall", "off", "full")}
            for a in anchors:
                if a not in rows:
                    continue
                for k in acc:
                    v = rows[a][k]
                    if v == v:
                        acc[k].append(v)
            out[split] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
            out[split]["n"] = len([a for a in anchors if a in rows])
        return out

    def objective(self, rows: dict, anchors: list[str]) -> float:
        """What thresholds are selected on: both targets weighted equally."""
        w = [rows[a]["wall"] for a in anchors if a in rows and rows[a]["wall"] == rows[a]["wall"]]
        o = [rows[a]["off"] for a in anchors if a in rows and rows[a]["off"] == rows[a]["off"]]
        return (np.mean(w) if w else 0.0) + (np.mean(o) if o else 0.0)

    def pick_threshold(self, scores: dict, anchors: list[str], grid) -> float:
        best, best_t = -1e9, float(grid[0])
        for t in grid:
            rows = {a: self.row(a, scores[a] >= t) for a in anchors}
            v = self.objective(rows, anchors)
            if v > best:
                best, best_t = v, float(t)
        return best_t


def run_lovo(fit_fn, bench: Bench, grid, *, verbose=False):
    """``fit_fn(train_anchors) -> predict(anchor) -> per-node score``."""
    rows, ths = {}, {}
    for held in bench.fit:
        tr = [a for a in bench.fit if a != held]
        predict = fit_fn(tr)
        sc = {a: predict(a) for a in tr + [held]}
        t = bench.pick_threshold(sc, tr, grid)
        rows[held] = bench.row(held, sc[held] >= t)
        ths[held] = t
        if verbose:
            r = rows[held]
            print("      [lovo] %-12s t=%.3f wall %.4f off %s"
                  % (held, t, r["wall"],
                     ("%.4f" % r["off"]) if r["off"] == r["off"] else "n/a"), flush=True)
    predict = fit_fn(bench.fit)
    sc = {a: predict(a) for a in bench.fit + bench.dev}
    t_all = bench.pick_threshold(sc, bench.fit, grid)
    for a in bench.dev:
        rows[a] = bench.row(a, sc[a] >= t_all)
        ths[a] = t_all
    return bench.summarise(rows), rows, ths
