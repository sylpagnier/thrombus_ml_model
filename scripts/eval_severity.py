"""Re-score saved predictions under BOTH metrics, split by protocol and geometry class.

Changing the metric invalidates comparison with every earlier number, so this reports the
old and the new side by side and shows what moved and why.

    python scripts/eval_severity.py --tags rec3s,rec5s,rec3s6,rec3o
    python scripts/eval_severity.py --tags rec3s,rec5s,rec3s6,rec3o --tau-sweep
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

from eval_readouts import load_scores  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache, splits  # noqa: E402
from src.clot_ml.geometry_class import classify, is_priority, width_stats  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityConfig, SeverityScorer  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"
GRID = np.linspace(0.02, 0.998, 60)


def geometry_classes(anchors):
    out = {}
    for a in anchors:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        s = width_stats(d)
        out[a] = dict(cls=classify(s, a), **s)
    return out


def build_scorers(cache, cfg):
    return {a: SeverityScorer(S["edge_index"], S["y"] > 0.5, len(S["wall"]), cfg)
            for a, S in cache.items()}


def tune_thresholds(scorers, cache, scores, anchors):
    """Independent wall / off-wall cuts, tuned under whichever metric ``scorers`` carries."""
    def best(domain_of):
        bv, bt = -1e9, float(GRID[0])
        for t in GRID:
            vals = []
            for a in anchors:
                S = cache[a]
                d = domain_of(S)
                v = scorers[a].score((scores[a] >= t) & d, d)
                if v == v:
                    vals.append(v)
            if vals and np.mean(vals) > bv:
                bv, bt = float(np.mean(vals)), float(t)
        return bt
    return best(lambda S: S["wall"]), best(lambda S: ~S["wall"])


def evaluate(scorers, cache, store, folds, fit, dev):
    rows = {}
    for k, held in folds.items():
        tr = [a for a in fit if a not in held]
        sc = {a: store["%d|%s" % (k, a)] for a in fit}
        tw, to = tune_thresholds(scorers, cache, sc, tr)
        for a in held:
            S = cache[a]
            pred = ((sc[a] >= tw) & S["wall"]) | ((sc[a] >= to) & ~S["wall"])
            rows[a] = dict(wall=scorers[a].score(pred, S["wall"]),
                           off=scorers[a].score(pred, ~S["wall"]))
    sc = {a: store["final|%s" % a] for a in fit + dev}
    tw, to = tune_thresholds(scorers, cache, sc, fit)
    for a in dev:
        S = cache[a]
        pred = ((sc[a] >= tw) & S["wall"]) | ((sc[a] >= to) & ~S["wall"])
        rows[a] = dict(wall=scorers[a].score(pred, S["wall"]),
                       off=scorers[a].score(pred, ~S["wall"]))
    return rows


def agg(rows, anchors, key):
    v = [rows[a][key] for a in anchors if a in rows and rows[a][key] == rows[a][key]]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--tau-sweep", action="store_true")
    ap.add_argument("--save", default="outputs/phase10_severity.json")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    cache = attach_physics(load_cache("gt"))
    fit, dev = splits(cache)
    store, folds = load_scores(tags)
    geo = geometry_classes(fit + dev)

    print("[geometry classes]")
    print("%-12s %6s %8s %10s  %s" % ("vessel", "split", "bulge", "narrowing", "class"))
    for a in fit + dev:
        if a not in geo:
            continue
        g = geo[a]
        print("%-12s %6s %8.3f %10.3f  %s%s"
              % (a, "dev" if a in dev else "fit", g["bulge"], g["narrowing"], g["cls"],
                 "   <- priority" if is_priority(g["cls"]) else ""))
    prio = [a for a in fit + dev if a in geo and is_priority(geo[a]["cls"])]
    print("priority (stenosis/aneurysm) n=%d: %s" % (len(prio), ", ".join(prio)))

    results = {}
    for name, cfg in (("legacy", LEGACY), ("severity", DEFAULT)):
        sc_ = build_scorers(cache, cfg)
        rows = evaluate(sc_, cache, store, folds, fit, dev)
        results[name] = rows
        print("\n=== %s ===" % name)
        for label, anchors in (("FIT", fit), ("DEV", dev),
                               ("PRIORITY(sten/aneu)", prio),
                               ("BASELINE geom", [a for a in fit + dev if a not in prio])):
            w, nw = agg(rows, anchors, "wall")
            o, no = agg(rows, anchors, "off")
            print("  %-22s wall %.4f (n=%2d)   off %.4f (n=%2d)" % (label, w, nw, o, no))

    print("\n[per-vessel, legacy -> severity]")
    print("%-12s %6s %-18s %15s %15s" % ("vessel", "split", "class", "wall", "off"))
    for a in fit + dev:
        if a not in results["legacy"]:
            continue
        lw, ow = results["legacy"][a]["wall"], results["legacy"][a]["off"]
        sw, so = results["severity"][a]["wall"], results["severity"][a]["off"]
        f = lambda x, y: ("%.3f->%.3f" % (x, y)) if x == x else "   n/a    "
        print("%-12s %6s %-18s %15s %15s"
              % (a, "dev" if a in dev else "fit", geo.get(a, {}).get("cls", "?"),
                 f(lw, sw), f(ow, so)))

    if args.tau_sweep:
        print("\n[tau_abs / rho sweep on the off-wall domain]")
        print("%8s %6s | %8s %8s %8s" % ("tau_abs", "rho", "FIT off", "DEV off", "PRIO off"))
        for tau in (0.0, 3.0, 5.0, 8.0, 12.0):
            for rho in (0.15, 0.25, 0.35):
                cfg = SeverityConfig(tau_abs=tau, rho=rho)
                r = evaluate(build_scorers(cache, cfg), cache, store, folds, fit, dev)
                print("%8.0f %6.2f | %8.4f %8.4f %8.4f"
                      % (tau, rho, agg(r, fit, "off")[0], agg(r, dev, "off")[0],
                         agg(r, prio, "off")[0]))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(tags=tags, geometry=geo,
                                   legacy=results["legacy"], severity=results["severity"],
                                   config=DEFAULT.as_dict()), indent=2, default=float))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
