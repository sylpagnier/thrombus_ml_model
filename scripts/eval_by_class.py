"""Out-of-fold performance by geometry class, under both metrics.

Every number here is out-of-fold on the geometry-stratified K-fold of
`src/clot_ml/geometry_splits.py`, so a vessel's score always comes from a model that never
saw it.  Readout thresholds are tuned inside each fold on that fold's own training vessels.

    python scripts/eval_by_class.py --tags cv5a,cv5b,cv5c
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, LEGACY, SeverityScorer  # noqa: E402

OUT = REPO / "outputs/phase9_scores"
GRID = np.linspace(0.02, 0.998, 60)


def load(tags):
    zs = [np.load(OUT / f"{t}.npz", allow_pickle=True) for t in tags]
    pool = [str(x) for x in zs[0]["pool"]]
    classes = dict(zip(pool, [str(x) for x in zs[0]["classes"]]))
    folds = {int(k.split("|")[1]): [str(x) for x in zs[0][k]]
             for k in zs[0].files if k.startswith("held|")}
    store = {k: np.mean([z[k] for z in zs], axis=0)
             for k in zs[0].files if "|" in k and not k.startswith("held")}
    return pool, classes, folds, store


def tune(scorers, cache, scores, anchors):
    def best(dom):
        bv, bt = -1e9, float(GRID[0])
        for t in GRID:
            v = [scorers[a].score((scores[a] >= t) & dom(cache[a]), dom(cache[a]))
                 for a in anchors]
            v = [x for x in v if x == x]
            if v and np.mean(v) > bv:
                bv, bt = float(np.mean(v)), float(t)
        return bt
    return best(lambda S: S["wall"]), best(lambda S: ~S["wall"])


def out_of_fold(cfg, cache, pool, folds, store):
    scorers = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                                 len(cache[a]["wall"]), cfg) for a in pool}
    rows = {}
    for k, held in folds.items():
        tr = [a for a in pool if a not in held]
        sc = {a: store["%d|%s" % (k, a)] for a in pool}
        tw, to = tune(scorers, cache, sc, tr)
        for a in held:
            S = cache[a]
            pred = ((sc[a] >= tw) & S["wall"]) | ((sc[a] >= to) & ~S["wall"])
            rows[a] = dict(wall=scorers[a].score(pred, S["wall"]),
                           off=scorers[a].score(pred, ~S["wall"]),
                           n_off=int(((S["y"] > 0.5) & ~S["wall"]).sum()),
                           n_wall=int(((S["y"] > 0.5) & S["wall"]).sum()))
    return rows


def physics_rows(cfg, cache, pool):
    from src.core_physics.physics_lumen_model import adjacency, grow_into_lumen
    from src.config import BiochemConfig
    import predict_wall_clot as P
    bio = BiochemConfig(phase="biochem")
    rows = {}
    for a in pool:
        S = cache[a]
        wall, ei = S["wall"], S["edge_index"]
        A = adjacency(ei, len(wall)).astype(np.int8)
        cur = (S["gate"] > 0) & wall
        adm = (S["sr"] < float(bio.lss) * P.RELAX) & wall
        for _ in range(P.GROW_HOPS):
            cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
        off = grow_into_lumen(cur, wall, A, S["spd"], S["sr"],
                              lumen_hops=P.LUMEN_HOPS, speed_thresh=P.LUMEN_SPEED)
        sc = SeverityScorer(ei, S["y"] > 0.5, len(wall), cfg)
        rows[a] = dict(wall=sc.score(cur | off, wall), off=sc.score(cur | off, ~wall))
    return rows


def agg(rows, anchors, key):
    v = [rows[a][key] for a in anchors if a in rows and rows[a][key] == rows[a][key]]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--save", default="outputs/phase10_by_class.json")
    args = ap.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    cache = attach_physics(load_cache("gt"))
    pool, classes, folds, store = load(tags)

    groups = {
        "ALL": pool,
        "baseline": [a for a in pool if not is_priority(classes[a])],
        "PRIORITY (sten+aneu)": [a for a in pool if is_priority(classes[a])],
        "  aneurysm": [a for a in pool if classes[a] == "aneurysm"],
        "  stenosis": [a for a in pool if classes[a] == "stenosis"],
    }
    results = {}
    for name, cfg in (("legacy", LEGACY), ("severity", DEFAULT)):
        model = out_of_fold(cfg, cache, pool, folds, store)
        phys = physics_rows(cfg, cache, pool)
        results[name] = dict(model=model, physics=phys)
        print("\n=== %s metric, OUT-OF-FOLD (geometry-stratified 5-fold) ===" % name)
        print("%-22s %5s | %-16s %-16s | %-16s %-16s"
              % ("group", "n", "wall model", "wall physics", "off model", "off physics"))
        for g, anchors in groups.items():
            mw, nw = agg(model, anchors, "wall")
            pw, _ = agg(phys, anchors, "wall")
            mo, no = agg(model, anchors, "off")
            po, _ = agg(phys, anchors, "off")
            print("%-22s %5d | %-16s %-16s | %-16s %-16s"
                  % (g, nw, "%.4f" % mw, "%.4f" % pw,
                     ("%.4f (n=%d)" % (mo, no)) if no else "n/a",
                     ("%.4f" % po) if no else "n/a"))

    print("\n[per-vessel, out-of-fold, severity metric]")
    print("%-12s %-18s %6s %6s | %8s %8s | %8s %8s"
          % ("vessel", "class", "nWall", "nOff", "wall", "phys", "off", "phys"))
    m, p = results["severity"]["model"], results["severity"]["physics"]
    for a in sorted(pool, key=lambda x: (not is_priority(classes[x]), x)):
        r, q = m[a], p[a]
        f = lambda v: ("%.4f" % v) if v == v else "   n/a  "
        print("%-12s %-18s %6d %6d | %8s %8s | %8s %8s"
              % (a, classes[a], r["n_wall"], r["n_off"],
                 f(r["wall"]), f(q["wall"]), f(r["off"]), f(q["off"])))

    Path(args.save).write_text(json.dumps(
        dict(tags=tags, classes=classes, folds={str(k): v for k, v in folds.items()},
             results=results), indent=2, default=float))
    print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
