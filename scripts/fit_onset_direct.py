"""Predict ONSET TIME directly from deploy-legal physics features.  Ridge first.

PHASE6_HANDOFF 9: "The GNN residual does not beat ridge on the same features -> do not ship
the network."  This is that ridge, and it is also the experiment that decides whether a
direct onset model is worth building at all.

WHY DIRECT.  Every field oracle in ``scripts/diag_lever_panel.py`` -- perfect wall AP,
perfect time-varying flow, both -- scored BELOW the frozen-ap baseline on mean-over-time,
while the onset oracle scored +0.099.  The ODE's first-crossing readout loses about a
quarter of the ordering that its own inputs carry (|rho| 0.877 as a feature -> 0.649 as an
onset).  So the physics keeps the committed SET, which is already at the flow-oracle
ceiling, and the regression takes over the timing.

PROTOCOL.  Train on FIT, select on DEV, **SEALED is not opened at all in this script** --
it stays closed until a final model is frozen.  The bar to beat is the physics ODE's own
onset ordering on the same vessels.

    python scripts/fit_onset_direct.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.onset_features import (  # noqa: E402
    FEATURE_NAMES, build_features, hop_distance, onset_target,
)

CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/onset_direct")


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def load(names, bio, C):
    out = {}
    for n in names:
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        X, S = build_features(z, bio, C=C)
        y, valid = onset_target(z)
        m = S & valid                       # train/score only where BOTH commit
        if m.sum() < 8:
            continue
        out[n] = dict(X=X, S=S, y=y, valid=valid, mask=m, z=z,
                      gate=X[:, FEATURE_NAMES.index("gate")],
                      sr=z["sr0"], hop=X[:, FEATURE_NAMES.index("hop")])
    return out


def per_vessel_rho(data, names, predict) -> dict:
    return {n: spear(predict(data[n]), data[n]["y"][data[n]["mask"]]) for n in names}


def summary(d):
    v = np.array([x for x in d.values() if np.isfinite(x)])
    return (float(np.mean(v)), float(np.median(v)), len(v)) if len(v) else (np.nan, np.nan, 0)


def main() -> int:
    bio = BiochemConfig(phase="biochem")
    OUT.mkdir(parents=True, exist_ok=True)
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    fit, dev, sealed = prot["fit"], prot["dev"], prot["sealed"]
    C = float(prot["best_cl"]["C"])
    print("FIT n=%d   DEV n=%d   SEALED n=%d (NOT OPENED HERE)" % (len(fit), len(dev), len(sealed)))

    data = load(fit + dev, bio, C)
    fit = [n for n in fit if n in data]
    dev = [n for n in dev if n in data]
    print("usable: FIT %d, DEV %d\n" % (len(fit), len(dev)))

    # ------------------------------------------------- single-feature reference points
    print("=" * 86)
    print("A. WHAT EACH FEATURE CARRIES ON ITS OWN   (|rho| with GT onset, per vessel)")
    print("=" * 86)
    singles = {}
    for i, f in enumerate(FEATURE_NAMES):
        r = {n: spear(data[n]["X"][data[n]["mask"], i], data[n]["y"][data[n]["mask"]])
             for n in fit + dev}
        v = np.array([abs(x) for x in r.values() if np.isfinite(x)])
        singles[f] = float(np.mean(v)) if len(v) else np.nan
    for f, v in sorted(singles.items(), key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -1)):
        print("   %-16s mean |rho| %.3f" % (f, v))

    # ------------------------------------------------------------------- ridge / trees
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xtr = np.concatenate([data[n]["X"][data[n]["mask"]] for n in fit])
    ytr = np.concatenate([data[n]["y"][data[n]["mask"]] for n in fit])
    # rank-normalise the target WITHIN vessel: absolute onset fractions differ 3x across
    # vessels and the metric is a per-vessel rank correlation, so fitting absolute times
    # would spend capacity on a between-vessel offset nobody scores.
    ytr_rank = np.concatenate([
        (np.argsort(np.argsort(data[n]["y"][data[n]["mask"]])).astype(float)
         / max(data[n]["mask"].sum() - 1, 1)) for n in fit])
    sc = StandardScaler().fit(Xtr)

    print("\n" + "=" * 86)
    print("B. MODELS  (train FIT, select DEV; SEALED untouched)")
    print("=" * 86)
    results = {}

    def evaluate(tag, predict_fn):
        rf = per_vessel_rho(data, fit, predict_fn)
        rd = per_vessel_rho(data, dev, predict_fn)
        results[tag] = dict(fit=summary(rf)[0], dev=summary(rd)[0],
                            fit_med=summary(rf)[1], dev_med=summary(rd)[1],
                            per_vessel={**rf, **rd})
        print("   %-28s FIT mean rho %+.3f | DEV mean rho %+.3f  median %+.3f"
              % (tag, results[tag]["fit"], results[tag]["dev"], results[tag]["dev_med"]))

    # physics reference: the ODE's ordering proxy (gate*ap_closure), and hop alone
    gi, ai, hi = (FEATURE_NAMES.index(k) for k in ("gate", "ap_closure", "hop"))
    evaluate("physics: gate*ap_closure",
             lambda d: -(d["X"][d["mask"], gi] * d["X"][d["mask"], ai]))
    evaluate("physics: hop distance", lambda d: d["X"][d["mask"], hi])
    evaluate("physics: gate alone", lambda d: -d["X"][d["mask"], gi])

    for alpha in (0.1, 1.0, 10.0, 100.0):
        m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr_rank)
        evaluate("ridge alpha=%g" % alpha, lambda d, m=m: m.predict(sc.transform(d["X"][d["mask"]])))

    for nest, depth in ((200, 2), (300, 3)):
        g = GradientBoostingRegressor(n_estimators=nest, max_depth=depth, learning_rate=0.05,
                                      subsample=0.8, random_state=0).fit(Xtr, ytr_rank)
        evaluate("gbm n=%d d=%d" % (nest, depth),
                 lambda d, g=g: g.predict(d["X"][d["mask"]]))

    best = max((k for k in results if k.startswith(("ridge", "gbm"))),
               key=lambda k: results[k]["dev"])
    phys = max((k for k in results if k.startswith("physics")), key=lambda k: results[k]["dev"])
    print("\n   best learned : %-24s DEV %+.3f" % (best, results[best]["dev"]))
    print("   best physics : %-24s DEV %+.3f" % (phys, results[phys]["dev"]))
    print("   learned - physics on DEV: %+.3f" % (results[best]["dev"] - results[phys]["dev"]))
    print("\n   NOTE the ODE's own onset ordering on held-out vessels was rho +0.587")
    print("   (scripts/diag_lever_panel.py).  That is the number to beat end-to-end.")

    # ridge coefficients, for the physics story
    m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr_rank)
    print("\n   ridge(alpha=1) coefficients, largest first (+ => predicts LATER onset):")
    for i in np.argsort(-np.abs(m.coef_))[:10]:
        print("      %-16s %+.4f" % (FEATURE_NAMES[i], m.coef_[i]))

    (OUT / "ridge_study.json").write_text(json.dumps(
        dict(singles=singles, results={k: {kk: vv for kk, vv in v.items() if kk != "per_vessel"}
                                       for k, v in results.items()},
             per_vessel={k: v["per_vessel"] for k, v in results.items()},
             fit=fit, dev=dev, C=C), indent=2, default=float), encoding="utf-8")
    print("\nwrote %s" % (OUT / "ridge_study.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
