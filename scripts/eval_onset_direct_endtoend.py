"""Does predicting onset DIRECTLY convert into mean-over-time score?

``scripts/fit_onset_direct.py`` measured onset ORDERING and found two things:

  * the gate, used straight as an onset ranker, reaches rho 0.62 (FIT) / 0.85 (DEV) --
    while the ODE it feeds produces only 0.587.  The readout is throwing away ordering its
    own input already has.
  * no learned model beat that one-feature baseline (best GBM DEV 0.802 vs gate 0.850,
    and GBM is badly overfit: FIT 0.966).  PHASE6_HANDOFF 9 says do not ship the network.

But rho is not the deliverable, and this project has now twice seen rho gains fail to
convert (the AP closure; every field oracle in the lever panel).  So this evaluates the
direct-onset predictors END TO END on the metric of record, with the mask held at ``S``.

THE LEVEL PROBLEM.  A ranker gives order, not time.  Mapping ranks onto the horizon
uniformly would spread onsets far wider than GT and over-shoot ``spread_ratio``, which the
lever panel showed is itself costly.  So ranks are mapped through an empirical quantile
function of GT onset fractions **pooled over FIT vessels only** -- learned, deployable, and
it gives the arm the right marginal distribution without seeing the test vessel.

SEALED IS NOT OPENED HERE.

    python scripts/eval_onset_direct_endtoend.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.onset_features import (  # noqa: E402
    FEATURE_NAMES, build_features, onset_target,
)

CACHE = Path("outputs/wall_species_cache")
OUT = Path("outputs/onset_direct")


def _ev():
    spec = importlib.util.spec_from_file_location(
        "ev", str(REPO / "scripts" / "eval_ap_closure_protocol.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def quantile_map(fit_fracs: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Map ranks in [0,1] onto the FIT cohort's empirical GT onset-fraction distribution."""
    q = np.quantile(fit_fracs, np.clip(ranks, 0.0, 1.0))
    return np.clip(q, 0.0, 1.0)


def to_onset(c, frac_wall, nt):
    """Wall-only fractional onset -> full-node index array, restricted to the shipped mask.

    Features are built on the wall subgraph (a few hundred nodes); the scoring context is
    the full mesh (~20k).  ``c['S']`` stays the authoritative mask so every arm here is
    scored on exactly the set the physics model ships -- only the timing differs.
    """
    full = np.zeros(len(c["w"]), dtype=np.float64)
    full[c["wall_idx"]] = np.asarray(frac_wall, dtype=np.float64)
    idx = np.clip(np.round(full * (nt - 1)), 0, nt - 1).astype(int)
    return np.where(c["S"], idx, -1)


def main() -> int:
    ev = _ev()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    fit, dev = prot["fit"], prot["dev"]
    C = float(prot["best_cl"]["C"])
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    feats, ctx = {}, {}
    for n in fit + dev:
        p = CACHE / f"{n}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        X, S = build_features(z, bio, C=C)
        y, valid = onset_target(z)
        c = ev.build_context(n, bio, phys, "gt")
        if c is None:
            continue
        feats[n] = dict(X=X, S=S, y=y, valid=valid, z=z)
        ctx[n] = c
    fit = [n for n in fit if n in ctx]
    dev = [n for n in dev if n in ctx]
    print("contexts %d in %.0fs   FIT %d  DEV %d\n" % (len(ctx), time.time() - t0, len(fit), len(dev)))

    # ---- the FIT-only onset distribution and the FIT-only models
    fit_fracs = np.concatenate([feats[n]["y"][feats[n]["S"] & feats[n]["valid"]] for n in fit])
    print("FIT GT onset fraction: p10 %.2f  median %.2f  p90 %.2f"
          % tuple(np.percentile(fit_fracs, [10, 50, 90])))

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    Xtr = np.concatenate([feats[n]["X"][feats[n]["S"] & feats[n]["valid"]] for n in fit])
    ytr = np.concatenate([feats[n]["y"][feats[n]["S"] & feats[n]["valid"]] for n in fit])
    sc = StandardScaler().fit(Xtr)
    ridge = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=0).fit(Xtr, ytr)
    gi = FEATURE_NAMES.index("gate")

    def rank01(v):
        return np.argsort(np.argsort(v)).astype(float) / max(len(v) - 1, 1)

    def arms(n):
        f, c = feats[n], ctx[n]
        nt = len(c["t"])
        out = {}
        on_base, _ = ev.rollout_onset(c, bio, None, prot["base_da"])
        out["physics ODE"] = on_base
        # gate ordering + FIT-learned onset distribution.  Zero learned parameters in the
        # ORDERING; the only learned object is the marginal, taken from FIT vessels.
        r = 1.0 - rank01(f["X"][:, gi])            # bigger gate -> earlier
        out["gate rank + FIT dist"] = to_onset(c, quantile_map(fit_fracs, r), nt)
        out["ridge (absolute)"] = to_onset(c, ridge.predict(sc.transform(f["X"])), nt)
        out["gbm (absolute)"] = to_onset(c, gbm.predict(f["X"]), nt)
        out["gbm rank + FIT dist"] = to_onset(
            c, quantile_map(fit_fracs, rank01(gbm.predict(f["X"]))), nt)
        out["onset oracle"] = np.where(
            c["S"], np.where(c["gt_onset"] >= 0, c["gt_onset"], nt - 1), -1)
        return out

    labels = ["physics ODE", "gate rank + FIT dist", "ridge (absolute)", "gbm (absolute)",
              "gbm rank + FIT dist", "onset oracle"]
    R = {k: {} for k in labels}
    for n in fit + dev:
        a = arms(n)
        for k in labels:
            R[k][n] = ev.arm_metrics(ctx[n], a[k])

    for tag, names in (("DEV (selection)", dev), ("FIT (in-sample for the models)", fit)):
        print("\n" + "=" * 96)
        print("%s   n=%d   MEAN-over-time" % (tag, len(names)))
        print("=" * 96)
        print("%-24s %8s %8s %8s | %8s %8s %8s | %8s"
              % ("arm", "mean", "early", "late", "curveL1", "rho", "spread", "vs ODE"))
        base = None
        for k in labels:
            g = lambda kk: float(np.nanmean([R[k][n][kk] for n in names]))     # noqa: E731
            m = g("score")
            if base is None:
                base = m
            print("%-24s %8.4f %8.4f %8.4f | %8.4f %+8.3f %8.3f | %+8.4f"
                  % (k, m, g("score_early"), g("score_late"), g("curve_l1"), g("rho"),
                     g("spread_ratio"), m - base))

    dev_scores = {k: float(np.nanmean([R[k][n]["score"] for n in dev])) for k in labels}
    ode, orac = dev_scores["physics ODE"], dev_scores["onset oracle"]
    best = max((k for k in labels if k not in ("physics ODE", "onset oracle")),
               key=lambda k: dev_scores[k])
    print("\n   DEV prize %+.4f;  best direct-onset arm %s recovers %+.4f (%.0f%%)"
          % (orac - ode, best, dev_scores[best] - ode,
             100.0 * (dev_scores[best] - ode) / (orac - ode) if abs(orac - ode) > 1e-9 else 0))
    print("\n   SEALED was not opened.")

    (OUT / "endtoend.json").write_text(json.dumps(
        dict(dev=dev, fit=fit, per_vessel={k: R[k] for k in labels},
             dev_scores=dev_scores), indent=2, default=float), encoding="utf-8")
    print("   wrote %s   (%.0fs)" % (OUT / "endtoend.json", time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
