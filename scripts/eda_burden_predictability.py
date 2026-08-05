"""Step 3: is per-vessel clot burden predictable from deployable conditioning?

Per-vessel graded burden spans ~6.5x across the cohort and nothing tells the model which
regime a new vessel is in (docs/WALL_MODEL_PLAN.md s4 Step 3). This regresses the graded
burden on the geometry/flow descriptors the model already receives and reports whether
that mapping exists.

**The headline number is leave-one-out CV R-squared, not in-sample R-squared.** With ~43
vessels and 9 features, in-sample R-squared is inflated by construction and would happily
report "burden is learnable" from pure noise. A permutation test on the same LOO statistic
gives the p-value. Ridge alpha is chosen by an *inner* CV on each training fold, so the
selection never sees the held-out vessel.

Deliberately excluded from the feature set: ``clot_frac``, ``n_active``, ``offwall_frac``,
``mat_max``, ``onset_frac``. Those are computed from the clot field itself -- regressing
burden on them would be leakage, not prediction. ``nn_train_dist`` is also excluded by
default because it encodes a train/test split choice rather than a property of the vessel
(enable with ``--include-nn-dist`` if you want it).

Reading the result, per the plan:
  * **High LOO R-squared** -> thickness is learnable from what the model sees; the fix is
    conditioning / loss, and a retrain should target it directly.
  * **LOO R-squared at or below 0** -> no wall model can infer burden from its inputs. You
    need an explicit burden-conditioning input or a calibrated operating point, and cohort
    permutations are not going to help.

Pure analysis: CPU only, no model, no rollout, no GPU.

    python scripts/eda_burden_predictability.py
    python scripts/eda_burden_predictability.py --min-burden 0.005 --target offwall_share
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eda_clot_burden import ANCHOR_DIR, anchor_burden  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

GEN_EDA_JSON = "outputs/biochem/eda/generalization_eda.json"
BURDEN_JSON = "outputs/biochem/eda/clot_burden_graded.json"
OUT_JSON = "outputs/biochem/eda/burden_predictability.json"

# Deployable conditioning only -- nothing derived from the clot field (see module docstring).
DEFAULT_FEATURES = [
    "re_actual", "d_bar", "w_p5", "w_med", "w_p95",
    "stenosis_ratio", "expansion_ratio", "curvature", "aspect",
]
LEAKY = {"clot_frac", "n_active", "offwall_frac", "mat_max", "onset_frac"}


def _loo_r2(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    *,
    fixed_alpha: float | None = None,
) -> tuple[float, np.ndarray]:
    """Leave-one-out R-squared. Returns ``(r2, preds)``.

    ``fixed_alpha=None`` selects alpha by inner CV on each training fold, so the held-out
    vessel is never seen -- that is the honest headline. The permutation null passes the
    full-data alpha instead: refitting the inner CV thousands of times costs minutes and
    the null is a *reference distribution for the same estimator*, so holding the
    regularisation fixed across permutations is the right comparison anyway.
    """
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    preds = np.zeros(n)
    for i in range(n):
        tr = np.ones(n, dtype=bool)
        tr[i] = False
        if fixed_alpha is None:
            est = RidgeCV(alphas=alphas, cv=min(5, n - 1))
        else:
            est = Ridge(alpha=float(fixed_alpha))
        model = make_pipeline(StandardScaler(), est)
        model.fit(X[tr], y[tr])
        preds[i] = model.predict(X[i : i + 1])[0]
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")), preds


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 3: burden predictability from deployable features")
    ap.add_argument("--gen-eda", default=GEN_EDA_JSON)
    ap.add_argument("--burden-json", default=BURDEN_JSON, help="Cache; recomputed for missing anchors")
    ap.add_argument("--target", default="burden", choices=("burden", "offwall_share", "n_clot"))
    ap.add_argument("--min-burden", type=float, default=0.0,
                    help="Restrict to clot-rich vessels with burden above this (e.g. 0.005)")
    ap.add_argument("--include-nn-dist", action="store_true", help="Add nn_train_dist as a feature")
    ap.add_argument(
        "--include-mirrors",
        action="store_true",
        help="Keep *_mirror_y vessels. Off by default: a mirror is the same vessel, so it "
             "would sit in its twin's LOO training fold and leak the answer.",
    )
    ap.add_argument("--n-perm", type=int, default=2000, help="Permutation-test resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    from scipy import stats

    root = get_project_root()
    gen_path = Path(args.gen_eda)
    if not gen_path.is_absolute():
        gen_path = root / gen_path
    gen = {r["name"]: r for r in json.loads(gen_path.read_text(encoding="utf-8"))}

    burden_path = Path(args.burden_json)
    if not burden_path.is_absolute():
        burden_path = root / burden_path
    cache: dict[str, dict] = {}
    if burden_path.is_file():
        cache = dict((json.loads(burden_path.read_text(encoding="utf-8")).get("per_anchor") or {}))

    feats = list(DEFAULT_FEATURES) + (["nn_train_dist"] if args.include_nn_dist else [])
    bad = LEAKY & set(feats)
    if bad:
        raise ValueError(f"clot-derived features are leakage, refusing: {sorted(bad)}")

    phys = PhysicsConfig(phase="biochem")
    rows: list[dict] = []
    computed = 0
    for name in sorted(gen):
        graph = ANCHOR_DIR / f"{name}.pt"
        if not graph.is_file():
            continue
        if any(gen[name].get(f) is None for f in feats):
            print(f"[skip] {name}: missing descriptor", flush=True)
            continue
        if name not in cache:
            cache[name] = anchor_burden(graph, phys)
            computed += 1
            print(f"[calc] {name} burden={cache[name]['burden']*100:.2f}%", flush=True)
        b = cache[name]
        rows.append({"name": name, "target": float(b[args.target]),
                     "burden": float(b["burden"]),
                     "x": [float(gen[name][f]) for f in feats]})
    if computed:
        burden_path.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(burden_path.read_text(encoding="utf-8")) if burden_path.is_file() else {}
        prev["per_anchor"] = cache
        burden_path.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print(f"[i] burden cache updated ({computed} new) -> {burden_path}", flush=True)

    rows = [r for r in rows if r["burden"] >= args.min_burden]
    if not args.include_mirrors:
        dropped = [r["name"] for r in rows if r["name"].endswith("_mirror_y")]
        rows = [r for r in rows if not r["name"].endswith("_mirror_y")]
        if dropped:
            print(f"[i] dropped {len(dropped)} mirrored duplicates (LOO leakage): {dropped}", flush=True)
    if len(rows) < 8:
        print(f"[ERROR] only {len(rows)} vessels after filtering; too few to regress")
        return 1

    names = [r["name"] for r in rows]
    X = np.array([r["x"] for r in rows], dtype=float)
    y = np.array([r["target"] for r in rows], dtype=float)

    # Constant columns carry no information and break Spearman; drop them explicitly.
    keep = [j for j in range(X.shape[1]) if float(X[:, j].std()) > 1e-12]
    if len(keep) != X.shape[1]:
        const = [feats[j] for j in range(X.shape[1]) if j not in keep]
        print(f"[i] dropped constant features: {const}", flush=True)
        X = X[:, keep]
        feats = [feats[j] for j in keep]
    n, p = X.shape

    print(f"\n=== target={args.target}  n={n} vessels  p={p} features  min_burden={args.min_burden} ===")
    lo, hi = float(y.min()), float(y.max())
    nz = y[y > 0]
    spread = (hi / float(nz.min())) if nz.size else float("nan")
    print(f"target: mean={y.mean():.5f} sd={y.std(ddof=1):.5f} min={lo:.5f} max={hi:.5f} "
          f"n_zero={int((y == 0).sum())} spread(nonzero)={spread:.1f}x")

    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-3, 3, 25)
    full = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=min(5, n - 1))).fit(X, y)
    in_sample_r2 = float(full.score(X, y))
    loo_r2, preds = _loo_r2(X, y, alphas)

    alpha_full = float(full[-1].alpha_)
    rng = np.random.default_rng(args.seed)
    null = np.empty(args.n_perm)
    for k in range(args.n_perm):
        null[k], _ = _loo_r2(X, rng.permutation(y), alphas, fixed_alpha=alpha_full)
    p_value = float((1.0 + (null >= loo_r2).sum()) / (1.0 + args.n_perm))

    print(f"\nin-sample R2 = {in_sample_r2:+.4f}   <- inflated, do not quote")
    print(f"LOO-CV   R2 = {loo_r2:+.4f}   <- headline")
    print(f"permutation p = {p_value:.4f}  (null LOO R2: mean {null.mean():+.4f}, "
          f"p95 {np.quantile(null, 0.95):+.4f}, n={args.n_perm})")

    print(f"\n{'feature':>18} {'spearman':>10} {'p':>9} {'ridge_coef':>11}")
    coefs = full[-1].coef_
    uni = {}
    for j, f in enumerate(feats):
        rho, pv = stats.spearmanr(X[:, j], y)
        uni[f] = {"spearman": float(rho), "p": float(pv), "ridge_coef_std": float(coefs[j])}
        print(f"{f:>18} {rho:+10.3f} {pv:9.4f} {coefs[j]:+11.4f}")

    if loo_r2 <= 0.0:
        verdict = (
            "LOO R2 <= 0: the deployable descriptors carry NO usable information about burden -- "
            "the fitted model is worse than predicting the cohort mean. Per the plan, no wall "
            "model can infer thickness from what it sees; you need explicit burden conditioning "
            "or a calibrated operating point. Cohort re-selection will not fix this."
        )
    elif p_value > 0.05:
        verdict = (
            f"LOO R2 = {loo_r2:+.3f} but permutation p = {p_value:.3f}: not distinguishable from "
            "chance at this sample size. Treat burden as unpredictable unless more vessels move it."
        )
    elif loo_r2 < 0.30:
        verdict = (
            f"LOO R2 = {loo_r2:+.3f} (p = {p_value:.3f}): a real but weak signal. Geometry explains "
            "a minority of burden variance -- conditioning may help at the margin, but it will not "
            "close a 6.5x spread on its own."
        )
    else:
        verdict = (
            f"LOO R2 = {loo_r2:+.3f} (p = {p_value:.3f}): thickness IS learnable from deployable "
            "conditioning. The fix is conditioning / loss, and the Step 4 retrain should target it."
        )
    print(f"\n=> {verdict}")

    print(f"\n{'vessel':>14} {'actual':>9} {'loo_pred':>9} {'err':>9}")
    for i in np.argsort(-y):
        print(f"{names[i]:>14} {y[i]:9.5f} {preds[i]:9.5f} {preds[i] - y[i]:+9.5f}")

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "target": args.target, "n": n, "features": feats, "min_burden": args.min_burden,
        "in_sample_r2": in_sample_r2, "loo_cv_r2": loo_r2, "permutation_p": p_value,
        "n_perm": args.n_perm, "null_loo_r2_mean": float(null.mean()),
        "null_loo_r2_p95": float(np.quantile(null, 0.95)),
        "univariate": uni, "verdict": verdict,
        "per_vessel": [
            {"name": names[i], "actual": float(y[i]), "loo_pred": float(preds[i])}
            for i in range(n)
        ],
    }, indent=2), encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
