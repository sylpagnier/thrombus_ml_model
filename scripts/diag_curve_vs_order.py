"""Does the mean-over-time score reward the growth CURVE, or the onset ORDER?

The four-arm run produced a result that only makes sense if these two are different
things.  The AP closure took onset ``spread_ratio`` from 0.392 to 0.739 (GT = 1.0, oracle
0.897) and ``curve_l1`` from 0.1075 to 0.0876 -- a large, real improvement in the shape of
the aggregate growth curve -- and moved the mean-over-time deploy score by **-0.0001**.
Meanwhile the oracle, which differs from the model mainly in getting every node's ORDER
right (rho 1.000 vs 0.602), is +0.099 ahead.

So before any more effort goes into the curve, this settles which quantity the metric
actually pays for, by degrading the ORACLE in one dimension at a time:

  compress   keep GT's onset ORDER, shrink the spread to the model's 0.39  -> costs ?
  shuffle    keep GT's spread and marginal distribution, permute WHO gets which onset -> ?

Both start from the same committed set and the same onset times, so the comparison is
clean.  Whichever degradation costs more is what the metric is measuring, and therefore
what any model -- physics or learned -- has to target.

TRAIN only (FIT+DEV).  Nothing is fitted or selected; SEALED is not touched.

    python scripts/diag_curve_vs_order.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import importlib.util  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, onset_metrics  # noqa: E402

OUT = Path("outputs/ap_closure")
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.20, 0.39, 0.60, 1.00)


def _ev():
    spec = importlib.util.spec_from_file_location(
        "ev", str(REPO / "scripts" / "eval_ap_closure_protocol.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compress(onset, alpha, nt):
    """Keep the order, shrink the spread about the median."""
    ok = onset >= 0
    if ok.sum() < 2:
        return onset.copy()
    med = float(np.median(onset[ok]))
    out = onset.copy().astype(np.float64)
    out[ok] = np.clip(np.round(med + alpha * (onset[ok] - med)), 0, nt - 1)
    return np.where(ok, out.astype(int), -1)


def shuffle(onset, rng):
    """Keep the spread and the exact multiset of onset times; destroy WHO gets which."""
    ok = onset >= 0
    out = onset.copy()
    v = onset[ok].copy()
    rng.shuffle(v)
    out[ok] = v
    return out


def main() -> int:
    ev = _ev()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    prot = json.load(open(OUT / "protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in names:
        c = ev.build_context(n, bio, phys, "gt")
        if c is None:
            continue
        nt = len(c["t"])
        orac = np.where(c["S"], np.where(c["gt_onset"] >= 0, c["gt_onset"], nt - 1), -1)
        model, _ = ev.rollout_onset(c, bio, None, prot["base_da"])
        r = dict(name=n, oracle=ev.arm_metrics(c, orac)["score"],
                 model=ev.arm_metrics(c, model)["score"])
        for a in ALPHAS:
            o = compress(orac, a, nt)
            m = onset_metrics(o, c["gt_onset"], c["t"], c["w"])
            r["compress_%.2f" % a] = ev.arm_metrics(c, o)["score"]
            r["compress_%.2f_spread" % a] = float(m["spread_ratio"])
        sh = []
        for s in SEEDS:
            o = shuffle(orac, np.random.default_rng(s))
            sh.append(ev.arm_metrics(c, o)["score"])
        r["shuffle"] = float(np.mean(sh))
        r["shuffle_sd"] = float(np.std(sh))
        r["curve_l1_shuffle"] = float(curve_l1(shuffle(orac, np.random.default_rng(0)),
                                               c["gt_onset"], c["t"], c["w"]))
        r["curve_l1_oracle"] = float(curve_l1(orac, c["gt_onset"], c["t"], c["w"]))
        rows.append(r)
        print("%-12s oracle %.4f  shuffled %.4f  compressed(0.39) %.4f  model %.4f"
              % (n, r["oracle"], r["shuffle"], r["compress_0.39"], r["model"]))

    print("\n" + "=" * 84)
    print("WHAT THE MEAN-OVER-TIME SCORE PAYS FOR   (TRAIN n=%d)" % len(rows))
    print("=" * 84)
    o = np.array([r["oracle"] for r in rows])
    print("%-34s %8s %9s" % ("arm", "mean", "vs oracle"))
    print("%-34s %8.4f %9s" % ("oracle (perfect order + spread)", o.mean(), "--"))
    for a in ALPHAS:
        v = np.array([r["compress_%.2f" % a] for r in rows])
        sp = np.median([r["compress_%.2f_spread" % a] for r in rows])
        print("%-34s %8.4f %+9.4f   (spread_ratio %.2f)"
              % ("  order KEPT, spread x%.2f" % a, v.mean(), v.mean() - o.mean(), sp))
    v = np.array([r["shuffle"] for r in rows])
    print("%-34s %8.4f %+9.4f   (spread and curve UNCHANGED)"
          % ("  spread KEPT, order DESTROYED", v.mean(), v.mean() - o.mean()))
    m = np.array([r["model"] for r in rows])
    print("%-34s %8.4f %+9.4f" % ("the shipped model", m.mean(), m.mean() - o.mean()))

    cl_o = np.mean([r["curve_l1_oracle"] for r in rows])
    cl_s = np.mean([r["curve_l1_shuffle"] for r in rows])
    print("\n  Shuffling leaves the aggregate growth curve IDENTICAL by construction:")
    print("     curve_l1 oracle %.4f -> shuffled %.4f" % (cl_o, cl_s))
    print("  so any score it loses is PURELY the cost of getting the order wrong.")

    d_ord = o.mean() - v.mean()
    d_spr = o.mean() - np.array([r["compress_0.39"] for r in rows]).mean()
    print("\n  cost of destroying ORDER   : %+.4f" % -d_ord)
    print("  cost of destroying SPREAD  : %+.4f  (to the shipped model's 0.39)" % -d_spr)
    print("\n  VERDICT: the metric is driven by %s."
          % ("ORDER" if d_ord > d_spr else "SPREAD"))

    (OUT / "curve_vs_order.json").write_text(json.dumps(rows, indent=2, default=float),
                                             encoding="utf-8")
    print("\nwrote %s" % (OUT / "curve_vs_order.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
