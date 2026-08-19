"""Aggregate every ML-corrector run into one verdict table, with paired statistics.

Reads ``outputs/ml_clean_protocol/*.json`` (v1: endpoint-only loss, unbounded residual)
and ``outputs/ml_v2/*.json`` (v2: trajectory supervision, split objectives, bounded +
uncertainty-gated residuals), and compares both against the zero-parameter physics model
on the SEALED set.

Per-vessel deltas are paired, so the test is a paired t / Wilcoxon over the 8 sealed
vessels, not an unpaired comparison of means.  Seeds are pooled per arm and also reported
individually, because a single run's sealed number is a peak-picked estimate off a noisy
DEV trace and is not on its own trustworthy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except Exception:                                          # pragma: no cover
    stats = None

V1 = Path("outputs/ml_clean_protocol")
V2 = Path("outputs/ml_v2")


def load(dirpath, is_v2):
    runs = []
    for p in sorted(dirpath.glob("*.json")):
        if p.stem.startswith("_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sealed = r.get("sealed")
        phys = r.get("physics")
        if not sealed or not phys:
            continue
        if is_v2:
            ml = {k: v["score"] for k, v in r.get("ml", {}).items()}
            curve = {k: v.get("curve_l1") for k, v in r.get("ml", {}).items()}
            flow = r.get("flow", "gt")
        else:
            ml = r.get("ml_sealed", {})
            curve = {}
            flow = "gt"
        if not ml:
            continue
        runs.append(dict(name=p.stem, version="v2" if is_v2 else "v1", flow=flow,
                         sealed=sealed, phys=phys, ml=ml, curve=curve,
                         dev=r.get("best_dev"), dev_bar=r.get("dev_bar")))
    return runs


def paired(ml, phys, names):
    d = np.array([ml[n] - phys[n] for n in names if n in ml])
    out = dict(n=len(d), mean=float(d.mean()), sd=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
               pos=int((d > 0).sum()))
    if stats is not None and len(d) > 1:
        out["t_p"] = float(stats.ttest_1samp(d, 0.0).pvalue)
        try:
            out["w_p"] = float(stats.wilcoxon(d).pvalue)
        except Exception:
            out["w_p"] = float("nan")
        rng = np.random.default_rng(0)
        bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(20000)]
        out["ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def main() -> int:
    runs = load(V1, False) + load(V2, True)
    if not runs:
        print("no runs found in %s or %s" % (V1, V2))
        return 1
    print("=" * 92)
    print("ML CORRECTOR vs ZERO-PARAMETER PHYSICS -- sealed set, clean protocol")
    print("=" * 92)
    print("%-22s %-4s %-5s %8s %8s %9s %8s %6s %7s"
          % ("run", "ver", "flow", "DEV ML", "DEVbar", "SEALED ML", "SEALbar", "delta", "pos"))
    for r in runs:
        names = [n for n in r["sealed"] if n in r["ml"]]
        ml_m = float(np.mean([r["ml"][n] for n in names]))
        ph_m = float(np.mean([r["phys"][n] for n in names]))
        st = paired(r["ml"], r["phys"], names)
        print("%-22s %-4s %-5s %8s %8s %9.4f %8.4f %+8.4f %3d/%d"
              % (r["name"], r["version"], r["flow"],
                 "%.4f" % r["dev"] if r["dev"] else "--",
                 "%.4f" % r["dev_bar"] if r.get("dev_bar") else "--",
                 ml_m, ph_m, ml_m - ph_m, st["pos"], st["n"]))

    for ver, flow in (("v1", "gt"), ("v2", "gt"), ("v2", "pred")):
        sel = [r for r in runs if r["version"] == ver and r["flow"] == flow]
        if not sel:
            continue
        allnames = sorted({n for r in sel for n in r["sealed"] if n in r["ml"]})
        pooled = np.array([r["ml"][n] - r["phys"][n] for r in sel for n in allnames
                           if n in r["ml"]])
        # seed-averaged per-vessel delta: the honest unit, one number per vessel
        per_vessel = {n: np.mean([r["ml"][n] - r["phys"][n] for r in sel if n in r["ml"]])
                      for n in allnames}
        d = np.array(list(per_vessel.values()))
        print("\n--- %s / flow=%s   (%d seed%s pooled) ---"
              % (ver.upper(), flow, len(sel), "s" if len(sel) > 1 else ""))
        print("  seed-averaged per-vessel delta:")
        for n in allnames:
            print("     %-12s %+.4f" % (n, per_vessel[n]))
        print("  mean %+.4f   sd %.4f   %d/%d vessels positive"
              % (d.mean(), d.std(ddof=1) if len(d) > 1 else 0.0, (d > 0).sum(), len(d)))
        if stats is not None and len(d) > 1:
            print("  paired t p=%.3f   wilcoxon p=%.3f"
                  % (stats.ttest_1samp(d, 0.0).pvalue,
                     stats.wilcoxon(d).pvalue if len(d) > 2 else float("nan")))
            rng = np.random.default_rng(0)
            bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(20000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print("  bootstrap 95%% CI [%+.4f, %+.4f]  ->  %s"
                  % (lo, hi, "SIGNIFICANT" if lo > 0 else "not distinguishable from zero"))
        cv = [r["curve"] for r in sel if r["curve"]]
        if cv:
            allc = [v for c in cv for v in c.values() if v is not None]
            if allc:
                print("  trajectory curve_l1 on sealed: mean %.4f  (physics arm-A ref 0.0903)"
                      % float(np.mean(allc)))

    print("\n" + "=" * 92)
    v1 = [r for r in runs if r["version"] == "v1"]
    v2g = [r for r in runs if r["version"] == "v2" and r["flow"] == "gt"]
    if v1 and v2g:
        def m(sel):
            return float(np.mean([np.mean([r["ml"][n] - r["phys"][n]
                                           for n in r["sealed"] if n in r["ml"]]) for r in sel]))
        print("v1 (endpoint loss, unbounded)      sealed delta %+.4f" % m(v1))
        print("v2 (traj loss, bounded+gated)      sealed delta %+.4f" % m(v2g))
        print("fixes 1-4 changed the sealed delta by %+.4f" % (m(v2g) - m(v1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
