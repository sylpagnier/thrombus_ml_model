"""Verdict table for the temporal-only runs: does the head beat the physics baseline?

Reports the three-way decomposition per arm -- hard physics / tuned differentiable base /
base + temporal head -- so the head's marginal contribution is separated from the base's,
which is what every earlier round conflated.

The headline the user asked for is the MEDIAN deploy_clot_score on sealed; mean and the
trajectory metric are reported alongside because a head that lifts the median by breaking
the curve has not done its job.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except Exception:                                          # pragma: no cover
    stats = None

OUT = Path("outputs/temporal_only")


def main() -> int:
    runs = []
    for p in sorted(OUT.glob("*.json")):
        if p.stem.startswith("_"):
            continue
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not runs:
        print("no runs in %s" % OUT)
        return 1

    print("=" * 94)
    print("TEMPORAL HEAD ONLY -- sealed set, clean protocol, base parity-gated")
    print("=" * 94)
    print("%-14s %-5s %6s %8s | %8s %8s %8s | %9s %9s"
          % ("run", "flow", "parity", "DEVmed", "PHYSmed", "BASEmed", "HEADmed",
             "head-base", "head-phys"))
    for r in runs:
        sealed = r["sealed"]
        ph = [r["sealed_physics"][n] for n in sealed]
        ba = [r["sealed_base"][n]["score"] for n in sealed]
        hd = [r["sealed_head"][n]["score"] for n in sealed]
        print("%-14s %-5s %+6.3f %8.4f | %8.4f %8.4f %8.4f | %+9.4f %+9.4f"
              % ("%s_s%d" % (r["flow"], r["seed"]), r["flow"], r["parity_gap"],
                 r["best_dev_median"], np.median(ph), np.median(ba), np.median(hd),
                 np.median(hd) - np.median(ba), np.median(hd) - np.median(ph)))

    for flow in ("gt", "pred"):
        sel = [r for r in runs if r["flow"] == flow]
        if not sel:
            continue
        sealed = sel[0]["sealed"]
        # one number per vessel, averaged over seeds -- the honest unit
        ph = {n: sel[0]["sealed_physics"][n] for n in sealed}
        ba = {n: float(np.mean([r["sealed_base"][n]["score"] for r in sel])) for n in sealed}
        hd = {n: float(np.mean([r["sealed_head"][n]["score"] for r in sel])) for n in sealed}
        cb = float(np.mean([r["sealed_base"][n]["curve_l1"] for r in sel for n in sealed]))
        ch = float(np.mean([r["sealed_head"][n]["curve_l1"] for r in sel for n in sealed]))
        print("\n--- flow=%s  (%d seed%s) ---" % (flow, len(sel), "s" if len(sel) > 1 else ""))
        print("  %-12s %9s %9s %9s | %9s" % ("vessel", "physics", "base", "base+head", "head-base"))
        for n in sealed:
            print("  %-12s %9.4f %9.4f %9.4f | %+9.4f" % (n, ph[n], ba[n], hd[n], hd[n] - ba[n]))
        pv, bv, hv = (np.array([ph[n] for n in sealed]), np.array([ba[n] for n in sealed]),
                      np.array([hd[n] for n in sealed]))
        print("  %-22s median %.4f  mean %.4f" % ("physics baseline", np.median(pv), pv.mean()))
        print("  %-22s median %.4f  mean %.4f  curveL1 %.4f"
              % ("tuned base (no head)", np.median(bv), bv.mean(), cb))
        print("  %-22s median %.4f  mean %.4f  curveL1 %.4f"
              % ("base + temporal head", np.median(hv), hv.mean(), ch))
        print("  HEAD marginal   : median %+.4f  mean %+.4f  curveL1 %+.4f  (%d/%d vessels up)"
              % (np.median(hv) - np.median(bv), hv.mean() - bv.mean(), ch - cb,
                 int((hv > bv).sum()), len(hv)))
        print("  vs PHYSICS      : median %+.4f  mean %+.4f"
              % (np.median(hv) - np.median(pv), hv.mean() - pv.mean()))
        if stats is not None and len(hv) > 2:
            d = hv - bv
            if np.any(d != 0):
                print("  head-vs-base paired t p=%.3f  wilcoxon p=%.3f"
                      % (stats.ttest_1samp(d, 0.0).pvalue, stats.wilcoxon(d).pvalue))
            else:
                print("  head-vs-base: identical on every vessel -- the head is INERT")
            d2 = hv - pv
            print("  head-vs-physics paired t p=%.3f" % stats.ttest_1samp(d2, 0.0).pvalue)
    print("\n" + "=" * 94)
    print("Read the HEAD marginal row: that is the only number attributable to the ML head.")
    print("'vs PHYSICS' folds in the base, which is physics re-implemented differentiably.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
