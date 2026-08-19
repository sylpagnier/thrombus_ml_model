"""Verdict for the controlled MeshGraphNet + cGNODE ladder.

Five rungs on identical splits and seeds, so each component's marginal contribution is
separable:

    physics   the zero-parameter model (the bar)
    base      parity-gated differentiable reimplementation of it, no heads
    +MGN      base + MeshGraphNet spatial residual   (bounded, gate OFF, logit-clamped)
    +cGNODE   base + rate-multiplier temporal head   (small-init, trajectory loss)
    +both     both heads

"vs physics" is the number that decides deployment; "marginal vs base" is the number
attributable to the ML heads, because the base is physics re-implemented differentiably
and carries its own error.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except Exception:                                          # pragma: no cover
    stats = None

OUT = Path("outputs/ml_ladder")
RUNGS = ("base", "mgn", "cgnode", "both")
LABEL = {"base": "base (no heads)", "mgn": "+MeshGraphNet",
         "cgnode": "+cGNODE", "both": "+both heads"}


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

    print("=" * 96)
    print("CONTROLLED LADDER -- MeshGraphNet + cGNODE vs the zero-parameter physics model")
    print("=" * 96)
    for r in runs:
        print("  %s_seed%d  parity %+.4f  FIT n=%d (%d augmented)  params: MGN %d / cGNODE %d"
              % (r["flow"], r["seed"], r["parity"], len(r["fit"]), len(r.get("augmented", [])),
                 r["rungs"]["mgn"]["n_params"], r["rungs"]["cgnode"]["n_params"]))

    for flow in ("gt", "pred"):
        sel = [r for r in runs if r["flow"] == flow]
        if not sel:
            continue
        sealed = sel[0]["sealed"]
        ph = {n: float(np.mean([r["physics"][n] for r in sel])) for n in sealed}
        got = {rg: {n: float(np.mean([r["rungs"][rg]["sealed"][n]["score"] for r in sel]))
                    for n in sealed} for rg in RUNGS}
        cl = {rg: float(np.mean([r["rungs"][rg]["sealed"][n]["curve_l1"]
                                 for r in sel for n in sealed])) for rg in RUNGS}
        print("\n--- flow=%s  (%d seed%s, seed-averaged per vessel) ---"
              % (flow, len(sel), "s" if len(sel) > 1 else ""))
        print("  %-12s %9s %9s %9s %9s %9s"
              % ("vessel", "physics", "base", "+MGN", "+cGNODE", "+both"))
        for n in sealed:
            print("  %-12s %9.4f %9.4f %9.4f %9.4f %9.4f"
                  % (n, ph[n], got["base"][n], got["mgn"][n], got["cgnode"][n], got["both"][n]))
        pv = np.array([ph[n] for n in sealed])
        print("\n  %-18s median %.4f  mean %.4f" % ("physics (0 param)", np.median(pv), pv.mean()))
        for rg in RUNGS:
            v = np.array([got[rg][n] for n in sealed])
            line = ("  %-18s median %.4f  mean %.4f  curveL1 %.4f | vs physics %+.4f  vs base %+.4f"
                    % (LABEL[rg], np.median(v), v.mean(), cl[rg], v.mean() - pv.mean(),
                       v.mean() - np.array([got["base"][n] for n in sealed]).mean()))
            print(line)
        if stats is not None and len(sealed) > 2:
            print()
            bv = np.array([got["base"][n] for n in sealed])
            for rg in ("mgn", "cgnode", "both"):
                v = np.array([got[rg][n] for n in sealed])
                d_ph, d_ba = v - pv, v - bv
                pp = stats.ttest_1samp(d_ph, 0.0).pvalue if np.any(d_ph != 0) else float("nan")
                pb = stats.ttest_1samp(d_ba, 0.0).pvalue if np.any(d_ba != 0) else float("nan")
                tag = "INERT (identical to base)" if not np.any(d_ba != 0) else ""
                print("  %-18s vs physics p=%.3f (%d/%d up) | vs base p=%.3f (%d/%d up) %s"
                      % (LABEL[rg], pp, int((d_ph > 0).sum()), len(pv),
                         pb, int((d_ba > 0).sum()), len(pv), tag))
        best = max(RUNGS, key=lambda rg: np.mean([got[rg][n] for n in sealed]))
        bm = np.mean([got[best][n] for n in sealed])
        print("\n  VERDICT (%s): best rung is %s at %.4f vs physics %.4f -> %s"
              % (flow, LABEL[best], bm, pv.mean(),
                 "ML BEATS physics" if bm > pv.mean() else "physics still wins"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
