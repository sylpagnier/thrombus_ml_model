"""Push the FINAL off-wall set: couple it to the wall set, and to the shell.

The final-time score IS the committed set's quality, and off-wall sits at 0.7372 against a
per-vessel oracle cut of 0.8275 (`scripts/diag_readout_ceiling.py`).  Every readout tried so
far chooses the off-wall set **independently of the wall set** -- which throws away the
single strongest structural fact the project has measured about off-wall clot:

  * PHASE7 3.1: an off-wall GT node's nearest wall node is itself GT-committed **99.9%** of
    the time.  There is no other source -- `Mat` is advected from the wall flux and nothing
    else makes it (PHASE7 1.1).
  * PHASE7 3.1 again: off-wall GT clot is **one boundary-layer node row**, normal offset
    1.66-1.80 median edge lengths, `p50 ~ p90`.

So two hard constraints are available for free, and neither has been applied at `t_final`:

    owner   an off-wall node may be committed only if its OWNER is in the committed WALL set
    shell   ... and only if it is on the topological first shell

`docs/PHASE9_ML.md` 4 reports the shell restriction as a LOSS (FIT off 0.534 against 0.597),
but that was measured against a plain threshold, where removing nodes cannot be compensated.
Under the expected-score readout the budget is chosen after the restriction, so recall lost
to the shell can be bought back by committing further down the ranking -- which is exactly
the interaction that measurement could not see.

    python scripts/eval_offwall_final.py --tags v5a,v5b,v5c --cache v5
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

from eval_expected_score_readout import expected_curve  # noqa: E402
from eval_strict import FAMILIES, GRID, apply_adapt, load_scores, tune_adapt  # noqa: E402
from src.clot_ml.data import attach_physics, load_cache  # noqa: E402
from src.clot_ml.geometry_splits import classes_for, is_priority  # noqa: E402
from src.clot_ml.severity_metric import DEFAULT, SeverityScorer  # noqa: E402
from src.clot_ml.softmetric import dilation_operator, to_torch_sparse  # noqa: E402

PACKS = REPO / "data/processed/graphs_biochem_anchors"
GAMMA = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
KSCALE = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
#: Anti-compression exponent on the budget.  1.0 = the expected-score readout unchanged.
#: DELIBERATELY TINY.  A 7x6 (beta, alpha) grid on top of the existing 6x7 (gamma, kscale)
#: one collapsed the held-out score from 0.7359 to 0.6244 while the SELECTION score rose --
#: 1764 combinations fitted on 14 vessels is pure selection noise.  See docs/PHASE10_V4.md.
#: Budget anti-compression, and the reason both grids are SINGLETONS.
#: The measured failure it was meant to fix is real -- `patient032` commits 63 nodes for 120
#: GT (recall 0.417) while `patient005` commits 17 for 4 (precision 0.267), so the chosen
#: budget is compressed toward the cohort middle.  But every widening of the search loses,
#: held out, while the SELECTION score rises (same arm, same code, only the grid size):
#:     beta {1.0}            42 combos   0.7359
#:     beta {1.0,1.4,2.0}   126 combos   0.6852
#: At n=19 the size of the readout search space is itself a hyperparameter, and expanding it
#: costs more than the bias it removes.  Fix the compression with a mechanism, not a knob.
BETA = [1.0]
ALPHA = [0.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cache", default="v5")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = attach_physics(load_cache(args.cache))
    pool, folds, sc_all = load_scores(args.tags.split(","))
    pool = [a for a in pool if a in cache]
    classes = classes_for(pool, PACKS)
    fo = {a: k for k, held in folds.items() for a in held}
    sc = {a: sc_all[(fo[a], a)] for a in pool}
    vs = {a: SeverityScorer(cache[a]["edge_index"], cache[a]["y"] > 0.5,
                            len(cache[a]["wall"]), DEFAULT) for a in pool}
    Dt = {a: to_torch_sparse(dilation_operator(cache[a]["edge_index"],
                                               len(cache[a]["wall"]), 2), dev) for a in pool}

    def wall_of(S):
        return S["wall"]

    def off_of(S):
        return ~S["wall"]

    # --- the allowed off-wall region under each constraint --------------------------
    def region(a, wall_set, use_owner, use_shell):
        S = cache[a]
        d = ~S["wall"]
        if use_owner:
            d = d & wall_set[S["owner"]]
        if use_shell:
            d = d & S["shell"].astype(bool)
        return d

    print("[i] building expected-score curves per region ...", flush=True)
    ARMS = [("base", 0, 0), ("owner", 1, 0), ("shell", 0, 1), ("owner+shell", 1, 1)]
    rows = {n: {} for n, _, _ in ARMS}
    rows["nested_pick"] = {}
    per_vessel_pick = {}

    for k, held in sorted(folds.items()):
        sel = [a for a in pool if a not in held]
        # the WALL set, exactly as the shipped readout builds it
        sub = {a: sc[a] for a in sel}
        th_w = FAMILIES["resid"][0](cache, vs, sel, sub, GRID)
        b_w, med_w = tune_adapt(cache, vs, sel, sub, "resid", th_w, wall_of)
        wset = {a: apply_adapt(cache[a], sc[a], "resid", th_w, wall_of, b_w, med_w)
                & cache[a]["wall"] for a in pool}

        cands = {}
        for name, uo, us in ARMS:
            reg_ = {a: region(a, wset[a], uo, us) for a in pool}
            curves = {(a, g): expected_curve(sc[a], reg_[a], Dt[a], dev, g)
                      for a in pool for g in GAMMA}

            def raw_k(a, g, _c=curves):
                ks, vals = _c[(a, g)]
                if len(ks) < 2:
                    return None, None
                return int(ks[int(np.argmax(vals))]), int(ks[-1])

            def mass(a, _r=reg_):
                d = _r[a]
                return float(sc[a][d].sum()) if d.any() else 0.0

            def mask_for(a, g, kscl, beta, alpha, kmed, mmed, _r=reg_,
                         _rk=raw_k, _ms=mass):
                k0, kmax = _rk(a, g)
                if k0 is None:
                    return np.zeros(len(sc[a]), bool)
                # ANTI-COMPRESSION.  The measured failure is that the chosen budget is
                # compressed toward the cohort middle: `patient032` commits 63 nodes for 120
                # GT (recall 0.417) while `patient005` commits 17 for 4 (precision 0.267).
                # `beta > 1` expands the spread about the cohort median budget; `alpha`
                # leans the same way on the off-wall confidence mass.  beta=1, alpha=0
                # reproduces the expected-score readout EXACTLY, so this can only move if it
                # pays -- the same "perturb, do not replace" shape as the adaptive threshold.
                kk = kmed * (max(k0, 1) / max(kmed, 1e-9)) ** beta
                if alpha:
                    kk *= (max(_ms(a), 1e-9) / max(mmed, 1e-9)) ** alpha
                kk = int(np.clip(round(kk * kscl), 1, kmax))
                d = _r[a]
                order = np.flatnonzero(d)[np.argsort(-sc[a][d])]
                m = np.zeros(len(sc[a]), bool)
                m[order[:kk]] = True
                return m

            best = None
            for g in GAMMA:
                ks_sel = [raw_k(a, g)[0] for a in sel]
                ks_sel = [x for x in ks_sel if x]
                kmed = float(np.median(ks_sel)) if ks_sel else 1.0
                mmed = float(np.median([mass(a) for a in sel])) or 1.0
                for kscl in KSCALE:
                    for beta in BETA:
                        for alpha in ALPHA:
                            v = [x for x in (vs[a].score(
                                mask_for(a, g, kscl, beta, alpha, kmed, mmed),
                                off_of(cache[a])) for a in sel) if x == x]
                            q = float(np.mean(v)) if v else -1e9
                            if best is None or q > best[0]:
                                best = (q, g, kscl, beta, alpha, kmed, mmed)
            cands[name] = (best[0],
                           lambda a, _m=mask_for, _b=best: _m(a, _b[1], _b[2], _b[3], _b[4],
                                                              _b[5], _b[6]))

        pick = max(cands, key=lambda n: cands[n][0])
        per_vessel_pick[k] = pick
        for a in held:
            d = off_of(cache[a])
            for name, _, _ in ARMS:
                rows[name][a] = vs[a].score(cands[name][1](a), d)
            rows["nested_pick"][a] = vs[a].score(cands[pick][1](a), d)
        print("  fold %d  pick=%-11s  sel %s" % (
            k, pick, "  ".join("%s %.3f" % (n, cands[n][0]) for n, _, _ in ARMS)),
            flush=True)

    prio = [a for a in pool if is_priority(classes.get(a, ""))]
    print("\nFINAL-TIME OFF-WALL, strictly nested (tags=%s)\n" % args.tags)
    print("%-14s | %9s %9s %9s" % ("arm", "ALL", "baseline", "PRIORITY"))
    for name in [n for n, _, _ in ARMS] + ["nested_pick"]:
        R = rows[name]
        base = [a for a in pool if a not in prio]
        print("%-14s | %9.4f %9.4f %9.4f"
              % (name, np.nanmean([R[a] for a in pool]),
                 np.nanmean([R[a] for a in base]), np.nanmean([R[a] for a in prio])))
    print("\nper vessel: base -> nested_pick")
    for a in sorted(pool):
        if rows["base"][a] != rows["base"][a]:
            continue
        print("   %-11s %.4f -> %.4f" % (a, rows["base"][a], rows["nested_pick"][a]))
    if args.save:
        Path(args.save).write_text(json.dumps(rows, indent=2, default=float))
        print("\nwrote %s" % args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
