#!/usr/bin/env python
"""Grade a training leg on the two things WALL_MODEL_PLAN.md s12.3/s12.5 says matter.

1. OBJECTIVE ALIGNMENT -- does decreasing loss track increasing deploy score?
   Reported three ways, because s12.3 showed the naive one is not trustworthy at n=6:
     * Spearman(loss, deploy_clot_score) -- the pre-registered statistic, with an EXACT
       permutation p-value and a leave-one-out jackknife (v6 read -0.406 with p=0.217 and
       flipped to +0.05 when a single epoch was dropped).
     * z-separation -- deploy score in these legs is bimodal, so the question that actually
       resolves is "can the loss tell the good epoch from the bad ones?". |z| < 0.5 means no.
     * effective resolution -- how many distinct deploy states the epochs actually visited.
       A leg whose score is constant to 1e-3 cannot support ANY correlation claim.

2. EXCURSION DEPTH (s12.5 item 2) -- the gate-independent readout that replaces loss as the
   way to rank legs: min deploy_clot_fp / min mass / how many epochs left the saturated basin.

Usage:
    python scripts/diag_leg_alignment.py --legs v3,v6,v7,v8,v9
    python scripts/diag_leg_alignment.py --logs path/to/train_log.jsonl,other/train_log.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

SUBCOHORT_ROOT = Path("outputs/biochem/eda/wall_gen_stenosis_subcohort")
EXTRA_ROOTS = [Path("outputs/biochem/eda/autonomous_6h")]
SATURATED_FP = 292  # the basin every sub-cohort leg falls into (s12.4)


def _spearman(a: list[float], b: list[float]) -> float:
    def rank(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def _resolve_log(name: str) -> Path | None:
    if name.endswith(".jsonl"):
        p = Path(name)
        return p if p.exists() else None
    leg = name if name.startswith("WG_") else f"WG_stenosis_subcohort_ft_{name}"
    leg = leg.replace("_v1", "") if leg.endswith("_v1") else leg
    for root in [SUBCOHORT_ROOT, *EXTRA_ROOTS]:
        p = root / leg / "train_log.jsonl"
        if p.exists():
            return p
    return None


def analyse(name: str, rows: list[dict]) -> dict:
    loss = [float(r["loss"]) for r in rows]
    sc = [float(r.get("deploy_clot_score", 0.0)) for r in rows]
    f1 = [float(r.get("deploy_clot_f1", 0.0)) for r in rows]
    fp = [int(r.get("deploy_clot_fp", 0)) for r in rows]
    mass = [float(r.get("deploy_clot_mass_ratio", 0.0)) for r in rows]
    n = len(rows)
    out: dict = {"leg": name, "n": n}

    if n >= 3:
        rho = _spearman(loss, sc)
        out["spearman"] = rho
        # exact permutation p-value (one-sided: how often is chance at least this negative)
        if n <= 8:
            perms = list(itertools.permutations(range(n)))
            hits = sum(1 for p in perms if _spearman(loss, [sc[i] for i in p]) <= rho + 1e-12)
            out["perm_p"] = hits / len(perms)
        # jackknife: does one epoch carry the sign?
        jk = []
        for i in range(n):
            jk.append(_spearman([v for j, v in enumerate(loss) if j != i],
                                [v for j, v in enumerate(sc) if j != i]))
        out["jk_min"], out["jk_max"] = min(jk), max(jk)
        out["sign_stable"] = all((x < 0) == (rho < 0) for x in jk)

    # z-separation of the best epoch's loss against the rest
    if n >= 3:
        bi = max(range(n), key=lambda i: sc[i])
        rest = [loss[i] for i in range(n) if i != bi]
        m, s = st.mean(rest), st.pstdev(rest)
        out["best_ep"] = int(rows[bi].get("epoch", bi + 1))
        out["best_score"] = sc[bi]
        out["best_f1"] = f1[bi]
        out["bad_score"] = st.mean([sc[i] for i in range(n) if i != bi])
        out["z"] = (loss[bi] - m) / s if s > 0 else float("nan")

    # effective resolution: distinct deploy states actually visited
    out["distinct_fp"] = len(set(fp))
    out["score_spread"] = max(sc) - min(sc)
    out["loss_spread_pct"] = 100.0 * (max(loss) - min(loss)) / max(abs(max(loss)), 1e-9)
    # excursion depth (s12.5 item 2)
    out["min_fp"] = min(fp)
    out["min_mass"] = min(mass)
    out["n_excursion"] = sum(1 for v in fp if v < SATURATED_FP)
    out["max_score"] = max(sc)
    out["max_f1"] = max(f1)
    out["salvage"] = rows[-1].get("salvage_best_score")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legs", default="", help="comma list, e.g. v3,v6,v7,v8,v9")
    ap.add_argument("--logs", default="", help="comma list of explicit train_log.jsonl paths")
    ap.add_argument("--per-epoch", action="store_true", help="also dump each epoch")
    args = ap.parse_args()

    names = [s.strip() for s in (args.legs or args.logs).split(",") if s.strip()]
    if not names:
        ap.error("pass --legs or --logs")

    results = []
    for nm in names:
        p = _resolve_log(nm)
        if p is None:
            print(f"[skip] {nm}: no train_log.jsonl found")
            continue
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        if not rows:
            print(f"[skip] {nm}: empty log")
            continue
        results.append(analyse(nm, rows))
        if args.per_epoch:
            print(f"\n=== {nm} ({p}) ===")
            print(f"{'ep':>3} {'loss':>11} {'unr':>4} {'score':>8} {'f1':>8} {'mass':>7} {'fp':>5} {'select':>26}")
            for r in rows:
                print(f"{r.get('epoch',0):>3} {r['loss']:>11.4f} {r.get('cur_unroll',-1):>4} "
                      f"{r.get('deploy_clot_score',0):>8.4f} {r.get('deploy_clot_f1',0):>8.4f} "
                      f"{r.get('deploy_clot_mass_ratio',0):>7.3f} {int(r.get('deploy_clot_fp',0)):>5} "
                      f"{str(r.get('select_mode','')):>26}")

    print("\n" + "=" * 118)
    print("1. OBJECTIVE ALIGNMENT  (target: rho < 0 AND z < -0.5 AND distinct_fp > 2)")
    print("=" * 118)
    print(f"{'leg':<10} {'n':>3} {'rho':>7} {'perm_p':>7} {'jk range':>17} {'stable':>7} "
          f"{'z':>7} {'best sc':>8} {'bad sc':>7} {'#fp states':>11} {'loss spr':>9}")
    for r in results:
        jk = f"[{r.get('jk_min', float('nan')):+.2f},{r.get('jk_max', float('nan')):+.2f}]"
        print(f"{r['leg']:<10} {r['n']:>3} {r.get('spearman', float('nan')):>+7.3f} "
              f"{r.get('perm_p', float('nan')):>7.3f} {jk:>17} {str(r.get('sign_stable', '')):>7} "
              f"{r.get('z', float('nan')):>+7.2f} {r.get('best_score', 0):>8.4f} "
              f"{r.get('bad_score', 0):>7.4f} {r['distinct_fp']:>11} {r['loss_spread_pct']:>8.2f}%")

    print("\n" + "=" * 118)
    print("2. EXCURSION DEPTH  (s12.5 item 2 -- the gate-independent way to rank legs)")
    print("=" * 118)
    print(f"{'leg':<10} {'min fp':>7} {'min mass':>9} {'#excursions':>12} {'max score':>10} "
          f"{'max f1':>8} {'salvage':>9}")
    for r in results:
        sv = r.get("salvage")
        print(f"{r['leg']:<10} {r['min_fp']:>7} {r['min_mass']:>9.3f} {r['n_excursion']:>12} "
              f"{r['max_score']:>10.4f} {r['max_f1']:>8.4f} "
              f"{(f'{sv:.4f}' if isinstance(sv, (int, float)) else '-'):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
