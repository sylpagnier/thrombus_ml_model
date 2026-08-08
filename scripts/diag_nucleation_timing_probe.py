"""Are EARLY and LATE nucleation sites the same kind of place?

This decides C1's nucleation-head target, and it is the cheapest question on the Phase 3
critical path (WALL_MODEL_PLAN.md 26.12.1, PHASE3_HANDOFF.md 1.4a).

20.4 settled the head target as the `t=20` seeds because seeds are +0.097 AUC more predictable
and 2.7x more consistent across vessels than the final map. That reasoning holds **only if early
and late nucleation happen in the same kinds of places** -- then a head trained on the cleanest
examples generalises to all of them. The census (26.12) found that 7 of 35 vessels nucleate
predominantly LATE, so if the two populations differ, a seed-trained head silently misses a
fifth of the inventory.

Two measurements per vessel, over the t=0 deploy-legal field features:

  DISCRIMINABILITY  AUC(feature; Q1 sites vs Q4 sites)
                    ~0.50 => indistinguishable => same kind of place => train on seeds.
                    far from 0.50 => they separate => the head needs time conditioning.

  TRANSFER          AUC(feature; Q1 sites vs band negatives)  and  (Q4 vs band negatives)
                    If a feature ranks Q1 sites well but Q4 sites near chance, a head fitted on
                    Q1 will not rank Q4. This is the practical consequence of the above.

Usage:
    python scripts/diag_nucleation_timing_probe.py
    python scripts/diag_nucleation_timing_probe.py --anchors patient001,patient032 --out probe.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_growth_masks import resolve_ceiling_mask  # noqa: E402
from src.core_physics.clot_t0_extended_probe import build_feature_table_at_time  # noqa: E402

MAT_REL_FRAC = 0.10
CEILING_HOPS = 3
N_PERM = 24
# The leaked CFD channels (16.1c) and anything read from `y` are not deploy-legal.
ILLEGAL = {"u_prior", "v_prior", "mu_prior", "speed_mismatch_nd"}
LEGAL_GROUPS = {"geometry", "topology", "kine_x", "bio_x"}


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUC. 0.5 = indistinguishable."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    order = allv.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, allv.numel() + 1, dtype=torch.float64)
    # average ranks for ties
    uniq, inv, cnt = torch.unique(allv, return_inverse=True, return_counts=True)
    if int(cnt.max()) > 1:
        sums = torch.zeros(uniq.numel(), dtype=torch.float64).scatter_add_(0, inv, ranks)
        ranks = (sums / cnt.double())[inv]
    n1 = pos.numel()
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n1 * neg.numel()))


def commit_times(data, band, device, t_max):
    n = int(data.num_nodes)
    first = torch.full((n,), -1, dtype=torch.long, device=device)
    names = data.y_channel_names.split(",")
    series = data.y[:, :, names.index("Mat_log1p_nd")].to(device=device)
    thr = float(series.max()) * MAT_REL_FRAC
    if thr <= 0:
        return first
    for t in range(t_max):
        newly = (series[t] > thr) & band & (first < 0)
        if bool(newly.any()):
            first[newly] = t
    return first


def probe_one(path, device, phys, bio):
    data = torch.load(path, map_location="cpu", weights_only=False)
    t_max = int(data.y.shape[0])
    band = resolve_ceiling_mask(data, device, bio, ceiling_hops=CEILING_HOPS).reshape(-1).bool()
    if not bool(band.any()):
        return None
    first = commit_times(data, band, device, t_max)
    committed = first >= 0
    if int(committed.sum()) == 0:
        return None

    ei = data.edge_index.to(device=device)
    ft = first.clone()
    ft[~committed] = torch.iinfo(torch.long).max
    nbr_min = torch.full_like(ft, torch.iinfo(torch.long).max)
    nbr_min = nbr_min.scatter_reduce(0, ei[1], ft[ei[0]], reduce="amin", include_self=True)
    nucleation = committed & ~(nbr_min < first)

    q1_hi, q4_lo = t_max // 4, 3 * t_max // 4
    q1 = nucleation & (first < q1_hi)
    q4 = nucleation & (first >= q4_lo)
    neg = band & ~committed
    if int(q1.sum()) < 5 or int(q4.sum()) < 5:
        return {"anchor": path.stem, "skipped": "needs >=5 nucleation sites in both Q1 and Q4",
                "q1": int(q1.sum()), "q4": int(q4.sum())}

    feats = build_feature_table_at_time(data, 0, device=device, phys_cfg=phys, bio_cfg=bio)
    rows = []
    # Permutation null: with these sample sizes and 80+ features, some separation appears by
    # chance. Re-label the SAME sites at random into two groups of the same sizes and measure
    # the same statistic, so the verdict is against chance rather than against 0.5.
    idx = torch.nonzero(q1 | q4).reshape(-1)
    n1 = int(q1.sum())
    g = torch.Generator().manual_seed(0)
    perms = [idx[torch.randperm(idx.numel(), generator=g)] for _ in range(N_PERM)]
    for k, (v, g_, _s) in sorted(feats.items()):
        if g_ not in LEGAL_GROUPS or k in ILLEGAL:
            continue
        v = v.reshape(-1)
        if v.numel() != int(data.num_nodes) or not bool(torch.isfinite(v).all()):
            continue
        if float(v.std()) <= 0:
            continue
        null = [abs(auc(v[pm[:n1]], v[pm[n1:]]) - 0.5) for pm in perms]
        rows.append({
            "feature": k,
            "auc_q1_vs_q4": auc(v[q1], v[q4]),
            "null_sep_mean": sum(null) / len(null),
            "auc_q1_vs_neg": auc(v[q1], v[neg]),
            "auc_q4_vs_neg": auc(v[q4], v[neg]),
        })
    return {"anchor": path.stem, "q1": int(q1.sum()), "q4": int(q4.sum()),
            "n_neg": int(neg.sum()), "features": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cpu")
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")

    if args.anchors.strip():
        paths = [Path(f"data/processed/graphs_biochem_anchors/{a.strip()}.pt")
                 for a in args.anchors.split(",") if a.strip()]
    else:
        paths = [Path(p) for p in sorted(glob.glob("data/processed/graphs_biochem_anchors/patient*.pt"))
                 if "mirror" not in p]

    out = []
    for p in paths:
        if not p.exists():
            continue
        try:
            r = probe_one(p, device, phys, bio)
        except Exception as e:
            print(f"[skip] {p.stem}: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        if "skipped" in r:
            print(f"[skip] {r['anchor']}: {r['skipped']} (q1={r['q1']}, q4={r['q4']})")
            continue
        out.append(r)
        best = max(r["features"], key=lambda f: abs(f["auc_q1_vs_q4"] - 0.5))
        print(f"  {r['anchor']:>12} q1={r['q1']:>4} q4={r['q4']:>4}  "
              f"most-separating: {best['feature']:<26} AUC(Q1|Q4)={best['auc_q1_vs_q4']:.3f}")

    if not out:
        print("[ERR] no vessel had >=5 nucleation sites in both Q1 and Q4")
        return 1

    # Aggregate: how far from 0.50 does the typical feature get?
    per_feat: dict[str, list[float]] = {}
    per_null: dict[str, list[float]] = {}
    trans: dict[str, list[tuple[float, float]]] = {}
    for r in out:
        for f in r["features"]:
            # mean of |AUC-0.5|, NOT |mean(AUC)-0.5|: a feature that separates in opposite
            # directions on different vessels still separates, and the latter cancels it to 0.
            per_feat.setdefault(f["feature"], []).append(abs(f["auc_q1_vs_q4"] - 0.5))
            per_null.setdefault(f["feature"], []).append(f["null_sep_mean"])
            trans.setdefault(f["feature"], []).append((f["auc_q1_vs_neg"], f["auc_q4_vs_neg"]))

    def _m(x):
        return sum(x) / len(x)

    def excess(k: str) -> float:
        """Separation minus what random re-labelling of the same sites produces."""
        return _m(per_feat[k]) - _m(per_null[k])

    q1n = sum(r["q1"] for r in out)
    q4n = sum(r["q4"] for r in out)
    print(f"\n  n={len(out)} vessels with >=5 nucleation sites in BOTH Q1 and Q4"
          f"   (Q1 sites={q1n}, Q4 sites={q4n})\n")
    print(f"  {'feature':<28} {'sep':>7} {'null':>7} {'excess':>8} {'AUC(Q1|neg)':>12} {'AUC(Q4|neg)':>12}")
    for k in sorted(per_feat, key=lambda kk: -excess(kk))[:14]:
        t = trans[k]
        a1 = _m([x for x, _ in t])
        a4 = _m([y for _, y in t])
        print(f"  {k:<28} {_m(per_feat[k]):>7.3f} {_m(per_null[k]):>7.3f} {excess(k):>+8.3f}"
              f" {a1:>12.3f} {a4:>12.3f}")

    print(f"\n  sep    = mean |AUC(Q1 vs Q4) - 0.5| across those vessels")
    print(f"  null   = the same statistic under {N_PERM} random re-labellings of the SAME sites")
    print(f"  excess = sep - null. THIS is the only column that means anything -- with 80+")
    print(f"           features and small site counts, raw `sep` is large by chance.")
    print(f"  The two AUC(*|neg) columns say HOW they differ: a feature that ranks Q1 sites")
    print(f"  above chance and Q4 sites below it points in opposite directions for the two.")

    worst = max(excess(k) for k in per_feat)
    print(f"\n  VERDICT: largest excess separation over chance = {worst:+.3f}")
    if worst < 0.10:
        print("  -> Q1 and Q4 nucleation sites are NOT distinguishable by any deploy-legal")
        print("     t=0 feature. Same kind of place. Train the head on seeds per 20.4.")
    else:
        print("  -> They separate. A seed-trained head will not rank late nucleation sites;")
        print("     the nucleation head needs a time input or a slow rate modulation.")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
