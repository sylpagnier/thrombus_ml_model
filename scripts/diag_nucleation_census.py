"""Growth-vs-nucleation census across the WHOLE vessel inventory.

C1's premise (WALL_MODEL_PLAN.md 14.6) is that 27-58% of clot commits are *nucleation* -- the
node had no committed neighbour when it turned on -- and that the current multiplicative
architecture (`spatial_gate * magnitude * autocat`) structurally cannot express them.

That number comes from **six vessels** (039-044). This project has already been burned once for
generalising from exactly those six: 16.3 found that the 6-vessel feature rules do not hold
cohort-wide and that `mu_eff` -- "diagnostic" on the small cohort -- is anti-predictive on 19 of
35. Before building an architecture on the 14.6 number, it is worth the CPU to check it on all
of them.

Definitions follow the deploy metric, not the training labels:
  * committed at t  = `gt_growth_commit_mask_at_time` (relu(mu_eff(t) - mu_eff(0)) >= thresh)
  * band           = `resolve_ceiling_mask` (wall + CLOT_PHI_CEILING_HOPS), wall clot only
  * commit time    = first t at which a band node is committed
  * nucleation     = at its commit time, NO graph neighbour was already committed
  * growth         = at least one neighbour was

Also reports the seed-reachability that 14.6 measured at 43-81%: what fraction of the final
committed set lies within k hops of the t=20 committed set. That is the ceiling on how much of
the final map growth-from-seeds can reach at all.

Usage:
    python scripts/diag_nucleation_census.py                       # all packs
    python scripts/diag_nucleation_census.py --anchors patient039,patient043
    python scripts/diag_nucleation_census.py --seed-t 20 --hops 3 --out outputs/nuc.json
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
CEILING_HOPS: int | None = None
GROWTH_HOPS = 1
MAT_REL_FRAC = 0.10  # 21.2: rel_max at 10% of each vessel's peak Mat

from src.core_physics.clot_growth_masks import (  # noqa: E402
    graph_dilate_hops,
    gt_growth_commit_mask_at_time,
    resolve_ceiling_mask,
)


def commit_times(data, band: torch.Tensor, device: torch.device,
                 phys: PhysicsConfig, t_max: int, label: str) -> torch.Tensor:
    """First timestep at which each band node is committed; -1 if never.

    `label` selects the GT definition, and it matters enormously -- 14.6 predates both the
    canonical metric (20.1) and the per-vessel rel_max labels (21.2), so its numbers are not
    reproducible without knowing which it used:
      mu  -- relu(mu_eff(t) - mu_eff(0)) >= thresh. The DEPLOY METRIC's GT.
      mat -- Mat_log1p_nd > frac * (vessel peak Mat). The TRAINING label (21.2).
    """
    n = int(data.num_nodes)
    first = torch.full((n,), -1, dtype=torch.long, device=device)
    if label == "mat":
        names = data.y_channel_names.split(",")
        col = names.index("Mat_log1p_nd")
        series = data.y[:, :, col].to(device=device)
        thr = float(series.max()) * MAT_REL_FRAC
        if thr <= 0:
            return first
        for t in range(t_max):
            m = (series[t] > thr) & band
            newly = m & (first < 0)
            if bool(newly.any()):
                first[newly] = t
        return first
    for t in range(t_max):
        m = gt_growth_commit_mask_at_time(data, t, phys, device).reshape(-1).bool() & band
        newly = m & (first < 0)
        if bool(newly.any()):
            first[newly] = t
    return first


def census_one(path: Path, device: torch.device, phys: PhysicsConfig,
               bio: BiochemConfig, seed_t: int, hops: int, label: str) -> dict | None:
    data = torch.load(path, map_location="cpu", weights_only=False)
    t_max = int(data.y.shape[0])
    band = resolve_ceiling_mask(data, device, bio, ceiling_hops=CEILING_HOPS).reshape(-1).bool()
    if not bool(band.any()):
        return None
    ei = data.edge_index.to(device=device)

    first = commit_times(data, band, device, phys, t_max, label)
    committed = first >= 0
    n_commit = int(committed.sum())
    if n_commit == 0:
        return {"anchor": path.stem, "commits": 0}

    # A commit at time t is GROWTH if any neighbour committed STRICTLY EARLIER. Using "earlier"
    # rather than "same step or earlier" keeps simultaneous commits from being credited to each
    # other, which would inflate growth and understate the term C1 exists to add.
    src, dst = ei[0], ei[1]
    ft = first.clone()
    ft[~committed] = torch.iinfo(torch.long).max
    # min over neighbours of each node's commit time, propagated GROWTH_HOPS times so that
    # "had committed material nearby" tolerates more than one hop. s26.13.2: with a strict
    # 1-hop rule, EVERY late-quartile "nucleation" site in six vessels sat exactly 2 hops from
    # existing clot -- i.e. it was growth, misclassified. 1 reproduces the original rule.
    nbr_min = ft.clone()
    for _ in range(max(GROWTH_HOPS, 1)):
        prop = torch.full_like(ft, torch.iinfo(torch.long).max)
        prop = prop.scatter_reduce(0, dst, nbr_min[src], reduce="amin", include_self=True)
        nbr_min = torch.minimum(nbr_min, prop)
    is_growth = committed & (nbr_min < first)
    is_nucleation = committed & ~is_growth

    n_nuc = int(is_nucleation.sum())
    # time-quartile profile of nucleation, as 14.6 reported
    q = []
    ct = first[is_nucleation]
    for i in range(4):
        lo, hi = i * t_max // 4, (i + 1) * t_max // 4
        q.append(int(((ct >= lo) & (ct < hi)).sum()))

    # seed reachability: how much of the FINAL committed set is within `hops` of the t=seed_t set
    seeds = (first >= 0) & (first <= seed_t)
    reach = graph_dilate_hops(seeds, ei, hops).reshape(-1).bool() & band
    final = committed
    frac_reachable = float((final & reach).sum()) / max(int(final.sum()), 1)

    return {
        "anchor": path.stem,
        "commits": n_commit,
        "growth": n_commit - n_nuc,
        "nucleation": n_nuc,
        "nucleation_pct": 100.0 * n_nuc / n_commit,
        "nucleation_by_quartile": q,
        "seed_count": int(seeds.sum()),
        "final_within_%dhops_of_seeds_pct" % hops: 100.0 * frac_reachable,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="", help="comma-separated; default = every pack")
    ap.add_argument("--seed-t", type=int, default=20)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--label", choices=("mu", "mat"), default="mu",
                    help="mu = deploy-metric GT; mat = rel_max training label")
    ap.add_argument("--ceiling-hops", type=int, default=None)
    ap.add_argument("--growth-hops", type=int, default=1,
                    help="a commit counts as GROWTH if committed material is within "
                         "this many hops. 1 = the original (too strict) rule.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    global CEILING_HOPS, GROWTH_HOPS
    CEILING_HOPS = args.ceiling_hops
    GROWTH_HOPS = args.growth_hops
    device = torch.device("cpu")
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")

    if args.anchors.strip():
        paths = [Path(f"data/processed/graphs_biochem_anchors/{a.strip()}.pt")
                 for a in args.anchors.split(",") if a.strip()]
    else:
        paths = [Path(p) for p in sorted(glob.glob("data/processed/graphs_biochem_anchors/patient*.pt"))]

    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"[skip] {p.name} missing")
            continue
        try:
            r = census_one(p, device, phys, bio, args.seed_t, args.hops, args.label)
        except Exception as e:
            print(f"[skip] {p.stem}: {e}")
            continue
        if r is None or r.get("commits", 0) == 0:
            print(f"[skip] {p.stem}: no committed nodes in band")
            continue
        rows.append(r)
        key = f"final_within_{args.hops}hops_of_seeds_pct"
        print(f"  {r['anchor']:>12} commits={r['commits']:>5} nucleation={r['nucleation_pct']:>5.1f}% "
              f"q={r['nucleation_by_quartile']} seed_reach={r[key]:>5.1f}%")

    if not rows:
        print("[ERR] no vessels produced a census")
        return 1

    nuc = [r["nucleation_pct"] for r in rows]
    key = f"final_within_{args.hops}hops_of_seeds_pct"
    reach = [r[key] for r in rows]
    n = len(nuc)
    mean = sum(nuc) / n
    sd = (sum((x - mean) ** 2 for x in nuc) / max(n - 1, 1)) ** 0.5
    print(f"\n  n={n} vessels")
    print(f"  nucleation %:   mean {mean:5.1f}  sd {sd:5.1f}  range {min(nuc):5.1f} .. {max(nuc):5.1f}")
    rm = sum(reach) / n
    print(f"  seed reach %:   mean {rm:5.1f}  range {min(reach):5.1f} .. {max(reach):5.1f}")
    # 14.6's six-vessel claim, restated for comparison
    print("\n  14.6 measured 27-58% nucleation and 43-81% seed reach on SIX vessels (039-044).")
    six = [r for r in rows if r["anchor"][-3:] in {"039", "040", "041", "042", "043", "044"}]
    if six:
        s = [r["nucleation_pct"] for r in six]
        print(f"  those six here: mean {sum(s)/len(s):5.1f}  range {min(s):5.1f} .. {max(s):5.1f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
