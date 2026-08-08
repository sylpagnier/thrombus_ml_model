"""Does the COMSOL gate structure actually discriminate clot on OUR data?

This is the empirical check the physics-mirroring Phase 3 plan rests on
(docs/PHASE3_HANDOFF.md 1.4), and there is a real tension to resolve:

  * 10.1 says the low-shear gate `[sr < lss]`, lss = 25 1/s, fires for **79.7%** of growing
    nodes and is the dominant deposition mechanism.
  * 16.3 measured flow proxies (`-anaSpd`) at only **0.768** AUC for final Mat, and Z1 put the
    whole flow channel's *marginal* contribution at **0.041** AUC over a zero-prior.

If the gate does not separate committing from non-committing wall-band nodes on our packs, then
building a model around the law buys nothing and the plan needs rethinking BEFORE any code.

Measured per vessel, on wall-band nodes:
  * fraction of committing nodes with `gamma_si < lss`   -- 10.1's 79.7% claim, on our data
  * AUC of `-gamma_si` for "ever commits"                -- does low shear rank clot at all
  * AUC of `is_low_shear` (the soft gate as used)        -- the gate as the kernel applies it
  * AUC of `neg_dgamma_dx` (separation gate input)       -- the 21% minority mechanism
  * base rate, for context: a high AUC on a 2% base rate is a different claim than on 40%

Usage:
    python scripts/diag_physics_gate_support.py
    python scripts/diag_physics_gate_support.py --anchors patient041,patient020
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


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    order = allv.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, allv.numel() + 1, dtype=torch.float64)
    uniq, inv, cnt = torch.unique(allv, return_inverse=True, return_counts=True)
    if int(cnt.max()) > 1:
        sums = torch.zeros(uniq.numel(), dtype=torch.float64).scatter_add_(0, inv, ranks)
        ranks = (sums / cnt.double())[inv]
    n1 = pos.numel()
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * neg.numel()))


def one(path: Path, device, phys, bio, lss: float) -> dict | None:
    data = torch.load(path, map_location="cpu", weights_only=False)
    band = resolve_ceiling_mask(data, device, bio, ceiling_hops=CEILING_HOPS).reshape(-1).bool()
    if not bool(band.any()):
        return None
    names = data.y_channel_names.split(",")
    series = data.y[:, :, names.index("Mat_log1p_nd")].to(device=device)
    thr = float(series.max()) * MAT_REL_FRAC
    if thr <= 0:
        return None
    commits = (series > thr).any(dim=0) & band
    if int(commits.sum()) == 0:
        return None
    neg = band & ~commits

    f = build_feature_table_at_time(data, 0, device=device, phys_cfg=phys, bio_cfg=bio)

    def col(k):
        v = f.get(k)
        return None if v is None else v[0].reshape(-1)

    g = col("gamma_si")
    ils = col("is_low_shear")
    ndg = col("neg_dgamma_dx")
    if g is None:
        return None

    return {
        "anchor": path.stem,
        "n_band": int(band.sum()),
        "n_commit": int(commits.sum()),
        "base_rate": float(commits.sum()) / float(band.sum()),
        # 10.1's claim, on our data: do committing nodes sit below the low-shear threshold?
        "frac_commit_below_lss": float((g[commits] < lss).float().mean()),
        "frac_noncommit_below_lss": float((g[neg] < lss).float().mean()),
        "auc_neg_gamma": auc(-g[commits], -g[neg]),
        "auc_is_low_shear": auc(ils[commits], ils[neg]) if ils is not None else float("nan"),
        "auc_neg_dgamma_dx": auc(ndg[commits], ndg[neg]) if ndg is not None else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = torch.device("cpu")
    phys, bio = PhysicsConfig(phase="biochem"), BiochemConfig(phase="biochem")
    lss = float(bio.lss)

    if args.anchors.strip():
        paths = [Path(f"data/processed/graphs_biochem_anchors/{a.strip()}.pt")
                 for a in args.anchors.split(",") if a.strip()]
    else:
        paths = [Path(p) for p in sorted(glob.glob("data/processed/graphs_biochem_anchors/patient*.pt"))
                 if "mirror" not in p]

    print(f"lss = {lss} 1/s   band = wall + {CEILING_HOPS} hops   commit = Mat > 10% of vessel peak\n")
    print(f"  {'vessel':>12} {'band':>6} {'commit':>7} {'base':>6} "
          f"{'%commit<lss':>12} {'%non<lss':>9} {'AUC -sr':>8} {'AUC gate':>9} {'AUC dsr':>8}")
    rows = []
    for p in paths:
        if not p.exists():
            continue
        try:
            r = one(p, device, phys, bio, lss)
        except Exception as e:
            print(f"  {p.stem:>12}  skip: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['anchor']:>12} {r['n_band']:>6} {r['n_commit']:>7} {r['base_rate']:>6.1%} "
              f"{r['frac_commit_below_lss']:>12.1%} {r['frac_noncommit_below_lss']:>9.1%} "
              f"{r['auc_neg_gamma']:>8.3f} {r['auc_is_low_shear']:>9.3f} {r['auc_neg_dgamma_dx']:>8.3f}")

    if not rows:
        print("[ERR] no vessel produced a result")
        return 1

    def m(k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return sum(v) / len(v)

    n = len(rows)
    print(f"\n  n={n} vessels")
    print(f"  10.1 claims 79.7% of GROWING nodes are below lss. Measured here: "
          f"{m('frac_commit_below_lss'):.1%}")
    print(f"  ... but non-committing band nodes below lss:        {m('frac_noncommit_below_lss'):.1%}")
    print(f"      (if these two are close, the gate fires nearly everywhere and separates nothing)")
    print(f"\n  mean AUC  -gamma_si (low shear ranks clot) : {m('auc_neg_gamma'):.3f}")
    print(f"  mean AUC  is_low_shear (soft gate)         : {m('auc_is_low_shear'):.3f}")
    print(f"  mean AUC  neg_dgamma_dx (separation gate)  : {m('auc_neg_dgamma_dx'):.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
