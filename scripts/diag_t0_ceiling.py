"""What is the CEILING for a model that sees only t=0 flow, geometry, ICs and BCs?

Phase 3 (as of 2026-08-08) is scoped to a temporary bandaid: assume the GT flow field is
available **at t=0 only**, plus geometry / ICs / BCs, and ask whether a wall-only clot model can
generalize to unseen vessels at `deploy_clot_score > 0.6`. Phase 5 replaces the bandaid with the
deployable ML kinematic model.

Before building that model it is worth knowing what the information content of t=0 actually
supports. This measures the **oracle-thresholded ranking ceiling**: for each vessel, rank
wall-band nodes by the best deploy-legal t=0 feature (and by a simple leave-one-vessel-out
logistic combination of them), sweep the threshold, and take the best achievable F1.

That is an upper bound for any t=0-only *ranking* model, and it is deliberately generous:
  * the threshold is chosen per vessel with knowledge of the labels (oracle);
  * no rollout error, no commit dynamics, no calibration loss.

It is NOT an upper bound on the full model, because the autocatalytic rollout can sharpen a weak
spatial prior into a committed set, and `deploy_clot_score` is relaxed precision with a 2-hop
dilation and a recall floor, which is more forgiving than raw F1. Read it as "what ranking
quality is available at t=0", not as a verdict.

Usage:
    python scripts/diag_t0_ceiling.py
    python scripts/diag_t0_ceiling.py --anchors patient041,patient043
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
# Leaked CFD channels (16.1c) are not deploy-legal even under the t=0-GT-flow bandaid: they are
# the CONVERGED clot-affected solution, not the t=0 field.
# 16.1c: the converged clot-affected CFD solution. These appear BOTH bare and with a
# `kine_x_`/`bio_x_` prefix, and an earlier version of this script only excluded the bare names --
# so the leak won on 3 of 35 vessels and inflated the ceiling. Match on substring instead.
ILLEGAL_SUBSTR = ("u_prior", "v_prior", "mu_prior", "speed_mismatch", "wss_prior")
LEGAL_GROUPS = {"geometry", "topology", "kine_x", "bio_x", "flow", "flow_derived", "shear_grad"}


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


def best_f1(score: torch.Tensor, label: torch.Tensor) -> float:
    """Oracle-threshold F1: best achievable by any threshold on this ranking."""
    order = score.argsort(descending=True)
    lab = label[order].double()
    tp = torch.cumsum(lab, 0)
    k = torch.arange(1, lab.numel() + 1, dtype=torch.float64)
    n_pos = float(lab.sum())
    if n_pos <= 0:
        return float("nan")
    f1 = 2 * tp / (k + n_pos)
    return float(f1.max())


def vessel_rows(path: Path, device, phys, bio):
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
    if int(commits.sum()) == 0 or int((band & ~commits).sum()) == 0:
        return None
    feats = build_feature_table_at_time(data, 0, device=device, phys_cfg=phys, bio_cfg=bio)
    cols: dict[str, torch.Tensor] = {}
    for k, (v, g, _s) in sorted(feats.items()):
        if g not in LEGAL_GROUPS or any(b in k for b in ILLEGAL_SUBSTR):
            continue
        v = v.reshape(-1)
        if v.numel() != int(data.num_nodes) or not bool(torch.isfinite(v).all()):
            continue
        if float(v[band].std()) <= 0:
            continue
        cols[k] = v
    return {"anchor": path.stem, "band": band, "y": commits, "cols": cols}


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

    V = []
    for p in paths:
        if not p.exists():
            continue
        try:
            r = vessel_rows(p, device, phys, bio)
        except Exception as e:
            print(f"  skip {p.stem}: {type(e).__name__}: {e}")
            continue
        if r:
            V.append(r)
    if not V:
        print("[ERR] nothing usable")
        return 1

    shared = set(V[0]["cols"])
    for r in V[1:]:
        shared &= set(r["cols"])
    shared = sorted(shared)

    # --- per-vessel: best SINGLE feature, oracle threshold -----------------------------------
    print(f"n={len(V)} vessels, {len(shared)} deploy-legal t=0 features in common\n")
    print(f"  {'vessel':>12} {'base':>6} {'bestAUC':>8} {'oracle F1':>10}  {'best feature':<24}")
    rows = []
    for r in V:
        b, y = r["band"], r["y"]
        best = (0.5, None, 0.0)
        for k in shared:
            v = r["cols"][k][b]
            a = auc(v[y[b]], v[~y[b]])
            a_dir = max(a, 1 - a)
            if a_dir > best[0]:
                sc = v if a >= 0.5 else -v
                best = (a_dir, k, best_f1(sc, y[b]))
        rows.append({"anchor": r["anchor"], "base_rate": float(y.sum()) / float(b.sum()),
                     "best_auc": best[0], "best_f1": best[2], "best_feature": best[1]})
        print(f"  {r['anchor']:>12} {rows[-1]['base_rate']:>6.1%} {best[0]:>8.3f} "
              f"{best[2]:>10.3f}  {str(best[1]):<24}")

    def m(k):
        v = [x[k] for x in rows if x[k] == x[k]]
        return sum(v) / len(v)

    print(f"\n  mean best-single-feature AUC : {m('best_auc'):.3f}")
    print(f"  mean ORACLE-THRESHOLD F1     : {m('best_f1'):.3f}")
    print(f"  vessels with oracle F1 >= 0.6: {sum(1 for x in rows if x['best_f1'] >= 0.6)} / {len(rows)}")
    print(f"  vessels with oracle F1 >= 0.5: {sum(1 for x in rows if x['best_f1'] >= 0.5)} / {len(rows)}")
    print("\n  This is an OPTIMISTIC bound for a t=0 ranking model: the threshold is oracle-chosen")
    print("  per vessel, and there is no rollout, calibration or commit-dynamics loss.")
    print("  It is NOT the full-model ceiling -- the autocatalytic rollout can sharpen a weak")
    print("  prior, and deploy_clot_score relaxes with a 2-hop dilation and a recall floor.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
