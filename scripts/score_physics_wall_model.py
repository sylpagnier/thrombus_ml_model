"""Score the Phase-3 physics wall model with the CANONICAL deploy metric.

Uses ``gt_clot_phi_at_time`` + ``compute_clot_relaxed_metrics`` + ``clot_score_from_deploy_dict``
exactly as ``grade_deploy_clot_series`` does (wall-masked, 3-hop relaxed, guiding blend),
so the number printed here is comparable to ``deploy_clot_score`` elsewhere in the project.

Cohort splits are honoured: the sealed set is scored but reported separately and is never
used to pick anything.

Usage:
    python scripts/score_physics_wall_model.py --mode gate
    python scripts/score_physics_wall_model.py --mode ode --da-sweep 50,100,150,200,400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_DEV,
    WALL_COHORT_V2_GENERALIZATION,
    WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import predict_phi, t0_flow_fields  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict,
    compute_clot_relaxed_metrics,
    metrics_to_deploy_prefix,
)

ANCHOR_DIR = Path("data/processed/graphs_biochem_anchors")


def score_one(data, phi_pred, phys, device):
    n_times = int(data.y.shape[0])
    from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index

    t_eval = resolve_deploy_eval_time_index(n_times)
    phi_gt = gt_clot_phi_at_time(data, t_eval, phys, device=device).reshape(-1)
    wall = data.mask_wall.reshape(-1).bool()
    phi_pred = phi_pred.reshape(-1) * wall.float()
    phi_gt = phi_gt * wall.float()
    m = compute_clot_relaxed_metrics(phi_pred, phi_gt, data.edge_index, wall_mask=wall)
    out = metrics_to_deploy_prefix(m)
    out["deploy_clot_score"] = clot_score_from_deploy_dict(out)
    out["t_eval"] = int(t_eval)
    out["n_gt"] = int(phi_gt.sum())
    out["n_pred"] = int(phi_pred.sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="ode", choices=["gate", "ode"])
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--da-sweep", default="1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cpu")
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    splits = {"train": WALL_COHORT_V2_TRAIN, "dev": WALL_COHORT_V2_DEV,
              "sealed": WALL_COHORT_V2_GENERALIZATION}
    names = sorted({a for v in splits.values() for a in v})
    das = [float(x) for x in args.da_sweep.split(",") if x.strip()]

    # cache the expensive part (MLS build) across the da sweep
    packs, fields = {}, {}
    for a in names:
        p = ANCHOR_DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        packs[a] = d
        fields[a] = t0_flow_fields(d, bio, hops=args.hops)
    print("loaded %d packs (hops=%d)" % (len(packs), args.hops))

    all_rows = []
    for da in das:
        rows = {}
        for a, d in packs.items():
            wall = d.mask_wall.reshape(-1).bool().numpy()
            if args.mode == "ode":
                from src.core_physics.physics_wall_model import integrate_mat
                mat = integrate_mat(d, bio, fields[a], da_scale=da)
                phi = torch.tensor(((mat >= float(bio.viscosity_mat_crit)) & wall).astype(np.float32))
            else:
                phi = torch.tensor(((fields[a].gate > 0) & wall).astype(np.float32))
            r = score_one(d, phi, phys, device)
            r["anchor"] = a
            r["da_scale"] = da
            rows[a] = r
        for split, members in splits.items():
            sc = [rows[a]["deploy_clot_score"] for a in members if a in rows]
            pr = [rows[a]["deploy_clot_relaxed_prec"] for a in members if a in rows]
            rc = [rows[a]["deploy_clot_relaxed_rec"] for a in members if a in rows]
            f1 = [rows[a]["deploy_clot_f1"] for a in members if a in rows]
            if not sc:
                continue
            print("  da=%-8g %-7s n=%2d  score %.4f (>=0.6: %d)  relP %.3f relR %.3f strictF1 %.3f"
                  % (da, split, len(sc), float(np.mean(sc)), sum(s >= 0.6 for s in sc),
                     float(np.mean(pr)), float(np.mean(rc)), float(np.mean(f1))))
        all_rows.extend(rows.values())

    best_da = None
    if len(das) > 1:
        tr = {}
        for r in all_rows:
            if r["anchor"] in WALL_COHORT_V2_TRAIN:
                tr.setdefault(r["da_scale"], []).append(r["deploy_clot_score"])
        best_da = max(tr, key=lambda k: np.mean(tr[k]))
        print("\nbest da_scale on TRAIN only: %g (train mean %.4f)"
              % (best_da, float(np.mean(tr[best_da]))))

    show = best_da if best_da is not None else das[0]
    print("\nper-vessel at da_scale=%g" % show)
    print("%12s %8s %7s %7s %7s %7s %7s" % ("vessel", "split", "score", "relP", "relR", "F1", "n_pred/gt"))
    for r in sorted((r for r in all_rows if r["da_scale"] == show), key=lambda z: z["anchor"]):
        sp = next((k for k, v in splits.items() if r["anchor"] in v), "?")
        print("%12s %8s %7.4f %7.3f %7.3f %7.3f   %d/%d"
              % (r["anchor"], sp, r["deploy_clot_score"], r["deploy_clot_relaxed_prec"],
                 r["deploy_clot_relaxed_rec"], r["deploy_clot_f1"], r["n_pred"], r["n_gt"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
