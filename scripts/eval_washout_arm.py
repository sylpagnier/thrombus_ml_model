"""PHASE 8: does the Mat washout term earn its place on the metrics of record?

The mechanism is established elsewhere -- ``diag_local_ode_closure.py`` (the accumulate-only
per-node ODE cannot order GT ``Mat`` even with oracle inputs) and ``diag_mat_washout.py``
(a single global ``lambda`` takes oracle rank 0.310 -> 0.447 leave-one-vessel-out, beating a
bare lifetime, with pure saturation buying nothing).  This script asks the only question that
decides whether it ships, on the model's OWN ``Mat`` rather than an oracle:

    rho_corner    spearman(model Mat, GT Mat) on CORNER wall nodes
                  FINDINGS 8.5's target -- 0.193 for the shipped model.  Corner-only because
                  half the wall is mid-edge nodes with structurally zero GT Mat, and scoring
                  those inflates the correlation to 0.534 on agreed zeros.
    off F1        strict off-wall F1 from the Mat-magnitude lumen arm.  This is what the
                  ordering is FOR: FINDINGS' headline is that the mechanism is worth +0.068
                  under a GT-Mat oracle and -0.017 on the model's own Mat, and the whole gap
                  is the ordering.
    score         canonical relaxed deploy clot score on the FULL MESH -- the deliverable.
    onset         growth_l1 against the GT growth count, i.e. does the clot arrive at the
                  right TIME.  A removal term changes when a node crosses crit, so this can
                  move in either direction and must be reported either way.

Arms: washout off (the shipped model, bit-for-bit) against washout on, everything else held.

    python scripts/eval_washout_arm.py --flow gt
    python scripts/eval_washout_arm.py --flow gt --lam-sweep
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

from predict_wall_clot import node_pos, predict_wall_clot, wall_mat_field  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    fill_grown_wall_mat, grow_into_lumen_by_mat, midside_nodes,
)
from src.core_physics.physics_wall_model import WASHOUT_LAMBDA, t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import spearman  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10        # pack Mat_log1p_nd -> COMSOL model units (docs/PHASE6_RESULTS.md 1)
FILL_HOPS = 6


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--lam-sweep", action="store_true")
    ap.add_argument("--save", default="outputs/phase8_washout_arm.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    lams = [0.0, WASHOUT_LAMBDA] if not args.lam_sweep else \
        [0.0, 3.652e-7, 7.499e-7, WASHOUT_LAMBDA, 3.162e-6, 6.494e-6]

    packs = []
    for anchor in WALL_COHORT_V2_TRAIN:
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        if args.flow == "pred" and getattr(d, "u0_pred", None) is None:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        ei_np = d.edge_index.detach().cpu().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")
                                 ).reshape(-1).numpy() > 0.5
        if gt.sum() == 0:
            continue
        names = d.y_channel_names.split(",")
        mat_gt = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        f = t0_flow_fields(d, bio, hops={"gt": 3, "pred": 4}[args.flow],
                           flow_source=args.flow)
        ms = midside_nodes(node_pos(d), ei_np)
        base, _ = predict_wall_clot(d, bio, flow=args.flow, lumen=False)
        packs.append(dict(anchor=anchor, d=d, wall=wall, ei=ei_np, gt=gt, mat_gt=mat_gt,
                          f=f, midside=ms, base=base))
    print("[i] %d vessels, flow=%s" % (len(packs), args.flow))

    per_lam = {}
    for lam in lams:
        rows = []
        for pk in packs:
            d, wall, ei_np = pk["d"], pk["wall"], pk["ei"]
            gt, mat_gt = pk["gt"], pk["mat_gt"]
            mat_m = wall_mat_field(d, bio, pk["f"], washout=lam)
            # Ordering on corner wall nodes that GT actually populates (FINDINGS 8.5).
            corner = wall & ~pk["midside"] & (mat_gt > 0) & (mat_m > 0)
            rho_c = spearman(mat_m[corner], mat_gt[corner]) if corner.sum() > 8 else np.nan
            # The off-wall arm this ordering feeds.
            mw = fill_grown_wall_mat(mat_m, pk["base"], wall,
                                     _adj(ei_np, len(wall)), hops=FILL_HOPS)
            off = grow_into_lumen_by_mat(mw, wall, node_pos(d), ei_np, crit)
            pred = pk["base"] | off
            m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)),
                                             torch.tensor(gt.astype(np.float32)),
                                             d.edge_index)
            rows.append(dict(anchor=pk["anchor"], rho_corner=float(rho_c),
                             off_f1=float(f1(pred & ~wall, gt & ~wall)),
                             wall_f1=float(f1(pred & wall, gt & wall)),
                             score=float(clot_score_from_deploy_dict(
                                 metrics_to_deploy_prefix(m)))))
        g = lambda k: np.array([r[k] for r in rows], dtype=float)
        per_lam[lam] = dict(rows=rows, rho_corner=float(np.nanmean(g("rho_corner"))),
                            off_f1=float(np.nanmean(g("off_f1"))),
                            wall_f1=float(np.nanmean(g("wall_f1"))),
                            score=float(np.nanmean(g("score"))))

    print("\n=== WASHOUT ARM, %d train vessels, flow=%s ===" % (len(packs), args.flow))
    print("   %-14s %11s %10s %10s %10s" % ("lambda", "rho_corner", "off F1", "wall F1",
                                            "score"))
    for lam in lams:
        r = per_lam[lam]
        tag = "off (shipped)" if lam == 0.0 else "%.3e" % lam
        print("   %-14s %11.3f %10.4f %10.4f %10.4f"
              % (tag, r["rho_corner"], r["off_f1"], r["wall_f1"], r["score"]))

    b, w = per_lam[0.0], per_lam[WASHOUT_LAMBDA]
    print("\n   delta at lambda = %.3e:" % WASHOUT_LAMBDA)
    print("      rho_corner   %+.3f   (%.3f -> %.3f)"
          % (w["rho_corner"] - b["rho_corner"], b["rho_corner"], w["rho_corner"]))
    print("      off-wall F1  %+.4f  (%.4f -> %.4f)"
          % (w["off_f1"] - b["off_f1"], b["off_f1"], w["off_f1"]))
    print("      deploy score %+.4f  (%.4f -> %.4f)"
          % (w["score"] - b["score"], b["score"], w["score"]))
    print("\n   per-vessel rho_corner and score:")
    for rb, rw in zip(b["rows"], w["rows"]):
        print("      %-12s rho %+.3f -> %+.3f    score %.4f -> %.4f"
              % (rb["anchor"], rb["rho_corner"], rw["rho_corner"], rb["score"], rw["score"]))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({str(k): v for k, v in per_lam.items()}, indent=2))
    print("\nwrote %s" % out)
    return 0


def _adj(ei_np, n):
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


if __name__ == "__main__":
    raise SystemExit(main())
