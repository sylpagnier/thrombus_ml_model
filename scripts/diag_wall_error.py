"""PHASE 8: what is the remaining WALL error made of?

Off-wall: the model's own Mat cannot beat the shipped speed arm, and putting that arm
on the species shell raises strict F1 while *lowering* the relaxed deploy score (the
metric pays for phantom-band near-hits).  Wall-only is 0.765 of 0.783, wall F1 0.822 --
that is the pool that still moves the number of record without fighting the metric.

The wall mask is t=0 gates plus shear-admitted graph growth.  This script splits the
errors so the next physics change has a target:

    FN, ungated     GT clot whose t=0 gate is closed -- frozen-flow miss
    FN, gated       GT clot inside a gate that the ODE/growth never committed
    FN, grown-out   GT clot outside the gate but inside the admission band, not reached
    FN, other       GT clot the admission band cannot reach at all
    FP, gated       predicted clot on a t=0 gate that GT never clots -- over-ignition
    FP, grown       predicted clot that arrived by graph growth, GT never clots

    python scripts/diag_wall_error.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from predict_wall_clot import GROW_HOPS, RELAX, node_pos, predict_wall_clot  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402

DIR = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10
CRIT = 2.0e7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_wall_error.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    rows = []

    print("%-12s %5s %5s %5s  %7s %7s %7s %7s  %7s %7s"
          % ("anchor", "GT", "pred", "F1", "FN_ung", "FN_gat", "FN_adm", "FN_out",
             "FP_gat", "FP_grw"))
    tot = dict(gt=0, pred=0, tp=0,
               fn_ung=0, fn_gat=0, fn_adm=0, fn_out=0, fp_gat=0, fp_grw=0,
               fn_mat_hi=0, fp_mat_lo=0)

    for anchor in WALL_COHORT_V2_TRAIN:
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        ei = d.edge_index.detach().cpu().numpy()
        n = len(wall)
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")
                                 ).reshape(-1).numpy() > 0.5
        gt_w = gt & wall
        if gt_w.sum() == 0:
            continue
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        pred, _ = predict_wall_clot(d, bio, flow="gt", lumen=False)
        pred_w = pred & wall
        gate = (f.gate > 0) & wall
        adm = (f.sr < float(bio.lss) * RELAX) & wall
        names = d.y_channel_names.split(",")
        mat = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S

        fn = gt_w & ~pred_w
        fp = pred_w & ~gt_w
        fn_ung = fn & ~gate
        fn_gat = fn & gate
        fn_adm = fn & ~gate & adm
        fn_out = fn & ~gate & ~adm
        fp_gat = fp & gate
        fp_grw = fp & ~gate
        f1 = 2 * int((pred_w & gt_w).sum()) / max(int(pred_w.sum()) + int(gt_w.sum()), 1)
        print("   %-12s %5d %5d %5.3f  %7d %7d %7d %7d  %7d %7d"
              % (anchor, int(gt_w.sum()), int(pred_w.sum()), f1,
                 int(fn_ung.sum()), int(fn_gat.sum()), int(fn_adm.sum()),
                 int(fn_out.sum()), int(fp_gat.sum()), int(fp_grw.sum())))
        tot["gt"] += int(gt_w.sum())
        tot["pred"] += int(pred_w.sum())
        tot["tp"] += int((pred_w & gt_w).sum())
        tot["fn_ung"] += int(fn_ung.sum())
        tot["fn_gat"] += int(fn_gat.sum())
        tot["fn_adm"] += int(fn_adm.sum())
        tot["fn_out"] += int(fn_out.sum())
        tot["fp_gat"] += int(fp_gat.sum())
        tot["fp_grw"] += int(fp_grw.sum())
        tot["fn_mat_hi"] += int((fn & (mat >= CRIT)).sum())
        tot["fp_mat_lo"] += int((fp & (mat < CRIT)).sum())
        rows.append(dict(anchor=anchor, gt=int(gt_w.sum()), pred=int(pred_w.sum()),
                         f1=float(f1)))

    ngt = tot["gt"]
    print("\n=== POOLED WALL ERROR, %d vessels, %d GT wall clot nodes ===" % (len(rows), ngt))
    print("   predicted                         %5d" % tot["pred"])
    print("   TP / FN / FP                      %5d / %5d / %5d"
          % (tot["tp"], ngt - tot["tp"], tot["pred"] - tot["tp"]))
    print("   FN ungated (t=0 gate closed)      %5d  %5.1f%% of GT"
          % (tot["fn_ung"], 100 * tot["fn_ung"] / ngt))
    print("      of those, inside admission     %5d" % tot["fn_adm"])
    print("      of those, outside admission    %5d" % tot["fn_out"])
    print("   FN gated (ignited, never committed)%5d  %5.1f%% of GT"
          % (tot["fn_gat"], 100 * tot["fn_gat"] / ngt))
    print("   FP on a t=0 gate (over-ignition)  %5d  %5.1f%% of pred"
          % (tot["fp_gat"], 100 * tot["fp_gat"] / max(tot["pred"], 1)))
    print("   FP by graph growth                %5d  %5.1f%% of pred"
          % (tot["fp_grw"], 100 * tot["fp_grw"] / max(tot["pred"], 1)))
    print("\n   sanity vs the field itself (GT Mat >= crit):")
    print("      FN that ARE Mat>=crit          %5d  <- mask missed a real high-Mat node"
          % tot["fn_mat_hi"])
    print("      FP that are Mat<crit           %5d  <- mask invented clot the field denies"
          % tot["fp_mat_lo"])

    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(tot=tot, rows=rows), indent=2))
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
