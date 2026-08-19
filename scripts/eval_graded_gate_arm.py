"""PHASE 8: t=0 surrogate for the time-averaged gate, plus longer along-wall growth.

``eval_wall_gate_ceiling.py`` found the wall-mask prize:

    union of the COMSOL gate over GT flow, no growth     +0.051 deploy score
    field {Mat >= crit}                                   +0.096
    hops=12, same admission                               +0.013   (deployable)

The union is an oracle.  ``graded_gate`` (already in the wall model) was written as the
t=0 surrogate for it: a node just ABOVE ``lss`` at t=0 is the one most likely to ENTER the
gate as neighbouring clot slows the flow.  This script sweeps that surrogate and the hop
count, with leave-one-vessel-out so a longer hop is not just reading TRAIN.

    python scripts/eval_graded_gate_arm.py
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

from predict_wall_clot import GROW_HOPS, LUMEN_HOPS, LUMEN_SPEED, RELAX  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import grow_into_lumen, speed_nd  # noqa: E402
from src.core_physics.physics_wall_model import graded_gate, t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"


def f1(pred, gt):
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def sc(pred, gt_t, ei):
    m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)), gt_t, ei)
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def grow_mask(seed, wall, A, sr, bio, hops, relax):
    cur = seed.copy()
    adm = (sr < float(bio.lss) * relax) & wall
    for _ in range(hops):
        cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_graded_gate_arm.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")

    packs = []
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
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if (gt & wall).sum() == 0:
            continue
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        packs.append(dict(anchor=anchor, d=d, wall=wall, A=A, gt=gt, gt_f=gt_f,
                          f=f, spd=speed_nd(d)))
    print("[i] %d vessels" % len(packs))

    def pred_of(p, hops, relax, mode="hard", tau=0.25, gcut=0.0):
        if mode == "hard":
            seed = (p["f"].gate > 0) & p["wall"]
        else:
            g = graded_gate(p["f"], bio, mode=mode, tau_low=tau)
            seed = (g > gcut) & p["wall"]
        msk = grow_mask(seed, p["wall"], p["A"], p["f"].sr, bio, hops, relax)
        off = grow_into_lumen(msk, p["wall"], p["A"], p["spd"], p["f"].sr,
                              lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
        return msk | off

    def mean_score(hops, relax, mode="hard", tau=0.25, gcut=0.0):
        return float(np.mean([sc(pred_of(p, hops, relax, mode, tau, gcut),
                                 p["gt_f"], p["d"].edge_index) for p in packs]))

    shipped = mean_score(GROW_HOPS, RELAX)
    print("   shipped hops=%d relax=%.1f           %.4f" % (GROW_HOPS, RELAX, shipped))

    print("\n=== HARD GATE, hop count (relax=2.0) ===")
    hop_scores = {}
    for h in (6, 8, 10, 12, 16, 20, 40, 80):
        s = mean_score(h, 2.0)
        hop_scores[h] = s
        print("   hops=%-2d  %.4f  %+.4f" % (h, s, s - shipped))

    print("\n=== GROW TO SATURATION of the admission band (relax=2.0) ===")
    sat_scores, sat_hops = [], []
    for p in packs:
        seed = (p["f"].gate > 0) & p["wall"]
        cur = seed.copy()
        hops_used = 0
        adm = (p["f"].sr < float(bio.lss) * 2.0) & p["wall"]
        while hops_used < 200:
            nxt = cur | (((p["A"] @ cur.astype(np.int8)) > 0) & adm)
            hops_used += 1
            if np.array_equal(nxt, cur):
                break
            cur = nxt
        off = grow_into_lumen(cur, p["wall"], p["A"], p["spd"], p["f"].sr,
                              lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
        sat_scores.append(sc(cur | off, p["gt_f"], p["d"].edge_index))
        sat_hops.append(hops_used)
        print("   %-12s hops_to_sat %3d  score %.4f" % (p["anchor"], hops_used, sat_scores[-1]))
    print("   MEAN score %.4f  %+.4f   median hops-to-sat %d"
          % (float(np.mean(sat_scores)), float(np.mean(sat_scores)) - shipped,
             int(np.median(sat_hops))))
    print("\n=== GRADED GATE (sigmoid_low), hops=6 -- expected to lose, kept as the control ===")
    grade_rows = []
    for tau in (0.10, 0.25, 0.50, 1.00):
        for gcut in (0.05, 0.15, 0.30, 0.50):
            s = mean_score(6, 2.0, mode="sigmoid_low", tau=tau, gcut=gcut)
            print("   tau=%.2f cut=%.2f  %.4f  %+.4f" % (tau, gcut, s, s - shipped))
            grade_rows.append(dict(tau=tau, gcut=gcut, hops=6, score=s))
    best_g = max(grade_rows, key=lambda r: r["score"])

    print("\n=== GRADED + hops=12, same tau/cut grid ===")
    for tau in (0.10, 0.25, 0.50, 1.00):
        for gcut in (0.15, 0.50):
            s = mean_score(12, 2.0, mode="sigmoid_low", tau=tau, gcut=gcut)
            print("   tau=%.2f cut=%.2f hops=12  %.4f  %+.4f" % (tau, gcut, s, s - shipped))
            grade_rows.append(dict(tau=tau, gcut=gcut, hops=12, score=s))
    best_all = max(grade_rows, key=lambda r: r["score"])

    # LOO on hop count: pick hops on 18, score the held-out one. Hard gate, relax=2.
    print("\n=== LEAVE-ONE-VESSEL-OUT hops (hard gate, relax=2) ===")
    hop_list = (6, 12, 20, 40)
    per = {h: [] for h in hop_list}
    for p in packs:
        for h in hop_list:
            per[h].append(sc(pred_of(p, h, 2.0), p["gt_f"], p["d"].edge_index))
    loo = []
    print("   %-12s %8s %8s %8s" % ("held out", "hops=6", "chosen", "score"))
    for i, p in enumerate(packs):
        keep = [j for j in range(len(packs)) if j != i]
        h_star = max(hop_list, key=lambda h: float(np.mean([per[h][j] for j in keep])))
        loo.append(per[h_star][i])
        print("   %-12s %8.4f %8d %8.4f"
              % (p["anchor"], per[6][i], h_star, per[h_star][i]))
    print("   MEAN LOO %.4f   shipped-on-same %.4f   delta %+.4f"
          % (float(np.mean(loo)), float(np.mean(per[6])),
             float(np.mean(loo)) - float(np.mean(per[6]))))

    print("\n   best graded: tau=%.2f cut=%.2f hops=%d  %.4f"
          % (best_all["tau"], best_all["gcut"], best_all["hops"], best_all["score"]))

    path = Path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(
        shipped=shipped, hops=hop_scores, graded=grade_rows,
        loo_hops=float(np.mean(loo)), loo_base=float(np.mean(per[6]))), indent=2))
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
