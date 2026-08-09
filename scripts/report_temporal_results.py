"""Consolidated temporal result: mask quality AND growth-curve quality, both flow arms.

Variants, all zero-learned-parameter:
  A  gate + graph growth      -- the shipped model (docs/PHASE3_RESULTS.md 4). No time axis.
  B  ODE + wake feedback      -- arm 2 with the corrected sign; has a real growth curve.
  C  B + graph growth         -- do the ad-hoc dilation and the physics feedback stack?

Reference ceiling: ``diag_timevarying_gate_oracle.py`` integrates the same ODE against
GT's own time-varying gates, which upper-bounds ANY evolving-flow model, learned or not.
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    WALL_COHORT_V2_GENERALIZATION, WALL_COHORT_V2_TRAIN,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import node_positions  # noqa: E402
from src.core_physics.physics_wall_model import (  # noqa: E402
    first_crossing, graded_gate, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.shear_redistribution import (  # noqa: E402
    build_crosssection_operator, make_blockage, sdf_nd,
)
from src.core_physics.species_pushforward_continuous import resolve_deploy_eval_time_index  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.temporal_metrics import curve_l1, gt_onset_index, onset_metrics  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = Path("data/processed/graphs_biochem_anchors")
STENCIL = {"gt": 3, "pred": 4}
RELAX, GROW = 2.0, 6
DA, RM, WAKE, EVERY = 40.0, 0.30, 8.0, 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/temporal_final.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    names = sorted(set(WALL_COHORT_V2_TRAIN) | set(WALL_COHORT_V2_GENERALIZATION))
    rows = []
    for a in names:
        p = DIR / f"{a}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        n = len(wall)
        ei = d.edge_index.numpy()
        A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        pos, sd = node_positions(d), sdf_nd(d)
        B = build_crosssection_operator(pos, sd, wall, radius_mult=RM)
        gt_idx = gt_onset_index(d, phys, wall)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        pg = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        pg = pg * torch.tensor(wall.astype(np.float32))
        split = "sealed" if a in WALL_COHORT_V2_GENERALIZATION else "train"

        def measure(mask, idx, t):
            m = {"score": clot_score_from_deploy_dict(metrics_to_deploy_prefix(
                compute_clot_relaxed_metrics(torch.tensor(mask.astype(np.float32)), pg,
                                             d.edge_index, wall_mask=torch.tensor(wall))))}
            if idx is not None:
                om = onset_metrics(idx, gt_idx, t, wall)
                m["rho"] = om["rho"]
                m["spread_ratio"] = om["spread_ratio"]
                m["curve_l1"] = curve_l1(idx, gt_idx, t, wall)
            else:
                m["rho"] = m["spread_ratio"] = m["curve_l1"] = float("nan")
            return m

        r = {"anchor": a, "split": split}
        for arm in ("gt", "pred"):
            try:
                f = t0_flow_fields(d, bio, hops=STENCIL[arm], flow_source=arm)
            except ValueError:
                continue
            # A: gate + graph growth (no time axis)
            cur = (f.gate > 0) & wall
            adm = (f.sr < float(bio.lss) * RELAX) & wall
            for _ in range(GROW):
                cur = cur | (((A @ cur.astype(np.int8)) > 0) & adm)
            r[f"A_{arm}"] = measure(cur, None, None)
            # B: ODE + wake
            g0 = graded_gate(f, bio, mode="hard") * wall
            blk = make_blockage(f, bio, B, wall, every=EVERY, feedback="wake", wake=WAKE)
            traj, t = integrate_mat_trajectory(d, bio, g0, da_scale=DA, blockage=blk)
            idxB = first_crossing(traj, float(bio.viscosity_mat_crit))
            maskB = (idxB >= 0) & wall
            r[f"B_{arm}"] = measure(maskB, idxB, t)
            # C: B + graph growth
            curC = maskB.copy()
            for _ in range(GROW):
                curC = curC | (((A @ curC.astype(np.int8)) > 0) & adm)
            idxC = np.where(curC & (idxB < 0), len(t) - 1, idxB)
            r[f"C_{arm}"] = measure(curC, idxC, t)
        rows.append(r)

    def agg(split, key, metric):
        v = [r[key][metric] for r in rows if r["split"] == split and key in r
             and r[key][metric] == r[key][metric]]
        return float(np.mean(v)) if v else float("nan")

    print("full-horizon vessels: %d train, %d sealed\n"
          % (sum(r["split"] == "train" for r in rows), sum(r["split"] == "sealed" for r in rows)))
    hdr = ("variant", "flow", "score", "curveL1", "rho", "sprRat")
    for split in ("train", "sealed"):
        print("--- %s ---" % split.upper())
        print("%-26s %5s %8s %8s %7s %7s" % hdr)
        for key, lbl in (("A", "A gate+growth (shipped)"), ("B", "B ODE+wake"),
                         ("C", "C ODE+wake+growth")):
            for arm in ("gt", "pred"):
                k = f"{key}_{arm}"
                print("%-26s %5s %8.4f %8s %7s %7s"
                      % (lbl if arm == "gt" else "", arm, agg(split, k, "score"),
                         "--" if agg(split, k, "curve_l1") != agg(split, k, "curve_l1")
                         else "%.4f" % agg(split, k, "curve_l1"),
                         "--" if agg(split, k, "rho") != agg(split, k, "rho")
                         else "%.3f" % agg(split, k, "rho"),
                         "--" if agg(split, k, "spread_ratio") != agg(split, k, "spread_ratio")
                         else "%.3f" % agg(split, k, "spread_ratio")))
        print()
    print("reference: ODE with GT time-varying gates (ORACLE, illegal): "
          "train score 0.8913 curveL1 0.0670 rho 0.795 | sealed 0.9066 0.0821 0.665")
    print("reference: ODE, frozen hard t=0 gate, no wake:               "
          "train score 0.7919 curveL1 0.1018 rho 0.713 | sealed 0.8645 0.0903 0.682")
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
