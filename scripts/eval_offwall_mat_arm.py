"""PHASE 7 step 1-2: the Mat-magnitude lumen arm, and the flux-into-first-cell rate scale.

Reports, per arm, on the FULL MESH (which is the deliverable -- docs/PHASE6_RESULTS.md 20.3
showed every Phase-6 number was wall-masked and therefore blind to a sixth of the clot):

    deploy_clot_score   the canonical relaxed score, full mesh   <- the metric of record
    deploy (wall-only)  the same score restricted to the wall    <- comparable to Phase 6
    wall F1 / off F1    strict F1 split by domain

Arms:
    wall-only            no lumen arm at all
    speed (shipped)      grow_into_lumen, 2 hops, speed_nd < 0.2
    mat                  grow_into_lumen_by_mat on the rollout's own Mat
    mat + fill           ... with graph-grown wall nodes given an inherited Mat
    mat ORACLE           ... driven by GT Mat, i.e. the ceiling of the mechanism

    python scripts/eval_offwall_mat_arm.py --flow gt
    python scripts/eval_offwall_mat_arm.py --flow gt --da-sweep
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

from predict_wall_clot import node_pos, predict_wall_clot, wall_mat_field  # noqa: E402
from src.biochem_gnn.mat_growth_simple import WALL_COHORT_V2_TRAIN  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    MAT_ATTENUATION, MAT_ATTENUATION_TUNED, SHELL_SPECIES_HI, SHELL_SPECIES_LO,
    fill_grown_wall_mat, grow_into_lumen_by_mat,
)
from src.core_physics.physics_wall_model import t0_flow_fields  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10        # pack Mat_log1p_nd -> COMSOL model units (docs/PHASE6_RESULTS.md 1)


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = int((pred & gt).sum())
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def score(pred: np.ndarray, gt_t: torch.Tensor, ei: torch.Tensor, wall_t=None) -> float:
    m = compute_clot_relaxed_metrics(torch.tensor(pred.astype(np.float32)), gt_t, ei,
                                     wall_mask=wall_t)
    return clot_score_from_deploy_dict(metrics_to_deploy_prefix(m))


def load(anchor: str):
    p = DIR / f"{anchor}.pt"
    if not p.exists():
        return None
    d = torch.load(p, map_location="cpu", weights_only=False)
    if int(d.y.shape[0]) < 150:
        return None
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="gt", choices=["gt", "pred"])
    ap.add_argument("--cohort", default="train")
    ap.add_argument("--da-sweep", action="store_true",
                    help="also sweep da_scale (the flux-into-first-cell rate scale)")
    ap.add_argument("--save", default="outputs/phase7_offwall_mat_arm.json")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    anchors = list(WALL_COHORT_V2_TRAIN)
    arms = ["wall-only", "speed (shipped)", "mat", "mat + fill",
            "mat ORACLE", "mat ORACLE tuned"]
    acc = {a: {"score": [], "score_wall": [], "wall_f1": [], "off_f1": []} for a in arms}
    per_vessel = {}
    da_rows = []

    for anchor in anchors:
        d = load(anchor)
        if d is None:
            continue
        if args.flow == "pred" and getattr(d, "u0_pred", None) is None:
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        ei_np = d.edge_index.detach().cpu().numpy()
        n = len(wall)
        A = sp.coo_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.int8)
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if gt.sum() == 0:
            continue
        wall_t = torch.tensor(wall)
        names = d.y_channel_names.split(",")
        mat_gt = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S

        f = t0_flow_fields(d, bio, hops={"gt": 3, "pred": 4}[args.flow], flow_source=args.flow)
        mat_m = wall_mat_field(d, bio, f)
        pos = node_pos(d)

        base, _ = predict_wall_clot(d, bio, flow=args.flow, lumen=False)
        preds = {"wall-only": base}
        preds["speed (shipped)"] = predict_wall_clot(d, bio, flow=args.flow, lumen=True)[0]
        preds["mat"] = base | grow_into_lumen_by_mat(mat_m, wall, pos, ei_np, crit)
        preds["mat + fill"] = predict_wall_clot(
            d, bio, flow=args.flow, lumen="mat", mat_wall=mat_m)[0]
        preds["mat ORACLE"] = base | grow_into_lumen_by_mat(mat_gt, wall, pos, ei_np, crit)
        preds["mat ORACLE tuned"] = base | grow_into_lumen_by_mat(
            mat_gt, wall, pos, ei_np, crit, attenuation=MAT_ATTENUATION_TUNED)

        row = {}
        for a in arms:
            p = preds[a]
            s = score(p, gt_f, d.edge_index)
            sw = score(p & wall, gt_f * wall_t.float(), d.edge_index, wall_t)
            wf, of = f1(p & wall, gt & wall), f1(p & ~wall, gt & ~wall)
            acc[a]["score"].append(s)
            acc[a]["score_wall"].append(sw)
            acc[a]["wall_f1"].append(wf)
            if not np.isnan(of):
                acc[a]["off_f1"].append(of)
            row[a] = dict(score=s, score_wall=sw, wall_f1=wf, off_f1=of)
        per_vessel[anchor] = row
        print("%-12s nGT %4d (off %3d) | score  %s"
              % (anchor, int(gt.sum()), int((gt & ~wall).sum()),
                 "  ".join("%-16s %.4f" % (a, row[a]["score"]) for a in ("speed (shipped)", "mat + fill", "mat ORACLE"))))

        if args.da_sweep:
            for da in (20.0, 28.0, 40.0, 60.0, 100.0):
                mm = wall_mat_field(d, bio, f, da_scale=da)
                mf = fill_grown_wall_mat(mm, base, wall, A, hops=6)
                pr = base | grow_into_lumen_by_mat(mf, wall, pos, ei_np, crit)
                da_rows.append((da, score(pr, gt_f, d.edge_index),
                                f1(pr & wall, gt & wall), f1(pr & ~wall, gt & ~wall)))

    nv = len(per_vessel)
    print("\n%d vessels, flow=%s.  MAT_ATTENUATION=%.2f, shell = topological"
          " (first_corner_shell; band fallback %.2f-%.2f median edge lengths)"
          % (nv, args.flow, MAT_ATTENUATION, SHELL_SPECIES_LO, SHELL_SPECIES_HI))
    print("%-18s %11s %11s %9s %8s" % ("arm", "score(FULL)", "score(wall)", "wall F1", "off F1"))
    for a in arms:
        print("%-18s %11.4f %11.4f %9.4f %8.4f"
              % (a, np.mean(acc[a]["score"]), np.mean(acc[a]["score_wall"]),
                 np.nanmean(acc[a]["wall_f1"]),
                 np.mean(acc[a]["off_f1"]) if acc[a]["off_f1"] else float("nan")))
    base_s = np.mean(acc["speed (shipped)"]["score"])
    for a in arms:
        print("   delta vs shipped  %-18s %+0.4f" % (a, np.mean(acc[a]["score"]) - base_s))

    if da_rows:
        print("\nda_scale sweep (mat + fill arm).  1/h from the mesh = 28.1, shipped = 40")
        print("%8s %11s %9s %8s" % ("da_scale", "score(FULL)", "wall F1", "off F1"))
        for da in (20.0, 28.0, 40.0, 60.0, 100.0):
            r = [x for x in da_rows if x[0] == da]
            print("%8.0f %11.4f %9.4f %8.4f"
                  % (da, np.mean([x[1] for x in r]), np.nanmean([x[2] for x in r]),
                     np.nanmean([x[3] for x in r])))

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"flow": args.flow, "n_vessels": nv, "per_vessel": per_vessel,
         "summary": {a: {k: float(np.nanmean(v)) if v else None for k, v in acc[a].items()}
                     for a in arms}}, indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
