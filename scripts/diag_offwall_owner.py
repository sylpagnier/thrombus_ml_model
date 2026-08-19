"""PHASE 8: is the off-wall arm using the wrong owner, and what is the true ceiling?

GT clot IS the level set {Mat >= 2e7} on wall AND off-wall (diag_fibrin_clot_route.py).
So the physically right off-wall rule is ``shell & (Mat_self >= crit)``, not
``att * Mat_nearest_wall >= crit``.  ``grow_into_lumen_by_mat`` uses the second form
because the model only integrates Mat at the wall.  This script splits three errors
the Phase-7 table conflates:

    A. ceiling of the FIELD:        shell & (GT Mat_self >= crit)
    B. ceiling of WALL-Mat + att:   shell & (att * GT Mat_owner >= crit)   [current form]
    C. deployable:                  same as B, model's own Mat, vs the shipped speed arm

and two owner definitions:

    nearest     KD-tree nearest wall node     (shipped)
    topo        wall node on the far side of the wall-normal mid-side bridge
                -- the element the shell node actually sits in, no length

If A >> B, the attenuation form is the bottleneck and no wall-Mat model can close it.
If topo beats nearest on B, the KD-tree owner is a bug and we should ship the fix.
If C with topo/att beats speed, the mat arm finally earns its place.

    python scripts/diag_offwall_owner.py
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
    MAT_ATTENUATION, grow_into_lumen_by_mat, resolve_offwall_shell,
    topological_owner, wall_normal_projection,
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
MAT_S = 7e10
CRIT = 2.0e7


def f1_pr(pred, gt):
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


def off_from(mat, owner, shell, att):
    ok = owner >= 0
    pred = np.zeros(len(shell), dtype=bool)
    pred[ok] = shell[ok] & (float(att) * mat[owner[ok]] >= CRIT)
    return pred


def deploy_score(pred, gt_t, ei):
    m = compute_clot_relaxed_metrics(
        torch.tensor(pred.astype(np.float32)), gt_t, ei)
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="outputs/phase8_offwall_owner.json")
    args = ap.parse_args()
    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)

    packs = []
    for anchor in WALL_COHORT_V2_TRAIN:
        p = DIR / f"{anchor}.pt"
        if not p.exists():
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < 150:
            continue
        names = d.y_channel_names.split(",")
        wall = d.mask_wall.reshape(-1).bool().numpy()
        pos, ei = node_pos(d), d.edge_index.detach().cpu().numpy()
        mat = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if gt.sum() == 0:
            continue
        shell = resolve_offwall_shell(pos, wall, ei)
        _, near = wall_normal_projection(pos, wall)
        topo = topological_owner(pos, wall, ei)
        f = t0_flow_fields(d, bio, hops=3, flow_source="gt")
        mat_m = wall_mat_field(d, bio, f)
        base, _ = predict_wall_clot(d, bio, flow="gt", lumen=False)
        speed, _ = predict_wall_clot(d, bio, flow="gt", lumen=True)
        packs.append(dict(anchor=anchor, d=d, wall=wall, pos=pos, ei=ei, mat=mat,
                          mat_m=mat_m, shell=shell, near=near, topo=topo, gt=gt,
                          gt_f=gt_f, base=base, speed=speed, gt_off=gt & ~wall))

    print("[i] %d vessels" % len(packs))
    n_off = sum(int(p["gt_off"].sum()) for p in packs)
    n_cov = sum(int((p["gt_off"] & (p["topo"] >= 0)).sum()) for p in packs)
    print("   topological owner covers %d / %d off-wall GT clot nodes (%.1f%%)"
          % (n_cov, n_off, 100.0 * n_cov / max(n_off, 1)))

    # --- A. true field ceiling ---
    print("\n=== A. TRUE CEILING: threshold GT Mat ON THE NODE ITSELF ===")
    fs, ps, rs, sc = [], [], [], []
    for p in packs:
        pred = p["shell"] & (p["mat"] >= CRIT)
        a, b, c = f1_pr(pred, p["gt_off"])
        fs.append(a); ps.append(b); rs.append(c)
        sc.append(deploy_score(p["base"] | pred, p["gt_f"], p["d"].edge_index))
    print("   shell & (Mat_self >= crit)   off F1 %.4f  P %.3f  R %.3f  score %.4f"
          % (np.nanmean(fs), np.nanmean(ps), np.nanmean(rs), np.mean(sc)))
    fs2, ps2, rs2 = [], [], []
    for p in packs:
        pred = (~p["wall"]) & (p["mat"] >= CRIT)     # no shell restriction
        a, b, c = f1_pr(pred, p["gt_off"])
        fs2.append(a); ps2.append(b); rs2.append(c)
    print("   ALL off-wall & (Mat_self >= crit)  off F1 %.4f  P %.3f  R %.3f"
          % (np.nanmean(fs2), np.nanmean(ps2), np.nanmean(rs2)))

    # --- B. wall-Mat + att, two owners, GT Mat ---
    print("\n=== B. WALL-Mat FORM: att * GT Mat_owner >= crit  (off F1) ===")
    atts = [0.10, 0.16, 0.18, 0.25, 0.35, 0.50, 1.00]
    print("   %-8s %10s %10s" % ("att", "nearest", "topo"))
    best = {}
    for att in atts:
        row = {}
        for key in ("near", "topo"):
            vals = [f1_pr(off_from(p["mat"], p[key], p["shell"], att), p["gt_off"])[0]
                    for p in packs]
            row[key] = float(np.nanmean(vals))
        print("   %-8.2f %10.4f %10.4f" % (att, row["near"], row["topo"]))
        best[att] = row

    # finer sweep for each owner
    print("\n   best constant on a finer grid:")
    fine = np.logspace(-1.3, 0.0, 25)
    for key, lbl in (("near", "nearest"), ("topo", "topo")):
        scores = []
        for att in fine:
            vals = [f1_pr(off_from(p["mat"], p[key], p["shell"], att), p["gt_off"])[0]
                    for p in packs]
            scores.append((float(att), float(np.nanmean(vals))))
        a, v = max(scores, key=lambda kv: kv[1])
        print("      %-8s att=%.3f  off F1 %.4f" % (lbl, a, v))

    # --- C. deployable: model Mat vs speed ---
    print("\n=== C. DEPLOY SCORE, model's own Mat, vs shipped speed arm ===")
    speed_s = np.mean([deploy_score(p["speed"], p["gt_f"], p["d"].edge_index) for p in packs])
    wall_s = np.mean([deploy_score(p["base"], p["gt_f"], p["d"].edge_index) for p in packs])
    print("   wall-only                         %.4f" % wall_s)
    print("   speed (shipped)                   %.4f" % speed_s)

    print("   %-8s %10s %10s %12s" % ("att", "nearest", "topo", "topo OR speed"))
    c_rows = []
    for att in atts:
        near_s, topo_s, or_s = [], [], []
        for p in packs:
            pn = p["base"] | off_from(p["mat_m"], p["near"], p["shell"], att)
            pt = p["base"] | off_from(p["mat_m"], p["topo"], p["shell"], att)
            po = pt | p["speed"]
            near_s.append(deploy_score(pn, p["gt_f"], p["d"].edge_index))
            topo_s.append(deploy_score(pt, p["gt_f"], p["d"].edge_index))
            or_s.append(deploy_score(po, p["gt_f"], p["d"].edge_index))
        print("   %-8.2f %10.4f %10.4f %12.4f"
              % (att, np.mean(near_s), np.mean(topo_s), np.mean(or_s)))
        c_rows.append(dict(att=att, near=float(np.mean(near_s)),
                           topo=float(np.mean(topo_s)), or_speed=float(np.mean(or_s))))

    # shipped grow_into_lumen_by_mat for reference
    ship_mat = np.mean([
        deploy_score(p["base"] | grow_into_lumen_by_mat(
            p["mat_m"], p["wall"], p["pos"], p["ei"], crit), p["gt_f"], p["d"].edge_index)
        for p in packs])
    print("   grow_into_lumen_by_mat (shipped form) %.4f" % ship_mat)

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        n=len(packs), field_ceiling=dict(f1=float(np.nanmean(fs)), score=float(np.mean(sc))),
        gt_att=best, deploy=c_rows, speed=speed_s, wall=wall_s), indent=2))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
