"""Domain-restricted deploy scores against the 0.9 wall / 0.7 off-wall targets.

The blended full-mesh number hides a wall model that is close to 0.9 from an off-wall
arm that cannot reach 0.7.  This script scores the shipped model, the Mat-magnitude
lumen variants (constant att, flux/residence, D=0 convection), and the GT-Mat oracles
that are the ceilings, under the FIT/DEV protocol.  SEALED stays closed.

    python scripts/eval_domain_targets.py --flow pred
    python scripts/eval_domain_targets.py --flow gt --coupled
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

from predict_wall_clot import (  # noqa: E402
    GROW_HOPS, LUMEN_HOPS, LUMEN_MAT_FILL_HOPS, LUMEN_SPEED, RELAX, STENCIL, node_pos,
    predict_wall_clot, wall_mat_field,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.ap_closure import ApClosure, SHIPPED_DA_SCALE, make_rollout_hook  # noqa: E402
from src.core_physics.physics_lumen_model import (  # noqa: E402
    fill_grown_wall_mat, grow_into_lumen, grow_into_lumen_by_convection,
    grow_into_lumen_by_first_cell, grow_into_lumen_by_flux, grow_into_lumen_by_mat,
    grow_into_lumen_by_tds2, resolve_offwall_shell, uv_nd,
)
from src.core_physics.physics_wall_model import (  # noqa: E402
    WASHOUT_LAMBDA, integrate_mat_trajectory, t0_flow_fields,
)
from src.core_physics.shear_redistribution import (  # noqa: E402
    build_crosssection_operator, make_blockage, sdf_nd,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    resolve_deploy_eval_time_index,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.wall_cohort_splits import (  # noqa: E402
    DEV, FIT, MIN_T, SEALED, format_split_means, mean_by_split, split_of,
)
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    clot_score_from_deploy_dict, compute_clot_relaxed_metrics, metrics_to_deploy_prefix,
)

DIR = REPO / "data/processed/graphs_biochem_anchors"
MAT_S = 7e10
# C for kernel=mat_linear from the window-stability table in ap_closure.py (FIT windows).
MAT_LINEAR_C = 106.2
WALL_TARGET = 0.9
OFF_TARGET = 0.7


def f1(pred, gt):
    if gt.sum() == 0 and pred.sum() == 0:
        return float("nan")
    if gt.sum() == 0:
        return float("nan")
    tp = int((pred & gt).sum())
    p, r = tp / max(int(pred.sum()), 1), tp / max(int(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def domain_score(pred, gt, ei, domain, wall):
    if int((gt & domain).sum()) == 0:
        return float("nan")
    dom = torch.tensor(domain.astype(np.float32))
    m = compute_clot_relaxed_metrics(
        torch.tensor(pred.astype(np.float32)) * dom,
        torch.tensor(gt.astype(np.float32)) * dom,
        ei, wall_mask=torch.tensor(wall))
    return float(clot_score_from_deploy_dict(metrics_to_deploy_prefix(m)))


def adj(ei, n):
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(np.int8)


def coupled_mat_field(data, bio, f, wall, pos):
    """Deployable three-way: mat_linear AP + washout + algebraic-wake shear."""
    hook = make_rollout_hook(ApClosure(C=MAT_LINEAR_C, q=1.0, kernel="mat_linear"), bio, f.sr)
    B = build_crosssection_operator(pos, sdf_nd(data), wall)
    blk = make_blockage(f, bio, B, wall, feedback="wake", wake=1.0)
    crit = float(bio.viscosity_mat_crit)
    sr0 = f.sr

    def wsr(mat, _step):
        occ = (mat >= crit).astype(np.float64)
        phi = np.clip(np.asarray(B @ occ).reshape(-1), 0.0, 0.85)
        return sr0 * np.clip(1.0 - phi, 0.02, 1.0)

    traj, _ = integrate_mat_trajectory(
        data, bio, f.gate * wall,
        da_scale=SHIPPED_DA_SCALE, ap_closure=hook, blockage=blk,
        washout=WASHOUT_LAMBDA, washout_sr=wsr)
    return traj[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", default="pred", choices=["pred", "gt"])
    ap.add_argument("--coupled", action="store_true",
                    help="also integrate mat_linear + washout + wake (slow)")
    ap.add_argument("--save", default="outputs/phase8_domain_targets.json")
    args = ap.parse_args()

    bio, phys = BiochemConfig(phase="biochem"), PhysicsConfig(phase="biochem")
    crit = float(bio.viscosity_mat_crit)
    want = list(FIT) + list(DEV)
    print("[i] SEALED closed (%s)" % ", ".join(SEALED))
    print("[i] targets  wall deploy > %.2f   off-wall deploy > %.2f" % (WALL_TARGET, OFF_TARGET))
    print("[i] flow=%s  (pred = RGP-DEQ u0; gt = t=0 COMSOL, oracle/bandaid)" % args.flow)

    rows = []
    skipped = []
    for anchor in want:
        pth = DIR / f"{anchor}.pt"
        if not pth.exists():
            skipped.append((anchor, split_of(anchor), "no pack"))
            continue
        d = torch.load(pth, map_location="cpu", weights_only=False)
        if int(d.y.shape[0]) < MIN_T:
            skipped.append((anchor, split_of(anchor), "T=%d" % int(d.y.shape[0])))
            continue
        if args.flow == "pred" and getattr(d, "u0_pred", None) is None:
            skipped.append((anchor, split_of(anchor), "no u0_pred"))
            continue
        wall = d.mask_wall.reshape(-1).bool().numpy()
        t_eval = resolve_deploy_eval_time_index(int(d.y.shape[0]))
        gt_f = gt_clot_phi_at_time(d, t_eval, phys, device=torch.device("cpu")).reshape(-1)
        gt = gt_f.numpy() > 0.5
        if (gt & wall).sum() == 0:
            skipped.append((anchor, split_of(anchor), "empty GT"))
            continue
        ei = d.edge_index.detach().cpu().numpy()
        A = adj(ei, len(wall))
        f = t0_flow_fields(d, bio, hops=STENCIL[args.flow], flow_source=args.flow)
        pos = node_pos(d)
        u, v = uv_nd(d, flow=args.flow)
        names = d.y_channel_names.split(",")
        mat_gt = np.expm1(d.y[-1, :, names.index("Mat_log1p_nd")].double().numpy()) * MAT_S
        wall_msk, _ = predict_wall_clot(d, bio, flow=args.flow, lumen=False)
        speed = predict_wall_clot(d, bio, flow=args.flow, lumen=True)[0]
        mat_m = wall_mat_field(d, bio, f)
        mw = fill_grown_wall_mat(mat_m, wall_msk, wall, A, hops=LUMEN_MAT_FILL_HOPS)
        shell = resolve_offwall_shell(pos, wall, ei)
        spd = np.hypot(u, v)
        off_speed = grow_into_lumen(wall_msk, wall, A, spd, f.sr,
                                    lumen_hops=LUMEN_HOPS, speed_thresh=LUMEN_SPEED)
        off_att = grow_into_lumen_by_mat(mw, wall, pos, ei, crit)
        off_flux = grow_into_lumen_by_flux(mw, wall, pos, ei, crit, u, v)
        off_conv = grow_into_lumen_by_convection(mw, wall, pos, ei, crit, u, v)
        off_gt_att = grow_into_lumen_by_mat(mat_gt, wall, pos, ei, crit)
        off_gt_flux = grow_into_lumen_by_flux(mat_gt, wall, pos, ei, crit, u, v)
        off_gt_conv = grow_into_lumen_by_convection(mat_gt, wall, pos, ei, crit, u, v)
        off_tds2 = grow_into_lumen_by_tds2(mw, wall, pos, ei, crit, u, v)
        off_tds2_char = grow_into_lumen_by_tds2(
            mw, wall, pos, ei, crit, u, v, blend_nearest=False)
        off_gt_tds2 = grow_into_lumen_by_tds2(mat_gt, wall, pos, ei, crit, u, v)
        off_gt_tds2_char = grow_into_lumen_by_tds2(
            mat_gt, wall, pos, ei, crit, u, v, blend_nearest=False)
        off_gt_self = shell & (mat_gt >= crit)
        off_fc = grow_into_lumen_by_first_cell(mw, wall, pos, ei, crit, u, v)
        off_fc_c = grow_into_lumen_by_first_cell(
            mw, wall, pos, ei, crit, u, v, wall_clot=wall_msk)
        off_gt_fc = grow_into_lumen_by_first_cell(mat_gt, wall, pos, ei, crit, u, v)
        preds = {
            "wall-only": wall_msk,
            "speed (shipped)": speed,
            "mat+fill att": wall_msk | off_att,
            "mat+fill flux": wall_msk | off_flux,
            "mat+fill convect": wall_msk | off_conv,
            "tds2": wall_msk | off_tds2,
            "tds2 char": wall_msk | off_tds2_char,
            "first-cell": wall_msk | off_fc,
            "first-cell committed": wall_msk | off_fc_c,
            "speed AND flux": wall_msk | (off_speed & off_flux),
            "speed AND first-cell": wall_msk | (off_speed & off_fc),
            "speed OR tds2": wall_msk | off_speed | off_tds2,
            "speed OR att": wall_msk | off_speed | off_att,
            "speed OR first-cell": wall_msk | off_speed | off_fc,
            "oracle att*GT": wall_msk | off_gt_att,
            "oracle flux*GT": wall_msk | off_gt_flux,
            "oracle convect*GT": wall_msk | off_gt_conv,
            "oracle tds2*GT": wall_msk | off_gt_tds2,
            "oracle tds2 char*GT": wall_msk | off_gt_tds2_char,
            "oracle first-cell*GT": wall_msk | off_gt_fc,
            "oracle Mat_self": wall_msk | off_gt_self,
            "oracle GT field": gt,
        }
        if args.coupled:
            mat_c = coupled_mat_field(d, bio, f, wall, pos)
            mc = fill_grown_wall_mat(mat_c, wall_msk, wall, A, hops=LUMEN_MAT_FILL_HOPS)
            wall_c = wall_msk | ((mat_c >= crit) & wall)
            off_cf = grow_into_lumen_by_flux(mc, wall, pos, ei, crit, u, v)
            preds["coupled flux"] = wall_c | off_cf
            preds["coupled speed|flux"] = wall_c | off_speed | off_cf

        metrics = {}
        for name, pred in preds.items():
            metrics[name] = dict(
                full=domain_score(pred, gt, d.edge_index, np.ones(len(wall), dtype=bool), wall),
                wall=domain_score(pred, gt, d.edge_index, wall, wall),
                off=domain_score(pred, gt, d.edge_index, ~wall, wall),
                wall_f1=f1(pred & wall, gt & wall),
                off_f1=f1(pred & ~wall, gt & ~wall),
            )
        rows.append(dict(anchor=anchor, split=split_of(anchor),
                         n_gt=int(gt.sum()), n_off=int((gt & ~wall).sum()),
                         metrics=metrics))
        m0 = metrics["speed (shipped)"]
        print("%-12s %-4s nGT %4d off %3d | wall %.3f  off %.3f  full %.3f"
              % (anchor, split_of(anchor), int(gt.sum()), int((gt & ~wall).sum()),
                 m0["wall"], m0["off"] if m0["off"] == m0["off"] else float("nan"),
                 m0["full"]))

    by = {}
    for r in rows:
        by.setdefault(r["split"], []).append(r["anchor"])
    print("[i] eligible  FIT n=%d  DEV n=%d"
          % (len(by.get("fit", [])), len(by.get("dev", []))))
    if skipped:
        print("[i] dropped:")
        for a, sp, why in skipped:
            print("    %-12s %-6s %s" % (a, sp, why))

    arms = list(rows[0]["metrics"].keys()) if rows else []
    print("\n=== DOMAIN SCORES (select on DEV; SEALED closed) ===")
    print("%-22s %s" % ("arm", "FIT wall / off          DEV wall / off"))
    summary = {}
    for arm in arms:
        wall_s = {r["anchor"]: r["metrics"][arm]["wall"] for r in rows}
        off_s = {r["anchor"]: r["metrics"][arm]["off"] for r in rows}
        full_s = {r["anchor"]: r["metrics"][arm]["full"] for r in rows}
        wf1 = {r["anchor"]: r["metrics"][arm]["wall_f1"] for r in rows}
        of1 = {r["anchor"]: r["metrics"][arm]["off_f1"] for r in rows}
        print("%-22s wall %s" % (arm, format_split_means(wall_s)))
        print("%-22s off  %s" % ("", format_split_means(off_s)))
        print("%-22s full %s" % ("", format_split_means(full_s)))
        summary[arm] = dict(wall=wall_s, off=off_s, full=full_s, wall_f1=wf1, off_f1=of1)

    print("\n=== vs targets (FIT mean, DEV mean) ===")
    if "speed (shipped)" in summary:
        base_w, base_o = summary["speed (shipped)"]["wall"], summary["speed (shipped)"]["off"]
        print("   shipped  wall %s" % format_split_means(base_w))
        print("            off  %s" % format_split_means(base_o))
    freeze = []
    for arm in arms:
        if arm == "speed (shipped)" or arm.startswith("oracle"):
            continue
        dw = {a: summary[arm]["wall"][a] - summary["speed (shipped)"]["wall"][a]
              for a in summary[arm]["wall"]}
        do = {a: summary[arm]["off"][a] - summary["speed (shipped)"]["off"][a]
              for a in summary[arm]["off"]
              if summary[arm]["off"][a] == summary[arm]["off"][a]
              and summary["speed (shipped)"]["off"][a] == summary["speed (shipped)"]["off"][a]}
        print("   %-20s wall %s" % (arm, format_split_means(dw)))
        print("   %-20s off  %s" % ("", format_split_means(do)))
        mw, mo = mean_by_split(dw), mean_by_split(do)
        mf = mean_by_split({a: summary[arm]["full"][a] - summary["speed (shipped)"]["full"][a]
                            for a in summary[arm]["full"]})
        wall_up = (mw["fit"]["mean"] or 0) > 1e-6 and (mw["dev"]["mean"] or 0) > 1e-6
        off_up = (mo["fit"]["mean"] or 0) > 1e-6 and (mo["dev"]["mean"] or 0) > 1e-6
        full_ok = (mf["fit"]["mean"] or 0) > -1e-4 and (mf["dev"]["mean"] or 0) > -1e-4
        if wall_up and full_ok:
            freeze.append((arm, "wall", mw["fit"]["mean"], mw["dev"]["mean"]))
        if off_up and full_ok:
            freeze.append((arm, "off", mo["fit"]["mean"], mo["dev"]["mean"]))
    if freeze:
        print("   same-sign FIT+DEV gains (do not open SEALED):")
        for arm, dom, df, dd in freeze:
            print("      %s %s  FIT %+.4f  DEV %+.4f" % (arm, dom, df, dd))
    else:
        print("   no deployable arm improves both FIT and DEV on a domain score.")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        flow=args.flow, coupled=bool(args.coupled),
        targets=dict(wall=WALL_TARGET, off=OFF_TARGET),
        eligible={s: by.get(s, []) for s in ("fit", "dev")},
        skipped=[list(x) for x in skipped],
        per={r["anchor"]: r for r in rows},
    ), indent=2, default=float))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
