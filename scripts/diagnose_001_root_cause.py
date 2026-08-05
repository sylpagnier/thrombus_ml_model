"""Autonomous root-cause hunt for patient001 hop_ge2 hard-zero.

Hypotheses (most -> least likely):
  H1  vel-decay / lumen flow wipe (ablations)
  H3  graph/data integrity
  H2  teacher vs resting step-feature shift
  H4  compare 001 vs 007 lumen speed / Mat GT

Usage:
  python scripts/diagnose_001_root_cause.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.clot_growth_masks import graph_dilate_hops  # noqa: E402
from src.core_physics.clot_phi_simple import (  # noqa: E402
    _wall_mask_from_data,
    mat_si_for_gelation_from_log1p,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    build_continuous_step_features,
    clear_offwall_model_cache,
    compute_hop_distances,
    continuous_mat_commit_thresh,
    deploy_eval_time_index,
    eval_deploy_clot_f1,
    load_continuous_bundle,
    train_deploy_eval_flow_source,
)
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    induced_subgraph,
    snapshot_wall_hops,
    wall_band_mask,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.inference.corrector_coupling import resolve_kinematics_checkpoint  # noqa: E402
from src.training.biochem_species_scope import (  # noqa: E402
    MAT_CHANNEL,
    pushforward_local_index,
    pushforward_state_bulk_indices,
)
from src.training.train_offwall_growth import _band_static_to_device  # noqa: E402
from src.utils import species_channels as sc  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
)
from src.utils.paths import get_project_root  # noqa: E402


def _mat_idx() -> int:
    try:
        return int(pushforward_local_index("mat"))
    except Exception:
        return 0


def _summarize(m: dict) -> dict[str, float]:
    keys = (
        "deploy_clot_f1",
        "deploy_clot_offwall_n_pred",
        "deploy_clot_offwall_n_pred_hop_ge2",
        "deploy_clot_offwall_n_gt_hop_ge2",
    )
    return {k: float(m.get(k, 0.0) or 0.0) for k in keys}


@torch.no_grad()
def audit_graph(anchor: str, data, device, phys, bio) -> dict:
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    hop = compute_hop_distances(data.edge_index.to(device), wall, n)
    t_dep = int(deploy_eval_time_index(int(data.y.shape[0])))
    phi = gt_clot_phi_at_time(data, t_dep, phys, device).reshape(-1)
    clot = phi >= 0.5
    lumen = (hop >= 2) & (~wall)
    lumen_gt = lumen & clot
    wall_clot = wall & clot

    y = data.y[t_dep].to(device=device, dtype=torch.float32)
    mat_log = y[:, sc.SPECIES_BLOCK][:, int(MAT_CHANNEL)].reshape(-1)
    mat_si = mat_si_for_gelation_from_log1p(mat_log, bio)
    gel_need = float(bio.viscosity_mat_crit)

    band_mask = wall_band_mask(data, device, wall_hops=snapshot_wall_hops())
    band_nodes, _, _ = induced_subgraph(band_mask, data.edge_index)
    band_nodes = band_nodes.long()
    on_band = torch.zeros(n, dtype=torch.bool, device=device)
    on_band[band_nodes] = True
    lumen_gt_in_band = int((lumen_gt & on_band).sum().item())

    orphan = 0
    if lumen_gt.any() and wall_clot.any():
        reach = wall_clot.clone()
        for _ in range(12):
            reach = graph_dilate_hops(reach, data.edge_index.to(device), 1) | reach
        orphan = int((lumen_gt & (~reach)).sum().item())

    speed = torch.linalg.vector_norm(y[:, 0:2], dim=-1)
    out = {
        "n_nodes": n,
        "n_times": int(data.y.shape[0]),
        "n_wall": int(wall.sum().item()),
        "n_lumen": int(lumen.sum().item()),
        "n_lumen_gt": int(lumen_gt.sum().item()),
        "n_wall_clot": int(wall_clot.sum().item()),
        "n_band": int(band_nodes.numel()),
        "lumen_gt_in_band": lumen_gt_in_band,
        "lumen_gt_orphan_from_wall_clot": orphan,
        "mat_log_lumen_gt_mean": float(mat_log[lumen_gt].mean()) if lumen_gt.any() else 0.0,
        "mat_log_lumen_gt_max": float(mat_log[lumen_gt].max()) if lumen_gt.any() else 0.0,
        "mat_si_lumen_gt_max": float(mat_si[lumen_gt].max()) if lumen_gt.any() else 0.0,
        "mat_si_above_crit_lumen": int((mat_si[lumen_gt] >= gel_need).sum()) if lumen_gt.any() else 0,
        "speed_lumen_gt_mean": float(speed[lumen_gt].mean()) if lumen_gt.any() else 0.0,
        "speed_lumen_gt_p90": float(torch.quantile(speed[lumen_gt].float(), 0.9)) if lumen_gt.any() else 0.0,
        "speed_wall_clot_mean": float(speed[wall_clot].mean()) if wall_clot.any() else 0.0,
        "gelation_mat_crit": gel_need,
        "mat_commit_thresh": continuous_mat_commit_thresh(),
        "gelation_log1p_approx": float(torch.log1p(torch.tensor(gel_need / float(bio.Minf)))),
    }
    print(
        f"[H3] {anchor}: lumen_gt={out['n_lumen_gt']} in_band={lumen_gt_in_band}/{out['n_lumen_gt']} "
        f"orphan={orphan} mat_log_max={out['mat_log_lumen_gt_max']:.4e} "
        f"speed_p90={out['speed_lumen_gt_p90']:.4f} wall_speed={out['speed_wall_clot_mean']:.4f}",
        flush=True,
    )
    return out


@torch.no_grad()
def teacher_vs_rest_feats(anchor: str, data, static, device) -> dict:
    node_idx = static["node_idx"].long()
    base = static["base_feats"]
    edge = static["edge_index"]
    wall_m = static["wall_mask_band"].bool()
    pos = static.get("pos_band")
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data.to(device), device, n)
    hop = compute_hop_distances(data.edge_index.to(device), wall, n)[node_idx]
    lumen = (hop >= 2) & (~wall_m)
    t_dep = int(deploy_eval_time_index(int(data.y.shape[0])))
    t0 = max(0, t_dep - 8)

    bulk = pushforward_state_bulk_indices()
    y = data.y[t0].to(device=device, dtype=torch.float32)
    sp = y[:, sc.SPECIES_BLOCK]
    teacher = torch.stack([sp[:, int(c)] for c in bulk], dim=-1)[node_idx]
    resting = torch.zeros_like(teacher)
    vel = y[node_idx, 0:2].contiguous()

    def _feats(log_state):
        return build_continuous_step_features(
            base,
            log_state,
            training=False,
            time_index=t0 + 1,
            velocity=vel,
            pos_band=pos,
            edge_index=edge,
        )

    if not lumen.any():
        return {"anchor": anchor, "cos_step_feats_lumen_teacher_vs_rest": float("nan")}
    cos = float(F.cosine_similarity(_feats(teacher)[lumen], _feats(resting)[lumen], dim=-1).mean())
    midx = _mat_idx()
    out = {
        "anchor": anchor,
        "cos_step_feats_lumen_teacher_vs_rest": cos,
        "teacher_mat_lumen_mean": float(teacher[lumen, midx].mean()),
        "t0": t0,
    }
    print(
        f"[H2] {anchor}: cos(teacher_vs_rest)={cos:.4f} teacher_mat={out['teacher_mat_lumen_mean']:.4e}",
        flush=True,
    )
    return out


def run_ablations(wall_ckpt: Path, growth_ckpt: Path, data, static, phys, bio, device) -> dict:
    cases = [
        ("baseline", {}),
        ("vel_decay_off", {"SPECIES_CONTINUOUS_VEL_DECAY": "0"}),
        ("flow_gt", {"SPECIES_TRAIN_DEPLOY_EVAL_FLOW": "gt", "T0_R4_FLOW_SOURCE": "gt"}),
        (
            "vel_decay_off_and_flow_gt",
            {
                "SPECIES_CONTINUOUS_VEL_DECAY": "0",
                "SPECIES_TRAIN_DEPLOY_EVAL_FLOW": "gt",
                "T0_R4_FLOW_SOURCE": "gt",
            },
        ),
        ("coupling_off", {"SPECIES_CLOSED_LOOP_COUPLING": "0"}),
        (
            "all_suppressors_off",
            {
                "SPECIES_CONTINUOUS_VEL_DECAY": "0",
                "SPECIES_CLOSED_LOOP_COUPLING": "0",
                "SPECIES_TRAIN_DEPLOY_EVAL_FLOW": "gt",
                "T0_R4_FLOW_SOURCE": "gt",
            },
        ),
    ]
    watch = [
        "SPECIES_CONTINUOUS_VEL_DECAY",
        "SPECIES_TRAIN_DEPLOY_EVAL_FLOW",
        "T0_R4_FLOW_SOURCE",
        "SPECIES_CLOSED_LOOP_COUPLING",
    ]
    prev = {k: os.environ.get(k) for k in watch}
    results = {}

    def _restore():
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _apply_overrides(overrides: dict[str, str]) -> None:
        # load_continuous_bundle(meta) force-enables vel_decay=1; re-pin after every load.
        for k, v in overrides.items():
            os.environ[k] = v

    for name, overrides in cases:
        _restore()
        _apply_overrides(overrides)
        clear_offwall_model_cache()
        os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
        os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(growth_ckpt)
        os.environ["SPECIES_TWO_MODEL_ROUTE"] = "wall"
        wall_b = load_continuous_bundle(wall_ckpt, device=device, quiet=True)
        assert wall_b is not None
        _apply_overrides(overrides)  # meta restore wipes VEL_DECAY=0
        flow = train_deploy_eval_flow_source()
        deploy_ov = {"T0_R4_FLOW_SOURCE": os.environ.get("T0_R4_FLOW_SOURCE", flow)}
        deploy_ov.update(overrides)
        apply_deploy_env(overrides=deploy_ov)
        _apply_overrides(overrides)
        flow = train_deploy_eval_flow_source()
        eff_decay = os.environ.get("SPECIES_CONTINUOUS_VEL_DECAY", "?")
        eff_flow = os.environ.get("T0_R4_FLOW_SOURCE", "?")
        eff_cpl = os.environ.get("SPECIES_CLOSED_LOOP_COUPLING", "?")
        print(
            f"[H1] run={name} eff vel_decay={eff_decay} flow={eff_flow} coupling={eff_cpl}",
            flush=True,
        )
        m = _summarize(
            eval_deploy_clot_f1(
                wall_b.model,
                data.to(device),
                static,
                phys,
                bio,
                device,
                time_index=deploy_eval_time_index(int(data.y.shape[0])),
                flow_source=flow,
            )
        )
        m["eff_vel_decay"] = eff_decay
        m["eff_flow"] = eff_flow
        m["eff_coupling"] = eff_cpl
        results[name] = m
        print(
            f"[H1] ablation={name}: f1={m['deploy_clot_f1']:.3f} "
            f"offwall={m['deploy_clot_offwall_n_pred']:.0f} "
            f"ge2={m['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
            f"{m['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
            flush=True,
        )
        # Early stop if we opened 001
        if m["deploy_clot_offwall_n_pred_hop_ge2"] > 0.5:
            print(f"[OK] 001 lumen OPENED under ablation={name}", flush=True)
            break

    _restore()
    return results


def classify(report: dict) -> str:
    h1 = report.get("ablations_001") or {}
    base = float((h1.get("baseline") or {}).get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0)

    def ge2(name: str) -> float:
        return float((h1.get(name) or {}).get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0)

    a001 = (report.get("audit") or {}).get("patient001") or {}
    a007 = (report.get("audit") or {}).get("patient007") or {}
    h2 = report.get("feat_mismatch") or {}
    c001 = float((h2.get("patient001") or {}).get("cos_step_feats_lumen_teacher_vs_rest", 1) or 1)
    c007 = float((h2.get("patient007") or {}).get("cos_step_feats_lumen_teacher_vs_rest", 1) or 1)

    if a001.get("n_lumen_gt", 0) <= 0:
        return "data_no_lumen_gt_on_001"
    if a001.get("lumen_gt_orphan_from_wall_clot", 0) > 0:
        return "data_orphan_lumen_unreachable_from_wall_clot"
    if a001.get("lumen_gt_in_band", 0) < 0.5 * max(a001.get("n_lumen_gt", 1), 1):
        return "data_lumen_gt_mostly_outside_wall_band"
    if a001.get("mat_si_above_crit_lumen", 0) <= 0:
        return "data_gt_lumen_mat_never_reaches_gelation_crit"

    if ge2("vel_decay_off") > 0.5 and base < 0.5:
        return "ROOT_vel_decay_wipes_001_lumen"
    if ge2("flow_gt") > 0.5 and base < 0.5:
        return "ROOT_kinematics_flow_suppresses_001"
    if ge2("vel_decay_off_and_flow_gt") > 0.5 and base < 0.5:
        return "ROOT_vel_decay_and_or_flow_suppress_001"
    if ge2("coupling_off") > 0.5 and base < 0.5:
        return "ROOT_closed_loop_coupling_suppresses_001"
    if ge2("all_suppressors_off") > 0.5 and base < 0.5:
        return "ROOT_combined_deploy_physics_suppresses_001"
    if any(ge2(k) > 0.5 for k in h1):
        return "ROOT_ablation_opened_001_see_which"

    if c001 + 0.25 < c007 and c001 < 0.55:
        return "suspect_001_teacher_vs_rest_feat_shift_worse_than_007"
    sp001 = float(a001.get("speed_lumen_gt_p90", 0) or 0)
    sp007 = float(a007.get("speed_lumen_gt_p90", 0) or 0)
    if sp001 > 2.0 * max(sp007, 1e-6):
        return "suspect_001_lumen_speed_much_higher_than_007_vel_decay_risk"
    if all(ge2(k) < 0.5 for k in h1) and h1:
        return "NOT_simple_vel_decay_or_flow_or_coupling_growth_head_or_state_path"
    return "inconclusive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-ckpt", default="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")
    ap.add_argument(
        "--growth-ckpt",
        default="outputs/biochem/offwall_model/wc_v7_open001_6h/growth_D_Orig10_Band/best.pth",
    )
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument("--skip-ablations", action="store_true")
    ap.add_argument(
        "--out",
        default="outputs/biochem/offwall_model/wc_v7_open001_6h/diagnose_root_cause.json",
    )
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    wall_ckpt = Path(args.wall_ckpt)
    if not wall_ckpt.is_absolute():
        wall_ckpt = root / wall_ckpt
    growth_ckpt = Path(args.growth_ckpt)
    if not growth_ckpt.is_absolute():
        growth_ckpt = root / growth_ckpt
    if not growth_ckpt.is_file():
        alt = (
            root
            / "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/growth_frontier_ge2_prec/best.pth"
        )
        if alt.is_file():
            growth_ckpt = alt
    if not growth_ckpt.is_file():
        raise FileNotFoundError(growth_ckpt)

    apply_mat_growth_leg_env(args.mat_leg, force=True)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    graph_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir

    print("=" * 72, flush=True)
    print("001 ROOT CAUSE HUNT (ranked ablations)", flush=True)
    print(f"[i] growth={growth_ckpt}", flush=True)
    print("=" * 72, flush=True)

    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()),
        device,
        phys_cfg=PhysicsConfig(phase="kinematics"),
    )

    report: dict = {
        "growth_ckpt": str(growth_ckpt),
        "audit": {},
        "feat_mismatch": {},
        "ablations_001": {},
    }
    statics = {}
    datas = {}

    print("\n=== H3 data audit + H2 feat shift ===", flush=True)
    for anc in ("patient001", "patient007"):
        data = torch.load(graph_dir / f"{anc}.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            uv, z = predict_kinematics_and_latent(kine, data)
        data.u0_pred = uv[:, 0].detach().cpu()
        data.v0_pred = uv[:, 1].detach().cpu()
        datas[anc] = data
        report["audit"][anc] = audit_graph(anc, data, device, phys, bio)
        band = build_band_base_features(
            data, kine, device, wall_hops=snapshot_wall_hops(), z_kin_override=z
        )
        statics[anc] = _band_static_to_device(
            {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in band.items()},
            device,
        )
        report["feat_mismatch"][anc] = teacher_vs_rest_feats(anc, data, statics[anc], device)

    if not args.skip_ablations:
        print("\n=== H1 ablations on patient001 compound deploy ===", flush=True)
        report["ablations_001"] = run_ablations(
            wall_ckpt,
            growth_ckpt,
            datas["patient001"],
            statics["patient001"],
            phys,
            bio,
            device,
        )

    verdict = classify(report)
    report["verdict"] = verdict
    print("\n" + "=" * 72, flush=True)
    print(f"[i] verdict={verdict}", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
