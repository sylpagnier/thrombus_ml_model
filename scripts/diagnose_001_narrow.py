"""Quick narrowing ladder for patient001 hop_ge2 hard-zero.

Runs in ~minutes (no multi-epoch ladder). Goal: separate feature-path bug
from deploy-route / threshold issues and point at the fix.

Steps:
  A) band vs global feature cosine on lumen/wall (001 and 007)
  B) locked wall-only deploy on band static (sanity)
  C) signs-of-life mini-train band (40 steps) -> compound + growth-alone deploy
  D) same mini-train global -> compound deploy (expect still dead)

Usage:
  python scripts/diagnose_001_narrow.py
  python scripts/diagnose_001_narrow.py --steps 60
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

from scripts.diagnose_001_signs_of_life import (  # noqa: E402
    _build_late_tile,
    _lumen_delta_stats,
    _mini_train,
)
from src.biochem_gnn.config import apply_deploy_env  # noqa: E402
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig, VesselConfig  # noqa: E402
from src.core_physics.clot_phi_simple import _wall_mask_from_data  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    compute_hop_distances,
    deploy_eval_time_index,
    eval_deploy_clot_f1,
    load_continuous_bundle,
    noisy_teacher_log_state0,
    save_continuous_checkpoint,
    train_deploy_eval_flow_source,
)
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.species_snapshot_gnn import snapshot_wall_hops  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.inference.corrector_coupling import resolve_kinematics_checkpoint  # noqa: E402
from src.training.train_offwall_growth import (  # noqa: E402
    _band_static_to_device,
    build_global_base_features,
    freeze_growth_backbone,
)
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics_and_latent,
)
from src.utils.paths import get_project_root  # noqa: E402


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return float("nan")
    a2 = a.reshape(a.shape[0], -1).float()
    b2 = b.reshape(b.shape[0], -1).float()
    # match widths (band may have extra cols vs truncated global)
    w = min(a2.shape[1], b2.shape[1])
    a2, b2 = a2[:, :w], b2[:, :w]
    return float(F.cosine_similarity(a2, b2, dim=1).mean().item())


def _feat_alignment(anchor: str, data, kine, device, phys) -> dict:
    with torch.no_grad():
        _uv, z = predict_kinematics_and_latent(kine, data)
        band = build_band_base_features(
            data, kine, device, wall_hops=snapshot_wall_hops(), z_kin_override=z
        )
        glob = build_global_base_features(data, kine, device)
    node_idx = band["node_idx"].long()
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    hop = compute_hop_distances(data.edge_index.to(device), wall, n)
    t_dep = int(deploy_eval_time_index(int(data.y.shape[0])))
    phi = gt_clot_phi_at_time(data, t_dep, phys, device).reshape(-1)
    clot = phi >= 0.5

    band_feats = band["base_feats"]
    glob_on_band = glob[node_idx]
    hop_b = hop[node_idx]
    wall_b = wall[node_idx]
    clot_b = clot[node_idx]
    lumen = (hop_b >= 2) & (~wall_b)
    lumen_gt = lumen & clot_b
    wall_nodes = wall_b

    # Kin-only block = cols before SDF (latent_dim); SDF is 1 col after.
    lat = int(band.get("latent_dim") or max(band_feats.shape[1] - 1, 1))
    kin_b = band_feats[:, :lat]
    kin_g = glob_on_band[:, :lat]

    out = {
        "n_full": n,
        "n_band": int(node_idx.numel()),
        "feat_dim_band": int(band_feats.shape[1]),
        "feat_dim_global": int(glob.shape[1]),
        "cos_all_band": _cos(band_feats, glob_on_band),
        "cos_kin_all_band": _cos(kin_b, kin_g),
        "cos_lumen": _cos(band_feats[lumen], glob_on_band[lumen]),
        "cos_kin_lumen": _cos(kin_b[lumen], kin_g[lumen]),
        "cos_lumen_gt": _cos(band_feats[lumen_gt], glob_on_band[lumen_gt]),
        "cos_wall": _cos(band_feats[wall_nodes], glob_on_band[wall_nodes]),
        "n_lumen_band": int(lumen.sum().item()),
        "n_lumen_gt_band": int(lumen_gt.sum().item()),
        "has_kin_norm": bool(band.get("kin_mean") is not None),
    }
    print(
        f"[A] {anchor} cos_kin_lumen={out['cos_kin_lumen']:.3f} "
        f"cos_lumen={out['cos_lumen']:.3f} cos_wall={out['cos_wall']:.3f} "
        f"kin_norm={out['has_kin_norm']} "
        f"dims band/glob={out['feat_dim_band']}/{out['feat_dim_global']}",
        flush=True,
    )
    return out


def _deploy_summary(m: dict) -> dict[str, float]:
    keys = (
        "deploy_clot_f1",
        "deploy_clot_offwall_n_pred_hop_ge2",
        "deploy_clot_offwall_n_gt_hop_ge2",
        "deploy_clot_offwall_strict_f1_hop_ge2",
    )
    return {k: float(m.get(k, 0.0) or 0.0) for k in keys}


def _run_deploy(model, data, static, phys, bio, device) -> dict:
    flow_eval = train_deploy_eval_flow_source()
    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})
    return _deploy_summary(
        eval_deploy_clot_f1(
            model,
            data,
            static,
            phys,
            bio,
            device,
            time_index=deploy_eval_time_index(int(data.y.shape[0])),
            flow_source=flow_eval,
        )
    )


def _arm_mini_and_deploy(
    *,
    feat_source: str,
    wall_ckpt: Path,
    data,
    band_cpu: dict,
    glob_feats: torch.Tensor,
    phys,
    bio,
    device,
    steps: int,
    out_dir: Path,
) -> dict:
    print(f"\n[i] === arm feat_source={feat_source} ===", flush=True)
    clear_offwall_model_cache()
    bundle = load_continuous_bundle(wall_ckpt, device=device, quiet=True, architecture="dual")
    assert bundle is not None
    model = bundle.model
    n_fr, n_tr = freeze_growth_backbone(model)
    print(f"[i] freeze-backbone frozen={n_fr} trainable={n_tr}", flush=True)

    tile = _build_late_tile(
        data=data,
        pack_band=band_cpu,
        pack_global_feats=glob_feats,
        pack_global_flow=None,
        pack_global_flow_cols=None,
        phys=phys,
        device=device,
        feat_source=feat_source,
        hops_k=5,
        frontier_hops=2,
        unroll=8,
    )
    print(
        f"[i] tile mask={tile['n_mask']} lumen={tile['n_lumen_in_tile']} "
        f"win={tile['win'][0]}..{tile['win'][-1]}",
        flush=True,
    )

    model.eval()
    pre = _lumen_delta_stats(
        model,
        base_feats=tile["base_feats"],
        edge_index=tile["edge_index"],
        log_state0=noisy_teacher_log_state0(
            tile["series"][0], tile["edge_index"], training=False
        ),
        wall_mask=tile["wall_mask"],
        pos=tile["pos"],
        hop=tile["hop"],
        species0=tile["species_block"][0],
        vel0=tile["velocity"][0],
        flow_series=tile["flow_series"],
        flow_cols=tile["flow_cols"],
        t0=int(tile["win"][0]),
    )
    losses = _mini_train(
        model, tile, steps=steps, lr=3e-4, lumen_w=10.0, device=device
    )
    loss0 = next((x for x in losses if x == x), float("nan"))
    loss1 = next((x for x in reversed(losses) if x == x), float("nan"))
    model.eval()
    post = _lumen_delta_stats(
        model,
        base_feats=tile["base_feats"],
        edge_index=tile["edge_index"],
        log_state0=noisy_teacher_log_state0(
            tile["series"][0], tile["edge_index"], training=False
        ),
        wall_mask=tile["wall_mask"],
        pos=tile["pos"],
        hop=tile["hop"],
        species0=tile["species_block"][0],
        vel0=tile["velocity"][0],
        flow_series=tile["flow_series"],
        flow_cols=tile["flow_cols"],
        t0=int(tile["win"][0]),
    )
    print(
        f"[i] fire {pre['n_fire_gt_thr']:.0f}->{post['n_fire_gt_thr']:.0f} "
        f"loss {loss0:.4f}->{loss1:.4f}",
        flush=True,
    )

    tmp = out_dir / f"_narrow_growth_{feat_source}.pth"
    save_continuous_checkpoint(
        tmp,
        model,
        {"narrow": True, "train_feat_source": feat_source},
    )

    static = _band_static_to_device(band_cpu, device)
    data_g = data.to(device)

    # Compound wall-route
    clear_offwall_model_cache()
    os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
    os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(tmp)
    os.environ["SPECIES_TWO_MODEL_ROUTE"] = "wall"
    wall_b = load_continuous_bundle(wall_ckpt, device=device, quiet=True)
    assert wall_b is not None
    compound = _run_deploy(wall_b.model, data_g, static, phys, bio, device)
    print(
        f"[C] compound_{feat_source}: clot_f1={compound['deploy_clot_f1']:.3f} "
        f"hop_ge2={compound['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
        f"{compound['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
        flush=True,
    )

    # Growth-alone
    clear_offwall_model_cache()
    os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
    os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
    growth_b = load_continuous_bundle(tmp, device=device, quiet=True)
    assert growth_b is not None
    alone = _run_deploy(growth_b.model, data_g, static, phys, bio, device)
    print(
        f"[C] growth_alone_{feat_source}: clot_f1={alone['deploy_clot_f1']:.3f} "
        f"hop_ge2={alone['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
        f"{alone['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
        flush=True,
    )

    # Frontier route compound
    clear_offwall_model_cache()
    os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
    os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(tmp)
    os.environ["SPECIES_TWO_MODEL_ROUTE"] = "frontier"
    os.environ["SPECIES_TWO_MODEL_FRONTIER_HOPS"] = "2"
    wall_b2 = load_continuous_bundle(wall_ckpt, device=device, quiet=True)
    assert wall_b2 is not None
    frontier = _run_deploy(wall_b2.model, data_g, static, phys, bio, device)
    print(
        f"[C] frontier_{feat_source}: clot_f1={frontier['deploy_clot_f1']:.3f} "
        f"hop_ge2={frontier['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
        f"{frontier['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
        flush=True,
    )

    os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
    os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
    os.environ.pop("SPECIES_TWO_MODEL_ROUTE", None)
    clear_offwall_model_cache()

    return {
        "n_mask": tile["n_mask"],
        "loss_start": loss0,
        "loss_end": loss1,
        "pre_fire": pre["n_fire_gt_thr"],
        "post_fire": post["n_fire_gt_thr"],
        "mean_abs_post": post["mean_abs_per_lumen_step"],
        "compound": compound,
        "growth_alone": alone,
        "frontier": frontier,
        "ckpt": str(tmp),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument(
        "--wall-ckpt",
        default="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    )
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse")
    ap.add_argument(
        "--out",
        default="outputs/biochem/offwall_model/wc_v7_crack_001_3h/diagnose_narrow.json",
    )
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    wall_ckpt = Path(args.wall_ckpt)
    if not wall_ckpt.is_absolute():
        wall_ckpt = root / wall_ckpt
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    apply_mat_growth_leg_env(args.mat_leg, force=True)
    os.environ["SPECIES_LUMEN_SHAPE_FN_W"] = "25"
    os.environ["SPECIES_LUMEN_SHAPE_FP_W"] = "0.35"
    os.environ["SPECIES_CONTINUOUS_UNDERPRED_WEIGHT"] = "12.0"
    os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
    os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    graph_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir

    print("=" * 72, flush=True)
    print(f"NARROW 001 (steps={args.steps})", flush=True)
    print("=" * 72, flush=True)

    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()),
        device,
        phys_cfg=PhysicsConfig(phase="kinematics"),
    )

    report: dict = {"steps": args.steps, "feat_alignment": {}, "arms": {}}

    # --- A: feature alignment 001 vs 007 ---
    print("\n[A] Feature alignment (band vs global on overlapping nodes)", flush=True)
    for anc in ("patient001", "patient007"):
        data_a = torch.load(graph_dir / f"{anc}.pt", map_location="cpu", weights_only=False)
        report["feat_alignment"][anc] = _feat_alignment(anc, data_a, kine, device, phys)

    # Prepare 001 packs for mini-train / deploy
    data = torch.load(graph_dir / "patient001.pt", map_location="cpu", weights_only=False)
    with torch.no_grad():
        pred_uv, z = predict_kinematics_and_latent(kine, data)
    data.u0_pred = pred_uv[:, 0].detach().cpu()
    data.v0_pred = pred_uv[:, 1].detach().cpu()
    band = build_band_base_features(
        data, kine, device, wall_hops=snapshot_wall_hops(), z_kin_override=z
    )
    band_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in band.items()}
    glob_feats = build_global_base_features(data, kine, device).cpu()

    # --- B: wall-only baseline on band ---
    print("\n[B] Wall-only deploy on band static (patient001)", flush=True)
    clear_offwall_model_cache()
    wall_b = load_continuous_bundle(wall_ckpt, device=device, quiet=True)
    assert wall_b is not None
    static = _band_static_to_device(band_cpu, device)
    wall_only = _run_deploy(wall_b.model, data.to(device), static, phys, bio, device)
    report["wall_only_001"] = wall_only
    print(
        f"[B] wall_only clot_f1={wall_only['deploy_clot_f1']:.3f} "
        f"hop_ge2={wall_only['deploy_clot_offwall_n_pred_hop_ge2']:.0f}/"
        f"{wall_only['deploy_clot_offwall_n_gt_hop_ge2']:.0f}",
        flush=True,
    )

    # --- C/D: band then global mini-train + deploy ---
    for src in ("band", "global"):
        report["arms"][src] = _arm_mini_and_deploy(
            feat_source=src,
            wall_ckpt=wall_ckpt,
            data=data,
            band_cpu=band_cpu,
            glob_feats=glob_feats,
            phys=phys,
            bio=bio,
            device=device,
            steps=int(args.steps),
            out_dir=out.parent,
        )

    b = report["arms"]["band"]
    g = report["arms"]["global"]
    b_ge2 = b["compound"]["deploy_clot_offwall_n_pred_hop_ge2"]
    g_ge2 = g["compound"]["deploy_clot_offwall_n_pred_hop_ge2"]
    b_fire = b["post_fire"]
    cos001 = report["feat_alignment"]["patient001"]["cos_kin_lumen"]
    cos007 = report["feat_alignment"]["patient007"]["cos_kin_lumen"]

    if b_ge2 > 0.5 and g_ge2 < 0.5:
        verdict = "fix_confirmed_train_on_band_feats"
    elif b_fire > 50 and b_ge2 < 0.5:
        if b["frontier"]["deploy_clot_offwall_n_pred_hop_ge2"] > 0.5:
            verdict = "tile_life_but_wall_route_blocks_use_frontier_or_longer_train"
        elif b["growth_alone"]["deploy_clot_offwall_n_pred_hop_ge2"] > 0.5:
            verdict = "tile_life_compound_blend_blocks_growth_alone_ok"
        else:
            verdict = "tile_life_but_deploy_threshold_or_nucleation_gap"
    elif cos001 < 0.5 and cos007 > 0.7:
        verdict = "001_feat_mismatch_severe_vs_007_retrain_band"
    elif b_fire < 1 and g["post_fire"] < 1:
        verdict = "dead_everywhere_wiring"
    else:
        verdict = "partial_need_longer_band_train"

    report["verdict"] = verdict
    report["fix_hint"] = {
        "preferred": "train_feat_source=band (already default in crack launcher)",
        "cos_kin_lumen_001": cos001,
        "cos_kin_lumen_007": cos007,
        "band_compound_hop_ge2": b_ge2,
        "global_compound_hop_ge2": g_ge2,
    }

    print("\n" + "=" * 72, flush=True)
    print(f"[i] verdict={verdict}", flush=True)
    print(
        f"[i] fix_hint band_ge2={b_ge2:.0f} global_ge2={g_ge2:.0f} "
        f"cos001={cos001:.3f} cos007={cos007:.3f}",
        flush=True,
    )
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
