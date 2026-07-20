"""Train Phase 2.5 continuous log-delta pushforward + soft-commit memory.

Usage::

    python -m src.training.train_species_pushforward_continuous
    python -m src.training.train_species_pushforward_continuous --anchor patient007 --epochs 120
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.optim as optim

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.species_gelation_readout import (
    build_species_physics_ctx,
    continuous_physics_readout,
)
from src.biochem_gnn.config import PHASE_CKPT, apply_deploy_env, apply_train_recipe_env
from src.training.biochem_species_scope import (
    format_channel_list,
    pushforward_species_scope,
    pushforward_state_bulk_indices,
    scope_label_for_channels,
)
from src.evaluation.clot_relaxed_metrics import clot_score_from_deploy_dict, species_continuous_clout_score_mode
from src.core_physics.species_gnode_pushforward import species_pushforward_arch
from src.core_physics.species_pushforward_continuous import (
    parse_biochem_train_anchors,
    pushforward_train_t0_per_vessel,
    resolve_train_t0_max,
    species_latent_dropout_p,
    DEFAULT_S34_CKPT,
    SpeciesDualHeadContinuousGNN,
    band_speed_series,
    bind_band_geometry,
    build_continuous_gnn,
    closed_loop_init_prob,
    continuous_channel_weights,
    continuous_feature_dim,
    continuous_frontier_hops,
    continuous_gate_temp,
    continuous_mature_fp_exempt,
    continuous_nucleation_topk,
    continuous_neighbor_commit_alpha,
    continuous_neighbor_commit_gate,
    continuous_saturation_gate,
    continuous_temporal_gate,
    continuous_score_clot_weight,
    continuous_delta_residual,
    continuous_temporal_offset,
    temporal_lambda_bounds,
    continuous_delta_threshold,
    continuous_dual_head,
    continuous_final_state_all_band,
    continuous_final_state_weight,
    continuous_fp_weight,
    continuous_growth_only_loss,
    continuous_huber_beta,
    continuous_loss_scale,
    continuous_spatial_loss_weight,
    continuous_speed_fp_weight,
    continuous_teacher_blur,
    deploy_horizon_steps,
    deploy_eval_time_index,
    deploy_eval_clot_times,
    graph_last_time_index,
    legacy_capped_deploy_time_index,
    deploy_eval_dual_full_weight,
    deploy_horizon_aux_all_packs,
    deploy_horizon_aux_cap_steps,
    train_deploy_eval_flow_source,
    continuous_teacher_fp_frac,
    continuous_teacher_noise_sigma,
    continuous_vel_decay_enabled,
    curriculum_unroll_for_epoch,
    eval_continuous_window,
    eval_deploy_clot_f1,
    eval_full_rollout_fimat_f1,
    filter_continuous_windows,
    init_continuous_from_snapshot,
    init_dual_head_from_continuous,
    iter_pushforward_windows,
    mature_clot_frac,
    saturation_headroom_scale,
    load_continuous_bundle,
    load_pushforward_state_dict_partial,
    log_series_on_band,
    pushforward_feature_dim,
    pushforward_max_unroll_steps,
    pushforward_step_stride,
    pushforward_train_t0_max,
    pushforward_unroll_steps,
    pushforward_window_t0_weight,
    rollout_prefix_log_state,
    save_continuous_checkpoint,
    tbptt_tail_steps,
    unroll_continuous_loss,
)
from src.core_physics.species_pushforward_gnn import (
    build_band_base_features,
    flow_feats_drop_xy,
    flow_feats_dynamic,
    flow_feats_enabled,
    geom_feats_enabled,
    geom_feats_rich_enabled,
    pushforward_train_t0_min,
)
from src.core_physics.species_snapshot_gnn import (
    DEFAULT_SNAPSHOT_CKPT,
    kin_per_vessel_norm_enabled,
    snapshot_hidden_dim,
    snapshot_wall_hops,
)
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root
from src.utils import species_channels as sc


def _split_band_nodes(n_sub: int, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_sub, generator=g)
    n_val = max(1, int(round(n_sub * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.numel() == 0:
        train_idx = val_idx
    train_m = torch.zeros(n_sub, dtype=torch.bool)
    val_m = torch.zeros(n_sub, dtype=torch.bool)
    train_m[train_idx] = True
    val_m[val_idx] = True
    return train_m, val_m


@torch.no_grad()
def _prepare_static(data, *, device: torch.device, kine_model, wall_hops: int) -> dict:
    return build_band_base_features(data, kine_model, device, wall_hops=wall_hops)


def _release_pack_to_cpu(pack: dict) -> None:
    """PyG ``Data.to`` is in-place; move held training graphs back to CPU between packs."""
    data = pack.get("data")
    if data is not None and hasattr(data, "to"):
        pack["data"] = data.to("cpu")


def _val_windows(static: dict, *, unroll: int, stride: int) -> list[list[int]]:
    anchors = [10, 25, 28]
    wins: list[list[int]] = []
    n_times = int(static["n_times"])
    for t0 in anchors:
        win = [t0 + i * stride for i in range(unroll + 1)]
        if win[-1] < n_times:
            wins.append(win)
    return wins


def _parse_anchors(raw: str, *, all_anchors: bool, root: Path) -> list[str]:
    return parse_biochem_train_anchors(raw, all_anchors=all_anchors, root=root)


def _build_anchor_pack(
    anchor: str,
    *,
    root: Path,
    device: torch.device,
    kine_model,
    wall_hops: int,
    unroll: int,
    stride: int,
    max_windows: int,
    val_frac: float,
    seed: int,
    phys: PhysicsConfig,
    bio: BiochemConfig,
) -> dict:
    graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor.strip()}.pt"
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    # One GINO-DEQ solve per vessel: UV baseline + z_kin (joint cache). Local corrector uses UV later.
    from src.utils.kinematics_inference import predict_kinematics_and_latent

    data_dev = data.to(device)
    with torch.no_grad():
        pred_uv, z_kin = predict_kinematics_and_latent(kine_model, data_dev)
    data.u0_pred = pred_uv[:, 0].to("cpu").clone()
    data.v0_pred = pred_uv[:, 1].to("cpu").clone()

    static = build_band_base_features(
        data,
        kine_model,
        device,
        wall_hops=wall_hops,
        z_kin_override=z_kin,
    )
    train_m, val_m = _split_band_nodes(static["n_band"], val_frac, seed)
    train_m = train_m.to(device=device)
    val_m = val_m.to(device=device)
    windows = iter_pushforward_windows(static["n_times"], unroll=unroll, stride=stride)
    pack_t0_max = resolve_train_t0_max(int(static["n_times"]))
    windows = filter_continuous_windows(
        windows, data, static["node_idx"], device, t0_max=pack_t0_max, min_delta_mag=1e-8
    )
    if max_windows > 0:
        windows = windows[: int(max_windows)]

    # Move all static features, masks, and graphs to CPU to avoid GPU OOM
    static_cpu = {}
    for k, v in static.items():
        if isinstance(v, torch.Tensor):
            static_cpu[k] = v.to("cpu")
        else:
            static_cpu[k] = v
    train_m = train_m.to("cpu")
    val_m = val_m.to("cpu")
    data = data.to("cpu")

    return {
        "anchor": anchor.strip(),
        "data": data,
        "static": static_cpu,
        "train_m": train_m,
        "val_m": val_m,
        "windows": windows,
        "train_t0_max": pack_t0_max,
        "val_windows": _val_windows(static_cpu, unroll=unroll, stride=stride),
        "phys": phys,
        "bio": bio,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train species continuous pushforward (baseline biochem_gnn)")
    ap.add_argument(
        "--phase",
        choices=("biochem_gnn", "clot_deploy_gnn"),
        default="biochem_gnn",
        help="canonical deploy baseline GNN",
    )
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchors", default="", help="Comma-separated anchors for multi-vessel train")
    ap.add_argument("--all-anchors", action="store_true", help="Train on all biochem anchor graphs on disk")
    ap.add_argument("--val-anchor", default="patient007", help="Holdout anchor for val logging")
    ap.add_argument(
        "--exclude-val-from-train",
        action="store_true",
        help="LOAO: drop val-anchor from training packs (train only on other vessels)",
    )
    ap.add_argument("--init", default="", help="Optional checkpoint to warm-start")
    ap.add_argument(
        "--no-init",
        action="store_true",
        help="Random init (skip default snapshot / continuous warm-start)",
    )
    ap.add_argument(
        "--init-mode",
        choices=("full", "backbone", "mat_readout"),
        default="full",
        help="Warm-start policy when --init is a fi_mat dual-head ckpt (mat recipe)",
    )
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unroll", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--wall-hops", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--init-s1", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--early-stop", type=int, default=25)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument(
        "--recipe",
        choices=("default", "mat_growth_simple"),
        default="default",
        help="Training env recipe (mat_growth_simple = Mat-only single-head)",
    )
    mat_leg_choices = ("",)
    try:
        from src.biochem_gnn.mat_growth_simple import LADDER_LEG_ORDER

        mat_leg_choices = ("", *tuple(LADDER_LEG_ORDER))
    except Exception:
        # Keep trainer usable even if mat-growth helper import fails.
        mat_leg_choices = ("", "A_random", "B_backbone", "C_geom", "D_parity_single", "E_dual_mat", "F_single_fimat")
    ap.add_argument(
        "--leg",
        choices=mat_leg_choices,
        default="",
        help="Mat-growth ladder leg (applies per-leg env overrides)",
    )
    ap.add_argument(
        "--arch",
        choices=("sage", "gnode"),
        default="",
        help="Pushforward trunk: sage=GraphSAGE (default), gnode=GINO derivative",
    )
    args = ap.parse_args()

    if args.arch.strip():
        os.environ["SPECIES_PUSHFORWARD_ARCH"] = str(args.arch).strip().lower()
    pushforward_arch = species_pushforward_arch()

    phase = "biochem_gnn"
    if str(args.recipe).strip().lower() == "mat_growth_simple":
        from src.biochem_gnn.mat_growth_simple import (
            apply_mat_growth_leg_env,
            apply_mat_growth_simple_recipe_env,
            mat_growth_precision_selection_enabled,
        )

        if str(args.leg).strip():
            apply_mat_growth_leg_env(str(args.leg).strip(), force=True)
        else:
            apply_mat_growth_simple_recipe_env(force=True)
    else:
        apply_train_recipe_env()
    mat_growth_recipe = str(args.recipe).strip().lower() == "mat_growth_simple"
    mat_precision_select = False
    if mat_growth_recipe:
        from src.biochem_gnn.mat_growth_simple import mat_growth_precision_selection_enabled

        mat_precision_select = mat_growth_precision_selection_enabled()
    if args.unroll is None and not (os.environ.get("SPECIES_PUSHFORWARD_UNROLL") or "").strip():
        os.environ["SPECIES_PUSHFORWARD_UNROLL"] = "10"
    if args.unroll is not None:
        os.environ["SPECIES_PUSHFORWARD_UNROLL"] = str(args.unroll)
    if args.stride is not None:
        os.environ["SPECIES_PUSHFORWARD_STEP_STRIDE"] = str(args.stride)
    if args.wall_hops is not None:
        os.environ["SPECIES_SNAPSHOT_WALL_HOPS"] = str(args.wall_hops)
    if not (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL") or "").strip():
        os.environ["SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL"] = "1"

    unroll = pushforward_unroll_steps()
    max_unroll = pushforward_max_unroll_steps()
    stride = pushforward_step_stride()
    wall_hops = snapshot_wall_hops()
    hidden = snapshot_hidden_dim() if args.hidden is None else max(int(args.hidden), 16)
    ch_w = continuous_channel_weights()
    huber_b = continuous_huber_beta()
    t0_max = None if pushforward_train_t0_per_vessel() else pushforward_train_t0_max()
    growth_only = continuous_growth_only_loss()
    loss_scale = continuous_loss_scale()
    delta_thr = continuous_delta_threshold()
    fp_w = continuous_fp_weight()
    physics_on = continuous_physics_readout()
    dual_head = continuous_dual_head()
    phase_tag = PHASE_CKPT
    default_out = DEFAULT_S34_CKPT
    lr = float(args.lr) if args.lr is not None else (3e-4 if growth_only else 1e-3)
    grad_clip = (
        float(args.grad_clip)
        if args.grad_clip is not None
        else float(os.environ.get("SPECIES_CONTINUOUS_GRAD_CLIP", "1.0" if growth_only else "0") or "0")
    )
    grad_clip = max(grad_clip, 0.0)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from src.core_physics.t0_device import require_cuda_device
    device = require_cuda_device()
    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    train_anchors = _parse_anchors(args.anchors or args.anchor, all_anchors=bool(args.all_anchors), root=root)
    val_anchor = args.val_anchor.strip() or train_anchors[0]
    if bool(args.exclude_val_from_train):
        train_anchors = [a for a in train_anchors if a.strip() != val_anchor]
        if not train_anchors:
            raise ValueError(f"exclude-val-from-train left no train anchors (val={val_anchor})")

    kine_ckpt = str(resolve_kinematics_checkpoint())
    kine_model = load_kinematics_predictor(
        kine_ckpt, device, phys_cfg=PhysicsConfig(phase="kinematics"), cache=False
    )

    packs: list[dict] = []
    for anc in train_anchors:
        packs.append(
            _build_anchor_pack(
                anc,
                root=root,
                device=device,
                kine_model=kine_model,
                wall_hops=wall_hops,
                unroll=unroll,
                stride=stride,
                max_windows=int(args.max_windows),
                val_frac=float(args.val_frac),
                seed=int(args.seed),
                phys=phys,
                bio=bio,
            )
        )
        import gc
        gc.collect()

    # Free the large kinematics model from GPU VRAM now that dataset loading is complete
    from src.utils.kinematics_inference import clear_kinematics_predictor_cache

    clear_kinematics_predictor_cache()
    del kine_model
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    val_pack = next((p for p in packs if p["anchor"] == val_anchor), packs[0])
    ref_static = packs[0]["static"]
    latent_dim = int(ref_static["base_feats"].shape[1] - 1)
    prev_in_dim = pushforward_feature_dim(latent_dim)
    in_dim = continuous_feature_dim(latent_dim)
    model = build_continuous_gnn(in_dim, hidden=hidden, arch=pushforward_arch).to(device)
    # Latent leash: tell the model the z_kin slice width + dropout prob so the training forward can
    # stochastically zero the (clot-blind) latent and force reliance on the explicit flow features.
    model.kin_latent_dim = int(ref_static.get("latent_dim", 0) or 0)
    model.latent_dropout_p = species_latent_dropout_p()
    if model.latent_dropout_p > 0.0:
        print(
            f"[i] latent leash: dropout p={model.latent_dropout_p:.2f} on z_kin[:{model.kin_latent_dim}]",
            flush=True,
        )

    init_ckpt = "" if bool(args.no_init) else (args.init.strip() or str(root / DEFAULT_S34_CKPT))
    if bool(args.no_init):
        print("[i] random init (--no-init)", flush=True)
    elif pushforward_arch == "gnode":
        snap_path = args.init_s1.strip() or str(root / DEFAULT_SNAPSHOT_CKPT)
        if Path(snap_path).is_file():
            from src.core_physics.species_snapshot_gnn import load_snapshot_bundle

            snap = load_snapshot_bundle(snap_path, device=device, quiet=True)
            if snap is not None:
                init_dual_head_from_continuous(model, snap.model, quiet=False)
                print(f"[OK] gnode dual-head warm-start from snapshot {snap_path}", flush=True)
        elif init_ckpt and Path(init_ckpt).is_file():
            bundle = load_continuous_bundle(
                init_ckpt, device=device, quiet=True, architecture="dual", apply_meta_env=False
            )
            if bundle is not None:
                init_dual_head_from_continuous(model, bundle.model, quiet=False)
                print(f"[OK] gnode dual-head warm-start from continuous {init_ckpt}", flush=True)
    elif init_ckpt and Path(init_ckpt).is_file():
        init_meta = {}
        init_path = Path(init_ckpt)
        if init_path.is_file():
            init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
            init_meta = dict(init_payload.get("meta") or {})
        ckpt_is_dual = bool(init_meta.get("dual_head"))
        init_mode = str(args.init_mode).strip().lower()
        use_mat_warm = (
            str(args.recipe).strip().lower() == "mat_growth_simple"
            and not dual_head
            and ckpt_is_dual
            and init_mode in ("backbone", "mat_readout")
        )
        if use_mat_warm:
            from src.biochem_gnn.mat_growth_simple import init_mat_single_from_fimat_ckpt

            init_mat_single_from_fimat_ckpt(
                model,
                init_path,
                device=device,
                mode=init_mode,
                quiet=False,
            )
            print(f"[OK] mat-growth warm-start ({init_mode}) from {init_ckpt}", flush=True)
        else:
            arch = "single" if dual_head and not ckpt_is_dual else None
            bundle = load_continuous_bundle(
                init_ckpt, device=device, quiet=True, architecture=arch, apply_meta_env=False
            )
            if bundle is not None:
                if dual_head and not ckpt_is_dual and isinstance(model, SpeciesDualHeadContinuousGNN):
                    init_dual_head_from_continuous(model, bundle.model)
                else:
                    load_pushforward_state_dict_partial(
                        model, bundle.model.state_dict(), quiet=False
                    )
                print(f"[OK] warm-start from {init_ckpt}", flush=True)
    else:
        init_path = args.init_s1.strip() or str(root / DEFAULT_SNAPSHOT_CKPT)
        if Path(init_path).is_file():
            init_continuous_from_snapshot(model, init_path)
    if str(args.recipe).strip().lower() == "mat_growth_simple" and str(args.leg).strip():
        apply_mat_growth_leg_env(str(args.leg).strip(), force=True)
    # Bias readout toward small positive log-deltas (avoid zero-delta collapse).
    with torch.no_grad():
        bias_layers: list[torch.nn.Linear] = []
        if hasattr(model, "readout"):
            last = model.readout[-1]
            if isinstance(last, torch.nn.Linear):
                bias_layers.append(last)
        elif hasattr(model, "magnitude_head"):
            last = model.magnitude_head[-1]
            if isinstance(last, torch.nn.Linear):
                bias_layers.append(last)
        for last in bias_layers:
            if last.bias is not None:
                last.bias.fill_(0.5 if growth_only else 1e-4)

    n_windows = sum(len(p["windows"]) for p in packs)
    t0_caps = {p["anchor"]: int(p["train_t0_max"]) for p in packs}
    print(
        f"[i] phase={phase_tag} anchors={train_anchors} val={val_anchor} "
        f"unroll={unroll} max_unroll={max_unroll} tbptt_tail={tbptt_tail_steps()} "
        f"windows={n_windows} dual_head={int(dual_head)} "
        f"kin_norm={int(kin_per_vessel_norm_enabled())} physics={int(physics_on)} "
        f"vel_decay={int(continuous_vel_decay_enabled())} "
        f"sat_gate={int(continuous_saturation_gate())} sat_scale={saturation_headroom_scale():.0f} "
        f"mature_exempt={int(continuous_mature_fp_exempt())} mature_frac={mature_clot_frac():.2f} "
        f"temporal_gate={int(continuous_temporal_gate())} "
        f"delta_res={int(continuous_delta_residual())} "
        f"temp_off={int(continuous_temporal_offset())} "
        f"score_clot_w={continuous_score_clot_weight():.2f} "
        f"clout_score={species_continuous_clout_score_mode()} "
        f"lambda=({temporal_lambda_bounds()[0]:.1f},{temporal_lambda_bounds()[1]:.1f}) "
        f"closed_loop_init={closed_loop_init_prob():.2f} "
        f"final_state_w={continuous_final_state_weight():.2f} "
        f"teacher_noise={continuous_teacher_noise_sigma():.3f} "
        f"teacher_fp={continuous_teacher_fp_frac():.2f} blur={continuous_teacher_blur():.2f} "
        f"growth_only={int(growth_only)} delta_thr={delta_thr:.1e} fp_w={fp_w:.1f} "
        f"t0_min={pushforward_train_t0_min()} t0_max_per_vessel={int(pushforward_train_t0_per_vessel())} "
        f"t0_caps={t0_caps} "
        f"loss_scale={loss_scale:.0f} lr={lr:.1e} grad_clip={grad_clip:.1f} "
        f"huber_beta={huber_b:.2e} ch_w=({ch_w[0]:.1f},{ch_w[1]:.1f})",
        flush=True,
    )

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    out_raw = args.out.strip() or default_out
    out_path = Path(out_raw)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "train_log.jsonl"

    meta_base = {
        "anchors": train_anchors,
        "val_anchor": val_anchor,
        "phase": phase_tag,
        "unroll": unroll,
        "max_unroll": max_unroll,
        "tbptt_tail": tbptt_tail_steps(),
        "vel_decay": continuous_vel_decay_enabled(),
        "flow_feats": flow_feats_enabled(),
        "flow_dynamic": flow_feats_dynamic(),
        "flow_drop_xy": flow_feats_drop_xy(),
        "geom_feats": geom_feats_enabled(),
        "geom_feats_rich": geom_feats_rich_enabled(),
        "neighbor_commit_gate": continuous_neighbor_commit_gate(),
        "neighbor_commit_alpha": continuous_neighbor_commit_alpha(),
        "gate_temp": continuous_gate_temp(),
        "frontier_hops": continuous_frontier_hops(),
        "nucleation_topk": continuous_nucleation_topk(),
        "latent_dropout": species_latent_dropout_p(),
        "saturation_gate": continuous_saturation_gate(),
        "saturation_scale": saturation_headroom_scale(),
        "mature_fp_exempt": continuous_mature_fp_exempt(),
        "mature_frac": mature_clot_frac(),
        "temporal_gate": continuous_temporal_gate(),
        "temporal_lambda_min": temporal_lambda_bounds()[0],
        "temporal_lambda_max": temporal_lambda_bounds()[1],
        "delta_residual": continuous_delta_residual(),
        "temporal_offset": continuous_temporal_offset(),
        "score_clot_w": continuous_score_clot_weight(),
        "closed_loop_init": closed_loop_init_prob(),
        "final_state_weight": continuous_final_state_weight(),
        "final_state_all_band": continuous_final_state_all_band(),
        "speed_fp_weight": continuous_speed_fp_weight(),
        "deploy_horizon": deploy_horizon_steps(),
        "teacher_noise": continuous_teacher_noise_sigma(),
        "teacher_fp_frac": continuous_teacher_fp_frac(),
        "teacher_blur": continuous_teacher_blur(),
        "stride": stride,
        "wall_hops": wall_hops,
        "latent_dim": latent_dim,
        "hidden": hidden,
        "kine_ckpt": kine_ckpt,
        "n_band": ref_static["n_band"],
        "pushforward_species_scope": pushforward_species_scope(),
        "pushforward_species_channels": pushforward_state_bulk_indices(),
        "pushforward_species_label": scope_label_for_channels(pushforward_state_bulk_indices()),
        "n_windows": n_windows,
        "growth_only_loss": growth_only,
        "dual_head": dual_head,
        "kin_per_vessel_norm": kin_per_vessel_norm_enabled(),
        "spatial_loss_weight": continuous_spatial_loss_weight(),
        "physics_readout": physics_on,
        "delta_threshold": delta_thr,
        "fp_weight": fp_w,
        "loss_scale": loss_scale,
        "huber_beta": huber_b,
        "channel_weight_fi": ch_w[0],
        "channel_weight_mat": ch_w[1],
        "train_t0_max": t0_max,
        "train_t0_max_per_vessel": bool(pushforward_train_t0_per_vessel()),
        "train_t0_caps": {p["anchor"]: int(p["train_t0_max"]) for p in packs},
        "train_t0_min": pushforward_train_t0_min(),
        "arch": pushforward_arch,
        "leg": str(args.leg).strip() if args.leg else "",
        "env_overrides": (
            dict(__import__("src.biochem_gnn.mat_growth_simple", fromlist=["mat_growth_leg_spec"]).mat_growth_leg_spec(str(args.leg).strip()).env_overrides)
            if (str(args.recipe).strip().lower() == "mat_growth_simple" and str(args.leg).strip())
            else {}
        ),
    }

    best_score = -1.0
    stale = 0
    t0 = time.perf_counter()

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        cur_unroll = curriculum_unroll_for_epoch(ep)
        pack_order = packs[:]
        random.shuffle(pack_order)

        for pack in pack_order:
            wins = pack["windows"][:]
            random.shuffle(wins)
            static = pack["static"]
            # Move static and data to GPU for training
            static_gpu = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in static.items()
            }
            pack_data_gpu = pack["data"].to(device)
            train_m_gpu = pack["train_m"].to(device)

            for win in wins:
                win_use = win[: cur_unroll + 1]
                series = log_series_on_band(pack_data_gpu, win_use, device, static_gpu["node_idx"])
                speed_series = (
                    band_speed_series(
                        pack_data_gpu, win_use, device, static_gpu["node_idx"], for_training=True
                    )
                    if continuous_vel_decay_enabled()
                    else None
                )
                physics_ctx = None
                if physics_on:
                    physics_ctx = build_species_physics_ctx(
                        pack_data_gpu,
                        time_window=win_use,
                        node_idx=static_gpu["node_idx"],
                        phys_cfg=pack["phys"],
                        bio_cfg=pack["bio"],
                        device=device,
                    )
                w_t0 = pushforward_window_t0_weight(int(win_use[0]))
                if w_t0 <= 0.0:
                    continue
                log_state0 = series[0]
                if (
                    int(win_use[0]) > 0
                    and closed_loop_init_prob() > 0.0
                    and random.random() < closed_loop_init_prob()
                ):
                    log_state0 = rollout_prefix_log_state(
                        model,
                        pack_data_gpu,
                        static_gpu,
                        int(win_use[0]),
                        device,
                    )
                velocity_series = [pack_data_gpu.y[ti, static_gpu["node_idx"], 0:2] for ti in win_use]
                species_block_full = [pack_data_gpu.y[ti, static_gpu["node_idx"], sc.SPECIES_BLOCK] for ti in win_use]
                loss, _, _ = unroll_continuous_loss(
                    model,
                    base_feats=static_gpu["base_feats"],
                    edge_index=static_gpu["edge_index"],
                    log_series=series,
                    train_mask=train_m_gpu,
                    log_state0=log_state0,
                    speed_series=speed_series,
                    training=True,
                    physics_ctx=physics_ctx,
                    window_weight=w_t0,
                    tbptt_tail=tbptt_tail_steps(),
                    pos_band=static_gpu.get("pos_band"),
                    time_window=win_use,
                    flow_series=static_gpu.get("flow_series"),
                    flow_cols=static_gpu.get("flow_cols"),
                    wall_mask_band=pack_data_gpu.mask_wall[static_gpu["node_idx"]] if hasattr(pack_data_gpu, "mask_wall") and pack_data_gpu.mask_wall is not None else None,
                    species_block=species_block_full,
                    velocity=velocity_series,
                )
                if not loss.requires_grad:
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_losses.append(float(loss.item()))

            # Cleanup GPU pack memory (Data.to is in-place; restore CPU residency)
            del static_gpu, pack_data_gpu, train_m_gpu
            _release_pack_to_cpu(pack)

        if True:
            h = deploy_horizon_steps()
            dep_packs = packs if deploy_horizon_aux_all_packs() else [val_pack]
            aux_cap = deploy_horizon_aux_cap_steps()
            for vpack in dep_packs:
                n_times = int(vpack["static"]["n_times"])
                if h > 0:
                    t_end = min(int(h), n_times - 1)
                else:
                    t_end = graph_last_time_index(n_times)
                if aux_cap > 0:
                    t_end = min(t_end, aux_cap - 1)
                if t_end < 3:
                    continue
                win_dep = list(range(0, t_end + 1))
                static = vpack["static"]
                # Move to GPU for deploy horizon loss
                static_gpu = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in static.items()
                }
                vpack_data_gpu = vpack["data"].to(device)
                train_m_gpu = vpack["train_m"].to(device)

                series = log_series_on_band(vpack_data_gpu, win_dep, device, static_gpu["node_idx"])
                speed_series = band_speed_series(
                    vpack_data_gpu, win_dep, device, static_gpu["node_idx"], for_training=True
                )
                w_dep = 2.5 if vpack["anchor"] == val_anchor else 1.25
                velocity_series = [vpack_data_gpu.y[ti, static_gpu["node_idx"], 0:2] for ti in win_dep]
                species_block_full = [vpack_data_gpu.y[ti, static_gpu["node_idx"], sc.SPECIES_BLOCK] for ti in win_dep]
                loss_dep, _, _ = unroll_continuous_loss(
                    model,
                    base_feats=static_gpu["base_feats"],
                    edge_index=static_gpu["edge_index"],
                    log_series=series,
                    train_mask=train_m_gpu,
                    log_state0=series[0],
                    speed_series=speed_series,
                    training=True,
                    window_weight=w_dep,
                    tbptt_tail=min(tbptt_tail_steps(), max(5, len(win_dep) // 5)),
                    speed_fp_weight=continuous_speed_fp_weight(),
                    pos_band=static_gpu.get("pos_band"),
                    time_window=win_dep,
                    flow_series=static_gpu.get("flow_series"),
                    flow_cols=static_gpu.get("flow_cols"),
                    wall_mask_band=vpack_data_gpu.mask_wall[static_gpu["node_idx"]] if hasattr(vpack_data_gpu, "mask_wall") and vpack_data_gpu.mask_wall is not None else None,
                    species_block=species_block_full,
                    velocity=velocity_series,
                )
                if loss_dep.requires_grad:
                    opt.zero_grad(set_to_none=True)
                    loss_dep.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    ep_losses.append(float(loss_dep.item()))

                del static_gpu, vpack_data_gpu, train_m_gpu
                _release_pack_to_cpu(vpack)

        model.eval()
        val_state_f1: list[float] = []
        val_mat_f1: list[float] = []
        val_growth_f1: list[float] = []
        val_growth_mat_f1: list[float] = []
        val_init_f1: list[float] = []
        val_pred_delta: list[float] = []
        val_clot_phi_f1: list[float] = []
        deploy_mat_f1 = 0.0
        deploy_fi_f1 = 0.0
        deploy_clot_f1 = 0.0
        deploy_clot_guiding = 0.0
        deploy_clot_relaxed_f05 = 0.0
        deploy_clot_relaxed_prec = 0.0
        deploy_clot_relaxed_rec = 0.0
        deploy_clot_pred_pos_frac = 0.0
        deploy_clot_dil_iou = 0.0
        deploy_clot_score = 0.0
        deploy_clot_guiding_mid = 0.0
        with torch.no_grad():
            # Move val pack to GPU
            val_static_gpu = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in val_pack["static"].items()
            }
            val_data_gpu = val_pack["data"].to(device)
            val_m_gpu = val_pack["val_m"].to(device)

            for win in val_pack["val_windows"]:
                static = val_pack["static"]
                series = log_series_on_band(val_data_gpu, win, device, val_static_gpu["node_idx"])
                speed_series = (
                    band_speed_series(val_data_gpu, win, device, val_static_gpu["node_idx"])
                    if continuous_vel_decay_enabled()
                    else None
                )
                physics_ctx = None
                if physics_on:
                    physics_ctx = build_species_physics_ctx(
                        val_data_gpu,
                        time_window=win,
                        node_idx=val_static_gpu["node_idx"],
                        phys_cfg=val_pack["phys"],
                        bio_cfg=val_pack["bio"],
                        device=device,
                    )
                velocity_series = [val_data_gpu.y[ti, val_static_gpu["node_idx"], 0:2] for ti in win]
                species_block_full = [val_data_gpu.y[ti, val_static_gpu["node_idx"], sc.SPECIES_BLOCK] for ti in win]
                m = eval_continuous_window(
                    model,
                    base_feats=val_static_gpu["base_feats"],
                    edge_index=val_static_gpu["edge_index"],
                    log_series=series,
                    mask=val_m_gpu,
                    log_state0=series[0],
                    speed_series=speed_series,
                    physics_ctx=physics_ctx,
                    time_window=win,
                    flow_series=val_static_gpu.get("flow_series"),
                    flow_cols=val_static_gpu.get("flow_cols"),
                    wall_mask_band=val_data_gpu.mask_wall[val_static_gpu["node_idx"]] if hasattr(val_data_gpu, "mask_wall") and val_data_gpu.mask_wall is not None else None,
                    species_block=species_block_full,
                    velocity=velocity_series,
                )
                val_state_f1.append(m["final_state_f1"])
                val_mat_f1.append(m["final_state_mat_f1"])
                val_growth_f1.append(m["mean_growth_f1"])
                val_growth_mat_f1.append(m["mean_growth_mat_f1"])
                val_init_f1.append(m["init_state_f1"])
                val_pred_delta.append(m["mean_pred_delta"])
                val_clot_phi_f1.append(m.get("clot_phi_f1", 0.0))
            if True:
                n_val = int(val_data_gpu.y.shape[0])
                t_deploy = deploy_eval_time_index(n_val)
                dep = eval_full_rollout_fimat_f1(
                    model,
                    val_data_gpu,
                    val_static_gpu,
                    device,
                    time_index=t_deploy,
                )
                deploy_mat_f1 = float(dep["deploy_mat_f1"])
                deploy_fi_f1 = float(dep["deploy_fi_f1"])
                need_clot = (
                    bool(args.exclude_val_from_train)
                    or continuous_score_clot_weight() > 0.0
                    or mat_precision_select
                )
                if need_clot:
                    from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache

                    flow_eval = train_deploy_eval_flow_source()
                    env_snap = {
                        k: os.environ.get(k)
                        for k in (
                            "SPECIES_ROLLOUT_VEL_SOURCE",
                            "SPECIES_ROLLOUT_PIN_OTHER",
                            "SPECIES_ROLLOUT_IC_SOURCE",
                            "SPECIES_ROLLOUT_DEPLOY_FAITHFUL",
                            "T0_R4_FLOW_SOURCE",
                        )
                    }
                    apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})
                    clot_times = deploy_eval_clot_times(n_val)
                    clf_by_t: dict[int, dict] = {}
                    for t_clot in clot_times:
                        clf_by_t[int(t_clot)] = eval_deploy_clot_f1(
                            model,
                            val_data_gpu,
                            val_static_gpu,
                            val_pack["phys"],
                            val_pack["bio"],
                            device,
                            time_index=int(t_clot),
                            flow_source=flow_eval,
                        )
                    t_main = deploy_eval_time_index(n_val)
                    clf = clf_by_t[t_main]
                    if len(clf_by_t) > 1:
                        w_full = deploy_eval_dual_full_weight()
                        w_mid = 1.0 - w_full
                        mid_t = legacy_capped_deploy_time_index(n_val)
                        s_full = float(
                            clf_by_t[t_main].get(
                                "deploy_clot_score",
                                clot_score_from_deploy_dict(clf_by_t[t_main]),
                            )
                        )
                        s_mid = float(
                            clf_by_t[mid_t].get(
                                "deploy_clot_score",
                                clot_score_from_deploy_dict(clf_by_t[mid_t]),
                            )
                        )
                        deploy_clot_score = w_full * s_full + w_mid * s_mid
                    else:
                        deploy_clot_score = float(
                            clf.get("deploy_clot_score", clot_score_from_deploy_dict(clf))
                        )
                    deploy_clot_f1 = float(clf["deploy_clot_f1"])
                    deploy_clot_guiding = float(clf.get("deploy_clot_guiding", deploy_clot_f1))
                    deploy_clot_relaxed_f05 = float(clf.get("deploy_clot_relaxed_f05", deploy_clot_f1))
                    deploy_clot_relaxed_prec = float(clf.get("deploy_clot_relaxed_prec", 0.0))
                    deploy_clot_relaxed_rec = float(clf.get("deploy_clot_relaxed_rec", 0.0))
                    deploy_clot_pred_pos_frac = float(clf.get("deploy_clot_pred_pos_frac", 0.0))
                    deploy_clot_dil_iou = float(clf.get("deploy_clot_dil_iou", 0.0))
                    if len(clf_by_t) > 1:
                        mid_t = legacy_capped_deploy_time_index(n_val)
                        deploy_clot_guiding_mid = float(
                            clf_by_t[mid_t].get("deploy_clot_guiding", 0.0)
                        )
                    else:
                        deploy_clot_guiding_mid = deploy_clot_guiding
                    for k, v in env_snap.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                    reset_species_rollout_flow_cache()

            # Cleanup GPU val memory (keep packs on CPU for the next epoch)
            del val_static_gpu, val_data_gpu, val_m_gpu
            _release_pack_to_cpu(val_pack)

        row = {
            "epoch": ep,
            "loss": sum(ep_losses) / max(len(ep_losses), 1),
            "val_state_f1": sum(val_state_f1) / max(len(val_state_f1), 1),
            "val_mat_f1": sum(val_mat_f1) / max(len(val_mat_f1), 1),
            "val_growth_f1": sum(val_growth_f1) / max(len(val_growth_f1), 1),
            "val_growth_mat_f1": sum(val_growth_mat_f1) / max(len(val_growth_mat_f1), 1),
            "val_init_f1": sum(val_init_f1) / max(len(val_init_f1), 1),
            "val_pred_delta": sum(val_pred_delta) / max(len(val_pred_delta), 1),
            "val_clot_phi_f1": sum(val_clot_phi_f1) / max(len(val_clot_phi_f1), 1),
            "cur_unroll": cur_unroll,
        }
        if True:
            row["deploy_eval_t"] = t_deploy
            row["deploy_mat_f1"] = deploy_mat_f1
            row["deploy_fi_f1"] = deploy_fi_f1
            row["deploy_mat_f1_t53"] = deploy_mat_f1
            row["deploy_fi_f1_t53"] = deploy_fi_f1
            if bool(args.exclude_val_from_train) or continuous_score_clot_weight() > 0.0 or mat_precision_select:
                row["deploy_clot_f1"] = deploy_clot_f1
                row["deploy_clot_f1_t53"] = deploy_clot_f1
                row["deploy_clot_guiding"] = deploy_clot_guiding
                row["deploy_clot_relaxed_f05"] = deploy_clot_relaxed_f05
                row["deploy_clot_relaxed_prec"] = deploy_clot_relaxed_prec
                row["deploy_clot_relaxed_rec"] = deploy_clot_relaxed_rec
                row["deploy_clot_pred_pos_frac"] = deploy_clot_pred_pos_frac
                row["deploy_clot_dil_iou"] = deploy_clot_dil_iou
                row["deploy_clot_score"] = deploy_clot_score
                row["deploy_clot_guiding_mid"] = deploy_clot_guiding_mid
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        dep_msg = ""
        if True:
            dep_msg = f" deploy_mat_t={deploy_mat_f1:.3f} deploy_fi_t={deploy_fi_f1:.3f} t={t_deploy} unroll={cur_unroll}"
            if bool(args.exclude_val_from_train) or continuous_score_clot_weight() > 0.0 or mat_precision_select:
                dep_msg += (
                    f" deploy_clot_g={deploy_clot_guiding:.3f}"
                    f" f05={deploy_clot_relaxed_f05:.3f}"
                    f" rprec={deploy_clot_relaxed_prec:.3f}"
                    f" rrec={deploy_clot_relaxed_rec:.3f}"
                    f" pos={deploy_clot_pred_pos_frac:.3f}"
                    f" diou={deploy_clot_dil_iou:.3f}"
                    f" f1={deploy_clot_f1:.3f}"
                )
        print(
            f"[ep {ep:03d}] loss={row['loss']:.6f} "
            f"val_state_f1={row['val_state_f1']:.3f} val_mat_f1={row['val_mat_f1']:.3f} "
            f"val_growth_f1={row['val_growth_f1']:.3f} val_dlt={row['val_pred_delta']:.2e} "
            f"clot_phi_f1={row['val_clot_phi_f1']:.3f} init_f1={row['val_init_f1']:.3f}{dep_msg}",
            flush=True,
        )

        if physics_on:
            score = (
                0.50 * row["val_clot_phi_f1"]
                + 0.25 * row["val_growth_f1"]
                + 0.15 * row["val_state_f1"]
                + 0.10 * row["val_growth_mat_f1"]
            )
        elif bool(args.exclude_val_from_train):
            score = (
                0.55 * deploy_clot_score
                + 0.25 * deploy_mat_f1
                + 0.10 * row["val_state_f1"]
                + 0.10 * row["val_growth_f1"]
            )
        elif mat_precision_select:
            clot_w = continuous_score_clot_weight()
            pos_tgt = 0.08
            overpaint = max(0.0, deploy_clot_pred_pos_frac - pos_tgt)
            mat_score = 0.40 * deploy_mat_f1 + 0.10 * row["val_growth_f1"]
            score = clot_w * deploy_clot_score + (1.0 - clot_w) * mat_score - 0.25 * overpaint
        else:
            mat_score = (
                0.70 * deploy_mat_f1
                + 0.15 * row["val_growth_f1"]
                + 0.10 * row["val_state_f1"]
                + 0.05 * row["val_growth_mat_f1"]
            )
            clot_w = continuous_score_clot_weight()
            if clot_w > 0.0:
                score = (1.0 - clot_w) * mat_score + clot_w * deploy_clot_score
            else:
                score = mat_score
        if score > best_score:
            best_score = score
            stale = 0
            meta = {**meta_base, "best_score": best_score, "best_epoch": ep, **row}
            save_continuous_checkpoint(out_path, model, meta)
        else:
            stale += 1
            if stale >= int(args.early_stop):
                print(f"[i] early stop @ ep {ep} (best_score={best_score:.3f})", flush=True)
                break

    print(f"[OK] best_score={best_score:.3f} elapsed={time.perf_counter() - t0:.1f}s ckpt={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
