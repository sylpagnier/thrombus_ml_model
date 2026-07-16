"""Train decoupled off-wall clot growth model on tiled subgraphs.

This script trains a GNN model to predict species (Mat) growth specifically on
off-wall nodes. For each training window, it dynamically extracts a subgraph
consisting of nodes within k=4 hops of any active clot node. The loss is computed
only on the off-wall nodes in this subgraph.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.clot_growth_masks import (
    gt_growth_commit_mask_at_time,
    graph_dilate_hops,
)
from src.biochem_gnn.config import PHASE_CKPT, apply_deploy_env, apply_train_recipe_env
from src.training.biochem_species_scope import (
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
    build_continuous_gnn,
    closed_loop_init_prob,
    continuous_channel_weights,
    continuous_feature_dim,
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
    pushforward_unroll_steps,
    pushforward_step_stride,
    pushforward_train_t0_max,
    pushforward_window_t0_weight,
    rollout_prefix_log_state,
    save_continuous_checkpoint,
    tbptt_tail_steps,
    unroll_continuous_loss,
    smooth_hop1_log_targets,
    noisy_teacher_log_state0,
    model_vel_decay_alphas,
    compute_hop_distances,
    continuous_delta_threshold,
    continuous_delta_value_scale,
    continuous_huber_beta_growth,
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
    induced_subgraph,
)
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root
from src.utils import species_channels as sc


@torch.no_grad()
def build_global_base_features(data, kine_model, device) -> torch.Tensor:
    """Build kinematic, SDF, and flow features globally on the full graph."""
    n = int(data.num_nodes)
    z_kin = predict_kinematics_latent(kine_model, data)
    sdf = sdf_nd_from_data(data, device, n)
    
    from src.core_physics.species_snapshot_gnn import build_snapshot_features
    base_feats = build_snapshot_features(z_kin, sdf)
    
    from src.core_physics.species_pushforward_gnn import _resolve_flow_uv, _flow_feats_from_uv
    if flow_feats_enabled():
        u, v = _resolve_flow_uv(data, kine_model, device)
        flow = _flow_feats_from_uv(data, u, v, device, torch.arange(n, device=device))
        base_feats = torch.cat([base_feats, flow], dim=1)
    
    if geom_feats_enabled():
        from src.core_physics.species_pushforward_gnn import _geometry_band_features
        geom = _geometry_band_features(data, device, torch.arange(n, device=device)).to(dtype=base_feats.dtype)
        base_feats = torch.cat([base_feats, geom], dim=1)
        
    return base_feats


def _build_anchor_pack_offwall(
    anchor: str,
    *,
    root: Path,
    device: torch.device,
    kine_model,
    unroll: int,
    stride: int,
    max_windows: int,
    phys: PhysicsConfig,
    bio: BiochemConfig,
) -> dict:
    graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor.strip()}.pt"
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    
    # Pre-build global features
    base_feats_global = build_global_base_features(data, kine_model, device).to("cpu")
    
    # Identify time steps and windows
    n_times = int(data.y.shape[0])
    windows = iter_pushforward_windows(n_times, unroll=unroll, stride=stride)
    pack_t0_max = resolve_train_t0_max(n_times)
    
    # Filter windows that have some growth
    from src.core_physics.species_pushforward_continuous import filter_continuous_windows
    # For filtering, we temporarily construct a dummy node_idx to satisfy the signature
    dummy_node_idx = torch.arange(data.num_nodes, device=device)
    windows = filter_continuous_windows(
        windows, data, dummy_node_idx, device, t0_max=pack_t0_max, min_delta_mag=1e-8
    )
    if max_windows > 0:
        windows = windows[: int(max_windows)]
        
    # Get wall mask
    from src.core_physics.clot_phi_simple import _wall_mask_from_data
    wall_mask_full = _wall_mask_from_data(data, device, data.num_nodes).to("cpu")
    
    # Pre-build dynamic flow series if enabled
    flow_series_global = None
    flow_cols = None
    if flow_feats_enabled() and flow_feats_dynamic() and getattr(data, "y", None) is not None and data.y.dim() == 3:
        from src.core_physics.species_pushforward_gnn import _flow_feats_series_from_y
        flow_series_global = _flow_feats_series_from_y(data, device, torch.arange(data.num_nodes, device=device)).to("cpu")
        # Identify starting column of flow features: base snapshot features are z_kin + sdf
        z_kin = predict_kinematics_latent(kine_model, data)
        flow_start = int(z_kin.shape[1] + 1)
        flow_cols = (flow_start, 5) # standard 5-ch flow proxies
        
    # Prepare val windows using fixed target anchors
    val_anchors = [10, 25, 28]
    val_windows = []
    for t0 in val_anchors:
        win = [t0 + i * stride for i in range(unroll + 1)]
        if win[-1] < n_times:
            val_windows.append(win)

    return {
        "anchor": anchor.strip(),
        "data": data.to("cpu"),
        "base_feats_global": base_feats_global,
        "flow_series_global": flow_series_global,
        "flow_cols": flow_cols,
        "wall_mask_full": wall_mask_full,
        "windows": windows,
        "train_t0_max": pack_t0_max,
        "val_windows": val_windows,
        "phys": phys,
        "bio": bio,
    }


def diffuse_field(field: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, hops: int = 2) -> torch.Tensor:
    """GNN average-pooling (diffusion) to blur fields spatially."""
    if hops <= 0:
        return field
    row, col = edge_index
    deg = torch.zeros(num_nodes, device=field.device, dtype=field.dtype)
    deg.index_add_(0, row, torch.ones_like(row, dtype=field.dtype))
    deg_clamp = deg.clamp(min=1.0).unsqueeze(-1)
    
    current = field
    for _ in range(hops):
        neighbor_sum = torch.zeros_like(current)
        neighbor_sum.index_add_(0, col, current[row])
        current = neighbor_sum / deg_clamp
    return current


def compute_shape_loss(
    pred_delta: torch.Tensor,
    tgt_delta: torch.Tensor,
    mask: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    loss_mode: str,
) -> torch.Tensor:
    """Compute shape-aware losses on the extracted subgraph.
    
    `mask` indicates off-wall nodes to supervise.
    """
    p = pred_delta
    t = tgt_delta
    
    # Configurable variables
    val_scale = continuous_delta_value_scale()
    huber_beta = continuous_huber_beta_growth()
    active_thresh = continuous_delta_threshold()

    if loss_mode == "loss_blurring":
        # Diffuse both prediction and target deltas to smooth out high-frequency misalignments
        p_blurred = diffuse_field(p, edge_index, num_nodes, hops=2)
        t_blurred = diffuse_field(t, edge_index, num_nodes, hops=2)
        
        loss = F.huber_loss(
            p_blurred[mask] * val_scale,
            t_blurred[mask] * val_scale,
            delta=huber_beta,
            reduction="mean",
        )
        return loss

    elif loss_mode == "spatial_tolerance":
        # Identify active target growth channel (Mat)
        active = (t[:, 0] > active_thresh)
        
        # Dilate active targets by 2 hops to define the tolerance region
        gt_dilated = graph_dilate_hops(active, edge_index, hops=2)
        
        # FP nodes are where we predict delta but are completely outside dilated active zone
        pred_active = (p[:, 0] > active_thresh)
        fp_mask = pred_active & (~gt_dilated) & mask
        
        losses = []
        active_mask = active & mask
        if active_mask.any():
            losses.append(F.huber_loss(p[active_mask] * val_scale, t[active_mask] * val_scale, delta=huber_beta, reduction="mean"))
            
        if fp_mask.any():
            # False positives outside tolerance zone get penalized
            fp_loss = F.huber_loss(p[fp_mask] * val_scale, torch.zeros_like(p[fp_mask]), delta=huber_beta, reduction="mean")
            losses.append(2.0 * fp_loss)
            
        # Volume conservation term: enforce matching total mass growth on off-wall nodes
        pred_vol = (p[mask] * val_scale).sum()
        tgt_vol = (t[mask] * val_scale).sum()
        vol_loss = F.l1_loss(pred_vol, tgt_vol)
        
        base_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=p.device)
        return base_loss + 0.1 * vol_loss

    else:
        # Standard Huber loss
        loss = F.huber_loss(
            p[mask] * val_scale,
            t[mask] * val_scale,
            delta=huber_beta,
            reduction="mean",
        )
        return loss


def unroll_offwall_loss_custom(
    model: nn.Module,
    *,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    log_series: list[torch.Tensor],
    train_mask: torch.Tensor,
    pos_band: torch.Tensor,
    time_window: list[int] | None,
    flow_series: torch.Tensor | None,
    flow_cols: tuple[int, int] | None,
    wall_mask_band: torch.Tensor,
    species_block: list[torch.Tensor] | None,
    velocity: list[torch.Tensor] | None,
    loss_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Sequence unroller with customized shape loss function."""
    from src.core_physics.species_pushforward_continuous import (
        noisy_teacher_log_state0,
        model_vel_decay_alphas,
        tbptt_tail_steps,
        step_loss_weights,
        splice_dynamic_flow,
        maybe_drop_latent,
        build_continuous_step_features,
        align_continuous_feature_dim,
        log_delta_targets,
        pushforward_log_state_step,
        continuous_vel_decay_enabled,
        continuous_dual_head,
        bind_band_geometry,
    )
    
    n_steps = len(log_series) - 1
    if n_steps <= 0:
        return torch.tensor(0.0, device=device)
        
    bind_band_geometry(model, {
        "pos_band": pos_band,
        "edge_index": edge_index,
        "wall_mask_band": wall_mask_band,
    })
    
    log_state = noisy_teacher_log_state0(log_series[0], edge_index, training=model.training)
    vel_alphas = model_vel_decay_alphas(model)
    tail = tbptt_tail_steps()
    loss_start = max(0, n_steps - int(tail))
    step_w = step_loss_weights(n_steps)
    
    losses = []
    loss_ws = []
    
    for step in range(n_steps):
        grad_step = (not model.training) or step >= loss_start
        ctx = torch.enable_grad() if grad_step else torch.no_grad()
        with ctx:
            if step < loss_start and model.training:
                log_state = log_state.detach()
                
            flow_ti = int(time_window[step]) if time_window is not None else step
            step_base_feats = splice_dynamic_flow(base_feats, flow_series, flow_cols, flow_ti)
            step_base_feats = maybe_drop_latent(step_base_feats, model, model.training and grad_step)
            
            model.log_state = log_state
            model.species_block = species_block[step] if species_block is not None else log_series[step]
            model.velocity = velocity[step] if velocity is not None else None
            
            feats = build_continuous_step_features(
                step_base_feats,
                log_state,
                training=model.training and grad_step,
                time_index=time_window[step + 1] if time_window is not None else step + 1,
                velocity=velocity[step] if velocity is not None else None,
                pos_band=pos_band,
                edge_index=edge_index,
            )
            feats = align_continuous_feature_dim(feats, model)
            
            use_edge_index = getattr(model, "augmented_edge_index", None)
            if use_edge_index is None or os.environ.get("SPECIES_LONGRANGE_EDGES") != "1":
                use_edge_index = edge_index
                
            from src.core_physics.species_pushforward_continuous import delta_readout
            if continuous_dual_head() and hasattr(model, "forward_decoupled"):
                pred_delta, _, _ = model.forward_decoupled(feats, use_edge_index, log_state=log_state)
            else:
                pred_delta = delta_readout(model(feats, use_edge_index))
                
            tgt_delta = log_delta_targets(log_series[step], log_series[step + 1])
            
            # Compute custom shape-aware loss
            step_loss = compute_shape_loss(
                pred_delta,
                tgt_delta,
                train_mask,
                edge_index,
                base_feats.shape[0],
                loss_mode,
            )
            
            # Decay state for next step
            spd = None
            if continuous_vel_decay_enabled():
                vel = velocity[step + 1] if velocity is not None and step + 1 < len(velocity) else (velocity[step] if velocity is not None else None)
                spd = vel.norm(dim=1) if vel is not None else None
                
            log_state = pushforward_log_state_step(
                log_state,
                pred_delta,
                straight_through=model.training and grad_step,
                wall_speed=spd,
                vel_decay_alphas=vel_alphas,
            )
            
            if step_loss is not None and grad_step:
                losses.append(step_loss)
                loss_ws.append(float(step_w[step]))
                
    if not losses:
        return torch.tensor(0.0, device=device)
        
    wsum = max(sum(loss_ws), 1e-6)
    return sum(loss * w for loss, w in zip(losses, loss_ws)) / wsum


def collect_active_env_overrides() -> dict[str, str]:
    overrides = {}
    for k, v in os.environ.items():
        if k.startswith("SPECIES_") or k.startswith("BIOCHEM_") or k.startswith("CLOT_"):
            overrides[k] = v
    return overrides


def main() -> int:
    ap = argparse.ArgumentParser(description="Train decoupled off-wall species continuous GNN with shape losses")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchors", default="", help="Comma-separated anchors for multi-vessel train")
    ap.add_argument("--all-anchors", action="store_true", help="Train on all biochem anchor graphs on disk")
    ap.add_argument("--val-anchor", default="patient007", help="Holdout anchor for val logging")
    ap.add_argument("--init", default="", help="Optional checkpoint to warm-start")
    ap.add_argument("--no-init", action="store_true", help="Random init")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unroll", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default="")
    ap.add_argument("--early-stop", type=int, default=15)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--hops-k", type=int, default=4, help="k-hops dilation around active clots for subgraphs")
    ap.add_argument(
        "--loss-mode",
        choices=("standard", "spatial_tolerance", "loss_blurring"),
        default="standard",
        help="Loss strategy for off-wall growth optimization"
    )
    args = ap.parse_args()

    apply_train_recipe_env()
    
    # Parse early to apply meta env from initialization checkpoint
    init_ckpt = "" if bool(args.no_init) else (args.init.strip() or str(get_project_root() / DEFAULT_S34_CKPT))
    if init_ckpt and Path(init_ckpt).is_file():
        load_continuous_bundle(init_ckpt, quiet=True, apply_meta_env=True)
        print(f"[i] Applied meta environment overrides from {init_ckpt} early", flush=True)
    
    # Enforce off-wall scope by default
    os.environ["CLOT_PHI_PHYSICS_WALL_MAT_ONLY"] = "0"
    
    unroll = pushforward_unroll_steps() if args.unroll is None else int(args.unroll)
    stride = pushforward_step_stride() if args.stride is None else int(args.stride)
    hidden = int(args.hidden)
    
    bulk_channels = pushforward_state_bulk_indices()
    growth_only = continuous_growth_only_loss()
    lr = float(args.lr) if args.lr is not None else (3e-4 if growth_only else 1e-3)
    grad_clip = float(args.grad_clip) if args.grad_clip is not None else 1.0

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from src.core_physics.t0_device import require_cuda_device
    device = require_cuda_device()
    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    train_anchors = parse_biochem_train_anchors(args.anchors or args.anchor, all_anchors=bool(args.all_anchors), root=root)
    val_anchor = args.val_anchor.strip() or train_anchors[0]

    kine_ckpt = str(resolve_kinematics_checkpoint())
    kine_model = load_kinematics_predictor(kine_ckpt, device, phys_cfg=PhysicsConfig(phase="kinematics"))

    print(f"[i] Pre-processing graph datasets...", flush=True)
    packs: list[dict] = []
    for anc in train_anchors:
        packs.append(
            _build_anchor_pack_offwall(
                anc,
                root=root,
                device=device,
                kine_model=kine_model,
                unroll=unroll,
                stride=stride,
                max_windows=int(args.max_windows),
                phys=phys,
                bio=bio,
            )
        )
    val_pack = next((p for p in packs if p["anchor"] == val_anchor), packs[0])
    
    # Resolve dims using first pack
    ref_static = packs[0]["base_feats_global"]
    latent_dim = int(ref_static.shape[1] - 1)
    in_dim = continuous_feature_dim(latent_dim)
    
    pushforward_arch_name = species_pushforward_arch()
    model = build_continuous_gnn(in_dim, hidden=hidden, arch=pushforward_arch_name).to(device)
    model.kin_latent_dim = latent_dim
    model.latent_dropout_p = species_latent_dropout_p()

    init_ckpt = "" if bool(args.no_init) else (args.init.strip() or str(root / DEFAULT_S34_CKPT))
    if bool(args.no_init):
        print("[i] random init (--no-init)", flush=True)
    elif init_ckpt and Path(init_ckpt).is_file():
        bundle = load_continuous_bundle(init_ckpt, device=device, quiet=True, architecture="dual", apply_meta_env=False)
        if bundle is not None:
            init_dual_head_from_continuous(model, bundle.model, quiet=False)
            print(f"[OK] warm-start GNN from {init_ckpt}", flush=True)
            
    # Setup optimizer
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    out_raw = args.out.strip() or "outputs/biochem/offwall_model/best.pth"
    out_path = Path(out_raw)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "train_log.jsonl"

    best_score = -1.0
    stale = 0
    t0 = time.perf_counter()

    print(
        f"[i] Training decoupled off-wall growth model:\n"
        f"  epochs: {args.epochs}, lr: {lr:.1e}, early-stop limit: {args.early_stop}\n"
        f"  dilation hops k: {args.hops_k}, unroll window: {unroll}\n"
        f"  supervision: OFF-WALL nodes only (loss-masked)\n"
        f"  loss mode: {args.loss_mode}\n"
        f"  output: {out_path}",
        flush=True,
    )

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        cur_unroll = curriculum_unroll_for_epoch(ep)
        pack_order = packs[:]
        random.shuffle(pack_order)

        for pack in pack_order:
            wins = pack["windows"][:]
            random.shuffle(wins)
            base_feats_global = pack["base_feats_global"].to(device)
            wall_mask_full = pack["wall_mask_full"].to(device)
            data_gpu = pack["data"].to(device)

            for win in wins:
                win_use = win[: cur_unroll + 1]
                
                # 1. Compute active clot union across window
                clot_union = torch.zeros(data_gpu.num_nodes, dtype=torch.bool, device=device)
                for ti in win_use:
                    clot_t = gt_growth_commit_mask_at_time(data_gpu, ti, phys, device)
                    clot_union |= clot_t
                
                if not clot_union.any():
                    continue # skip window if there are absolutely no active clots
                    
                # 2. Dilate by k hops to build local subgraph mask
                subgraph_mask = graph_dilate_hops(clot_union, data_gpu.edge_index, args.hops_k)
                
                # 3. Extract subgraph nodes and edge index
                node_idx, edge_sub, remap = induced_subgraph(subgraph_mask, data_gpu.edge_index)
                
                # 4. Supervise only off-wall nodes within this neighborhood
                wall_mask_sub = wall_mask_full[node_idx]
                offwall_nodes = ~wall_mask_sub
                if not offwall_nodes.any():
                    continue # skip window if there are no off-wall nodes in this neighborhood
                
                # 5. Slice inputs to subgraph indices
                base_feats_sub = base_feats_global[node_idx]
                pos_sub = data_gpu.x[node_idx, :2].to(dtype=base_feats_sub.dtype)
                
                series = []
                for ti in win_use:
                    y = data_gpu.y[ti].to(device=device, dtype=torch.float32)
                    sp = y[:, sc.SPECIES_BLOCK]
                    sp_sub = torch.stack([sp[:, int(ch)] for ch in bulk_channels], dim=-1)[node_idx]
                    series.append(sp_sub)
                    
                speed_series = None
                if continuous_vel_decay_enabled():
                    speed_series = []
                    for ti in win_use:
                        y = data_gpu.y[ti].to(device=device, dtype=torch.float32)
                        vel = y[:, 0:2]
                        speed = vel.norm(dim=1)
                        speed_series.append(speed[node_idx])
                
                velocity_series = [data_gpu.y[ti, node_idx, 0:2] for ti in win_use]
                species_block_full = [data_gpu.y[ti, node_idx, sc.SPECIES_BLOCK] for ti in win_use]
                
                # Dynamic flow features sliced
                flow_series_sub = None
                if pack["flow_series_global"] is not None:
                    flow_series_sub = pack["flow_series_global"][:, node_idx.cpu()].to(device)
                
                # Custom off-wall shape loss
                loss = unroll_offwall_loss_custom(
                    model,
                    base_feats=base_feats_sub,
                    edge_index=edge_sub,
                    log_series=series,
                    train_mask=offwall_nodes,
                    pos_band=pos_sub,
                    time_window=win_use,
                    flow_series=flow_series_sub,
                    flow_cols=pack["flow_cols"],
                    wall_mask_band=wall_mask_sub,
                    species_block=species_block_full,
                    velocity=velocity_series,
                    loss_mode=args.loss_mode,
                    device=device,
                )
                
                if not loss.requires_grad:
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_losses.append(float(loss.item()))

            # Cleanup
            del base_feats_global, wall_mask_full, data_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_loss = sum(ep_losses) / max(len(ep_losses), 1)

        # --- Validation Loop (Standard deploy metrics evaluation) ---
        model.eval()
        val_static = val_pack["base_feats_global"].to(device)
        val_data = val_pack["data"].to(device)
        
        dummy_static = {
            "node_idx": torch.arange(val_data.num_nodes, device=device),
            "base_feats": val_static,
            "edge_index": val_data.edge_index,
            "pos_band": val_data.x[:, :2].to(dtype=val_static.dtype),
        }
        
        n_val = int(val_data.y.shape[0])
        t_deploy = deploy_eval_time_index(n_val)
        
        # Standard evaluate rollout F1
        dep_f1 = eval_full_rollout_fimat_f1(
            model,
            val_data,
            dummy_static,
            device,
            time_index=t_deploy,
        )
        
        # Standard evaluate clot metrics (relaxed precision/recall/F1)
        apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": "gt"})
        clf = eval_deploy_clot_f1(
            model,
            val_data,
            dummy_static,
            phys,
            bio,
            device,
            time_index=t_deploy,
            flow_source="gt",
        )
        
        val_score = float(clf.get("deploy_clot_score", 0.0))
        offwall_relaxed_f1 = float(clf.get("deploy_clot_offwall_relaxed_f1", 0.0))
        offwall_n_pred = float(clf.get("deploy_clot_offwall_n_pred", 0.0))
        offwall_n_gt = float(clf.get("deploy_clot_offwall_n_gt", 0.0))
        
        improved = False
        if val_score > best_score:
            best_score = val_score
            improved = True
            stale = 0
            
            meta = {
                "epoch": ep,
                "val_score": val_score,
                "offwall_relaxed_f1": offwall_relaxed_f1,
                "offwall_n_pred": offwall_n_pred,
                "offwall_n_gt": offwall_n_gt,
                "unroll": unroll,
                "hops_k": args.hops_k,
                "arch": pushforward_arch_name,
                "loss_mode": args.loss_mode,
                "env_overrides": collect_active_env_overrides(),
            }
            save_continuous_checkpoint(out_path, model, meta)
        else:
            stale += 1

        dt = time.perf_counter() - t0
        print(
            f"Epoch {ep:02d} | Loss: {mean_loss:.4f} | Val Score: {val_score:.3f} "
            f"| Off-wall Relaxed F1: {offwall_relaxed_f1:.3f} | Pred Off-wall: {offwall_n_pred:.1f}/{offwall_n_gt:.1f} "
            f"| {'[SAVED]' if improved else ''}",
            flush=True,
        )

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": ep,
                "loss": mean_loss,
                "val_score": val_score,
                "offwall_relaxed_f1": offwall_relaxed_f1,
                "offwall_n_pred": offwall_n_pred,
                "offwall_n_gt": offwall_n_gt,
                "dt": dt
            }) + "\n")

        if stale >= int(args.early_stop):
            print(f"[i] Early stopping triggered at epoch {ep} (stale={stale})", flush=True)
            break

    print(f"[OK] Training complete. Best score: {best_score:.3f}. Output saved to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
