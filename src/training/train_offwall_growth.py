"""Train decoupled clot-growth specialist on tiled subgraphs around existing clots.

For each training window, extract subgraph(s) within ``--hops-k`` of active clot(s),
then supervise either:
  * ``offwall``         - ~wall nodes only (legacy v6 blurring specialist)
  * ``frontier``        - k-hop neighborhood of committed clots (wall + lumen growth)
  * ``frontier_lumen``  - dilate(clot) & ~wall
  * ``frontier_ge2``    - dilate(clot) & hop>=2 (interior lumen only)
  * ``hop_ge2``         - all hop>=2 nodes

Tile modes (``--tile-mode``):
  * ``union``         - one subgraph per window around the union of all clots (default)
  * ``per_component`` - one subgraph per uninterrupted clot region (connected component)

Warm-start from WC_v7 and use ``--loss-mode loss_blurring`` for the compound A/B recipe.
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
    bool_mask_connected_components,
)
from src.biochem_gnn.config import PHASE_CKPT, apply_deploy_env, apply_train_recipe_env
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env
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
    clear_offwall_model_cache,
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
    predict_kinematics_and_latent,
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
    
    # Pre-build global features (tile training still indexes these on subgraphs).
    base_feats_global = build_global_base_features(data, kine_model, device).to("cpu")

    # Identify time steps and windows
    n_times = int(data.y.shape[0])
    windows = iter_pushforward_windows(n_times, unroll=unroll, stride=stride)
    pack_t0_max = resolve_train_t0_max(n_times)

    # Filter windows that have some growth (CPU — avoid parking full graphs on a 4GB GPU)
    from src.core_physics.species_pushforward_continuous import filter_continuous_windows
    cpu = torch.device("cpu")
    dummy_node_idx = torch.arange(data.num_nodes, device=cpu)
    windows = filter_continuous_windows(
        windows, data, dummy_node_idx, cpu, t0_max=pack_t0_max, min_delta_mag=1e-8
    )
    if max_windows > 0:
        # Prefer late windows (early t0 often has no committed clot yet).
        windows = windows[-int(max_windows) :]

    # Get wall mask
    from src.core_physics.clot_phi_simple import _wall_mask_from_data
    wall_mask_full = _wall_mask_from_data(data, device, data.num_nodes).to("cpu")

    # Pre-build dynamic flow series if enabled
    flow_series_global = None
    flow_cols = None
    if flow_feats_enabled() and flow_feats_dynamic() and getattr(data, "y", None) is not None and data.y.dim() == 3:
        from src.core_physics.species_pushforward_gnn import _flow_feats_series_from_y
        flow_series_global = _flow_feats_series_from_y(
            data, device, torch.arange(data.num_nodes, device=device)
        ).to("cpu")
        # Identify starting column of flow features: base snapshot features are z_kin + sdf
        z_kin = predict_kinematics_latent(kine_model, data)
        flow_start = int(z_kin.shape[1] + 1)
        flow_cols = (flow_start, 5)  # standard 5-ch flow proxies

    # Wall-band static matching eval_mat_growth_simple (compound-val / A floor).
    # Full-graph static silently zeros clot F1 on patient001 (diagnose_crack_001_root).
    with torch.no_grad():
        pred_uv, z_kin_band = predict_kinematics_and_latent(kine_model, data)
    data.u0_pred = pred_uv[:, 0].detach().to(device="cpu").clone()
    data.v0_pred = pred_uv[:, 1].detach().to(device="cpu").clone()
    band_static = build_band_base_features(
        data,
        kine_model,
        device,
        wall_hops=snapshot_wall_hops(),
        z_kin_override=z_kin_band,
    )
    band_static_cpu = {
        k: (v.detach().to("cpu") if torch.is_tensor(v) else v) for k, v in band_static.items()
    }

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
        "band_static": band_static_cpu,
        "flow_series_global": flow_series_global,
        "flow_cols": flow_cols,
        "wall_mask_full": wall_mask_full,
        "windows": windows,
        "train_t0_max": pack_t0_max,
        "val_windows": val_windows,
        "phys": phys,
        "bio": bio,
    }


def _band_static_to_device(band_static: dict, device: torch.device) -> dict:
    """Move packed wall-band deploy static to ``device`` for compound-val / A floor."""
    out = {}
    for k, v in band_static.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


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
    *,
    hop_dist: torch.Tensor | None = None,
    lumen_shape_weight: float = 2.0,
) -> torch.Tensor:
    """Compute shape-aware losses on the extracted subgraph.

    ``mask`` selects supervised nodes (off-wall and/or clot frontier).
    ``loss_lumen_shape`` = blurring_prec + soft Dice favoring hop>=2 GT shape.
    """
    p = pred_delta
    t = tgt_delta

    # Configurable variables
    val_scale = continuous_delta_value_scale()
    huber_beta = continuous_huber_beta_growth()
    active_thresh = continuous_delta_threshold()

    if loss_mode in ("loss_blurring", "loss_blurring_prec", "loss_lumen_shape"):
        # Diffuse both prediction and target deltas to smooth out high-frequency misalignments
        p_blurred = diffuse_field(p, edge_index, num_nodes, hops=2)
        t_blurred = diffuse_field(t, edge_index, num_nodes, hops=2)

        loss = F.huber_loss(
            p_blurred[mask] * val_scale,
            t_blurred[mask] * val_scale,
            delta=huber_beta,
            reduction="mean",
        )
        if loss_mode == "loss_blurring":
            return loss

        # Precision term: penalize raw (unblurred) growth outside 2-hop of GT active deltas.
        # Kept light vs spatial_tolerance's collapse (§203): shape term remains primary.
        if t.dim() == 1:
            active = t > active_thresh
            pred_ch = p
        else:
            active = t[:, 0] > active_thresh
            pred_ch = p[:, 0]
        gt_dilated = graph_dilate_hops(active, edge_index, hops=2)
        pred_active = pred_ch > active_thresh
        fp_mask = pred_active & (~gt_dilated) & mask.reshape(-1)
        if fp_mask.any():
            fp_pred = p[fp_mask] * val_scale
            fp_loss = F.huber_loss(
                fp_pred,
                torch.zeros_like(fp_pred),
                delta=huber_beta,
                reduction="mean",
            )
            loss = loss + 0.5 * fp_loss

        if loss_mode == "loss_lumen_shape" and hop_dist is not None:
            lumen = (hop_dist.reshape(-1) >= 2) & mask.reshape(-1)
            if bool(lumen.any().item()) and bool((active & lumen).any().item()):
                pred_s = torch.sigmoid((pred_ch[lumen] - active_thresh) * 8.0)
                gt_s = active[lumen].to(dtype=pred_s.dtype)
                inter = (pred_s * gt_s).sum()
                pred_sum = pred_s.sum()
                gt_sum = gt_s.sum()
                fn_w = float(os.environ.get("SPECIES_LUMEN_SHAPE_FN_W", "2.5"))
                fp_w = float(os.environ.get("SPECIES_LUMEN_SHAPE_FP_W", "1.0"))
                dice = (2.0 * inter) / (fp_w * pred_sum + fn_w * gt_sum + 1e-6)
                loss = loss + float(lumen_shape_weight) * (1.0 - dice)
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


def growth_specialist_ckpt_score(
    *,
    ckpt_metric: str,
    clot_score: float,
    offwall_relaxed_f1: float,
    offwall_n_pred: float,
    offwall_n_gt: float,
    hop_ge2_strict_f1: float = 0.0,
    hop_ge2_n_pred: float = 0.0,
    hop_ge2_n_gt: float = 0.0,
) -> float:
    """Checkpoint selection score for the decoupled growth specialist.

    - ``clot_score``: legacy full deploy clot score (growth-alone; poor for specialists).
    - ``offwall_relaxed``: maximize off-wall relaxed F1.
    - ``offwall_balanced``: blend relaxed F1 with volume match vs GT (best-practice Arm C).
    - ``hop_ge2_balanced``: prioritize hop>=2 strict F1 + lumen volume match.
    - ``hop_ge2_recall``: limit-analysis score — reward lumen recall (TP/GT), light FP penalty.
    """
    mode = (ckpt_metric or "clot_score").strip().lower()
    if mode in ("offwall_relaxed", "offwall_relaxed_f1", "relaxed"):
        return float(offwall_relaxed_f1)
    if mode in ("hop_ge2_recall", "lumen_recall", "recall_push"):
        n_gt = max(float(hop_ge2_n_gt), 0.0)
        n_pred = max(float(hop_ge2_n_pred), 0.0)
        if n_gt <= 0.0:
            # Prefer idle on no-GT vessels (spray still hurts a little).
            return -0.05 * min(n_pred, 40.0)
        # Proxy recall from volume when TP unknown: min(pred, gt)/gt, plus strict.
        recall_proxy = min(n_pred, n_gt) / n_gt
        overshoot = max(0.0, (n_pred - n_gt) / n_gt)
        return (
            0.55 * recall_proxy
            + 0.35 * float(hop_ge2_strict_f1)
            - 0.10 * min(overshoot, 2.0)
        )
    if mode in ("hop_ge2_balanced", "lumen_balanced", "hop2"):
        n_gt = max(float(hop_ge2_n_gt), 0.0)
        n_pred = max(float(hop_ge2_n_pred), 0.0)
        if n_gt <= 0.0:
            return float(hop_ge2_strict_f1) - 0.1 * min(n_pred, 20.0)
        vol_match = min(n_pred / n_gt, 1.0)
        overshoot = max(0.0, (n_pred - n_gt) / max(n_gt, 1.0))
        vol_match = max(0.0, vol_match - 0.25 * overshoot)
        return 0.65 * float(hop_ge2_strict_f1) + 0.35 * vol_match
    if mode in ("offwall_balanced", "balanced", "offwall"):
        n_gt = max(float(offwall_n_gt), 0.0)
        n_pred = max(float(offwall_n_pred), 0.0)
        if n_gt <= 0.0:
            # No GT off-wall mass: prefer predicting nothing (avoid free FPs).
            return float(offwall_relaxed_f1) - 0.1 * min(n_pred, 20.0)
        vol_match = min(n_pred / n_gt, 1.0)
        return 0.7 * float(offwall_relaxed_f1) + 0.3 * vol_match
    return float(clot_score)

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
    hop_dist: torch.Tensor | None = None,
    lumen_shape_weight: float = 2.0,
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
                hop_dist=hop_dist,
                lumen_shape_weight=lumen_shape_weight,
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


def freeze_growth_backbone(model: torch.nn.Module) -> tuple[int, int]:
    """Freeze SAGE/conv backbone; keep spatial/magnitude heads trainable.

    Returns ``(n_frozen, n_trainable)`` parameter tensors.
    """
    head_prefixes = (
        "spatial_head",
        "magnitude_head",
        "spatial_head_wall",
        "magnitude_head_wall",
        "spatial_head_offwall",
        "magnitude_head_offwall",
    )
    n_frozen = 0
    n_train = 0
    for name, param in model.named_parameters():
        is_head = any(name == p or name.startswith(p + ".") for p in head_prefixes)
        if is_head:
            param.requires_grad = True
            n_train += 1
        else:
            param.requires_grad = False
            n_frozen += 1
    return n_frozen, n_train


def _restore_two_model_env(prev: dict[str, str | None]) -> None:
    """Restore SPECIES_TWO_MODEL_* env keys after compound val."""
    clear_offwall_model_cache()
    for key, val in prev.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


@torch.no_grad()
def eval_compound_wall_route_deploy(
    *,
    growth_model: torch.nn.Module,
    wall_ckpt: Path,
    growth_tmp_ckpt: Path,
    val_pack: dict,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    hidden: int,
    meta_env: dict[str, str] | None = None,
) -> dict:
    """Wall-route compound deploy: frozen wall ckpt + current growth weights on disk.

    Saves ``growth_model`` to ``growth_tmp_ckpt`` so the two-model cache can load it,
    then evaluates with ``eval_deploy_clot_f1`` on the val pack.
    """
    prev = {
        "SPECIES_TWO_MODEL_MODE": os.environ.get("SPECIES_TWO_MODEL_MODE"),
        "SPECIES_OFFWALL_MODEL_CKPT": os.environ.get("SPECIES_OFFWALL_MODEL_CKPT"),
        "SPECIES_TWO_MODEL_ROUTE": os.environ.get("SPECIES_TWO_MODEL_ROUTE"),
    }
    save_continuous_checkpoint(
        growth_tmp_ckpt,
        growth_model,
        {
            "tmp_compound_val": True,
            "env_overrides": meta_env or collect_active_env_overrides(),
        },
    )
    clear_offwall_model_cache()
    wall_bundle = load_continuous_bundle(
        str(wall_ckpt),
        device=device,
        quiet=True,
        architecture="dual",
        apply_meta_env=False,
    )
    if wall_bundle is None:
        raise RuntimeError(f"failed to load wall ckpt for compound val: {wall_ckpt}")
    wall_model = wall_bundle.model
    wall_model.eval()

    val_static_pack = val_pack.get("band_static")
    if not isinstance(val_static_pack, dict) or "base_feats" not in val_static_pack:
        raise RuntimeError(
            "compound-val requires val_pack['band_static'] from _build_anchor_pack_offwall "
            "(wall-band static matching eval_mat_growth_simple)"
        )
    dummy_static = _band_static_to_device(val_static_pack, device)
    val_data = val_pack["data"].to(device)
    try:
        n_val = int(val_data.y.shape[0])
        t_deploy = deploy_eval_time_index(n_val)
        flow_eval = train_deploy_eval_flow_source()
        apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})

        os.environ["SPECIES_TWO_MODEL_MODE"] = "1"
        os.environ["SPECIES_OFFWALL_MODEL_CKPT"] = str(growth_tmp_ckpt)
        os.environ["SPECIES_TWO_MODEL_ROUTE"] = "wall"

        clf = eval_deploy_clot_f1(
            wall_model,
            val_data,
            dummy_static,
            phys,
            bio,
            device,
            time_index=t_deploy,
            flow_source=flow_eval,
        )
        return dict(clf)
    finally:
        _restore_two_model_env(prev)
        del wall_bundle, wall_model, val_data
        if device.type == "cuda":
            torch.cuda.empty_cache()


@torch.no_grad()
def eval_wall_only_deploy_floor(
    *,
    wall_ckpt: Path,
    val_pack: dict,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
) -> float:
    """Arm-A clot F1 on val pack (wall ckpt alone) used as compound save floor."""
    val_static_pack = val_pack.get("band_static")
    if not isinstance(val_static_pack, dict) or "base_feats" not in val_static_pack:
        raise RuntimeError(
            "A-floor requires val_pack['band_static'] from _build_anchor_pack_offwall "
            "(wall-band static matching eval_mat_growth_simple)"
        )
    prev = {
        "SPECIES_TWO_MODEL_MODE": os.environ.get("SPECIES_TWO_MODEL_MODE"),
        "SPECIES_OFFWALL_MODEL_CKPT": os.environ.get("SPECIES_OFFWALL_MODEL_CKPT"),
        "SPECIES_TWO_MODEL_ROUTE": os.environ.get("SPECIES_TWO_MODEL_ROUTE"),
    }
    clear_offwall_model_cache()
    os.environ["SPECIES_TWO_MODEL_MODE"] = "0"
    os.environ.pop("SPECIES_OFFWALL_MODEL_CKPT", None)
    os.environ.pop("SPECIES_TWO_MODEL_ROUTE", None)

    wall_bundle = load_continuous_bundle(
        str(wall_ckpt),
        device=device,
        quiet=True,
        architecture="dual",
        apply_meta_env=False,
    )
    if wall_bundle is None:
        raise RuntimeError(f"failed to load wall ckpt for A floor: {wall_ckpt}")
    wall_model = wall_bundle.model
    wall_model.eval()

    dummy_static = _band_static_to_device(val_static_pack, device)
    val_data = val_pack["data"].to(device)
    try:
        n_val = int(val_data.y.shape[0])
        t_deploy = deploy_eval_time_index(n_val)
        flow_eval = train_deploy_eval_flow_source()
        apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})
        clf = eval_deploy_clot_f1(
            wall_model,
            val_data,
            dummy_static,
            phys,
            bio,
            device,
            time_index=t_deploy,
            flow_source=flow_eval,
        )
        return float(clf.get("deploy_clot_f1", 0.0))
    finally:
        _restore_two_model_env(prev)
        del wall_bundle, wall_model, val_data
        if device.type == "cuda":
            torch.cuda.empty_cache()


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
        "--tile-mode",
        choices=("union", "per_component"),
        default="union",
        help="union=one tile per window around all clots; per_component=one tile per clot CC",
    )
    ap.add_argument(
        "--max-tiles-per-window",
        type=int,
        default=8,
        help="Cap connected-component tiles per window (largest first); ignored for union",
    )
    ap.add_argument(
        "--frontier-hops",
        type=int,
        default=2,
        help="When --supervise-mode=frontier*, loss mask = dilate(clot, this many hops)",
    )
    ap.add_argument(
        "--supervise-mode",
        choices=("offwall", "frontier", "frontier_lumen", "frontier_ge2", "hop_ge2"),
        default="offwall",
        help=(
            "offwall=~wall; frontier=dilate(clot); frontier_lumen=dilate&~wall; "
            "frontier_ge2=dilate&hop>=2; hop_ge2=BFS hops>=2"
        ),
    )
    ap.add_argument(
        "--loss-mode",
        choices=("standard", "spatial_tolerance", "loss_blurring", "loss_blurring_prec", "loss_lumen_shape"),
        default="standard",
        help="loss_lumen_shape = blurring_prec + soft Dice on hop>=2 GT shape",
    )
    ap.add_argument(
        "--ckpt-metric",
        choices=("clot_score", "offwall_relaxed", "offwall_balanced", "hop_ge2_balanced", "hop_ge2_recall"),
        default="clot_score",
        help="hop_ge2_balanced=prec+vol; hop_ge2_recall=limit-analysis recall push",
    )
    ap.add_argument(
        "--lumen-shape-weight",
        type=float,
        default=2.0,
        help="Weight on soft Dice lumen-shape term (loss_lumen_shape)",
    )
    ap.add_argument(
        "--cheap-val",
        action="store_true",
        help="Skip full deploy val (use -train_loss); for smoke tests on small GPUs",
    )
    ap.add_argument(
        "--compound-val",
        action="store_true",
        help="Val = wall-route compound deploy (wall ckpt + this growth model) on --val-anchor",
    )
    ap.add_argument(
        "--wall-ckpt",
        default="",
        help="Frozen wall/canonical ckpt for --compound-val (default: --init / locked)",
    )
    ap.add_argument(
        "--wall-clot-floor-delta",
        type=float,
        default=0.02,
        help="Reject ckpt save if compound clot F1 < A_floor - delta",
    )
    ap.add_argument(
        "--compound-val-every",
        type=int,
        default=2,
        help="Run compound deploy val every N epochs (1=every epoch)",
    )
    ap.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze SAGE/conv; train spatial/magnitude heads only (wall-protect)",
    )
    ap.add_argument(
        "--mat-leg",
        default="",
        help="Optional mat-growth leg env (e.g. WC_v7_clot_phi_mse) for feature compatibility",
    )
    args = ap.parse_args()

    apply_train_recipe_env()

    # Parse early to apply meta env from initialization checkpoint
    init_ckpt = "" if bool(args.no_init) else (args.init.strip() or str(get_project_root() / DEFAULT_S34_CKPT))
    if init_ckpt and Path(init_ckpt).is_file():
        load_continuous_bundle(init_ckpt, quiet=True, apply_meta_env=True)
        print(f"[i] Applied meta environment overrides from {init_ckpt} early", flush=True)

    # Mat-leg last (force) so WC_v7 feature/physics knobs win over stale init meta.
    if args.mat_leg.strip():
        apply_mat_growth_leg_env(args.mat_leg.strip(), force=True)
        print(f"[i] Applied mat-growth leg env: {args.mat_leg.strip()}", flush=True)

    # Never wall-only gelation for a growth specialist
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
        # Pack tensors stay on CPU; free transient GPU from kinematics / flow series.
        if device.type == "cuda":
            torch.cuda.empty_cache()
    val_pack = next((p for p in packs if p["anchor"] == val_anchor), packs[0])

    # Kinematics predictor is only needed for feature packing; free VRAM for the GNN.
    del kine_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[i] Packed {len(packs)} anchors on CPU "
        f"(peak GPU after pack free: "
        f"{(torch.cuda.memory_allocated() / (1024 ** 2)):.0f} MiB)"
        if device.type == "cuda"
        else f"[i] Packed {len(packs)} anchors",
        flush=True,
    )

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
            # Drop init teacher weights from VRAM (partial copy already applied).
            del bundle
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if bool(args.freeze_backbone):
        n_fr, n_tr = freeze_growth_backbone(model)
        print(f"[i] freeze-backbone: frozen={n_fr} trainable_heads={n_tr}", flush=True)

    # Setup optimizer (trainable params only when backbone frozen)
    opt = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=1e-5,
    )
    
    out_raw = args.out.strip() or "outputs/biochem/offwall_model/best.pth"
    out_path = Path(out_raw)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "train_log.jsonl"

    best_score = float("-inf")  # first epoch always eligible (cheap-val uses -loss)
    stale = 0
    t0 = time.perf_counter()

    supervise_mode = str(args.supervise_mode).strip().lower()
    tile_mode = str(args.tile_mode).strip().lower()
    max_tiles_per_window = max(int(args.max_tiles_per_window), 1)
    frontier_hops = max(int(args.frontier_hops), 0)
    ckpt_metric = str(args.ckpt_metric).strip().lower()
    lumen_shape_weight = float(args.lumen_shape_weight)
    if supervise_mode == "frontier":
        supervise_desc = f"FRONTIER dilate(clot,{frontier_hops}) including wall near commits"
    elif supervise_mode == "frontier_lumen":
        supervise_desc = (
            f"FRONTIER_LUMEN dilate(clot,{frontier_hops}) & ~wall (growth specialist, no wall loss)"
        )
    elif supervise_mode == "frontier_ge2":
        supervise_desc = (
            f"FRONTIER_GE2 dilate(clot,{frontier_hops}) & hop>=2 (interior lumen only)"
        )
    elif supervise_mode == "hop_ge2":
        supervise_desc = "HOP>=2 lumen nodes only (firewall target region)"
    else:
        supervise_desc = "OFF-WALL nodes only (loss-masked)"
    use_compound_val = bool(args.compound_val) and not bool(args.cheap_val)
    if bool(args.compound_val) and bool(args.cheap_val):
        print("[WARN] --cheap-val overrides --compound-val (smoke path)", flush=True)
    wall_ckpt_raw = (args.wall_ckpt or "").strip() or (args.init or "").strip() or str(root / DEFAULT_S34_CKPT)
    wall_ckpt_path = Path(wall_ckpt_raw)
    if not wall_ckpt_path.is_absolute():
        wall_ckpt_path = root / wall_ckpt_path
    compound_val_every = max(int(args.compound_val_every), 1)
    wall_floor_delta = float(args.wall_clot_floor_delta)
    a_floor_clot_f1: float | None = None
    growth_tmp_ckpt = out_path.parent / "_compound_val_growth_tmp.pth"
    print(
        f"[i] Training decoupled clot-growth specialist:\n"
        f"  epochs: {args.epochs}, lr: {lr:.1e}, early-stop limit: {args.early_stop}\n"
        f"  subgraph hops k: {args.hops_k}, unroll window: {unroll}\n"
        f"  tile_mode: {tile_mode} (max_tiles/window={max_tiles_per_window})\n"
        f"  supervision: {supervise_desc}\n"
        f"  loss mode: {args.loss_mode}\n"
        f"  ckpt metric: {ckpt_metric}\n"
        f"  lumen_shape_weight: {lumen_shape_weight}\n"
        f"  freeze_backbone: {int(bool(args.freeze_backbone))}\n"
        f"  compound_val: {int(use_compound_val)} every={compound_val_every} "
        f"floor_delta={wall_floor_delta}\n"
        f"  mat-leg: {args.mat_leg.strip() or '(none)'}\n"
        f"  output: {out_path}",
        flush=True,
    )

    if use_compound_val:
        if not wall_ckpt_path.is_file():
            raise FileNotFoundError(f"--compound-val needs wall ckpt: {wall_ckpt_path}")
        print(f"[i] Measuring Arm-A clot F1 floor on {val_anchor}...", flush=True)
        a_floor_clot_f1 = eval_wall_only_deploy_floor(
            wall_ckpt=wall_ckpt_path,
            val_pack=val_pack,
            phys=phys,
            bio=bio,
            device=device,
        )
        print(
            f"[i] A_floor deploy_clot_f1={a_floor_clot_f1:.4f} "
            f"(reject if compound < {a_floor_clot_f1 - wall_floor_delta:.4f})",
            flush=True,
        )

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        ep_tiles = 0
        cur_unroll = curriculum_unroll_for_epoch(ep)
        pack_order = packs[:]
        random.shuffle(pack_order)

        for pack in pack_order:
            wins = pack["windows"][:]
            random.shuffle(wins)
            # Keep full timeline graph on CPU (4GB GPUs OOM on data.to(cuda)).
            data_cpu = pack["data"]
            n_nodes = int(data_cpu.num_nodes)
            edge_index = data_cpu.edge_index.to(device=device)
            base_feats_global = pack["base_feats_global"].to(device=device)
            wall_mask_full = pack["wall_mask_full"].to(device=device)
            pos_cpu = data_cpu.x[:, :2]
            hop_full = compute_hop_distances(edge_index, wall_mask_full, n_nodes)

            for win in wins:
                win_use = win[: cur_unroll + 1]

                # 1. Active clot union (GT helper streams y[t] only onto device).
                clot_union = torch.zeros(n_nodes, dtype=torch.bool, device=device)
                for ti in win_use:
                    clot_t = gt_growth_commit_mask_at_time(data_cpu, ti, phys, device)
                    clot_union |= clot_t

                if not clot_union.any():
                    continue

                if tile_mode == "per_component":
                    tile_seeds = bool_mask_connected_components(clot_union, edge_index)
                    if max_tiles_per_window > 0:
                        tile_seeds = tile_seeds[:max_tiles_per_window]
                else:
                    tile_seeds = [clot_union]

                for clot_seed in tile_seeds:
                    # 2-3. Local subgraph around this clot seed (union or one CC)
                    subgraph_mask = graph_dilate_hops(clot_seed, edge_index, args.hops_k)
                    node_idx, edge_sub, remap = induced_subgraph(subgraph_mask, edge_index)

                    # 4. Supervision mask (relative to this tile's clot seed)
                    wall_mask_sub = wall_mask_full[node_idx]
                    hop_sub = hop_full[node_idx]
                    if supervise_mode == "frontier":
                        growth_zone_full = graph_dilate_hops(clot_seed, edge_index, frontier_hops)
                        train_mask = growth_zone_full[node_idx]
                    elif supervise_mode == "frontier_lumen":
                        growth_zone_full = graph_dilate_hops(clot_seed, edge_index, frontier_hops)
                        train_mask = growth_zone_full[node_idx] & (~wall_mask_sub)
                    elif supervise_mode == "frontier_ge2":
                        growth_zone_full = graph_dilate_hops(clot_seed, edge_index, frontier_hops)
                        train_mask = growth_zone_full[node_idx] & (hop_sub >= 2)
                    elif supervise_mode == "hop_ge2":
                        train_mask = hop_sub >= 2
                    else:
                        train_mask = ~wall_mask_sub
                    if not train_mask.any():
                        continue

                    # 5. Slice inputs (CPU y -> device only for subgraph nodes / times)
                    node_idx_cpu = node_idx.detach().cpu()
                    base_feats_sub = base_feats_global[node_idx]
                    pos_sub = pos_cpu[node_idx_cpu].to(device=device, dtype=base_feats_sub.dtype)

                    series = []
                    for ti in win_use:
                        y = data_cpu.y[int(ti)].to(device=device, dtype=torch.float32)
                        sp = y[:, sc.SPECIES_BLOCK]
                        sp_sub = torch.stack([sp[:, int(ch)] for ch in bulk_channels], dim=-1)[node_idx]
                        series.append(sp_sub)

                    velocity_series = [
                        data_cpu.y[int(ti), node_idx_cpu, 0:2].to(device=device, dtype=torch.float32)
                        for ti in win_use
                    ]
                    species_block_full = [
                        data_cpu.y[int(ti), node_idx_cpu, sc.SPECIES_BLOCK].to(
                            device=device, dtype=torch.float32
                        )
                        for ti in win_use
                    ]

                    flow_series_sub = None
                    if pack["flow_series_global"] is not None:
                        flow_series_sub = pack["flow_series_global"][:, node_idx_cpu].to(device)

                    loss = unroll_offwall_loss_custom(
                        model,
                        base_feats=base_feats_sub,
                        edge_index=edge_sub,
                        log_series=series,
                        train_mask=train_mask,
                        pos_band=pos_sub,
                        time_window=win_use,
                        flow_series=flow_series_sub,
                        flow_cols=pack["flow_cols"],
                        wall_mask_band=wall_mask_sub,
                        species_block=species_block_full,
                        velocity=velocity_series,
                        loss_mode=args.loss_mode,
                        device=device,
                        hop_dist=hop_sub,
                        lumen_shape_weight=lumen_shape_weight,
                    )

                    if not loss.requires_grad:
                        continue
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    ep_losses.append(float(loss.item()))
                    ep_tiles += 1

            del base_feats_global, wall_mask_full, edge_index, hop_full
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_loss = sum(ep_losses) / max(len(ep_losses), 1)
        if not ep_losses:
            msg = (
                f"epoch {ep}: no supervised windows produced a loss "
                f"(check --max-windows / clot presence on train anchors)"
            )
            print(f"[WARN] {msg}", flush=True)
            if ep == 1:
                raise RuntimeError(msg)
        else:
            print(
                f"[i] epoch {ep}: tiles={ep_tiles} mean_loss={mean_loss:.4f} "
                f"(tile_mode={tile_mode})",
                flush=True,
            )

        # --- Validation ---
        model.eval()
        wall_reject = False
        compound_clot_f1 = 0.0
        skipped_compound = False
        if bool(args.cheap_val):
            # Smoke path: avoid parking ~200MB y timelines on a 4GB GPU for deploy eval.
            val_clot_score = 0.0
            offwall_relaxed_f1 = 0.0
            offwall_strict_f1 = 0.0
            offwall_n_pred = 0.0
            offwall_n_gt = 0.0
            hop_ge2_strict_f1 = 0.0
            hop_ge2_n_pred = 0.0
            hop_ge2_n_gt = 0.0
            val_score = -float(mean_loss)
        elif use_compound_val and (ep % compound_val_every != 0) and ep != int(args.epochs):
            # Hold score between compound vals (do not cheap-save on -loss).
            skipped_compound = True
            val_clot_score = 0.0
            offwall_relaxed_f1 = 0.0
            offwall_strict_f1 = 0.0
            offwall_n_pred = 0.0
            offwall_n_gt = 0.0
            hop_ge2_strict_f1 = 0.0
            hop_ge2_n_pred = 0.0
            hop_ge2_n_gt = 0.0
            val_score = float("-inf")
            print(
                f"[i] epoch {ep}: skip compound val (every {compound_val_every})",
                flush=True,
            )
        elif use_compound_val:
            clf = eval_compound_wall_route_deploy(
                growth_model=model,
                wall_ckpt=wall_ckpt_path,
                growth_tmp_ckpt=growth_tmp_ckpt,
                val_pack=val_pack,
                phys=phys,
                bio=bio,
                device=device,
                hidden=hidden,
                meta_env=collect_active_env_overrides(),
            )
            val_clot_score = float(clf.get("deploy_clot_score", 0.0))
            compound_clot_f1 = float(clf.get("deploy_clot_f1", 0.0))
            offwall_relaxed_f1 = float(clf.get("deploy_clot_offwall_relaxed_f1", 0.0))
            offwall_strict_f1 = float(clf.get("deploy_clot_offwall_strict_f1", 0.0))
            offwall_n_pred = float(clf.get("deploy_clot_offwall_n_pred", 0.0))
            offwall_n_gt = float(clf.get("deploy_clot_offwall_n_gt", 0.0))
            hop_ge2_strict_f1 = float(clf.get("deploy_clot_offwall_strict_f1_hop_ge2", 0.0))
            hop_ge2_n_pred = float(clf.get("deploy_clot_offwall_n_pred_hop_ge2", 0.0))
            hop_ge2_n_gt = float(clf.get("deploy_clot_offwall_n_gt_hop_ge2", 0.0))
            val_score = growth_specialist_ckpt_score(
                ckpt_metric=ckpt_metric,
                clot_score=val_clot_score,
                offwall_relaxed_f1=offwall_relaxed_f1,
                offwall_n_pred=offwall_n_pred,
                offwall_n_gt=offwall_n_gt,
                hop_ge2_strict_f1=hop_ge2_strict_f1,
                hop_ge2_n_pred=hop_ge2_n_pred,
                hop_ge2_n_gt=hop_ge2_n_gt,
            )
            floor = float(a_floor_clot_f1 if a_floor_clot_f1 is not None else 0.0)
            if compound_clot_f1 < (floor - wall_floor_delta):
                wall_reject = True
                print(
                    f"[WARN] wall-floor reject: compound clot_f1={compound_clot_f1:.4f} "
                    f"< A_floor={floor:.4f} - {wall_floor_delta:.3f}",
                    flush=True,
                )
        else:
            val_static = val_pack["base_feats_global"].to(device)
            val_data = val_pack["data"].to(device)
            try:
                # Must bind wall_mask_band: dual-head sat / magnitude paths and hop
                # geometry depend on it. Omitting it made train-val offwall counts
                # non-reproducible vs eval_mat_growth_simple (firewall seq collapse).
                wall_band = val_pack["wall_mask_full"].to(device=device)
                dummy_static = {
                    "node_idx": torch.arange(int(val_data.num_nodes), device=device),
                    "base_feats": val_static,
                    "edge_index": val_data.edge_index,
                    "pos_band": val_data.x[:, :2].to(dtype=val_static.dtype),
                    "wall_mask_band": wall_band,
                }
                n_val = int(val_data.y.shape[0])
                t_deploy = deploy_eval_time_index(n_val)
                _ = eval_full_rollout_fimat_f1(
                    model,
                    val_data,
                    dummy_static,
                    device,
                    time_index=t_deploy,
                )
                # Match production mat-growth eval: kinematics flow, not GT oracle.
                flow_eval = train_deploy_eval_flow_source()
                apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})
                clf = eval_deploy_clot_f1(
                    model,
                    val_data,
                    dummy_static,
                    phys,
                    bio,
                    device,
                    time_index=t_deploy,
                    flow_source=flow_eval,
                )
            finally:
                del val_data, val_static
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            val_clot_score = float(clf.get("deploy_clot_score", 0.0))
            offwall_relaxed_f1 = float(clf.get("deploy_clot_offwall_relaxed_f1", 0.0))
            offwall_strict_f1 = float(clf.get("deploy_clot_offwall_strict_f1", 0.0))
            offwall_n_pred = float(clf.get("deploy_clot_offwall_n_pred", 0.0))
            offwall_n_gt = float(clf.get("deploy_clot_offwall_n_gt", 0.0))
            hop_ge2_strict_f1 = float(clf.get("deploy_clot_offwall_strict_f1_hop_ge2", 0.0))
            hop_ge2_n_pred = float(clf.get("deploy_clot_offwall_n_pred_hop_ge2", 0.0))
            hop_ge2_n_gt = float(clf.get("deploy_clot_offwall_n_gt_hop_ge2", 0.0))
            val_score = growth_specialist_ckpt_score(
                ckpt_metric=ckpt_metric,
                clot_score=val_clot_score,
                offwall_relaxed_f1=offwall_relaxed_f1,
                offwall_n_pred=offwall_n_pred,
                offwall_n_gt=offwall_n_gt,
                hop_ge2_strict_f1=hop_ge2_strict_f1,
                hop_ge2_n_pred=hop_ge2_n_pred,
                hop_ge2_n_gt=hop_ge2_n_gt,
            )

        improved = False
        if skipped_compound:
            # Do not count toward early-stop stale when we deliberately skip val.
            pass
        elif wall_reject:
            stale += 1
        elif val_score > best_score:
            best_score = val_score
            improved = True
            stale = 0

            meta = {
                "epoch": ep,
                "val_score": val_score,
                "val_clot_score": val_clot_score,
                "compound_clot_f1": compound_clot_f1 if use_compound_val else None,
                "a_floor_clot_f1": a_floor_clot_f1,
                "ckpt_metric": ckpt_metric,
                "offwall_relaxed_f1": offwall_relaxed_f1,
                "offwall_strict_f1": offwall_strict_f1,
                "offwall_n_pred": offwall_n_pred,
                "offwall_n_gt": offwall_n_gt,
                "hop_ge2_strict_f1": hop_ge2_strict_f1,
                "hop_ge2_n_pred": hop_ge2_n_pred,
                "hop_ge2_n_gt": hop_ge2_n_gt,
                "unroll": unroll,
                "hops_k": args.hops_k,
                "frontier_hops": frontier_hops,
                "supervise_mode": supervise_mode,
                "tile_mode": tile_mode,
                "max_tiles_per_window": max_tiles_per_window,
                "arch": pushforward_arch_name,
                "loss_mode": args.loss_mode,
                "mat_leg": args.mat_leg.strip() or None,
                "compound_val": use_compound_val,
                "freeze_backbone": bool(args.freeze_backbone),
                "env_overrides": collect_active_env_overrides(),
            }
            save_continuous_checkpoint(out_path, model, meta)
        else:
            stale += 1

        dt = time.perf_counter() - t0
        tag = ""
        if improved:
            tag = "[SAVED]"
        elif wall_reject:
            tag = "[WALL_REJECT]"
        elif skipped_compound:
            tag = "[SKIP_VAL]"
        print(
            f"Epoch {ep:02d} | Loss: {mean_loss:.4f} | CkptScore: {val_score:.3f} "
            f"(clot={val_clot_score:.3f}"
            f"{f' compound_f1={compound_clot_f1:.3f}' if use_compound_val and not skipped_compound else ''}"
            f") | Off-wall RelF1: {offwall_relaxed_f1:.3f} "
            f"| Pred Off-wall: {offwall_n_pred:.1f}/{offwall_n_gt:.1f} "
            f"| hop_ge2: {hop_ge2_n_pred:.1f}/{hop_ge2_n_gt:.1f} "
            f"(strict={hop_ge2_strict_f1:.3f}) "
            f"| {tag}",
            flush=True,
        )

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": ep,
                "loss": mean_loss,
                "n_tiles": ep_tiles,
                "tile_mode": tile_mode,
                "val_score": val_score,
                "val_clot_score": val_clot_score,
                "compound_clot_f1": compound_clot_f1 if use_compound_val else None,
                "a_floor_clot_f1": a_floor_clot_f1,
                "wall_reject": wall_reject,
                "skipped_compound": skipped_compound,
                "ckpt_metric": ckpt_metric,
                "offwall_relaxed_f1": offwall_relaxed_f1,
                "offwall_strict_f1": offwall_strict_f1,
                "offwall_n_pred": offwall_n_pred,
                "offwall_n_gt": offwall_n_gt,
                "hop_ge2_strict_f1": hop_ge2_strict_f1,
                "hop_ge2_n_pred": hop_ge2_n_pred,
                "hop_ge2_n_gt": hop_ge2_n_gt,
                "dt": dt
            }) + "\n")

        if (not skipped_compound) and stale >= int(args.early_stop):
            print(f"[i] Early stopping triggered at epoch {ep} (stale={stale})", flush=True)
            break

    if growth_tmp_ckpt.is_file():
        try:
            growth_tmp_ckpt.unlink()
        except OSError:
            pass

    if not out_path.is_file():
        print(f"[WARN] no checkpoint was saved (best_score={best_score}); writing last weights", flush=True)
        save_continuous_checkpoint(
            out_path,
            model,
            {
                "epoch": int(args.epochs),
                "val_score": float(best_score),
                "ckpt_metric": ckpt_metric,
                "supervise_mode": supervise_mode,
                "loss_mode": args.loss_mode,
                "freeze_backbone": bool(args.freeze_backbone),
                "compound_val": use_compound_val,
                "a_floor_clot_f1": a_floor_clot_f1,
                "env_overrides": collect_active_env_overrides(),
            },
        )
    print(f"[OK] Training complete. Best score: {best_score:.3f}. Output saved to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
