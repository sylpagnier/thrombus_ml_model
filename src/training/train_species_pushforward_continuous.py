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
import shutil
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
from src.core_physics.species_deploy_rollout import band_uv_for_model
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
    deploy_eval_time_fracs,
    graph_last_time_index,
    legacy_capped_deploy_time_index,
    deploy_eval_dual_full_weight,
    deploy_horizon_aux_all_packs,
    deploy_horizon_aux_cap_steps,
    train_deploy_eval_flow_source,
    use_vessel_mat_max,
    continuous_teacher_fp_frac,
    continuous_teacher_noise_sigma,
    continuous_vel_decay_enabled,
    continuous_vel_decay_wall_only,
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
    ensure_band_mesh_priors,
    flow_feats_drop_xy,
    flow_feats_dynamic,
    flow_feats_enabled,
    flux_stag_feats_enabled,
    geom_feats_enabled,
    geom_feats_rich_enabled,
    overlay_flow_series_for_window,
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


def _flow_series_for_unroll(
    data,
    static: dict,
    device: torch.device,
    time_window: list[int],
    velocity_series: list[torch.Tensor],
) -> torch.Tensor | None:
    """Deploy-faithful window overlay: match flow feats to ``band_uv_for_model`` UV.

    Pack ``flow_series`` stays as the GT path when ``flow_feats_source=gt``; otherwise
    absolute timeline rows for ``time_window`` are overwritten from ``velocity_series``.
    """
    base = static.get("flow_series")
    if not flow_feats_dynamic() or static.get("flow_cols") is None:
        return base
    return overlay_flow_series_for_window(
        base,
        data,
        device,
        static["node_idx"],
        time_window,
        velocity_series,
    )


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
    mirror_y: bool = False,
) -> dict:
    kine_stem = Path(kine_model.config.ckpt_path).stem if getattr(kine_model, "config", None) and hasattr(kine_model.config, "ckpt_path") else "deploy"
    suffix = "_mirror_y" if mirror_y else ""
    # Band-feature tag is required: geom/flux legs must not reuse control packs (same hops).
    from src.core_physics.species_pushforward_gnn import band_extra_feature_dim, band_feats_cache_tag

    feat_tag = band_feats_cache_tag()
    # Prior source is part of the pack identity: the DEQ latent baked into the cache is
    # conditioned on the prior columns, so reusing a `stored` pack under `analytic` would
    # silently preserve the leak (vessel_scope.prior_source_cache_tag).
    from src.core_physics.vessel_scope import prior_source_cache_tag

    pack_cache_dir = (
        root / ".cache" / "packs"
        / f"{kine_stem}_hops{wall_hops}_{feat_tag}{suffix}{prior_source_cache_tag()}"
    )
    pack_cache_path = pack_cache_dir / f"{anchor.strip()}{suffix}.pt"

    data = None
    static = None
    if pack_cache_path.exists():
        print(f"[i] Pre-loaded species graph cache for {anchor.strip()}{suffix}", flush=True)
        cached = torch.load(pack_cache_path, map_location="cpu", weights_only=False)
        data = cached["data"]
        static = cached["static"]
        # Refuse silent reuse of stale packs (wrong band extras / old pre-featfix caches).
        packed_w = int(static["base_feats"].shape[1])
        latent_w = int(static.get("latent_dim") or (packed_w - 1))
        expect_w = int(latent_w) + 1 + int(band_extra_feature_dim())
        if packed_w != expect_w:
            print(
                f"[WARN] stale pack {pack_cache_path.name}: base_feats width={packed_w} "
                f"!= expected {expect_w} (tag={feat_tag}); rebuilding",
                flush=True,
            )
            data = None
            static = None
        elif static is not None and data is not None:
            ensure_band_mesh_priors(static, data)

    if static is None:
        graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor.strip()}{suffix}.pt"
        data = torch.load(graph_path, map_location="cpu", weights_only=False)

        # Legal priors BEFORE the solve: the DEQ consumes UV_PRIOR / MU_PRIOR, so rewriting
        # them afterwards would leave z_kin conditioned on the leaked CFD field (16.1c).
        from src.core_physics.vessel_scope import prepare_vessel_data

        data, _ = prepare_vessel_data(data)

        # One GINO-DEQ solve per vessel: UV baseline + z_kin (joint cache). Local corrector uses UV later.
        from src.utils.kinematics_inference import predict_kinematics_and_latent

        data_dev = data.to(device)
        cache_dir = root / ".cache" / "kinematics_t0" / kine_stem

        with torch.no_grad():
            pred_uv, z_kin = predict_kinematics_and_latent(
                kine_model, 
                data_dev,
                disk_cache_dir=cache_dir,
                disk_cache_key=anchor.strip()
            )
        data.u0_pred = pred_uv[:, 0].to("cpu").clone()
        data.v0_pred = pred_uv[:, 1].to("cpu").clone()

        static = build_band_base_features(
            data,
            kine_model,
            device,
            wall_hops=wall_hops,
            z_kin_override=z_kin,
        )

        static_cpu = {k: (v.to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in static.items()}
        pack_cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"data": data.to("cpu"), "static": static_cpu}, pack_cache_path)
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

    from src.core_physics.vessel_scope import resolve_vessel_mat_max

    return {
        "anchor": anchor.strip(),
        "data": data,
        # Per-vessel label scale travels WITH the pack, so no loop can forget to bind it.
        # None on a clot-free pack -> mat_label_thresh() falls back to the absolute threshold.
        "mat_max": resolve_vessel_mat_max(data),
        "static": static_cpu,
        "train_m": train_m,
        "val_m": val_m,
        "windows": windows,
        "train_t0_max": pack_t0_max,
        "val_windows": _val_windows(static_cpu, unroll=unroll, stride=stride),
        "phys": phys,
        "bio": bio,
    }


# Window metrics from ``eval_continuous_window`` are teacher-forced: it is handed GT velocity
# and GT species blocks. They are debugging aids only and must never steer checkpoint choice.
GT_FED_DIAGNOSTIC_KEYS = (
    "val_state_f1",
    "val_mat_f1",
    "val_growth_f1",
    "val_growth_mat_f1",
    "val_init_f1",
    "val_clot_phi_f1",
)


def select_checkpoint_score(
    row: dict,
    *,
    held_out_val: bool,
    physics_on: bool,
    mat_precision_select: bool,
    deploy_clot_score: float,
    deploy_mat_f1: float,
    deploy_clot_pred_pos_frac: float,
    clot_weight: float,
    deploy_clot_mass_ratio: float = 1.0,
    select_clot_score_weight: float = 0.70,
    select_mat_f1_weight: float = 0.30,
    select_clot_f1_weight: float = 0.0,
    select_mass_soft_lambda: float = 0.0,
    select_mass_soft_target: float = 1.2,
    select_mass_hard_max: float = 0.0,
    select_mass_hard_min: float = 0.0,
    select_overpaint_lambda: float = 0.0,
    select_overpaint_frac_target: float = 0.08,
    select_seed_prec_lambda: float = 0.0,
    select_front_speed_lambda: float = 0.0,
    select_fn_fp_lambda: float = 0.0,
    select_fn_hard_max: float = 0.0,
    select_f1_min_hard_floor: float = 0.0,
    select_front_speed_target_lambda: float = 0.0,
    select_fp_fn_imbalance_lambda: float = 0.0,
) -> tuple[float, str]:
    """Checkpoint-selection score and the mode used.

    With a held-out val anchor the score is built *only* from closed-loop deploy metrics: the
    ``val_*`` window metrics are teacher-forced on GT, so letting them select a checkpoint means
    picking the model that best replays GT rather than the one that rolls out best unaided.

    Soft mass / overpaint terms borrow the mat_precision overpaint idea into held-out selection.
    Optional seed-panel bonuses (mat_seed_prec / front speed / FN-FP) stay deploy-faithful.
    Hard mass reject is catastrophe-only (disabled when ``select_mass_hard_max <= 0``).
    ``select_mass_hard_min`` rejects precision-mirage starvation (mass too low, score inflated).
    When ``select_clot_f1_weight > 0``, strict ``deploy_clot_f1`` is the primary term (wall-gen gate).
    ``select_f1_min_hard_floor`` rejects on ``deploy_clot_f1_min`` (s9.8 sliding-window grading,
    ``deploy_eval_time_fracs``) -- a checkpoint that looks fine at t_final but has already
    collapsed earlier in the rollout must not be promoted on the strength of the final point
    alone. No-op (0.0) unless sliding-window grading is active, since that field is otherwise
    identical to the single-point ``deploy_clot_f1``.
    """
    if held_out_val:
        mass = float(deploy_clot_mass_ratio)
        hard_max = float(select_mass_hard_max)
        hard_min = float(select_mass_hard_min)
        if hard_max > 0.0 and mass > hard_max:
            # Catastrophe spray: never promote; keep early-stop tied to a prior valid best.
            return -1.0e12, "deploy_only_mass_reject"
        if hard_min > 0.0 and mass < hard_min:
            # Starvation / precision mirage (score up, F1 down, mass << 1).
            return -1.0e12, "deploy_only_mass_reject"
        fn_hard = float(select_fn_hard_max)
        if fn_hard > 0.0:
            fn = float(row.get("deploy_clot_fn", row.get("clot_fn_median", 0.0)) or 0.0)
            if fn > fn_hard:
                return -1.0e12, "deploy_only_fn_reject"
        f1_min_floor = float(select_f1_min_hard_floor)
        if f1_min_floor > 0.0:
            f1_min = row.get("deploy_clot_f1_min")
            if f1_min is not None and float(f1_min) < f1_min_floor:
                return -1.0e12, "deploy_only_f1_min_reject"
        w_f1 = max(float(select_clot_f1_weight), 0.0)
        w_clot = max(float(select_clot_score_weight), 0.0)
        w_mat = max(float(select_mat_f1_weight), 0.0)
        w_sum = w_f1 + w_clot + w_mat
        if w_sum <= 0.0:
            w_clot, w_mat, w_sum = 0.70, 0.30, 1.0
            w_f1 = 0.0
        clot_f1 = float(row.get("deploy_clot_f1", 0.0) or 0.0)
        deploy_score = (
            w_f1 * clot_f1 + w_clot * float(deploy_clot_score) + w_mat * float(deploy_mat_f1)
        ) / w_sum
        lam_m = max(float(select_mass_soft_lambda), 0.0)
        if lam_m > 0.0:
            deploy_score -= lam_m * max(0.0, mass - float(select_mass_soft_target))
            # Also penalize under-mass (symmetric soft band around target).
            soft_lo = max(0.5, float(select_mass_soft_target) - 0.4)
            deploy_score -= lam_m * max(0.0, soft_lo - mass)
        lam_o = max(float(select_overpaint_lambda), 0.0)
        if lam_o > 0.0:
            overpaint = max(0.0, float(deploy_clot_pred_pos_frac) - float(select_overpaint_frac_target))
            deploy_score -= lam_o * overpaint
        lam_s = max(float(select_seed_prec_lambda), 0.0)
        if lam_s > 0.0:
            deploy_score += lam_s * float(row.get("mat_seed_prec", 0.0) or 0.0)
        lam_f = max(float(select_front_speed_lambda), 0.0)
        if lam_f > 0.0:
            # Cap so runaway front speed cannot dominate clot score.
            front = min(float(row.get("mat_front_speed_ratio", 0.0) or 0.0), 1.5)
            deploy_score += lam_f * front
        lam_e = max(float(select_fn_fp_lambda), 0.0)
        if lam_e > 0.0:
            fn = float(row.get("deploy_clot_fn", row.get("clot_fn_median", 0.0)) or 0.0)
            fp = float(row.get("deploy_clot_fp", row.get("clot_fp_median", 0.0)) or 0.0)
            # Penalize FN-heavy underseed relative to FP (deploy counts when present).
            deploy_score -= lam_e * max(0.0, fn - fp) / max(fn + fp, 1.0)
        lam_ft = max(float(select_front_speed_target_lambda), 0.0)
        if lam_ft > 0.0:
            # Uncapped, symmetric: penalizes distance from front_speed=1.0 either direction,
            # unlike select_front_speed_lambda above (monotonic reward, saturates+misdirects
            # once front_speed > 1.5 -- s9.10).
            front_raw = float(row.get("mat_front_speed_ratio", 1.0) or 1.0)
            deploy_score -= lam_ft * abs(front_raw - 1.0)
        lam_imb = max(float(select_fp_fn_imbalance_lambda), 0.0)
        if lam_imb > 0.0:
            # Symmetric counterpart to select_fn_fp_lambda above (which only fires FN-heavy).
            fn_i = float(row.get("deploy_clot_fn", row.get("clot_fn_median", 0.0)) or 0.0)
            fp_i = float(row.get("deploy_clot_fp", row.get("clot_fp_median", 0.0)) or 0.0)
            deploy_score -= lam_imb * abs(fn_i - fp_i) / max(fn_i + fp_i, 1.0)
        if row.get("deploy_eval_t", -1) > 0:
            return deploy_score, "deploy_only"
        return -row["loss"], "lowest_loss"
    if physics_on:
        score = (
            0.50 * row["val_clot_phi_f1"]
            + 0.25 * row["val_growth_f1"]
            + 0.15 * row["val_state_f1"]
            + 0.10 * row["val_growth_mat_f1"]
        )
        return score, "physics_gt_fed"
    if mat_precision_select:
        overpaint = max(0.0, deploy_clot_pred_pos_frac - 0.08)
        mat_score = 0.40 * deploy_mat_f1 + 0.10 * row["val_growth_f1"]
        score = clot_weight * deploy_clot_score + (1.0 - clot_weight) * mat_score - 0.25 * overpaint
        return score, "mat_precision"
    mat_score = (
        0.70 * deploy_mat_f1
        + 0.15 * row["val_growth_f1"]
        + 0.10 * row["val_state_f1"]
        + 0.05 * row["val_growth_mat_f1"]
    )
    if clot_weight > 0.0:
        return ((1.0 - clot_weight) * mat_score + clot_weight * deploy_clot_score), "mat_clot_mix"
    return mat_score, "mat_only"


def _held_out_select_kwargs_from_runtime() -> dict:
    """Typed scoring knobs for held-out deploy_only (defaults = legacy formula)."""
    out = {
        "select_clot_score_weight": 0.70,
        "select_mat_f1_weight": 0.30,
        "select_clot_f1_weight": 0.0,
        "select_mass_soft_lambda": 0.0,
        "select_mass_soft_target": 1.2,
        "select_mass_hard_max": 0.0,
        "select_mass_hard_min": 0.0,
        "select_overpaint_lambda": 0.0,
        "select_overpaint_frac_target": 0.08,
        "select_seed_prec_lambda": 0.0,
        "select_front_speed_lambda": 0.0,
        "select_fn_fp_lambda": 0.0,
        "select_fn_hard_max": 0.0,
        "select_f1_min_hard_floor": 0.0,
        "select_front_speed_target_lambda": 0.0,
        "select_fp_fn_imbalance_lambda": 0.0,
    }
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is None:
            return out
        sc = rt.scoring
        out.update(
            {
                "select_clot_score_weight": float(sc.select_clot_score_weight),
                "select_mat_f1_weight": float(sc.select_mat_f1_weight),
                "select_clot_f1_weight": float(sc.select_clot_f1_weight),
                "select_mass_soft_lambda": float(sc.select_mass_soft_lambda),
                "select_mass_soft_target": float(sc.select_mass_soft_target),
                "select_mass_hard_max": float(sc.select_mass_hard_max),
                "select_mass_hard_min": float(sc.select_mass_hard_min),
                "select_overpaint_lambda": float(sc.select_overpaint_lambda),
                "select_overpaint_frac_target": float(sc.select_overpaint_frac_target),
                "select_seed_prec_lambda": float(sc.select_seed_prec_lambda),
                "select_front_speed_lambda": float(sc.select_front_speed_lambda),
                "select_fn_fp_lambda": float(sc.select_fn_fp_lambda),
                "select_fn_hard_max": float(sc.select_fn_hard_max),
                "select_f1_min_hard_floor": float(sc.select_f1_min_hard_floor),
                "select_front_speed_target_lambda": float(sc.select_front_speed_target_lambda),
                "select_fp_fn_imbalance_lambda": float(sc.select_fp_fn_imbalance_lambda),
            }
        )
    except Exception:
        pass
    return out


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
    ap.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze SAGE/conv trunk; train spatial/magnitude heads only (light fine-tune)",
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
    ap.add_argument("--deploy-freq", type=int, default=0, help="Evaluate deploy metrics every N epochs")
    ap.add_argument(
        "--recipe",
        choices=("default", "mat_growth_simple"),
        default="default",
        help="Training recipe overrides (mat_growth_simple = Mat-only single-head)",
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
        default="",
        help="Mat-growth ladder leg (applies per-leg env overrides)",
    )
    ap.add_argument(
        "--arch",
        choices=("sage", "gat"),
        default="",
        help="Pushforward trunk: sage=GraphSAGE (default), gat=GATv2",
    )
    # Anti-memorization overrides (applied AFTER leg env to survive force=True reset).
    ap.add_argument(
        "--drop-xy",
        action="store_true",
        help="Zero x_norm/y_norm in flow features (anti spatial memorization)",
    )
    ap.add_argument(
        "--latent-dropout",
        type=float,
        default=None,
        help="Probability of zeroing z_kin per step (anti latent memorization)",
    )
    ap.add_argument(
        "--geom-rich",
        action="store_true",
        help="Enable 2-hop width expansion + curvature geometry features",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override weight decay for the optimizer",
    )
    ap.add_argument(
        "--list-legs",
        default="",
        help="Print matching ladder leg codes (e.g. v3) and exit",
    )
    args = ap.parse_args()

    if args.list_legs.strip():
        try:
            from src.biochem_gnn.mat_growth_simple import LADDER_LEG_ORDER
            filter_str = args.list_legs.strip()
            # Prefer prefix (WG_sweep_v3 -> WG_sweep_v3_*), else substring.
            matches = [leg for leg in LADDER_LEG_ORDER if leg.startswith(filter_str)]
            if not matches:
                fl = filter_str.lower()
                matches = [leg for leg in LADDER_LEG_ORDER if fl in leg.lower()]
            for leg in matches:
                print(leg)
            return 0
        except Exception as e:
            print(f"[ERROR] Failed to list legs: {e}")
            return 1

    cli_arch = str(args.arch).strip().lower()

    phase = "biochem_gnn"
    mat_growth_recipe = str(args.recipe).strip().lower() == "mat_growth_simple"
    leg_name = str(args.leg).strip()
    leg_config_kwargs: dict = {}
    leg_runtime_kwargs: dict = {}
    leg_no_init = False
    if mat_growth_recipe:
        from src.biochem_gnn.mat_growth_simple import (
            apply_mat_growth_leg_env,
            apply_mat_growth_simple_recipe_env,
            get_mat_growth_config_kwargs,
            get_mat_growth_runtime_kwargs,
            mat_growth_leg_spec,
        )

        # Residual unknown env only; architecture + runtime are typed below.
        if leg_name:
            apply_mat_growth_leg_env(leg_name, force=True)
            leg_config_kwargs = get_mat_growth_config_kwargs(leg_name)
            leg_runtime_kwargs = get_mat_growth_runtime_kwargs(leg_name)
            leg_no_init = bool(mat_growth_leg_spec(leg_name).no_init)
        else:
            apply_mat_growth_simple_recipe_env(force=True)
    else:
        apply_train_recipe_env()

    from src.architecture.pushforward_config import (
        PushforwardConfig,
        get_active_config,
        use_pushforward_config,
    )
    from src.architecture.runtime_config import (
        BiochemRuntimeConfig,
        get_active_runtime,
        use_biochem_runtime,
    )
    import dataclasses as _dc

    # Prefer configs already bound by apply_*_env; from_env is legacy fallback only.
    pushforward_config = get_active_config() or PushforwardConfig.from_env()
    if leg_config_kwargs:
        pushforward_config = _dc.replace(pushforward_config, **leg_config_kwargs)
    # CLI anti-memorization overrides win over leg defaults.
    cli_cfg: dict = {}
    if cli_arch:
        cli_cfg["arch"] = cli_arch
    if args.drop_xy:
        cli_cfg["flow_feats_drop_xy"] = True
    if args.geom_rich:
        cli_cfg["geom_feats"] = True
        cli_cfg["geom_feats_rich"] = True
    if args.unroll is not None:
        cli_cfg["unroll"] = int(args.unroll)
    if args.stride is not None:
        cli_cfg["step_stride"] = int(args.stride)
    if bool(getattr(args, "freeze_backbone", False)):
        cli_cfg["freeze_backbone"] = True
    if cli_cfg:
        pushforward_config = _dc.replace(pushforward_config, **cli_cfg)

    runtime_config = get_active_runtime() or BiochemRuntimeConfig.from_env()
    # Default train safety: GT velocity + no local-corrector coupling (historical speed crutch).
    # Legs may opt into deploy-faithful train by setting runtime_kwargs AFTER this baseline.
    runtime_config = runtime_config.with_overrides(
        closed_loop_coupling=False,
        corrector_coupling=False,
        rollout_vel_source="gt",
        train_vel_source="gt",
    )
    if leg_runtime_kwargs:
        runtime_config = runtime_config.with_overrides(**leg_runtime_kwargs)
    if args.latent_dropout is not None:
        runtime_config = runtime_config.with_overrides(latent_dropout=float(args.latent_dropout))
    if args.wall_hops is not None:
        runtime_config = runtime_config.with_overrides(wall_hops=int(args.wall_hops))
    try:
        pushforward_config.validate()
    except ValueError as e:
        print(f"[ERROR] Invalid config: {e}")
        return 1

    mat_precision_select = False
    if mat_growth_recipe:
        from src.biochem_gnn.mat_growth_simple import mat_growth_precision_selection_enabled

        mat_precision_select = bool(runtime_config.scoring.precision_select) or mat_growth_precision_selection_enabled()

    # Bind typed configs for helpers that still accept optional / context resolution.
    _cfg_cm = use_pushforward_config(pushforward_config)
    _cfg_cm.__enter__()
    _rt_cm = use_biochem_runtime(runtime_config)
    _rt_cm.__enter__()

    pushforward_arch = pushforward_config.arch

    unroll = pushforward_unroll_steps(pushforward_config)
    max_unroll = pushforward_max_unroll_steps(pushforward_config)
    stride = pushforward_step_stride(pushforward_config)
    wall_hops = snapshot_wall_hops()
    hidden = snapshot_hidden_dim() if args.hidden is None else max(int(args.hidden), 16)
    ch_w = continuous_channel_weights()
    huber_b = continuous_huber_beta(pushforward_config)
    t0_max = (
        None
        if pushforward_train_t0_per_vessel(pushforward_config)
        else pushforward_train_t0_max(pushforward_config)
    )
    growth_only = continuous_growth_only_loss(pushforward_config)
    loss_scale = continuous_loss_scale(pushforward_config)
    delta_thr = continuous_delta_threshold(pushforward_config)
    fp_w = continuous_fp_weight(pushforward_config)
    physics_on = bool(pushforward_config.physics_readout)
    dual_head = continuous_dual_head(pushforward_config)
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
        # Held-out selection needs deploy metrics every epoch; default --deploy-freq 0 only
        # evaluates on the final epoch, so early-stop falls back to lowest_loss (-loss).
        if int(args.deploy_freq) <= 0:
            args.deploy_freq = 1
            print(
                "[i] exclude-val-from-train: forcing --deploy-freq 1 "
                "(deploy-only checkpoint selection)",
                flush=True,
            )

    kine_ckpt = str(resolve_kinematics_checkpoint())
    kine_model = load_kinematics_predictor(
        kine_ckpt, device, phys_cfg=PhysicsConfig(phase="kinematics"), cache=False
    )

    mirror_y_flag = bool(getattr(runtime_config.rollout, "augment_mirror_y", False)) or (
        os.environ.get("SPECIES_AUGMENT_MIRROR_Y", "0") == "1"
    )
    packs: list[dict] = []
    for anc in train_anchors:
        # Always train on the native graph; mirror-Y is augmentation (double the set), not a replace.
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
                mirror_y=False,
            )
        )
        if mirror_y_flag:
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
                    mirror_y=True,
                )
            )
        import gc
        gc.collect()
    if mirror_y_flag:
        print(
            f"[i] mirror-Y augment: {len(train_anchors)} native + {len(train_anchors)} mirrored packs "
            f"(n_packs={len(packs)})",
            flush=True,
        )

    # Held-out val has no pack in `packs` by construction (it was dropped from train_anchors
    # above). Build it here, while kine_model is still resident, and keep it OUT of `packs` so
    # the optimizer can never see it.
    heldout_val_pack: dict | None = None
    if bool(args.exclude_val_from_train):
        heldout_val_pack = _build_anchor_pack(
            val_anchor,
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
            mirror_y=False,
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

    if heldout_val_pack is not None:
        val_pack = heldout_val_pack
    else:
        val_pack = next((p for p in packs if p["anchor"] == val_anchor), None)
    if val_pack is None or val_pack["anchor"] != val_anchor:
        # Never silently substitute another vessel: the old `next(..., packs[0])` fallback made
        # --exclude-val-from-train score the first *training* anchor, so every held-out number
        # was really train performance (docs/GENERALIZATION_PLAN.md s2b-sexies).
        got = None if val_pack is None else val_pack["anchor"]
        raise ValueError(
            f"val pack resolves to {got!r}, expected {val_anchor!r}; refusing to validate on "
            "the wrong vessel"
        )
    print(f"[i] val pack = {val_pack['anchor']} (held_out={heldout_val_pack is not None})", flush=True)
    ref_static = packs[0]["static"]
    # Prefer true z_kin width; fall back to legacy (base_feats - sdf) only if missing.
    latent_dim = int(ref_static.get("latent_dim") or (int(ref_static["base_feats"].shape[1]) - 1))
    in_dim = continuous_feature_dim(latent_dim)
    from src.core_physics.species_pushforward_gnn import band_extra_feature_dim, band_feats_cache_tag

    packed_w = int(ref_static["base_feats"].shape[1])
    expect_base = int(latent_dim) + 1 + int(band_extra_feature_dim())
    if packed_w != expect_base:
        raise RuntimeError(
            f"pack base_feats width={packed_w} != expected {expect_base} "
            f"(latent={latent_dim}, band_extras={band_extra_feature_dim()}, "
            f"tag={band_feats_cache_tag()}); "
            "refusing to train with mismatched band features (would silent-pad/truncate). "
            "Delete stale .cache/packs entries for this feat tag and retry."
        )
    print(
        f"[i] in_dim={in_dim} latent={latent_dim} band_extras={band_extra_feature_dim()} "
        f"pack_base={packed_w}",
        flush=True,
    )
    model = build_continuous_gnn(in_dim, hidden=hidden, arch=pushforward_arch, config=pushforward_config).to(device)
    # Latent leash: tell the model the z_kin slice width + dropout prob so the training forward can
    # stochastically zero the (clot-blind) latent and force reliance on the explicit flow features.
    model.kin_latent_dim = int(ref_static.get("latent_dim", 0) or 0)
    model.latent_dropout_p = species_latent_dropout_p()
    if model.latent_dropout_p > 0.0:
        print(
            f"[i] latent leash: dropout p={model.latent_dropout_p:.2f} on z_kin[:{model.kin_latent_dim}]",
            flush=True,
        )

    skip_init = bool(args.no_init) or bool(leg_no_init)
    init_ckpt = "" if skip_init else (args.init.strip() or str(root / DEFAULT_S34_CKPT))
    if skip_init:
        why = "--no-init" if args.no_init else f"leg {leg_name} no_init"
        print(f"[i] random init ({why})", flush=True)
    elif init_ckpt and Path(init_ckpt).is_file():
        init_meta = {}
        init_payload: dict = {}
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
                    src_in = int(
                        init_payload.get("in_dim")
                        or getattr(bundle.model, "in_dim", 0)
                        or 0
                    )
                    load_pushforward_state_dict_partial(
                        model,
                        bundle.model.state_dict(),
                        quiet=False,
                        src_in_dim=src_in if src_in > 0 else None,
                    )
                print(f"[OK] warm-start from {init_ckpt}", flush=True)
    else:
        init_path = args.init_s1.strip() or str(root / DEFAULT_SNAPSHOT_CKPT)
        if Path(init_path).is_file():
            init_continuous_from_snapshot(model, init_path)
    if mat_growth_recipe and leg_name:
        apply_mat_growth_leg_env(leg_name, force=True)
        # Re-apply CLI + train-safe defaults, then leg opt-in (deploy-faithful A/B).
        pushforward_config = get_active_config() or pushforward_config
        runtime_config = get_active_runtime() or runtime_config
        if leg_config_kwargs:
            pushforward_config = _dc.replace(pushforward_config, **leg_config_kwargs)
        if cli_cfg:
            pushforward_config = _dc.replace(pushforward_config, **cli_cfg)
        runtime_config = runtime_config.with_overrides(
            closed_loop_coupling=False,
            corrector_coupling=False,
            rollout_vel_source="gt",
            train_vel_source="gt",
        )
        if leg_runtime_kwargs:
            runtime_config = runtime_config.with_overrides(**leg_runtime_kwargs)
    if args.latent_dropout is not None:
        runtime_config = runtime_config.with_overrides(latent_dropout=float(args.latent_dropout))
    try:
        _rt_cm.__exit__(None, None, None)
    except Exception:
        pass
    _rt_cm = use_biochem_runtime(runtime_config)
    _rt_cm.__enter__()
    try:
        _cfg_cm.__exit__(None, None, None)
    except Exception:
        pass
    _cfg_cm = use_pushforward_config(pushforward_config)
    _cfg_cm.__enter__()

    freeze_backbone = bool(getattr(pushforward_config, "freeze_backbone", False))
    if freeze_backbone:
        from src.training.train_offwall_growth import freeze_growth_backbone

        n_fr, n_tr = freeze_growth_backbone(model)
        print(f"[i] freeze_backbone: frozen={n_fr} trainable_heads={n_tr}", flush=True)
        
    if str(args.recipe).strip().lower() == "mat_growth_simple":
        from src.biochem_gnn.mat_growth_simple import recipe_fingerprint
        import pprint
        print("\n[i] Resolved Configuration Fingerprint:")
        pprint.pprint(recipe_fingerprint(), indent=2, width=120)
        print()

    # Only bias-init readout on cold/random starts. Warm-start already has trained head
    # biases; overwriting them (esp. with freeze_backbone) forces spray/overpaint.
    warmed = bool(args.init.strip()) and not bool(args.no_init)
    if not warmed and not freeze_backbone:
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
    elif warmed:
        print("[i] skip readout bias fill (warm-start preserves head biases)", flush=True)

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

    wd = float(args.weight_decay) if getattr(args, "weight_decay", None) is not None else 1e-5
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters (freeze_backbone left an empty optimizer?)")
    opt = optim.Adam(trainable, lr=lr, weight_decay=wd)
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
        "pushforward_config": pushforward_config.to_env(),
        "unroll": unroll,
        "max_unroll": max_unroll,
        "tbptt_tail": tbptt_tail_steps(),
        "vel_decay": continuous_vel_decay_enabled(),
        "vel_decay_wall_only": continuous_vel_decay_wall_only(),
        "flow_feats": flow_feats_enabled(),
        "flow_dynamic": flow_feats_dynamic(),
        "flow_drop_xy": flow_feats_drop_xy(),
        "geom_feats": geom_feats_enabled(),
        "geom_feats_rich": geom_feats_rich_enabled(),
        "flux_stag_feat": flux_stag_feats_enabled(),
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
        "leg": leg_name,
        "config_kwargs": dict(leg_config_kwargs),
        "runtime_kwargs": dict(leg_runtime_kwargs),
        "env_overrides": (
            dict(
                __import__(
                    "src.biochem_gnn.mat_growth_simple", fromlist=["mat_growth_leg_spec"]
                ).mat_growth_leg_spec(leg_name).env_overrides
            )
            if (mat_growth_recipe and leg_name)
            else {}
        ),
    }

    best_score = -1e9
    stale = 0
    dead_phi_epochs = 0
    # Salvage retention (WALL_MODEL_PLAN.md s12.4/s12.5 item 1). The selection gates
    # (mass/FN hard reject) are correct as *selection* policy but they were also, silently,
    # the *retention* policy: a rejected epoch hits `continue` below and no weights are ever
    # written. Five consecutive stenosis sub-cohort legs (v2..v6) rejected every epoch and
    # left zero best.pth between them -- including v2 ep5 and v3 ep5 at deploy F1 0.6155 /
    # 0.6125, the two best states the cohort has ever reached, now unrecoverable.
    # This tracks the best raw deploy_clot_score across ALL epochs, gate or no gate, and
    # writes best_salvage.pth. It does NOT touch best_score, early stop, or which epoch the
    # gate calls best -- selection semantics are unchanged.
    salvage_score = -1e9
    salvage_epoch = -1
    salvage_path = out_path.parent / "best_salvage.pth"
    if bool(args.exclude_val_from_train):
        print(
            f"[i] select=deploy_only (val {args.val_anchor} held out); "
            f"GT-teacher-forced diagnostics excluded from selection: "
            f"{', '.join(GT_FED_DIAGNOSTIC_KEYS)}",
            flush=True,
        )
    fh = int(continuous_frontier_hops())
    tk = float(continuous_nucleation_topk())
    if fh > 0 or tk > 0.0:
        print(
            f"[i] sparse_commit ON frontier_hops={fh} nucleation_topk={tk:.4g} "
            f"gate_temp={float(continuous_gate_temp()):.4g} "
            "(train-time seed-then-frontier mask)",
            flush=True,
        )
    else:
        print("[i] sparse_commit OFF (frontier_hops=0, nucleation_topk=0)", flush=True)
    from src.core_physics.species_pushforward_continuous import (
        continuous_seed_aux_compact_weight,
        continuous_seed_aux_early_steps,
        continuous_seed_aux_weight,
    )

    saw = float(continuous_seed_aux_weight())
    if saw > 0.0:
        print(
            f"[i] seed_aux ON weight={saw:.4g} early_steps={int(continuous_seed_aux_early_steps())} "
            f"compact={float(continuous_seed_aux_compact_weight()):.4g} "
            "(location/early-commit; not a mass term)",
            flush=True,
        )
    from src.core_physics.species_pushforward_continuous import (
        continuous_pocket_contrast_early_steps,
        continuous_pocket_contrast_hops,
        continuous_pocket_contrast_inside_weight,
        continuous_pocket_contrast_weight,
    )

    pcw = float(continuous_pocket_contrast_weight())
    if pcw > 0.0:
        print(
            f"[i] pocket_contrast ON weight={pcw:.4g} hops={int(continuous_pocket_contrast_hops())} "
            f"early_steps={int(continuous_pocket_contrast_early_steps())} "
            f"inside_w={float(continuous_pocket_contrast_inside_weight()):.4g} "
            "(exclusive wrong-pocket soft loss; no hard forward mask)",
            flush=True,
        )
    from src.core_physics.species_pushforward_continuous import (
        continuous_gate_fp_weight,
        continuous_underpred_weight,
    )

    # Mechanism-engaged line for the per-vessel label scale (22.2). Without this, a rel_max
    # run that failed to bind mat_max would look identical to an absolute run in the log.
    try:
        from src.core_physics.species_pushforward_continuous import mat_label_thresh
        _lab_mode = getattr(pushforward_config, "mat_label_thresh_mode", "absolute")
        _samples = [(p["anchor"], p.get("mat_max")) for p in packs[:3]]
        with use_vessel_mat_max(_samples[0][1] if _samples else None):
            _thr = mat_label_thresh()
        print(
            f"[i] label threshold: mode={_lab_mode} frac="
            f"{getattr(pushforward_config, 'mat_label_rel_frac', 0.0):.3g} "
            f"-> {_samples[0][0] if _samples else 'n/a'} thr={_thr:.3e} "
            f"(mat_max per pack: "
            + ", ".join(f"{a}={('%.2e' % m) if m else 'None'}" for a, m in _samples) + ")",
            flush=True,
        )
    except Exception as _e:
        print(f"[WARN] could not report label threshold: {_e}", flush=True)

    print(
        f"[i] underpred_weight={float(continuous_underpred_weight()):.4g} "
        f"gate_fp_weight={float(continuous_gate_fp_weight()):.4g}",
        flush=True,
    )
    t0 = time.perf_counter()

    for ep in range(1, int(args.epochs) + 1):
        t0_ep = time.time()
        model.train()
        ep_losses: list[float] = []
        cur_unroll = curriculum_unroll_for_epoch(ep)
        pack_order = packs[:]
        random.shuffle(pack_order)

        for pack in pack_order:
          # Bind this vessel's label scale for every loss computed on its windows. Without
          # this, mat_label_thresh() silently falls back to the absolute threshold and
          # `rel_max` becomes a no-op -- the exact failure mode of v4/v5 (12.3).
          with use_vessel_mat_max(pack.get("mat_max")):
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
                  # Deploy-faithful UV for model.velocity (resolve via train_vel_source).
                  # Never feed raw COMSOL data.y UV into architecture paths.
                  velocity_series = [
                      band_uv_for_model(
                          pack_data_gpu, ti, device, static_gpu["node_idx"], for_training=True
                      )
                      for ti in win_use
                  ]
                  species_block_full = [pack_data_gpu.y[ti, static_gpu["node_idx"], sc.SPECIES_BLOCK] for ti in win_use]
                  flow_series_win = _flow_series_for_unroll(
                      pack_data_gpu, static_gpu, device, win_use, velocity_series
                  )
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
                      flow_series=flow_series_win,
                      flow_cols=static_gpu.get("flow_cols"),
                      wall_mask_band=pack_data_gpu.mask_wall[static_gpu["node_idx"]] if hasattr(pack_data_gpu, "mask_wall") and pack_data_gpu.mask_wall is not None else None,
                      wall_normals_band=static_gpu.get("wall_normals_band"),
                      sdf_band=static_gpu.get("sdf_band"),
                      edge_attr_band=static_gpu.get("edge_attr_band"),
                      species_block=species_block_full,
                      velocity=velocity_series,
                      epoch=ep,
                      max_epochs=int(args.epochs),
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
            if deploy_horizon_aux_all_packs():
                dep_packs = list(packs)
            else:
                # Single-pack aux: never use the held-out val pack for backprop.
                dep_packs = [p for p in packs if p["anchor"] != val_anchor][:1]
                if not dep_packs:
                    dep_packs = list(packs[:1])
            if heldout_val_pack is not None:
                # This aux term backprops, so a held-out anchor must never appear here.
                dep_packs = [p for p in dep_packs if p["anchor"] != val_anchor]
            aux_cap = deploy_horizon_aux_cap_steps()
            for vpack in dep_packs:
                # Bind this vessel's label scale for the deploy-horizon aux loss too, so the
                # aux and the main loop grade against the same label definition.
                with use_vessel_mat_max(vpack.get("mat_max")):
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
                    velocity_series = [
                        band_uv_for_model(
                            vpack_data_gpu, ti, device, static_gpu["node_idx"], for_training=True
                        )
                        for ti in win_dep
                    ]
                    species_block_full = [vpack_data_gpu.y[ti, static_gpu["node_idx"], sc.SPECIES_BLOCK] for ti in win_dep]
                    flow_series_win = _flow_series_for_unroll(
                        vpack_data_gpu, static_gpu, device, win_dep, velocity_series
                    )
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
                        flow_series=flow_series_win,
                        flow_cols=static_gpu.get("flow_cols"),
                        wall_mask_band=vpack_data_gpu.mask_wall[static_gpu["node_idx"]] if hasattr(vpack_data_gpu, "mask_wall") and vpack_data_gpu.mask_wall is not None else None,
                        wall_normals_band=static_gpu.get("wall_normals_band"),
                        sdf_band=static_gpu.get("sdf_band"),
                        edge_attr_band=static_gpu.get("edge_attr_band"),
                        species_block=species_block_full,
                        velocity=velocity_series,
                        epoch=ep,
                        max_epochs=int(args.epochs),
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
        deploy_wall_score = 0.0
        deploy_clot_mass_ratio = 0.0
        deploy_clot_empty_gt_score = 0.0
        mat_seed_prec = 0.0
        mat_seed_count = 0.0
        mat_front_speed_ratio = 0.0
        deploy_clot_fp = 0.0
        deploy_clot_fn = 0.0
        # The reported deploy metrics must use the SAME label definition the loss trained
        # against, or selection grades against a different notion of 'committed'.
        with torch.no_grad(), use_vessel_mat_max(val_pack.get("mat_max")):
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
                velocity_series = [
                    band_uv_for_model(
                        val_data_gpu, ti, device, val_static_gpu["node_idx"], for_training=True
                    )
                    for ti in win
                ]
                species_block_full = [val_data_gpu.y[ti, val_static_gpu["node_idx"], sc.SPECIES_BLOCK] for ti in win]
                flow_series_win = _flow_series_for_unroll(
                    val_data_gpu, val_static_gpu, device, win, velocity_series
                )
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
                    flow_series=flow_series_win,
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
            deploy_mat_f1, deploy_fi_f1, deploy_clot_f1, deploy_clot_guiding = 0.0, 0.0, 0.0, 0.0
            deploy_clot_f1_min = 0.0
            row_sliding_mass_mean, row_sliding_fp_mean = 0.0, 0.0
            deploy_clot_relaxed_f05, deploy_clot_relaxed_prec, deploy_clot_relaxed_rec = 0.0, 0.0, 0.0
            deploy_clot_pred_pos_frac, deploy_clot_dil_iou, deploy_clot_score, deploy_clot_guiding_mid = 0.0, 0.0, 0.0, 0.0
            deploy_wall_score, deploy_clot_mass_ratio, deploy_clot_empty_gt_score = 0.0, 0.0, 0.0
            t_deploy = -1
            if (args.deploy_freq > 0 and ep % args.deploy_freq == 0) or ep == args.epochs:
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
                mat_seed_prec = float(dep.get("mat_seed_prec", 0.0))
                mat_seed_count = float(dep.get("mat_seed_count", 0.0))
                mat_front_speed_ratio = float(dep.get("mat_front_speed_ratio", 0.0))
                need_clot = (
                    bool(args.exclude_val_from_train)
                    or continuous_score_clot_weight() > 0.0
                    or mat_precision_select
                )
                if need_clot:
                    from src.evaluation.canonical_clot_eval import canonical_deploy_clot_metrics

                    flow_eval = train_deploy_eval_flow_source()
                    clot_times = deploy_eval_clot_times(n_val)
                    clf_by_t: dict[int, dict] = {}
                    for t_clot in clot_times:
                        # Same protocol as scripts/eval_mat_growth_simple.py (flow-cache reset,
                        # data isolation, env restore) so the numbers stay comparable.
                        clf_by_t[int(t_clot)] = canonical_deploy_clot_metrics(
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
                    sliding_fracs = deploy_eval_time_fracs()
                    if sliding_fracs and len(clf_by_t) > 1:
                        # s9.8: mean over every sliding-window point (not just t_final) as the
                        # primary metric, so a checkpoint can't look good only at the last step.
                        # deploy_clot_f1_min tracks the WORST point for select_f1_min_hard_floor.
                        #
                        # s9.10 correction: mass/FP are NOT mean-averaged here, unlike F1/score.
                        # This is a growth-only system -- mass and FP only grow over the rollout
                        # (v2's own patient043 run: mass 1.41 at t=130 -> 2.75 at t=200 within
                        # ONE epoch) -- so t_final is always at least as bad as any earlier point,
                        # and the mean systematically understates it. Averaging select_mass_hard_max
                        # against the mean let a run promote-check itself against a diluted number;
                        # anchoring the hard guards to t_final closes that (v1/v2's original,
                        # single-point semantics), while F1/score/f1_min keep the sliding-window
                        # view they were built for.
                        def _mean(key: str, default: float = 0.0) -> float:
                            vals = [float(v.get(key, default) or default) for v in clf_by_t.values()]
                            return sum(vals) / max(len(vals), 1)

                        def _score(v: dict) -> float:
                            return float(v.get("deploy_clot_score", clot_score_from_deploy_dict(v)))

                        deploy_clot_score = sum(_score(v) for v in clf_by_t.values()) / len(clf_by_t)
                        deploy_clot_f1 = _mean("deploy_clot_f1")
                        deploy_clot_f1_min = min(float(v.get("deploy_clot_f1", 0.0)) for v in clf_by_t.values())
                        deploy_clot_guiding = _mean("deploy_clot_guiding", deploy_clot_f1)
                        deploy_clot_relaxed_f05 = _mean("deploy_clot_relaxed_f05", deploy_clot_f1)
                        deploy_clot_relaxed_prec = _mean("deploy_clot_relaxed_prec")
                        deploy_clot_relaxed_rec = _mean("deploy_clot_relaxed_rec")
                        deploy_clot_pred_pos_frac = _mean("deploy_clot_pred_pos_frac")
                        deploy_clot_dil_iou = _mean("deploy_clot_dil_iou")
                        deploy_wall_score = _mean("deploy_wall_score")
                        # t_final, not mean -- see s9.10 note above. clf == clf_by_t[t_main].
                        deploy_clot_mass_ratio = float(clf.get("deploy_clot_mass_ratio", 0.0))
                        deploy_clot_fp = float(clf.get("deploy_clot_fp", 0.0))
                        deploy_clot_fn = float(clf.get("deploy_clot_fn", 0.0))
                        deploy_clot_empty_gt_score = _mean("deploy_clot_empty_gt_score")
                        # Repurposed slot: worst-point guiding score (there is no single "mid"
                        # point once grading spans N sliding windows instead of 2 fixed ones).
                        deploy_clot_guiding_mid = min(
                            float(v.get("deploy_clot_guiding", deploy_clot_f1)) for v in clf_by_t.values()
                        )
                        # Visibility only -- not fed into any guard -- so a log reader can see
                        # how much worse t_final's mass/FP are than the sliding-window mean.
                        row_sliding_mass_mean = _mean("deploy_clot_mass_ratio")
                        row_sliding_fp_mean = _mean("deploy_clot_fp")
                    else:
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
                        deploy_clot_f1_min = deploy_clot_f1
                        deploy_clot_guiding = float(clf.get("deploy_clot_guiding", deploy_clot_f1))
                        deploy_clot_relaxed_f05 = float(clf.get("deploy_clot_relaxed_f05", deploy_clot_f1))
                        deploy_clot_relaxed_prec = float(clf.get("deploy_clot_relaxed_prec", 0.0))
                        deploy_clot_relaxed_rec = float(clf.get("deploy_clot_relaxed_rec", 0.0))
                        deploy_clot_pred_pos_frac = float(clf.get("deploy_clot_pred_pos_frac", 0.0))
                        deploy_clot_dil_iou = float(clf.get("deploy_clot_dil_iou", 0.0))
                        deploy_wall_score = float(clf.get("deploy_wall_score", 0.0))
                        deploy_clot_mass_ratio = float(clf.get("deploy_clot_mass_ratio", 0.0))
                        deploy_clot_empty_gt_score = float(clf.get("deploy_clot_empty_gt_score", 0.0))
                        deploy_clot_fp = float(clf.get("deploy_clot_fp", 0.0))
                        deploy_clot_fn = float(clf.get("deploy_clot_fn", 0.0))
                        row_sliding_mass_mean, row_sliding_fp_mean = deploy_clot_mass_ratio, deploy_clot_fp
                        if len(clf_by_t) > 1:
                            mid_t = legacy_capped_deploy_time_index(n_val)
                            deploy_clot_guiding_mid = float(
                                clf_by_t[mid_t].get("deploy_clot_guiding", 0.0)
                            )
                        else:
                            deploy_clot_guiding_mid = deploy_clot_guiding
                    # Env restore + flow-cache reset now live in canonical_deploy_clot_metrics.

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
            row["mat_seed_prec"] = mat_seed_prec
            row["mat_seed_count"] = mat_seed_count
            row["mat_front_speed_ratio"] = mat_front_speed_ratio
            if bool(args.exclude_val_from_train) or continuous_score_clot_weight() > 0.0 or mat_precision_select:
                row["deploy_clot_f1"] = deploy_clot_f1
                row["deploy_clot_f1_min"] = deploy_clot_f1_min
                row["deploy_clot_mass_ratio_sliding_mean"] = row_sliding_mass_mean
                row["deploy_clot_fp_sliding_mean"] = row_sliding_fp_mean
                row["deploy_clot_f1_t53"] = deploy_clot_f1
                row["deploy_clot_guiding"] = deploy_clot_guiding
                row["deploy_clot_relaxed_f05"] = deploy_clot_relaxed_f05
                row["deploy_clot_relaxed_prec"] = deploy_clot_relaxed_prec
                row["deploy_clot_relaxed_rec"] = deploy_clot_relaxed_rec
                row["deploy_clot_pred_pos_frac"] = deploy_clot_pred_pos_frac
                row["deploy_clot_dil_iou"] = deploy_clot_dil_iou
                row["deploy_clot_score"] = deploy_clot_score
                row["deploy_clot_guiding_mid"] = deploy_clot_guiding_mid
                row["deploy_wall_score"] = deploy_wall_score
                row["deploy_clot_mass_ratio"] = deploy_clot_mass_ratio
                row["deploy_clot_empty_gt_score"] = deploy_clot_empty_gt_score
                row["deploy_clot_fp"] = deploy_clot_fp
                row["deploy_clot_fn"] = deploy_clot_fn

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
                    f" wall={deploy_wall_score:.3f}"
                    f" mass={deploy_clot_mass_ratio:.3f}"
                    f" f1={deploy_clot_f1:.3f}"
                    f" seed_n={mat_seed_count:.1f}"
                    f" seed_p={mat_seed_prec:.3f}"
                    f" front={mat_front_speed_ratio:.3f}"
                    f" fp={deploy_clot_fp:.0f}"
                    f" fn={deploy_clot_fn:.0f}"
                )
        ep_time = time.time() - t0_ep
        print(
            f"[ep {ep:03d}] time={ep_time:.1f}s loss={row['loss']:.6f} "
            f"val_state_f1={row['val_state_f1']:.3f} val_mat_f1={row['val_mat_f1']:.3f} "
            f"val_growth_f1={row['val_growth_f1']:.3f} val_dlt={row['val_pred_delta']:.2e} "
            f"clot_phi_f1={row['val_clot_phi_f1']:.3f} init_f1={row['val_init_f1']:.3f}{dep_msg}",
            flush=True,
        )
        score_kwargs = _held_out_select_kwargs_from_runtime()
        score, select_mode = select_checkpoint_score(
            row,
            held_out_val=bool(args.exclude_val_from_train),
            physics_on=physics_on,
            mat_precision_select=mat_precision_select,
            deploy_clot_score=deploy_clot_score,
            deploy_mat_f1=deploy_mat_f1,
            deploy_clot_pred_pos_frac=deploy_clot_pred_pos_frac,
            clot_weight=continuous_score_clot_weight(),
            deploy_clot_mass_ratio=float(deploy_clot_mass_ratio),
            **score_kwargs,
        )
        row["select_mode"] = select_mode
        row["select_score"] = score
        # Gate-independent retention: keep the highest raw deploy score we ever saw.
        if float(deploy_clot_score) > salvage_score:
            salvage_score = float(deploy_clot_score)
            salvage_epoch = ep
            save_continuous_checkpoint(
                salvage_path,
                model,
                {
                    **meta_base,
                    "salvage_score": salvage_score,
                    "salvage_epoch": ep,
                    "salvage_select_mode": select_mode,
                    "salvage_gate_rejected": select_mode.endswith("_reject"),
                    **row,
                },
            )
        row["salvage_best_score"] = salvage_score
        row["salvage_best_epoch"] = salvage_epoch
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if select_mode == "deploy_only_mass_reject":
            print(
                f"[i] select=mass_reject mass={float(deploy_clot_mass_ratio):.3f} "
                f"(hard_min={score_kwargs.get('select_mass_hard_min', 0.0)} "
                f"hard_max={score_kwargs['select_mass_hard_max']})",
                flush=True,
            )
        if select_mode == "deploy_only_fn_reject":
            print(
                f"[i] select=fn_reject fn={float(row.get('deploy_clot_fn', row.get('clot_fn_median', 0.0)) or 0.0):.0f} "
                f"(hard_max={score_kwargs.get('select_fn_hard_max', 0.0)})",
                flush=True,
            )
        if select_mode == "physics_gt_fed" and row["val_clot_phi_f1"] <= 0.0:
            dead_phi_epochs += 1
            if dead_phi_epochs == 3:
                print(
                    "[WARN] val_clot_phi_f1 has been 0.000 for 3 epochs: half the selection "
                    "weight is dead and the rest is GT-teacher-forced. Pass "
                    "--exclude-val-from-train for deploy-only selection.",
                    flush=True,
                )
        # Hard mass-reject / FN-reject must never become "best" (score == best_score init sentinel).
        if select_mode in ("deploy_only_mass_reject", "deploy_only_fn_reject"):
            stale += 1
            if stale >= int(args.early_stop):
                print(f"[i] early stop @ ep {ep} (best_score={best_score:.3f})", flush=True)
                break
            continue
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

    last_path = out_path.parent / "last.pth"
    save_continuous_checkpoint(last_path, model, {**meta_base, "epoch": ep, "last_score": score})
    if salvage_epoch > 0:
        print(
            f"[i] salvage ckpt: ep {salvage_epoch} deploy_clot_score={salvage_score:.4f} -> {salvage_path.name}",
            flush=True,
        )
    if not out_path.exists() and salvage_path.exists():
        # Every epoch was gate-rejected. Promote the salvage so the leg is analysable at all
        # instead of leaving nothing behind (s12.4: this voided v2..v6). Meta records that the
        # gate rejected it, so nothing downstream can mistake it for a gate-passing selection.
        shutil.copyfile(salvage_path, out_path)
        print(
            f"[WARN] no epoch passed the selection gate; promoted salvage ep {salvage_epoch} "
            f"(deploy_clot_score={salvage_score:.4f}) to {out_path.name}",
            flush=True,
        )
        print(
            "[WARN]   this checkpoint is GATE-REJECTED -- meta.salvage_gate_rejected=True. "
            "Grade it, do not ship it.",
            flush=True,
        )
    print(f"[OK] best_score={best_score:.3f} elapsed={time.perf_counter() - t0:.1f}s ckpt={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
