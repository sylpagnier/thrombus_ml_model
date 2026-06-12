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
from src.core_physics.species_pushforward_continuous import (
    BIOCHEM_ANCHORS_6,
    DEFAULT_CONTINUOUS_CKPT,
    DEFAULT_S26_CKPT,
    DEFAULT_S30_CKPT,
    DEFAULT_S31_CKPT,
    DEFAULT_S32_CKPT,
    DEFAULT_S33_CKPT,
    DEFAULT_S34_CKPT,
    SpeciesDualHeadContinuousGNN,
    band_speed_series,
    build_continuous_gnn,
    closed_loop_init_prob,
    continuous_channel_weights,
    continuous_feature_dim,
    continuous_mature_fp_exempt,
    continuous_saturation_gate,
    continuous_temporal_gate,
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
    init_dual_head_widen_from_checkpoint,
    iter_pushforward_windows,
    mature_clot_frac,
    saturation_headroom_scale,
    load_continuous_bundle,
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
from src.core_physics.species_pushforward_gnn import build_band_base_features, pushforward_train_t0_min
from src.core_physics.species_snapshot_gnn import (
    DEFAULT_SNAPSHOT_CKPT,
    kin_per_vessel_norm_enabled,
    snapshot_hidden_dim,
    snapshot_wall_hops,
)
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root


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


def _val_windows(static: dict, *, unroll: int, stride: int) -> list[list[int]]:
    anchors = [10, 25, 28]
    wins: list[list[int]] = []
    n_times = int(static["n_times"])
    for t0 in anchors:
        win = [t0 + i * stride for i in range(unroll + 1)]
        if win[-1] < n_times:
            wins.append(win)
    return wins


def _parse_anchors(raw: str, *, all_anchors: bool) -> list[str]:
    if all_anchors:
        return list(BIOCHEM_ANCHORS_6)
    items = [a.strip() for a in raw.split(",") if a.strip()]
    return items or ["patient007"]


def _build_anchor_pack(
    anchor: str,
    *,
    root: Path,
    device: torch.device,
    kine_model,
    wall_hops: int,
    unroll: int,
    stride: int,
    t0_max: int | None,
    max_windows: int,
    val_frac: float,
    seed: int,
    phys: PhysicsConfig,
    bio: BiochemConfig,
) -> dict:
    graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor.strip()}.pt"
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    static = _prepare_static(data, device=device, kine_model=kine_model, wall_hops=wall_hops)
    train_m, val_m = _split_band_nodes(static["n_band"], val_frac, seed)
    train_m = train_m.to(device=device)
    val_m = val_m.to(device=device)
    windows = iter_pushforward_windows(static["n_times"], unroll=unroll, stride=stride)
    windows = filter_continuous_windows(
        windows, data, static["node_idx"], device, t0_max=t0_max, min_delta_mag=1e-8
    )
    if max_windows > 0:
        windows = windows[: int(max_windows)]
    return {
        "anchor": anchor.strip(),
        "data": data,
        "static": static,
        "train_m": train_m,
        "val_m": val_m,
        "windows": windows,
        "val_windows": _val_windows(static, unroll=unroll, stride=stride),
        "phys": phys,
        "bio": bio,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train species continuous pushforward (phase 2.5/2.6/3.0)")
    ap.add_argument("--phase", choices=("s25", "s26", "s30", "s31", "s32", "s33", "s34"), default="s25")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchors", default="", help="Comma-separated anchors for multi-vessel train")
    ap.add_argument("--all-anchors", action="store_true", help="Train on all 6 biochem anchors")
    ap.add_argument("--val-anchor", default="patient007", help="Holdout anchor for val logging")
    ap.add_argument(
        "--exclude-val-from-train",
        action="store_true",
        help="LOAO: drop val-anchor from training packs (train only on other vessels)",
    )
    ap.add_argument("--init-s26", default="", help="Optional s26 checkpoint to warm-start")
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
    args = ap.parse_args()

    phase = str(args.phase).strip().lower()
    if phase in ("s26", "s30", "s31", "s32", "s33", "s34"):
        os.environ["SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS"] = "1"
    if phase in ("s31", "s32", "s33", "s34"):
        os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] = "1"
        os.environ["SPECIES_CONTINUOUS_PHYSICS_READOUT"] = "0"
        os.environ["SPECIES_KIN_PER_VESSEL_NORM"] = "1"
    if phase == "s34":
        os.environ.setdefault("SPECIES_CONTINUOUS_SATURATION_GATE", "1")
        os.environ.setdefault("SPECIES_CONTINUOUS_MATURE_FP_EXEMPT", "1")
        os.environ.setdefault("SPECIES_CONTINUOUS_MATURE_FRAC", "0.95")
        os.environ.setdefault("SPECIES_CONTINUOUS_SATURATION_SCALE", "80")
        os.environ.setdefault("SPECIES_CONTINUOUS_TEMPORAL_GATE", "1")
        os.environ.setdefault("SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN", "0.5")
        os.environ.setdefault("SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX", "1.5")
        for key, val in (
            ("SPECIES_CONTINUOUS_VEL_DECAY", "1"),
            ("SPECIES_CONTINUOUS_TEACHER_NOISE", "0.02"),
            ("SPECIES_CONTINUOUS_TEACHER_FP_FRAC", "0.08"),
            ("SPECIES_CONTINUOUS_TEACHER_BLUR", "0.25"),
            ("SPECIES_CONTINUOUS_TBPTT_TAIL", "5"),
            ("SPECIES_CONTINUOUS_CURRICULUM_UNROLL", "1"),
            ("SPECIES_CONTINUOUS_CLOSED_LOOP_INIT", "0.45"),
            ("SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT", "0.35"),
            ("SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND", "1"),
            ("SPECIES_CONTINUOUS_SPEED_FP_WEIGHT", "4.0"),
            ("SPECIES_CONTINUOUS_DEPLOY_HORIZON", "53"),
        ):
            os.environ.setdefault(key, val)
        if args.unroll is None and not (os.environ.get("SPECIES_PUSHFORWARD_UNROLL") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_UNROLL"] = "10"
        os.environ.setdefault("SPECIES_PUSHFORWARD_MAX_UNROLL", "53")
        if not (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_MAX") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_TRAIN_T0_MAX"] = "35"
    if phase == "s33":
        os.environ.setdefault("SPECIES_CONTINUOUS_SATURATION_GATE", "1")
        os.environ.setdefault("SPECIES_CONTINUOUS_MATURE_FP_EXEMPT", "1")
        os.environ.setdefault("SPECIES_CONTINUOUS_MATURE_FRAC", "0.95")
        os.environ.setdefault("SPECIES_CONTINUOUS_SATURATION_SCALE", "80")
        for key, val in (
            ("SPECIES_CONTINUOUS_VEL_DECAY", "1"),
            ("SPECIES_CONTINUOUS_TEACHER_NOISE", "0.02"),
            ("SPECIES_CONTINUOUS_TEACHER_FP_FRAC", "0.08"),
            ("SPECIES_CONTINUOUS_TEACHER_BLUR", "0.25"),
            ("SPECIES_CONTINUOUS_TBPTT_TAIL", "5"),
            ("SPECIES_CONTINUOUS_CURRICULUM_UNROLL", "1"),
            ("SPECIES_CONTINUOUS_CLOSED_LOOP_INIT", "0.45"),
            ("SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT", "0.35"),
            ("SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND", "1"),
            ("SPECIES_CONTINUOUS_SPEED_FP_WEIGHT", "4.0"),
            ("SPECIES_CONTINUOUS_DEPLOY_HORIZON", "53"),
        ):
            os.environ.setdefault(key, val)
        if args.unroll is None and not (os.environ.get("SPECIES_PUSHFORWARD_UNROLL") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_UNROLL"] = "10"
        os.environ.setdefault("SPECIES_PUSHFORWARD_MAX_UNROLL", "53")
        if not (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_MAX") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_TRAIN_T0_MAX"] = "35"
    if phase == "s32":
        os.environ.setdefault("SPECIES_CONTINUOUS_VEL_DECAY", "1")
        os.environ.setdefault("SPECIES_CONTINUOUS_TEACHER_NOISE", "0.02")
        os.environ.setdefault("SPECIES_CONTINUOUS_TEACHER_FP_FRAC", "0.08")
        os.environ.setdefault("SPECIES_CONTINUOUS_TEACHER_BLUR", "0.25")
        os.environ.setdefault("SPECIES_CONTINUOUS_TBPTT_TAIL", "5")
        os.environ.setdefault("SPECIES_CONTINUOUS_CURRICULUM_UNROLL", "1")
        if args.unroll is None and not (os.environ.get("SPECIES_PUSHFORWARD_UNROLL") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_UNROLL"] = "10"
        os.environ.setdefault("SPECIES_PUSHFORWARD_MAX_UNROLL", "15")
        os.environ.setdefault("SPECIES_CONTINUOUS_CLOSED_LOOP_INIT", "0.45")
        os.environ.setdefault("SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT", "0.35")
        os.environ.setdefault("SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND", "1")
        os.environ.setdefault("SPECIES_CONTINUOUS_SPEED_FP_WEIGHT", "4.0")
        os.environ.setdefault("SPECIES_CONTINUOUS_DEPLOY_HORIZON", "53")
        if not (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_MAX") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_TRAIN_T0_MAX"] = "35"
        if not (os.environ.get("SPECIES_PUSHFORWARD_MAX_UNROLL") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_MAX_UNROLL"] = "53"
    if phase == "s30":
        os.environ["SPECIES_CONTINUOUS_PHYSICS_READOUT"] = "1"
        if not (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_MIN") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_TRAIN_T0_MIN"] = "17"
        if not (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_MAX") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_TRAIN_T0_MAX"] = "32"
        if not (os.environ.get("SPECIES_PUSHFORWARD_TAU_CENTER") or "").strip():
            os.environ["SPECIES_PUSHFORWARD_TAU_CENTER"] = "25"
    if args.unroll is not None:
        os.environ["SPECIES_PUSHFORWARD_UNROLL"] = str(args.unroll)
    if args.stride is not None:
        os.environ["SPECIES_PUSHFORWARD_STEP_STRIDE"] = str(args.stride)
    if args.wall_hops is not None:
        os.environ["SPECIES_SNAPSHOT_WALL_HOPS"] = str(args.wall_hops)
    if not (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_MAX") or "").strip():
        os.environ["SPECIES_PUSHFORWARD_TRAIN_T0_MAX"] = "22"

    unroll = pushforward_unroll_steps()
    max_unroll = pushforward_max_unroll_steps()
    stride = pushforward_step_stride()
    wall_hops = snapshot_wall_hops()
    hidden = snapshot_hidden_dim() if args.hidden is None else max(int(args.hidden), 16)
    ch_w = continuous_channel_weights()
    huber_b = continuous_huber_beta()
    t0_max = pushforward_train_t0_max()
    growth_only = continuous_growth_only_loss()
    loss_scale = continuous_loss_scale()
    delta_thr = continuous_delta_threshold()
    fp_w = continuous_fp_weight()
    physics_on = continuous_physics_readout()
    dual_head = continuous_dual_head()
    if phase == "s34":
        phase_tag = "s34_temporal_gate"
        default_out = DEFAULT_S34_CKPT
    elif phase == "s33":
        phase_tag = "s33_saturation_gate"
        default_out = DEFAULT_S33_CKPT
    elif phase == "s32":
        phase_tag = "s32_long_horizon"
        default_out = DEFAULT_S32_CKPT
    elif phase == "s31":
        phase_tag = "s31_dual_head"
        default_out = DEFAULT_S31_CKPT
    elif phase == "s30":
        phase_tag = "s30_continuous_physics"
        default_out = DEFAULT_S30_CKPT
    elif growth_only:
        phase_tag = "s26_continuous"
        default_out = DEFAULT_S26_CKPT
    else:
        phase_tag = "s25_continuous"
        default_out = DEFAULT_CONTINUOUS_CKPT
    lr = float(args.lr) if args.lr is not None else (3e-4 if growth_only else 1e-3)
    grad_clip = (
        float(args.grad_clip)
        if args.grad_clip is not None
        else float(os.environ.get("SPECIES_CONTINUOUS_GRAD_CLIP", "1.0" if growth_only else "0") or "0")
    )
    grad_clip = max(grad_clip, 0.0)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    train_anchors = _parse_anchors(args.anchors or args.anchor, all_anchors=bool(args.all_anchors))
    val_anchor = args.val_anchor.strip() or train_anchors[0]
    if bool(args.exclude_val_from_train):
        train_anchors = [a for a in train_anchors if a.strip() != val_anchor]
        if not train_anchors:
            raise ValueError(f"exclude-val-from-train left no train anchors (val={val_anchor})")

    kine_ckpt = str(resolve_kinematics_checkpoint())
    kine_model = load_kinematics_predictor(
        kine_ckpt, device, phys_cfg=PhysicsConfig(phase="kinematics")
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
                t0_max=t0_max,
                max_windows=int(args.max_windows),
                val_frac=float(args.val_frac),
                seed=int(args.seed),
                phys=phys,
                bio=bio,
            )
        )
    val_pack = next((p for p in packs if p["anchor"] == val_anchor), packs[0])
    ref_static = packs[0]["static"]
    latent_dim = int(ref_static["base_feats"].shape[1] - 1)
    prev_in_dim = pushforward_feature_dim(latent_dim)
    in_dim = continuous_feature_dim(latent_dim)
    model = build_continuous_gnn(in_dim, hidden=hidden).to(device)

    init_s26 = args.init_s26.strip() or (
        str(root / DEFAULT_S33_CKPT)
        if phase == "s34"
        else (
            str(root / DEFAULT_S32_CKPT)
            if phase == "s33"
            else (
                str(root / DEFAULT_S31_CKPT)
                if phase == "s32"
                else (str(root / DEFAULT_S26_CKPT) if phase in ("s30", "s31") else "")
            )
        )
    )
    if init_s26 and Path(init_s26).is_file():
        init_meta = {}
        init_path = Path(init_s26)
        if init_path.is_file():
            init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
            init_meta = dict(init_payload.get("meta") or {})
        ckpt_is_dual = bool(init_meta.get("dual_head"))
        arch = "single" if dual_head and not ckpt_is_dual else None
        if (
            phase == "s33"
            and continuous_saturation_gate()
            and in_dim > prev_in_dim
            and isinstance(model, SpeciesDualHeadContinuousGNN)
        ):
            init_dual_head_widen_from_checkpoint(
                model, init_s26, prev_in_dim=prev_in_dim, device=device
            )
        else:
            bundle = load_continuous_bundle(init_s26, device=device, quiet=True, architecture=arch)
            if bundle is not None:
                if dual_head and not ckpt_is_dual and isinstance(model, SpeciesDualHeadContinuousGNN):
                    init_dual_head_from_continuous(model, bundle.model)
                else:
                    model.load_state_dict(bundle.model.state_dict(), strict=False)
                print(f"[OK] warm-start from {init_s26}", flush=True)
    else:
        init_path = args.init_s1.strip() or str(root / DEFAULT_SNAPSHOT_CKPT)
        if Path(init_path).is_file():
            init_continuous_from_snapshot(model, init_path)
    # Bias readout toward small positive log-deltas (avoid zero-delta collapse).
    with torch.no_grad():
        last = model.readout[-1]
        if isinstance(last, torch.nn.Linear) and last.bias is not None:
            last.bias.fill_(0.5 if growth_only else 1e-4)

    n_windows = sum(len(p["windows"]) for p in packs)
    print(
        f"[i] phase={phase_tag} anchors={train_anchors} val={val_anchor} "
        f"unroll={unroll} max_unroll={max_unroll} tbptt_tail={tbptt_tail_steps()} "
        f"windows={n_windows} dual_head={int(dual_head)} "
        f"kin_norm={int(kin_per_vessel_norm_enabled())} physics={int(physics_on)} "
        f"vel_decay={int(continuous_vel_decay_enabled())} "
        f"sat_gate={int(continuous_saturation_gate())} sat_scale={saturation_headroom_scale():.0f} "
        f"mature_exempt={int(continuous_mature_fp_exempt())} mature_frac={mature_clot_frac():.2f} "
        f"temporal_gate={int(continuous_temporal_gate())} "
        f"lambda=({temporal_lambda_bounds()[0]:.1f},{temporal_lambda_bounds()[1]:.1f}) "
        f"closed_loop_init={closed_loop_init_prob():.2f} "
        f"final_state_w={continuous_final_state_weight():.2f} "
        f"teacher_noise={continuous_teacher_noise_sigma():.3f} "
        f"teacher_fp={continuous_teacher_fp_frac():.2f} blur={continuous_teacher_blur():.2f} "
        f"growth_only={int(growth_only)} delta_thr={delta_thr:.1e} fp_w={fp_w:.1f} "
        f"t0_min={pushforward_train_t0_min()} t0_max={t0_max} "
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
        "saturation_gate": continuous_saturation_gate(),
        "saturation_scale": saturation_headroom_scale(),
        "mature_fp_exempt": continuous_mature_fp_exempt(),
        "mature_frac": mature_clot_frac(),
        "temporal_gate": continuous_temporal_gate(),
        "temporal_lambda_min": temporal_lambda_bounds()[0],
        "temporal_lambda_max": temporal_lambda_bounds()[1],
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
        "train_t0_min": pushforward_train_t0_min(),
    }

    best_score = -1.0
    stale = 0
    t0 = time.perf_counter()

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        ep_losses: list[float] = []
        cur_unroll = curriculum_unroll_for_epoch(ep) if phase in ("s32", "s33", "s34") else unroll
        pack_order = packs[:]
        random.shuffle(pack_order)

        for pack in pack_order:
            wins = pack["windows"][:]
            random.shuffle(wins)
            static = pack["static"]
            for win in wins:
                win_use = win[: cur_unroll + 1]
                series = log_series_on_band(pack["data"], win_use, device, static["node_idx"])
                speed_series = (
                    band_speed_series(pack["data"], win_use, device, static["node_idx"])
                    if continuous_vel_decay_enabled()
                    else None
                )
                physics_ctx = None
                if physics_on:
                    physics_ctx = build_species_physics_ctx(
                        pack["data"],
                        time_window=win_use,
                        node_idx=static["node_idx"],
                        phys_cfg=pack["phys"],
                        bio_cfg=pack["bio"],
                        device=device,
                    )
                w_t0 = pushforward_window_t0_weight(int(win_use[0]))
                if w_t0 <= 0.0:
                    continue
                log_state0 = series[0]
                if (
                    phase in ("s32", "s33", "s34")
                    and int(win_use[0]) > 0
                    and closed_loop_init_prob() > 0.0
                    and random.random() < closed_loop_init_prob()
                ):
                    log_state0 = rollout_prefix_log_state(
                        model,
                        pack["data"],
                        static,
                        int(win_use[0]),
                        device,
                    )
                loss, _, _ = unroll_continuous_loss(
                    model,
                    base_feats=static["base_feats"],
                    edge_index=static["edge_index"],
                    log_series=series,
                    train_mask=pack["train_m"],
                    log_state0=log_state0,
                    speed_series=speed_series,
                    training=True,
                    physics_ctx=physics_ctx,
                    window_weight=w_t0,
                    tbptt_tail=tbptt_tail_steps(),
                )
                if not loss.requires_grad:
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_losses.append(float(loss.item()))

        if phase in ("s32", "s33", "s34") and deploy_horizon_steps() > 0:
            h = deploy_horizon_steps()
            vpack = val_pack
            n_times = int(vpack["static"]["n_times"])
            t_end = min(int(h), n_times - 1)
            if t_end >= 3:
                win_dep = list(range(0, t_end + 1))
                static = vpack["static"]
                series = log_series_on_band(vpack["data"], win_dep, device, static["node_idx"])
                speed_series = band_speed_series(vpack["data"], win_dep, device, static["node_idx"])
                loss_dep, _, _ = unroll_continuous_loss(
                    model,
                    base_feats=static["base_feats"],
                    edge_index=static["edge_index"],
                    log_series=series,
                    train_mask=vpack["train_m"],
                    log_state0=series[0],
                    speed_series=speed_series,
                    training=True,
                    window_weight=2.5,
                    tbptt_tail=min(tbptt_tail_steps(), max(5, len(win_dep) // 5)),
                    speed_fp_weight=continuous_speed_fp_weight(),
                )
                if loss_dep.requires_grad:
                    opt.zero_grad(set_to_none=True)
                    loss_dep.backward()
                    if grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    ep_losses.append(float(loss_dep.item()))

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
        with torch.no_grad():
            for win in val_pack["val_windows"]:
                static = val_pack["static"]
                series = log_series_on_band(val_pack["data"], win, device, static["node_idx"])
                speed_series = (
                    band_speed_series(val_pack["data"], win, device, static["node_idx"])
                    if continuous_vel_decay_enabled()
                    else None
                )
                physics_ctx = None
                if physics_on:
                    physics_ctx = build_species_physics_ctx(
                        val_pack["data"],
                        time_window=win,
                        node_idx=static["node_idx"],
                        phys_cfg=val_pack["phys"],
                        bio_cfg=val_pack["bio"],
                        device=device,
                    )
                m = eval_continuous_window(
                    model,
                    base_feats=static["base_feats"],
                    edge_index=static["edge_index"],
                    log_series=series,
                    mask=val_pack["val_m"],
                    log_state0=series[0],
                    speed_series=speed_series,
                    physics_ctx=physics_ctx,
                )
                val_state_f1.append(m["final_state_f1"])
                val_mat_f1.append(m["final_state_mat_f1"])
                val_growth_f1.append(m["mean_growth_f1"])
                val_growth_mat_f1.append(m["mean_growth_mat_f1"])
                val_init_f1.append(m["init_state_f1"])
                val_pred_delta.append(m["mean_pred_delta"])
                val_clot_phi_f1.append(m.get("clot_phi_f1", 0.0))
            if phase in ("s32", "s33", "s34"):
                dep = eval_full_rollout_fimat_f1(
                    model,
                    val_pack["data"],
                    val_pack["static"],
                    device,
                    time_index=53,
                )
                deploy_mat_f1 = float(dep["deploy_mat_f1"])
                deploy_fi_f1 = float(dep["deploy_fi_f1"])
                if bool(args.exclude_val_from_train):
                    clf = eval_deploy_clot_f1(
                        model,
                        val_pack["data"],
                        val_pack["static"],
                        val_pack["phys"],
                        val_pack["bio"],
                        device,
                        time_index=53,
                        flow_source="gt",
                    )
                    deploy_clot_f1 = float(clf["deploy_clot_f1"])

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
        if phase in ("s32", "s33", "s34"):
            row["deploy_mat_f1_t53"] = deploy_mat_f1
            row["deploy_fi_f1_t53"] = deploy_fi_f1
            if bool(args.exclude_val_from_train):
                row["deploy_clot_f1_t53"] = deploy_clot_f1
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        dep_msg = ""
        if phase in ("s32", "s33", "s34"):
            dep_msg = f" deploy_mat_t53={deploy_mat_f1:.3f} deploy_fi_t53={deploy_fi_f1:.3f} unroll={cur_unroll}"
            if bool(args.exclude_val_from_train):
                dep_msg += f" deploy_clot_t53={deploy_clot_f1:.3f}"
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
        elif phase in ("s32", "s33", "s34") and bool(args.exclude_val_from_train):
            score = (
                0.55 * deploy_clot_f1
                + 0.25 * deploy_mat_f1
                + 0.10 * row["val_state_f1"]
                + 0.10 * row["val_growth_f1"]
            )
        elif phase in ("s32", "s33", "s34"):
            score = (
                0.70 * deploy_mat_f1
                + 0.15 * row["val_growth_f1"]
                + 0.10 * row["val_state_f1"]
                + 0.05 * row["val_growth_mat_f1"]
            )
        elif dual_head or growth_only:
            score = (
                0.55 * row["val_growth_f1"]
                + 0.30 * row["val_state_f1"]
                + 0.15 * row["val_growth_mat_f1"]
            )
        else:
            score = row["val_state_f1"]
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
