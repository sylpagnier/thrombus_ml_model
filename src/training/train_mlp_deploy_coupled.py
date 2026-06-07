"""Coupled deploy finetune: train clot-phi MLP on pred-kine teacher rollouts (B_wired).

Step 2 v3: diagnose-aligned forward (``forward_mlp_fields_at_rollout_frame``),
frozen t=0 allowed mask, allowed-wide hinge, closed-loop rollouts.

Frozen: GNODE teacher. Trainable: clot-phi hybrid head (mu branch default).

Usage::

    python -m src.training.train_mlp_deploy_coupled

Launcher: ``scripts/go_mlp_deploy_coupled_finetune.ps1``
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_phi_mu_inject import (
    diagnose_deploy_gate_rollout_series,
    init_deploy_supervision_vision_mask,
    resolve_clot_mu_commit_thresh_si,
)
from src.core_physics.clot_phi_simple import (
    build_clot_phi_model,
    build_clot_phi_step,
    clot_phi_hybrid_enabled,
)
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.inference.biochem_teacher_loader import build_biochem_teacher, resolve_rollout_mu_ratio_max
from src.inference.clot_phi_inject_attach import attach_clot_phi_injector_to_teacher
from src.inference.deploy_mu_map_env import wire_deploy_mu_map
from src.training.deploy_coupled_forward import (
    compute_allowed_deploy_metrics,
    compute_deploy_coupled_step_losses,
    deploy_coupled_promote_score,
    forward_mlp_fields_at_rollout_frame,
    resolve_supervise_clot_mask,
)
from src.training.train_clot_phi_simple import (
    _dice_score,
    _env_bool,
    _list_anchor_paths,
    _load_init_checkpoint,
    _split_train_val,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.nondim import to_t_nd
from src.utils.paths import get_project_root


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _closed_loop_enabled() -> bool:
    return _env_bool("MLP_DEPLOY_COUPLED_CLOSED_LOOP", True)


def _graph_passes(train: bool) -> int:
    if not train or not _closed_loop_enabled():
        return 1
    return max(_env_int("MLP_DEPLOY_COUPLED_GRAPH_PASSES", 2), 1)


def _final_frame_only() -> bool:
    return _env_bool("MLP_DEPLOY_COUPLED_FINAL_FRAME_ONLY", False)


def _frame_loss_weight(si: int, n_frames: int) -> float:
    if n_frames <= 1:
        return 1.0
    if si == n_frames - 1:
        return max(_env_float("MLP_DEPLOY_COUPLED_FINAL_FRAME_WEIGHT", 3.0), 1.0)
    return 1.0


def _label_indices_for_rollout(data, bio_cfg, device, eval_t_nd: torch.Tensor) -> list[int]:
    t_si = bio_cfg.resolve_biochem_times(data, device)
    t_ref = float(bio_cfg.t_final)
    eval_times_si = eval_t_nd.reshape(-1).to(device=device, dtype=t_si.dtype) * t_ref
    out: list[int] = []
    for i in range(int(eval_times_si.numel())):
        t = float(eval_times_si[i].item())
        idx = int((t_si - t).abs().argmin().item())
        out.append(max(0, min(idx, int(data.y.shape[0]) - 1)))
    return out


def _deploy_coupled_fast_mode() -> bool:
    return _env_bool("MLP_DEPLOY_COUPLED_FAST", False)


def _apply_fast_rollout_env() -> None:
    os.environ["BIOCHEM_ADJOINT_RK4_SUBSTEPS"] = "1"
    os.environ.setdefault("BIOCHEM_ODEINT_USE_ADJOINT", "1")


def _eval_times_for_rollout(
    t_si: torch.Tensor,
    t_ref: float,
    *,
    time_stride: int,
    fast: bool = False,
) -> torch.Tensor:
    if fast:
        t_last = float(t_si[-1].item())
        t_mid = 0.5 * t_last
        times_si = sorted({0.0, t_mid, t_last})
        return to_t_nd(torch.tensor(times_si, device=t_si.device, dtype=t_si.dtype), t_ref)
    eval_t = to_t_nd(t_si, t_ref)
    stride = max(1, int(time_stride))
    if stride > 1 and eval_t.numel() > 2:
        idxs = list(range(0, int(eval_t.numel()), stride))
        if idxs[-1] != int(eval_t.numel()) - 1:
            idxs.append(int(eval_t.numel()) - 1)
        eval_t = eval_t[idxs]
    return eval_t


def _freeze_phi_branch(model: torch.nn.Module) -> None:
    if not _env_bool("CLOT_PHI_DEPLOY_TRAIN_MU_ONLY", True):
        return
    for name, p in model.named_parameters():
        if "phi_fc" in name:
            p.requires_grad = False
        elif name.endswith("net.3.weight") or name.endswith("net.3.bias"):
            p.requires_grad = False


def _init_dlog_bias_from_allowed(
    model: torch.nn.Module,
    train_paths: list[str],
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> None:
    """Set ``dlog_fc.bias`` from median log(mu_gt/mu_anchor) on GT clots in allowed mask."""
    if not _env_bool("MLP_DEPLOY_COUPLED_BIAS_INIT", True):
        return
    if not clot_phi_hybrid_enabled() or not hasattr(model, "dlog_fc"):
        return
    mu_thresh = resolve_clot_mu_commit_thresh_si(phys_cfg)
    log_ratios: list[torch.Tensor] = []
    for path in train_paths[: min(6, len(train_paths))]:
        data = torch.load(path, map_location="cpu", weights_only=False)
        allowed = init_deploy_supervision_vision_mask(
            data, torch.device("cpu"), 0, phys_cfg=phys_cfg, bio_cfg=bio_cfg
        )
        for ti in (0, int(data.y.shape[0]) - 1):
            step = build_clot_phi_step(data, ti, phys_cfg, bio_cfg, torch.device("cpu"))
            gt_clot = (step.mu_gt_cap >= mu_thresh) & allowed
            if not bool(gt_clot.any().item()):
                continue
            ratio = (
                step.mu_gt_cap[gt_clot] / step.mu_c_si[gt_clot].clamp(min=1e-8)
            ).clamp(min=1.05)
            log_ratios.append(torch.log(ratio))
    if not log_ratios:
        return
    med = torch.cat(log_ratios).median().item()
    bias = math.log(math.expm1(max(med, 1e-4))) if med > 1e-3 else float(med)
    with torch.no_grad():
        if getattr(model.dlog_fc, "bias", None) is not None:
            model.dlog_fc.bias.fill_(bias)
    print(f"[i]  dlog_fc bias init (allowed GT clots) ~ softplus^-1({med:.3f}) -> {bias:.3f}", flush=True)


def _sync_teacher_injector(teacher, model: torch.nn.Module) -> None:
    inj = getattr(teacher, "_clot_phi_injector", None)
    inner = getattr(inj, "_model", None)
    if inner is None:
        return
    inner.load_state_dict(model.state_dict())
    inner.eval()


def _load_teacher(
    ckpt_path: Path,
    device: torch.device,
    mu_ratio_max: float,
    *,
    clot_ckpt: Path,
    fast: bool = False,
) -> Any:
    if fast:
        _apply_fast_rollout_env()
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    from src.architecture.gnode_biochem import (
        apply_biochem_forward_policy_from_checkpoint_meta,
        restore_mlp_clot_inject_shell_env,
        snapshot_mlp_clot_inject_shell_env,
    )

    inject_shell = snapshot_mlp_clot_inject_shell_env()
    apply_biochem_forward_policy_from_checkpoint_meta(raw, quiet=True)
    restore_mlp_clot_inject_shell_env(inject_shell)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    teacher = build_biochem_teacher(
        raw,
        phys_cfg=phys,
        bio_cfg=bio,
        device=device,
        mu_ratio_max=mu_ratio_max,
        quiet=True,
    )
    wire_deploy_mu_map(clot_ckpt=clot_ckpt, wired=True)
    attach_clot_phi_injector_to_teacher(teacher, device, str(clot_ckpt))
    if fast:
        teacher.max_inner_iters = max(3, min(int(getattr(teacher, "max_inner_iters", 10)), 4))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, phys, bio


@torch.no_grad()
def _cache_teacher_rollout(
    teacher,
    data,
    bio_cfg,
    device: torch.device,
    *,
    time_stride: int,
    fast: bool = False,
    path_stem: str = "anchor",
    pass_idx: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    data = infer_missing_schema(data, phase_hint="biochem").to(device)
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
    t_si = bio_cfg.resolve_biochem_times(data, device)
    t_ref = float(bio_cfg.t_final)
    eval_t = _eval_times_for_rollout(t_si, t_ref, time_stride=time_stride, fast=fast)
    tag = "fast" if fast else f"stride={time_stride}"
    pass_tag = f" pass={pass_idx}" if pass_idx > 0 else ""
    print(
        f"[i]  teacher rollout {path_stem} ({tag}, T={int(eval_t.numel())}{pass_tag})...",
        flush=True,
    )
    t0 = time.perf_counter()
    pred_series = teacher(
        data,
        eval_t,
        y_true_trajectory=data.y,
        teacher_forcing_ratio=1.0,
        start_idx=0,
        initial_species=None,
        detach_macro_state=True,
    )
    if isinstance(pred_series, tuple):
        pred_series = pred_series[0]
    label_idx = _label_indices_for_rollout(data, bio_cfg, device, eval_t)
    print(f"[i]  rollout done {path_stem} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return pred_series, label_idx


@torch.no_grad()
def _diagnose_gate_final(
    model: torch.nn.Module,
    path: str,
    *,
    teacher,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    rollout_fast: bool,
    in_dim: int,
    hidden: int,
) -> dict[str, float]:
    """Run diagnose-equivalent gate series; return t_final frame (promotion truth)."""
    data = torch.load(path, weights_only=False).to(device)
    data = infer_missing_schema(data, phase_hint="biochem")
    _sync_teacher_injector(teacher, model)
    t_si = bio_cfg.resolve_biochem_times(data, device)
    t_ref = float(bio_cfg.t_final)
    time_stride = max(int(os.environ.get("MLP_DEPLOY_COUPLED_TIME_STRIDE", "5")), 1)
    eval_t = _eval_times_for_rollout(t_si, t_ref, time_stride=time_stride, fast=rollout_fast)
    pred_series, label_idx = _cache_teacher_rollout(
        teacher,
        data,
        bio_cfg,
        device,
        time_stride=time_stride,
        fast=rollout_fast,
        path_stem=Path(path).stem + "_gate",
        pass_idx=0,
    )
    eval_times_si = eval_t.reshape(-1).to(device=device, dtype=t_si.dtype) * t_ref
    gate_model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    gate_model.load_state_dict(model.state_dict(), strict=True)
    gate_model.eval()
    rows = diagnose_deploy_gate_rollout_series(
        gate_model,
        data,
        pred_series,
        eval_times_si,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        label_time_indices=label_idx,
    )
    if not rows:
        return {}
    last = rows[-1]
    return {
        "gate_frac_mu": float(last.frac_mu_mlp_ge_thr_in_allowed),
        "gate_frac_both": float(last.frac_both_in_allowed),
        "gate_frac_commit": float(last.frac_commit_in_allowed),
        "gate_mu_p90": float(last.mu_mlp_p90_allowed),
        "gate_n_gt_clot": float(last.n_gt_clot_in_allowed),
        "gate_n_allowed": float(last.n_allowed),
    }


def _run_epoch(
    model: torch.nn.Module,
    paths: list[str],
    *,
    teacher,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    train: bool,
    time_stride: int,
    optimizer: torch.optim.Optimizer | None = None,
    rollout_fast: bool = False,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    mu_thresh = resolve_clot_mu_commit_thresh_si(phys_cfg)
    phi_thresh = max(_env_float("BIOCHEM_MLP_MU_MAP_PHI_THRESH", 0.5), 0.0)
    mu_log_lambda = max(_env_float("CLOT_PHI_MU_LOG_LAMBDA", 1.5), 0.0)
    soft_commit_lambda = max(_env_float("CLOT_PHI_DEPLOY_SOFT_COMMIT_LAMBDA", 4.0), 0.0)
    hinge_lambda = max(_env_float("CLOT_PHI_DEPLOY_MU_HINGE_LAMBDA", 5.0), 0.0)
    allowed_hinge_lambda = max(_env_float("CLOT_PHI_DEPLOY_ALLOWED_HINGE_LAMBDA", 10.0), 0.0)
    phi_lambda = max(_env_float("CLOT_PHI_DEPLOY_PHI_LAMBDA", 0.0), 0.0)
    if _env_bool("CLOT_PHI_DEPLOY_TRAIN_MU_ONLY", True):
        phi_lambda = 0.0

    graph_passes = _graph_passes(train)
    final_only = _final_frame_only()

    mu_mse_sum = 0.0
    hinge_sum = 0.0
    allowed_hinge_sum = 0.0
    commit_sum = 0.0
    log_mae_sum = 0.0
    frac_mu_ok_sum = 0.0
    frac_both_sum = 0.0
    frac_rollout_ok_sum = 0.0
    mu_p90_sum = 0.0
    dice_sum = 0.0
    n_steps = 0
    n_graphs = 0
    n_skipped = 0

    frame_indices_fn = (
        lambda n: [n - 1]
        if final_only
        else list(range(n))
    )

    with torch.set_grad_enabled(train):
        for path in paths:
            data = torch.load(path, weights_only=False).to(device)
            data = infer_missing_schema(data, phase_hint="biochem")
            assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
            if not hasattr(data, "y") or data.y.dim() != 3:
                continue
            n_graphs += 1
            stem = Path(path).stem
            allowed = init_deploy_supervision_vision_mask(
                data, device, 0, phys_cfg=phys_cfg, bio_cfg=bio_cfg
            )
            if not bool(allowed.any().item()):
                continue

            for pass_idx in range(graph_passes):
                if train:
                    _sync_teacher_injector(teacher, model)
                pred_series, label_idx = _cache_teacher_rollout(
                    teacher,
                    data,
                    bio_cfg,
                    device,
                    time_stride=time_stride,
                    fast=rollout_fast,
                    path_stem=stem,
                    pass_idx=pass_idx,
                )
                n_frames = len(label_idx)
                for si in frame_indices_fn(n_frames):
                    li = label_idx[si]
                    pred_y = pred_series[si]
                    u_nd = pred_y[:, 0]
                    v_nd = pred_y[:, 1]
                    species = pred_y[:, 4:16]
                    mu_rollout_si = phys_cfg.viscosity_nd_to_si(
                        pred_y[:, STATE_CHANNEL_MU_EFF_ND]
                    ).reshape(-1)

                    fields = forward_mlp_fields_at_rollout_frame(
                        model,
                        data,
                        li,
                        u_nd=u_nd,
                        v_nd=v_nd,
                        species_log=species,
                        phys_cfg=phys_cfg,
                        bio_cfg=bio_cfg,
                        device=device,
                    )
                    phi_pred = fields["phi"]
                    mu_pred = fields["mu_mlp"]
                    mu_bulk = fields["mu_bulk"]
                    tgt = fields["phi_gt"]
                    mu_gt_cap = fields["mu_gt_cap"]
                    logits = fields["logits"]

                    gt_clot = (mu_gt_cap >= mu_thresh) & allowed
                    supervise = resolve_supervise_clot_mask(
                        gt_clot=gt_clot,
                        phi_gt=tgt,
                        allowed=allowed,
                        phi_thresh=phi_thresh,
                    )
                    if train and not bool(supervise.any().item()):
                        n_skipped += 1
                        continue

                    fw = _frame_loss_weight(si, n_frames)

                    if train:
                        if optimizer is None:
                            raise ValueError("optimizer required when train=True")
                        optimizer.zero_grad(set_to_none=True)
                        loss, terms = compute_deploy_coupled_step_losses(
                            phi_pred=phi_pred,
                            mu_mlp=mu_pred,
                            mu_bulk=mu_bulk,
                            mu_gt_cap=mu_gt_cap,
                            phi_gt=tgt,
                            allowed=allowed,
                            gt_clot=gt_clot,
                            phys_cfg=phys_cfg,
                            phi_thresh=phi_thresh,
                            mu_log_lambda=mu_log_lambda,
                            hinge_lambda=hinge_lambda,
                            allowed_hinge_lambda=allowed_hinge_lambda,
                            soft_commit_lambda=soft_commit_lambda,
                            phi_lambda=phi_lambda,
                            pos_weight=1.0,
                            logits=logits,
                        )
                        (loss * fw).backward()
                        optimizer.step()
                        mu_mse_v = float(terms["mu_mse"].item())
                        hinge_v = float(terms["hinge"].item())
                        allowed_hinge_v = float(terms["allowed_hinge"].item())
                        commit_v = float(terms["soft_commit"].item()) + float(
                            terms["commit_log"].item()
                        )
                    else:
                        with torch.no_grad():
                            _, terms = compute_deploy_coupled_step_losses(
                                phi_pred=phi_pred,
                                mu_mlp=mu_pred,
                                mu_bulk=mu_bulk,
                                mu_gt_cap=mu_gt_cap,
                                phi_gt=tgt,
                                allowed=allowed,
                                gt_clot=gt_clot,
                                phys_cfg=phys_cfg,
                                phi_thresh=phi_thresh,
                                mu_log_lambda=mu_log_lambda,
                                hinge_lambda=hinge_lambda,
                                allowed_hinge_lambda=allowed_hinge_lambda,
                                soft_commit_lambda=soft_commit_lambda,
                                phi_lambda=phi_lambda,
                                pos_weight=1.0,
                                logits=logits,
                            )
                            mu_mse_v = float(terms["mu_mse"].item())
                            hinge_v = float(terms["hinge"].item())
                            allowed_hinge_v = float(terms["allowed_hinge"].item())
                            commit_v = float(terms["soft_commit"].item()) + float(
                                terms["commit_log"].item()
                            )

                    with torch.no_grad():
                        m = compute_allowed_deploy_metrics(
                            phi=phi_pred,
                            mu_mlp=mu_pred,
                            mu_rollout_si=mu_rollout_si,
                            allowed=allowed,
                            mu_thresh=mu_thresh,
                            phi_thresh=phi_thresh,
                        )
                        frac_mu_ok_sum += m["frac_mu_ok_allowed"]
                        frac_both_sum += m["frac_both_allowed"]
                        frac_rollout_ok_sum += m["frac_rollout_mu_ok_allowed"]
                        mu_p90_sum += m["mu_mlp_p90_allowed"]
                        log_mae = (
                            torch.log(mu_pred[allowed].clamp(min=1e-8))
                            - torch.log(mu_gt_cap[allowed].clamp(min=1e-8))
                        ).abs().mean()
                        log_mae_sum += float(log_mae.item())
                        dice_sum += _dice_score(phi_pred[allowed], tgt[allowed])
                        mu_mse_sum += mu_mse_v
                        hinge_sum += hinge_v
                        allowed_hinge_sum += allowed_hinge_v
                        commit_sum += commit_v
                        n_steps += 1

    denom = max(n_steps, 1)
    return {
        "mu_log_mse": mu_mse_sum / denom,
        "mu_hinge": hinge_sum / denom,
        "allowed_hinge": allowed_hinge_sum / denom,
        "commit_bce": commit_sum / denom,
        "dice": dice_sum / denom,
        "mu_log_mae": log_mae_sum / denom,
        "frac_mu_ok_allowed": frac_mu_ok_sum / denom,
        "frac_both_allowed": frac_both_sum / denom,
        "frac_rollout_mu_ok_allowed": frac_rollout_ok_sum / denom,
        "mu_mlp_p90_allowed": mu_p90_sum / denom,
        "n_steps": float(n_steps),
        "n_skipped": float(n_skipped),
        "n_graphs": float(n_graphs),
        "graph_passes": float(graph_passes if train else 1),
    }


def main() -> None:
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    teacher_ckpt = Path(
        os.environ.get(
            "MLP_DEPLOY_COUPLED_TEACHER",
            "outputs/biochem/gnode10_sweep/gnode12_lane_a_promoted/biochem_teacher_best_high_mu.pth",
        )
    )
    if not teacher_ckpt.is_absolute():
        teacher_ckpt = root / teacher_ckpt
    clot_init = Path(
        os.environ.get(
            "MLP_DEPLOY_COUPLED_INIT_CLOTPHI",
            "outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/clot_phi_best.pth",
        )
    )
    if not clot_init.is_absolute():
        clot_init = root / clot_init
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Missing teacher ckpt: {teacher_ckpt}")
    if not clot_init.is_file():
        raise FileNotFoundError(f"Missing clot-phi init ckpt: {clot_init}")

    init_raw = torch.load(clot_init, map_location=device, weights_only=False)
    cfg = init_raw.get("config") or {}
    apply_clot_phi_config_from_checkpoint(cfg)
    apply_clot_phi_eval_defaults()
    os.environ.setdefault("CLOT_PHI_DGAMMA_FEATURE_TIME", "current")
    os.environ["CLOT_PHI_JOINT_USE_PRED_SPECIES"] = "1"
    wire_deploy_mu_map(clot_ckpt=clot_init, wired=True)

    raw_dir = (os.environ.get("CLOT_PHI_ANCHOR_DIR") or "").strip()
    if raw_dir:
        anchor_dir = Path(raw_dir).expanduser()
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    paths = _list_anchor_paths(anchor_dir.resolve())
    val_stem = (os.environ.get("CLOT_PHI_VAL_ANCHOR") or "patient007").strip()
    train_paths, val_paths = _split_train_val(paths, val_stem)

    if _env_bool("MLP_DEPLOY_COUPLED_TRAIN_ON_VAL", False):
        val_only = [p for p in paths if Path(p).stem.lower() == val_stem.lower()]
        if val_only:
            train_paths = val_only
            val_paths = val_only

    train_filter = (os.environ.get("MLP_DEPLOY_COUPLED_TRAIN_ANCHORS") or "").strip()
    if train_filter:
        allow = {s.strip().lower() for s in train_filter.split(",") if s.strip()}
        train_paths = [p for p in train_paths if Path(p).stem.lower() in allow]
    max_train = int(os.environ.get("MLP_DEPLOY_COUPLED_MAX_TRAIN", "0") or "0")
    if max_train > 0:
        train_paths = train_paths[:max_train]
    if not train_paths:
        raise RuntimeError("No train anchors after MLP_DEPLOY_COUPLED_TRAIN_ANCHORS / MAX_TRAIN filter")

    fast_mode = _deploy_coupled_fast_mode()
    epochs = max(int(os.environ.get("MLP_DEPLOY_COUPLED_EPOCHS", "12")), 1)
    lr = float(os.environ.get("MLP_DEPLOY_COUPLED_LR", "5e-4"))
    time_stride = max(int(os.environ.get("MLP_DEPLOY_COUPLED_TIME_STRIDE", "5")), 1)
    val_stride_raw = (os.environ.get("MLP_DEPLOY_COUPLED_VAL_STRIDE") or "").strip()
    val_stride = time_stride if fast_mode else max(int(val_stride_raw or "5"), 1)
    mu_ratio = resolve_rollout_mu_ratio_max(float(os.environ.get("BIOCHEM_TEACHER_MU_RATIO_MAX", "20")))

    hidden = max(int(cfg.get("hidden", 32)), 4)
    in_dim = max(int(cfg.get("in_dim", 6)), 1)
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    _load_init_checkpoint(model, None, clot_init, device)
    _freeze_phi_branch(model)
    _init_dlog_bias_from_allowed(
        model, train_paths, phys_cfg=phys_cfg, bio_cfg=bio_cfg
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable clot-phi parameters (check CLOT_PHI_DEPLOY_TRAIN_MU_ONLY)")
    n_train = sum(p.numel() for p in params)
    print(f"[i]  trainable params={n_train}", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=float(os.environ.get("CLOT_PHI_WEIGHT_DECAY", "1e-4")))

    out_dir = Path(
        os.environ.get("MLP_DEPLOY_COUPLED_OUT_DIR", "outputs/biochem/mlp_deploy_coupled")
    )
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "clot_phi_best.pth"
    log_path = out_dir / "deploy_coupled_train_log.jsonl"

    closed = _closed_loop_enabled()
    g_passes = _graph_passes(True)
    print(
        f"[NEW] deploy coupled v3 (diagnose-aligned allowed, closed_loop={int(closed)})",
        flush=True,
    )
    print(
        f"[i]  epochs={epochs} lr={lr:.2e} stride={time_stride} val_stride={val_stride} "
        f"fast={int(fast_mode)} final_only={int(_final_frame_only())} "
        f"graph_passes={g_passes} mu_only={int(_env_bool('CLOT_PHI_DEPLOY_TRAIN_MU_ONLY', True))}",
        flush=True,
    )
    print(
        f"[i]  teacher={teacher_ckpt} init_clot={clot_init} "
        f"train={len(train_paths)} val={len(val_paths)}",
        flush=True,
    )
    print(
        f"[i]  mu_commit_thresh={resolve_clot_mu_commit_thresh_si(phys_cfg):.4f} "
        f"allowed_hinge={_env_float('CLOT_PHI_DEPLOY_ALLOWED_HINGE_LAMBDA', 10.0):.1f} "
        f"hinge={_env_float('CLOT_PHI_DEPLOY_MU_HINGE_LAMBDA', 5.0):.1f} "
        f"soft_commit={_env_float('CLOT_PHI_DEPLOY_SOFT_COMMIT_LAMBDA', 4.0):.1f}",
        flush=True,
    )

    teacher, _, _ = _load_teacher(
        teacher_ckpt, device, mu_ratio, clot_ckpt=clot_init, fast=fast_mode
    )
    _sync_teacher_injector(teacher, model)
    best_score = -1.0
    best_gate_mu = -1.0
    val_gate_path = next((p for p in val_paths if Path(p).stem.lower() == val_stem.lower()), val_paths[0] if val_paths else "")

    for ep in range(epochs):
        t0 = time.perf_counter()
        tr = _run_epoch(
            model,
            train_paths,
            teacher=teacher,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            train=True,
            time_stride=time_stride,
            optimizer=opt,
            rollout_fast=fast_mode,
        )
        _sync_teacher_injector(teacher, model)
        va = _run_epoch(
            model,
            val_paths,
            teacher=teacher,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            train=False,
            time_stride=val_stride,
            optimizer=None,
            rollout_fast=fast_mode,
        )
        gate: dict[str, float] = {}
        if val_gate_path:
            gate = _diagnose_gate_final(
                model,
                val_gate_path,
                teacher=teacher,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                rollout_fast=fast_mode,
                in_dim=in_dim,
                hidden=hidden,
            )
        score = deploy_coupled_promote_score({**va, **gate})
        both = float(va.get("frac_both_allowed", 0.0))
        mu_ok = float(va.get("frac_mu_ok_allowed", 0.0))
        gate_mu = float(gate.get("gate_frac_mu", 0.0))
        elapsed = time.perf_counter() - t0
        row = {
            "epoch": ep + 1,
            "train": tr,
            "val": va,
            "gate_final": gate,
            "deploy_score": score,
            "elapsed_s": round(elapsed, 1),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[ep {ep + 1:02d}] val allowed mu={mu_ok:.3f} both={both:.3f} "
            f"p90={va.get('mu_mlp_p90_allowed', 0.0):.4f} "
            f"gate_mu={gate_mu:.3f} gate_p90={gate.get('gate_mu_p90', 0.0):.4f} "
            f"allowed_hinge={tr.get('allowed_hinge', 0.0):.4f} ({elapsed:.0f}s)",
            flush=True,
        )
        promote = score > best_score or (
            abs(score - best_score) < 1e-6 and gate_mu > best_gate_mu
        )
        if promote:
            best_score = score
            best_gate_mu = gate_mu
            save_cfg = dict(cfg)
            save_cfg.update(
                {
                    "deploy_coupled_v3": True,
                    "deploy_coupled_closed_loop": closed,
                    "teacher_ckpt": str(teacher_ckpt),
                    "init_clot_ckpt": str(clot_init),
                    "deploy_leg": "B_wired",
                    "hidden": hidden,
                    "in_dim": in_dim,
                    "deploy_mu_commit_thresh_si": resolve_clot_mu_commit_thresh_si(phys_cfg),
                    "anchor_dir": str(anchor_dir),
                }
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": save_cfg,
                    "epoch": ep + 1,
                    "val": va,
                    "gate_final": gate,
                    "deploy_score": score,
                },
                ckpt_path,
            )
            _sync_teacher_injector(teacher, model)

    print(
        f"[OK]  best deploy_score={best_score:.3f} gate_mu={best_gate_mu:.3f} -> {ckpt_path}",
        flush=True,
    )
    print(
        "[i]  verify: go_diagnose_deploy_gate.ps1 -ClotPhiCheckpoint "
        f"{ckpt_path.relative_to(root)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
