"""Clot forecast ladder (R0-R1): one-step mu(t) -> mu(t+dt) on GT flow."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    ClotPhiStepBatch,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_fixed_mu_from_phi_enabled,
    clot_phi_model_uses_mpnn,
    log_blend_mu_eff_si,
    mu_eff_from_carried_phi,
    mu_eff_from_delta_log_si,
    resolve_clot_support_band,
    supervision_region_mask,
)

if TYPE_CHECKING:
    import torch.nn as nn


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or ("1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def clot_forecast_mode() -> str:
    """``one_step`` = features @ t_in, labels @ t_out; empty = legacy per-frame."""
    return (os.environ.get("CLOT_FORECAST_MODE") or "").strip().lower()


def clot_forecast_one_step_enabled() -> bool:
    return clot_forecast_mode() in ("one_step", "1", "next", "forecast")


def clot_forecast_input_mu_enabled() -> bool:
    """Prong B: append log(mu_t) from the input frame to node features."""
    return _env_bool("CLOT_FORECAST_INPUT_MU", False)


def clot_forecast_mu_carry_enabled() -> bool:
    """Deploy: log(mu @ t_in) from carried pred mu, not COMSOL GT (after warm-up)."""
    return _env_bool("CLOT_FORECAST_MU_CARRY", False)


def clot_forecast_mu_carry_detach() -> bool:
    return _env_bool("CLOT_FORECAST_MU_CARRY_DETACH", True)


def clot_forecast_mu_init_mode() -> str:
    """Cold-start mu when no carry yet: ``carreau`` (deploy) or ``gt`` (train bridge)."""
    raw = (os.environ.get("CLOT_FORECAST_MU_INIT") or "carreau").strip().lower()
    if raw in ("gt", "comsol", "truth"):
        return "gt"
    return "carreau"


def clot_forecast_pair_stride() -> int:
    """COMSOL index gap between input and target (1 = adjacent snapshots)."""
    try:
        return max(1, int(os.environ.get("CLOT_FORECAST_PAIR_STRIDE", "1") or "1"))
    except ValueError:
        return 1


def clot_forecast_pair_schedule() -> str:
    """How (t_in, t_out) pairs are built for one-step forecast training.

    - ``rolling`` (default): adjacent pairs (t, t+stride) across the series.
    - ``static_final``: single pair (0, T_final) — localization / shell gate.
    - ``from_t0``: pairs (0, t_out) for each t_out — multi-horizon without carry.
    """
    raw = (os.environ.get("CLOT_FORECAST_PAIR_SCHEDULE") or "rolling").strip().lower()
    if raw in ("static_final", "static", "final", "t0_to_final", "shell"):
        return "static_final"
    if raw in ("from_t0", "from0", "multih", "multi_horizon", "horizons"):
        return "from_t0"
    return "rolling"


def iter_forecast_pairs(
    t_steps: int,
    *,
    time_stride: int = 1,
    pair_stride: int | None = None,
) -> list[tuple[int, int]]:
    """Valid (t_in, t_out) index pairs for one-step forecast (no extrapolation)."""
    t_steps = int(t_steps)
    if t_steps <= 0:
        return []
    stride = int(pair_stride if pair_stride is not None else clot_forecast_pair_stride())
    stride = max(1, stride)
    step = max(1, int(time_stride))
    schedule = clot_forecast_pair_schedule()
    t_final = t_steps - 1
    if schedule == "static_final":
        if t_final <= 0:
            return []
        return [(0, t_final)]
    if schedule == "from_t0":
        return [(0, int(t_out)) for t_out in range(stride, t_steps, step)]
    t_max = t_steps - stride
    return [(int(ti), int(ti + stride)) for ti in range(0, max(t_max, 0), step)]


def clot_forecast_mask_mode() -> str:
    """Loss/supervision band for one-step forecast.

    - ``target`` (default A/B/C): neighbor shell seeded from GT mu @ t_out (oracle).
    - ``input``: neighbor shell seeded from GT mu @ t_in (no future clot peek).
    - ``deploy_input``: deploy band with GT phi + GT mu @ t_in (oracle ablation; degenerate).
    - ``deploy_pred``: deploy band with model phi @ t_in + mu @ t_in (deploy-faithful).
    - ``deploy_band``: loss on frozen physics band from GT mu @ t=0 (no future mu peek).
    """
    raw = (os.environ.get("CLOT_FORECAST_MASK") or "target").strip().lower()
    if raw in ("input", "t_in", "mu_in", "current"):
        return "input"
    if raw in ("deploy_pred", "pred_deploy", "deploy_model"):
        return "deploy_pred"
    if raw in ("deploy_band", "frozen_t0", "b0_band", "t0_band"):
        return "deploy_band"
    if raw in ("deploy", "deploy_input", "deploy_in", "band"):
        return "deploy_input"
    return "target"


def clot_forecast_deploy_mask_needs_model() -> bool:
    return clot_forecast_mask_mode() == "deploy_pred"


def clot_forecast_deploy_loss_enabled() -> bool:
    return clot_forecast_mask_mode() in ("deploy_input", "deploy_pred")


def resolve_rollout_prev_mu_si(
    rollout_state,
    step: ClotPhiStepBatch,
    device: torch.device,
    *,
    time_index: int = 0,
    train_epoch: int | None = None,
) -> torch.Tensor:
    """Deploy band seeds: GT mu during carry bridge; else carried pred mu or GT @ step."""
    from src.core_physics.clot_phi_rollout import carry_gt_fade_alpha, carry_gt_warmup_active

    if clot_phi_fixed_mu_from_phi_enabled():
        phi_prev = rollout_state.phi_prev if rollout_state is not None else None
        return mu_eff_from_carried_phi(step.mu_c_si, phi_prev, device=device)

    mu_gt = step.mu_gt_cap.reshape(-1).to(device=device)
    fade_a = carry_gt_fade_alpha(train_epoch)
    if fade_a is not None:
        if rollout_state is not None and rollout_state.log_mu_prev is not None:
            mu_pred = torch.exp(rollout_state.log_mu_prev.clamp(max=20.0)).reshape(-1).to(device=device)
        else:
            mu_pred = mu_gt
        return (1.0 - fade_a) * mu_gt + fade_a * mu_pred
    if carry_gt_warmup_active(int(time_index), train_epoch):
        return mu_gt
    if rollout_state is not None and rollout_state.log_mu_prev is not None:
        return torch.exp(rollout_state.log_mu_prev.clamp(max=20.0)).reshape(-1).to(device=device)
    return mu_gt


def build_deploy_eligible_phi_gt(
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    use_soft: bool = False,
) -> torch.Tensor:
    """GT clot labels on eligible lumen (not oracle neighbor shell only)."""
    from src.core_physics.clot_phi_simple import (
        _lumen_supervision_eligible,
        _wall_mask_from_data,
        carreau_mu_si_from_uv,
        phi_gt_binary,
        phi_gt_soft,
    )

    y = data.y[int(time_index)].to(device)
    mu_gt = phys_cfg.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])
    mu_cap = cap_mu_eff_si(mu_gt)
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n)
    eligible = _lumen_supervision_eligible(data, device, wall, n)
    mu_c = carreau_mu_si_from_uv(data, y[:, 0], y[:, 1], phys_cfg)
    if use_soft:
        return phi_gt_soft(mu_cap, mu_c, eligible)
    return phi_gt_binary(mu_cap, eligible, phys_cfg)


def clot_forecast_extra_feature_dim() -> int:
    if not clot_forecast_one_step_enabled():
        return 0
    return 1 if clot_forecast_input_mu_enabled() else 0


def resolve_forecast_deploy_loss_region(
    data,
    *,
    step_in: ClotPhiStepBatch,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    phi: torch.Tensor,
    prev_mu_eff_si: torch.Tensor,
    mu_mlp_si: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deploy neighbor commit band for forecast training/eval."""
    from src.core_physics.clot_phi_mu_inject import resolve_deploy_neighbor_commit_mask

    return resolve_deploy_neighbor_commit_mask(
        data,
        device,
        phi=phi,
        prev_mu_eff_si=prev_mu_eff_si,
        phys_cfg=phys_cfg,
        u_nd=step_in.u_flow_nd,
        v_nd=step_in.v_flow_nd,
        bio_cfg=bio_cfg,
        mu_c_si=step_in.mu_c_si,
        mu_mlp_si=mu_mlp_si,
    )


def resolve_forecast_loss_region(
    data,
    *,
    step_in: ClotPhiStepBatch,
    mu_cap_out: torch.Tensor,
    t_in: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    phi_deploy: torch.Tensor | None = None,
    mu_mlp_deploy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervision / loss mask for one-step forecast pairs."""
    mode = clot_forecast_mask_mode()
    if mode == "input":
        return supervision_region_mask(data, device, step_in.mu_gt_cap, phys_cfg)
    if mode == "deploy_band":
        return resolve_clot_support_band(
            data,
            device,
            step_in.mu_gt_cap,
            phys_cfg,
            bio_cfg,
            frozen_t0=True,
        )
    if mode in ("deploy_input", "deploy_pred"):
        phi = phi_deploy if mode == "deploy_pred" else step_in.phi_gt
        if phi is None:
            phi = step_in.phi_gt
        return resolve_forecast_deploy_loss_region(
            data,
            step_in=step_in,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            phi=phi,
            prev_mu_eff_si=step_in.mu_gt_cap,
            mu_mlp_si=mu_mlp_deploy,
        )
    return supervision_region_mask(data, device, mu_cap_out, phys_cfg)


def resolve_forecast_deploy_mask_from_model(
    data,
    *,
    step: ClotPhiStepBatch,
    model: "nn.Module",
    edge_index: torch.Tensor | None,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    hybrid: bool,
) -> torch.Tensor:
    """Build deploy-faithful loss band using detached model outputs @ t_in."""
    if step.mu_in_cap is None:
        raise ValueError("forecast deploy mask requires mu_in_cap on step batch")
    step_in = ClotPhiStepBatch(
        features=step.features,
        phi_gt=step.phi_in_gt if step.phi_in_gt is not None else step.phi_gt,
        mu_c_si=step.mu_c_si,
        mu_gt_cap=step.mu_in_cap,
        region=step.region,
        loss_mask=step.loss_mask,
        species_log_gt=step.species_log_gt,
        u_flow_nd=step.u_flow_nd,
        v_flow_nd=step.v_flow_nd,
    )
    with torch.no_grad():
        if clot_phi_model_uses_mpnn(model):
            if edge_index is None:
                raise ValueError("mpnn model requires edge_index for deploy forecast mask")
            logits = model.forward_logits(step.features, edge_index)
            dlog_fn = getattr(model, "forward_delta_log_mu", None)
            dlog = dlog_fn(step.features, edge_index) if dlog_fn is not None else None
        else:
            logits = model.forward_logits(step.features)
            dlog_fn = getattr(model, "forward_delta_log_mu", None)
            dlog = dlog_fn(step.features) if dlog_fn is not None else None
        phi_pred = torch.sigmoid(logits.reshape(-1))
        mu_mlp = None
        if hybrid and dlog is not None:
            mu_mlp = mu_eff_from_delta_log_si(step.mu_c_si, dlog)
        elif not hybrid and clot_phi_fixed_mu_from_phi_enabled():
            mu_mlp = log_blend_mu_eff_si(step.mu_c_si, phi_pred)
    mode = clot_forecast_mask_mode()
    phi = phi_pred if mode == "deploy_pred" else (step.phi_in_gt if step.phi_in_gt is not None else step_in.phi_gt)
    return resolve_forecast_deploy_loss_region(
        data,
        step_in=step_in,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        phi=phi.reshape(-1),
        prev_mu_eff_si=step.mu_in_cap,
        mu_mlp_si=mu_mlp,
    )


def resolve_forecast_log_mu_in(
    *,
    gt_mu_cap_si: torch.Tensor,
    mu_c_si: torch.Tensor,
    forecast_state: object | None,
    time_index: int,
    train_epoch: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Resolve log(mu @ t_in) for forecast node features (GT, carry, or curriculum blend)."""
    from src.core_physics.clot_phi_rollout import (
        ClotPhiRolloutState,
        carry_gt_fade_alpha,
        carry_gt_warmup_active,
    )

    log_gt = torch.log(
        gt_mu_cap_si.reshape(-1).to(device=device, dtype=torch.float32).clamp(min=1e-8)
    )
    if not clot_forecast_mu_carry_enabled():
        return log_gt

    log_pred: torch.Tensor | None = None
    if isinstance(forecast_state, ClotPhiRolloutState) and forecast_state.log_mu_prev is not None:
        log_pred = forecast_state.log_mu_prev.reshape(-1).to(device=device, dtype=log_gt.dtype)

    fade_a = carry_gt_fade_alpha(train_epoch)
    if fade_a is not None:
        if log_pred is None:
            log_pred = log_gt
        return (1.0 - fade_a) * log_gt + fade_a * log_pred

    if carry_gt_warmup_active(int(time_index), train_epoch):
        return log_gt

    if log_pred is not None:
        return log_pred

    if clot_forecast_mu_init_mode() == "gt":
        return log_gt
    return torch.log(mu_c_si.reshape(-1).to(device=device, dtype=log_gt.dtype).clamp(min=1e-8))


def resolve_forecast_mu_in_si(
    *,
    gt_mu_cap_si: torch.Tensor,
    mu_c_si: torch.Tensor,
    forecast_state: object | None,
    time_index: int,
    train_epoch: int | None,
    device: torch.device,
) -> torch.Tensor:
    log_mu = resolve_forecast_log_mu_in(
        gt_mu_cap_si=gt_mu_cap_si,
        mu_c_si=mu_c_si,
        forecast_state=forecast_state,
        time_index=time_index,
        train_epoch=train_epoch,
        device=device,
    )
    return torch.exp(log_mu.clamp(max=20.0))


def append_forecast_input_features(
    feats: torch.Tensor,
    log_mu_t: torch.Tensor,
    *,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Concatenate ``log(mu_t)`` when ``CLOT_FORECAST_INPUT_MU=1``."""
    if not clot_forecast_input_mu_enabled():
        return feats
    col = log_mu_t.reshape(-1, 1).to(device=device, dtype=dtype)
    if col.shape[0] != n_nodes:
        col = col[:n_nodes]
    return torch.cat([feats, col], dim=1)


def build_clot_forecast_pair_step(
    data,
    t_in: int,
    t_out: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    forecast_state: object | None = None,
    train_epoch: int | None = None,
) -> ClotPhiStepBatch:
    """One-step pair: flow/features @ t_in, supervision labels @ t_out."""
    step_in = build_clot_phi_step(data, int(t_in), phys_cfg, bio_cfg, device)
    y_out = data.y[int(t_out)].to(device)
    mu_gt_out = phys_cfg.viscosity_nd_to_si(y_out[:, STATE_CHANNEL_MU_EFF_ND])
    mu_cap_out = cap_mu_eff_si(mu_gt_out)
    mode = clot_forecast_mask_mode()
    from src.core_physics.clot_phi_simple import (
        _lumen_supervision_eligible,
        _wall_mask_from_data,
        phi_gt_binary,
        phi_gt_soft,
    )

    use_soft = _env_bool("CLOT_PHI_SOFT_LABELS", False)
    if mode in ("deploy_input", "deploy_pred"):
        n = int(data.num_nodes)
        wall = _wall_mask_from_data(data, device, n)
        eligible = _lumen_supervision_eligible(data, device, wall, n)
        if use_soft:
            phi_gt_out = phi_gt_soft(mu_cap_out, step_in.mu_c_si, eligible)
        else:
            phi_gt_out = phi_gt_binary(mu_cap_out, eligible, phys_cfg)
        region = torch.zeros(n, device=device, dtype=torch.float32)
        loss_mask = region.bool()
    else:
        region = resolve_forecast_loss_region(
            data,
            step_in=step_in,
            mu_cap_out=mu_cap_out,
            t_in=int(t_in),
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
        )
        if use_soft:
            phi_gt_out = phi_gt_soft(mu_cap_out, step_in.mu_c_si, region)
        else:
            phi_gt_out = phi_gt_binary(mu_cap_out, region, phys_cfg)
        loss_mask = region.bool()

    feats = step_in.features
    if clot_forecast_input_mu_enabled():
        log_mu_in = resolve_forecast_log_mu_in(
            gt_mu_cap_si=step_in.mu_gt_cap,
            mu_c_si=step_in.mu_c_si,
            forecast_state=forecast_state,
            time_index=int(t_in),
            train_epoch=train_epoch,
            device=device,
        )
        feats = append_forecast_input_features(
            feats,
            log_mu_in,
            n_nodes=int(data.num_nodes),
            device=device,
            dtype=feats.dtype,
        )
        mu_in_cap = torch.exp(log_mu_in.clamp(max=20.0))
    else:
        mu_in_cap = step_in.mu_gt_cap.reshape(-1)

    species_out = y_out[:, 4:16].to(device=device, dtype=torch.float32)
    return ClotPhiStepBatch(
        features=feats,
        phi_gt=phi_gt_out,
        mu_c_si=step_in.mu_c_si,
        mu_gt_cap=mu_cap_out,
        region=region,
        loss_mask=loss_mask,
        species_log_gt=species_out,
        u_flow_nd=step_in.u_flow_nd,
        v_flow_nd=step_in.v_flow_nd,
        mu_in_cap=mu_in_cap,
        phi_in_gt=step_in.phi_gt,
    )


def snapshot_clot_forecast_config() -> dict[str, object]:
    from src.core_physics.clot_phi_rollout import snapshot_carry_gt_warmup_config
    from src.core_physics.clot_phi_simple import snapshot_clot_support_config

    return {
        "forecast_mode": clot_forecast_mode() or "legacy",
        "forecast_one_step": clot_forecast_one_step_enabled(),
        "forecast_input_mu": clot_forecast_input_mu_enabled(),
        "forecast_pair_stride": clot_forecast_pair_stride(),
        "forecast_pair_schedule": clot_forecast_pair_schedule(),
        "forecast_mask": clot_forecast_mask_mode(),
        "forecast_mu_carry": clot_forecast_mu_carry_enabled(),
        "forecast_mu_carry_detach": clot_forecast_mu_carry_detach(),
        "forecast_mu_init": clot_forecast_mu_init_mode(),
        **snapshot_carry_gt_warmup_config(),
        **snapshot_clot_support_config(),
    }
