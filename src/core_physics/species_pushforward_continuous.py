"""Phase 2.5/2.6: continuous log-delta pushforward + soft-commit memory gate.

Train on ``delta_log = log(1+X_{t+1}) - log(1+X_t)``.
Phase 2.5: dense Huber on all ceiling-band nodes (collapses to zero-delta).
Phase 2.6: ``ActiveGrowthHuberLoss`` on GT-active deltas only + FP penalty (ceiling scope).
Eval/deploy: threshold only the **accumulated** log state (binary readout).
See ``docs/SPECIES_GNN_LADDER.md``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core_physics.species_pushforward_gnn import (
    STATE_DIM,
    SpeciesPushforwardGNN,
    filter_pushforward_windows,
    init_pushforward_from_snapshot,
    iter_pushforward_windows,
    maybe_noise_state,
    pushforward_channel_weights,
    pushforward_focal_alpha_channels,
    pushforward_focal_gamma_channels,
    pushforward_feature_dim,
    pushforward_input_noise,
    pushforward_step_stride,
    pushforward_train_t0_max,
    pushforward_train_t0_min,
    pushforward_unroll_steps,
    pushforward_window_t0_weight,
    step_loss_weights,
)
from src.core_physics.species_snapshot_gnn import (
    SpeciesSnapshotGNN,
    build_snapshot_features,
    fi_mat_active_labels,
    fi_mat_log_targets,
    snapshot_active_log_nd,
    snapshot_hidden_dim,
    snapshot_loss,
    snapshot_wall_hops,
    trigger_metrics,
)
from src.training.biochem_loss_policy import ActiveGrowthHuberLoss
from src.utils.paths import get_project_root

DEFAULT_CONTINUOUS_CKPT = "outputs/biochem/species_snapshot_s25/best.pth"
DEFAULT_S26_CKPT = "outputs/biochem/species_snapshot_s26/best.pth"
DEFAULT_S30_CKPT = "outputs/biochem/species_snapshot_s30/best.pth"
DEFAULT_S31_CKPT = "outputs/biochem/species_snapshot_s31/best.pth"
DEFAULT_S32_CKPT = "outputs/biochem/species_snapshot_s32/best.pth"
DEFAULT_S33_CKPT = "outputs/biochem/species_snapshot_s33/best.pth"
DEFAULT_S34_CKPT = "outputs/biochem/species_snapshot_s34/best.pth"

BIOCHEM_ANCHORS_6 = (
    "patient001",
    "patient002",
    "patient003",
    "patient004",
    "patient006",
    "patient007",
)
CH_FI = 0
CH_MAT = 1



def continuous_ckpt_path() -> Path:
    raw = (os.environ.get("SPECIES_CONTINUOUS_CKPT") or DEFAULT_CONTINUOUS_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def continuous_growth_only_loss() -> bool:
    raw = (os.environ.get("SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_dual_head() -> bool:
    raw = (os.environ.get("SPECIES_CONTINUOUS_DUAL_HEAD") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_saturation_gate() -> bool:
    raw = (os.environ.get("SPECIES_CONTINUOUS_SATURATION_GATE") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def saturation_headroom_scale() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_SATURATION_SCALE") or "80").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 80.0


def mature_clot_frac() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_MATURE_FRAC") or "0.95").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.95


def continuous_mature_fp_exempt() -> bool:
    raw = (os.environ.get("SPECIES_CONTINUOUS_MATURE_FP_EXEMPT") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_temporal_gate() -> bool:
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEMPORAL_GATE") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def temporal_lambda_bounds() -> tuple[float, float]:
    lo_raw = (os.environ.get("SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN") or "0.5").strip()
    hi_raw = (os.environ.get("SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX") or "1.5").strip()
    try:
        lo = float(lo_raw)
        hi = float(hi_raw)
    except ValueError:
        lo, hi = 0.5, 1.5
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def global_species_mass(log_state: torch.Tensor) -> torch.Tensor:
    """Scalar wall-band mass: sum of FI + Mat log1p ND (for temporal gate input)."""
    st = log_state.reshape(-1, STATE_DIM)
    return st.sum()


def global_species_mass_feature(log_state: torch.Tensor) -> torch.Tensor:
    """Normalized scalar feature in ``(1, 1)`` for the temporal MLP."""
    mass = global_species_mass(log_state)
    n = max(int(log_state.reshape(-1, STATE_DIM).shape[0]), 1)
    fi_max, mat_max = continuous_max_sat_log()
    scale = float(n) * float(fi_max + mat_max)
    return torch.log1p(mass / max(scale, 1e-6)).reshape(1, 1).to(
        device=log_state.device, dtype=log_state.dtype
    )


def continuous_feature_dim(latent_dim: int) -> int:
    """GNN input dim: ``[z_kin, sdf, state_norm, (optional sat_ratio)]``."""
    base = pushforward_feature_dim(int(latent_dim))
    if continuous_saturation_gate():
        return base + STATE_DIM
    return base


def continuous_vel_decay_enabled() -> bool:
    raw = (os.environ.get("SPECIES_CONTINUOUS_VEL_DECAY") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_teacher_noise_sigma() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEACHER_NOISE") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_teacher_fp_frac() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEACHER_FP_FRAC") or "0").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.0


def continuous_teacher_blur() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEACHER_BLUR") or "0").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.0


def pushforward_max_unroll_steps() -> int:
    raw = (os.environ.get("SPECIES_PUSHFORWARD_MAX_UNROLL") or "").strip()
    if raw:
        try:
            return max(int(float(raw)), 1)
        except ValueError:
            pass
    return max(pushforward_unroll_steps(), 10)


def tbptt_tail_steps() -> int:
    raw = (os.environ.get("SPECIES_CONTINUOUS_TBPTT_TAIL") or "5").strip()
    try:
        return max(int(float(raw)), 1)
    except ValueError:
        return 5


def closed_loop_init_prob() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_CLOSED_LOOP_INIT") or "0").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.0


def curriculum_unroll_for_epoch(epoch: int) -> int:
    """Epoch-based unroll length (Phase 4 curriculum)."""
    if (os.environ.get("SPECIES_CONTINUOUS_CURRICULUM_UNROLL") or "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return pushforward_unroll_steps()
    max_u = pushforward_max_unroll_steps()
    if epoch <= 10:
        return min(5, max_u)
    if epoch <= 20:
        return min(10, max_u)
    if epoch <= 30:
        return min(15, max_u)
    if epoch <= 40:
        return min(25, max_u)
    return min(30, max_u)


def continuous_spatial_loss_weight() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT") or "1.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1.0


def continuous_delta_threshold() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_THRESH") or "1e-5").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1e-5


def continuous_delta_threshold_channels() -> tuple[float, float]:
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi = _env_float_channel("SPECIES_CONTINUOUS_DELTA_THRESH_FI", continuous_delta_threshold())
    mat = _env_float_channel("SPECIES_CONTINUOUS_DELTA_THRESH_MAT", 5e-6)
    return fi, mat


def continuous_delta_value_scale() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_VALUE_SCALE") or "100000").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 1e5


def continuous_underpred_weight() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_UNDERPRED_WEIGHT") or "2.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 2.0


def continuous_fp_weight() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_FP_WEIGHT") or "0.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_fp_threshold() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_FP_THRESH") or "2e-5").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 2e-5


def continuous_loss_scale() -> float:
    default = "1" if continuous_growth_only_loss() else "10000"
    raw = (os.environ.get("SPECIES_CONTINUOUS_LOSS_SCALE") or default).strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return float(default)


def continuous_huber_beta_growth() -> float:
    """Huber beta on value-scaled deltas (O(1) domain)."""
    if continuous_growth_only_loss():
        raw = (os.environ.get("SPECIES_CONTINUOUS_HUBER_BETA") or "1.0").strip()
        try:
            return max(float(raw), 1e-6)
        except ValueError:
            return 1.0
    return continuous_huber_beta()


def _growth_huber() -> ActiveGrowthHuberLoss:
    fi_max, mat_max = continuous_max_sat_log()
    return ActiveGrowthHuberLoss(
        delta_threshold=continuous_delta_threshold(),
        delta_threshold_channels=continuous_delta_threshold_channels(),
        beta=continuous_huber_beta_growth(),
        fp_weight=continuous_fp_weight(),
        fp_threshold=continuous_fp_threshold(),
        value_scale=continuous_delta_value_scale(),
        channel_weight=continuous_channel_weights(),
        underpred_weight=continuous_underpred_weight(),
        mature_frac=mature_clot_frac(),
        mature_exempt_fp=continuous_mature_fp_exempt(),
        mature_max_log=(fi_max, mat_max),
    )


def continuous_huber_beta() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_HUBER_BETA") or "1e-4").strip()
    try:
        return max(float(raw), 1e-8)
    except ValueError:
        return 1e-4


def continuous_channel_weights() -> tuple[float, float]:
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi = _env_float_channel("SPECIES_CONTINUOUS_CHANNEL_WEIGHT_FI", 1.0)
    mat = _env_float_channel("SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT", 2.0)
    return fi, mat


def continuous_max_sat_log() -> tuple[float, float]:
    """Per-channel log1p saturation after commit (freeze mature clot)."""
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi = _env_float_channel("SPECIES_CONTINUOUS_MAX_SAT_LOG_FI", 0.002)
    mat = _env_float_channel("SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT", 0.002)
    return fi, mat


def continuous_state_scale() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_STATE_SCALE") or "0.002").strip()
    try:
        return max(float(raw), 1e-8)
    except ValueError:
        return 0.002


def continuous_mat_commit_thresh() -> float:
    """Binary readout threshold on accumulated log Mat (deploy)."""
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    return _env_float_channel("SPECIES_CONTINUOUS_MAT_COMMIT_THRESH", snapshot_active_log_nd())


def log_series_on_band(
    data,
    time_indices: Sequence[int],
    device: torch.device,
    node_idx: torch.Tensor,
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for t in time_indices:
        out.append(fi_mat_log_targets(data, int(t), device)[node_idx])
    return out


def band_speed_at_time(
    data,
    time_index: int,
    device: torch.device,
    node_idx: torch.Tensor,
) -> torch.Tensor:
    """Normalized wall-band speed in [0, 1] from GT ``[u, v]``."""
    y = data.y[int(time_index)].to(device=device, dtype=torch.float32)
    u, v = y[:, 0], y[:, 1]
    speed = torch.sqrt(u * u + v * v + 1e-12)
    spd = speed[node_idx.reshape(-1)]
    mx = spd.max().clamp(min=1e-6)
    return (spd / mx).clamp(0.0, 1.0)


def band_speed_series(
    data,
    time_indices: Sequence[int],
    device: torch.device,
    node_idx: torch.Tensor,
) -> list[torch.Tensor]:
    return [band_speed_at_time(data, int(t), device, node_idx) for t in time_indices]


def _graph_blur_band_state(
    state: torch.Tensor,
    edge_index: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if alpha <= 0.0:
        return state
    row, col = edge_index
    n = int(state.shape[0])
    deg = torch.zeros(n, device=state.device, dtype=state.dtype)
    deg.index_add_(0, col, torch.ones(col.numel(), device=state.device, dtype=state.dtype))
    deg = deg.clamp(min=1.0)
    agg = torch.zeros_like(state)
    agg.index_add_(0, col, state[row])
    neighbor_mean = agg / deg.unsqueeze(-1)
    return ((1.0 - alpha) * state + alpha * neighbor_mean).clamp(min=0.0)


def noisy_teacher_log_state0(
    log_state0: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    """Corrupt GT-anchored window start to mimic deploy false-positive bleed."""
    sigma = continuous_teacher_noise_sigma()
    fp_frac = continuous_teacher_fp_frac()
    blur = continuous_teacher_blur()
    if not training or (sigma <= 0.0 and fp_frac <= 0.0 and blur <= 0.0):
        return log_state0.clone()
    st = log_state0.reshape(-1, STATE_DIM).clone()
    if sigma > 0.0:
        st = st + torch.randn_like(st) * sigma
    if fp_frac > 0.0:
        n = int(st.shape[0])
        k = max(1, int(round(n * fp_frac)))
        idx = torch.randperm(n, device=st.device)[:k]
        bump = torch.tensor(0.05, device=st.device, dtype=st.dtype)
        st[idx] = torch.maximum(st[idx], bump)
    if blur > 0.0:
        st = _graph_blur_band_state(st, edge_index, blur)
    return st.clamp(min=0.0)


def model_vel_decay_alphas(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not continuous_vel_decay_enabled():
        return None
    if not hasattr(model, "log_vel_decay_fi") or not hasattr(model, "log_vel_decay_mat"):
        return None
    a_fi = F.softplus(model.log_vel_decay_fi)
    a_mat = F.softplus(model.log_vel_decay_mat)
    return a_fi, a_mat


def apply_velocity_decay(
    log_state: torch.Tensor,
    wall_speed: torch.Tensor,
    alphas: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    st = log_state.reshape(-1, STATE_DIM)
    spd = wall_speed.reshape(-1, 1).to(device=st.device, dtype=st.dtype)
    a_fi, a_mat = alphas
    decay = torch.zeros_like(st)
    decay[:, CH_FI] = a_fi * spd.squeeze(-1) * st[:, CH_FI]
    decay[:, CH_MAT] = a_mat * spd.squeeze(-1) * st[:, CH_MAT]
    return (st - decay).clamp(min=0.0)


def log_delta_targets(log_prev: torch.Tensor, log_next: torch.Tensor) -> torch.Tensor:
    return (log_next.reshape(-1, STATE_DIM) - log_prev.reshape(-1, STATE_DIM)).to(dtype=torch.float32)


def normalize_log_state(log_state: torch.Tensor) -> torch.Tensor:
    scale = continuous_state_scale()
    return (log_state.reshape(-1, STATE_DIM) / scale).clamp(0.0, 1.0)


def saturation_ratio_features(log_state: torch.Tensor) -> torch.Tensor:
    """``X_t / X_max`` per channel in ``[0, 1]`` (local concentration inhibitor)."""
    st = log_state.reshape(-1, STATE_DIM)
    fi_max, mat_max = continuous_max_sat_log()
    maxes = torch.tensor([fi_max, mat_max], device=st.device, dtype=st.dtype).clamp(min=1e-8)
    return (st / maxes.unsqueeze(0)).clamp(0.0, 1.0)


def build_continuous_step_features(
    base_feats: torch.Tensor,
    log_state: torch.Tensor,
    *,
    training: bool = True,
) -> torch.Tensor:
    state_norm = normalize_log_state(log_state)
    state_in = maybe_noise_log_state(state_norm, training=training)
    feats = torch.cat([base_feats, state_in], dim=-1)
    if continuous_saturation_gate():
        feats = torch.cat([feats, saturation_ratio_features(log_state)], dim=-1)
    return feats


def apply_magnitude_headroom_clamp(
    magnitude: torch.Tensor,
    log_state: torch.Tensor,
) -> torch.Tensor:
    """Differentiable soft-clamp: crush delta when near Mat/FI saturation ceiling."""
    st = log_state.reshape(-1, STATE_DIM)
    fi_max, mat_max = continuous_max_sat_log()
    maxes = torch.tensor([fi_max, mat_max], device=st.device, dtype=st.dtype)
    margin = maxes.unsqueeze(0) - st
    scale = saturation_headroom_scale()
    headroom = F.softplus(margin * scale) * continuous_delta_out_scale()
    return torch.minimum(magnitude.reshape(-1, STATE_DIM), headroom)


def denormalize_log_state(norm_state: torch.Tensor) -> torch.Tensor:
    scale = continuous_state_scale()
    return norm_state.reshape(-1, STATE_DIM) * scale


def soft_commit_log_state(
    log_state: torch.Tensor,
    *,
    straight_through: bool = False,
) -> torch.Tensor:
    """Freeze committed nodes at saturation (cannot un-clot)."""
    st = log_state.reshape(-1, STATE_DIM).clone()
    thr = snapshot_active_log_nd()
    sat_fi, sat_mat = continuous_max_sat_log()
    sats = (sat_fi, sat_mat)
    out = st.clone()
    for ch in range(STATE_DIM):
        commit_thr = continuous_mat_commit_thresh() if ch == CH_MAT else thr
        committed = st[:, ch] > commit_thr
        sat_v = torch.tensor(sats[ch], device=st.device, dtype=st.dtype)
        frozen = torch.where(committed, sat_v, st[:, ch])
        if straight_through and bool(committed.any().item()):
            hard = committed.float()
            frozen = hard * sat_v + (1.0 - hard) * st[:, ch]
            frozen = frozen + (torch.where(committed, sat_v, st[:, ch]) - frozen).detach()
        out[:, ch] = frozen
    return out.clamp(min=0.0)


def log_state_to_active(log_state: torch.Tensor) -> torch.Tensor:
    """Binary readout from accumulated log state (eval/deploy)."""
    mat_thr = continuous_mat_commit_thresh()
    fi_thr = snapshot_active_log_nd()
    st = log_state.reshape(-1, STATE_DIM)
    out = torch.zeros_like(st)
    out[:, CH_FI] = (st[:, CH_FI] > fi_thr).float()
    out[:, CH_MAT] = (st[:, CH_MAT] > mat_thr).float()
    return out


def step_has_growth_supervision(tgt_delta: torch.Tensor, mask: torch.Tensor) -> bool:
    m = mask.reshape(-1).to(device=tgt_delta.device).bool()
    if not bool(m.any().item()):
        return False
    t = tgt_delta.reshape(-1, tgt_delta.shape[-1])[m]
    fi_thr, mat_thr = continuous_delta_threshold_channels()
    return bool((t[:, CH_FI] > fi_thr).any().item() or (t[:, CH_MAT] > mat_thr).any().item())


def continuous_delta_loss(
    pred_delta: torch.Tensor,
    tgt_delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float | None = None,
    channel_weight: tuple[float, float] | None = None,
    current_log_state: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Delta step loss inside deployable ceiling ``mask`` (wall + hops, not GT clot)."""
    m = mask.reshape(-1).to(device=pred_delta.device).bool()
    if not bool(m.any().item()):
        return None
    if continuous_growth_only_loss():
        if not step_has_growth_supervision(tgt_delta, m):
            return None
        loss = _growth_huber()(pred_delta, tgt_delta, m, current_log_state=current_log_state)
        return loss * continuous_loss_scale()
    b = continuous_huber_beta() if beta is None else float(beta)
    cw = channel_weight if channel_weight is not None else continuous_channel_weights()
    p = pred_delta[m]
    t = tgt_delta[m]
    lf = F.huber_loss(p[:, CH_FI], t[:, CH_FI], delta=b, reduction="mean")
    lm = F.huber_loss(p[:, CH_MAT], t[:, CH_MAT], delta=b, reduction="mean")
    return (cw[0] * lf + cw[1] * lm) * continuous_loss_scale()


def growth_supervision_mask(
    log_series: Sequence[torch.Tensor],
    *,
    delta_threshold: float | None = None,
) -> torch.Tensor:
    """Per-node channels that grew in a band-local log window."""
    thr = continuous_delta_threshold() if delta_threshold is None else float(delta_threshold)
    if len(log_series) < 2:
        n = int(log_series[0].shape[0]) if log_series else 0
        dev = log_series[0].device if log_series else torch.device("cpu")
        return torch.zeros(n, STATE_DIM, device=dev, dtype=torch.bool)
    fi_thr, mat_thr = continuous_delta_threshold_channels()
    grew = torch.zeros_like(log_series[0], dtype=torch.bool)
    for step in range(len(log_series) - 1):
        d = log_delta_targets(log_series[step], log_series[step + 1])
        grew[:, CH_FI] = grew[:, CH_FI] | (d[:, CH_FI] > fi_thr)
        grew[:, CH_MAT] = grew[:, CH_MAT] | (d[:, CH_MAT] > mat_thr)
    return grew


def growth_only_final_state_loss(
    final_pred: torch.Tensor,
    log_series: list[torch.Tensor],
    band_mask: torch.Tensor,
    *,
    beta: float | None = None,
) -> torch.Tensor:
    """Final-state Huber only on nodes that grew during the unroll window."""
    m = band_mask.reshape(-1).to(device=final_pred.device).bool()
    if not bool(m.any().item()) or len(log_series) < 2:
        return final_pred.sum() * 0.0
    grew = growth_supervision_mask(log_series).any(dim=-1) & m
    if not bool(grew.any().item()):
        return final_pred.sum() * 0.0
    b = continuous_huber_beta_growth() if beta is None else float(beta)
    scale = continuous_delta_value_scale()
    tgt = log_series[-1][grew] * scale
    pred = final_pred[grew] * scale
    return F.huber_loss(pred, tgt, delta=b, reduction="mean") * continuous_loss_scale()


def pushforward_log_state_step(
    log_state: torch.Tensor,
    pred_delta: torch.Tensor,
    *,
    straight_through: bool = False,
    wall_speed: torch.Tensor | None = None,
    vel_decay_alphas: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    nxt = log_state.reshape(-1, STATE_DIM) + pred_delta.reshape(-1, STATE_DIM)
    if wall_speed is not None and vel_decay_alphas is not None:
        nxt = apply_velocity_decay(nxt, wall_speed, vel_decay_alphas)
    nxt = nxt.clamp(min=0.0)
    return soft_commit_log_state(nxt, straight_through=straight_through)


def continuous_final_state_weight() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT") or "0.25").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.25


def continuous_final_state_all_band() -> bool:
    raw = (os.environ.get("SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_speed_fp_weight() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_SPEED_FP_WEIGHT") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def deploy_horizon_steps() -> int:
    raw = (os.environ.get("SPECIES_CONTINUOUS_DEPLOY_HORIZON") or "0").strip()
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return 0


def continuous_delta_out_scale() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_OUT_SCALE") or "1e-5").strip()
    try:
        return max(float(raw), 1e-9)
    except ValueError:
        return 1e-5


def continuous_delta_softplus_beta() -> float:
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_SOFTPLUS_BETA") or "20").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 20.0


def delta_readout(raw_delta: torch.Tensor) -> torch.Tensor:
    """Non-negative log increment (species do not decrease on clot front)."""
    st = raw_delta.reshape(-1, STATE_DIM)
    if continuous_growth_only_loss():
        # ReLU dead-zone on negative logits caused zero-delta collapse; softplus keeps grads.
        return F.softplus(st, beta=continuous_delta_softplus_beta()) * continuous_delta_out_scale()
    return F.relu(st)


def maybe_noise_log_state(norm_state: torch.Tensor, *, training: bool) -> torch.Tensor:
    sigma = pushforward_input_noise()
    if not training or sigma <= 0.0:
        return norm_state
    noise = torch.randn_like(norm_state) * sigma
    return (norm_state + noise).clamp(0.0, 1.0)


class SpeciesContinuousPushforwardGNN(SpeciesPushforwardGNN):
    """Same GraphSAGE backbone; readout predicts continuous log-delta (FI, Mat)."""

    def __init__(self, in_dim: int, *, hidden: int | None = None, out_dim: int = STATE_DIM):
        super().__init__(in_dim, hidden=hidden, out_dim=out_dim)
        self.log_vel_decay_fi = nn.Parameter(torch.tensor(-8.0))
        self.log_vel_decay_mat = nn.Parameter(torch.tensor(-8.0))


class SpeciesDualHeadContinuousGNN(SpeciesSnapshotGNN):
    """Phase 3.5: decoupled spatial gate * magnitude delta (FI, Mat)."""

    def __init__(self, in_dim: int, *, hidden: int | None = None, out_dim: int = STATE_DIM):
        super().__init__(in_dim, hidden=hidden, out_dim=out_dim)
        self.log_vel_decay_fi = nn.Parameter(torch.tensor(-8.0))
        self.log_vel_decay_mat = nn.Parameter(torch.tensor(-8.0))
        h = self.hidden
        fused = h + self.in_dim
        self.spatial_head = nn.Sequential(
            nn.Linear(fused, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(fused, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )
        for head in (self.spatial_head, self.magnitude_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        self.temporal_gate: nn.Sequential | None = None
        if continuous_temporal_gate():
            self.temporal_gate = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            )
            for m in self.temporal_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def temporal_lambda_from_state(self, log_state: torch.Tensor) -> torch.Tensor:
        """Scalar integration pace ``lambda in [lo, hi]`` from global band mass."""
        if self.temporal_gate is None:
            return torch.tensor(1.0, device=log_state.device, dtype=log_state.dtype)
        feat = global_species_mass_feature(log_state)
        raw = self.temporal_gate(feat).squeeze()
        lo, hi = temporal_lambda_bounds()
        return lo + torch.sigmoid(raw) * (hi - lo)

    def forward_decoupled(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        log_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_orig = x
        h = self.forward_hidden(x, edge_index)
        h_fused = torch.cat([h, x_orig], dim=-1)
        spatial_logits = self.spatial_head(h_fused)
        spatial_gate = torch.sigmoid(spatial_logits)
        mag_raw = self.magnitude_head(h_fused)
        magnitude = F.softplus(mag_raw, beta=continuous_delta_softplus_beta()) * continuous_delta_out_scale()
        if continuous_saturation_gate() and log_state is not None:
            magnitude = apply_magnitude_headroom_clamp(magnitude, log_state)
        if continuous_temporal_gate() and log_state is not None and self.temporal_gate is not None:
            lam = self.temporal_lambda_from_state(log_state)
            magnitude = magnitude * lam
        pred_delta = spatial_gate * magnitude
        return pred_delta, spatial_logits, magnitude

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        log_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_delta, _, _ = self.forward_decoupled(x, edge_index, log_state=log_state)
        return pred_delta


def build_continuous_gnn(in_dim: int, *, hidden: int | None = None) -> nn.Module:
    if continuous_dual_head():
        return SpeciesDualHeadContinuousGNN(in_dim, hidden=hidden)
    return SpeciesContinuousPushforwardGNN(in_dim, hidden=hidden)


def predict_continuous_step_delta(
    model: nn.Module,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    log_state: torch.Tensor,
    *,
    training: bool = False,
) -> torch.Tensor:
    """One closed-loop delta step (features + optional sat/temporal gates)."""
    feats = build_continuous_step_features(base_feats, log_state, training=training)
    if continuous_dual_head() and isinstance(model, SpeciesDualHeadContinuousGNN):
        pred_delta, _, _ = model.forward_decoupled(feats, edge_index, log_state=log_state)
        return pred_delta
    return delta_readout(model(feats, edge_index))


def growth_delta_labels(tgt_delta: torch.Tensor) -> torch.Tensor:
    """Per-channel binary growth indicator from log-delta targets."""
    fi_thr, mat_thr = continuous_delta_threshold_channels()
    t = tgt_delta.reshape(-1, STATE_DIM)
    out = torch.zeros_like(t)
    out[:, CH_FI] = (t[:, CH_FI] > fi_thr).float()
    out[:, CH_MAT] = (t[:, CH_MAT] > mat_thr).float()
    return out


def dual_head_step_loss(
    spatial_logits: torch.Tensor,
    magnitude: torch.Tensor,
    tgt_delta: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    current_log_state: torch.Tensor | None = None,
) -> torch.Tensor | None:
    m = train_mask.reshape(-1).to(device=spatial_logits.device).bool()
    if not bool(m.any().item()):
        return None
    growth_tgt = growth_delta_labels(tgt_delta)
    if not step_has_growth_supervision(tgt_delta, m):
        return None
    alpha = pushforward_focal_alpha_channels()
    gamma = pushforward_focal_gamma_channels()
    ch_w = pushforward_channel_weights()
    spatial_l = snapshot_loss(
        spatial_logits,
        tgt_delta,
        growth_tgt,
        m,
        focal_alpha=alpha,
        focal_gamma=gamma,
        channel_weight=ch_w,
    )
    mag_l = _growth_huber()(magnitude, tgt_delta, m, current_log_state=current_log_state)
    return continuous_spatial_loss_weight() * spatial_l + mag_l


def init_dual_head_widen_from_checkpoint(
    model: SpeciesDualHeadContinuousGNN,
    ckpt_path: Path | str,
    *,
    prev_in_dim: int,
    device: torch.device,
    quiet: bool = False,
) -> bool:
    """Load prior dual-head ckpt and zero-pad extra saturation-ratio input channels."""
    path = Path(ckpt_path)
    if not path.is_file():
        if not quiet:
            print(f"[WARN] widen-init missing: {path}")
        return False
    payload = torch.load(path, map_location=device, weights_only=False)
    s1 = dict(payload.get("model_state") or {})
    s2 = model.state_dict()
    s1_in = int(prev_in_dim)
    s2_in = int(model.in_dim)
    if s2_in < s1_in:
        if not quiet:
            print(f"[WARN] widen-init dim mismatch prev={s1_in} new={s2_in}")
        return False
    h = int(model.hidden)
    copied = 0
    for key, w1 in s1.items():
        if key not in s2:
            continue
        w2 = s2[key]
        if w1.shape == w2.shape:
            s2[key] = w1.clone()
            copied += 1
            continue
        if key.endswith("conv1.lin_l.weight") and w1.ndim == 2 and w2.shape[0] == w1.shape[0]:
            pad = torch.zeros(w2.shape[0], s2_in - s1_in, dtype=w1.dtype, device=w1.device)
            s2[key] = torch.cat([w1.to(w2.device), pad], dim=1)
            copied += 1
            continue
        if key.endswith("conv1.lin_r.weight") and w1.ndim == 2 and w2.shape[0] == w1.shape[0]:
            pad = torch.zeros(w2.shape[0], s2_in - s1_in, dtype=w1.dtype, device=w1.device)
            s2[key] = torch.cat([w1.to(w2.device), pad], dim=1)
            copied += 1
            continue
        for prefix in ("spatial_head.0.weight", "magnitude_head.0.weight"):
            if key == prefix and w1.ndim == 2 and w2.shape[0] == w1.shape[0]:
                s2[key][:, :h] = w1[:, :h]
                s2[key][:, h : h + s1_in] = w1[:, h : h + s1_in]
                copied += 1
                break
    model.load_state_dict(s2)
    if not quiet:
        print(f"[OK] widen-init from {path} ({copied} tensors) {s1_in} -> {s2_in}", flush=True)
    return copied > 0


def init_dual_head_from_continuous(
    dual: SpeciesDualHeadContinuousGNN,
    single: nn.Module,
    *,
    quiet: bool = False,
) -> None:
    """Warm-start dual heads from single-readout continuous / s26 checkpoint."""
    sd = single.state_dict()
    dual_sd = dual.state_dict()
    for key in dual_sd:
        if key in sd:
            dual_sd[key] = sd[key]
    for prefix in ("spatial_head", "magnitude_head"):
        for suffix in (".0.weight", ".0.bias", ".2.weight", ".2.bias"):
            rk = f"readout{suffix}"
            pk = f"{prefix}{suffix}"
            if rk in sd and pk in dual_sd:
                dual_sd[pk] = sd[rk].clone()
    dual.load_state_dict(dual_sd)
    if not quiet:
        print("[OK] dual-head warm-start from single-readout checkpoint", flush=True)


def unroll_continuous_loss(
    model: nn.Module,
    *,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    log_series: list[torch.Tensor],
    train_mask: torch.Tensor,
    log_state0: torch.Tensor | None = None,
    speed_series: list[torch.Tensor] | None = None,
    training: bool = True,
    physics_ctx: Any | None = None,
    window_weight: float = 1.0,
    tbptt_tail: int | None = None,
    speed_fp_weight: float | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    n_steps = len(log_series) - 1
    if n_steps <= 0:
        z = base_feats.sum() * 0.0
        return z, [], []

    if log_state0 is None:
        log_state = torch.zeros(base_feats.shape[0], STATE_DIM, device=base_feats.device, dtype=base_feats.dtype)
    else:
        log_state = noisy_teacher_log_state0(log_state0, edge_index, training=training)
    vel_alphas = model_vel_decay_alphas(model)
    tail = tbptt_tail if tbptt_tail is not None else tbptt_tail_steps()
    loss_start = max(0, n_steps - int(tail))
    step_w = step_loss_weights(n_steps)
    losses: list[torch.Tensor] = []
    loss_ws: list[float] = []
    pred_deltas: list[torch.Tensor] = []
    states: list[torch.Tensor] = [log_state.clone()]

    for step in range(n_steps):
        grad_step = (not training) or step >= loss_start
        ctx = torch.enable_grad() if grad_step else torch.no_grad()
        with ctx:
            if step < loss_start and training:
                log_state = log_state.detach()
            feats = build_continuous_step_features(
                base_feats, log_state, training=training and grad_step
            )
            tgt_delta = log_delta_targets(log_series[step], log_series[step + 1])
            gt_log = log_series[step]
            if continuous_dual_head() and isinstance(model, SpeciesDualHeadContinuousGNN):
                pred_delta, spatial_logits, magnitude = model.forward_decoupled(
                    feats, edge_index, log_state=log_state
                )
                step_loss = dual_head_step_loss(
                    spatial_logits,
                    magnitude,
                    tgt_delta,
                    train_mask,
                    current_log_state=gt_log,
                )
            else:
                pred_delta = delta_readout(model(feats, edge_index))
                step_loss = continuous_delta_loss(
                    pred_delta, tgt_delta, train_mask, current_log_state=gt_log
                )
            pred_deltas.append(pred_delta)
            if step_loss is not None and grad_step:
                losses.append(step_loss)
                loss_ws.append(float(step_w[step]))
            spd = None
            if speed_series is not None and step + 1 < len(speed_series):
                spd = speed_series[step + 1]
            log_state = pushforward_log_state_step(
                log_state,
                pred_delta,
                straight_through=training and grad_step,
                wall_speed=spd,
                vel_decay_alphas=vel_alphas,
            )
            states.append(log_state.clone())

        if physics_ctx is not None:
            from src.core_physics.species_gelation_readout import (
                continuous_mu_loss_weight,
                continuous_phi_loss_weight,
                continuous_physics_readout,
                physics_readout_losses,
            )

            if continuous_physics_readout():
                t_next = int(physics_ctx.time_window[step + 1])
                phi_l, mu_l = physics_readout_losses(
                    log_state,
                    physics_ctx,
                    train_mask,
                    time_index=t_next,
                    device=log_state.device,
                )
                pw = continuous_phi_loss_weight()
                mw = continuous_mu_loss_weight()
                if pw > 0.0:
                    losses.append(phi_l * pw)
                    loss_ws.append(float(step_w[step]))
                if mw > 0.0:
                    losses.append(mu_l * mw)
                    loss_ws.append(float(step_w[step]))

    if not losses:
        z = base_feats.sum() * 0.0
        return z, pred_deltas, states
    wsum = max(sum(loss_ws), 1e-6)
    step_loss = sum(loss * w for loss, w in zip(losses, loss_ws)) / wsum
    fw = continuous_final_state_weight()
    if fw > 0.0 and states:
        m = train_mask.reshape(-1).bool()
        if bool(m.any().item()):
            if continuous_growth_only_loss() and not continuous_final_state_all_band():
                final_loss = growth_only_final_state_loss(states[-1], log_series, m)
            else:
                beta = continuous_huber_beta()
                final_tgt = log_series[-1][m]
                final_pred = states[-1][m]
                final_loss = F.huber_loss(final_pred, final_tgt, delta=beta, reduction="mean")
                final_loss = final_loss * continuous_loss_scale()
            step_loss = step_loss + fw * final_loss

    if physics_ctx is not None and states:
        from src.core_physics.species_gelation_readout import (
            continuous_mu_loss_weight,
            continuous_phi_loss_weight,
            continuous_physics_readout,
            physics_readout_losses,
        )

        if continuous_physics_readout():
            t_final = int(physics_ctx.time_window[-1])
            phi_l, mu_l = physics_readout_losses(
                states[-1],
                physics_ctx,
                train_mask,
                time_index=t_final,
                device=states[-1].device,
            )
            pw = continuous_phi_loss_weight() * 0.5
            mw = continuous_mu_loss_weight() * 0.5
            if pw > 0.0:
                step_loss = step_loss + pw * phi_l
            if mw > 0.0:
                step_loss = step_loss + mw * mu_l

    spw = continuous_speed_fp_weight() if speed_fp_weight is None else float(speed_fp_weight)
    if spw > 0.0 and states and speed_series and len(speed_series) == len(states):
        pred_active = log_state_to_active(states[-1])
        gt_active = fi_mat_active_labels(log_series[-1])
        spd = speed_series[-1].reshape(-1, 1).to(device=states[-1].device, dtype=states[-1].dtype)
        fp_mask = (pred_active > 0.5) & (gt_active <= 0.5)
        bleed = (fp_mask.float() * spd).mean()
        step_loss = step_loss + spw * bleed

    return step_loss * float(window_weight), pred_deltas, states


@torch.no_grad()
def rollout_continuous_states(
    model: nn.Module,
    *,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    log_series: list[torch.Tensor],
    log_state0: torch.Tensor | None = None,
    speed_series: list[torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Returns log states, pred deltas, and binary readout states per step."""
    model.eval()
    n_steps = len(log_series) - 1
    if log_state0 is None:
        log_state = torch.zeros(base_feats.shape[0], STATE_DIM, device=base_feats.device, dtype=base_feats.dtype)
    else:
        log_state = log_state0.clone()
    vel_alphas = model_vel_decay_alphas(model)
    log_states = [log_state.clone()]
    deltas: list[torch.Tensor] = []
    actives: list[torch.Tensor] = [log_state_to_active(log_state)]
    for step in range(n_steps):
        pred_delta = predict_continuous_step_delta(
            model, base_feats, edge_index, log_state, training=False
        )
        deltas.append(pred_delta)
        spd = None
        if speed_series is not None and step + 1 < len(speed_series):
            spd = speed_series[step + 1]
        log_state = pushforward_log_state_step(
            log_state,
            pred_delta,
            straight_through=False,
            wall_speed=spd,
            vel_decay_alphas=vel_alphas,
        )
        log_states.append(log_state.clone())
        actives.append(log_state_to_active(log_state))
    return log_states, deltas, actives


def rollout_prefix_log_state(
    model: nn.Module,
    data,
    static: dict,
    time_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Detached closed-loop state at ``time_index`` from GT t=0 (scheduled sampling)."""
    was_training = model.training
    model.eval()
    node_idx = static["node_idx"]
    t_end = max(0, int(time_index))
    log_state = fi_mat_log_targets(data, 0, device)[node_idx]
    vel_alphas = model_vel_decay_alphas(model)
    with torch.no_grad():
        for t in range(t_end):
            spd = band_speed_at_time(data, t + 1, device, node_idx)
            pred_delta = predict_continuous_step_delta(
                model,
                static["base_feats"],
                static["edge_index"],
                log_state,
                training=False,
            )
            log_state = pushforward_log_state_step(
                log_state,
                pred_delta,
                straight_through=False,
                wall_speed=spd,
                vel_decay_alphas=vel_alphas,
            )
    if was_training:
        model.train()
    return log_state.detach()


@torch.no_grad()
def eval_full_rollout_fimat_f1(
    model: nn.Module,
    data,
    static: dict,
    device: torch.device,
    *,
    time_index: int = 53,
) -> dict[str, float]:
    """Closed-loop deploy Mat/FI F1 at ``time_index`` (full timeline from t=0)."""
    model.eval()
    node_idx = static["node_idx"]
    n_times = int(data.y.shape[0])
    t_eval = max(0, min(int(time_index), n_times - 1))
    log_state = fi_mat_log_targets(data, 0, device)[node_idx]
    vel_alphas = model_vel_decay_alphas(model)
    for t in range(t_eval):
        spd = band_speed_at_time(data, t + 1, device, node_idx)
        pred_delta = predict_continuous_step_delta(
            model,
            static["base_feats"],
            static["edge_index"],
            log_state,
            training=False,
        )
        log_state = pushforward_log_state_step(
            log_state,
            pred_delta,
            straight_through=False,
            wall_speed=spd,
            vel_decay_alphas=vel_alphas,
        )
    gt_log = fi_mat_log_targets(data, t_eval, device)[node_idx]
    gt_active = log_state_to_active(gt_log)
    pred_active = log_state_to_active(log_state)
    band_m = torch.ones(log_state.shape[0], dtype=torch.bool, device=device)
    sm = trigger_metrics(pred_active, gt_active, band_m)
    return {
        "deploy_fi_f1": float(sm["fi_f1"]),
        "deploy_mat_f1": float(sm["mat_f1"]),
        "deploy_trigger_f1": float(sm["trigger_f1"]),
        "time_index": int(t_eval),
    }


@torch.no_grad()
def eval_deploy_clot_f1(
    model: nn.Module,
    data,
    static: dict,
    phys_cfg,
    bio_cfg,
    device: torch.device,
    *,
    time_index: int = 53,
    flow_source: str = "gt",
) -> dict[str, float]:
    """Closed-loop species rollout -> nucleation clot F1 at ``time_index`` (deploy physics)."""
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, rollout_t0_clot_phi
    from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
    from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL
    from src.training.train_clot_phi_simple import _clot_metrics

    model.eval()
    node_idx = static["node_idx"]
    n_times = int(data.y.shape[0])
    t_eval = max(0, min(int(time_index), n_times - 1))
    out = data.y.clone().to(device=device)
    log_state = fi_mat_log_targets(data, 0, device)[node_idx]
    vel_alphas = model_vel_decay_alphas(model)
    for t in range(n_times):
        sp = data.y[t, :, 4:16].to(device=device, dtype=torch.float32).clone()
        idx = node_idx.reshape(-1)
        sp[idx, FI_CHANNEL] = log_state[:, 0]
        sp[idx, MAT_CHANNEL] = log_state[:, 1]
        out[t, :, 4:16] = sp.clamp(min=0.0)
        if t >= n_times - 1:
            break
        spd = band_speed_at_time(data, t + 1, device, node_idx)
        pred_delta = predict_continuous_step_delta(
            model,
            static["base_feats"],
            static["edge_index"],
            log_state,
            training=False,
        )
        log_state = pushforward_log_state_step(
            log_state,
            pred_delta,
            straight_through=False,
            wall_speed=spd,
            vel_decay_alphas=vel_alphas,
        )
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data,
            phys_cfg,
            bio_cfg,
            device,
            gamma_mode=RUNG2_GAMMA_MODE,
            flow_source=flow_source,
            pred_species_series=out,
            nucleation=True,
            nucleation_hops=1,
        )
    mask = torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)
    phi_gt = gt_clot_phi_at_time(data, t_eval, phys_cfg, device)
    phi_pred = traj[t_eval]["phi"]
    m = _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask)
    return {
        "deploy_clot_f1": float(m["clot_f1"]),
        "deploy_clot_prec": float(m["clot_prec"]),
        "deploy_clot_rec": float(m["clot_rec"]),
        "time_index": int(t_eval),
    }


@torch.no_grad()
def eval_continuous_window(
    model: nn.Module,
    *,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    log_series: list[torch.Tensor],
    mask: torch.Tensor,
    log_state0: torch.Tensor | None = None,
    speed_series: list[torch.Tensor] | None = None,
    physics_ctx: Any | None = None,
) -> dict[str, float]:
    """Primary metric: final cumulative state F1 after soft-commit readout (ceiling mask)."""
    log_states, deltas, actives = rollout_continuous_states(
        model,
        base_feats=base_feats,
        edge_index=edge_index,
        log_series=log_series,
        log_state0=log_state0 if log_state0 is not None else log_series[0],
        speed_series=speed_series,
    )
    gt_active = fi_mat_active_labels(log_series[-1])
    pred_active = actives[-1]
    sm = trigger_metrics(pred_active, gt_active, mask)
    init_active = fi_mat_active_labels(log_series[0])
    sm_init = trigger_metrics(init_active, gt_active, mask)

    mean_growth_f1 = 0.0
    mean_growth_mat_f1 = 0.0
    if deltas:
        thr = continuous_delta_threshold()
        m = mask.reshape(-1).bool()
        gf1: list[float] = []
        gmf1: list[float] = []
        for step, pred_d in enumerate(deltas):
            tgt_d = log_delta_targets(log_series[step], log_series[step + 1])
            grew = tgt_d[m] > thr
            if not bool(grew.any().item()):
                continue
            pred_g = (pred_d[m][grew] > thr).float()
            tgt_g = torch.ones_like(pred_g)
            tp = float((pred_g * tgt_g).sum().item())
            fp = float((pred_g * (1.0 - tgt_g)).sum().item())
            fn = float(((1.0 - pred_g) * tgt_g).sum().item())
            prec = tp / max(tp + fp, 1e-6)
            rec = tp / max(tp + fn, 1e-6)
            gf1.append((2.0 * prec * rec) / max(prec + rec, 1e-6))
            mat_g = grew[:, CH_MAT]
            if bool(mat_g.any().item()):
                pg = (pred_d[m][:, CH_MAT][mat_g] > thr).float()
                tg = torch.ones_like(pg)
                tp_m = float((pg * tg).sum().item())
                fp_m = float((pg * (1.0 - tg)).sum().item())
                fn_m = float(((1.0 - pg) * tg).sum().item())
                prec_m = tp_m / max(tp_m + fp_m, 1e-6)
                rec_m = tp_m / max(tp_m + fn_m, 1e-6)
                gmf1.append((2.0 * prec_m * rec_m) / max(prec_m + rec_m, 1e-6))
        if gf1:
            mean_growth_f1 = sum(gf1) / len(gf1)
        if gmf1:
            mean_growth_mat_f1 = sum(gmf1) / len(gmf1)

    mean_delta_mag = 0.0
    if deltas:
        mean_delta_mag = float(torch.stack([d.abs().mean() for d in deltas]).mean().item())

    clot_phi_f1 = 0.0
    if physics_ctx is not None and log_states:
        from src.core_physics.species_gelation_readout import (
            band_log_state_to_species12,
            differentiable_clot_phi_from_species12,
            gt_phi_band_at_time,
        )

        t_final = int(physics_ctx.time_window[-1])
        sp12 = band_log_state_to_species12(log_states[-1], physics_ctx.rest_band)
        phi_pred = differentiable_clot_phi_from_species12(sp12, physics_ctx.bio_cfg)
        phi_gt = gt_phi_band_at_time(physics_ctx, t_final, log_states[-1].device)
        pred_b = (phi_pred[mask.reshape(-1).bool()] > 0.5).float()
        tgt_b = (phi_gt[mask.reshape(-1).bool()] > 0.5).float()
        tp = float((pred_b * tgt_b).sum().item())
        fp = float((pred_b * (1.0 - tgt_b)).sum().item())
        fn = float(((1.0 - pred_b) * tgt_b).sum().item())
        prec = tp / max(tp + fp, 1e-6)
        rec = tp / max(tp + fn, 1e-6)
        clot_phi_f1 = (2.0 * prec * rec) / max(prec + rec, 1e-6)

    return {
        "final_state_f1": float(sm["trigger_f1"]),
        "final_state_mat_f1": float(sm["mat_f1"]),
        "final_state_fi_f1": float(sm["fi_f1"]),
        "final_state_prec": float(sm["trigger_prec"]),
        "final_state_rec": float(sm["trigger_rec"]),
        "init_state_f1": float(sm_init["trigger_f1"]),
        "mean_growth_f1": float(mean_growth_f1),
        "mean_growth_mat_f1": float(mean_growth_mat_f1),
        "mean_pred_delta": float(mean_delta_mag),
        "clot_phi_f1": float(clot_phi_f1),
    }


def filter_continuous_windows(
    windows: list[list[int]],
    data,
    node_idx: torch.Tensor,
    device: torch.device,
    *,
    t0_max: int | None = None,
    min_delta_mag: float = 0.0,
) -> list[list[int]]:
    t0_cap = pushforward_train_t0_max() if t0_max is None else t0_max
    t0_min = pushforward_train_t0_min()
    out: list[list[int]] = []
    for win in windows:
        if t0_min is not None and int(win[0]) < int(t0_min):
            continue
        if t0_cap is not None and int(win[0]) > int(t0_cap):
            continue
        if pushforward_window_t0_weight(int(win[0])) <= 0.0:
            continue
        if min_delta_mag <= 0.0:
            out.append(win)
            continue
        series = log_series_on_band(data, win, device, node_idx)
        mag = 0.0
        for step in range(len(series) - 1):
            d = log_delta_targets(series[step], series[step + 1])
            mag = max(mag, float(d.abs().max().item()))
        if mag >= float(min_delta_mag):
            out.append(win)
    return out


def save_continuous_checkpoint(
    path: Path | str,
    model: nn.Module,
    meta: dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "in_dim": model.in_dim,
        "hidden": model.hidden,
        "out_dim": model.out_dim,
        "phase": str(meta.get("phase", "s25_continuous")),
        "meta": meta,
    }
    torch.save(payload, p)
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class SpeciesContinuousBundle:
    model: SpeciesContinuousPushforwardGNN
    latent_dim: int
    hidden: int
    unroll: int
    stride: int
    device: torch.device


def load_continuous_bundle(
    ckpt_path: Path | str | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
    architecture: str | None = None,
) -> SpeciesContinuousBundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else continuous_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] continuous checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    in_dim = int(payload.get("in_dim", 0))
    hidden = int(payload.get("hidden", snapshot_hidden_dim()))
    meta = dict(payload.get("meta") or {})
    ckpt_dual = bool(meta.get("dual_head") or payload.get("dual_head"))
    if ckpt_dual:
        os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] = "1"
    if bool(meta.get("saturation_gate")):
        os.environ["SPECIES_CONTINUOUS_SATURATION_GATE"] = "1"
    if bool(meta.get("vel_decay")):
        os.environ["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    if bool(meta.get("temporal_gate")):
        os.environ["SPECIES_CONTINUOUS_TEMPORAL_GATE"] = "1"
    if bool(meta.get("kin_per_vessel_norm")):
        os.environ["SPECIES_KIN_PER_VESSEL_NORM"] = "1"
    if meta.get("mature_fp_exempt") is not None:
        os.environ["SPECIES_CONTINUOUS_MATURE_FP_EXEMPT"] = "1" if bool(meta.get("mature_fp_exempt")) else "0"
    if architecture == "single":
        use_dual = False
    elif architecture == "dual":
        use_dual = True
    else:
        use_dual = ckpt_dual or continuous_dual_head()
    if use_dual:
        model = SpeciesDualHeadContinuousGNN(in_dim, hidden=hidden).to(dev)
    else:
        model = SpeciesContinuousPushforwardGNN(in_dim, hidden=hidden).to(dev)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()
    return SpeciesContinuousBundle(
        model=model,
        latent_dim=int(meta.get("latent_dim", in_dim - 1 - STATE_DIM)),
        hidden=hidden,
        unroll=int(meta.get("unroll", pushforward_unroll_steps())),
        stride=int(meta.get("stride", pushforward_step_stride())),
        device=dev,
    )


def init_continuous_from_snapshot(
    model: SpeciesContinuousPushforwardGNN,
    snapshot_ckpt: Path | str,
    *,
    quiet: bool = False,
) -> bool:
    return init_pushforward_from_snapshot(model, snapshot_ckpt, quiet=quiet)
