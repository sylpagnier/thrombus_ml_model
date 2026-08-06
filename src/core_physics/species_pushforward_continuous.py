"""Phase 2.5/2.6: continuous log-delta pushforward + soft-commit memory gate.

Train on ``delta_log = log(1+X_{t+1}) - log(1+X_t)``.
Phase 2.5: dense Huber on all ceiling-band nodes (collapses to zero-delta).
Phase 2.6: ``ActiveGrowthHuberLoss`` on GT-active deltas only + FP penalty (ceiling scope).
Eval/deploy: threshold only the **accumulated** log state (binary readout).
See ``docs/SPECIES_GNN_LADDER.md``.
"""

from __future__ import annotations

import json
import math
import os
from src.core_physics.constants import *
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from src.architecture.pushforward_config import PushforwardConfig, resolve_config

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import species_channels as sc
from src.core_physics.species_pushforward_gnn import (
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
    species_log_targets,
    snapshot_active_log_nd,
    snapshot_hidden_dim,
    snapshot_loss,
    snapshot_wall_hops,
    trigger_metrics,
)
from src.training.biochem_species_scope import (
    FI_CHANNEL,
    MAT_CHANNEL,
    pushforward_local_index,
    pushforward_state_bulk_indices,
    pushforward_state_dim,
    scatter_log_state_to_species_block,
)
from src.config import VesselConfig
from src.training.biochem_loss_policy import ActiveGrowthHuberLoss
from src.utils.paths import get_project_root

# Legacy alias retained for compatibility; default now points at canonical biochem_gnn species checkpoint.
DEFAULT_CONTINUOUS_CKPT = "outputs/biochem/biochem_gnn/species/best.pth"
DEFAULT_S34_CKPT = "outputs/biochem/biochem_gnn/species/best.pth"

BIOCHEM_ANCHORS_6 = (
    "patient001",
    "patient002",
    "patient003",
    "patient004",
    "patient006",
    "patient007",
)
CH_FI = 0  # legacy alias when scope is fi_mat
CH_MAT = 1


def _local_bulk_index(bulk_channel: int) -> int | None:
    bulk = pushforward_state_bulk_indices()
    try:
        return bulk.index(int(bulk_channel))
    except ValueError:
        return None


def _local_fi_idx() -> int | None:
    return _local_bulk_index(FI_CHANNEL)


def _local_mat_idx() -> int | None:
    return _local_bulk_index(MAT_CHANNEL)


def continuous_max_sat_log_vec(
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-local-channel saturation ceilings, length ``_sd()``."""
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi_max, mat_max = continuous_max_sat_log()
    default = _env_float_channel("SPECIES_CONTINUOUS_MAX_SAT_LOG_DEFAULT", fi_max)
    vals: list[float] = []
    for bch in pushforward_state_bulk_indices():
        if bch == FI_CHANNEL:
            vals.append(fi_max)
        elif bch == MAT_CHANNEL:
            vals.append(mat_max)
        else:
            vals.append(default)
    dev = device or torch.device("cpu")
    return torch.tensor(vals, device=dev, dtype=dtype).clamp(min=1e-8)


def continuous_delta_threshold_vec() -> list[float]:
    fi_thr, mat_thr = continuous_delta_threshold_channels()
    default = continuous_delta_threshold()
    out: list[float] = []
    for bch in pushforward_state_bulk_indices():
        if bch == FI_CHANNEL:
            out.append(fi_thr)
        elif bch == MAT_CHANNEL:
            out.append(mat_thr)
        else:
            out.append(default)
    return out


def _scope_vec_from_fi_mat(fi_val: float, mat_val: float, *, default: float = 1.0) -> list[float]:
    out = [float(default)] * _sd()
    li_fi = _local_fi_idx()
    li_mat = _local_mat_idx()
    if li_fi is not None:
        out[li_fi] = float(fi_val)
    if li_mat is not None:
        out[li_mat] = float(mat_val)
    return out


def pushforward_focal_alpha_vec() -> list[float]:
    fi, mat = pushforward_focal_alpha_channels()
    return _scope_vec_from_fi_mat(fi, mat, default=0.9)


def pushforward_focal_gamma_vec() -> list[float]:
    from src.core_physics.species_snapshot_gnn import snapshot_focal_gamma

    fi, mat = pushforward_focal_gamma_channels()
    return _scope_vec_from_fi_mat(fi, mat, default=float(snapshot_focal_gamma()))


def _sd() -> int:
    return pushforward_state_dim()


def _ch_fi() -> int:
    return pushforward_local_index("fi")


def _ch_mat() -> int:
    return pushforward_local_index("mat")


def biochem_anchors_graph_dir(root: Path | None = None) -> Path:
    base = root if root is not None else get_project_root()
    return base / VesselConfig(phase="biochem_anchors").graph_output_dir


def discover_biochem_anchors(root: Path | None = None) -> list[str]:
    """All ``patient*.pt`` stems under ``graphs_biochem_anchors`` (sorted)."""
    anchor_dir = biochem_anchors_graph_dir(root)
    if not anchor_dir.is_dir():
        return list(BIOCHEM_ANCHORS_6)
    stems = sorted(p.stem for p in anchor_dir.glob("patient*.pt") if p.is_file())
    return stems or list(BIOCHEM_ANCHORS_6)


def parse_biochem_train_anchors(
    raw: str,
    *,
    all_anchors: bool,
    root: Path | None = None,
) -> list[str]:
    if all_anchors:
        return discover_biochem_anchors(root)
    items = [a.strip() for a in raw.split(",") if a.strip()]
    return items or ["patient007"]


# Historical short-horizon regime on patient007 (~53 macro-steps, ~2.2 h physical; F1 ~0.70-0.73).
# Not the default deploy eval horizon — use ``graph_last_time_index`` / ``deploy_eval_time_index``.
LEGACY_CAPPED_DEPLOY_HORIZON = 53


def graph_last_time_index(n_times: int) -> int:
    """Last macro-step index for a graph with ``n_times`` knots (0-based)."""
    return max(int(n_times) - 1, 0)


def legacy_capped_deploy_time_index(n_times: int) -> int:
    """Legacy short-horizon deploy checkpoint."""
    return min(LEGACY_CAPPED_DEPLOY_HORIZON, graph_last_time_index(n_times))


# Minimum steps left after the latest allowed t0, regardless of coverage_frac -- must clear the
# curriculum's largest unroll tier (15, at epoch>30; see curriculum_unroll_for_epoch) with margin.
TRAIN_T0_COVERAGE_MIN_RUNWAY = 20


def pushforward_train_t0_coverage_frac(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.train_t0_coverage_frac), 0.0)
    raw = (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_COVERAGE_FRAC") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def train_t0_max_for_n_times(n_times: int) -> int:
    """Per-vessel window start cap scaled to each graph's full timeline.

    Legacy formula caps at ~132 of a 200-step timeline (66%) -- the last third of the
    horizon is never sampled as a window START, only ever reached as a continuation of an
    earlier window. ``train_t0_coverage_frac`` (0 = this formula, unchanged) overrides that
    with an explicit fraction of the timeline, clamped to leave ``TRAIN_T0_COVERAGE_MIN_RUNWAY``
    steps of runway so a window starting near the end still has room for the curriculum's
    largest unroll.
    """
    last = graph_last_time_index(n_times)
    coverage = pushforward_train_t0_coverage_frac()
    if coverage > 0.0:
        cap = int(round(coverage * last))
        return max(0, min(cap, max(0, last - TRAIN_T0_COVERAGE_MIN_RUNWAY)))
    legacy_ref = legacy_capped_deploy_time_index(n_times)
    return max(legacy_ref, int(round(35.0 * last / max(legacy_ref, 1))))


def pushforward_train_t0_per_vessel(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return config.train_t0_per_vessel
    raw = (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def resolve_train_t0_max(n_times: int) -> int | None:
    if pushforward_train_t0_per_vessel():
        return train_t0_max_for_n_times(n_times)
    return pushforward_train_t0_max()



def continuous_ckpt_path() -> Path:
    raw = (os.environ.get("SPECIES_CONTINUOUS_CKPT") or DEFAULT_CONTINUOUS_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def continuous_growth_only_loss(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return config.growth_only_loss
    raw = (os.environ.get("SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_dual_head(config: PushforwardConfig | None = None) -> bool:

    config = resolve_config(config)
    if config is not None:
        return config.dual_head
    raw = (os.environ.get("SPECIES_CONTINUOUS_DUAL_HEAD") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_saturation_gate(config: PushforwardConfig | None = None) -> bool:

    config = resolve_config(config)
    if config is not None:
        return config.saturation_gate
    raw = (os.environ.get("SPECIES_CONTINUOUS_SATURATION_GATE") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def saturation_headroom_scale(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.saturation_scale
    raw = (os.environ.get("SPECIES_CONTINUOUS_SATURATION_SCALE") or "80").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 80.0


def saturation_headroom_scale_offwall(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.saturation_scale_offwall
    raw = os.environ.get("SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL")
    if raw is None:
        return saturation_headroom_scale()
    try:
        return max(float(raw.strip()), 1.0)
    except ValueError:
        return saturation_headroom_scale()


def mature_clot_frac(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.mature_frac
    raw = (os.environ.get("SPECIES_CONTINUOUS_MATURE_FRAC") or "0.95").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.95


def continuous_mature_fp_exempt(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return config.mature_fp_exempt
    raw = (os.environ.get("SPECIES_CONTINUOUS_MATURE_FP_EXEMPT") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_temporal_gate() -> bool:
    """Retired temporal lambda gate (kept off globally)."""
    return False


def temporal_lambda_bounds() -> tuple[float, float]:
    # Compatibility only; temporal gate is disabled.
    return 1.0, 1.0


def continuous_delta_residual(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return config.delta_residual
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_RESIDUAL") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_delta_residual_alpha(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.delta_residual_alpha
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_RESIDUAL_ALPHA") or "0.35").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.35


def continuous_temporal_offset(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return config.temporal_offset
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEMPORAL_OFFSET") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_temporal_offset_scale(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.temporal_offset_scale
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEMPORAL_OFFSET_SCALE") or "0.15").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.15


def continuous_neighbor_commit_gate(config: PushforwardConfig | None = None) -> bool:

    config = resolve_config(config)
    if config is not None:
        return config.neighbor_commit_gate
    raw = (os.environ.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE") or "0").strip().lower()
    return raw not in ("0", "false", "off", "no")


def continuous_neighbor_commit_alpha(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.neighbor_commit_alpha
    raw = (os.environ.get("SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA") or "0.8").strip()
    try:
        return max(0.0, min(float(raw), 1.0))
    except ValueError:
        return 0.8


def continuous_score_clot_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.score_clout_w
    raw = (os.environ.get("SPECIES_CONTINUOUS_SCORE_CLOUT_W") or "0").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.0


def global_species_mass(log_state: torch.Tensor) -> torch.Tensor:
    """Scalar wall-band mass: sum of FI + Mat log1p ND (for temporal gate input)."""
    st = log_state.reshape(-1, _sd())
    return st.sum()


def global_species_mass_feature(log_state: torch.Tensor) -> torch.Tensor:
    """Normalized scalar feature in ``(1, 1)`` for the temporal MLP."""
    mass = global_species_mass(log_state)
    n = max(int(log_state.reshape(-1, _sd()).shape[0]), 1)
    maxes = continuous_max_sat_log_vec(log_state.device, log_state.dtype)
    scale = float(n) * float(maxes.sum().item())
    return torch.log1p(mass / max(scale, 1e-6)).reshape(1, 1).to(
        device=log_state.device, dtype=log_state.dtype
    )


def continuous_time_context_enabled(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return config.time_context
    raw = (os.environ.get("SPECIES_CONTINUOUS_TIME_CONTEXT") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_time_ref_seconds(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.time_ref_s
    raw = (os.environ.get("SPECIES_CONTINUOUS_TIME_REF_S") or "3000").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 3000.0


def continuous_time_fourier_freqs(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return config.time_fourier_freqs
    raw = (os.environ.get("SPECIES_CONTINUOUS_TIME_FOURIER_FREQS") or "8").strip()
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return 8


def encode_continuous_time_features(
    time_index: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Global-time context: ``tau=t/t_ref`` + Fourier features shared across all graphs."""
    t_ref = continuous_time_ref_seconds()
    tau = max(float(time_index), 0.0) / t_ref
    tau = max(0.0, min(1.0, tau))
    out = [torch.tensor([tau], device=device, dtype=dtype)]
    n_freq = continuous_time_fourier_freqs()
    if n_freq > 0:
        ang = 2.0 * math.pi * tau
        for k in range(n_freq):
            w = float(2**k)
            out.append(torch.tensor([math.sin(w * ang)], device=device, dtype=dtype))
            out.append(torch.tensor([math.cos(w * ang)], device=device, dtype=dtype))
    return torch.cat(out, dim=0).reshape(1, -1)


def continuous_time_feature_dim() -> int:
    if not continuous_time_context_enabled():
        return 0
    return 1 + 2 * continuous_time_fourier_freqs()


def continuous_tail_feature_dim() -> int:
    """Trailing ``[state_norm, (sat_ratio), (time_context)]`` width after band base feats."""
    n = _sd()
    if continuous_saturation_gate():
        n += _sd()
    return n + continuous_time_feature_dim()


def continuous_feature_dim(latent_dim: int) -> int:
    """GNN input dim: ``[z_kin, sdf, band_extras..., state_norm, (sat), (time)]``."""
    base = pushforward_feature_dim(int(latent_dim))
    if continuous_saturation_gate():
        base += _sd()
    return base + continuous_time_feature_dim()


def continuous_vel_decay_enabled(config: PushforwardConfig | None = None) -> bool:

    config = resolve_config(config)
    if config is not None:
        return config.vel_decay
    raw = (os.environ.get("SPECIES_CONTINUOUS_VEL_DECAY") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_vel_decay_wall_only(config: PushforwardConfig | None = None) -> bool:

    config = resolve_config(config)
    if config is not None:
        return config.vel_decay_wall_only
    """If True, apply vel-decay only on wall nodes (lumen Mat is not washed out).

    Default ON: compound deploy must not let wall-trained alpha wipe lumen growth.
    Set SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY=0 for legacy full-band washout.
    """
    raw = (os.environ.get("SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_teacher_noise_sigma(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.teacher_noise
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEACHER_NOISE") or "0.02").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_teacher_fp_frac(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.teacher_fp_frac
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEACHER_FP_FRAC") or "0.08").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.0


def continuous_teacher_blur(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.teacher_blur
    raw = (os.environ.get("SPECIES_CONTINUOUS_TEACHER_BLUR") or "0.25").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.0


def pushforward_max_unroll_steps(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return config.max_unroll
    raw = (os.environ.get("SPECIES_PUSHFORWARD_MAX_UNROLL") or "200").strip()
    if raw:
        try:
            return max(int(float(raw)), 1)
        except ValueError:
            pass
    return max(pushforward_unroll_steps(), 10)


def tbptt_tail_steps(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return config.tbptt_tail
    raw = (os.environ.get("SPECIES_CONTINUOUS_TBPTT_TAIL") or "5").strip()
    try:
        return max(int(float(raw)), 1)
    except ValueError:
        return 5


def closed_loop_init_prob(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.closed_loop_init
    raw = (os.environ.get("SPECIES_CONTINUOUS_CLOSED_LOOP_INIT") or "0.45").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Scheduled Sampling (noisy-GT anchoring)
# ---------------------------------------------------------------------------

def scheduled_sampling_enabled(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return config.scheduled_sampling
    """Whether to inject intermittent GT anchoring during autoregressive unrolling."""
    return (os.environ.get("SPECIES_SCHEDULED_SAMPLING") or "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _ss_target_prob(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.ss_target_prob
    """Max probability of *keeping* the model's own prediction (vs resetting to GT).

    Ramps from 0 (pure GT anchoring) to this value over warmup, then stays.
    At deploy the model always keeps its own prediction (p=1), so higher target
    = closer to deploy distribution.
    """
    raw = (os.environ.get("SPECIES_SS_TARGET_PROB") or "0.5").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.5


def _ss_warmup_epochs(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return config.ss_warmup_epochs
    """Epochs of pure GT anchoring (p=0) before the ramp begins."""
    raw = (os.environ.get("SPECIES_SS_WARMUP_EPOCHS") or "3").strip()
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return 3


def _ss_anchor_stride(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return config.ss_anchor_stride
    """Only consider GT anchoring every *stride* steps (skip in-between).

    Default 10: potential reset points at step 10, 20, 30, ...
    """
    raw = (os.environ.get("SPECIES_SS_ANCHOR_STRIDE") or "10").strip()
    try:
        return max(int(float(raw)), 1)
    except ValueError:
        return 10


def _ss_noisy(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return config.ss_noisy
    """Use noisy GT (via noisy_teacher_log_state0) instead of clean GT.

    Noisy is strongly preferred: teaches error-recovery without oracle dependence.
    """
    return (os.environ.get("SPECIES_SS_NOISY") or "1").strip().lower() in (
        "1", "true", "yes", "on",
    )


def scheduled_sampling_keep_prob(epoch: int, max_epochs: int) -> float:
    """Probability of keeping model prediction (1-p = probability of GT reset).

    * Epochs < warmup: p = 0 (always reset to GT = pure teacher forcing).
    * Warmup -> max_epochs: linear ramp from 0 to target.
    * After max_epochs: p = target.
    """
    if not scheduled_sampling_enabled():
        return 1.0  # disabled: always keep own prediction (current behaviour)
    target = _ss_target_prob()
    warmup = _ss_warmup_epochs()
    if epoch < warmup:
        return 0.0
    ramp = max(max_epochs - warmup, 1)
    return min(target * (epoch - warmup) / ramp, target)


def curriculum_unroll_for_epoch(epoch: int, config: PushforwardConfig | None = None) -> int:
    """Epoch-based unroll length (Phase 4 curriculum)."""
    config = resolve_config(config)
    if config is not None:
        enabled = bool(config.curriculum_unroll)
    else:
        enabled = (os.environ.get("SPECIES_CONTINUOUS_CURRICULUM_UNROLL") or "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
    if not enabled:
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
    if max_u <= 60:
        return min(30, max_u)
    # Long-horizon graphs (full COMSOL export): ramp unroll past legacy 30-step cap.
    if epoch <= 50:
        return min(50, max_u)
    if epoch <= 60:
        return min(80, max_u)
    return min(120, max_u)


def continuous_spatial_loss_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.spatial_loss_weight
    raw = (os.environ.get("SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT") or "1.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1.0


def continuous_gate_temp(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.gate_temp
    """Temperature on the spatial-gate sigmoid: ``gate = sigmoid(logits / T)``.

    ``T < 1`` sharpens the gate toward a hard 0/1 decision (sparser support, fewer soft wall
    leaks -- the 6.16 "sparsify the gate" lever); ``T == 1`` is the original behaviour. Persisted in
    meta and restored at eval so the deploy field matches training.
    """
    raw = (os.environ.get("SPECIES_CONTINUOUS_GATE_TEMP") or "1.0").strip()
    try:
        return max(float(raw), 1e-3)
    except ValueError:
        return 1.0


def continuous_frontier_hops(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return config.frontier_hops
    """Restrict growth to the ``k``-hop frontier of *predicted* committed Mat (0 = off).

    Implements the "nucleate at a few sites, then propagate slowly" architecture: each step a node
    may only newly commit if it is within ``k`` hops of already-committed mass (or is a nucleation
    seed). ``k`` is the front advance per macro step -- the propagation-speed lever. DEPLOYABLE: the
    committed mask is read from the rollout ``log_state`` (the model's own prediction), never GT.
    """
    raw = (os.environ.get("SPECIES_CONTINUOUS_FRONTIER_HOPS") or "0").strip()
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return 0


def continuous_nucleation_topk(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.nucleation_topk
    """Deployable nucleation seed: fraction of band nodes allowed to nucleate by the model's own
    gate confidence (0 = off). When the frontier is empty (no committed mass yet, e.g. t0 where
    phi=0 at deploy), the top ``frac`` highest gate-logit nodes may seed -- the "choose a few areas"
    selector. Uses the model's spatial logits, NOT GT clot labels.
    """
    raw = (os.environ.get("SPECIES_CONTINUOUS_NUCLEATION_TOPK") or "0").strip()
    try:
        return min(max(float(raw), 0.0), 0.95)
    except ValueError:
        return 0.0


def continuous_delta_threshold(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.delta_thresh
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_THRESH") or str(DEFAULT_DELTA_THRESH)).strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 1e-5


def continuous_delta_threshold_channels() -> tuple[float, float]:
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi = _env_float_channel("SPECIES_CONTINUOUS_DELTA_THRESH_FI", continuous_delta_threshold())
    mat = _env_float_channel("SPECIES_CONTINUOUS_DELTA_THRESH_MAT", 5e-6)
    return fi, mat


def continuous_delta_value_scale(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.delta_value_scale
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_VALUE_SCALE") or str(DEFAULT_DELTA_VALUE_SCALE)).strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 1e5


def continuous_underpred_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.underpred_weight
    raw = (os.environ.get("SPECIES_CONTINUOUS_UNDERPRED_WEIGHT") or "2.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 2.0


def continuous_fp_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.fp_weight
    raw = (os.environ.get("SPECIES_CONTINUOUS_FP_WEIGHT") or "8").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_gate_fp_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.gate_fp_weight
    """Extra BCE pressure on spatial gate logits at zero-growth nodes (anti wall-paint)."""
    raw = (os.environ.get("SPECIES_CONTINUOUS_GATE_FP_WEIGHT") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_fp_threshold(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.fp_thresh
    raw = (os.environ.get("SPECIES_CONTINUOUS_FP_THRESH") or "2e-5").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 2e-5


def continuous_loss_scale(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.loss_scale
    default = "1" if continuous_growth_only_loss() else "10000"
    raw = (os.environ.get("SPECIES_CONTINUOUS_LOSS_SCALE") or default).strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return float(default)


def continuous_huber_beta_growth() -> float:
    """Huber beta on value-scaled deltas (O(1) domain)."""
    if continuous_growth_only_loss():
        raw = (os.environ.get("SPECIES_CONTINUOUS_HUBER_BETA") or "0.5").strip()
        try:
            return max(float(raw), 1e-6)
        except ValueError:
            return 0.5
    return continuous_huber_beta()


def _growth_huber() -> ActiveGrowthHuberLoss:
    return ActiveGrowthHuberLoss(
        delta_threshold=continuous_delta_threshold(),
        delta_threshold_channels=continuous_delta_threshold_vec(),
        beta=continuous_huber_beta_growth(),
        fp_weight=continuous_fp_weight(),
        fp_threshold=continuous_fp_threshold(),
        value_scale=continuous_delta_value_scale(),
        channel_weight=continuous_channel_weights_vec(),
        underpred_weight=continuous_underpred_weight(),
        mature_frac=mature_clot_frac(),
        mature_exempt_fp=continuous_mature_fp_exempt(),
        mature_max_log=continuous_max_sat_log_vec().tolist(),
    )


def continuous_huber_beta(config: PushforwardConfig | None = None) -> float:

    config = resolve_config(config)
    if config is not None:
        return config.huber_beta
    raw = (os.environ.get("SPECIES_CONTINUOUS_HUBER_BETA") or "1e-4").strip()
    try:
        return max(float(raw), 1e-8)
    except ValueError:
        return 1e-4


def continuous_channel_weights_vec() -> list[float]:
    bulk = pushforward_state_bulk_indices()
    cw_fi, cw_mat = continuous_channel_weights()
    weights = [1.0] * len(bulk)
    if FI_CHANNEL in bulk:
        weights[bulk.index(FI_CHANNEL)] = cw_fi
    if MAT_CHANNEL in bulk:
        weights[bulk.index(MAT_CHANNEL)] = cw_mat
    return weights


def continuous_channel_weights() -> tuple[float, float]:
    from src.architecture.pushforward_config import get_active_config
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    cfg = get_active_config()
    if cfg is not None:
        return float(cfg.channel_weight_fi), float(cfg.channel_weight_mat)
    fi = _env_float_channel("SPECIES_CONTINUOUS_CHANNEL_WEIGHT_FI", 1.0)
    mat = _env_float_channel("SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT", 4.0)
    return fi, mat


def continuous_max_sat_log() -> tuple[float, float]:
    """Per-channel log1p saturation after commit (freeze mature clot)."""
    cfg = resolve_config()
    if cfg is not None:
        return float(cfg.max_sat_log_fi), float(cfg.max_sat_log_mat)
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi = _env_float_channel("SPECIES_CONTINUOUS_MAX_SAT_LOG_FI", 0.002)
    mat = _env_float_channel("SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT", 0.002)
    return fi, mat


def continuous_state_scale(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return config.state_scale
    raw = (os.environ.get("SPECIES_CONTINUOUS_STATE_SCALE") or str(DEFAULT_STATE_SCALE)).strip()
    try:
        return max(float(raw), 1e-8)
    except ValueError:
        return 0.002


def continuous_mat_commit_thresh() -> float:
    """Binary readout threshold on accumulated log Mat (deploy)."""
    cfg = resolve_config()
    if cfg is not None and float(cfg.mat_commit_thresh) >= 0.0:
        return float(cfg.mat_commit_thresh)
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
        out.append(species_log_targets(data, int(t), device)[node_idx])
    return out


def band_speed_at_time(
    data,
    time_index: int,
    device: torch.device,
    node_idx: torch.Tensor,
    *,
    for_training: bool = False,
) -> torch.Tensor:
    """Normalized wall-band speed in [0, 1] (pred kine when deploy-faithful)."""
    from src.core_physics.species_deploy_rollout import band_speed_for_rollout

    return band_speed_for_rollout(
        data, time_index, device, node_idx, for_training=for_training
    )


def band_speed_series(
    data,
    time_indices: Sequence[int],
    device: torch.device,
    node_idx: torch.Tensor,
    *,
    for_training: bool = False,
) -> list[torch.Tensor]:
    return [
        band_speed_at_time(data, int(t), device, node_idx, for_training=for_training)
        for t in time_indices
    ]


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
    st = log_state0.reshape(-1, _sd()).clone()
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


def resolve_vel_decay_speed(
    wall_speed: torch.Tensor,
    wall_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-node speed used by vel-decay (optionally zeroed off-wall)."""
    spd = wall_speed.reshape(-1)
    if not continuous_vel_decay_wall_only():
        return spd
    if wall_mask is None:
        # Fail closed: never apply full-band washout when wall-only is requested.
        return torch.zeros_like(spd)
    m = wall_mask.reshape(-1).to(device=spd.device, dtype=spd.dtype)
    return spd * m


def apply_velocity_decay(
    log_state: torch.Tensor,
    wall_speed: torch.Tensor,
    alphas: tuple[torch.Tensor, torch.Tensor],
    *,
    wall_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    st = log_state.reshape(-1, _sd())
    spd = resolve_vel_decay_speed(wall_speed, wall_mask).reshape(-1, 1).to(
        device=st.device, dtype=st.dtype
    )
    a_fi, a_mat = alphas
    decay = torch.zeros_like(st)
    li_fi = _local_fi_idx()
    li_mat = _local_mat_idx()
    if li_fi is not None:
        decay[:, li_fi] = a_fi * spd.squeeze(-1) * st[:, li_fi]
    if li_mat is not None:
        decay[:, li_mat] = a_mat * spd.squeeze(-1) * st[:, li_mat]
    return (st - decay).clamp(min=0.0)


def log_delta_targets(log_prev: torch.Tensor, log_next: torch.Tensor) -> torch.Tensor:
    return (log_next.reshape(-1, _sd()) - log_prev.reshape(-1, _sd())).to(dtype=torch.float32)


def normalize_log_state(log_state: torch.Tensor) -> torch.Tensor:
    scale = continuous_state_scale()
    return (log_state.reshape(-1, _sd()) / scale).clamp(0.0, 1.0)


def saturation_ratio_features(log_state: torch.Tensor) -> torch.Tensor:
    """``X_t / X_max`` per channel in ``[0, 1]`` (local concentration inhibitor)."""
    st = log_state.reshape(-1, _sd())
    maxes = continuous_max_sat_log_vec(st.device, st.dtype)
    return (st / maxes.unsqueeze(0)).clamp(0.0, 1.0)


def build_continuous_step_features(
    base_feats: torch.Tensor,
    log_state: torch.Tensor,
    *,
    training: bool = True,
    time_index: int | None = None,
    velocity: torch.Tensor | None = None,
    pos_band: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
) -> torch.Tensor:
    state_norm = normalize_log_state(log_state)
    state_in = maybe_noise_log_state(state_norm, training=training)

    cfg = resolve_config()
    use_conv = bool(cfg.convection_aggr) if cfg is not None else (os.environ.get("SPECIES_CONVECTION_AGGR") == "1")
    if use_conv and velocity is not None and pos_band is not None and edge_index is not None:
        mat_idx = _ch_mat()
        if mat_idx is not None and 0 <= mat_idx < state_in.shape[1]:
            row, col = edge_index
            vel_from = velocity[row]  # [E, 2]
            dx = pos_band[col, 0] - pos_band[row, 0]
            dy = pos_band[col, 1] - pos_band[row, 1]
            alignment = vel_from[:, 0] * dx + vel_from[:, 1] * dy
            weight = F.relu(alignment)  # [E]
            
            mat_val = state_in[:, mat_idx]  # [n]
            mat_from = mat_val[row]  # [E]
            
            num_nodes = state_in.shape[0]
            accum = torch.zeros(num_nodes, device=state_in.device, dtype=state_in.dtype)
            accum.scatter_add_(0, col, weight * mat_from)
            
            sum_w = torch.zeros(num_nodes, device=state_in.device, dtype=state_in.dtype)
            sum_w.scatter_add_(0, col, weight)
            
            upwind_mat = accum / (sum_w + 1e-6)
            
            state_in = state_in.clone()
            if cfg is not None:
                alpha = float(cfg.convection_alpha)
            else:
                alpha = float(os.environ.get("SPECIES_CONVECTION_ALPHA", "0.5"))
            state_in[:, mat_idx] = (1 - alpha) * state_in[:, mat_idx] + alpha * upwind_mat

    feats = torch.cat([base_feats, state_in], dim=-1)
    if continuous_saturation_gate():
        feats = torch.cat([feats, saturation_ratio_features(log_state)], dim=-1)
    if continuous_time_context_enabled():
        ti = int(time_index) if time_index is not None else 0
        tctx = encode_continuous_time_features(ti, dtype=feats.dtype, device=feats.device)
        feats = torch.cat([feats, tctx.expand(feats.shape[0], -1)], dim=-1)
    return feats


def align_continuous_feature_dim(feats: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Pad/truncate trailing feature blocks to match model input width."""
    expected = int(getattr(model, "in_dim", feats.shape[1]))
    got = int(feats.shape[1])
    if got == expected:
        return feats
    if got > expected:
        return feats[:, :expected]
    pad = torch.zeros(feats.shape[0], expected - got, device=feats.device, dtype=feats.dtype)
    return torch.cat([feats, pad], dim=-1)


def apply_magnitude_headroom_clamp(
    magnitude: torch.Tensor,
    log_state: torch.Tensor,
    wall_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiable soft-clamp: crush delta when near species saturation ceiling."""
    st = log_state.reshape(-1, _sd())
    maxes = continuous_max_sat_log_vec(st.device, st.dtype)
    margin = maxes.unsqueeze(0) - st
    scale_wall = saturation_headroom_scale()
    scale_offwall = saturation_headroom_scale_offwall()

    if wall_mask is not None and scale_wall != scale_offwall:
        w_mask = wall_mask.reshape(-1, 1).bool()
        scale = torch.where(w_mask, scale_wall, scale_offwall)
    else:
        scale = scale_wall

    headroom = F.softplus(margin * scale) * continuous_delta_out_scale()
    return torch.minimum(magnitude.reshape(-1, _sd()), headroom)


def denormalize_log_state(norm_state: torch.Tensor) -> torch.Tensor:
    scale = continuous_state_scale()
    return norm_state.reshape(-1, _sd()) * scale


def soft_commit_log_state(
    log_state: torch.Tensor,
    *,
    straight_through: bool = False,
) -> torch.Tensor:
    """Freeze committed nodes at saturation (cannot un-clot)."""
    st = log_state.reshape(-1, _sd()).clone()
    thr = snapshot_active_log_nd()
    maxes = continuous_max_sat_log_vec(st.device, st.dtype)
    out = st.clone()
    for ch, bch in enumerate(pushforward_state_bulk_indices()):
        commit_thr = continuous_mat_commit_thresh() if bch == MAT_CHANNEL else thr
        committed = st[:, ch] > commit_thr
        sat_v = maxes[ch]
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
    st = log_state.reshape(-1, _sd())
    out = torch.zeros_like(st)
    for ch, bch in enumerate(pushforward_state_bulk_indices()):
        if bch == FI_CHANNEL:
            out[:, ch] = (st[:, ch] > fi_thr).float()
        elif bch == MAT_CHANNEL:
            out[:, ch] = (st[:, ch] > mat_thr).float()
    return out


def step_has_growth_supervision(tgt_delta: torch.Tensor, mask: torch.Tensor) -> bool:
    m = mask.reshape(-1).to(device=tgt_delta.device).bool()
    if not bool(m.any().item()):
        return False
    t = tgt_delta.reshape(-1, tgt_delta.shape[-1])[m]
    for ch, thr_v in enumerate(continuous_delta_threshold_vec()):
        if bool((t[:, ch] > thr_v).any().item()):
            return True
    return False


def continuous_delta_loss(
    pred_delta: torch.Tensor,
    tgt_delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float | None = None,
    channel_weight: tuple[float, float] | None = None,
    current_log_state: torch.Tensor | None = None,
    fp_weight_scale: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Delta step loss inside deployable ceiling ``mask`` (wall + hops, not GT clot)."""
    m = mask.reshape(-1).to(device=pred_delta.device).bool()
    if not bool(m.any().item()):
        return None
    if continuous_growth_only_loss():
        if not step_has_growth_supervision(tgt_delta, m):
            return None
        loss = _growth_huber()(
            pred_delta,
            tgt_delta,
            m,
            current_log_state=current_log_state,
            fp_weight_scale=fp_weight_scale,
        )
        return loss * continuous_loss_scale()
    b = continuous_huber_beta() if beta is None else float(beta)
    p = pred_delta[m]
    t = tgt_delta[m]
    weights = continuous_channel_weights_vec()
    losses = [
        w * F.huber_loss(p[:, ch], t[:, ch], delta=b, reduction="mean")
        for ch, w in enumerate(weights)
    ]
    wsum = max(sum(weights), 1e-6)
    return (sum(losses) / wsum) * continuous_loss_scale()


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
        return torch.zeros(n, _sd(), device=dev, dtype=torch.bool)
    thr_vec = continuous_delta_threshold_vec()
    grew = torch.zeros_like(log_series[0], dtype=torch.bool)
    for step in range(len(log_series) - 1):
        d = log_delta_targets(log_series[step], log_series[step + 1])
        for ch, thr_v in enumerate(thr_vec):
            grew[:, ch] = grew[:, ch] | (d[:, ch] > thr_v)
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
    wall_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    nxt = log_state.reshape(-1, _sd()) + pred_delta.reshape(-1, _sd())
    if wall_speed is not None and vel_decay_alphas is not None:
        nxt = apply_velocity_decay(
            nxt, wall_speed, vel_decay_alphas, wall_mask=wall_mask
        )
    nxt = nxt.clamp(min=0.0)
    return soft_commit_log_state(nxt, straight_through=straight_through)


def continuous_final_state_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.final_state_weight), 0.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT") or "0.35").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.25


def continuous_final_state_all_band(config: PushforwardConfig | None = None) -> bool:
    config = resolve_config(config)
    if config is not None:
        return bool(config.final_state_all_band)
    raw = (os.environ.get("SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_final_mass_penalty(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.final_mass_penalty), 0.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_FINAL_MASS_PENALTY") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_final_mass_target(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return float(config.final_mass_target)
    raw = (os.environ.get("SPECIES_CONTINUOUS_FINAL_MASS_TARGET") or "1.2").strip()
    try:
        return float(raw)
    except ValueError:
        return 1.2


def continuous_final_prec_fp_penalty(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.final_prec_fp_penalty), 0.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_FINAL_PREC_FP_PENALTY") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_step_mass_penalty(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.step_mass_penalty), 0.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_STEP_MASS_PENALTY") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_step_prec_fp_penalty(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.step_prec_fp_penalty), 0.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_STEP_PREC_FP_PENALTY") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_seed_aux_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.seed_aux_weight), 0.0)
    return 0.0


def continuous_seed_aux_early_steps(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return max(int(config.seed_aux_early_steps), 1)
    return 3


def continuous_seed_aux_compact_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.seed_aux_compact_weight), 0.0)
    return 0.0


def continuous_seed_aux_pos_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.seed_aux_pos_weight), 1.0)
    return 4.0


def continuous_pocket_contrast_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.pocket_contrast_weight), 0.0)
    return 0.0


def continuous_pocket_contrast_hops(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return max(int(config.pocket_contrast_hops), 0)
    return 4


def continuous_pocket_contrast_early_steps(config: PushforwardConfig | None = None) -> int:
    config = resolve_config(config)
    if config is not None:
        return max(int(config.pocket_contrast_early_steps), 1)
    return 8


def continuous_pocket_contrast_inside_weight(config: PushforwardConfig | None = None) -> float:
    config = resolve_config(config)
    if config is not None:
        return max(float(config.pocket_contrast_inside_weight), 0.0)
    return 0.0


def gt_first_seed_mat_mask(
    log_series: Sequence[torch.Tensor],
    *,
    early_steps: int,
    band_mask: torch.Tensor,
) -> torch.Tensor:
    """First-new GT Mat nodes within ``early_steps`` (multi-pocket seed set).

    Returns only the earliest timestep's newly active Mat -- the seed set for exclusive contrast.
    """
    li = _local_mat_idx()
    m = band_mask.reshape(-1).bool()
    if li is None or not log_series:
        return torch.zeros(int(m.numel()), dtype=torch.bool, device=m.device)
    active0 = log_state_to_active(log_series[0]).reshape(-1, log_series[0].shape[-1])[:, li].bool()
    k = min(max(int(early_steps), 1), max(len(log_series) - 1, 1))
    for t in range(1, k + 1):
        act = log_state_to_active(log_series[t]).reshape(-1, log_series[t].shape[-1])[:, li].bool()
        first_new = (act & ~active0) & m
        if bool(first_new.any().item()):
            return first_new.reshape(-1)
    return torch.zeros_like(active0)


def pocket_allowed_from_gt_seed(
    gt_seed: torch.Tensor,
    edge_index: torch.Tensor,
    hops: int,
) -> torch.Tensor:
    """k-hop dilation of GT first-seed (train labels only; never fed as a hard forward mask)."""
    from src.core_physics.clot_growth_masks import graph_dilate_hops

    return graph_dilate_hops(gt_seed.reshape(-1).bool(), edge_index, int(hops)).reshape(-1)


def soft_mat_commit_prob(log_state: torch.Tensor) -> torch.Tensor:
    """Differentiable soft Mat commit readout for pocket-contrast / seed aux."""
    li = _local_mat_idx()
    thr = continuous_mat_commit_thresh()
    st = log_state.reshape(-1, _sd())
    mat_col = st[:, li] if li is not None else st[:, 0]
    return torch.sigmoid((mat_col - thr) * 40.0)


def pocket_contrast_aux_loss(
    pred_soft: torch.Tensor,
    allowed: torch.Tensor,
    band_mask: torch.Tensor,
    *,
    weight: float = 1.0,
    inside_weight: float = 0.0,
) -> torch.Tensor:
    """Soft-penalize Mat probability outside the allowed GT-seed neighborhood.

    Does not hard-mask the forward. Inside the allowed pocket, growth is free unless
    ``inside_weight > 0`` (mild under-recall pressure).
    """
    w_out = max(float(weight), 0.0)
    w_in = max(float(inside_weight), 0.0)
    m = band_mask.reshape(-1).to(device=pred_soft.device).bool()
    allow = allowed.reshape(-1).to(device=pred_soft.device).bool()
    p = pred_soft.reshape(-1).clamp(0.0, 1.0)
    if not bool(m.any().item()) or (w_out <= 0.0 and w_in <= 0.0):
        return p.sum() * 0.0
    outside = m & ~allow
    loss = p.sum() * 0.0
    if w_out > 0.0 and bool(outside.any().item()):
        loss = loss + w_out * p[outside].mean()
    if w_in > 0.0 and bool((m & allow).any().item()):
        loss = loss + w_in * (1.0 - p[m & allow]).mean()
    return loss


def gt_early_mat_pocket(
    log_series: Sequence[torch.Tensor],
    *,
    early_steps: int,
    band_mask: torch.Tensor,
) -> torch.Tensor:
    """Binary GT early Mat pocket: newly active Mat within the first ``early_steps`` deltas.

    This is a *location* label (where commitment should start), not a mass target.
    """
    li = _local_mat_idx()
    m = band_mask.reshape(-1).bool()
    if li is None or not log_series:
        return torch.zeros(int(m.numel()), dtype=torch.bool, device=m.device)
    active0 = log_state_to_active(log_series[0]).reshape(-1, log_series[0].shape[-1])[:, li].bool()
    k = min(max(int(early_steps), 1), max(len(log_series) - 1, 1))
    pocket = torch.zeros_like(active0)
    for t in range(1, k + 1):
        act = log_state_to_active(log_series[t]).reshape(-1, log_series[t].shape[-1])[:, li].bool()
        pocket = pocket | (act & ~active0)
    return (pocket & m).reshape(-1)


def soft_seed_location_aux_loss(
    pred_soft: torch.Tensor,
    gt_pocket: torch.Tensor,
    band_mask: torch.Tensor,
    *,
    edge_index: torch.Tensor | None = None,
    pos_band: torch.Tensor | None = None,
    weight: float = 1.0,
    compact_weight: float = 0.0,
    pos_weight: float = 4.0,
) -> torch.Tensor:
    """BCE/focal-style location loss on soft early Mat probs vs GT early pocket.

    Intentionally *not* a mass ratio term: equal-mass wrong location still pays.
    Optional light compactness: edge smoothness of soft probs (connected pocket prior).
    """
    m = band_mask.reshape(-1).to(device=pred_soft.device).bool()
    w_loc = max(float(weight), 0.0)
    cw = max(float(compact_weight), 0.0)
    if not bool(m.any().item()) or (w_loc <= 0.0 and cw <= 0.0):
        return pred_soft.sum() * 0.0
    p = pred_soft.reshape(-1).clamp(1e-6, 1.0 - 1e-6)
    y = gt_pocket.reshape(-1).to(device=p.device, dtype=p.dtype)
    p_m = p[m]
    y_m = y[m]
    # Soft focal-ish: upweight false locations and missed GT pocket nodes.
    bce = F.binary_cross_entropy(p_m, y_m, reduction="none")
    pw = max(float(pos_weight), 1.0)
    w = torch.where(y_m > 0.5, torch.full_like(bce, pw), torch.ones_like(bce))
    # Extra FP location pressure (wrong place), not total mass.
    fp = (p_m * (1.0 - y_m)).detach()
    w = w * (1.0 + 2.0 * fp)
    loc = (bce * w).sum() / w.sum().clamp(min=1.0) if w_loc > 0.0 else p_m.sum() * 0.0

    compact = p_m.sum() * 0.0
    if cw > 0.0 and edge_index is not None and int(edge_index.numel()) > 0:
        row, col = edge_index[0], edge_index[1]
        # Smooth only where either endpoint is a soft seed candidate.
        edge_w = (p[row] + p[col]).clamp(min=0.0)
        if bool(edge_w.any().item()):
            compact = ((p[row] - p[col]) ** 2 * edge_w).sum() / edge_w.sum().clamp(min=1e-6)
    elif cw > 0.0 and pos_band is not None and bool((p_m > 0.05).any().item()):
        pos = pos_band.to(device=p.device, dtype=p.dtype).reshape(-1, pos_band.shape[-1])
        wpos = p.reshape(-1, 1) * m.reshape(-1, 1).to(dtype=p.dtype)
        mass = wpos.sum().clamp(min=1e-6)
        centroid = (pos * wpos).sum(dim=0) / mass
        compact = (((pos - centroid) ** 2) * wpos).sum() / mass

    return w_loc * loc + cw * compact


def rolled_final_mass_fp_penalty(
    final_pred: torch.Tensor,
    final_gt: torch.Tensor,
    band_mask: torch.Tensor,
    *,
    mass_weight: float | None = None,
    mass_target: float | None = None,
    fp_weight: float | None = None,
    soft_k: float = 40.0,
) -> torch.Tensor:
    """Differentiable soft mass-ratio + soft FP occupancy on rolled final Mat state.

    Cheap train–deploy alignment signal: penalize over-commit (mass_ratio > target) and
    predicted mass on GT-negative nodes. Uses sigmoid soft occupancy around the Mat commit
    threshold so gradients flow without a hard threshold.
    """
    mw = continuous_final_mass_penalty() if mass_weight is None else max(float(mass_weight), 0.0)
    fw = continuous_final_prec_fp_penalty() if fp_weight is None else max(float(fp_weight), 0.0)
    if mw <= 0.0 and fw <= 0.0:
        return final_pred.sum() * 0.0

    m = band_mask.reshape(-1).to(device=final_pred.device).bool()
    if not bool(m.any().item()):
        return final_pred.sum() * 0.0

    from src.training.biochem_species_scope import MAT_CHANNEL, pushforward_state_bulk_indices

    bulk = pushforward_state_bulk_indices()
    if MAT_CHANNEL not in bulk:
        return final_pred.sum() * 0.0
    mat_i = int(bulk.index(MAT_CHANNEL))
    thr = float(continuous_mat_commit_thresh())
    pred = final_pred.reshape(-1, final_pred.shape[-1])[m, mat_i]
    gt = final_gt.reshape(-1, final_gt.shape[-1])[m, mat_i]
    pred_soft = torch.sigmoid(float(soft_k) * (pred - thr))
    gt_pos = (gt > thr).to(dtype=pred_soft.dtype)
    n_gt = gt_pos.sum().clamp(min=1.0)
    mass_ratio = pred_soft.sum() / n_gt
    target = continuous_final_mass_target() if mass_target is None else float(mass_target)
    pen = final_pred.sum() * 0.0
    if mw > 0.0:
        pen = pen + mw * F.softplus(mass_ratio - target)
    if fw > 0.0:
        fp_soft = pred_soft * (1.0 - gt_pos)
        # Soft FP fraction of predicted occupancy (precision tilt without hard counts).
        fp_frac = fp_soft.sum() / pred_soft.sum().clamp(min=1e-6)
        pen = pen + fw * fp_frac
    return pen * continuous_loss_scale()


def continuous_rolled_soft_f1_weight() -> float:
    cfg = resolve_config()
    if cfg is not None:
        return max(float(cfg.rolled_soft_f1_weight), 0.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_ROLLED_SOFT_F1_WEIGHT") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_step_soft_f1_weight() -> float:
    cfg = resolve_config()
    if cfg is not None:
        return max(float(cfg.step_soft_f1_weight), 0.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_STEP_SOFT_F1_WEIGHT") or "0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def continuous_rolled_soft_f1_beta() -> float:
    cfg = resolve_config()
    if cfg is not None:
        return max(float(cfg.rolled_soft_f1_beta), 1e-3)
    raw = (os.environ.get("SPECIES_CONTINUOUS_ROLLED_SOFT_F1_BETA") or "1.0").strip()
    try:
        return max(float(raw), 1e-3)
    except ValueError:
        return 1.0


def continuous_rolled_soft_f1_k() -> float:
    cfg = resolve_config()
    if cfg is not None:
        return max(float(cfg.rolled_soft_f1_k), 1.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_ROLLED_SOFT_F1_K") or "40.0").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 40.0


def rolled_soft_f1_loss(
    final_pred: torch.Tensor,
    final_gt: torch.Tensor,
    band_mask: torch.Tensor,
    *,
    weight: float | None = None,
    beta: float | None = None,
    soft_k: float | None = None,
) -> torch.Tensor:
    """``weight * (1 - soft_F_beta)`` on the rolled Mat committed set (s12.5 change E).

    This is the deploy metric itself, softened -- the only term in the objective with a TP
    numerator, hence the only one that is monotone in F1 by construction.

    Why the existing rolled-state terms are not enough (s12.3):
      * ``final_mass_penalty`` = softplus(mass_ratio - target) is one-sided: identically 0
        below target, so it never pushes growth up, only down.
      * ``final_prec_fp_penalty`` = fp/(tp+fp) saturates toward 1.0 precisely in the
        fp=292 / mass=4.0 basin the model actually occupies, so its gradient vanishes where
        it is most needed.
    Soft occupancy is a sigmoid around the Mat commit threshold, matching
    ``rolled_final_mass_fp_penalty`` so the two terms see the same soft set.
    """
    w = continuous_rolled_soft_f1_weight() if weight is None else max(float(weight), 0.0)
    if w <= 0.0:
        return final_pred.sum() * 0.0

    m = band_mask.reshape(-1).to(device=final_pred.device).bool()
    if not bool(m.any().item()):
        return final_pred.sum() * 0.0

    from src.training.biochem_species_scope import MAT_CHANNEL, pushforward_state_bulk_indices

    bulk = pushforward_state_bulk_indices()
    if MAT_CHANNEL not in bulk:
        return final_pred.sum() * 0.0
    mat_i = int(bulk.index(MAT_CHANNEL))

    thr = float(continuous_mat_commit_thresh())
    k = continuous_rolled_soft_f1_k() if soft_k is None else max(float(soft_k), 1.0)
    b = continuous_rolled_soft_f1_beta() if beta is None else max(float(beta), 1e-3)

    pred = final_pred.reshape(-1, final_pred.shape[-1])[m, mat_i]
    gt = final_gt.reshape(-1, final_gt.shape[-1])[m, mat_i]
    p_soft = torch.sigmoid(k * (pred - thr))
    gt_pos = (gt > thr).to(dtype=p_soft.dtype)

    tp = (p_soft * gt_pos).sum()
    fp = (p_soft * (1.0 - gt_pos)).sum()
    fn = ((1.0 - p_soft) * gt_pos).sum()
    b2 = b * b
    denom = ((1.0 + b2) * tp + b2 * fn + fp).clamp(min=1e-6)
    soft_f = (1.0 + b2) * tp / denom
    return w * (1.0 - soft_f) * continuous_loss_scale()


def continuous_speed_fp_weight() -> float:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return max(float(rt.scoring.speed_fp_weight), 0.0)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_CONTINUOUS_SPEED_FP_WEIGHT") or "4.0").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def deploy_horizon_steps() -> int:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return max(int(rt.rollout.deploy_horizon), 0)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_CONTINUOUS_DEPLOY_HORIZON") or "0").strip()
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return 0


def deploy_eval_use_full_timeline() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.rollout.deploy_eval_full)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on", "full", "last")


def deploy_eval_time_index(n_times: int) -> int:
    """Deploy metric time index: per-graph last step unless explicitly capped."""
    last = graph_last_time_index(n_times)
    if deploy_eval_use_full_timeline():
        return last
    h = deploy_horizon_steps()
    if h > 0:
        return min(h, last)
    return last


def resolve_deploy_eval_time_index(n_times: int, *, time_index: int | None = None) -> int:
    """Resolve eval index from explicit override or deploy convention."""
    if time_index is not None:
        return max(0, min(int(time_index), graph_last_time_index(n_times)))
    return deploy_eval_time_index(n_times)


def default_deploy_metric_times(n_times: int) -> list[int]:
    """Standard deploy eval grid: t=0, mid probes, per-graph last."""
    last = graph_last_time_index(n_times)
    candidates = (0, 27, legacy_capped_deploy_time_index(n_times), last)
    return sorted({max(0, min(int(t), last)) for t in candidates})


def train_deploy_eval_flow_source() -> str:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            raw = str(rt.rollout.train_deploy_eval_flow or "auto").strip().lower()
        else:
            raw = (os.environ.get("SPECIES_TRAIN_DEPLOY_EVAL_FLOW") or "auto").strip().lower()
    except Exception:
        raw = (os.environ.get("SPECIES_TRAIN_DEPLOY_EVAL_FLOW") or "auto").strip().lower()
    if raw in ("gt", "comsol", "oracle"):
        return "gt"
    if raw in ("kine", "kinematics"):
        return "kinematics"
    return "auto"


def deploy_horizon_aux_all_packs() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.rollout.deploy_horizon_all_packs)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_DEPLOY_HORIZON_ALL_PACKS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def deploy_horizon_aux_cap_steps() -> int:
    """Cap TBPTT aux unroll length during training (VRAM); 0 = no cap."""
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return max(int(rt.rollout.deploy_horizon_aux_cap), 0)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_DEPLOY_HORIZON_AUX_CAP") or "72").strip()
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return 0


def deploy_eval_dual_times() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.scoring.deploy_eval_dual)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_CONTINUOUS_DEPLOY_EVAL_DUAL") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def deploy_eval_dual_full_weight() -> float:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return min(max(float(rt.scoring.deploy_dual_full_w), 0.0), 1.0)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_CONTINUOUS_DEPLOY_DUAL_FULL_W") or "0.65").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.65


def deploy_eval_time_fracs() -> list[float]:
    """Sliding-window fractions of the full horizon for deploy clot grading (s9.8).

    Parses ``deploy_eval_time_fracs`` ("0.5,0.75,1.0" style). Empty (default) disables this
    path entirely -- ``deploy_eval_clot_times`` falls back to the legacy dual/single behaviour,
    byte-identical to before this function existed.
    """
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        raw = str(rt.scoring.deploy_eval_time_fracs) if rt is not None else ""
        if rt is None:
            raw = os.environ.get("SPECIES_CONTINUOUS_DEPLOY_EVAL_TIME_FRACS") or ""
    except Exception:
        raw = os.environ.get("SPECIES_CONTINUOUS_DEPLOY_EVAL_TIME_FRACS") or ""
    raw = raw.strip()
    if not raw:
        return []
    out: list[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(min(max(float(tok), 0.0), 1.0))
        except ValueError:
            continue
    return out


def deploy_eval_clot_times(n_times: int) -> list[int]:
    """Time indices for deploy clot metric.

    Priority: explicit ``deploy_eval_time_fracs`` (sliding-window coverage of the whole
    horizon, s9.8 -- a checkpoint that only looks good at t_final can hide a rollout that
    has already gone badly wrong earlier) > legacy dual (mid+full) > single full. Unset
    ``deploy_eval_time_fracs`` (the default) preserves prior behaviour exactly.
    """
    last = deploy_eval_time_index(n_times)
    fracs = deploy_eval_time_fracs()
    if fracs:
        idxs = sorted({max(0, min(int(round(f * last)), last)) for f in fracs})
        return idxs or [last]
    if not deploy_eval_dual_times():
        return [last]
    mid = legacy_capped_deploy_time_index(n_times)
    if mid >= last:
        return [last]
    return [mid, last]


def continuous_delta_out_scale() -> float:
    cfg = resolve_config()
    if cfg is not None:
        return max(float(cfg.delta_out_scale), 1e-9)
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_OUT_SCALE") or "1e-5").strip()
    try:
        return max(float(raw), 1e-9)
    except ValueError:
        return 1e-5


def continuous_delta_softplus_beta() -> float:
    cfg = resolve_config()
    if cfg is not None:
        return max(float(cfg.delta_softplus_beta), 1.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_DELTA_SOFTPLUS_BETA") or "20").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 20.0


def continuous_autocatalytic_growth() -> bool:
    cfg = resolve_config()
    if cfg is not None:
        return bool(cfg.autocatalytic_growth)
    raw = (os.environ.get("SPECIES_CONTINUOUS_AUTOCAT_GROWTH") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def continuous_autocat_alpha() -> float:
    cfg = resolve_config()
    if cfg is not None:
        return min(max(float(cfg.autocat_alpha), 0.0), 1.0)
    raw = (os.environ.get("SPECIES_CONTINUOUS_AUTOCAT_ALPHA") or "0.8").strip()
    try:
        return min(max(float(raw), 0.0), 1.0)
    except ValueError:
        return 0.8


def continuous_autocat_inits() -> tuple[float, float]:
    cfg = resolve_config()
    if cfg is not None:
        return max(float(cfg.autocat_k_dep_init), 1e-6), max(float(cfg.autocat_k_auto_init), 1e-6)
    try:
        kd = max(float((os.environ.get("SPECIES_CONTINUOUS_AUTOCAT_K_DEP_INIT") or "1.0").strip()), 1e-6)
    except ValueError:
        kd = 1.0
    try:
        ka = max(float((os.environ.get("SPECIES_CONTINUOUS_AUTOCAT_K_AUTO_INIT") or "1.0").strip()), 1e-6)
    except ValueError:
        ka = 1.0
    return kd, ka


def delta_readout(raw_delta: torch.Tensor) -> torch.Tensor:
    """Non-negative log increment (species do not decrease on clot front)."""
    st = raw_delta.reshape(-1, _sd())
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

    def __init__(self, in_dim: int, *, hidden: int | None = None, out_dim: int | None = None, arch: str = "sage"):
        super().__init__(in_dim, hidden=hidden, out_dim=_sd() if out_dim is None else out_dim, arch=arch)
        self.log_vel_decay_fi = nn.Parameter(torch.tensor(-8.0))
        self.log_vel_decay_mat = nn.Parameter(torch.tensor(-8.0))


class SpeciesDualHeadContinuousGNN(SpeciesSnapshotGNN):
    """Phase 3.5: decoupled spatial gate * magnitude delta (FI, Mat)."""

    def __init__(self, in_dim: int, *, hidden: int | None = None, out_dim: int | None = None, arch: str = "sage"):
        super().__init__(in_dim, hidden=hidden, out_dim=_sd() if out_dim is None else out_dim, arch=arch)
        self.log_vel_decay_fi = nn.Parameter(torch.tensor(-8.0))
        self.log_vel_decay_mat = nn.Parameter(torch.tensor(-8.0))
        h = self.hidden
        fused = h + self.in_dim
        gate_in = fused + (1 if continuous_neighbor_commit_gate() else 0)
        od = self.out_dim
        self.spatial_head = nn.Sequential(
            nn.Linear(gate_in, h),
            nn.ReLU(),
            nn.Linear(h, od),
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(fused, h),
            nn.ReLU(),
            nn.Linear(h, od),
        )
        for head in (self.spatial_head, self.magnitude_head):
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.spatial_gate_heads = spatial_gate_heads_enabled()
        if self.spatial_gate_heads:
            self.spatial_head_wall = nn.Sequential(
                nn.Linear(gate_in, h),
                nn.ReLU(),
                nn.Linear(h, od),
            )
            self.magnitude_head_wall = nn.Sequential(
                nn.Linear(fused, h),
                nn.ReLU(),
                nn.Linear(h, od),
            )
            self.spatial_head_offwall = nn.Sequential(
                nn.Linear(gate_in, h),
                nn.ReLU(),
                nn.Linear(h, od),
            )
            self.magnitude_head_offwall = nn.Sequential(
                nn.Linear(fused, h),
                nn.ReLU(),
                nn.Linear(h, od),
            )
            for head in (self.spatial_head_wall, self.magnitude_head_wall, self.spatial_head_offwall, self.magnitude_head_offwall):
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
        # --- s11.3 change D: explicit gated-autocatalytic growth ---
        # COMSOL grows Mat as ~(Mas/Minf)*k_aa*AP -- rate proportional to material already
        # committed locally (~90% of real Mat growth, s11.3). The generic delta head has no
        # such term, so it can propagate but cannot ignite, which is what the two-basin
        # attractor in s12.4 looks like. Learnable, log-parameterised so both stay positive;
        # at k_dep=k_auto=1 an isolated node keeps its current rate and a fully-committed
        # neighbourhood doubles it, so the warm start is not disturbed at t=0.
        self.autocat_growth = continuous_autocatalytic_growth()
        self.log_k_dep: nn.Parameter | None = None
        self.log_k_auto: nn.Parameter | None = None
        if self.autocat_growth:
            kd, ka = continuous_autocat_inits()
            self.log_k_dep = nn.Parameter(torch.tensor(math.log(kd)))
            self.log_k_auto = nn.Parameter(torch.tensor(math.log(ka)))
        self.delta_residual: nn.Linear | None = None
        if continuous_delta_residual():
            self.delta_residual = nn.Linear(fused, out_dim)
            nn.init.zeros_(self.delta_residual.weight)
            nn.init.zeros_(self.delta_residual.bias)
        self.temporal_offset_gate: nn.Sequential | None = None
        if continuous_temporal_offset():
            self.temporal_offset_gate = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, out_dim),
            )
            for m in self.temporal_offset_gate.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            last = self.temporal_offset_gate[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _neighbor_commit_feature(self, log_state: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Mat commit-aware scalar prior for the spatial gate input."""
        st = log_state.reshape(-1, _sd())
        midx = _local_bulk_index(MAT_CHANNEL)
        if midx is None or midx < 0 or midx >= int(st.shape[1]):
            return torch.zeros((int(st.shape[0]), 1), device=st.device, dtype=st.dtype)
        mat_thr = continuous_mat_commit_thresh()
        committed = (st[:, midx] > mat_thr).to(dtype=st.dtype).unsqueeze(-1)
        alpha = continuous_neighbor_commit_alpha()
        neigh = _graph_blur_band_state(committed, edge_index, alpha)
        return neigh.clamp(min=0.0, max=1.0)

    def _frontier_nucleation_mask(
        self,
        spatial_logits: torch.Tensor,
        log_state: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Deployable hard mask: 1 where growth is allowed (k-hop frontier of predicted committed
        Mat, OR a top-k nucleation seed from the model's own gate logits). No GT is read.

        Returned detached (a non-differentiable structural mask); gradients still flow through the
        unmasked ``gate * magnitude`` at allowed nodes.
        """
        from src.core_physics.clot_growth_masks import graph_dilate_hops

        st = log_state.reshape(-1, _sd())
        n = int(st.shape[0])
        dev, dt = spatial_logits.device, spatial_logits.dtype
        midx = _local_bulk_index(MAT_CHANNEL)
        if midx is None or midx >= int(st.shape[1]):
            return torch.ones((n, 1), device=dev, dtype=dt)
        committed = (st[:, midx] > continuous_mat_commit_thresh()).reshape(-1).bool()
        allowed = graph_dilate_hops(committed, edge_index, continuous_frontier_hops()).to(device=dev)
        topk = continuous_nucleation_topk()
        if topk > 0.0 and not bool(committed.any().item()):
            logit_col = (
                spatial_logits[:, midx]
                if int(spatial_logits.shape[1]) > midx
                else spatial_logits.reshape(-1)
            ).detach().reshape(-1)
            k = min(max(int(math.ceil(topk * n)), 1), n)
            thr = torch.topk(logit_col, k).values.min()
            allowed = allowed | (logit_col >= thr)

        # Physics-inspired nucleation prior for Hop >= 2 nodes
        if physics_nucleation_enabled() and getattr(self, "velocity", None) is not None:
            u = self.velocity[:, 0].to(device=dev, dtype=dt)
            v = self.velocity[:, 1].to(device=dev, dtype=dt)
            speed = torch.sqrt(u * u + v * v)

            # BFS to compute hops from wall
            hops = torch.full((n,), -1, dtype=torch.long, device=dev)
            wall_m = getattr(self, "wall_mask_band", None)
            if wall_m is not None:
                wall_m = wall_m.to(device=dev).bool()
                hops[wall_m] = 0
                row, col = edge_index
                current_mask = wall_m.clone()
                current_hop = 0
                while True:
                    neighbor_mask = torch.zeros(n, dtype=torch.bool, device=dev)
                    neighbor_mask[col[current_mask[row]]] = True
                    next_mask = neighbor_mask & (hops == -1)
                    if not next_mask.any():
                        break
                    current_hop += 1
                    hops[next_mask] = current_hop
                    current_mask = next_mask
            hops[hops == -1] = 99

            # Compute shear proxy
            pos_b = getattr(self, "pos_band", None)
            if pos_b is not None:
                pos = pos_b.to(device=dev, dtype=dt)
                row, col = edge_index
                diff = pos[row] - pos[col]
                dist = diff.norm(dim=1).clamp(min=1e-6)
                grad = (speed[row] - speed[col]).abs() / dist
                deg = torch.zeros(n, device=dev, dtype=dt)
                deg.index_add_(0, row, torch.ones_like(grad))
                acc_g = torch.zeros(n, device=dev, dtype=dt)
                acc_g.index_add_(0, row, grad)
                shear_proxy = acc_g / deg.clamp(min=1.0)

                cfg_pn = resolve_config()
                if cfg_pn is not None:
                    speed_thresh = float(cfg_pn.physics_nuc_speed_thresh)
                    shear_thresh = float(cfg_pn.physics_nuc_shear_thresh)
                else:
                    speed_thresh = float(os.environ.get("SPECIES_PHYSICS_NUC_SPEED_THRESH", "0.15"))
                    shear_thresh = float(os.environ.get("SPECIES_PHYSICS_NUC_SHEAR_THRESH", "0.20"))

                stagnant = speed < speed_thresh
                low_shear = shear_proxy < shear_thresh
                off_wall = hops >= 2

                physics_eligible = stagnant & low_shear & off_wall
                allowed = allowed | physics_eligible

        return allowed.to(dtype=dt).unsqueeze(-1).detach()

    def _autocatalytic_factor(
        self, log_state: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor | None:
        """``k_dep + k_auto * local_committed_frac``, Mat column only (s11.3 change D).

        ``local_committed_frac`` is the graph-blurred committed-Mat indicator -- the same
        quantity ``_neighbor_commit_feature`` builds, but consumed here as a *multiplicative
        rate term* rather than as a gate input feature, which is what makes the law
        autocatalytic rather than merely commit-aware. Returns ``None`` when disabled.
        """
        if not self.autocat_growth or self.log_k_dep is None or self.log_k_auto is None:
            return None
        st = log_state.reshape(-1, _sd())
        midx = _local_bulk_index(MAT_CHANNEL)
        if midx is None or midx < 0 or midx >= int(st.shape[1]):
            return None
        committed = (st[:, midx] > continuous_mat_commit_thresh()).to(dtype=st.dtype).unsqueeze(-1)
        local = _graph_blur_band_state(committed, edge_index, continuous_autocat_alpha())
        local = local.clamp(min=0.0, max=1.0)
        k_dep = torch.exp(self.log_k_dep)
        k_auto = torch.exp(self.log_k_auto)
        return k_dep + k_auto * local

    def _apply_autocatalytic(
        self,
        pred_delta: torch.Tensor,
        log_state: torch.Tensor | None,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if log_state is None:
            return pred_delta
        fac = self._autocatalytic_factor(log_state, edge_index)
        if fac is None:
            return pred_delta
        mat_idx = _local_mat_idx()
        if mat_idx is None or mat_idx >= int(pred_delta.shape[1]):
            return pred_delta
        scale = torch.ones_like(pred_delta)
        scale[:, mat_idx] = fac.reshape(-1)
        return pred_delta * scale

    def temporal_lambda_from_state(self, log_state: torch.Tensor) -> torch.Tensor:
        """Scalar integration pace ``lambda in [lo, hi]`` from global band mass."""
        if self.temporal_gate is None:
            return torch.tensor(1.0, device=log_state.device, dtype=log_state.dtype)
        feat = global_species_mass_feature(log_state)
        raw = self.temporal_gate(feat).squeeze()
        lo, hi = temporal_lambda_bounds()
        return lo + torch.sigmoid(raw) * (hi - lo)

    def _get_sdf_band(self) -> torch.Tensor | None:
        if getattr(self, "pos_band", None) is not None and getattr(self, "wall_mask_band", None) is not None:
            wall_pos = self.pos_band[self.wall_mask_band.bool()]
            if wall_pos.numel() > 0:
                dists = torch.cdist(self.pos_band.unsqueeze(0), wall_pos.unsqueeze(0)).squeeze(0)
                sdf, _ = dists.min(dim=1)
                return sdf
        return None

    def forward_decoupled(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        log_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_orig = x
        h = self.forward_hidden(x, edge_index)
        h_fused = torch.cat([h, x_orig], dim=-1)
        spatial_in = h_fused
        if continuous_neighbor_commit_gate() and log_state is not None:
            spatial_in = torch.cat([h_fused, self._neighbor_commit_feature(log_state, edge_index)], dim=-1)

        if getattr(self, "spatial_gate_heads", False):
            # Wall Spatial and Magnitude
            spatial_logits_wall = self.spatial_head_wall(spatial_in)
            gate_temp = continuous_gate_temp()
            spatial_gate_wall = torch.sigmoid(spatial_logits_wall / gate_temp if gate_temp != 1.0 else spatial_logits_wall)
            if continuous_frontier_hops() > 0 and log_state is not None:
                spatial_gate_wall = spatial_gate_wall * self._frontier_nucleation_mask(
                    spatial_logits_wall, log_state, edge_index
                )
            cfg_df = resolve_config()
            use_df = bool(cfg_df.dynamic_frontier_mask) if cfg_df is not None else (os.environ.get("SPECIES_CONTINUOUS_DYNAMIC_FRONTIER_MASK") == "1")
            if use_df and log_state is not None:
                wall_m = getattr(self, "wall_mask_band", None)
                if wall_m is not None:
                    wall_m = wall_m.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
                    st = log_state.reshape(-1, _sd())
                    midx = _local_bulk_index(MAT_CHANNEL)
                    if midx is not None and 0 <= midx < int(st.shape[1]):
                        committed = (st[:, midx] > continuous_mat_commit_thresh()).reshape(-1).bool()
                        from src.core_physics.clot_growth_masks import graph_dilate_hops
                        neighbor = graph_dilate_hops(committed, edge_index, 1).to(device=x.device)
                        allowed = (wall_m.reshape(-1).bool() | neighbor).to(dtype=x.dtype).unsqueeze(-1)
                        spatial_gate_wall = spatial_gate_wall * allowed
            mag_raw_wall = self.magnitude_head_wall(h_fused)
            magnitude_wall = F.softplus(mag_raw_wall, beta=continuous_delta_softplus_beta()) * continuous_delta_out_scale()
            if continuous_saturation_gate() and log_state is not None:
                wall_m = getattr(self, "wall_mask_band", None)
                magnitude_wall = apply_magnitude_headroom_clamp(magnitude_wall, log_state, wall_mask=wall_m)
            pred_delta_wall = spatial_gate_wall * magnitude_wall

            # Off-wall Spatial and Magnitude
            spatial_logits_offwall = self.spatial_head_offwall(spatial_in)
            spatial_gate_offwall = torch.sigmoid(spatial_logits_offwall / gate_temp if gate_temp != 1.0 else spatial_logits_offwall)
            mag_raw_offwall = self.magnitude_head_offwall(h_fused)
            magnitude_offwall = F.softplus(mag_raw_offwall, beta=continuous_delta_softplus_beta()) * continuous_delta_out_scale()
            if continuous_saturation_gate() and log_state is not None:
                magnitude_offwall = apply_magnitude_headroom_clamp(magnitude_offwall, log_state, wall_mask=None)
            pred_delta_offwall = spatial_gate_offwall * magnitude_offwall

            sdf = self._get_sdf_band()
            if sdf is not None:
                cfg_gs = resolve_config()
                if cfg_gs is not None:
                    sdf_crit = float(cfg_gs.gate_sdf_crit)
                    sdf_temp = float(cfg_gs.gate_sdf_temp)
                else:
                    sdf_crit = float(os.environ.get("SPECIES_GATE_SDF_CRIT", "0.012"))
                    sdf_temp = float(os.environ.get("SPECIES_GATE_SDF_TEMP", "0.003"))
                gate = torch.sigmoid((sdf - sdf_crit) / max(sdf_temp, 1e-5)).unsqueeze(-1)
                pred_delta = gate * pred_delta_offwall + (1.0 - gate) * pred_delta_wall
                spatial_logits = spatial_logits_offwall
                magnitude = magnitude_offwall
            else:
                pred_delta = pred_delta_wall
                spatial_logits = spatial_logits_wall
                magnitude = magnitude_wall
        else:
            spatial_logits = self.spatial_head(spatial_in)
            gate_temp = continuous_gate_temp()
            spatial_gate = torch.sigmoid(spatial_logits / gate_temp if gate_temp != 1.0 else spatial_logits)
            if continuous_frontier_hops() > 0 and log_state is not None:
                spatial_gate = spatial_gate * self._frontier_nucleation_mask(
                    spatial_logits, log_state, edge_index
                )
            cfg_df = resolve_config()
            use_df = bool(cfg_df.dynamic_frontier_mask) if cfg_df is not None else (os.environ.get("SPECIES_CONTINUOUS_DYNAMIC_FRONTIER_MASK") == "1")
            if use_df and log_state is not None:
                wall_m = getattr(self, "wall_mask_band", None)
                if wall_m is not None:
                    wall_m = wall_m.reshape(-1, 1).to(device=x.device, dtype=x.dtype)
                    st = log_state.reshape(-1, _sd())
                    midx = _local_bulk_index(MAT_CHANNEL)
                    if midx is not None and 0 <= midx < int(st.shape[1]):
                        committed = (st[:, midx] > continuous_mat_commit_thresh()).reshape(-1).bool()
                        from src.core_physics.clot_growth_masks import graph_dilate_hops
                        neighbor = graph_dilate_hops(committed, edge_index, 1).to(device=x.device)
                        allowed = (wall_m.reshape(-1).bool() | neighbor).to(dtype=x.dtype).unsqueeze(-1)
                        spatial_gate = spatial_gate * allowed
            mag_raw = self.magnitude_head(h_fused)
            magnitude = F.softplus(mag_raw, beta=continuous_delta_softplus_beta()) * continuous_delta_out_scale()
            if continuous_saturation_gate() and log_state is not None:
                wall_m = getattr(self, "wall_mask_band", None)
                magnitude = apply_magnitude_headroom_clamp(magnitude, log_state, wall_mask=wall_m)
            if continuous_temporal_gate() and log_state is not None and self.temporal_gate is not None:
                lam = self.temporal_lambda_from_state(log_state)
                magnitude = magnitude * lam
            pred_delta = spatial_gate * magnitude

        # s11.3 change D: scale the Mat rate by locally-committed material before any
        # additive corrections, so the term is a true rate multiplier on the growth law.
        pred_delta = self._apply_autocatalytic(pred_delta, log_state, edge_index)

        if self.delta_residual is not None:
            alpha = continuous_delta_residual_alpha()
            pred_delta = pred_delta + alpha * self.delta_residual(h_fused)
        if self.temporal_offset_gate is not None and log_state is not None:
            off = self.temporal_offset_gate(global_species_mass_feature(log_state))
            pred_delta = pred_delta + off.squeeze(0) * (
                continuous_temporal_offset_scale() * continuous_delta_out_scale()
            )

        # Apply Skip-Hop GNN odd-hop node reconstruction
        if skip_hop_gnn_enabled() and getattr(self, "wall_mask_band", None) is not None:
            pred_delta = self._reconstruct_odd_nodes(pred_delta, edge_index)

        # Apply readout shear gate
        _cfg = resolve_config()
        use_shear = bool(_cfg.shear_readout_gate) if _cfg is not None else (os.environ.get("SPECIES_SHEAR_READOUT_GATE") == "1")
        if use_shear:
            ld = int(getattr(self, "kin_latent_dim", 0) or 0)
            if x.shape[1] > ld + 6:
                gamma_si = x[:, ld + 6].reshape(-1, 1)
                tau = getattr(self, "shear_gate_tau", None)
                lss = getattr(self, "shear_gate_lss", None)
                if tau is not None and lss is not None:
                    gate = torch.sigmoid((lss - gamma_si) / torch.clamp(tau, min=1e-3))
                    mat_idx = _local_mat_idx()
                    if mat_idx is not None and mat_idx < pred_delta.shape[1]:
                        mask = torch.ones_like(pred_delta)
                        mask[:, mat_idx] = gate.squeeze(-1)
                        pred_delta = pred_delta * mask

        # Apply frontier kinetics
        use_front_kin = bool(_cfg.frontier_kinetics) if _cfg is not None else (os.environ.get("SPECIES_FRONTIER_KINETICS") == "1")
        if use_front_kin:
            mat_idx = _local_mat_idx()
            pos_band = getattr(self, "pos_band", None)
            velocity = getattr(self, "velocity", None)
            species_block = getattr(self, "species_block", None)
            # Only apply on the band graph — skip when called with full-graph feats
            n_graph = pred_delta.size(0)
            band_size = pos_band.size(0) if pos_band is not None else -1
            if (mat_idx is not None and pos_band is not None and velocity is not None
                    and species_block is not None and log_state is not None
                    and n_graph == band_size
                    and species_block.size(0) == n_graph
                    and velocity.size(0) == n_graph):
                st = log_state.reshape(-1, _sd())
                committed = (st[:, mat_idx] > continuous_mat_commit_thresh()).reshape(-1).bool()
                from src.core_physics.clot_growth_masks import graph_dilate_hops
                frontier = graph_dilate_hops(committed, edge_index, 1).to(device=x.device) & ~committed
                from src.utils import species_channels as sc
                ap = torch.expm1(species_block[:, sc.block_index("AP")].reshape(-1))
                t_sp = torch.expm1(species_block[:, sc.block_index("T")].reshape(-1))
                u_vel = velocity[:, 0].reshape(-1)
                v_vel = velocity[:, 1].reshape(-1)
                from src.core_physics.species_snapshot_gnn import compute_frontier_fluxes
                flux_ap, flux_t = compute_frontier_fluxes(
                    pos_band, u_vel, v_vel, edge_index, committed, frontier, ap, t_sp
                )
                if _cfg is not None:
                    k_ap = float(_cfg.frontier_k_ap)
                    k_t = float(_cfg.frontier_k_t)
                else:
                    k_ap = float(os.environ.get("SPECIES_FRONTIER_K_AP", "0.5"))
                    k_t = float(os.environ.get("SPECIES_FRONTIER_K_T", "0.5"))
                K_kinetics = k_ap * flux_ap + k_t * flux_t
                mask = torch.zeros_like(pred_delta)
                mask[:, mat_idx] = K_kinetics
                pred_delta = pred_delta + mask

        return pred_delta, spatial_logits, magnitude

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        log_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_delta, _, _ = self.forward_decoupled(x, edge_index, log_state=log_state)
        return pred_delta


def build_continuous_gnn(in_dim: int, *, hidden: int | None = None, arch: str | None = None, config=None) -> nn.Module:
    from src.core_physics.species_gnode_pushforward import (
        SpeciesGnodeDualHeadContinuousGNN,
        species_pushforward_arch,
    )

    use_arch = (arch or species_pushforward_arch()).strip().lower()
    if use_arch == "gnode":
        if not continuous_dual_head():
            raise ValueError("gnode pushforward arch requires SPECIES_CONTINUOUS_DUAL_HEAD=1")
        return SpeciesGnodeDualHeadContinuousGNN(in_dim, hidden=hidden)
    if continuous_dual_head():
        return SpeciesDualHeadContinuousGNN(in_dim, hidden=hidden, arch=use_arch)
    return SpeciesContinuousPushforwardGNN(in_dim, hidden=hidden, arch=use_arch)


def bind_band_geometry(model: nn.Module, static: dict) -> None:
    if hasattr(model, "set_band_geometry"):
        model.set_band_geometry(
            static.get("pos_band"),
            static.get("edge_index"),
            static.get("wall_mask_band"),
            wall_normals=static.get("wall_normals_band"),
            sdf=static.get("sdf_band"),
            edge_attr=static.get("edge_attr_band"),
        )


def species_latent_dropout_p() -> float:
    """Latent leash probability: chance to zero the z_kin slice per training step.

    Breaks 'latent dominance' -- the species teacher otherwise leans 100% on the (clot-blind) DEQ
    latent and ignores the explicit flow features, capping the corrector's value (see the +0.008
    gt-flow ceiling). Forcing z_kin to vanish a fraction of the time makes the model learn backup
    weights on the flow channels. Training-only; 0 disables.
    """
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return min(max(float(rt.rollout.latent_dropout), 0.0), 0.95)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_LATENT_DROPOUT") or "0").strip()
    try:
        return min(max(float(raw), 0.0), 0.95)
    except ValueError:
        return 0.0


_SPLICE_SCRATCH: torch.Tensor | None = None
_SPLICE_SCRATCH_KEY: tuple | None = None


def splice_dynamic_flow(
    base_feats: torch.Tensor,
    flow_series: torch.Tensor | None,
    flow_cols: tuple[int, int] | None,
    time_index: int | None,
) -> torch.Tensor:
    """Replace the flow block of ``base_feats`` with the time-``time_index`` slice (Trap C).

    No-op unless dynamic flow is active (``flow_series``/``flow_cols`` present) and a time index is
    given. ``flow_cols = (start, width)`` is the flow block span produced by
    ``build_band_base_features``; ``flow_series`` is ``[n_times, n_band, width]``. Time is clamped.

    Under ``torch.no_grad()`` (eval/deploy), reuses a scratch buffer to avoid per-step allocs.
    With grads enabled, clones so autograd graphs stay intact across unroll steps.
    """
    if flow_series is None or flow_cols is None or time_index is None:
        return base_feats
    start, width = int(flow_cols[0]), int(flow_cols[1])
    if width <= 0 or int(base_feats.shape[1]) < start + width:
        return base_feats
    n_t = int(flow_series.shape[0])
    ti = max(0, min(int(time_index), n_t - 1))
    src = flow_series[ti].to(device=base_feats.device, dtype=base_feats.dtype)
    if torch.is_grad_enabled():
        out = base_feats.clone()
        out[:, start : start + width] = src
        return out
    global _SPLICE_SCRATCH, _SPLICE_SCRATCH_KEY
    key = (tuple(base_feats.shape), str(base_feats.device), base_feats.dtype)
    if _SPLICE_SCRATCH is None or _SPLICE_SCRATCH_KEY != key:
        _SPLICE_SCRATCH = base_feats.new_empty(base_feats.shape)
        _SPLICE_SCRATCH_KEY = key
    _SPLICE_SCRATCH.copy_(base_feats)
    _SPLICE_SCRATCH[:, start : start + width] = src
    return _SPLICE_SCRATCH

def maybe_drop_latent(base_feats: torch.Tensor, model: nn.Module, training: bool) -> torch.Tensor:
    """Stochastically zero the z_kin slice of ``base_feats`` (the latent leash). No-op at eval.

    Reads ``model.latent_dropout_p`` (prob) and ``model.kin_latent_dim`` (z_kin width = the first
    columns of ``base_feats``, ahead of sdf + flow features). Resampled per call so each unrolled
    step independently sees latent or not.
    """
    if not training:
        return base_feats
    p = float(getattr(model, "latent_dropout_p", 0.0) or 0.0)
    ld = int(getattr(model, "kin_latent_dim", 0) or 0)
    if p <= 0.0 or ld <= 0 or int(base_feats.shape[1]) < ld:
        return base_feats
    if torch.rand((), device=base_feats.device).item() >= p:
        return base_feats
    out = base_feats.clone()
    out[:, :ld] = 0.0
    return out


_OFFWALL_MODEL_CACHE = None
_OFFWALL_MODEL_CACHE_PATH: str | None = None


def clear_offwall_model_cache() -> None:
    """Drop cached growth specialist (needed between A/B evals with different ckpts)."""
    global _OFFWALL_MODEL_CACHE, _OFFWALL_MODEL_CACHE_PATH
    _OFFWALL_MODEL_CACHE = None
    _OFFWALL_MODEL_CACHE_PATH = None


def two_model_route() -> str:
    """Route for two-model ownership blending.

    ``wall`` keeps the canonical model on wall nodes and hands ``~wall`` to the
    growth specialist. ``frontier`` preserves the legacy behavior where the
    specialist owns the committed clot neighborhood, including committed wall
    nodes. ``frontier_offwall`` keeps wall ownership with the canonical model
    and only hands off-wall frontier nodes to the specialist.
    """
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None and rt.offwall.two_model_route:
            raw = str(rt.offwall.two_model_route).strip().lower()
        else:
            raw = (os.environ.get("SPECIES_TWO_MODEL_ROUTE") or "wall").strip().lower()
    except Exception:
        raw = (os.environ.get("SPECIES_TWO_MODEL_ROUTE") or "wall").strip().lower()
    if raw in ("frontier_offwall", "frontier-offwall", "frontier_lumen_only", "frontier-lumen-only"):
        return "frontier_offwall"
    if raw in ("frontier", "growth", "committed", "existing"):
        return "frontier"
    return "wall"


def two_model_frontier_hops() -> float:
    """BFS hops around committed Mat where the growth specialist owns the delta.

    Supports fractional hops in (0, 1): 0.5 = committed + off-wall 1-hop shell only
    (tighter than hops=1). Per-anchor overrides via SPECIES_TWO_MODEL_FRONTIER_HOPS_MAP
    (e.g. ``patient010:0.5,default:1``).
    """
    anchor = ""
    hop_map_raw = ""
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            anchor = str(rt.offwall.frontier_hops_anchor or "").strip()
            hop_map_raw = str(rt.offwall.frontier_hops_map or "")
    except Exception:
        pass
    if not anchor:
        anchor = (os.environ.get("SPECIES_TWO_MODEL_FRONTIER_HOPS_ANCHOR") or "").strip()
    if not hop_map_raw:
        hop_map_raw = os.environ.get("SPECIES_TWO_MODEL_FRONTIER_HOPS_MAP", "")
    if anchor:
        hop_map = parse_two_model_frontier_hops_map(hop_map_raw)
        if anchor in hop_map:
            return float(hop_map[anchor])
        if "default" in hop_map:
            return float(hop_map["default"])
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return max(float(rt.offwall.two_model_frontier_hops), 0.0)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_TWO_MODEL_FRONTIER_HOPS", "2") or "2").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 2.0


def parse_two_model_frontier_hops_map(raw: str) -> dict[str, float]:
    """Parse ``anchor:hops`` comma map (``default`` key optional)."""
    out: dict[str, float] = {}
    for part in (raw or "").split(","):
        piece = part.strip()
        if not piece or ":" not in piece:
            continue
        name, val = piece.split(":", 1)
        name = name.strip()
        if not name:
            continue
        try:
            out[name] = max(float(val.strip()), 0.0)
        except ValueError:
            continue
    return out


def _frontier_growth_zone(
    *,
    committed: torch.Tensor,
    edge_index: torch.Tensor,
    wall_mask: torch.Tensor,
    hops: float,
) -> torch.Tensor:
    """Nodes where the growth specialist owns the delta under frontier routing."""
    from src.core_physics.clot_growth_masks import graph_dilate_hops

    hops_f = max(float(hops), 0.0)
    if hops_f <= 0.0:
        return committed.reshape(-1).bool()
    if hops_f < 1.0:
        # Tighter than hops=1: committed + off-wall 1-hop shell (hops=0.5 equivalent).
        shell = graph_dilate_hops(committed, edge_index, 1).reshape(-1).bool()
        shell = shell & ~committed.reshape(-1).bool() & ~wall_mask.reshape(-1).bool()
        return (committed.reshape(-1).bool() | shell)
    hop_int = max(int(hops_f), 1)
    return graph_dilate_hops(committed, edge_index, hop_int).reshape(-1).bool()


def _frontier_offwall_growth_zone(
    *,
    committed: torch.Tensor,
    edge_index: torch.Tensor,
    wall_mask: torch.Tensor,
    hops: float,
) -> torch.Tensor:
    """Off-wall-only frontier for the growth specialist.

    Unlike legacy ``frontier``, this route never hands committed wall nodes to
    the growth specialist. Fractional hops keep the 1-hop shell semantics but
    remain off-wall only.
    """
    zone = _frontier_growth_zone(
        committed=committed,
        edge_index=edge_index,
        wall_mask=wall_mask,
        hops=hops,
    )
    return zone & ~wall_mask.reshape(-1).bool()


def get_cached_offwall_model(device, in_dim, hidden, out_dim):
    global _OFFWALL_MODEL_CACHE, _OFFWALL_MODEL_CACHE_PATH
    ckpt_path = ""
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None and rt.offwall.offwall_model_ckpt:
            ckpt_path = str(rt.offwall.offwall_model_ckpt).strip()
    except Exception:
        pass
    if not ckpt_path:
        ckpt_path = (os.environ.get("SPECIES_OFFWALL_MODEL_CKPT") or "").strip()
    if not ckpt_path:
        raise ValueError("SPECIES_OFFWALL_MODEL_CKPT environment variable must be set when SPECIES_TWO_MODEL_MODE=1")
    if _OFFWALL_MODEL_CACHE is not None and _OFFWALL_MODEL_CACHE_PATH == ckpt_path:
        return _OFFWALL_MODEL_CACHE
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    offwall_model = SpeciesDualHeadContinuousGNN(in_dim=in_dim, hidden=hidden, out_dim=out_dim).to(device)
    sd = payload.get("model_state") or payload.get("model_state_dict") or payload

    # Pre-validate shapes to raise helpful size mismatch errors
    model_sd = offwall_model.state_dict()
    for name, param in sd.items():
        if name in model_sd:
            if param.shape != model_sd[name].shape:
                raise ValueError(
                    f"Dimension mismatch for parameter '{name}' in off-wall GNN: "
                    f"checkpoint shape is {param.shape}, but model expects {model_sd[name].shape}. "
                    f"Please ensure the off-wall growth model was trained with the same environment "
                    f"settings (e.g. species scope, GNN features) as the main model."
                )

    offwall_model.load_state_dict(sd, strict=False)
    offwall_model.eval()
    _OFFWALL_MODEL_CACHE = offwall_model
    _OFFWALL_MODEL_CACHE_PATH = ckpt_path
    return _OFFWALL_MODEL_CACHE


def two_model_mode_enabled() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.offwall.two_model_mode)
    except Exception:
        pass
    return os.environ.get("SPECIES_TWO_MODEL_MODE") == "1"


def isolate_offwall_loss_enabled() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.offwall.isolate_offwall_loss)
    except Exception:
        pass
    return os.environ.get("SPECIES_ISOLATE_OFFWALL_LOSS") == "1"


def offwall_loss_scale() -> float:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return float(rt.offwall.offwall_loss_scale)
    except Exception:
        pass
    return float(os.environ.get("SPECIES_OFFWALL_LOSS_SCALE", "2.0"))


def spatial_gate_heads_enabled() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.offwall.spatial_gate_heads)
    except Exception:
        pass
    return os.environ.get("SPECIES_SPATIAL_GATE_HEADS") == "1"


def physics_nucleation_enabled() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.offwall.physics_nucleation)
    except Exception:
        pass
    return os.environ.get("SPECIES_PHYSICS_NUCLEATION") == "1"


def skip_hop_gnn_enabled() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.offwall.skip_hop_gnn)
    except Exception:
        pass
    return os.environ.get("SPECIES_SKIP_HOP_GNN") == "1"


def wall_mat_only_enabled() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.rollout.wall_mat_only)
    except Exception:
        pass
    return os.environ.get("CLOT_PHI_PHYSICS_WALL_MAT_ONLY") == "1"


def _two_model_blend_mask(
    *,
    route: str,
    wall_mask: torch.Tensor,
    log_state: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """True where the *wall/canonical* model keeps ownership; False -> growth specialist.

    - ``wall``: canonical on wall nodes (nucleation + wall paint); growth on ~wall.
    - ``frontier``: legacy growth ownership on the committed clot neighborhood
      (wall or lumen); canonical elsewhere.
    - ``frontier_offwall``: growth only on the off-wall frontier neighborhood;
      canonical always keeps wall ownership.
    """
    w_m = wall_mask.reshape(-1).bool()
    if route == "wall":
        return w_m

    from src.training.biochem_species_scope import pushforward_local_index

    st = log_state.reshape(-1, log_state.shape[-1])
    try:
        midx = int(pushforward_local_index("mat"))
    except (KeyError, ValueError):
        midx = 0 if st.shape[1] == 1 else min(1, st.shape[1] - 1)
    committed = (st[:, midx] > continuous_mat_commit_thresh()).reshape(-1).bool()
    if not bool(committed.any().item()):
        return torch.ones_like(w_m)
    hops = two_model_frontier_hops()
    if route == "frontier_offwall":
        growth_zone = _frontier_offwall_growth_zone(
            committed=committed,
            edge_index=edge_index,
            wall_mask=w_m,
            hops=hops,
        )
    else:
        growth_zone = _frontier_growth_zone(
            committed=committed,
            edge_index=edge_index,
            wall_mask=w_m,
            hops=hops,
        )
    # Canonical owns nodes outside the committed neighborhood (nucleation / idle lumen).
    return ~growth_zone.to(device=w_m.device)


def predict_continuous_step_delta(
    model: nn.Module,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    log_state: torch.Tensor,
    *,
    training: bool = False,
    pos_band: torch.Tensor | None = None,
    time_index: int | None = None,
    flow_series: torch.Tensor | None = None,
    flow_cols: tuple[int, int] | None = None,
    flow_time_index: int | None = None,
    wall_mask_band: torch.Tensor | None = None,
    species_block: torch.Tensor | None = None,
    velocity: torch.Tensor | None = None,
) -> torch.Tensor:
    """One closed-loop delta step (features + optional sat/temporal gates).

    ``time_index`` drives the (retired) temporal gate; ``flow_time_index`` selects the dynamic-flow
    snapshot (the current state's time) and falls back to ``time_index`` when not given.

    When ``SPECIES_TWO_MODEL_MODE=1``, blends the wall/canonical model with a growth specialist
    (``SPECIES_OFFWALL_MODEL_CKPT``) using ``SPECIES_TWO_MODEL_ROUTE`` (``wall`` or ``frontier``).
    """
    if hasattr(model, "set_band_geometry"):
        model.set_band_geometry(pos_band, edge_index, wall_mask_band)
    flow_ti = flow_time_index if flow_time_index is not None else time_index
    base_feats = splice_dynamic_flow(base_feats, flow_series, flow_cols, flow_ti)
    base_feats = maybe_drop_latent(base_feats, model, training)

    model.log_state = log_state
    if species_block is not None:
        model.species_block = species_block
    model.velocity = velocity

    feats = build_continuous_step_features(
        base_feats,
        log_state,
        training=training,
        time_index=time_index,
        velocity=velocity,
        pos_band=pos_band,
        edge_index=edge_index,
    )
    feats = align_continuous_feature_dim(feats, model)
    use_edge_index = getattr(model, "augmented_edge_index", None)
    cfg_lr = resolve_config()
    use_lr = bool(cfg_lr.longrange_edges) if cfg_lr is not None else (os.environ.get("SPECIES_LONGRANGE_EDGES") == "1")
    if use_edge_index is None or not use_lr:
        use_edge_index = edge_index

    # Helper function to forward model
    def _run_forward(m_obj, fts, e_idx, lst):
        if continuous_dual_head() and hasattr(m_obj, "forward_decoupled"):
            pd, _, _ = m_obj.forward_decoupled(fts, e_idx, log_state=lst)
            return pd
        return delta_readout(m_obj(fts, e_idx))

    pred_delta = _run_forward(model, feats, use_edge_index, log_state)

    w_mask = wall_mask_band if wall_mask_band is not None else getattr(model, "wall_mask_band", None)
    if two_model_mode_enabled() and w_mask is not None and log_state is not None:
        try:
            offwall_model = get_cached_offwall_model(
                feats.device, model.in_dim, model.hidden, model.out_dim
            )
            # Set offwall model state
            if hasattr(offwall_model, "set_band_geometry"):
                offwall_model.set_band_geometry(pos_band, edge_index, w_mask)
            offwall_model.log_state = log_state
            if species_block is not None:
                offwall_model.species_block = species_block
            offwall_model.velocity = velocity

            pred_delta_off = _run_forward(offwall_model, feats, use_edge_index, log_state)
            keep_wall = _two_model_blend_mask(
                route=two_model_route(),
                wall_mask=w_mask,
                log_state=log_state,
                edge_index=use_edge_index,
            ).reshape(-1, 1).to(device=feats.device)
            pred_delta = torch.where(keep_wall, pred_delta, pred_delta_off)
        except Exception as e:
            print(f"[WARN] Failed to apply two-model offwall blend: {e}")

    return pred_delta


def growth_delta_labels(tgt_delta: torch.Tensor) -> torch.Tensor:
    """Per-channel binary growth indicator from log-delta targets."""
    thr_vec = continuous_delta_threshold_vec()
    t = tgt_delta.reshape(-1, _sd())
    out = torch.zeros_like(t)
    for ch, thr_v in enumerate(thr_vec):
        out[:, ch] = (t[:, ch] > thr_v).float()
    return out


def dual_head_step_loss(
    spatial_logits: torch.Tensor,
    magnitude: torch.Tensor,
    tgt_delta: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    current_log_state: torch.Tensor | None = None,
    fp_weight_scale: torch.Tensor | None = None,
    hops: torch.Tensor | None = None,
) -> torch.Tensor | None:
    m = train_mask.reshape(-1).to(device=spatial_logits.device).bool()
    if not bool(m.any().item()):
        return None
    growth_tgt = growth_delta_labels(tgt_delta)

    alpha = pushforward_focal_alpha_vec()
    gamma = pushforward_focal_gamma_vec()
    ch_w = continuous_channel_weights_vec()
    gate_temp = continuous_gate_temp()
    gfw = continuous_gate_fp_weight()

    isolate_offwall = isolate_offwall_loss_enabled()
    if isolate_offwall and hops is not None:
        m_wall = m & (hops.to(device=m.device) <= 1)
        m_offwall = m & (hops.to(device=m.device) >= 2)

        # Helper to compute loss for a sub-mask
        def _sub_loss(sub_m):
            if not bool(sub_m.any().item()) or not step_has_growth_supervision(tgt_delta, sub_m):
                return None
            spatial_l = snapshot_loss(
                spatial_logits, tgt_delta, growth_tgt, sub_m,
                focal_alpha=alpha, focal_gamma=gamma, channel_weight=ch_w,
            )
            mag_l = _growth_huber()(
                magnitude, tgt_delta, sub_m,
                current_log_state=current_log_state, fp_weight_scale=fp_weight_scale,
            )
            sub_l = continuous_spatial_loss_weight() * spatial_l + mag_l
            if gfw > 0.0:
                logits_m = spatial_logits.reshape(-1, spatial_logits.shape[-1])[sub_m]
                growth_m = growth_tgt.reshape(-1, growth_tgt.shape[-1])[sub_m]
                gate_prob = torch.sigmoid(logits_m / gate_temp if gate_temp != 1.0 else logits_m)
                inactive = (growth_m <= 0.5).float()
                if bool(inactive.any().item()):
                    bce = F.binary_cross_entropy(gate_prob, growth_m, reduction="none")
                    if fp_weight_scale is not None:
                        scale_m = fp_weight_scale.reshape(-1)[sub_m].unsqueeze(-1)
                        bce = bce * scale_m
                    sub_l = sub_l + gfw * (bce * inactive).sum() / inactive.sum().clamp(min=1.0)
            return sub_l

        loss_wall = _sub_loss(m_wall)
        loss_offwall = _sub_loss(m_offwall)

        if loss_wall is not None and loss_offwall is not None:
            offwall_scale = offwall_loss_scale()
            return 0.5 * loss_wall + 0.5 * offwall_scale * loss_offwall
        elif loss_wall is not None:
            return loss_wall
        elif loss_offwall is not None:
            offwall_scale = offwall_loss_scale()
            return offwall_scale * loss_offwall
        else:
            return None

    if not step_has_growth_supervision(tgt_delta, m):
        return None

    spatial_l = snapshot_loss(
        spatial_logits,
        tgt_delta,
        growth_tgt,
        m,
        focal_alpha=alpha,
        focal_gamma=gamma,
        channel_weight=ch_w,
    )
    mag_l = _growth_huber()(
        magnitude,
        tgt_delta,
        m,
        current_log_state=current_log_state,
        fp_weight_scale=fp_weight_scale,
    )
    loss = continuous_spatial_loss_weight() * spatial_l + mag_l
    if gfw > 0.0:
        logits_m = spatial_logits.reshape(-1, spatial_logits.shape[-1])[m]
        growth_m = growth_tgt.reshape(-1, growth_tgt.shape[-1])[m]
        gate_prob = torch.sigmoid(logits_m / gate_temp if gate_temp != 1.0 else logits_m)
        inactive = (growth_m <= 0.5).float()
        if bool(inactive.any().item()):
            bce = F.binary_cross_entropy(gate_prob, growth_m, reduction="none")
            if fp_weight_scale is not None:
                scale_m = fp_weight_scale.reshape(-1)[m].unsqueeze(-1)
                bce = bce * scale_m
            loss = loss + gfw * (bce * inactive).sum() / inactive.sum().clamp(min=1.0)
    return loss


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
    dual: nn.Module,
    single: nn.Module,
    *,
    quiet: bool = False,
) -> None:
    """Warm-start dual heads from single-readout continuous / snapshot checkpoint."""
    sd = single.state_dict()
    dual_sd = dual.state_dict()
    copied = 0
    for key in list(dual_sd.keys()):
        if key in sd and dual_sd[key].shape == sd[key].shape:
            dual_sd[key] = sd[key].clone()
            copied += 1
    for prefix in ("spatial_head", "magnitude_head"):
        for suffix in (".0.weight", ".0.bias", ".2.weight", ".2.bias"):
            rk = f"readout{suffix}"
            pk = f"{prefix}{suffix}"
            if rk not in sd or pk not in dual_sd:
                continue
            w_src = sd[rk]
            w_dst = dual_sd[pk]
            if w_src.shape == w_dst.shape:
                dual_sd[pk] = w_src.clone()
                copied += 1
            elif suffix == ".0.weight" and w_src.ndim == 2 and w_dst.ndim == 2:
                h = min(w_src.shape[0], w_dst.shape[0])
                in_c = min(w_src.shape[1], w_dst.shape[1])
                dual_sd[pk][:, :in_c] = w_src[:, :in_c]
                copied += 1
            elif suffix.endswith(".bias") and w_src.shape == w_dst.shape[: w_src.ndim]:
                dual_sd[pk] = w_src.clone()
                copied += 1
    dual.load_state_dict(dual_sd, strict=False)
    if not quiet:
        print(f"[OK] dual-head partial warm-start ({copied} tensors)", flush=True)


def smooth_hop1_log_targets(log_series: list[torch.Tensor], edge_index: torch.Tensor, wall_mask_band: torch.Tensor) -> list[torch.Tensor]:
    if not log_series or edge_index is None or wall_mask_band is None:
        return log_series
    num_nodes = log_series[0].shape[0]
    device = edge_index.device
    hops = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    wall_mask = wall_mask_band.to(device=device).bool()
    hops[wall_mask] = 0
    row, col = edge_index
    current_mask = wall_mask.clone()
    current_hop = 0
    while True:
        neighbor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        neighbor_mask[col[current_mask[row]]] = True
        next_mask = neighbor_mask & (hops == -1)
        if not next_mask.any():
            break
        current_hop += 1
        hops[next_mask] = current_hop
        current_mask = next_mask
        if current_hop >= 2:
            break
    hop1_mask = (hops == 1)
    if not hop1_mask.any():
        return log_series
    source_mask = (hops == 0) | (hops == 2)
    edge_mask = source_mask[row] & hop1_mask[col]
    row_e = row[edge_mask]
    col_e = col[edge_mask]
    mat_idx = _ch_mat()
    if mat_idx is None or mat_idx < 0:
        return log_series
    cfg_h1 = resolve_config()
    alpha = float(cfg_h1.hop1_smooth_alpha) if cfg_h1 is not None else float(os.environ.get("SPECIES_HOP1_SMOOTH_ALPHA", "0.4"))
    smoothed_series = []
    for step_tensor in log_series:
        st = step_tensor.clone()
        mat_vals = st[:, mat_idx]
        accum = torch.zeros(num_nodes, device=device, dtype=st.dtype)
        accum.scatter_add_(0, col_e, mat_vals[row_e])
        counts = torch.zeros(num_nodes, device=device, dtype=st.dtype)
        counts.scatter_add_(0, col_e, torch.ones_like(col_e, dtype=st.dtype))
        avg_neigh = torch.zeros_like(mat_vals)
        avg_neigh[hop1_mask] = accum[hop1_mask] / (counts[hop1_mask] + 1e-6)
        st[hop1_mask, mat_idx] = alpha * avg_neigh[hop1_mask] + (1 - alpha) * mat_vals[hop1_mask]
        smoothed_series.append(st)
    return smoothed_series


def compute_hop_distances(
    edge_index: torch.Tensor,
    wall_mask_band: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """BFS to compute the exact hop distance from the wall for all band nodes."""
    dev = edge_index.device
    hops = torch.full((num_nodes,), -1, dtype=torch.long, device=dev)
    wall_m = wall_mask_band.to(device=dev).bool()
    hops[wall_m] = 0
    row, col = edge_index
    current_mask = wall_m.clone()
    current_hop = 0
    while True:
        neighbor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
        neighbor_mask[col[current_mask[row]]] = True
        next_mask = neighbor_mask & (hops == -1)
        if not next_mask.any():
            break
        current_hop += 1
        hops[next_mask] = current_hop
        current_mask = next_mask
    hops[hops == -1] = 99
    return hops


def unroll_continuous_loss(
    model: nn.Module,
    *,
    config=None,
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
    pos_band: torch.Tensor | None = None,
    time_window: Sequence[int] | None = None,
    flow_series: torch.Tensor | None = None,
    flow_cols: tuple[int, int] | None = None,
    wall_mask_band: torch.Tensor | None = None,
    wall_normals_band: torch.Tensor | None = None,
    sdf_band: torch.Tensor | None = None,
    edge_attr_band: torch.Tensor | None = None,
    species_block: torch.Tensor | None = None,
    velocity: torch.Tensor | None = None,
    epoch: int = 0,
    max_epochs: int = 100,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    cfg_h1b = resolve_config()
    use_h1 = bool(cfg_h1b.hop1_smooth) if cfg_h1b is not None else (os.environ.get("SPECIES_HOP1_SMOOTH") == "1")
    if use_h1 and wall_mask_band is not None and edge_index is not None:
        log_series = smooth_hop1_log_targets(log_series, edge_index, wall_mask_band)
    n_steps = len(log_series) - 1
    if n_steps <= 0:
        z = base_feats.sum() * 0.0
        return z, [], []

    bind_band_geometry(model, {
        "pos_band": pos_band,
        "edge_index": edge_index,
        "wall_mask_band": wall_mask_band,
        "wall_normals_band": wall_normals_band,
        "sdf_band": sdf_band,
        "edge_attr_band": edge_attr_band,
    })

    if wall_mat_only_enabled() and wall_mask_band is not None:
        train_mask = train_mask & wall_mask_band.bool()

    if log_state0 is None:
        log_state = torch.zeros(base_feats.shape[0], _sd(), device=base_feats.device, dtype=base_feats.dtype)
    else:
        log_state = noisy_teacher_log_state0(log_state0, edge_index, training=training)
    vel_alphas = model_vel_decay_alphas(model)
    tail = tbptt_tail if tbptt_tail is not None else tbptt_tail_steps()
    loss_start = max(0, n_steps - int(tail))
    seed_aux_w = continuous_seed_aux_weight()
    seed_aux_k = min(continuous_seed_aux_early_steps(), n_steps) if seed_aux_w > 0.0 else 0
    pc_w = continuous_pocket_contrast_weight()
    pc_k = min(continuous_pocket_contrast_early_steps(), n_steps) if pc_w > 0.0 else 0
    pc_allowed: torch.Tensor | None = None
    if pc_w > 0.0 and edge_index is not None and log_series:
        gt_seed = gt_first_seed_mat_mask(
            log_series, early_steps=max(pc_k, 1), band_mask=train_mask
        )
        pc_allowed = pocket_allowed_from_gt_seed(
            gt_seed, edge_index, continuous_pocket_contrast_hops()
        )
    step_w = step_loss_weights(n_steps)
    losses: list[torch.Tensor] = []
    loss_ws: list[float] = []
    pred_deltas: list[torch.Tensor] = []
    states: list[torch.Tensor] = [log_state.clone()]
    early_gate_probs: list[torch.Tensor] = []

    cfg_loss = resolve_config()
    if cfg_loss is not None:
        midside_blind = "1" if cfg_loss.midside_blind_loss else "0"
        phys_fp_gating = bool(cfg_loss.physical_fp_gating)
        sdf_fp_gating = bool(cfg_loss.sdf_fp_gating)
    else:
        midside_blind = os.environ.get("SPECIES_MIDSIDE_BLIND_LOSS")
        phys_fp_gating = os.environ.get("SPECIES_PHYSICAL_FP_GATING") == "1"
        sdf_fp_gating = os.environ.get("SPECIES_SDF_FP_GATING") == "1"
    isolate_offwall = isolate_offwall_loss_enabled()
    hops = None
    if (midside_blind is not None or phys_fp_gating or sdf_fp_gating or isolate_offwall) and wall_mask_band is not None and edge_index is not None:
        hops = compute_hop_distances(edge_index, wall_mask_band, base_feats.shape[0])

    for step in range(n_steps):
        seed_early = seed_aux_w > 0.0 and step < seed_aux_k
        pocket_live = pc_w > 0.0 and pc_allowed is not None and step < max(pc_k, 1)
        # TBPTT: keep grads on the tail; also keep early steps live for seed/pocket aux.
        grad_step = (not training) or step >= loss_start or seed_early or pocket_live
        ctx = torch.enable_grad() if grad_step else torch.no_grad()
        with ctx:
            # Detach only after early aux windows so pocket/seed retain a grad path.
            if step < loss_start and training and step >= seed_aux_k and not pocket_live:
                log_state = log_state.detach()
            # Trap C: splice the time-varying flow block for the CURRENT state's time, then leash.
            flow_ti = int(time_window[step]) if time_window is not None and step < len(time_window) else step
            step_base_feats = splice_dynamic_flow(base_feats, flow_series, flow_cols, flow_ti)
            step_base_feats = maybe_drop_latent(step_base_feats, model, training and grad_step)

            # Set stateful inputs on the model for readout gates / kinetics
            model.log_state = log_state
            if species_block is not None and step < len(species_block):
                model.species_block = species_block[step]
            else:
                model.species_block = log_series[step]
            if velocity is not None and step < len(velocity):
                model.velocity = velocity[step]
            else:
                model.velocity = None

            feats = build_continuous_step_features(
                step_base_feats,
                log_state,
                training=training and grad_step,
                time_index=(
                    int(time_window[step + 1])
                    if time_window is not None and step + 1 < len(time_window)
                    else step + 1
                ),
                velocity=velocity[step] if velocity is not None and step < len(velocity) else None,
                pos_band=pos_band,
                edge_index=edge_index,
            )
            feats = align_continuous_feature_dim(feats, model)
            tgt_delta = log_delta_targets(log_series[step], log_series[step + 1])
            gt_log = log_series[step]
            use_edge_index = getattr(model, "augmented_edge_index", None)
            cfg_lr = resolve_config()
            use_lr = bool(cfg_lr.longrange_edges) if cfg_lr is not None else (os.environ.get("SPECIES_LONGRANGE_EDGES") == "1")
            if use_edge_index is None or not use_lr:
                use_edge_index = edge_index

            # Midside-blind loss masking
            step_train_mask = train_mask
            if midside_blind is not None and hops is not None:
                step_train_mask = train_mask.clone()
                if midside_blind == "all_odd":
                    step_train_mask = step_train_mask & (hops % 2 == 0)
                else:
                    step_train_mask = step_train_mask & (hops != 1)

            # Physical FP Gating
            fp_weight_scale = None
            if phys_fp_gating and hops is not None and velocity is not None and step < len(velocity):
                vel_t = velocity[step]
                if vel_t is not None:
                    speed = vel_t.norm(dim=1)
                    if pos_band is not None:
                        pos = pos_band.to(device=edge_index.device, dtype=speed.dtype)
                        row, col = edge_index
                        diff = pos[row] - pos[col]
                        dist = diff.norm(dim=1).clamp(min=1e-6)
                        grad = (speed[row] - speed[col]).abs() / dist
                        deg = torch.zeros(base_feats.shape[0], device=edge_index.device, dtype=speed.dtype)
                        deg.index_add_(0, row, torch.ones_like(grad))
                        acc_g = torch.zeros(base_feats.shape[0], device=edge_index.device, dtype=speed.dtype)
                        acc_g.index_add_(0, row, grad)
                        shear = acc_g / deg.clamp(min=1.0)

                        cfg_pf = resolve_config()
                        if cfg_pf is not None:
                            s_crit = float(cfg_pf.physical_fp_speed_crit)
                            s_width = float(cfg_pf.physical_fp_speed_width)
                            g_crit = float(cfg_pf.physical_fp_shear_crit)
                            g_width = float(cfg_pf.physical_fp_shear_width)
                            w_min = float(cfg_pf.physical_fp_min_weight)
                        else:
                            s_crit = float(os.environ.get("SPECIES_PHYSICAL_FP_SPEED_CRIT", "0.05"))
                            s_width = float(os.environ.get("SPECIES_PHYSICAL_FP_SPEED_WIDTH", "0.01"))
                            g_crit = float(os.environ.get("SPECIES_PHYSICAL_FP_SHEAR_CRIT", "10.0"))
                            g_width = float(os.environ.get("SPECIES_PHYSICAL_FP_SHEAR_WIDTH", "2.0"))
                            w_min = float(os.environ.get("SPECIES_PHYSICAL_FP_MIN_WEIGHT", "0.1"))

                        s_val = torch.sigmoid((speed - s_crit) / max(s_width, 1e-4))
                        g_val = torch.sigmoid((shear - g_crit) / max(g_width, 1e-4))
                        fp_weight_scale = w_min + (1.0 - w_min) * torch.max(s_val, g_val)

            # SDF-Weighted FP Gating (Direction 4)
            cfg_sdf = resolve_config()
            use_sdf_fp = bool(cfg_sdf.sdf_fp_gating) if cfg_sdf is not None else (os.environ.get("SPECIES_SDF_FP_GATING") == "1")
            if use_sdf_fp and pos_band is not None and wall_mask_band is not None:
                try:
                    wall_pos = pos_band[wall_mask_band.bool()]
                    if wall_pos.numel() > 0:
                        dists = torch.cdist(pos_band.unsqueeze(0), wall_pos.unsqueeze(0)).squeeze(0)
                        sdf_val, _ = dists.min(dim=1)
                        cfg_sdf2 = resolve_config()
                        if cfg_sdf2 is not None:
                            decay_scale = float(cfg_sdf2.sdf_fp_decay_scale)
                            min_weight = float(cfg_sdf2.sdf_fp_min)
                        else:
                            decay_scale = float(os.environ.get("SPECIES_SDF_FP_DECAY_SCALE", "0.015"))
                            min_weight = float(os.environ.get("SPECIES_SDF_FP_MIN", "0.1"))
                        sdf_weight = torch.exp(-sdf_val / max(decay_scale, 1e-4))
                        sdf_weight = torch.clamp(sdf_weight, min=min_weight, max=1.0)
                        if fp_weight_scale is not None:
                            fp_weight_scale = fp_weight_scale * sdf_weight
                        else:
                            fp_weight_scale = sdf_weight
                except Exception as e:
                    print(f"[WARN] Failed to compute SDF-weighted FP gating: {e}")

            spatial_logits = None
            if continuous_dual_head() and hasattr(model, "forward_decoupled"):
                pred_delta, spatial_logits, magnitude = model.forward_decoupled(
                    feats, use_edge_index, log_state=log_state
                )
                step_loss = dual_head_step_loss(
                    spatial_logits,
                    magnitude,
                    tgt_delta,
                    step_train_mask,
                    current_log_state=gt_log,
                    fp_weight_scale=fp_weight_scale,
                    hops=hops,
                )
            else:
                pred_delta = delta_readout(model(feats, use_edge_index))
                step_loss = continuous_delta_loss(
                    pred_delta,
                    tgt_delta,
                    step_train_mask,
                    current_log_state=gt_log,
                    fp_weight_scale=fp_weight_scale,
                )
            pred_deltas.append(pred_delta)
            # Primary step losses only on the TBPTT tail (seed aux is separate).
            if step_loss is not None and grad_step and step >= loss_start:
                losses.append(step_loss)
                loss_ws.append(float(step_w[step]))
            if seed_early and spatial_logits is not None:
                li = _local_mat_idx()
                gate_temp = continuous_gate_temp()
                logits = spatial_logits.reshape(-1, spatial_logits.shape[-1])
                col = logits[:, li] if li is not None and li < int(logits.shape[1]) else logits[:, 0]
                early_gate_probs.append(
                    torch.sigmoid(col / gate_temp if gate_temp != 1.0 else col)
                )
            spd = None
            if speed_series is not None and step + 1 < len(speed_series):
                spd = speed_series[step + 1]
            log_state = pushforward_log_state_step(
                log_state,
                pred_delta,
                straight_through=training and grad_step,
                wall_speed=spd,
                vel_decay_alphas=vel_alphas,
                wall_mask=wall_mask_band,
            )
            states.append(log_state.clone())

            # After the early seed horizon: location aux vs GT early pocket (not mass).
            if seed_early and training and (step + 1) == seed_aux_k:
                if early_gate_probs:
                    stacked = torch.stack(early_gate_probs, dim=0)
                    pred_soft = stacked.max(dim=0).values
                else:
                    # Soft commit fallback when dual-head logits are unavailable.
                    li = _local_mat_idx()
                    thr = continuous_mat_commit_thresh()
                    st = log_state.reshape(-1, _sd())
                    mat_col = st[:, li] if li is not None else st[:, 0]
                    pred_soft = torch.sigmoid((mat_col - thr) * 40.0)
                gt_pocket = gt_early_mat_pocket(
                    log_series, early_steps=seed_aux_k, band_mask=train_mask
                )
                seed_l = soft_seed_location_aux_loss(
                    pred_soft,
                    gt_pocket,
                    train_mask,
                    edge_index=use_edge_index,
                    pos_band=pos_band,
                    weight=seed_aux_w,
                    compact_weight=continuous_seed_aux_compact_weight(),
                    pos_weight=continuous_seed_aux_pos_weight(),
                )
                losses.append(seed_l)
                loss_ws.append(1.0)

            # Exclusive wrong-pocket contrast: soft Mat outside k-hop of GT first-seed.
            # Train-only label; never hard-masks the forward (true pocket stays free to grow).
            if training and grad_step and pc_w > 0.0 and pc_allowed is not None:
                pred_soft_pc = soft_mat_commit_prob(log_state)
                pc_l = pocket_contrast_aux_loss(
                    pred_soft_pc,
                    pc_allowed,
                    step_train_mask,
                    weight=pc_w,
                    inside_weight=continuous_pocket_contrast_inside_weight(),
                )
                losses.append(pc_l)
                loss_ws.append(float(step_w[step]))

            # Per-step soft mass / FP vs GT occupancy (binds spray during TBPTT).
            if training and grad_step and step >= loss_start and step + 1 < len(log_series):
                smw = continuous_step_mass_penalty()
                spw = continuous_step_prec_fp_penalty()
                if smw > 0.0 or spw > 0.0:
                    mass_fp_step = rolled_final_mass_fp_penalty(
                        log_state,
                        log_series[step + 1],
                        step_train_mask,
                        mass_weight=smw,
                        fp_weight=spw,
                    )
                    losses.append(mass_fp_step)
                    loss_ws.append(float(step_w[step]))
                # Soft-F_beta on the rolled state at this step vs that step's GT (change E).
                ssf = continuous_step_soft_f1_weight()
                if ssf > 0.0:
                    losses.append(
                        rolled_soft_f1_loss(
                            log_state,
                            log_series[step + 1],
                            step_train_mask,
                            weight=ssf,
                        )
                    )
                    loss_ws.append(float(step_w[step]))

            # --- Scheduled sampling: intermittent noisy-GT anchoring ---
            # With probability (1 - keep_prob), reset log_state to (noisy) GT
            # at anchor-stride boundaries. Stabilises long-window training
            # while converging to fully autoregressive by late epochs.
            if (
                training
                and scheduled_sampling_enabled()
                and step + 1 < n_steps
                and (step + 1) % _ss_anchor_stride() == 0
            ):
                keep_p = scheduled_sampling_keep_prob(epoch, max_epochs)
                if keep_p < 1.0 and random.random() > keep_p:
                    gt_anchor = log_series[step + 1].detach()
                    if _ss_noisy():
                        gt_anchor = noisy_teacher_log_state0(
                            gt_anchor, edge_index, training=True,
                        )
                    log_state = gt_anchor

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
    if states:
        m = train_mask.reshape(-1).bool()
        if bool(m.any().item()):
            if fw > 0.0:
                if continuous_growth_only_loss() and not continuous_final_state_all_band():
                    final_loss = growth_only_final_state_loss(states[-1], log_series, m)
                else:
                    beta = continuous_huber_beta()
                    final_tgt = log_series[-1][m]
                    final_pred = states[-1][m]
                    final_loss = F.huber_loss(final_pred, final_tgt, delta=beta, reduction="mean")
                    final_loss = final_loss * continuous_loss_scale()
                step_loss = step_loss + fw * final_loss
            # Soft mass / FP on the rolled final state (train–deploy alignment).
            mass_fp = rolled_final_mass_fp_penalty(states[-1], log_series[-1], m)
            step_loss = step_loss + mass_fp
            # Soft-F_beta on the rolled final state: the deploy metric itself, softened.
            # This is the term v6 proved the objective was missing (s12.3 change E).
            step_loss = step_loss + rolled_soft_f1_loss(states[-1], log_series[-1], m)

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
    time_window: Sequence[int] | None = None,
    flow_series: torch.Tensor | None = None,
    flow_cols: tuple[int, int] | None = None,
    wall_mask_band: torch.Tensor | None = None,
    species_block: torch.Tensor | None = None,
    velocity: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Returns log states, pred deltas, and binary readout states per step."""
    model.eval()
    n_steps = len(log_series) - 1
    if log_state0 is None:
        log_state = torch.zeros(base_feats.shape[0], _sd(), device=base_feats.device, dtype=base_feats.dtype)
    else:
        log_state = log_state0.clone()
    vel_alphas = model_vel_decay_alphas(model)
    log_states = [log_state.clone()]
    deltas: list[torch.Tensor] = []
    actives: list[torch.Tensor] = [log_state_to_active(log_state)]
    for step in range(n_steps):
        pred_delta = predict_continuous_step_delta(
            model,
            base_feats,
            edge_index,
            log_state,
            training=False,
            time_index=(
                int(time_window[step + 1])
                if time_window is not None and step + 1 < len(time_window)
                else step + 1
            ),
            flow_series=flow_series,
            flow_cols=flow_cols,
            flow_time_index=(
                int(time_window[step]) if time_window is not None and step < len(time_window) else step
            ),
            wall_mask_band=wall_mask_band,
            species_block=species_block[step] if species_block is not None and step < len(species_block) else log_series[step],
            velocity=velocity[step] if velocity is not None and step < len(velocity) else None,
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
            wall_mask=wall_mask_band,
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
    bind_band_geometry(model, static)
    node_idx = static["node_idx"]
    t_end = max(0, int(time_index))
    from src.core_physics.species_deploy_rollout import deploy_fimat_log_init

    log_state = deploy_fimat_log_init(data, device, node_idx)
    vel_alphas = model_vel_decay_alphas(model)
    pos_band = static.get("pos_band")
    wall_m = static.get("wall_mask_band")
    from src.core_physics.species_deploy_rollout import resolve_species_rollout_uv
    with torch.no_grad():
        for t in range(t_end):
            spd = band_speed_at_time(data, t + 1, device, node_idx, for_training=True)
            u, v = resolve_species_rollout_uv(data, t + 1, device, for_training=True)
            vel_val = torch.stack([u[node_idx], v[node_idx]], dim=1)
            pred_delta = predict_continuous_step_delta(
                model,
                static["base_feats"],
                static["edge_index"],
                log_state,
                training=False,
                pos_band=pos_band,
                time_index=t + 1,
                velocity=vel_val,
            )
            log_state = pushforward_log_state_step(
                log_state,
                pred_delta,
                straight_through=False,
                wall_speed=spd,
                vel_decay_alphas=vel_alphas,
                wall_mask=wall_m,
            )
    if was_training:
        model.train()
    return log_state.detach()


def mat_front_diagnostics(
    pred_actives: Sequence[torch.Tensor],
    gt_actives: Sequence[torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Decompose Mat rollout errors into sparse-seed, front-speed, and overpaint terms.

    This is intentionally eval-only diagnostics: GT active states are used only as labels after the
    model has produced its closed-loop predicted states. No value from this helper feeds the model.
    """
    li_mat = _local_mat_idx()
    if li_mat is None or not pred_actives or not gt_actives:
        return {
            "mat_seed_prec": 0.0,
            "mat_seed_count": 0.0,
            "mat_front_prec": 0.0,
            "mat_front_speed_ratio": 0.0,
            "mat_overpaint_frac": 0.0,
            "mat_overpaint_per_gt": 0.0,
        }
    n = min(len(pred_actives), len(gt_actives))
    pred_m = [a.reshape(-1, a.shape[-1])[:, li_mat].bool() for a in pred_actives[:n]]
    gt_m = [a.reshape(-1, a.shape[-1])[:, li_mat].bool() for a in gt_actives[:n]]
    band = (
        torch.ones_like(pred_m[-1], dtype=torch.bool)
        if mask is None
        else mask.reshape(-1).to(device=pred_m[-1].device).bool()
    )
    pred_final = pred_m[-1] & band
    gt_final = gt_m[-1] & band
    fp_final = pred_final & ~gt_final
    overpaint_frac = float(fp_final.sum().item()) / max(float(band.sum().item()), 1.0)
    overpaint_per_gt = float(fp_final.sum().item()) / max(float(gt_final.sum().item()), 1.0)

    first_new: torch.Tensor | None = None
    first_new_count = 0.0
    front_tp = front_pred = pred_new_tot = gt_new_tot = 0.0
    for i in range(1, n):
        pred_new = (pred_m[i] & ~pred_m[i - 1]) & band
        gt_new = (gt_m[i] & ~gt_m[i - 1]) & band
        if first_new is None and bool(pred_new.any().item()):
            first_new = pred_new
            first_new_count = float(pred_new.sum().item())
        front_tp += float((pred_new & gt_final).sum().item())
        front_pred += float(pred_new.sum().item())
        pred_new_tot += float(pred_new.sum().item())
        gt_new_tot += float(gt_new.sum().item())
    if first_new is not None:
        seed_prec = float((first_new & gt_final).sum().item()) / max(first_new_count, 1e-6)
    else:
        seed_prec = 0.0
    return {
        "mat_seed_prec": float(seed_prec),
        "mat_seed_count": float(first_new_count),
        "mat_front_prec": float(front_tp / max(front_pred, 1e-6)),
        "mat_front_speed_ratio": float(pred_new_tot / max(gt_new_tot, 1e-6)),
        "mat_overpaint_frac": float(overpaint_frac),
        "mat_overpaint_per_gt": float(overpaint_per_gt),
    }


@torch.no_grad()
def eval_full_rollout_fimat_f1(
    model: nn.Module,
    data,
    static: dict,
    device: torch.device,
    *,
    time_index: int | None = None,
) -> dict[str, float]:
    """Closed-loop deploy Mat/FI F1 at ``time_index`` (full timeline from t=0)."""
    model.eval()
    bind_band_geometry(model, static)
    node_idx = static["node_idx"]
    n_times = int(data.y.shape[0])
    t_eval = resolve_deploy_eval_time_index(n_times, time_index=time_index)
    from src.core_physics.species_deploy_rollout import deploy_fimat_log_init

    log_state = deploy_fimat_log_init(data, device, node_idx)
    vel_alphas = model_vel_decay_alphas(model)
    pos_band = static.get("pos_band")
    wall_m = static.get("wall_mask_band")
    pred_actives = [log_state_to_active(log_state)]
    gt_actives = [log_state_to_active(species_log_targets(data, 0, device)[node_idx])]
    from src.core_physics.species_deploy_rollout import resolve_species_rollout_uv
    for t in range(t_eval):
        spd = band_speed_at_time(data, t + 1, device, node_idx)
        u, v = resolve_species_rollout_uv(data, t + 1, device, for_training=False)
        vel_val = torch.stack([u[node_idx], v[node_idx]], dim=1)
        pred_delta = predict_continuous_step_delta(
            model,
            static["base_feats"],
            static["edge_index"],
            log_state,
            training=False,
            pos_band=pos_band,
            time_index=t + 1,
            velocity=vel_val,
        )
        log_state = pushforward_log_state_step(
            log_state,
            pred_delta,
            straight_through=False,
            wall_speed=spd,
            vel_decay_alphas=vel_alphas,
            wall_mask=wall_m,
        )
        pred_actives.append(log_state_to_active(log_state))
        gt_actives.append(log_state_to_active(species_log_targets(data, min(t + 1, n_times - 1), device)[node_idx]))
    gt_log = species_log_targets(data, t_eval, device)[node_idx]
    gt_active = log_state_to_active(gt_log)
    pred_active = log_state_to_active(log_state)
    band_m = torch.ones(log_state.shape[0], dtype=torch.bool, device=device)
    sm = trigger_metrics(pred_active, gt_active, band_m)
    out = {
        "deploy_fi_f1": float(sm["fi_f1"]),
        "deploy_mat_f1": float(sm["mat_f1"]),
        "deploy_trigger_f1": float(sm["trigger_f1"]),
        "time_index": int(t_eval),
    }
    out.update(mat_front_diagnostics(pred_actives, gt_actives, mask=band_m))
    return out


@torch.no_grad()
@torch.no_grad()
def deploy_species_rollout_series(
    model: nn.Module,
    data,
    static: dict,
    phys_cfg,
    bio_cfg,
    device: torch.device,
    *,
    flow_source: str = "gt",
    gelation_beta: float | None = None,
) -> tuple[torch.Tensor, Any]:
    """Closed-loop deploy species rollout -> ``(species_series, data)``.

    Extracted from :func:`eval_deploy_clot_f1` so offline diagnostics can re-grade the
    *same* rollout at several readout gains without re-running the GNN. ``data`` comes
    back because closed-loop coupling clones it and writes diverted UV into ``data.y``.
    """
    import os

    gel_beta = gelation_beta
    model.eval()
    bind_band_geometry(model, static)
    # Isolate training packs: closed-loop coupling writes diverted UV into data.y in-place.
    if (os.environ.get("SPECIES_CLOSED_LOOP_COUPLING") == "1" or flow_source == "auto") and hasattr(data, "clone"):
        data = data.clone()
    node_idx = static["node_idx"]
    n_times = int(data.y.shape[0])
    from src.core_physics.species_deploy_rollout import (
        alloc_species_y_series,
        deploy_fimat_log_init,
        pin_species_block,
    )

    out = alloc_species_y_series(data, device)
    log_state = deploy_fimat_log_init(data, device, node_idx)
    vel_alphas = model_vel_decay_alphas(model)
    pos_band = static.get("pos_band")
    wall_m = static.get("wall_mask_band")

    coupler = None
    mu_bulk_si = None
    if os.environ.get("SPECIES_CLOSED_LOOP_COUPLING") == "1" or flow_source == "auto":
        try:
            flow_device = device
            from src.inference.corrector_coupling import ClotAwareFlow, resolve_kinematics_checkpoint, resolve_corrector_checkpoint, reset_coupled_flow_registry
            from src.core_physics.coupled_shear_gnn import LocalKinematicCorrector
            from src.utils.kinematics_inference import load_kinematics_predictor
            from src.core_physics.clot_growth_masks import resolve_bulk_carreau_mu_si
            reset_coupled_flow_registry()
            
            global _CLOSED_LOOP_MODELS_CACHE
            if "_CLOSED_LOOP_MODELS_CACHE" not in globals():
                _CLOSED_LOOP_MODELS_CACHE = {}
                
            kine_ckpt = resolve_kinematics_checkpoint()
            corr_ckpt = resolve_corrector_checkpoint()
            
            if os.environ.get("BIOCHEM_KINE_RESOLVE_ON_CLOT") == "1":
                cache_key = (kine_ckpt, corr_ckpt, str(flow_device))
                if cache_key in _CLOSED_LOOP_MODELS_CACHE:
                    kine, corr_model = _CLOSED_LOOP_MODELS_CACHE[cache_key]
                else:
                    kine = load_kinematics_predictor(kine_ckpt, flow_device)
                    kine.eval()
                    for p in kine.parameters():
                        p.requires_grad = False
                    from src.core_physics.coupled_shear_gnn import load_local_corrector
                    corr_model = load_local_corrector(corr_ckpt, flow_device)
                    corr_model.eval()
                    for p in corr_model.parameters():
                        p.requires_grad = False
                    _CLOSED_LOOP_MODELS_CACHE[cache_key] = (kine, corr_model)
            else:
                kine = None
                corr_cache_key = (corr_ckpt, str(flow_device))
                if corr_cache_key in _CLOSED_LOOP_MODELS_CACHE:
                    corr_model = _CLOSED_LOOP_MODELS_CACHE[corr_cache_key]
                else:
                    from src.core_physics.coupled_shear_gnn import load_local_corrector
                    corr_model = load_local_corrector(corr_ckpt, flow_device)
                    corr_model.eval()
                    for p in corr_model.parameters():
                        p.requires_grad = False
                    _CLOSED_LOOP_MODELS_CACHE[corr_cache_key] = corr_model
                
            coupler = ClotAwareFlow(flow_device, phys_cfg=phys_cfg)
            coupler._kine = kine
            coupler._corrector = corr_model
            u0, v0 = coupler.base_flow(data)
            mu_bulk_si = resolve_bulk_carreau_mu_si(data, 0, phys_cfg, flow_device, u_nd=u0, v_nd=v0).reshape(-1)
        except Exception as e:
            print(f"[WARN] Failed to initialize closed-loop flow coupler in deploy rollout: {e}")

    for t in range(n_times):
        sp = pin_species_block(data, t, device)
        sp = scatter_log_state_to_species_block(sp, log_state, node_idx)
        out[t, :, sc.SPECIES_BLOCK] = sp.clamp(min=0.0)
        if t >= n_times - 1:
            break
        from src.core_physics.species_deploy_rollout import resolve_species_rollout_uv
        if coupler is not None:
            from src.inference.corrector_coupling import get_coupled_flow
            coupled = get_coupled_flow(data, device)
            if coupled is not None:
                u, v = coupled
            else:
                u, v = resolve_species_rollout_uv(data, t + 1, device, for_training=False)
        else:
            u, v = resolve_species_rollout_uv(data, t + 1, device, for_training=False)
            
        from src.core_physics.species_deploy_rollout import band_speed_from_uv
        spd = band_speed_from_uv(u, v, node_idx)
        vel_val = torch.stack([u[node_idx], v[node_idx]], dim=1)
        pred_delta = predict_continuous_step_delta(
            model,
            static["base_feats"],
            static["edge_index"],
            log_state,
            training=False,
            pos_band=pos_band,
            time_index=t + 1,
            velocity=vel_val,
        )
        log_state = pushforward_log_state_step(
            log_state,
            pred_delta,
            straight_through=False,
            wall_speed=spd,
            vel_decay_alphas=vel_alphas,
            wall_mask=wall_m,
        )

        if coupler is not None and t + 1 < n_times:
            try:
                from src.core_physics.species_gelation_readout import differentiable_clot_phi_from_species12, differentiable_mu_eff_from_species12
                from src.core_physics.clot_phi_simple import comsol_carreau_mu_si_from_uv
                from src.inference.corrector_coupling import write_coupled_flow_into_y, set_coupled_flow
                
                sp_next = pin_species_block(data, t + 1, device)
                sp_next = scatter_log_state_to_species_block(sp_next, log_state, node_idx)
                species_log12 = sp_next
                
                phi_clot = differentiable_clot_phi_from_species12(species_log12, bio_cfg)
                u_t1 = data.y[t + 1, :, 0]
                v_t1 = data.y[t + 1, :, 1]
                gel_factor = torch.ones_like(u_t1)
                mu_carreau_si = comsol_carreau_mu_si_from_uv(
                    data,
                    u_t1,
                    v_t1,
                    gel_factor,
                    phys_cfg,
                    device=device,
                )
                mu_eff_si = differentiable_mu_eff_from_species12(
                    species_log12, mu_carreau_si, phi_clot, bio_cfg, gelation_beta=gel_beta
                ).reshape(-1)

                state = coupler.update(data, mu_eff_si, mu_bulk_si=mu_bulk_si, publish=False)
                set_coupled_flow(data, state.u, state.v)
                write_coupled_flow_into_y(data, state.u, state.v, time_index=t + 1)
            except Exception as e:
                import traceback
                print(f"[WARN] Failed to apply closed-loop flow coupling at eval step {t+1}: {e}")
                traceback.print_exc()
    return out, data


@torch.no_grad()
def deploy_clot_phi_trajectory(
    data,
    species_series: torch.Tensor,
    static: dict,
    phys_cfg,
    bio_cfg,
    device: torch.device,
    *,
    time_index: int | None = None,
    flow_source: str = "gt",
    gelation_beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """``(phi_series[T, N], phi_gt, wall_mask, t_eval)`` -- the full graded clot timeline.

    :func:`deploy_clot_phi_fields` is the ``t_eval`` slice of this. Kept as one function so
    a commit-time diagnostic (docs/WALL_MODEL_PLAN.md s4 Step 1b) reads *when* each node
    crossed the same threshold the score is computed from, instead of re-rolling the
    physics readout with its own masking and drifting from the scored path.
    """
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, rollout_t0_clot_phi
    from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
    import os

    n_times = int(data.y.shape[0])
    t_eval = resolve_deploy_eval_time_index(n_times, time_index=time_index)
    gel_beta = gelation_beta
    with t0_rung2_env():
        nuc_hops = int(os.environ.get("CLOT_V2_NUCLEATION_HOPS", "1"))
        traj = rollout_t0_clot_phi(
            data,
            phys_cfg,
            bio_cfg,
            device,
            gamma_mode=RUNG2_GAMMA_MODE,
            flow_source=flow_source,
            pred_species_series=species_series,
            nucleation=True,
            nucleation_hops=nuc_hops,
            gelation_beta=gel_beta,
        )
    phi_gt = gt_clot_phi_at_time(data, t_eval, phys_cfg, device)
    phi_series = torch.stack([traj[t]["phi"].reshape(-1) for t in range(n_times)], dim=0)

    wall_mask = None
    if hasattr(data, "mask_wall") and data.mask_wall is not None:
        wall_mask = data.mask_wall.bool().to(device=phi_series.device)
    elif hasattr(data, "wall_mask") and data.wall_mask is not None:
        wall_mask = data.wall_mask.bool().to(device=phi_series.device)
    elif wall_mat_only_enabled():
        wm = static.get("wall_mask_band")
        if wm is not None:
            wall_mask = wm.bool().to(device=phi_series.device)

    # Enforce wall-only evaluation for the wall model
    if wall_mask is not None:
        wm_row = wall_mask.reshape(1, -1).to(dtype=phi_series.dtype)
        phi_series = phi_series * wm_row
        phi_gt = phi_gt * wall_mask
    return phi_series, phi_gt.reshape(-1), wall_mask, int(t_eval)


@torch.no_grad()
def deploy_clot_phi_fields(
    data,
    species_series: torch.Tensor,
    static: dict,
    phys_cfg,
    bio_cfg,
    device: torch.device,
    *,
    time_index: int | None = None,
    flow_source: str = "gt",
    gelation_beta: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """``(phi_pred, phi_gt, wall_mask, t_eval)`` for the graded clot label.

    Split out of :func:`grade_deploy_clot_series` so diagnostics can inspect the very
    fields the metric is computed from -- which nodes are TP/FP/FN -- without
    re-deriving the masking and risking a protocol that drifts from the scored one.
    """
    phi_series, phi_gt, wall_mask, t_eval = deploy_clot_phi_trajectory(
        data,
        species_series,
        static,
        phys_cfg,
        bio_cfg,
        device,
        time_index=time_index,
        flow_source=flow_source,
        gelation_beta=gelation_beta,
    )
    return phi_series[t_eval].reshape(-1), phi_gt, wall_mask, t_eval


@torch.no_grad()
def grade_deploy_clot_series(
    data,
    species_series: torch.Tensor,
    static: dict,
    phys_cfg,
    bio_cfg,
    device: torch.device,
    *,
    time_index: int | None = None,
    flow_source: str = "gt",
    gelation_beta: float | None = None,
) -> dict[str, float]:
    """Graded clot metrics from an already-computed species rollout.

    The readout is cheap relative to the rollout, so a diagnostic can call this repeatedly
    at different ``gelation_beta`` against one cached series to trace the beta curve with
    the closed loop held fixed.
    """
    from src.evaluation.clot_relaxed_metrics import (
        clot_score_from_deploy_dict,
        compute_clot_relaxed_metrics,
        metrics_to_deploy_prefix,
    )

    gel_beta = gelation_beta
    phi_pred, phi_gt, wall_mask, t_eval = deploy_clot_phi_fields(
        data,
        species_series,
        static,
        phys_cfg,
        bio_cfg,
        device,
        time_index=time_index,
        flow_source=flow_source,
        gelation_beta=gel_beta,
    )

    from src.evaluation.pocket_gate import apply_pocket_gate, resolve_pocket_gate_percentile

    gate_pct = resolve_pocket_gate_percentile()
    gate_stats: dict[str, float] = {}
    if gate_pct is not None:
        phi_pred, gate_stats = apply_pocket_gate(
            phi_pred, data, device, percentile=gate_pct, wall_mask=wall_mask
        )

    edge_index = data.edge_index.to(device=device)
    m = compute_clot_relaxed_metrics(
        phi_pred,
        phi_gt,
        edge_index,
        wall_mask=wall_mask,
    )
    out = metrics_to_deploy_prefix(m)
    out["deploy_clot_score"] = clot_score_from_deploy_dict(out)
    out["time_index"] = int(t_eval)
    out["deploy_gelation_beta"] = float(gel_beta) if gel_beta is not None else 1.0
    out.update(gate_stats)
    return out


@torch.no_grad()
def eval_deploy_clot_f1(
    model: nn.Module,
    data,
    static: dict,
    phys_cfg,
    bio_cfg,
    device: torch.device,
    *,
    time_index: int | None = None,
    flow_source: str = "gt",
) -> dict[str, float]:
    """Closed-loop species rollout -> nucleation clot F1 at ``time_index`` (deploy physics)."""
    from src.core_physics.species_viscosity_calibration import resolve_clot_readout_beta

    # Readout gain on the gelation leg. None => beta 1 (historical grading). Resolved once
    # per anchor so the same value drives the closed loop and the final graded readout.
    gel_beta = resolve_clot_readout_beta()
    if gel_beta is not None:
        print(f"[i] deploy clot readout gelation_beta={gel_beta:.4f}", flush=True)

    species_series, data = deploy_species_rollout_series(
        model,
        data,
        static,
        phys_cfg,
        bio_cfg,
        device,
        flow_source=flow_source,
        gelation_beta=gel_beta,
    )
    return grade_deploy_clot_series(
        data,
        species_series,
        static,
        phys_cfg,
        bio_cfg,
        device,
        time_index=time_index,
        flow_source=flow_source,
        gelation_beta=gel_beta,
    )


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
    time_window: Sequence[int] | None = None,
    flow_series: torch.Tensor | None = None,
    flow_cols: tuple[int, int] | None = None,
    wall_mask_band: torch.Tensor | None = None,
    species_block: torch.Tensor | None = None,
    velocity: torch.Tensor | None = None,
) -> dict[str, float]:
    """Primary metric: final cumulative state F1 after soft-commit readout (ceiling mask)."""
    log_states, deltas, actives = rollout_continuous_states(
        model,
        base_feats=base_feats,
        edge_index=edge_index,
        log_series=log_series,
        log_state0=log_state0 if log_state0 is not None else log_series[0],
        speed_series=speed_series,
        time_window=time_window,
        flow_series=flow_series,
        flow_cols=flow_cols,
        wall_mask_band=wall_mask_band,
        species_block=species_block,
        velocity=velocity,
    )
    gt_active = log_state_to_active(log_series[-1])
    pred_active = actives[-1]
    sm = trigger_metrics(pred_active, gt_active, mask)
    init_active = log_state_to_active(log_series[0])
    sm_init = trigger_metrics(init_active, gt_active, mask)
    gt_actives = [log_state_to_active(s) for s in log_series[: len(actives)]]
    front_diag = mat_front_diagnostics(actives, gt_actives, mask=mask)

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
            li_mat = _local_mat_idx()
            if li_mat is not None:
                mat_g = grew[:, li_mat]
                if bool(mat_g.any().item()):
                    pg = (pred_d[m][:, li_mat][mat_g] > thr).float()
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
        **front_diag,
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
        "phase": str(meta.get("phase", "biochem_gnn")),
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


def _widen_input_rows(
    src: torch.Tensor,
    tgt: torch.Tensor,
    *,
    old_in: int,
    new_in: int,
    insert_at: int,
) -> torch.Tensor:
    """Copy ``src`` into ``tgt`` inserting zero columns for new band extras before the tail.

    Layout: ``[shared_prefix | NEW_EXTRAS | state/sat/time_tail]``.
    """
    out = tgt.clone()
    insert_w = int(new_in) - int(old_in)
    if insert_w <= 0 or src.shape[-1] < old_in or out.shape[-1] < new_in:
        n = min(int(src.shape[-1]), int(out.shape[-1]))
        out[..., :n] = src[..., :n].to(device=out.device, dtype=out.dtype)
        return out
    pre = int(insert_at)
    # Direct input weights: [*, in_dim]
    if src.shape[-1] == old_in and out.shape[-1] == new_in:
        out[..., :pre] = src[..., :pre].to(device=out.device, dtype=out.dtype)
        out[..., pre : pre + insert_w] = 0
        out[..., pre + insert_w :] = src[..., pre:].to(device=out.device, dtype=out.dtype)
        return out
    # Fused heads: [*, hidden + in_dim] — widen only the trailing in_dim block.
    if src.shape[-1] > old_in and out.shape[-1] == src.shape[-1] + insert_w:
        h = int(src.shape[-1] - old_in)
        out[..., :h] = src[..., :h].to(device=out.device, dtype=out.dtype)
        out[..., h : h + pre] = src[..., h : h + pre].to(device=out.device, dtype=out.dtype)
        out[..., h + pre : h + pre + insert_w] = 0
        out[..., h + pre + insert_w :] = src[..., h + pre :].to(device=out.device, dtype=out.dtype)
        return out
    n = min(int(src.shape[-1]), int(out.shape[-1]))
    out[..., :n] = src[..., :n].to(device=out.device, dtype=out.dtype)
    return out


def load_pushforward_state_dict_partial(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    quiet: bool = False,
    src_in_dim: int | None = None,
) -> int:
    """Load compatible tensors; copy overlapping output rows when scope/out_dim widens.

    When ``src_in_dim`` is set and the model input is wider (new geom/flux band channels
    inserted before the state/sat/time tail), input weights are widened with zero-filled
    insert columns instead of truncating extras or randomly reiniting the first layer.
    """
    # Warm-start Spatially-Gated Readout Heads (Direction 6.1)
    if getattr(model, "spatial_gate_heads", False):
        new_state_dict = {}
        for key, val in state_dict.items():
            if key.startswith("spatial_head."):
                suffix = key[len("spatial_head."):]
                new_state_dict["spatial_head_wall." + suffix] = val.clone()
                new_state_dict["spatial_head_offwall." + suffix] = val.clone()
            elif key.startswith("magnitude_head."):
                suffix = key[len("magnitude_head."):]
                new_state_dict["magnitude_head_wall." + suffix] = val.clone()
                new_state_dict["magnitude_head_offwall." + suffix] = val.clone()
        state_dict = {**state_dict, **new_state_dict}

    dst = dict(model.state_dict())
    new_in = int(getattr(model, "in_dim", 0) or 0)
    old_in = int(src_in_dim) if src_in_dim is not None else 0
    widen = bool(old_in > 0 and new_in > old_in)
    insert_at = 0
    if widen:
        tail = continuous_tail_feature_dim()
        insert_at = max(int(old_in) - int(tail), 0)
    copied = 0
    skipped: list[str] = []
    widened = 0
    for key, src in state_dict.items():
        if key not in dst:
            skipped.append(key)
            continue
        tgt = dst[key]
        if src.shape == tgt.shape:
            dst[key] = src.to(device=tgt.device, dtype=tgt.dtype)
            copied += 1
            continue
        if (
            widen
            and src.ndim == 2
            and tgt.ndim == 2
            and src.shape[0] == tgt.shape[0]
            and int(tgt.shape[1]) > int(src.shape[1])
        ):
            dst[key] = _widen_input_rows(
                src, tgt, old_in=old_in, new_in=new_in, insert_at=insert_at
            )
            copied += 1
            widened += 1
            continue
        if (
            key.endswith(".weight")
            and src.ndim == 2
            and tgt.ndim == 2
            and src.shape[1] == tgt.shape[1]
        ):
            n = min(int(src.shape[0]), int(tgt.shape[0]))
            dst[key][:n] = src[:n].to(device=tgt.device, dtype=tgt.dtype)
            copied += 1
            continue
        if key.endswith(".bias") and src.ndim == 1 and tgt.ndim == 1:
            n = min(int(src.shape[0]), int(tgt.shape[0]))
            dst[key][:n] = src[:n].to(device=tgt.device, dtype=tgt.dtype)
            copied += 1
            continue
        if (
            key.endswith(".0.weight")
            and src.ndim == 2
            and tgt.ndim == 2
            and src.shape[0] == tgt.shape[0]
        ):
            in_c = min(int(src.shape[1]), int(tgt.shape[1]))
            dst[key][:, :in_c] = src[:, :in_c].to(device=tgt.device, dtype=tgt.dtype)
            copied += 1
            continue
        skipped.append(key)
    model.load_state_dict(dst)
    if not quiet:
        ckpt_out = int(state_dict.get("readout.2.weight", torch.empty(0)).shape[0]) if state_dict else 0
        msg = f"[OK] partial ckpt load ({copied} tensors"
        if widened:
            msg += f", widened_in {old_in}->{new_in} insert@{insert_at}"
        if skipped:
            msg += f", skipped {len(skipped)}"
        if hasattr(model, "out_dim"):
            msg += f", out {ckpt_out}->{int(model.out_dim)}"
        print(msg + ")", flush=True)
    return copied


def load_continuous_bundle(
    ckpt_path: Path | str | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
    architecture: str | None = None,
    apply_meta_env: bool = True,
) -> SpeciesContinuousBundle | None:
    """Load a continuous pushforward checkpoint.

    Architecture is reconstructed from typed ``PushforwardConfig.from_meta(meta)``.
    ``apply_meta_env`` is retained for API compatibility but no longer mutates
    ``os.environ`` for architecture knobs.
    """
    del apply_meta_env  # architecture comes from typed meta, not os.environ
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
    config = PushforwardConfig.from_meta(meta)
    if payload.get("dual_head") is not None and meta.get("dual_head") is None:
        from dataclasses import replace as _replace

        config = _replace(config, dual_head=bool(payload.get("dual_head")))

    ckpt_dual = bool(config.dual_head)
    if architecture == "single":
        use_dual = False
    elif architecture == "dual":
        use_dual = True
    else:
        use_dual = ckpt_dual

    out_dim = len(config.channels) if config.channels else None
    ckpt_arch = str(config.arch or meta.get("pushforward_arch") or "").strip().lower()

    from src.architecture.pushforward_config import use_pushforward_config

    with use_pushforward_config(config):
        if use_dual:
            if ckpt_arch == "gnode":
                from src.core_physics.species_gnode_pushforward import SpeciesGnodeDualHeadContinuousGNN

                model = SpeciesGnodeDualHeadContinuousGNN(in_dim, hidden=hidden).to(dev)
            else:
                model = SpeciesDualHeadContinuousGNN(
                    in_dim, hidden=hidden, out_dim=out_dim, arch=ckpt_arch or "sage"
                ).to(dev)
        else:
            model = SpeciesContinuousPushforwardGNN(
                in_dim, hidden=hidden, out_dim=out_dim, arch=ckpt_arch or "sage"
            ).to(dev)
        load_pushforward_state_dict_partial(model, payload["model_state"], quiet=quiet)
        model.eval()
        return SpeciesContinuousBundle(
            model=model,
            latent_dim=int(meta.get("latent_dim", in_dim - 1 - _sd())),
            hidden=hidden,
            unroll=int(meta.get("unroll", pushforward_unroll_steps(config))),
            stride=int(meta.get("stride", pushforward_step_stride(config))),
            device=dev,
        )


def init_continuous_from_snapshot(
    model: SpeciesContinuousPushforwardGNN,
    snapshot_ckpt: Path | str,
    *,
    quiet: bool = False,
) -> bool:
    return init_pushforward_from_snapshot(model, snapshot_ckpt, quiet=quiet)
