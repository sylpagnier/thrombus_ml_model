"""Phase 2: discrete autoregressive species pushforward on wall-band subgraph.

Predicts per-step **growth** (0->1 transitions) with state carried across unrolled steps.
See ``docs/SPECIES_GNN_LADDER.md`` Phase 2.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.core_physics.species_snapshot_gnn import (
    SpeciesSnapshotGNN,
    build_snapshot_features,
    fi_mat_active_labels,
    fi_mat_log_targets,
    induced_subgraph,
    kin_per_vessel_norm_enabled,
    kinematic_latent_band_stats,
    snapshot_active_log_nd,
    snapshot_focal_gamma_channels,
    snapshot_hidden_dim,
    snapshot_loss,
    snapshot_loss_mode,
    snapshot_wall_hops,
    trigger_metrics,
    wall_band_mask,
)
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.utils.paths import get_project_root

DEFAULT_PUSHFORWARD_CKPT = "outputs/biochem/species_snapshot_s2/best.pth"
STATE_DIM = 2  # FI commit, Mat commit


def pushforward_ckpt_path() -> Path:
    raw = (os.environ.get("SPECIES_PUSHFORWARD_CKPT") or DEFAULT_PUSHFORWARD_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def pushforward_unroll_steps() -> int:
    return max(int(float(os.environ.get("SPECIES_PUSHFORWARD_UNROLL", "5") or "5")), 1)


def pushforward_step_stride() -> int:
    return max(int(float(os.environ.get("SPECIES_PUSHFORWARD_STEP_STRIDE", "1") or "1")), 1)


def pushforward_input_noise() -> float:
    raw = (os.environ.get("SPECIES_PUSHFORWARD_INPUT_NOISE") or "0.05").strip()
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.05


def pushforward_focal_alpha_channels() -> tuple[float, float]:
    """Higher default alpha for sparse growth-front targets."""
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi = _env_float_channel("SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI", 0.95)
    mat = _env_float_channel("SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT", 0.92)
    return fi, mat


def pushforward_focal_gamma_channels() -> tuple[float, float]:
    from src.core_physics.species_snapshot_gnn import _env_float_channel, snapshot_focal_gamma

    base = snapshot_focal_gamma()
    fi = _env_float_channel("SPECIES_PUSHFORWARD_FOCAL_GAMMA_FI", base)
    mat = _env_float_channel("SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT", base)
    return fi, mat


def pushforward_channel_weights() -> tuple[float, float]:
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi = _env_float_channel("SPECIES_PUSHFORWARD_CHANNEL_WEIGHT_FI", 1.0)
    mat = _env_float_channel("SPECIES_PUSHFORWARD_CHANNEL_WEIGHT_MAT", 1.5)
    return fi, mat


def pushforward_growth_thresholds() -> tuple[float, float]:
    """Sigmoid cutoffs for growth logits at eval/viz (Mat often needs a higher cut)."""
    from src.core_physics.species_snapshot_gnn import _env_float_channel

    fi = _env_float_channel("SPECIES_PUSHFORWARD_GROWTH_THRESH_FI", 0.5)
    mat = _env_float_channel("SPECIES_PUSHFORWARD_GROWTH_THRESH_MAT", 0.65)
    return fi, mat


def pushforward_train_t0_max() -> int | None:
    raw = (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_MAX") or "").strip()
    if not raw:
        return None
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return None


def pushforward_train_t0_min() -> int | None:
    raw = (os.environ.get("SPECIES_PUSHFORWARD_TRAIN_T0_MIN") or "").strip()
    if not raw:
        return None
    try:
        return max(int(float(raw)), 0)
    except ValueError:
        return None


def pushforward_tau_center() -> int:
    raw = (os.environ.get("SPECIES_PUSHFORWARD_TAU_CENTER") or "25").strip()
    try:
        return int(float(raw))
    except ValueError:
        return 25


def pushforward_tau_sigma() -> float:
    raw = (os.environ.get("SPECIES_PUSHFORWARD_TAU_SIGMA") or "6").strip()
    try:
        return max(float(raw), 1.0)
    except ValueError:
        return 6.0


def pushforward_window_t0_weight(t0: int) -> float:
    """Higher weight near critical gelation window (default T20-T30)."""
    t0_min = pushforward_train_t0_min()
    if t0_min is not None and int(t0) < int(t0_min):
        return 0.0
    center = pushforward_tau_center()
    sigma = pushforward_tau_sigma()
    bump = float(math.exp(-0.5 * ((int(t0) - center) / sigma) ** 2))
    floor = 0.35
    return floor + (1.0 - floor) * bump


def pushforward_val_score_weights() -> tuple[float, float]:
    try:
        gw = float(os.environ.get("SPECIES_PUSHFORWARD_SCORE_GROWTH_W", "0.75") or "0.75")
    except ValueError:
        gw = 0.75
    try:
        sw = float(os.environ.get("SPECIES_PUSHFORWARD_SCORE_STATE_W", "0.25") or "0.25")
    except ValueError:
        sw = 0.25
    norm = max(gw + sw, 1e-6)
    return gw / norm, sw / norm


def apply_growth_thresholds(probs: torch.Tensor) -> torch.Tensor:
    fi_thr, mat_thr = pushforward_growth_thresholds()
    out = probs.reshape(-1, STATE_DIM).clone()
    if fi_thr != 0.5:
        out[:, 0] = (out[:, 0] >= fi_thr).float()
    if mat_thr != 0.5:
        out[:, 1] = (out[:, 1] >= mat_thr).float()
    return out


def filter_pushforward_windows(
    windows: list[list[int]],
    data,
    node_idx: torch.Tensor,
    device: torch.device,
    *,
    t0_max: int | None = None,
    min_growth_nodes: int = 0,
) -> list[list[int]]:
    """Drop late/quiet windows; keep those with enough growth positives."""
    out: list[list[int]] = []
    for win in windows:
        if t0_max is not None and int(win[0]) > int(t0_max):
            continue
        if min_growth_nodes <= 0:
            out.append(win)
            continue
        series = active_series_on_band(data, win, device, node_idx)
        n_growth = 0
        for step in range(len(series) - 1):
            g = growth_active_labels(series[step], series[step + 1])
            n_growth += int((g > 0.5).any(dim=-1).sum().item())
        if n_growth >= int(min_growth_nodes):
            out.append(win)
    return out


def step_loss_weights(n_steps: int) -> list[float]:
    mode = (os.environ.get("SPECIES_PUSHFORWARD_STEP_LOSS") or "linear").strip().lower()
    if mode in ("uniform", "flat", "1"):
        return [1.0] * max(n_steps, 1)
    # linear: weight later unroll steps more (error accumulation)
    denom = max(n_steps * (n_steps + 1) // 2, 1)
    return [(step + 1) / denom for step in range(n_steps)]


def pushforward_feature_dim(latent_dim: int, *, state_dim: int = STATE_DIM) -> int:
    return int(latent_dim) + 1 + int(state_dim)


def build_band_base_features(
    data,
    kine_model,
    device: torch.device,
    *,
    wall_hops: int | None = None,
) -> dict:
    """Wall-band subgraph + optional per-vessel ``z_kin`` standardization."""
    hops = snapshot_wall_hops() if wall_hops is None else int(wall_hops)
    data = data.to(device)
    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=hops)
    node_idx, edge_sub, _ = induced_subgraph(band, data.edge_index)
    from src.utils.kinematics_inference import predict_kinematics_latent

    z_kin = predict_kinematics_latent(kine_model, data)
    kin_mean = kin_std = None
    if kin_per_vessel_norm_enabled():
        kin_mean, kin_std = kinematic_latent_band_stats(z_kin, node_idx)
    sdf = sdf_nd_from_data(data, device, n)
    base_feats = build_snapshot_features(z_kin, sdf, kin_mean=kin_mean, kin_std=kin_std)[node_idx]
    return {
        "base_feats": base_feats,
        "edge_index": edge_sub,
        "node_idx": node_idx,
        "n_band": int(node_idx.numel()),
        "n_times": int(data.y.shape[0]),
        "kin_mean": kin_mean,
        "kin_std": kin_std,
    }


def build_pushforward_features(
    z_kin: torch.Tensor,
    sdf_nd: torch.Tensor,
    state_prev: torch.Tensor,
) -> torch.Tensor:
    """``[z_kin, sdf_n, state_prev]`` on band nodes."""
    base = build_snapshot_features(z_kin, sdf_nd)
    st = state_prev.reshape(-1, STATE_DIM).to(device=base.device, dtype=base.dtype).clamp(0.0, 1.0)
    return torch.cat([base, st], dim=-1)


def growth_active_labels(
    active_prev: torch.Tensor,
    active_next: torch.Tensor,
) -> torch.Tensor:
    """Per-channel 0->1 growth: ``(next > thr) & (prev <= thr)``."""
    prev = active_prev.reshape(-1, STATE_DIM).float()
    nxt = active_next.reshape(-1, STATE_DIM).float()
    return ((nxt > 0.5) & (prev <= 0.5)).to(dtype=torch.float32)


def active_series_on_band(
    data,
    time_indices: Sequence[int],
    device: torch.device,
    node_idx: torch.Tensor,
    *,
    thresh_log_nd: float | None = None,
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for t in time_indices:
        log = fi_mat_log_targets(data, int(t), device)[node_idx]
        out.append(fi_mat_active_labels(log, thresh_log_nd=thresh_log_nd))
    return out


def pushforward_state_step(
    state: torch.Tensor,
    logits: torch.Tensor,
    *,
    straight_through: bool = True,
) -> torch.Tensor:
    """OR new growth onto cumulative commit state (straight-through binary optional)."""
    growth = torch.sigmoid(logits.reshape(-1, STATE_DIM))
    if straight_through:
        hard = (growth > 0.5).float()
        growth = hard + growth - growth.detach()
    return torch.maximum(state.reshape(-1, STATE_DIM), growth).clamp(0.0, 1.0)


def maybe_noise_state(state: torch.Tensor, *, training: bool) -> torch.Tensor:
    sigma = pushforward_input_noise()
    if not training or sigma <= 0.0:
        return state
    noise = torch.randn_like(state) * sigma
    return (state + noise).clamp(0.0, 1.0)


class SpeciesPushforwardGNN(SpeciesSnapshotGNN):
    """Phase 2 GNN: same GraphSAGE + residual readout, wider input for prior state."""

    def __init__(self, in_dim: int, *, hidden: int | None = None, out_dim: int = STATE_DIM):
        super().__init__(in_dim, hidden=hidden, out_dim=out_dim)


def iter_pushforward_windows(
    n_times: int,
    *,
    unroll: int | None = None,
    stride: int | None = None,
) -> list[list[int]]:
    """Valid time index windows ``[t0, t1, ..., t_unroll]`` (length unroll+1)."""
    u = pushforward_unroll_steps() if unroll is None else max(int(unroll), 1)
    s = pushforward_step_stride() if stride is None else max(int(stride), 1)
    span = u * s
    wins: list[list[int]] = []
    for t0 in range(0, n_times - span):
        wins.append([t0 + i * s for i in range(u + 1)])
    return wins


def unroll_pushforward_loss(
    model: nn.Module,
    *,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    active_series: list[torch.Tensor],
    train_mask: torch.Tensor,
    state0: torch.Tensor | None = None,
    focal_alpha: tuple[float, float] | None = None,
    focal_gamma: tuple[float, float] | None = None,
    training: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Unroll ``len(active_series)-1`` growth steps; sum focal on growth targets."""
    n_steps = len(active_series) - 1
    if n_steps <= 0:
        z = base_feats.sum() * 0.0
        return z, [], []

    state = (
        torch.zeros(base_feats.shape[0], STATE_DIM, device=base_feats.device, dtype=base_feats.dtype)
        if state0 is None
        else state0.clone()
    )
    alpha = focal_alpha if focal_alpha is not None else pushforward_focal_alpha_channels()
    gamma = focal_gamma if focal_gamma is not None else pushforward_focal_gamma_channels()
    ch_w = pushforward_channel_weights()
    step_w = step_loss_weights(n_steps)

    losses: list[torch.Tensor] = []
    pred_growths: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []

    for step in range(n_steps):
        state_in = maybe_noise_state(state, training=training)
        feats = torch.cat([base_feats, state_in], dim=-1)
        logits = model(feats, edge_index)
        logits_all.append(logits)
        growth_tgt = growth_active_labels(active_series[step], active_series[step + 1])
        pred_growths.append(torch.sigmoid(logits))
        losses.append(
            snapshot_loss(
                logits,
                growth_tgt,
                growth_tgt,
                train_mask,
                focal_alpha=alpha,
                focal_gamma=gamma,
                channel_weight=ch_w,
            )
            * float(step_w[step])
        )
        state = pushforward_state_step(state, logits, straight_through=training)

    wsum = max(sum(step_w), 1e-6)
    return torch.stack(losses).sum() / wsum, pred_growths, logits_all


@torch.no_grad()
def rollout_pushforward_states(
    model: nn.Module,
    *,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    active_series: list[torch.Tensor],
    state0: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Closed-loop rollout; returns cumulative states and growth probs per step."""
    model.eval()
    n_steps = len(active_series) - 1
    state = (
        torch.zeros(base_feats.shape[0], STATE_DIM, device=base_feats.device, dtype=base_feats.dtype)
        if state0 is None
        else state0.clone()
    )
    states = [state.clone()]
    growths: list[torch.Tensor] = []
    for step in range(n_steps):
        feats = torch.cat([base_feats, state], dim=-1)
        logits = model(feats, edge_index)
        growth = torch.sigmoid(logits)
        growths.append(growth)
        state = pushforward_state_step(state, logits, straight_through=False)
        states.append(state.clone())
    return states, growths


def growth_metrics(
    pred_growth: torch.Tensor,
    tgt_growth: torch.Tensor,
    mask: torch.Tensor,
    *,
    apply_thresh: bool = True,
) -> dict[str, float]:
    pred = apply_growth_thresholds(pred_growth) if apply_thresh else pred_growth
    return trigger_metrics(pred, tgt_growth, mask)


@torch.no_grad()
def eval_pushforward_window(
    model: nn.Module,
    *,
    base_feats: torch.Tensor,
    edge_index: torch.Tensor,
    active_series: list[torch.Tensor],
    mask: torch.Tensor,
    state0: torch.Tensor | None = None,
) -> dict[str, float]:
    """Mean growth F1 across unroll steps + final cumulative state F1."""
    states, growths = rollout_pushforward_states(
        model,
        base_feats=base_feats,
        edge_index=edge_index,
        active_series=active_series,
        state0=state0 if state0 is not None else active_series[0],
    )
    if not growths:
        return {
            "mean_growth_f1": 0.0,
            "mean_growth_mat_f1": 0.0,
            "final_state_f1": 0.0,
            "final_state_mat_f1": 0.0,
        }
    gf1: list[float] = []
    gmf1: list[float] = []
    for step, pred_g in enumerate(growths):
        gt_g = growth_active_labels(active_series[step], active_series[step + 1])
        m = growth_metrics(pred_g, gt_g, mask)
        gf1.append(float(m["trigger_f1"]))
        gmf1.append(float(m["mat_f1"]))
    sm = trigger_metrics(states[-1], active_series[-1], mask)
    return {
        "mean_growth_f1": sum(gf1) / len(gf1),
        "mean_growth_mat_f1": sum(gmf1) / len(gmf1),
        "final_state_f1": float(sm["trigger_f1"]),
        "final_state_mat_f1": float(sm["mat_f1"]),
        "per_step_growth_f1": gf1,
    }


def save_pushforward_checkpoint(
    path: Path | str,
    model: SpeciesPushforwardGNN,
    meta: dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "in_dim": model.in_dim,
        "hidden": model.hidden,
        "out_dim": model.out_dim,
        "phase": "s2_pushforward",
        "meta": meta,
    }
    torch.save(payload, p)
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class SpeciesPushforwardBundle:
    model: SpeciesPushforwardGNN
    latent_dim: int
    hidden: int
    unroll: int
    stride: int
    device: torch.device


def init_pushforward_from_snapshot(
    model: SpeciesPushforwardGNN,
    snapshot_ckpt: Path | str,
    *,
    quiet: bool = False,
) -> bool:
    """Warm-start phase-2 from phase-1 ``SpeciesSnapshotGNN`` (pad +2 state input dims)."""
    path = Path(snapshot_ckpt)
    if not path.is_file():
        if not quiet:
            print(f"[WARN] snapshot init missing: {path}")
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    s1 = dict(payload.get("model_state") or {})
    s2 = model.state_dict()
    s1_in = int(payload.get("in_dim", 0))
    s2_in = int(model.in_dim)
    if s1_in <= 0 or s2_in != s1_in + STATE_DIM:
        if not quiet:
            print(f"[WARN] snapshot init dim mismatch s1={s1_in} s2={s2_in}")
        return False

    copied = 0
    for key, w1 in s1.items():
        if key not in s2:
            continue
        w2 = s2[key]
        if w1.shape == w2.shape:
            s2[key] = w1.clone()
            copied += 1
            continue
        # conv1: widen input by STATE_DIM (zero-init new state channels)
        if key.endswith("conv1.lin_l.weight") and w1.ndim == 2 and w2.shape[0] == w1.shape[0]:
            pad = torch.zeros(w2.shape[0], s2_in - s1_in, dtype=w1.dtype)
            s2[key] = torch.cat([w1, pad], dim=1)
            copied += 1
            continue
        if key.endswith("conv1.lin_r.weight") and w1.ndim == 2 and w2.shape[0] == w1.shape[0]:
            pad = torch.zeros(w2.shape[0], s2_in - s1_in, dtype=w1.dtype)
            s2[key] = torch.cat([w1, pad], dim=1)
            copied += 1
            continue
        # readout: [h + x_orig] -> widen x_orig slice
        if key.startswith("readout.0.weight") and w1.ndim == 2:
            h = w1.shape[0]
            s2[key][:, :h] = w1[:, :h]
            s2[key][:, h : h + s1_in] = w1[:, h : h + s1_in]
            copied += 1
            continue

    model.load_state_dict(s2)
    if not quiet:
        print(f"[OK] init pushforward from snapshot ({copied} tensors) s1_in={s1_in} -> s2_in={s2_in}")
    return copied > 0


def load_pushforward_bundle(
    ckpt_path: Path | str | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> SpeciesPushforwardBundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else pushforward_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] pushforward checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    in_dim = int(payload.get("in_dim", 0))
    hidden = int(payload.get("hidden", snapshot_hidden_dim()))
    meta = dict(payload.get("meta") or {})
    model = SpeciesPushforwardGNN(in_dim, hidden=hidden).to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return SpeciesPushforwardBundle(
        model=model,
        latent_dim=int(meta.get("latent_dim", in_dim - 1 - STATE_DIM)),
        hidden=hidden,
        unroll=int(meta.get("unroll", pushforward_unroll_steps())),
        stride=int(meta.get("stride", pushforward_step_stride())),
        device=dev,
    )
