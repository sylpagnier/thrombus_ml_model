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
from src.config import NodeFeat
from src.utils.paths import get_project_root

DEFAULT_PUSHFORWARD_CKPT = "outputs/biochem/species_snapshot_s2/best.pth"
from src.training.biochem_species_scope import pushforward_state_dim

STATE_DIM = 2  # legacy default; use pushforward_state_dim() for dynamic scope


def _sd() -> int:
    return pushforward_state_dim()


def pushforward_ckpt_path() -> Path:
    raw = (os.environ.get("SPECIES_PUSHFORWARD_CKPT") or DEFAULT_PUSHFORWARD_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def pushforward_unroll_steps() -> int:
    return max(int(float(os.environ.get("SPECIES_PUSHFORWARD_UNROLL", "10") or "10")), 1)


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
    out = probs.reshape(-1, _sd()).clone()
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


def pushforward_feature_dim(latent_dim: int, *, state_dim: int | None = None) -> int:
    sd = _sd() if state_dim is None else int(state_dim)
    return int(latent_dim) + 1 + sd


def stagnation_feats_enabled() -> bool:
    """Move 4: append deployable low-shear/stagnation proxy features to band inputs."""
    raw = (os.environ.get("SPECIES_STAGNATION_FEATS") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def geom_feats_enabled() -> bool:
    """Append deployable NON-FLOW geometry discriminators (width / expansion / wall curvature).

    Distinct from flow/stagnation feats: these are STATIC geometry only (no kine solve), encoding
    *where* recirculation/low-shear pockets form (expansions, stenoses, bends) -- signal the z_kin
    latent carries only implicitly. The open precision-lever test (leg C) for wall false positives.
    """
    raw = (os.environ.get("SPECIES_GEOM_FEATS") or "0").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    return geom_feats_rich_enabled()


def geom_feats_rich_enabled() -> bool:
    """Enrich the leg-C geometry block with the proven 2-hop commit-vs-eligible discriminators.

    Adds ``width_grad_2hop`` and ``curvature_2hop`` on top of the 3 static channels. The probe
    (docs/archive/SPECIES_LEARNING_STRATEGY.md s6.13) found multi-hop expansion/curvature -- not the 1-hop
    versions alone -- separate *committed* clot pockets from merely *eligible* wall nodes. Static,
    clot-blind, no kine solve, so it stays deployable.
    """
    raw = (os.environ.get("SPECIES_GEOM_FEATS_RICH") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def flow_feats_enabled() -> bool:
    """Full clot-aware flow feature set on band inputs: speed + shear + divergence + geometry.

    The GraphSAGE input is otherwise `[z_kin, sdf]` -- a *clot-blind* latent -- so the teacher
    has no channel that responds to a growing clot's flow diversion (proven by the `--gt-flow`
    gate: feeding perfect velocity is a no-op). This feature set is meant to be **trained on the
    clot-aware GT COMSOL velocity** (`SPECIES_FLOW_FEATS_SOURCE=gt`) and **deployed on the
    corrector-coupled flow** (auto source), so flow accuracy finally maps to Mat localization.
    Distinct from the clot-blind `SPECIES_STAGNATION_FEATS` proxy.
    """
    raw = (os.environ.get("SPECIES_FLOW_FEATS") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def flow_feats_source() -> str:
    """Velocity source for the flow features.

    * ``gt``   -- COMSOL ``data.y[t][:, 0:2]`` (clot-aware ground truth; training).
    * ``kine`` -- frozen GINO-DEQ prediction (clot-blind).
    * ``auto`` -- kine base flow, overridden by the corrector-coupled flow when coupling is on
      (the deploy default; closes the loop with the running clot prediction).
    """
    return (os.environ.get("SPECIES_FLOW_FEATS_SOURCE") or "auto").strip().lower()


def flow_feats_time() -> int:
    """Representative GT time index for the (static) flow features; <0 = last (formed clot)."""
    raw = (os.environ.get("SPECIES_FLOW_FEATS_TIME") or "-1").strip()
    try:
        return int(float(raw))
    except ValueError:
        return -1


def flow_feats_ablate() -> bool:
    """Verification knob: zero the flow feature block (keep its width) to test the latent leash.

    A latent-leashed teacher that actually reads the explicit flow should LOSE F1 when these are
    zeroed; a latent-dominant teacher is unaffected (the `+0.008` ceiling).
    """
    raw = (os.environ.get("SPECIES_FLOW_FEATS_ABLATE") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def flow_feats_drop_xy() -> bool:
    """Ablate only spatial coordinates from flow block while keeping dynamics channels.

    Keeps ``[log_speed, log_shear, tanh_div]`` and zeroes ``[x_norm, y_norm]``. This targets
    inlet/outlet spatial memorization without removing the physically meaningful flow cues.
    """
    raw = (os.environ.get("SPECIES_FLOW_FEATS_DROP_XY") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def flow_feats_dynamic() -> bool:
    """Time-varying flow features (Trap C): recompute the flow block per rollout step from the
    velocity at *that* step's time, instead of a single static (final-clot) snapshot.

    When on, ``build_band_base_features`` also returns a per-time ``flow_series`` (from
    ``data.y[t][:, 0:2]`` -- GT velocity in training/gate, per-step coupled velocity at deploy after
    ``write_coupled_flow_into_y``); the rollout splices ``flow_series[t]`` into the flow columns each
    step. The static base_feats flow block stays at the representative time for step-0 / fallback.
    """
    raw = (os.environ.get("SPECIES_FLOW_FEATS_DYNAMIC") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


@torch.no_grad()
def _base_uv_from_data(data, kine_model, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Clot-blind base flow: prefer cached ``u0_pred``/``v0_pred``, else one GINO-DEQ solve (no clone)."""
    if getattr(data, "u0_pred", None) is not None and getattr(data, "v0_pred", None) is not None:
        u = data.u0_pred.to(device=device, dtype=torch.float32).reshape(-1)
        v = data.v0_pred.to(device=device, dtype=torch.float32).reshape(-1)
        return u, v
    from src.utils.kinematics_inference import predict_kinematics

    try:
        # Do not clone: the UV/latent cache is keyed on ``data.x`` storage pointer.
        uv = predict_kinematics(kine_model, data).to(device=device)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        try:
            uv = predict_kinematics(kine_model, data).to(device=device)
            print("[i] flow-feature kine solve recovered on GPU after empty_cache.")
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                "flow-feature kine solve OOM on CUDA. Silent fallbacks to CPU are disabled "
                "by Hardware Execution Policy to prevent hangs."
            ) from e
    return uv[:, 0].reshape(-1), uv[:, 1].reshape(-1)


@torch.no_grad()
def _resolve_flow_uv(data, kine_model, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-graph velocity ``(u, v)`` (ND) for the flow features per ``SPECIES_FLOW_FEATS_SOURCE``."""
    src = flow_feats_source()
    if src == "gt" and getattr(data, "y", None) is not None and data.y.dim() == 3:
        n_t = int(data.y.shape[0])
        t = flow_feats_time()
        ti = (n_t - 1) if t < 0 else min(int(t), n_t - 1)
        y = data.y[ti].to(device=device, dtype=torch.float32)
        return y[:, 0].reshape(-1), y[:, 1].reshape(-1)
    u, v = _base_uv_from_data(data, kine_model, device)
    if src != "kine":  # auto: corrector-coupled override at deploy (Step F)
        from src.inference.corrector_coupling import corrector_coupling_enabled, get_coupled_flow

        if corrector_coupling_enabled():
            coupled = get_coupled_flow(data, device)
            if coupled is not None:
                u = coupled[0].reshape(-1).to(dtype=u.dtype)
                v = coupled[1].reshape(-1).to(dtype=v.dtype)
    return u, v


FLOW_FEATS_DIM = 5  # [log1p(speed), log1p(shear), tanh(div), x_n, y_n]


@torch.no_grad()
def _flow_feats_from_uv(
    data, u: torch.Tensor, v: torch.Tensor, device: torch.device, node_idx: torch.Tensor
) -> torch.Tensor:
    """The 5-ch flow proxy block for an explicit velocity field ``(u, v)`` on the band nodes.

    * ``speed``      = ``|u|`` (stagnation when low).
    * ``shear``      = mean neighbour speed-gradient magnitude (wall/clot shear layer).
    * ``divergence`` = graph divergence of ``u`` (``sum_j (u_j-u_i).(x_j-x_i)/dist^2 / deg``),
      ``tanh``-bounded; negative = converging/stagnating flow where mature clot accumulates.
    """
    u = u.reshape(-1).to(device=device)
    v = v.reshape(-1).to(device=device)
    speed = torch.sqrt(u * u + v * v)
    pos = data.x[:, :2].to(device=device, dtype=speed.dtype)
    row, col = data.edge_index.to(device=device)
    diff = pos[row] - pos[col]
    dist = diff.norm(dim=1).clamp(min=1e-6)
    grad = (speed[row] - speed[col]).abs() / dist
    div_edge = ((u[row] - u[col]) * diff[:, 0] + (v[row] - v[col]) * diff[:, 1]) / (dist * dist)
    n = int(data.num_nodes)
    acc_g = torch.zeros(n, device=device, dtype=speed.dtype)
    acc_d = torch.zeros(n, device=device, dtype=speed.dtype)
    deg = torch.zeros(n, device=device, dtype=speed.dtype)
    acc_g.index_add_(0, row, grad)
    acc_d.index_add_(0, row, div_edge)
    deg.index_add_(0, row, torch.ones_like(grad))
    shear_proxy = acc_g / deg.clamp(min=1.0)
    divergence = acc_d / deg.clamp(min=1.0)
    pn = (pos - pos.mean(0)) / pos.std(0).clamp(min=1e-6)
    xy = pn[:, :2]
    if flow_feats_drop_xy():
        xy = torch.zeros_like(xy)

    use_shear = (
        os.environ.get("SPECIES_SHEAR_READOUT_GATE") == "1"
        or os.environ.get("SPECIES_FRONTIER_KINETICS") == "1"
    )
    if use_shear:
        from src.utils.rheology import compute_shear_rate
        if hasattr(data, "G_x") and hasattr(data, "G_y") and data.G_x is not None and data.G_y is not None:
            du_dx = torch.sparse.mm(data.G_x, u.unsqueeze(1).to(dtype=torch.float32)).squeeze(1)
            du_dy = torch.sparse.mm(data.G_y, u.unsqueeze(1).to(dtype=torch.float32)).squeeze(1)
            dv_dx = torch.sparse.mm(data.G_x, v.unsqueeze(1).to(dtype=torch.float32)).squeeze(1)
            dv_dy = torch.sparse.mm(data.G_y, v.unsqueeze(1).to(dtype=torch.float32)).squeeze(1)
            gamma_dot_nd = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy, eps=1e-6).to(dtype=speed.dtype)
            if hasattr(data, "u_ref") and data.u_ref is not None:
                if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == data.num_nodes:
                    u_ref = data.u_ref.to(device=device, dtype=speed.dtype).reshape(-1)[:1]
                    d_bar = data.d_bar.to(device=device, dtype=speed.dtype).reshape(-1)[:1]
                else:
                    u_ref = torch.as_tensor(data.u_ref, device=device, dtype=speed.dtype).reshape(1)
                    d_bar = torch.as_tensor(data.d_bar, device=device, dtype=speed.dtype).reshape(1)
                d_safe = torch.clamp(d_bar, min=1e-8)
                scale_si = u_ref / d_safe
                gamma_si = gamma_dot_nd * scale_si
            else:
                gamma_si = gamma_dot_nd
        else:
            gamma_si = torch.zeros(n, device=device, dtype=speed.dtype)

        feats_list = [
            torch.log1p(speed.clamp(min=0)),
            torch.log1p(shear_proxy.clamp(min=0)),
            torch.tanh(divergence),
            xy[:, 0],
            xy[:, 1],
            gamma_si,
        ]
    else:
        feats_list = [
            torch.log1p(speed.clamp(min=0)),
            torch.log1p(shear_proxy.clamp(min=0)),
            torch.tanh(divergence),
            xy[:, 0],
            xy[:, 1],
        ]
    feats = torch.stack(feats_list, dim=1)
    return feats[node_idx]


@torch.no_grad()
def _flow_band_features(data, kine_model, device: torch.device, node_idx: torch.Tensor) -> torch.Tensor:
    """Static (single representative time) clot-aware flow proxies on band nodes.

    Velocity per :func:`flow_feats_source` -- GT COMSOL in training, corrector-coupled at deploy.
    """
    u, v = _resolve_flow_uv(data, kine_model, device)
    return _flow_feats_from_uv(data, u, v, device, node_idx)


@torch.no_grad()
def _flow_feats_series_from_y(data, device: torch.device, node_idx: torch.Tensor) -> torch.Tensor:
    """Per-time flow block ``[n_times, n_band, FLOW_FEATS_DIM]`` from the velocity in ``data.y``.

    ``data.y[t][:, 0:2]`` holds the time-``t`` velocity: GT COMSOL (clot-aware) in training / the
    gt-flow gate, or the per-step corrector-coupled field at deploy (see ``write_coupled_flow_into_y``).
    This is the time-varying signal the static snapshot cannot represent (Trap C).
    """
    n_t = int(data.y.shape[0])
    series = []
    for ti in range(n_t):
        y = data.y[ti].to(device=device, dtype=torch.float32)
        series.append(_flow_feats_from_uv(data, y[:, 0], y[:, 1], device, node_idx))
    return torch.stack(series, dim=0)


@torch.no_grad()
def _stagnation_band_features(data, kine_model, device: torch.device, node_idx: torch.Tensor) -> torch.Tensor:
    """Deployable stagnation proxies on band nodes: [log1p(speed), log1p(shear_proxy), x_norm, y_norm].

    Shear proxy = mean neighbour speed-gradient magnitude (a clot-free flow stagnation signal the
    z_kin latent encodes only implicitly). All from the kinematic flow + geometry -> deployable.
    """
    u, v = _base_uv_from_data(data, kine_model, device)
    # Corrector coupling: bend the base flow around the active clot so the deployable
    # stagnation/shear proxies the GraphSAGE sees reflect the diverted field (Step F).
    from src.inference.corrector_coupling import corrector_coupling_enabled, get_coupled_flow

    if corrector_coupling_enabled():
        coupled = get_coupled_flow(data, device)
        if coupled is not None:
            u, v = coupled[0].to(dtype=u.dtype), coupled[1].to(dtype=v.dtype)
    speed = torch.sqrt(u * u + v * v)
    pos = data.x[:, :2].to(device=device, dtype=speed.dtype)
    row, col = data.edge_index.to(device=device)
    dist = (pos[row] - pos[col]).norm(dim=1).clamp(min=1e-6)
    grad = (speed[row] - speed[col]).abs() / dist
    n = int(data.num_nodes)
    acc = torch.zeros(n, device=device, dtype=speed.dtype)
    deg = torch.zeros(n, device=device, dtype=speed.dtype)
    acc.index_add_(0, row, grad)
    deg.index_add_(0, row, torch.ones_like(grad))
    shear_proxy = acc / deg.clamp(min=1.0)
    pn = (pos - pos.mean(0)) / pos.std(0).clamp(min=1e-6)
    feats = torch.stack(
        [torch.log1p(speed.clamp(min=0)), torch.log1p(shear_proxy.clamp(min=0)), pn[:, 0], pn[:, 1]],
        dim=1,
    )
    return feats[node_idx]


GEOM_FEATS_DIM = 3  # [width, width_gradient, wall_curvature]
GEOM_FEATS_RICH_DIM = 5  # + [width_gradient_2hop, wall_curvature_2hop]


def geom_feats_dim() -> int:
    return GEOM_FEATS_RICH_DIM if geom_feats_rich_enabled() else GEOM_FEATS_DIM


def _hop_mean(values: torch.Tensor, row: torch.Tensor, col: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    """One graph diffusion step: ``mean_j(values_j)`` over each node's neighbours."""
    acc = torch.zeros_like(values)
    acc.index_add_(0, row, values[col])
    return acc / deg


@torch.no_grad()
def _geometry_band_features(data, device: torch.device, node_idx: torch.Tensor) -> torch.Tensor:
    """Deployable NON-FLOW geometry proxies on band nodes, per-band standardized.

    Channels (all static, clot-blind, no kine solve):
      * ``width``         = local lumen width (``NodeFeat.WIDTH_ND``); stenoses/expansions.
      * ``width_grad``    = ``mean_j(width_j) - width_i`` over neighbours; signed expansion(>0) /
        contraction(<0) -- recirculation forms just downstream of an expansion.
      * ``curvature``     = ``mean_j (1 - cos(n_i, n_j))`` of adjacent wall normals; vessel bends
        seed a low-shear pocket on one wall.

    When ``SPECIES_GEOM_FEATS_RICH`` is set, append two further channels:
      * ``width_grad_2hop`` = expansion measured over the 2-hop neighbourhood; captures the
        *extent* of an expansion, the proven commit-vs-eligible discriminator.
      * ``curvature_2hop``  = curvature smoothed over 2 hops; a sustained bend rather than a
        single-edge normal flip.

    These encode *where* coupled stagnation lives, which the z_kin latent carries only implicitly.
    """
    x = data.x.to(device)
    n = int(data.num_nodes)
    width = x[:, NodeFeat.WIDTH_ND].reshape(-1).to(torch.float32)
    wn = x[:, NodeFeat.WALL_NORMAL].to(torch.float32)  # (n, 2) wall-normal direction
    row, col = data.edge_index.to(device)
    ones = torch.ones(row.shape[0], device=device, dtype=torch.float32)
    deg = torch.zeros(n, device=device, dtype=torch.float32)
    deg.index_add_(0, row, ones)
    deg = deg.clamp(min=1.0)
    nbr_w = _hop_mean(width, row, col, deg)
    width_grad = nbr_w - width
    dot = (wn[row] * wn[col]).sum(dim=1)
    cacc = torch.zeros(n, device=device, dtype=torch.float32)
    cacc.index_add_(0, row, (1.0 - dot))
    curv = cacc / deg
    channels = [width, width_grad, curv]
    if geom_feats_rich_enabled():
        width_2hop = _hop_mean(nbr_w, row, col, deg)
        width_grad_2hop = width_2hop - width
        curv_2hop = _hop_mean(curv, row, col, deg)
        channels.extend([width_grad_2hop, curv_2hop])
    feats = torch.stack(channels, dim=1)[node_idx]
    mu = feats.mean(dim=0, keepdim=True)
    sd = feats.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (feats - mu) / sd


def build_band_base_features(
    data,
    kine_model,
    device: torch.device,
    *,
    wall_hops: int | None = None,
    z_kin_override: torch.Tensor | None = None,
) -> dict:
    """Wall-band subgraph + optional per-vessel ``z_kin`` standardization.

    ``z_kin_override`` lets the caller supply a *clot-aware* DEQ latent (from a re-solve with
    the clot ``mu`` injected into ``MU_PRIOR``) so the GraphSAGE teacher's primary flow input
    reflects the rerouted field instead of the frozen clot-free latent.
    """
    hops = snapshot_wall_hops() if wall_hops is None else int(wall_hops)
    data = data.to(device)
    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=hops)
    node_idx, edge_sub, _ = induced_subgraph(band, data.edge_index)
    from src.utils.kinematics_inference import predict_kinematics_latent

    if z_kin_override is not None:
        z_kin = z_kin_override.to(device=device)
    else:
        z_kin = predict_kinematics_latent(kine_model, data)
    kin_mean = kin_std = None
    if kin_per_vessel_norm_enabled():
        kin_mean, kin_std = kinematic_latent_band_stats(z_kin, node_idx)
    sdf = sdf_nd_from_data(data, device, n)
    base_feats = build_snapshot_features(z_kin, sdf, kin_mean=kin_mean, kin_std=kin_std)[node_idx]
    flow_series = None
    flow_cols = None
    if flow_feats_enabled():
        flow_start = int(base_feats.shape[1])  # flow block follows [z_kin, sdf]
        flow = _flow_band_features(data, kine_model, device, node_idx).to(dtype=base_feats.dtype)
        if flow_feats_ablate():
            flow = torch.zeros_like(flow)  # verification: width preserved, signal removed
        base_feats = torch.cat([base_feats, flow], dim=1)
        # Trap C: per-time flow block so the rollout can splice the time-varying field each step.
        if flow_feats_dynamic() and getattr(data, "y", None) is not None and data.y.dim() == 3:
            flow_series = _flow_feats_series_from_y(data, device, node_idx).to(dtype=base_feats.dtype)
            if flow_feats_ablate():
                flow_series = torch.zeros_like(flow_series)
            flow_cols = (flow_start, int(flow.shape[1]))
    elif stagnation_feats_enabled():
        stag = _stagnation_band_features(data, kine_model, device, node_idx).to(dtype=base_feats.dtype)
        base_feats = torch.cat([base_feats, stag], dim=1)
    if geom_feats_enabled():
        # Independent of flow/stagnation: append static non-flow geometry discriminators (leg C).
        geom = _geometry_band_features(data, device, node_idx).to(dtype=base_feats.dtype)
        base_feats = torch.cat([base_feats, geom], dim=1)
    pos_band = data.x[node_idx, :2].to(device=device, dtype=base_feats.dtype)
    from src.core_physics.clot_phi_simple import _wall_mask_from_data
    wall_mask_full = _wall_mask_from_data(data, device, n)
    wall_mask_band = wall_mask_full[node_idx]
    return {
        "base_feats": base_feats,
        "pos_band": pos_band,
        "edge_index": edge_sub,
        "node_idx": node_idx,
        "n_band": int(node_idx.numel()),
        "n_times": int(data.y.shape[0]),
        "kin_mean": kin_mean,
        "kin_std": kin_std,
        "latent_dim": int(z_kin.shape[1]),  # true z_kin width (first cols) for the latent leash
        "flow_series": flow_series,  # [n_t, n_band, flow_dim] when dynamic, else None
        "flow_cols": flow_cols,      # (start, width) of the flow block in base_feats, else None
        "wall_mask_band": wall_mask_band,
    }


def build_pushforward_features(
    z_kin: torch.Tensor,
    sdf_nd: torch.Tensor,
    state_prev: torch.Tensor,
) -> torch.Tensor:
    """``[z_kin, sdf_n, state_prev]`` on band nodes."""
    base = build_snapshot_features(z_kin, sdf_nd)
    st = state_prev.reshape(-1, _sd()).to(device=base.device, dtype=base.dtype).clamp(0.0, 1.0)
    return torch.cat([base, st], dim=-1)


def growth_active_labels(
    active_prev: torch.Tensor,
    active_next: torch.Tensor,
) -> torch.Tensor:
    """Per-channel 0->1 growth: ``(next > thr) & (prev <= thr)``."""
    prev = active_prev.reshape(-1, _sd()).float()
    nxt = active_next.reshape(-1, _sd()).float()
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
    growth = torch.sigmoid(logits.reshape(-1, _sd()))
    if straight_through:
        hard = (growth > 0.5).float()
        growth = hard + growth - growth.detach()
    return torch.maximum(state.reshape(-1, _sd()), growth).clamp(0.0, 1.0)


def maybe_noise_state(state: torch.Tensor, *, training: bool) -> torch.Tensor:
    sigma = pushforward_input_noise()
    if not training or sigma <= 0.0:
        return state
    noise = torch.randn_like(state) * sigma
    return (state + noise).clamp(0.0, 1.0)


class SpeciesPushforwardGNN(SpeciesSnapshotGNN):
    """Phase 2 GNN: same GraphSAGE + residual readout, wider input for prior state."""

    def __init__(self, in_dim: int, *, hidden: int | None = None, out_dim: int | None = None):
        super().__init__(in_dim, hidden=hidden, out_dim=_sd() if out_dim is None else out_dim)


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
        torch.zeros(base_feats.shape[0], _sd(), device=base_feats.device, dtype=base_feats.dtype)
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
        torch.zeros(base_feats.shape[0], _sd(), device=base_feats.device, dtype=base_feats.dtype)
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
    if s1_in <= 0 or s2_in != s1_in + _sd():
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
        latent_dim=int(meta.get("latent_dim", in_dim - 1 - _sd())),
        hidden=hidden,
        unroll=int(meta.get("unroll", pushforward_unroll_steps())),
        stride=int(meta.get("stride", pushforward_step_stride())),
        device=dev,
    )
