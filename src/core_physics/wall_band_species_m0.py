"""M0: wall-band graph reaction head (1-step teacher-forced species).

Predicts departures from resting species on a wall + hop band using GT flow
and previous species state. Channel sets:

  fimat     -- FI + Mat (2 ch)
  cascade4  -- APR + APS + FI + Mat (4 ch)

See ``src/docs/SPECIES_TEMPORAL_ML.md``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from src.config import BiochemConfig, PhysicsConfig, PredChannels
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.kinematics_clot_prior import shear_rate_si
from src.core_physics.species_snapshot_gnn import wall_band_mask
from src.core_physics.t0_mu_physics import predict_clot_phi_at_time
from src.core_physics.t0_rung4_ladder import resting_species_log_nd
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.utils.paths import get_project_root

DEFAULT_M0_CKPT = "outputs/biochem/wall_band_species_m0_fimat/best.pth"

# FI/Mat log-ND departures from rest are ~100-200x smaller than APR/APS on the wall
# band; up-weight gelation channels so cascade4 does not wash out FI/Mat learning.
_GELATION_LOG_WEIGHT = 200.0

CHANNEL_SETS: dict[str, dict[str, Any]] = {
    "fimat": {
        "indices": [8, 11],
        "names": ["FI", "Mat"],
        "weights": [_GELATION_LOG_WEIGHT, _GELATION_LOG_WEIGHT],
        "desc": "Gelation pair only (FI bulk + Mat wall)",
    },
    "cascade4": {
        "indices": [2, 3, 8, 11],
        "names": ["APR", "APS", "FI", "Mat"],
        "weights": [1.0, 1.0, _GELATION_LOG_WEIGHT, _GELATION_LOG_WEIGHT],
        "desc": "Cascade intermediates + gelation pair",
    },
}


def m0_ckpt_path() -> Path:
    raw = (os.environ.get("WALL_BAND_M0_CKPT") or DEFAULT_M0_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def m0_channel_set() -> str:
    raw = (os.environ.get("WALL_BAND_M0_CHANNEL_SET") or "fimat").strip().lower()
    if raw not in CHANNEL_SETS:
        known = ", ".join(sorted(CHANNEL_SETS))
        raise KeyError(f"Unknown channel set {raw!r}; known: {known}")
    return raw


def m0_wall_hops() -> int:
    return max(int(float(os.environ.get("WALL_BAND_M0_WALL_HOPS", "1") or "1")), 0)


def m0_delta_scale() -> float:
    try:
        return max(float(os.environ.get("WALL_BAND_M0_DELTA_SCALE", "0.5") or "0.5"), 1e-6)
    except ValueError:
        return 0.5


def resolve_channel_set(name: str) -> tuple[list[int], list[str], list[float]]:
    key = name.strip().lower()
    if key not in CHANNEL_SETS:
        known = ", ".join(sorted(CHANNEL_SETS))
        raise KeyError(f"Unknown channel set {name!r}; known: {known}")
    spec = CHANNEL_SETS[key]
    indices = list(spec["indices"])
    names = list(spec["names"])
    weights = list(spec.get("weights") or [1.0] * len(indices))
    if len(weights) != len(indices):
        raise ValueError(f"channel_set {key!r}: weights length {len(weights)} != {len(indices)}")
    return indices, names, [float(w) for w in weights]


class _BandConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))


class WallBandSpeciesGNN(nn.Module):
    """2L mesh GNN: wall-band features -> tanh-bounded species delta."""

    def __init__(self, in_dim: int, out_dim: int, *, hidden: int = 48):
        super().__init__()
        h = max(int(hidden), 16)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = h
        self.conv1 = _BandConv(self.in_dim, h)
        self.conv2 = _BandConv(h, h)
        self.head = nn.Linear(h, self.out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x, edge_index))
        h = F.silu(self.conv2(h, edge_index))
        return self.head(h)


@dataclass(frozen=True)
class WallBandM0Bundle:
    model: WallBandSpeciesGNN
    channel_set: str
    channel_indices: list[int]
    channel_names: list[str]
    channel_weights: list[float]
    in_dim: int
    hidden: int
    delta_scale: float
    wall_hops: int
    device: torch.device


def feature_dim(n_channels: int) -> int:
    # prev species (n_ch) + tau + sdf + wall + log_shear + u + v
    return int(n_channels) + 6


def build_m0_features(
    data,
    time_index: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    species_prev: torch.Tensor,
    channel_indices: list[int],
    band: torch.Tensor,
) -> torch.Tensor:
    """Node features for one macro step (full mesh, N x F)."""
    from src.core_physics.clot_phi_simple import _anchor_flow_props

    t = int(time_index)
    n = int(data.num_nodes)
    y = data.y[t].to(device=device, dtype=torch.float32)
    u = y[:, PredChannels.U]
    v = y[:, PredChannels.V]
    props = _anchor_flow_props(data, device)
    tau = float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))
    sdf = sdf_nd_from_data(data, device, n).reshape(-1)
    wall = band.reshape(-1).to(dtype=torch.float32)
    shear = shear_rate_si(data, u, v, props).reshape(-1).clamp(min=1e-8)
    log_sh = torch.log10(shear)
    sp_cols = [species_prev[:, int(ci)].reshape(-1) for ci in channel_indices]
    cols = sp_cols + [
        torch.full((n,), tau, device=device, dtype=torch.float32),
        sdf,
        wall,
        log_sh,
        u.reshape(-1),
        v.reshape(-1),
    ]
    return torch.stack(cols, dim=1)


def predict_delta_at_time(
    data,
    time_index: int,
    device: torch.device,
    bundle: WallBandM0Bundle,
    *,
    species_prev: torch.Tensor,
    bio_cfg: BiochemConfig | None = None,
) -> torch.Tensor:
    """Raw model delta (N x n_ch), tanh-bounded."""
    bio = bio_cfg or BiochemConfig(phase="biochem")
    band = wall_band_mask(data, device, wall_hops=bundle.wall_hops).reshape(-1).bool()
    feats = build_m0_features(
        data, time_index, device, bio,
        species_prev=species_prev, channel_indices=bundle.channel_indices, band=band,
    )
    edge_index = data.edge_index.to(device=device)
    return torch.tanh(bundle.model(feats, edge_index)) * float(bundle.delta_scale)


def apply_delta_to_species(
    species: torch.Tensor,
    delta: torch.Tensor,
    *,
    rest: torch.Tensor,
    channel_indices: list[int],
    band: torch.Tensor,
) -> torch.Tensor:
    """Write rest + delta on band for selected channels only."""
    out = species.clone()
    m = band.reshape(-1).bool()
    if not bool(m.any().item()):
        return out
    for j, ci in enumerate(channel_indices):
        out[m, int(ci)] = rest[m, int(ci)] + delta[m, j]
    return out.clamp(min=0.0)


@torch.no_grad()
def rollout_m0_species_series(
    data,
    bundle: WallBandM0Bundle,
    *,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    pin_other_species: str = "gt",
) -> torch.Tensor:
    """Build full 12-ch series: teacher-forced prev, hybrid non-selected channels.

    ``pin_other_species``: ``gt`` (default) keeps non-modeled channels from GT;
    ``rest`` pins them to resting IC.
    """
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or bundle.device
    n_steps = int(data.y.shape[0])
    out = data.y.clone().to(device=dev)
    rest = resting_species_log_nd(data, dev)
    band = wall_band_mask(data, dev, wall_hops=bundle.wall_hops).reshape(-1).bool()
    species_prev = rest.clone()

    for t in range(n_steps):
        if pin_other_species == "gt":
            sp = data.y[t, :, 4:16].to(device=dev, dtype=torch.float32).clone()
        else:
            sp = rest.clone()
        if t > 0:
            delta = predict_delta_at_time(
                data, t, dev, bundle, species_prev=species_prev, bio_cfg=bio,
            )
            sp = apply_delta_to_species(
                sp, delta, rest=rest, channel_indices=bundle.channel_indices, band=band,
            )
        out[t, :, 4:16] = sp
        species_prev = sp.detach()

    return out


@torch.no_grad()
def rollout_m0_phi_trajectory(
    data,
    bundle: WallBandM0Bundle,
    *,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
) -> dict[int, torch.Tensor]:
    from src.core_physics.t0_mu_physics import rollout_t0_clot_phi

    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or bundle.device
    pred = rollout_m0_species_series(data, bundle, phys_cfg=phys, bio_cfg=bio, device=dev)
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, dev,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt",
            pred_species_series=pred, nucleation=True, nucleation_hops=1,
        )
    return {int(t): v["phi"] for t, v in traj.items()}


def build_m0_bundle(
    channel_set: str,
    device: torch.device,
    *,
    hidden: int = 48,
    delta_scale: float | None = None,
    wall_hops: int | None = None,
) -> WallBandM0Bundle:
    indices, names, weights = resolve_channel_set(channel_set)
    n_ch = len(indices)
    in_dim = feature_dim(n_ch)
    model = WallBandSpeciesGNN(in_dim, n_ch, hidden=hidden).to(device)
    return WallBandM0Bundle(
        model=model,
        channel_set=channel_set.strip().lower(),
        channel_indices=indices,
        channel_names=names,
        channel_weights=weights,
        in_dim=in_dim,
        hidden=max(int(hidden), 16),
        delta_scale=float(delta_scale if delta_scale is not None else m0_delta_scale()),
        wall_hops=int(wall_hops if wall_hops is not None else m0_wall_hops()),
        device=device,
    )


def save_m0_checkpoint(path: Path | str, bundle: WallBandM0Bundle, *, meta: dict[str, Any] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": bundle.model.state_dict(),
        "channel_set": bundle.channel_set,
        "channel_indices": bundle.channel_indices,
        "channel_names": bundle.channel_names,
        "channel_weights": bundle.channel_weights,
        "in_dim": bundle.in_dim,
        "hidden": bundle.hidden,
        "delta_scale": bundle.delta_scale,
        "wall_hops": bundle.wall_hops,
        "meta": meta or {},
    }
    torch.save(payload, p)
    side = {
        k: v for k, v in payload.items()
        if k != "model_state"
    }
    p.with_suffix(".json").write_text(json.dumps(side, indent=2), encoding="utf-8")


def load_m0_bundle(
    ckpt_path: Path | str | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> WallBandM0Bundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else m0_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] wall-band M0 checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    cs = str(payload.get("channel_set", "fimat"))
    bundle = build_m0_bundle(
        cs, dev,
        hidden=int(payload.get("hidden", 48)),
        delta_scale=float(payload.get("delta_scale", m0_delta_scale())),
        wall_hops=int(payload.get("wall_hops", m0_wall_hops())),
    )
    bundle.model.load_state_dict(payload["model_state"])
    bundle.model.eval()
    return bundle


_GELATION_SPECIES_IDX = frozenset({8, 11})  # FI, Mat (log-ND channel index within species block)


def one_step_loss(
    data,
    time_index: int,
    bundle: WallBandM0Bundle,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
) -> torch.Tensor:
    """Teacher-forced weighted MSE on wall band at ``time_index``.

    Gelation channels (FI/Mat) use growth masking and per-node weights so
  cascade4 APR/APS gradients do not drown out tiny FI/Mat targets.
    """
    t = int(time_index)
    rest = resting_species_log_nd(data, device)
    band = wall_band_mask(data, device, wall_hops=bundle.wall_hops).reshape(-1).bool()
    if t <= 0:
        species_prev = rest
    else:
        species_prev = data.y[t - 1, :, 4:16].to(device=device, dtype=torch.float32)
    delta = predict_delta_at_time(
        data, t, device, bundle, species_prev=species_prev, bio_cfg=bio_cfg,
    )
    sp_gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
    growth_eps = 1e-7
    fi_mat_tau = 1e-4
    cascade_tau = 1e-2
    losses: list[torch.Tensor] = []
    for j, ci in enumerate(bundle.channel_indices):
        tgt = sp_gt[:, int(ci)] - rest[:, int(ci)]
        pred = delta[:, j]
        ch_w = float(bundle.channel_weights[j]) if j < len(bundle.channel_weights) else 1.0
        if int(ci) in _GELATION_SPECIES_IDX:
            mask = band & (tgt.abs() > growth_eps)
            if not bool(mask.any().item()):
                continue
            tau = fi_mat_tau
        else:
            mask = band
            tau = cascade_tau
        node_w = 1.0 + (tgt[mask].abs() / tau).clamp(max=50.0)
        err = (pred[mask] - tgt[mask]) ** 2
        losses.append(ch_w * (node_w * err).sum() / node_w.sum())
    if not losses:
        return delta.sum() * 0.0
    return torch.stack(losses).mean()
