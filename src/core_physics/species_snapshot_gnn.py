"""Wall-band GraphSAGE species operator (canonical: ``species_graphsage``).

SciML type: discrete-time graph autoregressive operator on a wall-band subgraph.
Backbone: 3-layer GraphSAGE; inputs = frozen ``GINO_DEQ`` latent ``z_kin`` + SDF.
Deploy continuous variant extends this with dual-head pushforward (FI/Mat only).

See ``docs/MODEL_NOMENCLATURE.md`` and ``docs/SPECIES_GNN_LADDER.md``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.config import BiochemConfig, PhysicsConfig
from src.utils import species_channels as sc
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.training.biochem_loss_policy import SpatialFocalLoss
from src.training.biochem_species_scope import (
    FI_CHANNEL,
    MAT_CHANNEL,
    pushforward_state_bulk_indices,
    scatter_log_state_to_species_block,
)
from src.utils.paths import get_project_root

DEFAULT_SNAPSHOT_CKPT = "outputs/biochem/species_snapshot_s1/best.pth"
DEFAULT_SPECIES_GNN_VIZ_DIR = "outputs/biochem/viz/species_gnn"


def species_gnn_viz_dir() -> Path:
    raw = (os.environ.get("SPECIES_GNN_VIZ_DIR") or DEFAULT_SPECIES_GNN_VIZ_DIR).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def species_gnn_viz_stem(*, phase: str = "s1", anchor: str) -> str:
    """Basename under ``species_gnn_viz_dir()`` (no extension)."""
    return f"{phase.strip().lower()}_{anchor.strip().lower()}"


def snapshot_ckpt_path() -> Path:
    raw = (os.environ.get("SPECIES_SNAPSHOT_CKPT") or DEFAULT_SNAPSHOT_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def snapshot_wall_hops() -> int:
    return max(int(float(os.environ.get("SPECIES_SNAPSHOT_WALL_HOPS", "3") or "3")), 0)


def snapshot_time_s_default() -> float:
    raw = (os.environ.get("SPECIES_SNAPSHOT_TIME_S") or "5000").strip()
    try:
        return float(raw)
    except ValueError:
        return 5000.0


def snapshot_active_log_nd() -> float:
    raw = (os.environ.get("SPECIES_SNAPSHOT_ACTIVE_LOG_ND") or "1e-4").strip()
    try:
        return float(raw)
    except ValueError:
        return 1e-4


def snapshot_loss_mode() -> str:
    raw = (os.environ.get("SPECIES_SNAPSHOT_LOSS") or "focal").strip().lower()
    if raw in ("bce", "focal"):
        return "focal"
    return raw if raw == "mse" else "focal"


def snapshot_focal_auto_alpha() -> bool:
    return (os.environ.get("SPECIES_SNAPSHOT_FOCAL_AUTO_ALPHA") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def focal_alpha_from_imbalance(n_neg: int, n_pos: int, *, floor: float = 0.75, cap: float = 0.98) -> float:
    """Map neg:pos ratio to focal alpha (higher alpha -> more positive weight)."""
    if n_pos <= 0:
        return cap
    ratio = float(n_neg) / float(max(n_pos, 1))
    # ratio ~6 -> ~0.86; ratio ~31 -> ~0.97
    alpha = 1.0 - (1.0 / (1.0 + 0.15 * ratio))
    return float(min(max(alpha, floor), cap))


def snapshot_focal_alpha(*, n_neg: int | None = None, n_pos: int | None = None) -> float:
    raw = (os.environ.get("SPECIES_SNAPSHOT_FOCAL_ALPHA") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    if snapshot_focal_auto_alpha() and n_neg is not None and n_pos is not None:
        return focal_alpha_from_imbalance(n_neg, n_pos)
    return 0.90


def snapshot_focal_gamma() -> float:
    raw = (os.environ.get("SPECIES_SNAPSHOT_FOCAL_GAMMA") or "2.0").strip()
    try:
        return float(raw)
    except ValueError:
        return 2.0


def _env_float_channel(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def snapshot_focal_alpha_channels(
    *,
    n_neg: int | None = None,
    n_pos: int | None = None,
) -> tuple[float, float]:
    """Per-channel focal alpha: FI (ch0), Mat (ch1). Lower Mat alpha reduces wall halo."""
    base = snapshot_focal_alpha(n_neg=n_neg, n_pos=n_pos)
    fi = _env_float_channel("SPECIES_SNAPSHOT_FOCAL_ALPHA_FI", base)
    mat = _env_float_channel("SPECIES_SNAPSHOT_FOCAL_ALPHA_MAT", base)
    return fi, mat


def snapshot_focal_gamma_channels() -> tuple[float, float]:
    base = snapshot_focal_gamma()
    fi = _env_float_channel("SPECIES_SNAPSHOT_FOCAL_GAMMA_FI", base)
    mat = _env_float_channel("SPECIES_SNAPSHOT_FOCAL_GAMMA_MAT", base)
    return fi, mat


def snapshot_channel_weights() -> tuple[float, float]:
    fi = _env_float_channel("SPECIES_SNAPSHOT_CHANNEL_WEIGHT_FI", 1.0)
    mat = _env_float_channel("SPECIES_SNAPSHOT_CHANNEL_WEIGHT_MAT", 1.0)
    return fi, mat


def snapshot_pred_thresholds() -> tuple[float, float]:
    """Sigmoid thresholds for FI / Mat at eval (Mat often needs a higher cut)."""
    fi = _env_float_channel("SPECIES_SNAPSHOT_FI_THRESH", 0.5)
    mat = _env_float_channel("SPECIES_SNAPSHOT_MAT_THRESH", 0.55)
    return fi, mat


def snapshot_hidden_dim() -> int:
    return max(int(float(os.environ.get("SPECIES_SNAPSHOT_HIDDEN", "64") or "64")), 16)


def resolve_time_index(data, *, time_s: float | None = None) -> int:
    """Map physical seconds to nearest COMSOL macro-step index."""
    target_s = snapshot_time_s_default() if time_s is None else float(time_s)
    n = int(data.y.shape[0])
    if hasattr(data, "t") and data.t is not None:
        t = data.t.to(dtype=torch.float32).reshape(-1)
        if t.numel() >= 1:
            idx = int((t - target_s).abs().argmin().item())
            return max(0, min(idx, n - 1))
    # Fallback: map target_s onto [0, t_final] using bio t_final if present.
    ref = float(BiochemConfig(phase="biochem").t_final)
    frac = max(min(target_s / max(ref, 1e-6), 1.0), 0.0)
    return max(0, min(int(round(frac * (n - 1))), n - 1))


def wall_band_mask(data, device: torch.device, *, wall_hops: int | None = None) -> torch.Tensor:
    """``mask_wall`` + ``wall_hops`` off-wall dilation (drops inner lumen)."""
    hops = snapshot_wall_hops() if wall_hops is None else int(wall_hops)
    return resolve_ceiling_mask(data, device, BiochemConfig(phase="biochem"), ceiling_hops=hops)


def induced_subgraph(
    band: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Node-induced subgraph: ``(node_idx, edge_index_sub, remap_full_to_sub)``."""
    node_idx = band.nonzero(as_tuple=False).view(-1)
    n_full = int(band.numel())
    remap = torch.full((n_full,), -1, dtype=torch.long, device=band.device)
    remap[node_idx] = torch.arange(node_idx.numel(), device=band.device, dtype=torch.long)
    row, col = edge_index
    keep = band[row] & band[col]
    sub_ei = torch.stack([remap[row[keep]], remap[col[keep]]], dim=0)
    return node_idx, sub_ei, remap


def species_log_targets(
    data,
    time_index: int,
    device: torch.device,
    *,
    bulk_channels: list[int] | None = None,
) -> torch.Tensor:
    """GT log1p ND for active pushforward species channels, shape ``[N, C]``."""
    y = data.y[int(time_index)].to(device=device, dtype=torch.float32)
    sp = y[:, sc.SPECIES_BLOCK]
    bulk = bulk_channels if bulk_channels is not None else pushforward_state_bulk_indices()
    return torch.stack([sp[:, int(ch)] for ch in bulk], dim=-1)


def fi_mat_log_targets(
    data,
    time_index: int,
    device: torch.device,
) -> torch.Tensor:
    """GT FI/Mat log1p channels, shape ``[N, 2]`` (legacy alias)."""
    return species_log_targets(data, time_index, device, bulk_channels=[FI_CHANNEL, MAT_CHANNEL])


def fi_mat_active_labels(
    tgt_log: torch.Tensor,
    *,
    thresh_log_nd: float | None = None,
) -> torch.Tensor:
    """Binary trigger labels per channel from log1p ND magnitude."""
    thr = snapshot_active_log_nd() if thresh_log_nd is None else float(thresh_log_nd)
    return (tgt_log > thr).to(dtype=torch.float32)


def kin_per_vessel_norm_enabled() -> bool:
    raw = (os.environ.get("SPECIES_KIN_PER_VESSEL_NORM") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def kinematic_latent_band_stats(
    z_kin: torch.Tensor,
    node_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vessel mean/std of ``z_kin`` on wall-band nodes (geometry-relative scaling)."""
    idx = node_idx.reshape(-1)
    zb = z_kin[idx]
    mean = zb.mean(dim=0)
    std = zb.std(dim=0).clamp(min=1e-6)
    return mean, std


def normalize_kinematic_latent(
    z_kin: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    m = mean.reshape(1, -1).to(device=z_kin.device, dtype=z_kin.dtype)
    s = std.reshape(1, -1).to(device=z_kin.device, dtype=z_kin.dtype).clamp(min=1e-6)
    return (z_kin - m) / s


def build_snapshot_features(
    z_kin: torch.Tensor,
    sdf_nd: torch.Tensor,
    *,
    kin_mean: torch.Tensor | None = None,
    kin_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concat frozen latent + normalized wall distance."""
    z = z_kin
    if kin_mean is not None and kin_std is not None:
        z = normalize_kinematic_latent(z_kin, kin_mean, kin_std)
    sdf = sdf_nd.reshape(-1, 1).to(device=z.device, dtype=z.dtype).clamp(min=0.0)
    sdf_cap = max(float(sdf.max().item()), 1e-6)
    sdf_n = sdf / sdf_cap
    return torch.cat([z, sdf_n], dim=-1)


def snapshot_feature_dim(latent_dim: int) -> int:
    return int(latent_dim) + 1


class SpeciesSnapshotGNN(nn.Module):
    """3-layer GraphSAGE + skip readout (``z_kin`` + SDF -> FI/Mat). Not a generic GNN/GNO."""

    def __init__(self, in_dim: int, *, hidden: int | None = None, out_dim: int = 2):
        super().__init__()
        h = snapshot_hidden_dim() if hidden is None else max(int(hidden), 16)
        self.in_dim = int(in_dim)
        self.hidden = h
        self.out_dim = int(out_dim)
        self.conv1 = SAGEConv(self.in_dim, h)
        self.conv2 = SAGEConv(h, h)
        self.conv3 = SAGEConv(h, h)
        self.readout = nn.Sequential(
            nn.Linear(h + self.in_dim, h),
            nn.ReLU(),
            nn.Linear(h, self.out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_hidden(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return F.relu(self.conv3(h, edge_index))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_orig = x
        h = self.forward_hidden(x, edge_index)
        h_fused = torch.cat([h, x_orig], dim=-1)
        return self.readout(h_fused)


@dataclass(frozen=True)
class SpeciesSnapshotBundle:
    model: SpeciesSnapshotGNN
    latent_dim: int
    hidden: int
    loss_mode: str
    time_index: int
    time_s: float
    wall_hops: int
    active_log_nd: float
    device: torch.device


def logits_to_probs(
    logits: torch.Tensor,
    *,
    loss_mode: str = "focal",
    thresh_log_nd: float | None = None,
    mat_thresh: float | None = None,
    fi_thresh: float | None = None,
) -> torch.Tensor:
    if loss_mode in ("bce", "focal"):
        out = torch.sigmoid(logits)
        env_fi, env_mat = snapshot_pred_thresholds()
        fi_thr = env_fi if fi_thresh is None else float(fi_thresh)
        mat_thr = env_mat if mat_thresh is None else float(mat_thresh)
        if fi_thr != 0.5 or mat_thr != 0.5:
            out = out.clone()
            if fi_thr != 0.5:
                out[:, 0] = (out[:, 0] >= fi_thr).float()
            if mat_thr != 0.5:
                out[:, 1] = (out[:, 1] >= mat_thr).float()
        return out
    thr = snapshot_active_log_nd() if thresh_log_nd is None else float(thresh_log_nd)
    return fi_mat_active_labels(logits, thresh_log_nd=thr)


def trigger_metrics(
    pred: torch.Tensor,
    tgt_active: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Precision/recall/F1 for combined FI|Mat trigger inside ``mask``."""
    if not bool(mask.any().item()):
        return {
            "trigger_prec": 0.0,
            "trigger_rec": 0.0,
            "trigger_f1": 0.0,
            "fi_f1": 0.0,
            "mat_f1": 0.0,
            "pred_pos_frac": 0.0,
            "gt_pos_frac": 0.0,
        }
    m = mask.reshape(-1).bool()
    pred_m = pred.reshape(pred.shape[0], -1)[m]
    tgt_m = tgt_active.reshape(tgt_active.shape[0], -1)[m]
    pred_b = (pred_m > 0.5).any(dim=-1).float()
    tgt_b = (tgt_m > 0.5).any(dim=-1).float()
    tp = float((pred_b * tgt_b).sum().item())
    fp = float((pred_b * (1.0 - tgt_b)).sum().item())
    fn = float(((1.0 - pred_b) * tgt_b).sum().item())
    prec = tp / max(tp + fp, 1e-6)
    rec = tp / max(tp + fn, 1e-6)
    f1 = (2.0 * prec * rec) / max(prec + rec, 1e-6)

    def _ch_f1(ch: int | None) -> float:
        if ch is None or ch < 0 or ch >= int(pred.shape[-1]):
            return 0.0
        pb = (pred[m, ch] > 0.5).float()
        tb = (tgt_active[m, ch] > 0.5).float()
        tpi = float((pb * tb).sum().item())
        fpi = float((pb * (1.0 - tb)).sum().item())
        fni = float(((1.0 - pb) * tb).sum().item())
        pi = tpi / max(tpi + fpi, 1e-6)
        ri = tpi / max(tpi + fni, 1e-6)
        return (2.0 * pi * ri) / max(pi + ri, 1e-6)

    bulk = pushforward_state_bulk_indices()
    li_fi = bulk.index(FI_CHANNEL) if FI_CHANNEL in bulk else None
    li_mat = bulk.index(MAT_CHANNEL) if MAT_CHANNEL in bulk else None

    n = float(m.sum().item())
    return {
        "trigger_prec": prec,
        "trigger_rec": rec,
        "trigger_f1": f1,
        "fi_f1": _ch_f1(li_fi),
        "mat_f1": _ch_f1(li_mat),
        "pred_pos_frac": float(pred_b.sum().item()) / max(n, 1.0),
        "gt_pos_frac": float(tgt_b.sum().item()) / max(n, 1.0),
    }


def snapshot_loss(
    pred: torch.Tensor,
    tgt_log: torch.Tensor,
    tgt_active: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_mode: str | None = None,
    pos_weight: float = 5.0,
    focal_alpha: float | tuple[float, float] | None = None,
    focal_gamma: float | tuple[float, float] | None = None,
    channel_weight: tuple[float, float] | None = None,
) -> torch.Tensor:
    mode = snapshot_loss_mode() if loss_mode is None else loss_mode
    if mode == "bce":
        mode = "focal"
    m = mask.reshape(-1).bool()
    if not bool(m.any().item()):
        return pred.sum() * 0.0
    p = pred[m]
    if mode == "mse":
        return F.mse_loss(p, tgt_log[m])
    active = tgt_active[m]
    n_pos = int((active > 0.5).sum().item())
    n_neg = int(active.numel()) - n_pos
    if focal_alpha is None:
        alpha: float | tuple[float, float] = snapshot_focal_alpha_channels(
            n_neg=n_neg, n_pos=max(n_pos, 1)
        )
    elif isinstance(focal_alpha, (list, tuple)):
        alpha = tuple(float(x) for x in focal_alpha)
    else:
        alpha = float(focal_alpha)
    if focal_gamma is None:
        gamma: float | tuple[float, float] = snapshot_focal_gamma_channels()
    elif isinstance(focal_gamma, (list, tuple)):
        gamma = tuple(float(x) for x in focal_gamma)
    else:
        gamma = float(focal_gamma)
    cw = channel_weight if channel_weight is not None else snapshot_channel_weights()
    return SpatialFocalLoss(alpha=alpha, gamma=gamma, channel_weight=cw)(p, active)


def save_snapshot_checkpoint(
    path: Path | str,
    model: SpeciesSnapshotGNN,
    meta: dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "in_dim": model.in_dim,
        "hidden": model.hidden,
        "out_dim": model.out_dim,
        "meta": meta,
    }
    torch.save(payload, p)
    side = p.with_suffix(".json")
    side.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_snapshot_bundle(
    ckpt_path: Path | str | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> SpeciesSnapshotBundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else snapshot_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] species snapshot checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    in_dim = int(payload.get("in_dim", 0))
    hidden = int(payload.get("hidden", snapshot_hidden_dim()))
    meta = dict(payload.get("meta") or {})
    model = SpeciesSnapshotGNN(in_dim, hidden=hidden).to(dev)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return SpeciesSnapshotBundle(
        model=model,
        latent_dim=int(meta.get("latent_dim", in_dim - 1)),
        hidden=hidden,
        loss_mode=str(meta.get("loss_mode", snapshot_loss_mode())),
        time_index=int(meta.get("time_index", 0)),
        time_s=float(meta.get("time_s", snapshot_time_s_default())),
        wall_hops=int(meta.get("wall_hops", snapshot_wall_hops())),
        active_log_nd=float(meta.get("active_log_nd", snapshot_active_log_nd())),
        device=dev,
    )
