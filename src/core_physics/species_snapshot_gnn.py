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


def build_even_hop_edges(edge_index: torch.Tensor, even_mask: torch.Tensor) -> torch.Tensor:
    row, col = edge_index
    even_to_odd = even_mask[row] & (~even_mask[col])
    even_nodes = row[even_to_odd]
    odd_nodes = col[even_to_odd]
    if odd_nodes.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    perm = torch.argsort(odd_nodes)
    even_sorted = even_nodes[perm]
    odd_sorted = odd_nodes[perm]
    unique_odds, counts = torch.unique_consecutive(odd_sorted, return_counts=True)
    cum_counts = torch.cat([torch.tensor([0], device=edge_index.device), torch.cumsum(counts, dim=0)])
    u_list = []
    v_list = []
    max_count = int(counts.max().item())
    for i in range(max_count):
        for j in range(i + 1, max_count):
            valid_mask = counts > j
            if not valid_mask.any():
                continue
            idx_i = cum_counts[:-1][valid_mask] + i
            idx_j = cum_counts[:-1][valid_mask] + j
            u_list.append(even_sorted[idx_i])
            v_list.append(even_sorted[idx_j])
    if len(u_list) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    u = torch.cat(u_list)
    v = torch.cat(v_list)
    sub_edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], dim=0)
    return torch.unique(sub_edge_index, dim=1)


def find_odd_to_even_neighbors(edge_index: torch.Tensor, even_mask: torch.Tensor, odd_mask: torch.Tensor):
    num_nodes = even_mask.size(0)
    device = edge_index.device
    row, col = edge_index
    even_to_odd = even_mask[row] & odd_mask[col]
    even_nodes = row[even_to_odd]
    odd_nodes = col[even_to_odd]
    if odd_nodes.numel() == 0:
        return torch.zeros(num_nodes, dtype=torch.long, device=device), torch.zeros(num_nodes, dtype=torch.long, device=device)
    perm = torch.argsort(odd_nodes)
    even_sorted = even_nodes[perm]
    odd_sorted = odd_nodes[perm]
    unique_odds, counts = torch.unique_consecutive(odd_sorted, return_counts=True)
    cum_counts = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)])
    neighbor_1 = torch.zeros(num_nodes, dtype=torch.long, device=device)
    neighbor_2 = torch.zeros(num_nodes, dtype=torch.long, device=device)
    valid_1 = counts >= 1
    idx_1 = cum_counts[:-1][valid_1]
    neighbor_1[unique_odds[valid_1]] = even_sorted[idx_1]
    valid_2 = counts >= 2
    idx_2 = cum_counts[:-1][valid_2] + 1
    neighbor_2[unique_odds[valid_2]] = even_sorted[idx_2]
    single_neighbor = (counts == 1)
    if single_neighbor.any():
        odd_singles = unique_odds[single_neighbor]
        neighbor_2[odd_singles] = neighbor_1[odd_singles]
    return neighbor_1, neighbor_2


def compute_frontier_fluxes(
    pos: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    edge_index: torch.Tensor,
    committed: torch.Tensor,
    frontier: torch.Tensor,
    ap: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos.device
    num_nodes = pos.size(0)
    
    flux_ap = torch.zeros(num_nodes, device=device, dtype=pos.dtype)
    flux_t = torch.zeros(num_nodes, device=device, dtype=pos.dtype)
    
    row, col = edge_index
    valid_edges = frontier[row] & committed[col]
    
    if not valid_edges.any():
        return flux_ap, flux_t
        
    f_nodes = row[valid_edges]
    c_nodes = col[valid_edges]
    
    diff = pos[c_nodes] - pos[f_nodes]
    dist = diff.norm(dim=1).clamp(min=1e-6)
    dir_vec = diff / dist.unsqueeze(-1)
    
    vel = torch.stack([u[f_nodes], v[f_nodes]], dim=1)
    dot = (vel * dir_vec).sum(dim=1)
    incoming = torch.clamp(dot, min=0.0)
    
    flux_ap_edge = incoming * ap[f_nodes]
    flux_t_edge = incoming * t[f_nodes]
    
    flux_ap.index_add_(0, f_nodes, flux_ap_edge)
    flux_t.index_add_(0, f_nodes, flux_t_edge)
    
    return flux_ap, flux_t


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
        # Multiscale skip-hop convolution layers
        self.skip_conv1 = SAGEConv(self.in_dim, h)
        self.skip_conv2 = SAGEConv(h, h)
        self.skip_conv3 = SAGEConv(h, h)
        
        # Convective upwind projection layers
        self.conv_proj1 = nn.Linear(h, h)
        self.conv_proj2 = nn.Linear(h, h)
        self.conv_proj3 = nn.Linear(h, h)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.pos_band = None
        self.edge_index = None
        self.wall_mask_band = None
        self._cached_hops = None
        self._cached_even_edges = None
        self._cached_even_mask = None
        self._cached_odd_mask = None
        self._cached_odd_to_even_neighbors = None
        self.shear_gate_tau = nn.Parameter(torch.tensor(10.0))
        self.shear_gate_lss = nn.Parameter(torch.tensor(50.0))
        self.augmented_edge_index = None
        self.skip_edge_index = None

    def set_band_geometry(self, pos_band, edge_index, wall_mask_band=None):
        self.pos_band = pos_band
        self.edge_index = edge_index
        if wall_mask_band is not None:
            self.wall_mask_band = wall_mask_band
            self._cached_hops = None
            self._cached_even_edges = None
            self._cached_even_mask = None
            self._cached_odd_mask = None
            self._cached_odd_to_even_neighbors = None
        if os.environ.get("SPECIES_LONGRANGE_EDGES") == "1" and pos_band is not None and edge_index is not None and wall_mask_band is not None:
            mult = float(os.environ.get("SPECIES_LONGRANGE_DIST_MULT", "2.5"))
            self.augmented_edge_index = build_longrange_edges(pos_band, edge_index, wall_mask_band, max_mult=mult)
        else:
            self.augmented_edge_index = None

        if os.environ.get("SPECIES_MULTISCALE_SKIP_HOP") == "1" and pos_band is not None and edge_index is not None and wall_mask_band is not None:
            mult = float(os.environ.get("SPECIES_MULTISCALE_SKIP_HOP_MULT", "3.0"))
            self.skip_edge_index = build_skip_only_edges(pos_band, edge_index, wall_mask_band, max_mult=mult)
        else:
            self.skip_edge_index = None

    def _get_hops_and_subgraphs(self, edge_index: torch.Tensor, num_nodes: int):
        if not hasattr(self, "wall_mask_band") or self.wall_mask_band is None:
            return None
        # Only valid when operating on the exact band graph where wall_mask was set
        if self.wall_mask_band.size(0) != num_nodes:
            return None
        if getattr(self, "_cached_hops", None) is not None:
            return self._cached_hops, self._cached_even_mask, self._cached_odd_mask, self._cached_even_edges
        wall_mask = self.wall_mask_band.to(device=edge_index.device)
        hops = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        hops[wall_mask] = 0
        row, col = edge_index
        current_mask = wall_mask.clone()
        current_hop = 0
        while True:
            neighbor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
            neighbor_mask[col[current_mask[row]]] = True
            next_mask = neighbor_mask & (hops == -1)
            if not next_mask.any():
                break
            current_hop += 1
            hops[next_mask] = current_hop
            current_mask = next_mask
        hops[hops == -1] = 99
        even_mask = (hops % 2 == 0)
        odd_mask = ~even_mask
        even_edge_index = build_even_hop_edges(edge_index, even_mask)
        self._cached_hops = hops
        self._cached_even_mask = even_mask
        self._cached_odd_mask = odd_mask
        self._cached_even_edges = even_edge_index
        return hops, even_mask, odd_mask, even_edge_index

    def _reconstruct_odd_nodes(self, values: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = values.size(0)
        # Invalidate cache if graph size has changed (e.g. full graph vs band)
        if self._cached_even_mask is not None and self._cached_even_mask.size(0) != num_nodes:
            self._cached_hops = None
            self._cached_even_mask = None
            self._cached_odd_mask = None
            self._cached_even_edges = None
            self._cached_odd_to_even_neighbors = None
        # _get_hops_and_subgraphs already guards against band-size mismatch; returns None if not applicable
        res = self._get_hops_and_subgraphs(edge_index, num_nodes)
        if res is None:
            return values
        hops, even_mask, odd_mask, even_edge_index = res
        # Also invalidate neighbor cache when graph changed
        if getattr(self, "_cached_odd_to_even_neighbors", None) is not None:
            if self._cached_odd_to_even_neighbors[0].size(0) != num_nodes:
                self._cached_odd_to_even_neighbors = None
        if getattr(self, "_cached_odd_to_even_neighbors", None) is None:
            self._cached_odd_to_even_neighbors = find_odd_to_even_neighbors(edge_index, even_mask, odd_mask)
        neighbor_1, neighbor_2 = self._cached_odd_to_even_neighbors
        out = values.clone()
        out[odd_mask] = 0.5 * (values[neighbor_1[odd_mask]] + values[neighbor_2[odd_mask]])
        return out

    def forward_hidden(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if os.environ.get("SPECIES_SKIP_HOP_GNN") == "1" and getattr(self, "wall_mask_band", None) is not None:
            num_nodes = x.size(0)
            res = self._get_hops_and_subgraphs(edge_index, num_nodes)
            if res is not None:
                hops, even_mask, odd_mask, even_edge_index = res
                even_idx = even_mask.nonzero(as_tuple=False).view(-1)
                num_even = even_idx.numel()
                if num_even > 0:
                    remap = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
                    remap[even_idx] = torch.arange(num_even, device=x.device, dtype=torch.long)
                    row_even, col_even = even_edge_index
                    remap_row = remap[row_even]
                    remap_col = remap[col_even]
                    keep = (remap_row != -1) & (remap_col != -1)
                    sub_edge_index_even = torch.stack([remap_row[keep], remap_col[keep]], dim=0)
                    x_even = x[even_idx]
                    h_even = F.relu(self.conv1(x_even, sub_edge_index_even))
                    h_even = F.relu(self.conv2(h_even, sub_edge_index_even))
                    h_even = F.relu(self.conv3(h_even, sub_edge_index_even))
                    h = torch.zeros((num_nodes, self.hidden), device=x.device, dtype=x.dtype)
                    h[even_idx] = h_even
                    return h
        # Layer 1
        h1 = F.relu(self.conv1(x, edge_index))
        if os.environ.get("SPECIES_CONVECTIVE_UPWIND") == "1" and getattr(self, "velocity", None) is not None and getattr(self, "pos_band", None) is not None:
            h1_up = convective_upwind_message_passing(h1, edge_index, self.velocity, self.pos_band)
            h1 = h1 + F.relu(self.conv_proj1(h1_up))
            
        # Layer 2
        h2 = F.relu(self.conv2(h1, edge_index))
        if os.environ.get("SPECIES_CONVECTIVE_UPWIND") == "1" and getattr(self, "velocity", None) is not None and getattr(self, "pos_band", None) is not None:
            h2_up = convective_upwind_message_passing(h2, edge_index, self.velocity, self.pos_band)
            h2 = h2 + F.relu(self.conv_proj2(h2_up))
            
        # Layer 3
        h3 = F.relu(self.conv3(h2, edge_index))
        if os.environ.get("SPECIES_CONVECTIVE_UPWIND") == "1" and getattr(self, "velocity", None) is not None and getattr(self, "pos_band", None) is not None:
            h3_up = convective_upwind_message_passing(h3, edge_index, self.velocity, self.pos_band)
            h3 = h3 + F.relu(self.conv_proj3(h3_up))
            
        # Multiscale Skip-Hop path
        if os.environ.get("SPECIES_MULTISCALE_SKIP_HOP") == "1" and getattr(self, "skip_edge_index", None) is not None:
            skip_edge_index = self.skip_edge_index.to(device=x.device)
            if skip_edge_index.numel() > 0:
                h_skip1 = F.relu(self.skip_conv1(x, skip_edge_index))
                h_skip2 = F.relu(self.skip_conv2(h_skip1, skip_edge_index))
                h_skip3 = F.relu(self.skip_conv3(h_skip2, skip_edge_index))
                blend_scale = float(os.environ.get("SPECIES_MULTISCALE_SKIP_HOP_SCALE", "0.5"))
                h3 = h3 + blend_scale * h_skip3
                
        return h3

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_orig = x
        h = self.forward_hidden(x, edge_index)
        h_fused = torch.cat([h, x_orig], dim=-1)
        out = self.readout(h_fused)
        if os.environ.get("SPECIES_SKIP_HOP_GNN") == "1" and getattr(self, "wall_mask_band", None) is not None:
            out = self._reconstruct_odd_nodes(out, edge_index)

        # Apply readout shear gate
        if os.environ.get("SPECIES_SHEAR_READOUT_GATE") == "1":
            ld = int(getattr(self, "kin_latent_dim", 0) or 0)
            if x.shape[1] > ld + 6:
                gamma_si = x[:, ld + 6].reshape(-1, 1)
                tau = getattr(self, "shear_gate_tau", None)
                lss = getattr(self, "shear_gate_lss", None)
                if tau is not None and lss is not None:
                    gate = torch.sigmoid((lss - gamma_si) / torch.clamp(tau, min=1e-3))
                    from src.training.biochem_species_scope import _local_mat_idx
                    mat_idx = _local_mat_idx()
                    if mat_idx is not None and mat_idx < out.shape[1]:
                        mask = torch.ones_like(out)
                        mask[:, mat_idx] = gate.squeeze(-1)
                        out = out * mask

        # Apply frontier kinetics
        if os.environ.get("SPECIES_FRONTIER_KINETICS") == "1":
            from src.training.biochem_species_scope import _local_mat_idx
            mat_idx = _local_mat_idx()
            pos_band = getattr(self, "pos_band", None)
            velocity = getattr(self, "velocity", None)
            species_block = getattr(self, "species_block", None)
            log_state = getattr(self, "log_state", None)
            if mat_idx is not None and pos_band is not None and velocity is not None and species_block is not None and log_state is not None:
                from src.training.biochem_species_scope import pushforward_state_dim, continuous_mat_commit_thresh
                st = log_state.reshape(-1, pushforward_state_dim())
                committed = (st[:, mat_idx] > continuous_mat_commit_thresh()).reshape(-1).bool()
                from src.core_physics.clot_growth_masks import graph_dilate_hops
                frontier = graph_dilate_hops(committed, edge_index, 1).to(device=x.device) & ~committed
                ap = torch.expm1(species_block[:, sc.block_index("AP")].reshape(-1))
                t_sp = torch.expm1(species_block[:, sc.block_index("T")].reshape(-1))
                u_vel = velocity[:, 0].reshape(-1)
                v_vel = velocity[:, 1].reshape(-1)
                flux_ap, flux_t = compute_frontier_fluxes(
                    pos_band, u_vel, v_vel, edge_index, committed, frontier, ap, t_sp
                )
                k_ap = float(os.environ.get("SPECIES_FRONTIER_K_AP", "0.5"))
                k_t = float(os.environ.get("SPECIES_FRONTIER_K_T", "0.5"))
                K_kinetics = k_ap * flux_ap + k_t * flux_t
                mask = torch.zeros_like(out)
                mask[:, mat_idx] = K_kinetics
                out = out + mask

        return out


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


def build_longrange_edges(pos_band: torch.Tensor, edge_index: torch.Tensor, wall_mask_band: torch.Tensor, max_mult: float = 2.5) -> torch.Tensor:
    if pos_band is None or edge_index is None or wall_mask_band is None:
        return edge_index
    num_nodes = pos_band.shape[0]
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
        if current_hop >= 4:
            break
    hop0_idx = (hops == 0).nonzero(as_tuple=False).view(-1)
    hop23_idx = ((hops == 2) | (hops == 3)).nonzero(as_tuple=False).view(-1)
    if hop0_idx.numel() == 0 or hop23_idx.numel() == 0:
        return edge_index
    pos_row = pos_band[row]
    pos_col = pos_band[col]
    edge_lengths = torch.linalg.norm(pos_row - pos_col, dim=-1)
    mean_length = float(edge_lengths.mean().item()) if edge_lengths.numel() > 0 else 0.3
    dist_thresh = mean_length * max_mult
    p0 = pos_band[hop0_idx]
    p23 = pos_band[hop23_idx]
    dists = torch.cdist(p0.unsqueeze(0), p23.unsqueeze(0)).squeeze(0)
    pairs = (dists < dist_thresh).nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return edge_index
    u_new = hop0_idx[pairs[:, 0]]
    v_new = hop23_idx[pairs[:, 1]]
    new_edges = torch.stack([
        torch.cat([u_new, v_new]),
        torch.cat([v_new, u_new])
    ], dim=0)
    combined = torch.cat([edge_index, new_edges], dim=1)
    return torch.unique(combined, dim=1)


def build_skip_only_edges(
    pos_band: torch.Tensor,
    edge_index: torch.Tensor,
    wall_mask_band: torch.Tensor,
    max_mult: float = 3.0,
) -> torch.Tensor:
    if pos_band is None or edge_index is None or wall_mask_band is None:
        return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
    num_nodes = pos_band.shape[0]
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
        if current_hop >= 4:
            break
    hop0_idx = (hops == 0).nonzero(as_tuple=False).view(-1)
    hop23_idx = ((hops == 2) | (hops == 3)).nonzero(as_tuple=False).view(-1)
    if hop0_idx.numel() == 0 or hop23_idx.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    pos_row = pos_band[row]
    pos_col = pos_band[col]
    edge_lengths = torch.linalg.norm(pos_row - pos_col, dim=-1)
    mean_length = (
        float(edge_lengths.mean().item()) if edge_lengths.numel() > 0 else 0.3
    )
    dist_thresh = mean_length * max_mult
    p0 = pos_band[hop0_idx]
    p23 = pos_band[hop23_idx]
    dists = torch.cdist(p0.unsqueeze(0), p23.unsqueeze(0)).squeeze(0)
    pairs = (dists < dist_thresh).nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    u_new = hop0_idx[pairs[:, 0]]
    v_new = hop23_idx[pairs[:, 1]]
    new_edges = torch.stack(
        [torch.cat([u_new, v_new]), torch.cat([v_new, u_new])], dim=0
    )
    return new_edges


def convective_upwind_message_passing(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    velocity: torch.Tensor,
    pos_band: torch.Tensor,
) -> torch.Tensor:
    """Propagate GNN messages upwind along local velocity vectors."""
    if (
        velocity is None
        or pos_band is None
        or edge_index is None
        or edge_index.numel() == 0
    ):
        return x
    device = x.device
    num_nodes = x.size(0)
    row, col = edge_index
    vel_from = velocity[row].to(device=device, dtype=x.dtype)
    diff = (pos_band[col] - pos_band[row]).to(device=device, dtype=x.dtype)
    alignment = vel_from[:, 0] * diff[:, 0] + vel_from[:, 1] * diff[:, 1]
    weights = F.relu(alignment)

    sum_w = torch.zeros(num_nodes, device=device, dtype=x.dtype)
    sum_w.scatter_add_(0, col, weights)

    norm_w = weights / (sum_w[col] + 1e-6)
    weighted_feats = x[row] * norm_w.unsqueeze(-1)

    out = torch.zeros((num_nodes, x.shape[1]), device=device, dtype=x.dtype)
    out.index_add_(0, col, weighted_feats)
    return out
