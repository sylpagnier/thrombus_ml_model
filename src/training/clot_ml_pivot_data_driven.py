"""Pivot C: data-driven band GNN phi (pred kine + geometry; no hand risk features)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.clot_temporal_growth_rules import (
    _resolve_uv_for_temporal_risk,
    _time_frac_at_index,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_pivot_common import eval_phi_rollout_on_anchor, pivot_out_dir


def data_driven_feature_dim() -> int:
    return 5  # sdf, u, v, p, t_frac


class _PhiGNNConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))


class ClotDataDrivenPhiGNN(nn.Module):
    """Per-node phi logits from pred-kine state + time fraction (no shear physics)."""

    def __init__(self, in_dim: int = 5, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 8)
        self.conv1 = _PhiGNNConv(in_dim, h)
        self.conv2 = _PhiGNNConv(h, h)
        self.head = nn.Linear(h, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_logits(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x, edge_index))
        h = F.silu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


def band_features_data_driven(
    data,
    *,
    device: torch.device,
    t_frac: float,
    flow_time: int = 0,
) -> torch.Tensor:
    n = int(data.num_nodes)
    u, v = _resolve_uv_for_temporal_risk(data, flow_time, device)
    ti = max(0, min(int(flow_time), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=device, dtype=torch.float32)
    p = y[:, 2]
    sdf = sdf_nd_from_data(data, device, n)
    t_col = torch.full((n,), float(t_frac), device=device, dtype=torch.float32)
    return torch.stack([sdf, u, v, p, t_col], dim=1)


def rollout_data_driven_phi(
    data,
    model: ClotDataDrivenPhiGNN,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
) -> dict[int, torch.Tensor]:
    data = data.to(device)
    n_times = int(data.y.shape[0])
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    edge_index = data.edge_index.to(device)
    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    for t_out in range(n_times):
        t_frac = _time_frac_at_index(data, int(t_out))
        feats = band_features_data_driven(data, device=device, t_frac=t_frac)
        logits = model.forward_logits(feats, edge_index)
        phi_step = torch.sigmoid(logits) * ceiling.reshape(-1).float()
        if phi_prev is not None:
            phi = torch.maximum(phi_step, phi_prev)
        else:
            phi = phi_step
        phi_by_t[int(t_out)] = phi
        phi_prev = phi.detach() if model.training else phi
    return phi_by_t


@dataclass
class PivotDataTrainConfig:
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 40
    early_paint_weight: float = 0.25
    final_bce_weight: float = 2.0


def train_one_graph(
    model: ClotDataDrivenPhiGNN,
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    early_paint_weight: float,
    final_bce_weight: float,
) -> torch.Tensor:
    reset_temporal_kinematics_cache()
    data = data.to(device)
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return torch.tensor(0.0, device=device)
    t_final = int(pairs[-1][1])
    phi_by_t = rollout_data_driven_phi(data, model, device=device, bio_cfg=bio_cfg)
    total = torch.tensor(0.0, device=device)
    n = 0
    for t_in, t_out in pairs:
        phi = phi_by_t[int(t_out)]
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        mask = step.loss_mask.reshape(-1).bool()
        if not bool(mask.any().item()):
            continue
        target = step.phi_gt.reshape(-1).to(phi.dtype)
        w = final_bce_weight if int(t_out) == t_final else 1.0
        bce = F.binary_cross_entropy(phi[mask], target[mask], reduction="mean")
        loss = w * bce
        t_frac = _time_frac_at_index(data, int(t_out))
        if t_frac <= 0.35:
            loss = loss + early_paint_weight * phi[mask].mean()
        total = total + loss
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_data_driven_on_anchor(
    model: ClotDataDrivenPhiGNN,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
) -> dict[str, Any]:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    phi_by_t = rollout_data_driven_phi(data, model, device=device, bio_cfg=bio_cfg)
    return eval_phi_rollout_on_anchor(
        phi_by_t,
        graph_path=graph_path,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="pivot_data_driven",
        pair_stride=pair_stride,
    )


def default_data_driven_out_dir() -> Path:
    return pivot_out_dir("data_driven")


def build_data_driven_model(meta: dict[str, Any]) -> ClotDataDrivenPhiGNN:
    return ClotDataDrivenPhiGNN(
        in_dim=data_driven_feature_dim(),
        hidden=int(meta.get("hidden", 32)),
    )
