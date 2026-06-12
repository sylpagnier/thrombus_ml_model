"""Step 7 ML ladder: end-to-end band GNN phi (ceiling + onset gate + monotonic carry).

Uses minimal pred-kine features + hand risk (Step-0 shell) and hard inc40 onset gate.
Pivot lessons: no soft commit (under-pred); no raw data-driven without onset (wall paint).
"""

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
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _resolve_pool_risk,
    _time_frac_at_index,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_pivot_common import eval_phi_rollout_on_anchor
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step2_band_gnn import band_features_pred_kine, step2_feature_dim
from src.utils.paths import get_project_root


def step7_feature_dim() -> int:
    return step2_feature_dim() + 2  # minimal kine (3) + hand risk + t_frac


class _PhiGNNConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))


class ClotBandPhiGNN(nn.Module):
    """Band GNN -> phi logits; init sparse (negative head bias)."""

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
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.0)

    def forward_logits(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x, edge_index))
        h = F.silu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


def band_features_step7(
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    rule_cfg: TemporalGrowthRuleConfig,
    ceiling: torch.Tensor,
    t_frac: float,
    t_out: int,
) -> torch.Tensor:
    kine = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        flow_time=rule_cfg.risk_flow_time,
    )
    _, risk = _resolve_pool_risk(
        data,
        device=device,
        bio_cfg=bio_cfg,
        ceiling=ceiling,
        cfg=rule_cfg,
        t_out=int(t_out),
    )
    n = int(kine.shape[0])
    t_col = torch.full((n,), float(t_frac), device=device, dtype=kine.dtype)
    return torch.cat([kine, risk.reshape(-1, 1), t_col.reshape(-1, 1)], dim=1)


def rollout_step7_phi(
    data,
    rule_cfg: TemporalGrowthRuleConfig,
    model: ClotBandPhiGNN,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    onset_frac: float | None = None,
) -> dict[int, torch.Tensor]:
    data = data.to(device)
    n_times = int(data.y.shape[0])
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    onset = float(onset_frac if onset_frac is not None else rule_cfg.global_onset_frac)
    edge_index = data.edge_index.to(device)
    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    for t_out in range(n_times):
        t_frac = _time_frac_at_index(data, int(t_out))
        if onset > 0 and t_frac < onset:
            phi = torch.zeros(int(data.num_nodes), device=device)
            phi_by_t[int(t_out)] = phi
            phi_prev = phi
            continue
        feats = band_features_step7(
            data,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            rule_cfg=rule_cfg,
            ceiling=ceiling,
            t_frac=t_frac,
            t_out=t_out,
        )
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
class Step7TrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 50
    early_paint_weight: float = 1.0
    final_bce_weight: float = 2.0


def train_one_graph(
    model: ClotBandPhiGNN,
    data,
    rule_cfg: TemporalGrowthRuleConfig,
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
    phi_by_t = rollout_step7_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
    )
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
def eval_step7_on_anchor(
    model: ClotBandPhiGNN,
    rule_cfg: TemporalGrowthRuleConfig,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
) -> dict[str, Any]:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    phi_by_t = rollout_step7_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
    )
    return eval_phi_rollout_on_anchor(
        phi_by_t,
        graph_path=graph_path,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="ml_step7_band_phi",
        pair_stride=pair_stride,
    )


def default_step7_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder/step7_band_phi"


def save_step7_checkpoint(
    path: Path,
    *,
    model: ClotBandPhiGNN,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_step7_checkpoint(
    path: Path | str,
    *,
    device: torch.device,
) -> tuple[ClotBandPhiGNN, dict[str, Any]]:
    raw = torch.load(Path(path), map_location=device, weights_only=False)
    meta = dict(raw.get("meta") or {})
    hidden = int(meta.get("hidden", 32))
    model = ClotBandPhiGNN(in_dim=step7_feature_dim(), hidden=hidden).to(device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()
    return model, meta


def resolve_step7_rule_cfg(step0_json: str | Path) -> TemporalGrowthRuleConfig:
    return load_step0_coef_json(step0_json).to_rule_config(name="ml_step0_coef")
