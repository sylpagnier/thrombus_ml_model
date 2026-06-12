"""V3.2: GNN ranks commits inside frozen progressive rule budget + nucleation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import feature_time_index, rollout_time_indices
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import pred_clot_mask
from src.core_physics.clot_nucleation_mask import (
    project_phi_with_nucleation,
    resolve_catalytic_hood,
    resolve_nucleation_eligibility,
    resolve_wall_mask,
)
from src.core_physics.clot_temporal_growth_rules import (
    _time_frac_at_index,
    reset_temporal_kinematics_cache,
)
from src.training.clot_growth_eval import eval_phi_trajectory_on_anchor
from src.training.clot_growth_loss import (
    ClotGrowthLossConfig,
    clot_growth_frame_loss,
    hop_penalty_from_gt_clot,
    temporal_frame_weight,
)
from src.training.clot_ml_step1_residual import rollout_frozen_rule_phi, resolve_step1_rule_cfg
from src.training.clot_ml_v2_growth_gnn import (
    ClotGrowthRateGNN,
    apply_step3_v3_env,
    growth_gnn_feature_dim,
    growth_gnn_features,
    hard_commit_for_recipe,
    rollout_v3_growth_gnn,
    teacher_phi_by_t_from_step1,
    train_one_graph_v31,
)
from src.utils.paths import get_project_root


class _RankGNNConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))


class ClotGrowthRankGNN(nn.Module):
    """Score eligible nodes for this step's rule increment budget."""

    def __init__(self, in_dim: int = 8, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 8)
        self.conv1 = _RankGNNConv(in_dim, h)
        self.conv2 = _RankGNNConv(h, h)
        self.head = nn.Linear(h, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def forward_logits(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x, edge_index))
        h = F.silu(self.conv2(h, edge_index))
        return self.head(h).squeeze(-1)


def apply_v32_env() -> None:
    apply_step3_v3_env(v31=True)
    os.environ["CLOT_V32_RECIPE"] = "1"
    os.environ.setdefault("CLOT_V31_HARD_COMMIT", "1")


def _rule_increment(
    phi_rule: torch.Tensor,
    phi_prev: torch.Tensor | None,
) -> torch.Tensor:
    prev = (
        phi_prev.reshape(-1).float()
        if phi_prev is not None
        else torch.zeros_like(phi_rule.reshape(-1).float())
    )
    return (phi_rule.reshape(-1).float() - prev).clamp(min=0.0)


def _soft_topk_mask(
    scores: torch.Tensor,
    candidate: torch.Tensor,
    k: int,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Differentiable soft top-k over candidate nodes."""
    n = int(scores.numel())
    out = torch.zeros(n, device=scores.device, dtype=scores.dtype)
    if k <= 0 or not bool(candidate.any().item()):
        return out
    k = min(int(k), int(candidate.sum().item()))
    s = scores.masked_fill(~candidate.reshape(-1).bool(), -1e4)
    w = torch.softmax(s / max(float(temperature), 0.05), dim=0)
    cand_idx = torch.where(candidate.reshape(-1).bool())[0]
    top_idx = cand_idx[torch.topk(w[cand_idx], k=k).indices]
    out[top_idx] = 1.0
    return out


def rollout_v32_ranker(
    model: ClotGrowthRankGNN,
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    commit_thresh: float = 0.5,
    soft_topk_temp: float = 0.5,
) -> dict[int, torch.Tensor]:
    apply_v32_env()
    data = data.to(device)
    edge_index = data.edge_index.to(device)
    n = int(data.num_nodes)
    hard_commit = hard_commit_for_recipe()
    phi_rule_by_t = rollout_frozen_rule_phi(
        data, rule_cfg, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg, sim_end_scale=1.0
    )
    out: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    for t_out in sorted(phi_rule_by_t.keys()):
        t_out = int(t_out)
        if t_out == 0:
            phi0 = torch.zeros(n, device=device, dtype=torch.float32)
            out[t_out] = phi0
            phi_prev = phi0
            continue
        phi_rule = phi_rule_by_t[t_out].reshape(-1).float()
        rule_inc = _rule_increment(phi_rule, phi_prev)
        elig = resolve_nucleation_eligibility(
            data, t_out, device, phys_cfg, bio_cfg, growth_seed="pred", phi_pred_by_time=out
        )
        k = max(int((rule_inc > 0.05).sum().item()), 1 if rule_inc.max() > 0.1 else 0)
        cand = elig & (rule_inc > 1e-4)
        if not bool(cand.any().item()):
            cand = elig
        t_feat = feature_time_index(data, t_out)
        commits = (
            pred_clot_mask(phi_prev, thresh=commit_thresh)
            if phi_prev is not None
            else torch.zeros(n, device=device, dtype=torch.bool)
        )
        hood = resolve_catalytic_hood(commits, edge_index)
        feats = growth_gnn_features(
            data,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            phi_prev=phi_prev,
            catalytic_hood=hood,
            t_out=t_out,
            flow_time=t_feat,
        )
        logits = model.forward_logits(feats, edge_index)
        assert phi_prev is not None
        if model.training:
            pick = _soft_topk_mask(logits, cand, k, temperature=soft_topk_temp)
            gate = torch.sigmoid(logits) * pick
            phi_prop = torch.maximum(phi_prev, phi_prev + elig.float() * rule_inc * gate)
        else:
            phi_prop = phi_prev.clone()
            if k > 0 and bool(cand.any().item()):
                scores = logits.masked_fill(~cand, float("-inf"))
                topk = torch.topk(scores, k=min(k, int(cand.sum().item())))
                phi_prop[topk.indices] = torch.maximum(
                    phi_rule[topk.indices], phi_prev[topk.indices]
                )
        phi = project_phi_with_nucleation(
            phi_prop, phi_prev, elig, commit_thresh=commit_thresh, hard_commit=hard_commit
        )
        out[t_out] = phi
        phi_prev = phi
    return out


@dataclass
class V32SweepLegConfig:
    name: str
    arch: str = "ranker"  # ranker | euler
    epochs: int = 6
    lr: float = 8e-4
    onset_weight: float = 1.0
    ring_weight: float = 0.4
    temporal_equal: bool = True
    init_ckpt: str = ""
    teacher_weight: float = 0.05


def train_one_graph_leg(
    model: nn.Module,
    data,
    rule_cfg,
    *,
    leg: V32SweepLegConfig,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    teacher_phi_by_t: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    loss_cfg = ClotGrowthLossConfig(
        onset_weight=leg.onset_weight,
        ring_weight=leg.ring_weight,
        temporal_equal=leg.temporal_equal,
    )
    if leg.arch == "euler":
        return train_one_graph_v31(
            model,
            data,
            rule_cfg,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            loss_cfg=loss_cfg,
            teacher_phi_by_t=teacher_phi_by_t,
            teacher_weight=leg.teacher_weight,
            final_frame_boost=1.2,
        )
    reset_temporal_kinematics_cache()
    data = data.to(device)
    assert isinstance(model, ClotGrowthRankGNN)
    phi_by_t = rollout_v32_ranker(
        model, data, rule_cfg, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
    )
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return phi_by_t[int(min(phi_by_t.keys()))].sum() * 0.0
    wall = resolve_wall_mask(data, device)
    edge_index = data.edge_index
    n_nodes = int(data.num_nodes)
    total = torch.tensor(0.0, device=device)
    wsum = 0.0
    for t_in, t_out in pairs:
        phi = phi_by_t[int(t_out)]
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        mask = step.loss_mask.reshape(-1).bool()
        if not bool(mask.any().item()):
            continue
        target = step.phi_gt.reshape(-1).to(phi.dtype)
        gt_bin = (target >= 0.5).detach().cpu().numpy()
        hop_np = hop_penalty_from_gt_clot(edge_index, n_nodes, gt_bin)
        hop_t = torch.from_numpy(hop_np).to(device=device, dtype=phi.dtype)
        parts = clot_growth_frame_loss(
            phi, target, mask, hop_t, cfg=loss_cfg, wall_mask=wall
        )
        t_frac = _time_frac_at_index(data, int(t_out))
        fw = temporal_frame_weight(t_frac, equal=leg.temporal_equal)
        loss = parts["loss"]
        if teacher_phi_by_t is not None and leg.teacher_weight > 0 and int(t_out) in teacher_phi_by_t:
            teach = teacher_phi_by_t[int(t_out)].reshape(-1).to(device=device, dtype=phi.dtype)
            loss = loss + float(leg.teacher_weight) * F.mse_loss(phi[mask], teach[mask])
        total = total + fw * loss
        wsum += fw
    return total / max(wsum, 1e-6)


@torch.no_grad()
def eval_leg_on_anchor(
    model: nn.Module,
    rule_cfg,
    *,
    graph_path: Path,
    leg: V32SweepLegConfig,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    if leg.arch == "euler":
        assert isinstance(model, ClotGrowthRateGNN)
        phi_by_t = rollout_v3_growth_gnn(
            model, data, rule_cfg, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
        )
        tag = "v32_euler"
    else:
        assert isinstance(model, ClotGrowthRankGNN)
        phi_by_t = rollout_v32_ranker(
            model, data, rule_cfg, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
        )
        tag = "v32_ranker"
    return eval_phi_trajectory_on_anchor(
        phi_by_t,
        data,
        anchor=graph_path.stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag=tag,
    )


def default_v32_sweep_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder_v2/v32_sweep_30m"


def build_model_for_leg(leg: V32SweepLegConfig, *, device: torch.device) -> nn.Module:
    if leg.arch == "euler":
        return ClotGrowthRateGNN(in_dim=growth_gnn_feature_dim(), hidden=32).to(device)
    return ClotGrowthRankGNN(in_dim=growth_gnn_feature_dim(), hidden=32).to(device)


def save_leg_checkpoint(path: Path, *, model: nn.Module, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_leg_checkpoint(
    path: Path,
    leg: V32SweepLegConfig,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    raw = torch.load(path, map_location=device, weights_only=False)
    meta = dict(raw.get("meta") or {})
    model = build_model_for_leg(leg, device=device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()
    return model, meta


def resolve_rule_cfg(step0_json: str | Path) -> Any:
    return resolve_step1_rule_cfg(get_project_root() / step0_json)
