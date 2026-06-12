"""Pivot B: learned mixture over shear-risk rule-family experts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_localized_spatial import LocalizedSpatialConfig
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _resolve_pool_risk,
    _time_frac_at_index,
    predict_phi_temporal_at_time,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_pivot_common import eval_phi_rollout_on_anchor, pivot_out_dir
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step2_band_gnn import (
    ClotBandRiskGNN,
    band_features_pred_kine,
    step2_feature_dim,
)

# Compact expert family (subset of shear_risk_rule_grid).
EXPERT_SPECS: tuple[tuple[str, float, float, float, float, str], ...] = (
    ("base", 0.25, 0.0, 0.0, 0.0, ""),
    ("neg55", 0.55, 0.15, 0.15, 0.15, ""),
    ("neg70", 0.70, 0.10, 0.10, 0.10, ""),
    ("sep40", 0.25, 0.40, 0.20, 0.15, ""),
    ("stag40", 0.20, 0.15, 0.40, 0.25, ""),
    ("lgrad35", 0.25, 0.15, 0.25, 0.35, ""),
    ("combo", 0.40, 0.25, 0.20, 0.15, ""),
    ("auto", 0.30, 0.20, 0.30, 0.20, "auto"),
)


def n_rule_experts() -> int:
    return len(EXPERT_SPECS)


def rule_cfg_for_expert(
    base: TemporalGrowthRuleConfig,
    spec: tuple[str, float, float, float, float, str],
) -> TemporalGrowthRuleConfig:
    tag, ndx, sep, stag, lgrad, sz = spec
    loc = base.localized
    if loc is None:
        raise ValueError("rule mixture requires localized config")
    loc = replace(
        loc,
        neg_dx_risk_weight=float(ndx),
        sep_stream_risk_weight=float(sep),
        stasis_risk_weight=float(stag),
        low_grad_risk_weight=float(lgrad),
        aneurysm_size_mode=str(sz),
    )
    return replace(base, name=f"mix_{tag}", localized=loc)


def stack_expert_risks(
    data,
    base_cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    ceiling: torch.Tensor,
    t_out: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pool, risks) with shape [n_experts, n_nodes]."""
    pool: torch.Tensor | None = None
    rows: list[torch.Tensor] = []
    for spec in EXPERT_SPECS:
        cfg_e = rule_cfg_for_expert(base_cfg, spec)
        pool_e, risk_e = _resolve_pool_risk(
            data,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            cfg=cfg_e,
            t_out=int(t_out),
        )
        if pool is None:
            pool = pool_e
        rows.append(risk_e.reshape(-1))
    assert pool is not None
    return pool, torch.stack(rows, dim=0)


class ClotRuleMixtureModel(nn.Module):
    """GNN pools band features -> softmax weights over rule-family expert risks."""

    def __init__(self, in_dim: int = 3, hidden: int = 32, n_experts: int | None = None):
        super().__init__()
        self.n_experts = int(n_experts or n_rule_experts())
        self.encoder = ClotBandRiskGNN(in_dim=in_dim, hidden=hidden)
        self.weight_head = nn.Linear(max(hidden, 8), self.n_experts)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def expert_weights(
        self,
        data,
        pool: torch.Tensor,
        feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder.forward_hidden(feats, edge_index)
        pool_f = pool.reshape(-1).float()
        if bool(pool.any().item()):
            h_pool = (h * pool_f.unsqueeze(1)).sum(dim=0) / pool_f.sum().clamp(min=1.0)
        else:
            h_pool = h.mean(dim=0)
        return F.softmax(self.weight_head(h_pool), dim=0)

    def mixed_risk(
        self,
        expert_risks: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return (weights.unsqueeze(1) * expert_risks).sum(dim=0)


def rollout_rule_mixture_phi(
    data,
    rule_cfg: TemporalGrowthRuleConfig,
    model: ClotRuleMixtureModel,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict[int, torch.Tensor]:
    data = data.to(device)
    n_times = int(data.y.shape[0])
    t_final = n_times - 1
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    feats = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        flow_time=rule_cfg.risk_flow_time,
    )
    edge_index = data.edge_index.to(device)
    pool0, _ = stack_expert_risks(
        data, rule_cfg, device=device, bio_cfg=bio_cfg, ceiling=ceiling, t_out=0
    )
    weights = model.expert_weights(data, pool0, feats, edge_index)

    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    for t_out in range(n_times):
        pool, expert_risks = stack_expert_risks(
            data,
            rule_cfg,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            t_out=t_out,
        )
        risk = model.mixed_risk(expert_risks, weights)
        phi = predict_phi_temporal_at_time(
            data,
            t_out,
            device=device,
            bio_cfg=bio_cfg,
            cfg=rule_cfg,
            ceiling=ceiling,
            risk=risk,
            phi_prev=phi_prev,
            t_final=t_final,
            use_provided_risk=True,
        )
        phi_by_t[int(t_out)] = phi
        phi_prev = phi
    return phi_by_t


@dataclass
class PivotMixtureTrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 40
    early_paint_weight: float = 0.20
    final_rank_weight: float = 2.0


def _pairwise_rank_loss(
    risk: torch.Tensor,
    phi_gt: torch.Tensor,
    pool: torch.Tensor,
    *,
    margin: float = 0.05,
    max_pairs: int = 384,
) -> torch.Tensor:
    idx = pool.reshape(-1).bool().nonzero(as_tuple=True)[0]
    if int(idx.numel()) < 2:
        return risk.sum() * 0.0
    y = phi_gt[idx].reshape(-1)
    pos = y > 0.35
    neg = y < 0.08
    if not bool(pos.any().item()) or not bool(neg.any().item()):
        return F.mse_loss(risk[idx], y.clamp(0.0, 1.0))
    rp = risk[idx[pos]]
    rn = risk[idx[neg]]
    n_sample = min(max_pairs, max(int(rp.numel()) * int(rn.numel()), 1))
    gp = torch.randint(0, int(rp.numel()), (n_sample,), device=risk.device)
    gn = torch.randint(0, int(rn.numel()), (n_sample,), device=risk.device)
    return F.relu(float(margin) + rn[gn] - rp[gp]).mean()


def train_one_graph(
    model: ClotRuleMixtureModel,
    data,
    rule_cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    early_paint_weight: float,
    final_rank_weight: float,
) -> torch.Tensor:
    reset_temporal_kinematics_cache()
    data = data.to(device)
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return torch.tensor(0.0, device=device)
    t_final = int(pairs[-1][1])
    feats = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        flow_time=rule_cfg.risk_flow_time,
    )
    edge_index = data.edge_index.to(device)
    pool0, _ = stack_expert_risks(
        data, rule_cfg, device=device, bio_cfg=bio_cfg, ceiling=ceiling, t_out=0
    )
    weights = model.expert_weights(data, pool0, feats, edge_index)
    total = torch.tensor(0.0, device=device)
    n = 0
    for t_in, t_out in pairs:
        pool, expert_risks = stack_expert_risks(
            data,
            rule_cfg,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            t_out=int(t_out),
        )
        risk = model.mixed_risk(expert_risks, weights)
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        phi_gt = step.phi_gt.reshape(-1).to(risk.dtype)
        w = final_rank_weight if int(t_out) == t_final else 1.0
        loss = w * _pairwise_rank_loss(risk, phi_gt, pool)
        t_frac = _time_frac_at_index(data, int(t_out))
        if t_frac <= 0.35 and bool(pool.any().item()):
            loss = loss + early_paint_weight * risk[pool].mean()
        total = total + loss
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_rule_mixture_on_anchor(
    model: ClotRuleMixtureModel,
    rule_cfg: TemporalGrowthRuleConfig,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
) -> dict[str, Any]:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    feats = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        flow_time=rule_cfg.risk_flow_time,
    )
    pool0, _ = stack_expert_risks(
        data, rule_cfg, device=device, bio_cfg=bio_cfg, ceiling=ceiling, t_out=0
    )
    w = model.expert_weights(data, pool0, feats, data.edge_index.to(device))
    phi_by_t = rollout_rule_mixture_phi(
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
        rule_tag="pivot_rule_mixture",
        pair_stride=pair_stride,
        extra={
            "expert_weights": {EXPERT_SPECS[i][0]: float(w[i].item()) for i in range(len(EXPERT_SPECS))},
        },
    )


def default_mixture_out_dir() -> Path:
    return pivot_out_dir("rule_mixture")


def resolve_mixture_rule_cfg(step0_json: str | Path) -> TemporalGrowthRuleConfig:
    return load_step0_coef_json(step0_json).to_rule_config(name="ml_step0_coef")


def build_rule_mixture_model(meta: dict[str, Any]) -> ClotRuleMixtureModel:
    return ClotRuleMixtureModel(
        in_dim=step2_feature_dim(),
        hidden=int(meta.get("hidden", 32)),
        n_experts=int(meta.get("n_experts", n_rule_experts())),
    )
