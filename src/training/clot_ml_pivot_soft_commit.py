"""Pivot A: differentiable soft commit (no hard top-k) + optional GNN risk residual."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_localized_spatial import LocalizedSpatialConfig, active_wall_halves
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _growth_time_frac,
    _progressive_frac_from_growth_u,
    _resolve_pool_risk,
    _time_frac_at_index,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_pivot_common import eval_phi_rollout_on_anchor, pivot_out_dir
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step2_band_gnn import (
    ClotBandRiskGNN,
    band_features_pred_kine,
    combine_gnn_risk,
    step2_feature_dim,
)


def soft_commit_phi_at_time(
    risk: torch.Tensor,
    pool: torch.Tensor,
    ceiling: torch.Tensor,
    data,
    *,
    device: torch.device,
    loc: LocalizedSpatialConfig,
    cfg: TemporalGrowthRuleConfig,
    t_out: int,
    t_final: int,
    onset_frac: float,
    temperature: float,
    phi_prev: torch.Tensor | None,
) -> torch.Tensor:
    """Soft mass on pool per wall half; monotonic carry; ceiling projection."""
    n = int(risk.numel())
    t_frac = _time_frac_at_index(data, int(t_out))
    if float(onset_frac) > 0 and t_frac < float(onset_frac):
        return torch.zeros(n, device=device, dtype=risk.dtype)

    u_grow = _growth_time_frac(t_frac, float(onset_frac))
    frac = min(_progressive_frac_from_growth_u(cfg, u_grow), 0.95)
    temp = max(float(temperature), 1e-4)
    out = torch.zeros(n, device=device, dtype=risk.dtype)

    for half in active_wall_halves(data, device, loc):
        seg = pool.reshape(-1).bool() & half
        if not bool(seg.any().item()):
            continue
        r = risk.reshape(-1)[seg]
        z = torch.sigmoid((r - r.median()) / temp)
        scale = float(frac) / z.mean().clamp(min=1e-6)
        out[seg] = (z * scale).clamp(0.0, 1.0)

    out = out * ceiling.reshape(-1).float()
    if phi_prev is not None:
        out = torch.maximum(out, phi_prev.reshape(-1).to(out.dtype))
    return out


class ClotSoftCommitModel(nn.Module):
    """Hand risk + GNN residual; learnable soft-commit temperature."""

    def __init__(self, in_dim: int = 3, hidden: int = 32, *, init_temp: float = 0.08):
        super().__init__()
        self.gnn = ClotBandRiskGNN(in_dim=in_dim, hidden=hidden)
        self.log_temp = nn.Parameter(torch.tensor(float(init_temp)).log())
        self.delta_scale = 0.30

    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp().clamp(min=1e-3, max=0.5)

    def risk_logits(
        self,
        data,
        rule_cfg: TemporalGrowthRuleConfig,
        *,
        device: torch.device,
        phys_cfg: PhysicsConfig,
        bio_cfg: BiochemConfig,
        ceiling: torch.Tensor,
        t_out: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pool, hand = _resolve_pool_risk(
            data,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            cfg=rule_cfg,
            t_out=int(t_out),
        )
        feats = band_features_pred_kine(
            data,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            flow_time=rule_cfg.risk_flow_time,
        )
        logits = self.gnn.forward_logits(feats, data.edge_index.to(device))
        risk = combine_gnn_risk(
            hand,
            logits,
            pool,
            data,
            rule_cfg,
            device=device,
            delta_scale=self.delta_scale,
        )
        return pool, risk


def rollout_soft_commit_phi(
    data,
    rule_cfg: TemporalGrowthRuleConfig,
    model: ClotSoftCommitModel,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    onset_frac: float | None = None,
) -> dict[int, torch.Tensor]:
    data = data.to(device)
    loc = rule_cfg.localized
    if loc is None:
        raise ValueError("soft commit pivot requires localized rule config")
    n_times = int(data.y.shape[0])
    t_final = n_times - 1
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    onset = float(onset_frac if onset_frac is not None else rule_cfg.global_onset_frac)
    temp = float(model.temperature().item())
    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    for t_out in range(n_times):
        pool, risk = model.risk_logits(
            data,
            rule_cfg,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            t_out=t_out,
        )
        phi = soft_commit_phi_at_time(
            risk,
            pool,
            ceiling,
            data,
            device=device,
            loc=loc,
            cfg=rule_cfg,
            t_out=t_out,
            t_final=t_final,
            onset_frac=onset,
            temperature=temp,
            phi_prev=phi_prev,
        )
        phi_by_t[int(t_out)] = phi
        phi_prev = phi.detach() if model.training else phi  # truncated BPTT
    return phi_by_t


@dataclass
class PivotSoftTrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 40
    early_paint_weight: float = 0.25
    final_bce_weight: float = 2.0


def train_one_graph(
    model: ClotSoftCommitModel,
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
    phi_by_t = rollout_soft_commit_phi(
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
def eval_soft_commit_on_anchor(
    model: ClotSoftCommitModel,
    rule_cfg: TemporalGrowthRuleConfig,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
) -> dict[str, Any]:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    phi_by_t = rollout_soft_commit_phi(
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
        rule_tag="pivot_soft_commit",
        pair_stride=pair_stride,
        extra={"temperature": float(model.temperature().item())},
    )


def default_soft_out_dir() -> Path:
    return pivot_out_dir("soft_commit")


def resolve_soft_rule_cfg(step0_json: str | Path) -> TemporalGrowthRuleConfig:
    return load_step0_coef_json(step0_json).to_rule_config(name="ml_step0_coef")


def build_soft_commit_model(meta: dict[str, Any]) -> ClotSoftCommitModel:
    return ClotSoftCommitModel(
        in_dim=step2_feature_dim(),
        hidden=int(meta.get("hidden", 32)),
        init_temp=float(meta.get("init_temp", 0.08)),
    )
