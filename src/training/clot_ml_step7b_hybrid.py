"""Step 7b ML ladder: MLP residual on frozen pivot-B rule_mixture phi.

Starts at rule_mixture deploy (~0.516 mean) via zero-init residual; BCE fine-tunes commits
without bare e2e sigmoid collapse (Step 7 / pivot A failure mode).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _time_frac_at_index,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_pivot_common import eval_phi_rollout_on_anchor, load_pivot_checkpoint
from src.training.clot_ml_pivot_rule_mixture import (
    ClotRuleMixtureModel,
    build_rule_mixture_model,
    rollout_rule_mixture_phi,
)
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step1_residual import (
    ClotRuleResidualMLP,
    _pair_loss,
    band_features_pred_kine,
    combine_rule_residual,
    step1_feature_dim,
)
from src.utils.paths import get_project_root


def step7b_feature_dim() -> int:
    return step1_feature_dim()


@torch.no_grad()
def rollout_frozen_mixture_phi(
    data,
    rule_cfg: TemporalGrowthRuleConfig,
    mixture: ClotRuleMixtureModel,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict[int, torch.Tensor]:
    reset_temporal_kinematics_cache()
    mixture.eval()
    return rollout_rule_mixture_phi(
        data,
        rule_cfg,
        mixture,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
    )


def apply_step7b_phi(
    residual: ClotRuleResidualMLP,
    data,
    phi_base_by_t: dict[int, torch.Tensor],
    t_out: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    ceiling: torch.Tensor,
    alpha: float,
    flow_time: int = 0,
) -> torch.Tensor:
    phi_base = phi_base_by_t[int(t_out)].to(device=device)
    feats = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        phi_rule=phi_base,
        flow_time=flow_time,
    )
    delta = residual.forward_logits(feats)
    return combine_rule_residual(phi_base, delta, ceiling, alpha=alpha)


def rollout_step7b_phi(
    data,
    rule_cfg: TemporalGrowthRuleConfig,
    mixture: ClotRuleMixtureModel,
    residual: ClotRuleResidualMLP,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float = 0.35,
    flow_time: int = 0,
) -> dict[int, torch.Tensor]:
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    phi_base_by_t = rollout_frozen_mixture_phi(
        data,
        rule_cfg,
        mixture,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
    )
    out: dict[int, torch.Tensor] = {}
    for t_out in sorted(phi_base_by_t.keys()):
        out[int(t_out)] = apply_step7b_phi(
            residual,
            data,
            phi_base_by_t,
            int(t_out),
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            alpha=alpha,
            flow_time=flow_time,
        )
    return out


@dataclass
class Step7bTrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    mixture_ckpt: str = (
        "outputs/biochem/clot_ml_ladder/pivot_rule_mixture/clot_ml_pivot_rule_mixture_best.pth"
    )
    alpha: float = 0.35
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 40
    early_paint_weight: float = 0.25
    final_bce_weight: float = 2.0


def load_frozen_mixture(
    ckpt_path: str | Path,
    *,
    device: torch.device,
) -> tuple[ClotRuleMixtureModel, dict[str, Any]]:
    return load_pivot_checkpoint(
        ckpt_path,
        device=device,
        model_ctor=build_rule_mixture_model,
    )


def train_one_graph(
    residual: ClotRuleResidualMLP,
    mixture: ClotRuleMixtureModel,
    data,
    rule_cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
    early_paint_weight: float,
    final_bce_weight: float,
) -> torch.Tensor:
    reset_temporal_kinematics_cache()
    data = data.to(device)
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    with torch.no_grad():
        phi_base_by_t = rollout_frozen_mixture_phi(
            data,
            rule_cfg,
            mixture,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
        )
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return torch.tensor(0.0, device=device)
    t_final = int(pairs[-1][1])
    total = torch.tensor(0.0, device=device)
    n = 0
    for t_in, t_out in pairs:
        phi_base = phi_base_by_t[int(t_out)]
        feats = band_features_pred_kine(
            data,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            phi_rule=phi_base,
            flow_time=rule_cfg.risk_flow_time,
        )
        delta = residual.forward_logits(feats)
        phi_pred = combine_rule_residual(phi_base, delta, ceiling, alpha=alpha)
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        t_frac = _time_frac_at_index(data, int(t_out))
        total = total + _pair_loss(
            phi_pred,
            step,
            t_frac=t_frac,
            early_paint_weight=early_paint_weight,
            final_bce_weight=final_bce_weight,
            is_final=int(t_out) == t_final,
        )
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def eval_step7b_on_anchor(
    residual: ClotRuleResidualMLP,
    mixture: ClotRuleMixtureModel,
    rule_cfg: TemporalGrowthRuleConfig,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
    pair_stride: int = 1,
) -> dict[str, Any]:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    phi_by_t = rollout_step7b_phi(
        data,
        rule_cfg,
        mixture,
        residual,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        flow_time=rule_cfg.risk_flow_time,
    )
    return eval_phi_rollout_on_anchor(
        phi_by_t,
        graph_path=graph_path,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="ml_step7b_hybrid",
        pair_stride=pair_stride,
    )


def default_step7b_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder/step7b_hybrid"


def save_step7b_checkpoint(
    path: Path,
    *,
    residual: ClotRuleResidualMLP,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": residual.state_dict(), "meta": meta}, path)


def load_step7b_checkpoint(
    path: Path | str,
    *,
    device: torch.device,
) -> tuple[ClotRuleMixtureModel, ClotRuleResidualMLP, dict[str, Any]]:
    raw = torch.load(Path(path), map_location=device, weights_only=False)
    meta = dict(raw.get("meta") or {})
    mixture_path = str(
        meta.get("mixture_ckpt")
        or "outputs/biochem/clot_ml_ladder/pivot_rule_mixture/clot_ml_pivot_rule_mixture_best.pth"
    )
    mixture, _ = load_frozen_mixture(mixture_path, device=device)
    hidden = int(meta.get("hidden", 32))
    residual = ClotRuleResidualMLP(in_dim=step7b_feature_dim(), hidden=hidden).to(device)
    residual.load_state_dict(raw["model"], strict=True)
    residual.eval()
    mixture.eval()
    return mixture, residual, meta


def resolve_step7b_rule_cfg(step0_json: str | Path) -> TemporalGrowthRuleConfig:
    return load_step0_coef_json(step0_json).to_rule_config(name="ml_step0_coef")
