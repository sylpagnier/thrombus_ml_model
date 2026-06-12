"""Step 1 ML ladder: MLP residual on frozen Step-0 rule phi (pred kine)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_phi_simple import node_features_from_gt
from src.core_physics.clot_temporal_growth_rules import (
    _resolve_uv_for_temporal_risk,
    _shape_from_phi_at_time,
    _time_frac_at_index,
    deploy_score_from_eval_row,
    reset_temporal_kinematics_cache,
    rollout_temporal_phi,
)
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def step1_feature_dim() -> int:
    return 4  # minimal physics (3) + phi_rule


def apply_step1_eval_env() -> None:
    """Match ``train_clot_ml_step1_residual`` env (minimal 3-feature + pred kine)."""
    os.environ["BIOCHEM_PRIOR_COMSOL_ALIGNED"] = "1"
    os.environ["BIOCHEM_PRIOR_NORM_MASK"] = "adjacent"
    os.environ["CLOT_PHI_DGAMMA_SLICE"] = "1"
    os.environ["CLOT_PHI_CEILING_HOPS"] = "2"
    os.environ["CLOT_FORECAST_MASK"] = "ceiling_growth"
    os.environ["CLOT_PHI_MINIMAL_FEATURES"] = "1"
    os.environ.setdefault("CLOT_TEMPORAL_VEL_SOURCE", "kinematics")
    os.environ.setdefault("CLOT_PHI_KINE_CKPT", "outputs/kinematics/kinematics_best.pth")


class ClotRuleResidualMLP(nn.Module):
    """Per-node residual logit on ceiling band."""

    def __init__(self, in_dim: int = 4, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 8)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.25)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def combine_rule_residual(
    phi_rule: torch.Tensor,
    delta_logit: torch.Tensor,
    ceiling: torch.Tensor,
    *,
    alpha: float = 0.35,
) -> torch.Tensor:
    """Additive residual inside ceiling: phi = clip(rule + alpha*sigmoid(delta))."""
    base = phi_rule.reshape(-1).clamp(0.0, 1.0)
    delta = torch.sigmoid(delta_logit.reshape(-1))
    phi = (base + float(alpha) * delta).clamp(0.0, 1.0)
    return phi * ceiling.reshape(-1).float()


def band_features_pred_kine(
    data,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    phi_rule: torch.Tensor,
    flow_time: int = 0,
) -> torch.Tensor:
    u, v = _resolve_uv_for_temporal_risk(data, flow_time, device)
    ti = max(0, min(int(flow_time), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=device, dtype=torch.float32)
    feats = node_features_from_gt(
        data,
        y,
        phys_cfg,
        bio_cfg,
        device=device,
        time_index=ti,
        u_nd_override=u,
        v_nd_override=v,
    )
    return torch.cat([feats, phi_rule.reshape(-1, 1).to(device=device, dtype=feats.dtype)], dim=1)


@torch.no_grad()
def rollout_frozen_rule_phi(
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    sim_end_scale: float | None = None,
    coupled: bool = False,
) -> dict[int, torch.Tensor]:
    reset_temporal_kinematics_cache()
    if coupled:
        from src.core_physics.clot_coupled_rollout import (
            reset_coupled_uv_cache,
            rollout_temporal_phi_coupled,
        )

        reset_coupled_uv_cache()
        return rollout_temporal_phi_coupled(
            data,
            rule_cfg,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            sim_end_scale=sim_end_scale,
        )
    return rollout_temporal_phi(
        data,
        rule_cfg,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        sim_end_scale=sim_end_scale,
    )


def apply_step1_phi(
    model: ClotRuleResidualMLP,
    data,
    phi_rule_by_t: dict[int, torch.Tensor],
    t_out: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    ceiling: torch.Tensor,
    alpha: float,
    flow_time: int = 0,
) -> torch.Tensor:
    phi_rule = phi_rule_by_t[int(t_out)].to(device=device)
    feats = band_features_pred_kine(
        data,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        phi_rule=phi_rule,
        flow_time=flow_time,
    )
    delta = model.forward_logits(feats)
    return combine_rule_residual(phi_rule, delta, ceiling, alpha=alpha)


def rollout_step1_phi(
    data,
    rule_cfg,
    model: ClotRuleResidualMLP,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float = 0.35,
    flow_time: int = 0,
    sim_end_scale: float | None = None,
    coupled: bool = False,
) -> dict[int, torch.Tensor]:
    apply_step1_eval_env()
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    phi_rule_by_t = rollout_frozen_rule_phi(
        data,
        rule_cfg,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        sim_end_scale=sim_end_scale,
        coupled=coupled,
    )
    out: dict[int, torch.Tensor] = {}
    for t_out in sorted(phi_rule_by_t.keys()):
        out[int(t_out)] = apply_step1_phi(
            model,
            data,
            phi_rule_by_t,
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
class Step1TrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    alpha: float = 0.35
    hidden: int = 32
    lr: float = 1e-3
    epochs: int = 40
    early_paint_weight: float = 0.25
    final_bce_weight: float = 2.0


def _pair_loss(
    phi_pred: torch.Tensor,
    step,
    *,
    t_frac: float,
    early_paint_weight: float,
    final_bce_weight: float,
    is_final: bool,
) -> torch.Tensor:
    mask = step.loss_mask.reshape(-1).bool()
    if not bool(mask.any().item()):
        return phi_pred.sum() * 0.0
    target = step.phi_gt.reshape(-1).to(phi_pred.dtype)
    w = final_bce_weight if is_final else 1.0
    bce = F.binary_cross_entropy(phi_pred[mask], target[mask], reduction="mean")
    loss = w * bce
    if t_frac <= 0.35:
        loss = loss + early_paint_weight * phi_pred[mask].mean()
    return loss


def train_one_graph(
    model: ClotRuleResidualMLP,
    data,
    rule_cfg,
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
    phi_rule_by_t = rollout_frozen_rule_phi(
        data, rule_cfg, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
    )
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1)
    if not pairs:
        return torch.tensor(0.0, device=device)
    t_final = int(pairs[-1][1])
    total = torch.tensor(0.0, device=device)
    n = 0
    for t_in, t_out in pairs:
        phi_rule = phi_rule_by_t[int(t_out)]
        feats = band_features_pred_kine(
            data,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            phi_rule=phi_rule,
            flow_time=0,
        )
        delta = model.forward_logits(feats)
        phi_pred = combine_rule_residual(phi_rule, delta, ceiling, alpha=alpha)
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
def eval_step1_on_anchor(
    model: ClotRuleResidualMLP,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
    pair_stride: int = 1,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    stem = graph_path.stem
    phi_by_t = rollout_step1_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
    )
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1, pair_stride=pair_stride)
    if not pairs:
        return {"anchor": stem, "n_pairs": 0}

    rows: list[dict[str, float]] = []
    for t_in, t_out in pairs:
        phi = phi_by_t[int(t_out)]
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
        rows.append(
            {
                "t_frac": _time_frac_at_index(data, int(t_out)),
                **{k: float(v) for k, v in band.items()},
            }
        )

    tfinal = rows[-1]
    early = [r for r in rows if r["t_frac"] <= 0.35]
    t_final_idx = int(pairs[-1][1])
    t_in_final = int(pairs[-1][0])
    phi_final = phi_by_t[t_final_idx]
    shape_final = _shape_from_phi_at_time(
        data,
        phi_final,
        t_in_final,
        t_final_idx,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
    )

    def _mean(key: str, subset: list[dict]) -> float:
        if not subset:
            return float("nan")
        return float(sum(r[key] for r in subset) / len(subset))

    out = {
        "anchor": stem,
        "rule": "ml_step1_residual",
        "n_pairs": len(rows),
        "mean_band_f1": _mean("clot_f1", rows),
        "early_mean_pred_frac": _mean("pred_pos_frac", early),
        "early_mean_gt_pos_frac": _mean("gt_pos_frac", early),
        "tfinal_band_f1": float(tfinal["clot_f1"]),
        "tfinal_band_pred_frac": float(tfinal["pred_pos_frac"]),
        "tfinal_gt_pos_frac": float(tfinal["gt_pos_frac"]),
        "tfinal_clot_shape": float(shape_final.get("clot_shape", float("nan"))),
        "tfinal_clot_shape_bal": float(shape_final.get("clot_shape_balanced", float("nan"))),
    }
    out["deploy_score"] = deploy_score_from_eval_row(out)
    return out


def default_step1_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder/step1_residual"


def save_step1_checkpoint(
    path: Path,
    *,
    model: ClotRuleResidualMLP,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_step1_checkpoint(
    path: Path | str,
    *,
    device: torch.device,
) -> tuple[ClotRuleResidualMLP, dict[str, Any]]:
    apply_step1_eval_env()
    raw = torch.load(Path(path), map_location=device, weights_only=False)
    meta = dict(raw.get("meta") or {})
    hidden = int(meta.get("hidden", 32))
    model = ClotRuleResidualMLP(in_dim=step1_feature_dim(), hidden=hidden).to(device)
    model.load_state_dict(raw["model"], strict=True)
    model.eval()
    return model, meta


def resolve_step1_rule_cfg(step0_json: str | Path) -> Any:
    return load_step0_coef_json(step0_json).to_rule_config(name="ml_step0_coef")
