"""Step 3b: finetune step1 shell with in-window GT + unsupervised extrap growth prior."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import (
    comsol_final_index,
    extrapolated_t_out_max,
    macro_tau_at_index,
    rollout_time_indices,
)
from src.core_physics.clot_growth_masks import resolve_ceiling_mask
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step1_residual import (
    ClotRuleResidualMLP,
    apply_step1_eval_env,
    eval_step1_on_anchor,
    load_step1_checkpoint,
    resolve_step1_rule_cfg,
    rollout_step1_phi,
    save_step1_checkpoint,
    train_one_graph,
)
from src.utils.paths import get_project_root


def apply_step3b_train_env(*, sim_end_scale: float = 2.0) -> None:
    apply_step1_eval_env()
    os.environ["CLOT_ML_USE_MACRO_TAU"] = "1"
    os.environ["CLOT_ML_CONTINUOUS_EXTRAP"] = "1"
    os.environ["CLOT_ML_SIM_END_SCALE"] = str(float(sim_end_scale))


@dataclass
class Step3bTrainConfig:
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    init_step1_ckpt: str = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth"
    alpha: float = 0.35
    hidden: int = 32
    lr: float = 5e-4
    epochs: int = 24
    sim_end_scale: float = 2.0
    extrap_weight: float = 0.15
    slow_growth_weight: float = 0.05
    paint_cap_frac: float = 0.15
    max_pred_frac_delta: float = 0.20


def default_step3b_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder/step3b_extrap"


def _commit_frac_on_ceiling(phi: torch.Tensor, ceiling: torch.Tensor) -> torch.Tensor:
    on = phi.reshape(-1).float() * ceiling.reshape(-1).float()
    return (on > 0.5).float().mean()


def extrap_unsup_loss(
    phi_by_t: dict[int, torch.Tensor],
    data,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    sim_end_scale: float,
    slow_growth_weight: float,
    paint_cap_frac: float,
    max_pred_frac_delta: float,
) -> torch.Tensor:
    """Unsupervised axis-C prior beyond COMSOL labels."""
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    t_final = comsol_final_index(data)
    t_max = extrapolated_t_out_max(data, sim_end_scale=sim_end_scale)
    t_indices = [t for t in rollout_time_indices(data, sim_end_scale=sim_end_scale) if t in phi_by_t]
    if len(t_indices) < 2:
        return torch.tensor(0.0, device=device)

    fracs: list[torch.Tensor] = []
    taus: list[float] = []
    for t in t_indices:
        phi = phi_by_t[int(t)].to(device)
        fracs.append(_commit_frac_on_ceiling(phi, ceiling))
        taus.append(macro_tau_at_index(data, int(t), bio_cfg=bio_cfg))

    loss = torch.tensor(0.0, device=device)
    mono_viol = torch.tensor(0.0, device=device)
    for i in range(1, len(fracs)):
        drop = (fracs[i - 1] - fracs[i]).clamp(min=0.0)
        mono_viol = mono_viol + drop

    in_window = [i for i, t in enumerate(t_indices) if int(t) <= t_final]
    frac_in = fracs[in_window[-1]] if in_window else fracs[0]
    frac_ex = fracs[-1]
    cap = float(frac_in.detach().item()) + float(paint_cap_frac)
    over_paint = (frac_ex - cap).clamp(min=0.0)

    slow = torch.tensor(0.0, device=device)
    for i in range(1, len(fracs)):
        dtau = max(taus[i] - taus[i - 1], 1e-6)
        rate = (fracs[i] - fracs[i - 1]) / dtau
        slow = slow + F.relu(rate - float(max_pred_frac_delta))

    loss = loss + mono_viol + over_paint + float(slow_growth_weight) * slow
    return loss


def train_one_graph_step3b(
    model: ClotRuleResidualMLP,
    data,
    rule_cfg,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
    cfg: Step3bTrainConfig,
    early_paint_weight: float,
    final_bce_weight: float,
) -> torch.Tensor:
    reset_temporal_kinematics_cache()
    loss_in = train_one_graph(
        model,
        data,
        rule_cfg,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        early_paint_weight=early_paint_weight,
        final_bce_weight=final_bce_weight,
    )
    phi_by_t = rollout_step1_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        sim_end_scale=float(cfg.sim_end_scale),
    )
    loss_ex = extrap_unsup_loss(
        phi_by_t,
        data,
        device=device,
        bio_cfg=bio_cfg,
        sim_end_scale=float(cfg.sim_end_scale),
        slow_growth_weight=float(cfg.slow_growth_weight),
        paint_cap_frac=float(cfg.paint_cap_frac),
        max_pred_frac_delta=float(cfg.max_pred_frac_delta),
    )
    return loss_in + float(cfg.extrap_weight) * loss_ex


def eval_step3b_on_anchor(
    model: ClotRuleResidualMLP,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
    sim_end_scale: float,
    pair_stride: int = 1,
) -> dict[str, Any]:
    apply_step3b_train_env(sim_end_scale=sim_end_scale)
    row = eval_step1_on_anchor(
        model,
        rule_cfg,
        graph_path=graph_path,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        pair_stride=pair_stride,
    )
    data = torch.load(graph_path, map_location=device, weights_only=False)
    phi_by_t = rollout_step1_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        sim_end_scale=float(sim_end_scale),
    )
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    t_final = comsol_final_index(data)
    t_max = extrapolated_t_out_max(data, sim_end_scale=sim_end_scale)
    phi_in = phi_by_t.get(t_final, phi_by_t[max(phi_by_t.keys())])
    phi_ex = phi_by_t.get(t_max, phi_by_t[max(phi_by_t.keys())])
    row["pred_frac_tfinal"] = float(_commit_frac_on_ceiling(phi_in, ceiling).item())
    row["pred_frac_extrap"] = float(_commit_frac_on_ceiling(phi_ex, ceiling).item())
    row["extrap_growth_delta"] = float(row["pred_frac_extrap"] - row["pred_frac_tfinal"])
    return row


def load_step3b_checkpoint(path: Path, device: torch.device) -> tuple[ClotRuleResidualMLP, dict[str, Any]]:
    return load_step1_checkpoint(path, device=device)
