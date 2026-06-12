"""Deploy v1: frozen step1 phi + 5a mu + optional 5b coupled kine + 3b horizon."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_continuous_time import (
    extrapolated_t_out_max,
    macro_tau_at_index,
    rollout_time_indices,
)
from src.core_physics.clot_coupled_rollout import reset_coupled_uv_cache
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_temporal_growth_rules import (
    _shape_from_phi_at_time,
    deploy_score_from_eval_row,
    reset_temporal_kinematics_cache,
    temporal_vel_source,
)
from src.evaluation.clot_extrap_plausibility import compute_extrap_plausibility
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step1_residual import (
    apply_step1_eval_env,
    load_step1_checkpoint,
    resolve_step1_rule_cfg,
    rollout_step1_phi,
)
from src.training.clot_ml_step5a_mu_readout import mu_eff_carreau_blend_from_phi
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


@dataclass
class DeployV1Recipe:
    """Frozen ML deploy v1 manifest (no retrain Steps 0-7)."""

    name: str = "clot_ml_deploy_v1"
    phi_shell: str = "step1"
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    step1_ckpt: str = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth"
    kine_ckpt: str = "outputs/kinematics/kinematics_best.pth"
    vel_source: str = "kinematics"
    use_macro_tau: bool = True
    continuous_extrap: bool = False
    sim_end_scale: float = 1.0
    coupled: bool = False
    env: dict[str, str] = field(default_factory=dict)

    def apply_env(self) -> None:
        apply_step1_eval_env()
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "coupled" if self.coupled else self.vel_source
        os.environ["CLOT_PHI_KINE_CKPT"] = self.kine_ckpt
        os.environ["CLOT_ML_USE_MACRO_TAU"] = "1" if self.use_macro_tau else "0"
        os.environ["CLOT_ML_CONTINUOUS_EXTRAP"] = "1" if self.continuous_extrap else "0"
        os.environ["CLOT_ML_SIM_END_SCALE"] = str(self.sim_end_scale)
        for k, v in (self.env or {}).items():
            os.environ[k] = str(v)


def default_deploy_v1_recipe(*, coupled: bool = False, sim_end_scale: float = 1.0) -> DeployV1Recipe:
    return DeployV1Recipe(coupled=coupled, sim_end_scale=sim_end_scale)


def load_deploy_v1_recipe(path: str | Path | None = None) -> DeployV1Recipe:
    root = get_project_root()
    p = Path(path) if path else root / "data/reference/clot_ml_deploy_v1.json"
    if not p.is_file():
        return default_deploy_v1_recipe()
    raw = json.loads(p.read_text(encoding="utf-8"))
    known = {f.name for f in DeployV1Recipe.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in raw.items() if k in known}
    return DeployV1Recipe(**kwargs)


def save_deploy_v1_recipe(recipe: DeployV1Recipe, path: Path | None = None) -> Path:
    out = path or (get_project_root() / "data/reference/clot_ml_deploy_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(recipe), indent=2), encoding="utf-8")
    return out


@torch.no_grad()
def rollout_deploy_v1_phi(
    data,
    recipe: DeployV1Recipe,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    sim_end_scale: float | None = None,
    coupled: bool | None = None,
) -> dict[int, torch.Tensor]:
    """Roll out phi commits for deploy v1 (step1 shell)."""
    recipe.apply_env()
    reset_temporal_kinematics_cache()
    reset_coupled_uv_cache()
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    root = get_project_root()
    scale = float(sim_end_scale if sim_end_scale is not None else recipe.sim_end_scale)
    use_coupled = bool(recipe.coupled if coupled is None else coupled)

    if recipe.phi_shell == "inc40":
        rule_cfg = load_step0_coef_json(root / recipe.step0_json).to_rule_config(name="deploy_v1_inc40")
        from src.core_physics.clot_temporal_growth_rules import rollout_temporal_phi

        if use_coupled:
            from src.core_physics.clot_coupled_rollout import rollout_temporal_phi_coupled

            return rollout_temporal_phi_coupled(
                data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio, sim_end_scale=scale
            )
        return rollout_temporal_phi(
            data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio, sim_end_scale=scale
        )

    rule_cfg = resolve_step1_rule_cfg(root / recipe.step0_json)
    model, meta = load_step1_checkpoint(root / recipe.step1_ckpt, device=device)
    alpha = float(meta.get("alpha", 0.35))
    return rollout_step1_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys,
        bio_cfg=bio,
        alpha=alpha,
        sim_end_scale=scale,
        coupled=use_coupled,
    )


@torch.no_grad()
def rollout_deploy_v1_mu(
    data,
    phi_by_t: dict[int, torch.Tensor],
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict[int, torch.Tensor]:
    mu_by_t: dict[int, torch.Tensor] = {}
    for t_out, phi in phi_by_t.items():
        mu_by_t[int(t_out)] = mu_eff_carreau_blend_from_phi(
            data, phi, int(t_out), device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
        )
    return mu_by_t


@torch.no_grad()
def eval_deploy_v1_on_graph(
    data,
    recipe: DeployV1Recipe,
    *,
    device: torch.device,
    sim_end_scale: float | None = None,
    coupled: bool | None = None,
    anchor: str = "",
) -> dict[str, Any]:
    """In-window deploy metrics at COMSOL t_final (Step 5a/5c/11 in-window leg)."""
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    scale = float(sim_end_scale if sim_end_scale is not None else recipe.sim_end_scale)
    phi_by_t = rollout_deploy_v1_phi(
        data, recipe, device=device, sim_end_scale=scale, coupled=coupled
    )
    n = int(data.y.shape[0])
    pairs = iter_forecast_pairs(n, time_stride=1)
    if not pairs:
        return {"anchor": anchor, "n_pairs": 0}

    t_in_f, t_out_f = pairs[-1]
    phi_final = phi_by_t.get(int(t_out_f))
    if phi_final is None:
        in_keys = [k for k in phi_by_t if k < n]
        t_out_f = max(in_keys) if in_keys else max(phi_by_t)
        phi_final = phi_by_t[int(t_out_f)]
        t_in_f = max(0, int(t_out_f) - 1)

    step = build_clot_forecast_pair_step(data, int(t_in_f), int(t_out_f), phys, bio, device)
    band = _clot_metrics(phi_final, step.phi_gt, step.loss_mask)
    shape = _shape_from_phi_at_time(
        data, phi_final, int(t_in_f), int(t_out_f), device=device, phys_cfg=phys, bio_cfg=bio
    )
    gt_pos = float((step.phi_gt[step.loss_mask] > 0.5).float().mean().item()) if bool(
        step.loss_mask.any()
    ) else 0.0

    row: dict[str, Any] = {
        "anchor": anchor,
        "phi_shell": recipe.phi_shell,
        "vel_source": temporal_vel_source(),
        "sim_end_scale": scale,
        "coupled": bool(recipe.coupled if coupled is None else coupled),
        "n_comsol_steps": n,
        "extrap_t_out_max": extrapolated_t_out_max(data, sim_end_scale=scale),
        "n_phi_steps": len(phi_by_t),
        "tfinal_band_f1": float(band["clot_f1"]),
        "tfinal_band_pred_frac": float(band["pred_pos_frac"]),
        "tfinal_gt_pos_frac": gt_pos,
        "tfinal_clot_shape": float(shape.get("clot_shape", float("nan"))),
        "tfinal_clot_shape_bal": float(shape.get("clot_shape_balanced", float("nan"))),
        "deploy_score": 0.0,
    }
    row["deploy_score"] = float(deploy_score_from_eval_row(row))
    if scale > 1.0 + 1e-6:
        row["extrap"] = compute_extrap_plausibility(
            data, phi_by_t, sim_end_scale=scale, bio_cfg=bio
        )
    return row


def audit_y_invariance(
    data,
    recipe: DeployV1Recipe,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Step 9: phi at t_final unchanged when GT clot/mu channels zeroed."""
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    recipe.apply_env()
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"

    data_a = data
    data_b = data.clone() if hasattr(data, "clone") else data
    if hasattr(data_b, "y") and data_b.y is not None and data_b is not data:
        y = data_b.y.clone()
        y[..., STATE_CHANNEL_MU_EFF_ND] = 0.0
        data_b.y = y

    phi_a = rollout_deploy_v1_phi(data_a, recipe, device=device, coupled=False)
    reset_temporal_kinematics_cache()
    reset_coupled_uv_cache()
    phi_b = rollout_deploy_v1_phi(data_b, recipe, device=device, coupled=False)

    n = int(data.y.shape[0])
    t_f = n - 1
    pa = phi_a.get(t_f, phi_a[max(phi_a)])
    pb = phi_b.get(t_f, phi_b[max(phi_b)])
    max_delta = float((pa - pb).abs().max().item())
    return {
        "max_phi_delta": max_delta,
        "pass_y_invariance": max_delta < 1e-5,
        "vel_source": temporal_vel_source(),
    }
