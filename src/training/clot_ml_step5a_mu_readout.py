"""Step 5a: mu_eff readout from frozen phi shells (Carreau + log-blend cap)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import resolve_bulk_carreau_mu_si
from src.core_physics.clot_phi_mu_inject import (
    ClotPhiMuInjector,
    biochem_mlp_mu_map_enabled,
    build_clot_phi_mu_injector,
)
from src.core_physics.clot_phi_simple import log_blend_mu_eff_si, project_deploy_mu_with_support
from src.core_physics.clot_temporal_growth_rules import (
    TemporalGrowthRuleConfig,
    _shape_from_phi_at_time,
    deploy_score_from_eval_row,
    reset_temporal_kinematics_cache,
    rollout_temporal_phi,
    temporal_rule_config_from_env,
    temporal_vel_source,
)
from src.core_physics.clot_temporal_growth_rules import _resolve_uv_for_temporal_risk
from src.evaluation.clot_shape_score import compute_clot_shape_metrics
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step1_residual import (
    load_step1_checkpoint,
    resolve_step1_rule_cfg,
    rollout_step1_phi,
)
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


class PhiShellKind(str, Enum):
    INC40 = "inc40"
    STEP1 = "step1"
    LANE_B = "lane_b"


@dataclass
class Step5aEvalConfig:
    anchor_dir: str = "data/processed/graphs_biochem_anchors"
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json"
    step1_ckpt: str = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth"
    lane_b_ckpt: str = ""
    shell: PhiShellKind = PhiShellKind.STEP1
    pair_stride: int = 1


def _species_log_zeros(data, device: torch.device) -> torch.Tensor:
    n = int(data.num_nodes)
    return torch.zeros(n, 12, device=device, dtype=torch.float32)


@torch.no_grad()
def mu_eff_carreau_blend_from_phi(
    data,
    phi: torch.Tensor,
    t_out: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    project_support: bool = True,
) -> torch.Tensor:
    """Default 5a recipe: log-blend Carreau bulk toward clot mu cap along phi."""
    u, v = _resolve_uv_for_temporal_risk(data, 0, device)
    mu_c = resolve_bulk_carreau_mu_si(
        data, t_out, phys_cfg, device, u_nd=u, v_nd=v
    )
    mu = log_blend_mu_eff_si(mu_c, phi.reshape(-1))
    if not project_support:
        return mu.reshape(-1)
    step = build_clot_forecast_pair_step(data, 0, min(t_out, int(data.y.shape[0]) - 1), phys_cfg, bio_cfg, device)
    return project_deploy_mu_with_support(
        data=data,
        step=step,
        mu_pred=mu.reshape(-1),
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        forecast_one_step=True,
        time_index=int(t_out),
        bulk_time_index=int(t_out),
    ).reshape(-1)


@torch.no_grad()
def mu_eff_lane_b_from_injector(
    injector: ClotPhiMuInjector,
    data,
    t_out: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    ti = min(int(t_out), int(data.y.shape[0]) - 1)
    u, v = _resolve_uv_for_temporal_risk(data, 0, device)
    sp = _species_log_zeros(data, device)
    mu = injector.apply_mu_map(
        data,
        ti,
        u_nd=u.reshape(-1),
        v_nd=v.reshape(-1),
        species_log=sp,
        macro_step_index=max(ti, 0),
    )
    return mu.reshape(-1)


@torch.no_grad()
def rollout_phi_for_shell(
    data,
    cfg: Step5aEvalConfig,
    rule_cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    step1_model=None,
    step1_alpha: float = 0.35,
) -> dict[int, torch.Tensor]:
    if cfg.shell == PhiShellKind.STEP1:
        if step1_model is None:
            raise ValueError("step1 shell requires loaded model")
        return rollout_step1_phi(
            data,
            rule_cfg,
            step1_model,
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            alpha=step1_alpha,
        )
    return rollout_temporal_phi(
        data, rule_cfg, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
    )


@torch.no_grad()
def eval_step5a_mu_on_anchor(
    cfg: Step5aEvalConfig,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    injector: ClotPhiMuInjector | None = None,
    step1_model=None,
    step1_alpha: float = 0.35,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    data = torch.load(graph_path, map_location=device, weights_only=False)
    stem = graph_path.stem

    if cfg.shell == PhiShellKind.INC40:
        rule_cfg = load_step0_coef_json(get_project_root() / cfg.step0_json).to_rule_config(
            name="step5a_inc40"
        )
    else:
        rule_cfg = resolve_step1_rule_cfg(get_project_root() / cfg.step0_json)

    phi_by_t = rollout_phi_for_shell(
        data,
        cfg,
        rule_cfg,
        device=device,
        phys_cfg=phys,
        bio_cfg=bio,
        step1_model=step1_model,
        step1_alpha=step1_alpha,
    )
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1, pair_stride=cfg.pair_stride)
    if not pairs:
        return {"anchor": stem, "shell": cfg.shell.value, "n_pairs": 0}

    t_in_f, t_out_f = pairs[-1]
    phi_final = phi_by_t[int(t_out_f)]

    if cfg.shell == PhiShellKind.LANE_B:
        if injector is None:
            raise ValueError("lane_b shell requires ClotPhiMuInjector")
        mu_pred = mu_eff_lane_b_from_injector(injector, data, int(t_out_f), device=device)
    else:
        mu_pred = mu_eff_carreau_blend_from_phi(
            data,
            phi_final,
            int(t_out_f),
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )

    step = build_clot_forecast_pair_step(data, int(t_in_f), int(t_out_f), phys, bio, device)
    band = _clot_metrics(phi_final, step.phi_gt, step.loss_mask)
    shape = _shape_from_phi_at_time(
        data,
        phi_final,
        int(t_in_f),
        int(t_out_f),
        device=device,
        phys_cfg=phys,
        bio_cfg=bio,
    )

    y_gt = data.y[int(t_out_f)].to(device=device, dtype=torch.float32)
    mu_gt_si = phys.viscosity_nd_to_si(y_gt[:, STATE_CHANNEL_MU_EFF_ND])
    pred_state = y_gt.clone()
    pred_state[:, STATE_CHANNEL_MU_EFF_ND] = phys.viscosity_si_to_nd(mu_pred)
    from src.core_physics.clot_growth_masks import resolve_ceiling_mask

    shape_direct = compute_clot_shape_metrics(
        pred_state=pred_state,
        gt_state=y_gt,
        edge_index=data.edge_index.to(device),
        phys_cfg=phys,
        node_mask=resolve_ceiling_mask(data, device, bio),
    )

    gt_pos = float((step.phi_gt[step.loss_mask] > 0.5).float().mean().item()) if bool(
        step.loss_mask.any()
    ) else 0.0

    row = {
        "anchor": stem,
        "shell": cfg.shell.value,
        "rule": rule_cfg.name,
        "n_pairs": len(pairs),
        "vel_source": temporal_vel_source(),
        "tfinal_band_f1": float(band["clot_f1"]),
        "tfinal_band_pred_frac": float(band["pred_pos_frac"]),
        "tfinal_gt_pos_frac": gt_pos,
        "tfinal_clot_shape": float(shape.get("clot_shape", float("nan"))),
        "tfinal_clot_shape_bal": float(shape.get("clot_shape_balanced", float("nan"))),
        "mu_shape_f1": float(shape_direct.get("clot_shape", float("nan"))),
        "mu_recall": float(shape_direct.get("clot_recall", float("nan"))),
        "mu_flow_ok": float(shape_direct.get("flow_ok", 1.0)),
        "deploy_score": 0.0,
    }
    row["deploy_score"] = float(deploy_score_from_eval_row(row))
    return row


def resolve_lane_b_clot_phi_ckpt(lane_b_ckpt: str = "") -> Path | None:
    """Resolve clot-phi ``.pth`` from manifest JSON or direct checkpoint path."""
    from src.inference.clot_baseline_recipe import baseline_manifest_path, load_manifest

    root = get_project_root()
    arg = lane_b_ckpt.strip()
    manifest_path = Path(arg) if arg else baseline_manifest_path()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    if manifest_path.is_file() and manifest_path.suffix.lower() == ".json":
        recipe, _eval = load_manifest(manifest_path)
        paths = recipe.resolved_paths(root)
        ckpt = paths.get("clot_phi_ckpt")
        return ckpt if ckpt is not None and ckpt.is_file() else None

    if manifest_path.is_file() and manifest_path.suffix.lower() == ".pth":
        return manifest_path
    return None


def load_step5a_injector(
    lane_b_ckpt: str,
    device: torch.device,
) -> ClotPhiMuInjector | None:
    from src.inference.clot_baseline_recipe import load_manifest, baseline_manifest_path
    from src.inference.deploy_mu_map_env import apply_deploy_mu_map_env

    root = get_project_root()
    arg = lane_b_ckpt.strip()
    manifest_path = Path(arg) if arg else baseline_manifest_path()
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    if manifest_path.is_file() and manifest_path.suffix.lower() == ".json":
        recipe, eval_meta = load_manifest(manifest_path)
        deploy_overrides = dict(recipe.deploy_mu_map_env or {})
        deploy_overrides.update(dict(eval_meta.get("deploy_mu_map_env") or {}))
        apply_deploy_mu_map_env(deploy_overrides)
        recipe.apply_clot_phi_env()
        ckpt = resolve_lane_b_clot_phi_ckpt(str(manifest_path))
    else:
        os.environ["BIOCHEM_MLP_MU_MAP"] = "1"
        os.environ["BIOCHEM_MLP_CLOT_INJECT"] = "1"
        ckpt = resolve_lane_b_clot_phi_ckpt(arg)

    if ckpt is None:
        return None
    os.environ["BIOCHEM_MLP_CLOT_INJECT"] = "1"
    return build_clot_phi_mu_injector(device, ckpt)


@torch.no_grad()
def eval_loao_step5a(
    cfg: Step5aEvalConfig,
    *,
    device: torch.device,
    anchor_dir: Path | None = None,
) -> dict[str, Any]:
    root = get_project_root()
    adir = anchor_dir or (root / cfg.anchor_dir)
    paths = sorted(adir.glob("patient*.pt"))
    if not paths:
        raise FileNotFoundError(f"No anchors under {adir}")

    injector = None
    step1_model = None
    step1_alpha = 0.35
    if cfg.shell == PhiShellKind.LANE_B:
        injector = load_step5a_injector(cfg.lane_b_ckpt, device)
        if injector is None:
            raise FileNotFoundError("lane_b shell: set --lane-b-ckpt or clot_baseline manifest")
    elif cfg.shell == PhiShellKind.STEP1:
        ckpt = root / cfg.step1_ckpt
        step1_model, meta = load_step1_checkpoint(ckpt, device=device)
        step1_alpha = float(meta.get("alpha", 0.35))

    per_anchor = [
        eval_step5a_mu_on_anchor(
            cfg,
            graph_path=p,
            device=device,
            injector=injector,
            step1_model=step1_model,
            step1_alpha=step1_alpha,
        )
        for p in paths
    ]
    mean_deploy = sum(r["deploy_score"] for r in per_anchor) / len(per_anchor)
    return {
        "step": "5a",
        "shell": cfg.shell.value,
        "mean_deploy": mean_deploy,
        "per_anchor": per_anchor,
        "config": {
            "anchor_dir": str(adir),
            "step0_json": cfg.step0_json,
            "step1_ckpt": cfg.step1_ckpt,
            "lane_b_ckpt": cfg.lane_b_ckpt,
            "mlp_mu_map": biochem_mlp_mu_map_enabled(),
        },
    }
