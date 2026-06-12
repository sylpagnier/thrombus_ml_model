"""Step 5b smoke: one DEQ re-solve with mu_eff prior vs frozen kinematics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_forecast import iter_forecast_pairs
from src.core_physics.clot_phi_rollout import KinematicsUvProvider
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache, temporal_vel_source
from src.training.clot_ml_step5a_mu_readout import (
    PhiShellKind,
    Step5aEvalConfig,
    mu_eff_carreau_blend_from_phi,
    rollout_phi_for_shell,
)
from src.training.clot_ml_step0_coef import load_step0_coef_json
from src.training.clot_ml_step1_residual import load_step1_checkpoint, resolve_step1_rule_cfg
from src.utils.kinematics_inference import load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint
from src.utils.metrics import rel_l2_uvp
from src.utils.paths import get_project_root


@torch.no_grad()
def smoke_coupled_kine_one_anchor(
    graph_path: Path,
    *,
    shell: str = "step1",
    step0_json: str = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    step1_ckpt: str = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    kine_ckpt: str = "",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Compare frozen GINO-DEQ vs mu-prior DEQ at t_final."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_temporal_kinematics_cache()
    phys_k = PhysicsConfig(phase="kinematics")
    phys_b = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(graph_path, map_location=dev, weights_only=False)
    stem = graph_path.stem

    cfg5 = Step5aEvalConfig(shell=PhiShellKind(shell), step0_json=step0_json, step1_ckpt=step1_ckpt)
    root = get_project_root()
    if cfg5.shell == PhiShellKind.INC40:
        rule_cfg = load_step0_coef_json(root / step0_json).to_rule_config(name="step5b_smoke")
    else:
        rule_cfg = resolve_step1_rule_cfg(root / step0_json)
    step1_model = None
    step1_alpha = 0.35
    if cfg5.shell == PhiShellKind.STEP1:
        step1_model, meta = load_step1_checkpoint(root / step1_ckpt, device=dev)
        step1_alpha = float(meta.get("alpha", 0.35))

    phi_by_t = rollout_phi_for_shell(
        data, cfg5, rule_cfg, device=dev, phys_cfg=phys_b, bio_cfg=bio,
        step1_model=step1_model, step1_alpha=step1_alpha,
    )
    pairs = iter_forecast_pairs(int(data.y.shape[0]))
    t_in, t_out = pairs[-1]
    phi_final = phi_by_t[int(t_out)]
    mu_eff = mu_eff_carreau_blend_from_phi(
        data, phi_final, int(t_out), device=dev, phys_cfg=phys_b, bio_cfg=bio,
    )

    ckpt = kine_ckpt.strip() or str(resolve_kinematics_checkpoint())
    kine_model = load_kinematics_predictor(ckpt, dev, phys_cfg=phys_k)
    batch = data.to(dev)
    pred_frozen = predict_kinematics(kine_model, batch)
    u_f, v_f, p_f = pred_frozen[:, 0], pred_frozen[:, 1], pred_frozen[:, 2]

    provider = KinematicsUvProvider(dev)
    u_c, v_c = provider.uv_nd_from_mu_si(batch, mu_eff)

    gt = data.y[int(t_out)].to(dev)
    gt_uvp = gt[:, :3]
    pred_frozen_uvp = torch.stack([u_f, v_f, p_f], dim=1)
    pred_coupled_uvp = torch.stack([u_c, v_c, p_f], dim=1)

    rel_frozen = float(rel_l2_uvp(pred_frozen_uvp, gt_uvp))
    rel_coupled = float(rel_l2_uvp(pred_coupled_uvp, gt_uvp))
    rel_delta = float(rel_l2_uvp(pred_coupled_uvp, pred_frozen_uvp))
    mu_ratio = float((mu_eff / mu_eff.mean().clamp(min=1e-8)).max().item())

    return {
        "anchor": stem,
        "shell": shell,
        "t_out": int(t_out),
        "vel_source": temporal_vel_source(),
        "rel_l2_frozen_kine": rel_frozen,
        "rel_l2_coupled_kine": rel_coupled,
        "rel_l2_coupled_vs_frozen": rel_delta,
        "mu_eff_max_over_mean": mu_ratio,
        "phi_commit_frac": float((phi_final > 0.5).float().mean().item()),
    }
