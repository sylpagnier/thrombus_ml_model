"""V2: frozen step1_a35 + nucleation + continuous macro tau / extrap indices (Step 3b clock)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import (
    comsol_final_index,
    extrapolated_t_out_max,
    feature_time_index,
    macro_tau_at_index,
    rollout_time_indices,
    sim_end_scale_from_env,
)
from src.core_physics.clot_nucleation_mask import (
    project_phi_with_nucleation,
    resolve_nucleation_eligibility,
    snapshot_nucleation_config,
)
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache
from src.training.clot_ml_step1_residual import (
    ClotRuleResidualMLP,
    rollout_frozen_rule_phi,
    resolve_step1_rule_cfg,
)
from src.training.clot_ml_v2_step1_nucleation import (
    apply_step1_phi_raw,
    apply_step1_v2_env,
    eval_phi_by_t_on_anchor,
    rollout_step1_v1_nucleation,
)


def apply_step2_v2_env(*, sim_end_scale: float = 1.0) -> None:
    """V1 nucleation shell + continuous extrap growth clock (axis C wiring)."""
    apply_step1_v2_env()
    os.environ["CLOT_ML_CONTINUOUS_EXTRAP"] = "1"
    os.environ["CLOT_ML_SIM_END_SCALE"] = str(max(float(sim_end_scale), 1.0))


def v2_continuous_tau_enabled() -> bool:
    return (os.environ.get("CLOT_ML_CONTINUOUS_EXTRAP") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@torch.no_grad()
def rollout_step1_v2_continuous_tau(
    data,
    rule_cfg,
    model: ClotRuleResidualMLP,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float = 0.35,
    sim_end_scale: float | None = None,
    coupled: bool = False,
    commit_thresh: float = 0.5,
) -> dict[int, torch.Tensor]:
    """Nucleation rollout on virtual tau indices; species/kine features clamp to COMSOL window."""
    scale = float(sim_end_scale if sim_end_scale is not None else sim_end_scale_from_env())
    apply_step2_v2_env(sim_end_scale=scale)
    phi_rule_by_t = rollout_frozen_rule_phi(
        data,
        rule_cfg,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        sim_end_scale=scale,
        coupled=coupled,
    )
    out: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    for t_out in sorted(phi_rule_by_t.keys()):
        t_feat = feature_time_index(data, int(t_out))
        phi_raw = apply_step1_phi_raw(
            model,
            data,
            phi_rule_by_t,
            int(t_out),
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            alpha=alpha,
            flow_time=int(t_feat),
        )
        elig = resolve_nucleation_eligibility(
            data,
            int(t_out),
            device,
            phys_cfg,
            bio_cfg,
            growth_seed="pred",
            phi_pred_by_time=out,
            commit_thresh=commit_thresh,
        )
        phi = project_phi_with_nucleation(
            phi_raw,
            phi_prev,
            elig,
            commit_thresh=commit_thresh,
        )
        out[int(t_out)] = phi
        phi_prev = phi
    return out


def _pred_commit_frac(phi: torch.Tensor, *, thresh: float = 0.5) -> float:
    return float((phi.reshape(-1).float() >= float(thresh)).float().mean().item())


@torch.no_grad()
def extrap_rollout_metrics(
    phi_by_t: dict[int, torch.Tensor],
    data,
    *,
    bio_cfg: BiochemConfig,
    sim_end_scale: float,
    commit_thresh: float = 0.5,
) -> dict[str, Any]:
    """Axis-C plausibility on virtual indices past COMSOL end."""
    t_final = comsol_final_index(data)
    t_max = extrapolated_t_out_max(data, sim_end_scale=float(sim_end_scale))
    t_indices = [
        t
        for t in rollout_time_indices(data, sim_end_scale=float(sim_end_scale))
        if int(t) in phi_by_t
    ]
    if not t_indices:
        return {
            "sim_end_scale": float(sim_end_scale),
            "n_virtual_steps": 0,
            "n_extrap_steps": 0,
            "monotone_commit_frac": True,
            "commit_frac_comsol_end": float("nan"),
            "commit_frac_extrap_end": float("nan"),
            "commit_frac_delta_extrap": float("nan"),
        }

    fracs: list[float] = []
    taus: list[float] = []
    for t in t_indices:
        phi = phi_by_t[int(t)]
        fracs.append(_pred_commit_frac(phi, thresh=commit_thresh))
        taus.append(macro_tau_at_index(data, int(t), bio_cfg=bio_cfg))

    mono = True
    max_drop = 0.0
    for i in range(1, len(fracs)):
        drop = fracs[i - 1] - fracs[i]
        if drop > 1e-6:
            mono = False
        max_drop = max(max_drop, drop)

    in_idx = [i for i, t in enumerate(t_indices) if int(t) <= t_final]
    ex_idx = [i for i, t in enumerate(t_indices) if int(t) > t_final]
    frac_comsol = fracs[in_idx[-1]] if in_idx else fracs[0]
    frac_extrap = fracs[ex_idx[-1]] if ex_idx else frac_comsol

    return {
        "sim_end_scale": float(sim_end_scale),
        "n_virtual_steps": len(t_indices),
        "n_extrap_steps": len(ex_idx),
        "tau_comsol_end": float(taus[in_idx[-1]] if in_idx else taus[0]),
        "tau_extrap_end": float(taus[-1]),
        "monotone_commit_frac": bool(mono),
        "max_commit_frac_drop": float(max_drop),
        "commit_frac_comsol_end": float(frac_comsol),
        "commit_frac_extrap_end": float(frac_extrap),
        "commit_frac_delta_extrap": float(frac_extrap - frac_comsol),
    }


@torch.no_grad()
def eval_step1_v2_on_anchor(
    model: ClotRuleResidualMLP,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
    sim_end_scale: float = 1.0,
    pair_stride: int = 1,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    phi_by_t = rollout_step1_v2_continuous_tau(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        sim_end_scale=float(sim_end_scale),
    )
    row = eval_phi_by_t_on_anchor(
        phi_by_t,
        data,
        anchor=graph_path.stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        pair_stride=pair_stride,
        rule_tag="ml_step1_v2_continuous_tau",
    )
    if float(sim_end_scale) > 1.0 + 1e-6:
        row["extrap"] = extrap_rollout_metrics(
            phi_by_t,
            data,
            bio_cfg=bio_cfg,
            sim_end_scale=float(sim_end_scale),
        )
    return row


@torch.no_grad()
def compare_v1_vs_v2_on_anchor(
    model: ClotRuleResidualMLP,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
    sim_end_scale_extrap: float = 2.0,
) -> dict[str, Any]:
    """LOAO row: V1 nucleation (in-window) vs V2 in-window parity + extrap metrics."""
    stem = graph_path.stem
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)

    phi_v1 = rollout_step1_v1_nucleation(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        sim_end_scale=1.0,
    )
    row_v1 = eval_phi_by_t_on_anchor(
        phi_v1,
        data,
        anchor=stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="ml_step1_v1_nucleation",
    )

    reset_temporal_kinematics_cache()
    phi_v2_in = rollout_step1_v2_continuous_tau(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        sim_end_scale=1.0,
    )
    row_v2_in = eval_phi_by_t_on_anchor(
        phi_v2_in,
        data,
        anchor=stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="ml_step1_v2_continuous_tau",
    )

    reset_temporal_kinematics_cache()
    phi_v2_ex = rollout_step1_v2_continuous_tau(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        sim_end_scale=float(sim_end_scale_extrap),
    )
    extrap = extrap_rollout_metrics(
        phi_v2_ex,
        data,
        bio_cfg=bio_cfg,
        sim_end_scale=float(sim_end_scale_extrap),
    )

    return {
        "anchor": stem,
        "v1": row_v1,
        "v2_inwindow": row_v2_in,
        "v2_extrap": extrap,
        "delta_deploy_v1_v2": float(row_v2_in["deploy_score"] - row_v1["deploy_score"]),
        "delta_tfinal_f1": float(row_v2_in["tfinal_band_f1"] - row_v1["tfinal_band_f1"]),
        "delta_tfinal_pred_frac": float(
            row_v2_in["tfinal_band_pred_frac"] - row_v1["tfinal_band_pred_frac"]
        ),
    }


def default_v2_out_dir() -> Path:
    from src.utils.paths import get_project_root

    return get_project_root() / "outputs/biochem/clot_ml_ladder_v2/v2_continuous_tau"


def v2_manifest_dict(
    *,
    step1_ckpt: str = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    sim_end_scale: float = 1.0,
) -> dict[str, Any]:
    return {
        "name": "clot_ml_v2_s2_continuous_tau",
        "track": "v2",
        "step": "v2",
        "phi_shell": "step1_v2_continuous_tau",
        "step0_json": "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
        "step1_ckpt": step1_ckpt,
        "kine_ckpt": "outputs/kinematics/kinematics_best.pth",
        "vel_source": "kinematics",
        "use_macro_tau": True,
        "continuous_extrap": True,
        "sim_end_scale": float(sim_end_scale),
        "coupled": False,
        "env": {
            "CLOT_V2_NUCLEATION": "1",
            "CLOT_ML_USE_MACRO_TAU": "1",
            "CLOT_ML_CONTINUOUS_EXTRAP": "1",
            "CLOT_ML_SIM_END_SCALE": str(float(sim_end_scale)),
        },
        "rollout_growth_seed": "pred",
        "nucleation_config": snapshot_nucleation_config(),
    }


def resolve_step1_rule_cfg_from_root(step0_json: str | Path) -> Any:
    from src.utils.paths import get_project_root

    return resolve_step1_rule_cfg(get_project_root() / step0_json)
