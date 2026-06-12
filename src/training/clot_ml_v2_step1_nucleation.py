"""V1: frozen step1_a35 MLP + nucleation projection (no ceiling multiply)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_nucleation_mask import (
    project_phi_with_nucleation,
    resolve_nucleation_eligibility,
    snapshot_nucleation_config,
)
from src.core_physics.clot_temporal_growth_rules import (
    _shape_from_phi_at_time,
    _time_frac_at_index,
    deploy_score_from_eval_row,
    reset_temporal_kinematics_cache,
)
from src.training.clot_ml_step1_residual import (
    ClotRuleResidualMLP,
    apply_step1_eval_env,
    apply_step1_phi,
    band_features_pred_kine,
    combine_rule_residual,
    load_step1_checkpoint,
    resolve_step1_rule_cfg,
    rollout_frozen_rule_phi,
    rollout_step1_phi,
)
from src.training.train_clot_phi_simple import _clot_metrics
from src.utils.paths import get_project_root


def v2_nucleation_rollout_enabled() -> bool:
    return (os.environ.get("CLOT_V2_NUCLEATION") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def apply_step1_v2_env() -> None:
    """Step1 eval env + V2 nucleation rollout flag (nucleation uses pred seed in code)."""
    apply_step1_eval_env()
    os.environ["CLOT_V2_NUCLEATION"] = "1"
    os.environ.setdefault("CLOT_V2_NUCLEATION_HOPS", "1")
    os.environ.setdefault("CLOT_V2_CATALYTIC_HOPS", "1")
    os.environ.setdefault("CLOT_ML_USE_MACRO_TAU", "1")


def combine_rule_residual_uncapped(
    phi_rule: torch.Tensor,
    delta_logit: torch.Tensor,
    *,
    alpha: float = 0.35,
) -> torch.Tensor:
    """MLP residual without ceiling mask (nucleation projects afterward)."""
    base = phi_rule.reshape(-1).clamp(0.0, 1.0)
    delta = torch.sigmoid(delta_logit.reshape(-1))
    return (base + float(alpha) * delta).clamp(0.0, 1.0)


@torch.no_grad()
def apply_step1_phi_raw(
    model: ClotRuleResidualMLP,
    data,
    phi_rule_by_t: dict[int, torch.Tensor],
    t_out: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
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
    return combine_rule_residual_uncapped(phi_rule, delta, alpha=alpha)


@torch.no_grad()
def rollout_step1_v1_nucleation(
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
    commit_thresh: float = 0.5,
) -> dict[int, torch.Tensor]:
    """Temporal step1 rollout with monotone nucleation projection (E_seed, pred seed)."""
    apply_step1_v2_env()
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
    phi_prev: torch.Tensor | None = None
    for t_out in sorted(phi_rule_by_t.keys()):
        phi_raw = apply_step1_phi_raw(
            model,
            data,
            phi_rule_by_t,
            int(t_out),
            device=device,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            alpha=alpha,
            flow_time=flow_time,
        )
        elig = resolve_nucleation_eligibility(
            data,
            int(t_out),
            device,
            phys_cfg,
            bio_cfg,
            growth_seed="pred",
            phi_pred_by_time=out,
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


@torch.no_grad()
def eval_phi_by_t_on_anchor(
    phi_by_t: dict[int, torch.Tensor],
    data,
    *,
    anchor: str,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
    rule_tag: str = "ml_step1",
) -> dict[str, Any]:
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1, pair_stride=pair_stride)
    if not pairs:
        return {"anchor": anchor, "rule": rule_tag, "n_pairs": 0}

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
        "anchor": anchor,
        "rule": rule_tag,
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


@torch.no_grad()
def eval_step1_v1_on_anchor(
    model: ClotRuleResidualMLP,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
    pair_stride: int = 1,
    sim_end_scale: float | None = None,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    phi_by_t = rollout_step1_v1_nucleation(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
        sim_end_scale=sim_end_scale,
    )
    return eval_phi_by_t_on_anchor(
        phi_by_t,
        data,
        anchor=graph_path.stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        pair_stride=pair_stride,
        rule_tag="ml_step1_v1_nucleation",
    )


@torch.no_grad()
def compare_ceiling_vs_nucleation_on_anchor(
    model: ClotRuleResidualMLP,
    rule_cfg,
    *,
    graph_path: Path,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    alpha: float,
) -> dict[str, Any]:
    reset_temporal_kinematics_cache()
    data = torch.load(graph_path, map_location=device, weights_only=False)
    stem = graph_path.stem

    apply_step1_eval_env()
    phi_ceiling = rollout_step1_phi(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
    )
    row_ceil = eval_phi_by_t_on_anchor(
        phi_ceiling,
        data,
        anchor=stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="ml_step1_ceiling",
    )

    reset_temporal_kinematics_cache()
    phi_nuc = rollout_step1_v1_nucleation(
        data,
        rule_cfg,
        model,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        alpha=alpha,
    )
    row_nuc = eval_phi_by_t_on_anchor(
        phi_nuc,
        data,
        anchor=stem,
        device=device,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        rule_tag="ml_step1_v1_nucleation",
    )

    return {
        "anchor": stem,
        "ceiling": row_ceil,
        "nucleation": row_nuc,
        "delta_deploy_score": float(row_nuc["deploy_score"] - row_ceil["deploy_score"]),
        "delta_tfinal_f1": float(row_nuc["tfinal_band_f1"] - row_ceil["tfinal_band_f1"]),
        "delta_tfinal_pred_frac": float(
            row_nuc["tfinal_band_pred_frac"] - row_ceil["tfinal_band_pred_frac"]
        ),
    }


def default_v1_out_dir() -> Path:
    return get_project_root() / "outputs/biochem/clot_ml_ladder_v2/v1_nucleation"


def v1_manifest_dict(
    *,
    step1_ckpt: str = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
) -> dict[str, Any]:
    return {
        "name": "clot_ml_v2_s1_nucleation",
        "track": "v2",
        "step": "v1",
        "phi_shell": "step1_v1_nucleation",
        "step0_json": "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
        "step1_ckpt": step1_ckpt,
        "kine_ckpt": "outputs/kinematics/kinematics_best.pth",
        "vel_source": "kinematics",
        "use_macro_tau": True,
        "continuous_extrap": False,
        "sim_end_scale": 1.0,
        "coupled": False,
        "env": {
            "CLOT_V2_NUCLEATION": "1",
            "CLOT_V2_NUCLEATION_HOPS": "1",
        },
        "rollout_growth_seed": "pred",
        "nucleation_config": snapshot_nucleation_config(),
    }
