"""Rung 4 mini-ladder: deployable species rules -> verified gelation physics -> clot phi.

Steps (increasing complexity; all use deploy-faithful nucleation mask E(t)):

  s0          Rules-only FI/Mat in E(t) from flow + geometry + macro time (NO GT species)
  s0_phi_rule Prior spatial rule -> phi directly (no species path; baseline)
  s0_oracle_* Audit ceilings only (read GT species in E(t) -- never deploy)

  s1_mlp_phi  Residual MLP on phi or rule phi in E(t) [planned]
  s2_loc      Risk reweight before s0 top-frac gate in E(t)
  s2_delta    Deprecated per-node FI/Mat delta (wall carpet)
  s3_temporal s2_loc + wall-band GRU
  s4_band_ml  FI/Mat GNN in E(t); FN/FP + soft clot; coupled
  s5_gnode_fimat  Narrow GNODE: 2 species, E(t)/ceiling mask
  s6_gnode    Full 12-ch GNODE teacher

Env: ``T0_RUNG4_STEP`` (default ``s0``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.clot_phi_simple import _anchor_flow_props, predict_phi_prior_rule
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat

FI_SLICE_IDX = 8
MAT_SLICE_IDX = 11


def resting_species_log_nd(data, device: torch.device) -> torch.Tensor:
    from src.architecture.gnode_biochem import _default_resting_species

    n = int(data.num_nodes)
    return _default_resting_species(n, device, data)


def species_log_mae_in_mask(
    pred_series: torch.Tensor,
    data,
    time_index: int,
    mask: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    t = int(time_index)
    m = mask.reshape(-1).bool()
    if not bool(m.any().item()):
        return {"species_log_mae": float("nan"), "fi_log_mae": float("nan"), "mat_log_mae": float("nan")}
    pred = pred_series[t, :, 4:16].to(device=device, dtype=torch.float32)
    gt = data.y[t, :, 4:16].to(device=device, dtype=torch.float32)
    diff = (pred - gt).abs()
    return {
        "species_log_mae": float(diff[m].mean().item()),
        "fi_log_mae": float(diff[m, FI_SLICE_IDX].mean().item()),
        "mat_log_mae": float(diff[m, MAT_SLICE_IDX].mean().item()),
        "n_mask": int(m.sum().item()),
    }

Rung4Step = Literal[
    "s0",
    "s0_phi_rule",
    "s0_oracle_fi_mat",
    "s0_oracle_all",
    "s1_mlp_phi",
    "s2_species",
    "s2_loc",
    "s2_delta",
    "s3_temporal",
    "s4_band_gnn",
    "s4_band_ml",
    "s5_gnode",
    "s5_gnode_fimat",
    "s6_gnode",
]

_STEP_META: dict[str, dict[str, object]] = {
    "s0": {
        "deploy": True,
        "uses_gt_species": False,
        "desc": "p92 risk norm x tau onset x top-frac FI/Mat ramp in E(t)",
    },
    "s0_phi_rule": {
        "deploy": True,
        "uses_gt_species": False,
        "desc": "spatial prior rule -> phi (no species); physics bypass",
    },
    "s0_oracle_fi_mat": {
        "deploy": False,
        "uses_gt_species": True,
        "desc": "AUDIT: GT FI/Mat in E(t)",
    },
    "s0_oracle_all": {
        "deploy": False,
        "uses_gt_species": True,
        "desc": "AUDIT: full GT species in E(t)",
    },
    "s1_mlp_phi": {"deploy": True, "uses_gt_species": False, "desc": "s0 species + physics phi + residual MLP in E(t)"},
    "s2_species": {"deploy": True, "uses_gt_species": False, "desc": "s2_loc or s2_delta via T0_R4_S2_MODE"},
    "s2_loc": {"deploy": True, "uses_gt_species": False, "desc": "s0 + risk reweight MLP before top-frac gate in E(t)"},
    "s2_delta": {"deploy": True, "uses_gt_species": False, "desc": "deprecated: s0 + FI/Mat delta MLP (wall carpet)"},
    "s3_temporal": {"deploy": True, "uses_gt_species": False, "desc": "s2_loc + per-node GRU on wall-band feats in E(t)"},
    "s4_band_gnn": {"deploy": True, "uses_gt_species": False, "desc": "alias s4_band_ml"},
    "s4_band_ml": {"deploy": True, "uses_gt_species": False, "desc": "2L band GNN gate residual in E(t)"},
    "s5_gnode": {"deploy": True, "uses_gt_species": False, "desc": "alias s6_gnode full teacher"},
    "s5_gnode_fimat": {"deploy": True, "uses_gt_species": False, "desc": "narrow 2-ch band GNN FI/Mat delta (T0_R4_S5_CKPT)"},
    "s5_fimat": {"deploy": True, "uses_gt_species": False, "desc": "alias s5_gnode_fimat"},
    "species_gnn": {
        "deploy": True,
        "uses_gt_species": False,
        "desc": "Wall-band continuous GNN FI/Mat (s34) + viscosity beta (s35); pred kine features",
    },
    "s34_gnn": {
        "deploy": True,
        "uses_gt_species": False,
        "desc": "alias species_gnn (s34+s35 deploy stack)",
    },
    "s_star_g0_rules": {"deploy": True, "uses_gt_species": False, "desc": "S* G0: s0 deploy rules (eval only)"},
    "s_star_gate": {"deploy": True, "uses_gt_species": False, "desc": "S* G4: gate GNN + commit loss (sweep ckpt)"},
    "s_star_species": {"deploy": True, "uses_gt_species": False, "desc": "S* M2: FI/Mat delta on frozen gate (sweep ckpt)"},
    "s_star_dyn": {"deploy": True, "uses_gt_species": False, "desc": "S* T2: GRU temporal smooth (sweep ckpt)"},
    "s_star_gate_species": {"deploy": True, "uses_gt_species": False, "desc": "S* GM: gate + species (sweep ckpt)"},
    "s_star_full": {"deploy": True, "uses_gt_species": False, "desc": "S* GMT: gate + species + dyn (sweep ckpt)"},
    "s_star_small_ml": {"deploy": True, "uses_gt_species": False, "desc": "S* tiny gate+species, high w_commit"},
    "s6_gnode": {"deploy": True, "uses_gt_species": False, "desc": "full 12-ch GNODE teacher"},
}


def rung4_step_from_env() -> str:
    raw = (os.environ.get("T0_RUNG4_STEP") or os.environ.get("T0_RULES_SPECIES_MODE") or "s0").strip().lower()
    # legacy mode names
    legacy = {
        "rest_nuc_oracle_fi_mat": "s0_oracle_fi_mat",
        "rest_nuc_oracle_all": "s0_oracle_all",
        "wall_dilate_fi_mat": "s0_oracle_fi_mat",  # was partial-oracle; use s0 instead
        "s1_mlp": "s1_mlp_phi",
        "s2": "s2_species",
        "s2_loc": "s2_species",
        "s3": "s3_temporal",
        "s4": "s4_band_ml",
        "s4_band_gnn": "s4_band_ml",
        "s4_band_ml": "s4_band_ml",
        "s5_gnode_fimat": "s5_gnode_fimat",
        "s5": "s5_gnode_fimat",
        "s_star": "s_star_full",
        "s_star_g0": "s_star_g0_rules",
        "s_star_gm": "s_star_gate_species",
        "s_star_gmt": "s_star_full",
        "s6": "s6_gnode",
        "s6_gnode": "s6_gnode",
    }
    return legacy.get(raw, raw)


def _species_step_for(step: str) -> str:
    """Species builder step (s1 reuses deploy s0 species)."""
    if step == "s1_mlp_phi":
        return "s0"
    return step


def rung4_step_meta(step: str | None = None) -> dict[str, object]:
    s = (step or rung4_step_from_env()).strip().lower()
    return dict(_STEP_META.get(s, {"deploy": False, "desc": f"unknown step {s!r}"}))


def rung4_step_is_deploy(step: str | None = None) -> bool:
    return bool(rung4_step_meta(step).get("deploy", False))


def rung4_step_uses_gt_species(step: str | None = None) -> bool:
    return bool(rung4_step_meta(step).get("uses_gt_species", False))


def rung4_use_dgamma_wall_seed() -> bool:
    """Deploy band for Rung4 / S-star: full wall at t=0 (default False).

    Set ``T0_RUNG4_USE_DGAMMA_WALL_SEED=1`` only for legacy oracle-band replay.
    """
    raw = (os.environ.get("T0_RUNG4_USE_DGAMMA_WALL_SEED") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def rung4_step_uses_coupled_species_rollout(step: str | None = None) -> bool:
    """Steps that need full coupled species series (GRU/GNN/sweep), not per-t s0 builder."""
    s = (step or rung4_step_from_env()).strip().lower()
    if s in (
        "s3_temporal",
        "s4_band_ml",
        "s4_band_gnn",
        "s5_gnode_fimat",
        "s5_fimat",
        "species_gnn",
        "s34_gnn",
    ):
        return True
    return s not in ("s_star_g0_rules",) and (
        s.startswith("s_star") or s.startswith("s4_")
    )


def _log1p_nd_for_fi_si(fi_si: float, bio_cfg: BiochemConfig, device: torch.device) -> float:
    scales = bio_cfg.get_species_scales(device=device)[:12]
    scale = max(float(scales[FI_SLICE_IDX]), 1e-12)
    return float(math.log1p(max(fi_si, 0.0) / scale))


def _log1p_nd_for_mat_si(mat_si: float, bio_cfg: BiochemConfig) -> float:
    minf = max(float(bio_cfg.Minf), 1e-12)
    return float(math.log1p(max(mat_si, 0.0) / minf))


def _s0_env_float(key: str, default: float) -> float:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _s0_onset_power() -> float:
    return max(_s0_env_float("T0_R4_S0_ONSET_POWER", 1.0), 0.0)


def _s0_fi_mat_gain() -> float:
    return max(_s0_env_float("T0_R4_S0_FI_MAT_GAIN", 1.15), 1.0)


def _s0_onset_factor(tau: float) -> float:
    """Macro-time ramp: 0 before tau_start, 1 after tau_end (power-shaped)."""
    t0 = _s0_env_float("T0_R4_S0_ONSET_TAU_START", 0.06)
    t1 = _s0_env_float("T0_R4_S0_ONSET_TAU_END", 0.32)
    if t1 <= t0:
        return 1.0 if tau >= t1 else 0.0
    x = (float(tau) - t0) / (t1 - t0)
    return max(0.0, min(1.0, x)) ** _s0_onset_power()


def _s0_risk_normalized(
    risk: torch.Tensor,
    elig: torch.Tensor,
) -> torch.Tensor:
    """Scale shear-risk to ~[0,1] on E(t) via band percentile (COMSOL-aligned prior)."""
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    r = risk.reshape(-1).float()
    mask = elig.reshape(-1).bool()
    if not bool(mask.any().item()):
        return torch.zeros_like(r)
    pctl = max(min(_s0_env_float("T0_R4_S0_RISK_NORM_PCTL", 0.92), 0.99), 0.50)
    floor = max(_s0_env_float("T0_R4_S0_RISK_NORM_FLOOR", 0.02), 1e-8)
    scale = max(float(torch.quantile(r[mask], pctl).item()), floor)
    return (r / scale).clamp(0.0, 1.0)


def _s0_spatial_weight(
    risk_norm: torch.Tensor,
    elig: torch.Tensor,
) -> torch.Tensor:
    """Keep top hotspot fraction inside E(t); 0 disables (soft risk only)."""
    top_frac = _s0_env_float("T0_R4_S0_SPATIAL_TOP_FRAC", 0.08)
    rn = risk_norm.reshape(-1)
    mask = elig.reshape(-1).bool()
    if top_frac <= 0.0 or not bool(mask.any().item()):
        return rn
    q = 1.0 - max(min(top_frac, 0.50), 0.01)
    thr = torch.quantile(rn[mask].float(), q)
    return torch.where(mask & (rn >= thr), rn, torch.zeros_like(rn))


@torch.no_grad()
def _build_s0_deploy_species(
    data,
    time_index: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    elig: torch.Tensor,
    commits_prev: torch.Tensor | None = None,
    risk_n_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deployable s0: localized FI/Mat ramp toward gelation crit inside E(t)."""
    from src.core_physics.clot_nucleation_mask import resolve_catalytic_hood

    t = int(time_index)
    sp = resting_species_log_nd(data, device)
    y = data.y[t].to(device=device, dtype=torch.float32)
    u_nd = y[:, 0]
    v_nd = y[:, 1]
    props = _anchor_flow_props(data, device)
    risk = clot_prior_score_flat(data, u_nd, v_nd, bio_cfg, props).reshape(-1).clamp(min=0.0)
    tau = float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))
    onset = _s0_onset_factor(tau)
    elig_b = elig.reshape(-1).bool()
    if risk_n_override is not None:
        risk_n = risk_n_override.reshape(-1).to(device=device, dtype=torch.float32).clamp(min=0.0)
    else:
        risk_n = _s0_risk_normalized(risk, elig)
    spatial = _s0_spatial_weight(risk_n, elig)
    gate = (onset * spatial).clamp(0.0, 1.0)

    # Catalytic spread: uniform gelation push on neighborhood of prior commits (growth front).
    spread_decay = max(_s0_env_float("T0_R4_S0_SPREAD_DECAY", 0.85), 0.0)
    spread_hops = max(int(_s0_env_float("T0_R4_S0_SPREAD_HOPS", 0)), 0)
    if (
        spread_decay > 0.0
        and spread_hops > 0
        and commits_prev is not None
        and bool(commits_prev.reshape(-1).bool().any().item())
        and hasattr(data, "edge_index")
    ):
        hood = resolve_catalytic_hood(
            commits_prev.reshape(-1).bool().to(device=device),
            data.edge_index.to(device=device),
            catalytic_hops=spread_hops,
        )
        if bool(hood.any().item()):
            spread_gate = min(1.0, max(0.0, float(onset) * spread_decay))
            gate = torch.where(hood, torch.maximum(gate, torch.full_like(gate, spread_gate)), gate)

    gain = _s0_fi_mat_gain()
    fi_tgt = _log1p_nd_for_fi_si(float(bio_cfg.viscosity_fi_crit) * gain, bio_cfg, device)
    mat_tgt = _log1p_nd_for_mat_si(float(bio_cfg.viscosity_mat_crit) * gain, bio_cfg)

    sp = sp.clone()
    fi_rest = sp[:, FI_SLICE_IDX]
    mat_rest = sp[:, MAT_SLICE_IDX]
    sp[:, FI_SLICE_IDX] = fi_rest + gate * (fi_tgt - fi_rest)
    sp[:, MAT_SLICE_IDX] = mat_rest + gate * (mat_tgt - mat_rest)
    return sp


@torch.no_grad()
def _build_oracle_species(
    data,
    time_index: int,
    device: torch.device,
    *,
    elig: torch.Tensor,
    step: str,
) -> torch.Tensor:
    sp_gt = data.y[int(time_index), :, 4:16].to(device=device, dtype=torch.float32)
    sp_rest = resting_species_log_nd(data, device)
    if step == "s0_oracle_all":
        return torch.where(elig.unsqueeze(-1), sp_gt, sp_rest)
    sp = sp_rest.clone()
    sp[elig, FI_SLICE_IDX] = sp_gt[elig, FI_SLICE_IDX]
    sp[elig, MAT_SLICE_IDX] = sp_gt[elig, MAT_SLICE_IDX]
    return sp


def _assert_step_implemented(step: str) -> None:
    if step in ():
        raise NotImplementedError(
            f"Rung 4 step {step!r} not implemented yet; see docs/T0_RUNG_LADDER.md"
        )


@torch.no_grad()
def build_rung4_species_log_nd_at_time(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    *,
    commits_prev: torch.Tensor | None,
    step: str | None = None,
    nucleation_hops: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Species log1p ND (N, 12) and nucleation eligibility E(t) for Rung 4 mini-ladder."""
    step_s = (step or rung4_step_from_env()).strip().lower()
    _assert_step_implemented(step_s)
    sp_step = _species_step_for(step_s)

    elig = resolve_nucleation_eligibility(
        data,
        int(time_index),
        device,
        phys_cfg,
        bio_cfg,
        commits_prev=commits_prev,
        growth_seed="pred",
        nucleation_hops=nucleation_hops,
        use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
    ).reshape(-1).bool()

    if sp_step == "s0":
        sp = _build_s0_deploy_species(
            data,
            int(time_index),
            device,
            bio_cfg,
            elig=elig,
            commits_prev=commits_prev,
        )
    elif sp_step in ("s0_oracle_fi_mat", "s0_oracle_all"):
        sp = _build_oracle_species(data, int(time_index), device, elig=elig, step=sp_step)
    elif sp_step == "s0_phi_rule":
        sp = resting_species_log_nd(data, device)
    elif step_s in ("s2_species", "s2_loc", "s2_delta"):
        from src.core_physics.t0_r4_s2_species import build_s2_species_log_nd_at_time, load_s2_bundle

        bundle = load_s2_bundle()
        if bundle is None:
            raise FileNotFoundError(
                "s2_species requires checkpoint; train with python -m src.training.train_t0_r4_s2_species"
            )
        sp, elig = build_s2_species_log_nd_at_time(
            data,
            int(time_index),
            device,
            phys_cfg,
            bio_cfg,
            bundle,
            commits_prev=commits_prev,
            nucleation_hops=nucleation_hops,
        )
        return sp, elig
    elif step_s == "s3_temporal":
        raise ValueError(
            "s3_temporal requires coupled GRU state; use rollout_rung4_species_series or rollout_s3_species_series"
        )
    elif step_s in ("s4_band_ml", "s4_band_gnn"):
        raise ValueError(
            "s4_band_ml requires coupled GNN rollout; use rollout_rung4_species_series or rollout_s4_species_series"
        )
    else:
        raise ValueError(f"unknown rung4 species step={sp_step!r} (from {step_s!r})")

    return sp, elig


@torch.no_grad()
def rollout_rung4_species_series(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    step: str | None = None,
    nucleation_hops: int = 1,
) -> torch.Tensor:
    """Coupled species rollout; returns ``(T, N, 16)`` y-shaped tensor (ch 4:16 = species)."""
    step_s = (step or rung4_step_from_env()).strip().lower()
    if step_s in ("s0_phi_rule", "s1_mlp_phi"):
        raise ValueError(f"{step_s} bypasses species series; use rollout_rung4_phi_trajectory")

    if step_s in ("species_gnn", "s34_gnn"):
        from src.core_physics.species_gnn_clot_rollout import (
            load_species_gnn_rollout_bundle,
            prepare_species_gnn_rollout_static,
            rollout_species_gnn_species_series,
            species_gnn_rollout_ckpt,
        )

        ckpt = species_gnn_rollout_ckpt()
        bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
        if bundle is None:
            raise FileNotFoundError(f"species_gnn requires checkpoint: {ckpt}")
        static = prepare_species_gnn_rollout_static(data, device=device)
        return rollout_species_gnn_species_series(
            data, bundle, static, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device,
        )

    if step_s in ("s2_species", "s2_loc", "s2_delta"):
        from src.core_physics.t0_r4_s2_species import load_s2_bundle, rollout_s2_species_series

        bundle = load_s2_bundle()
        if bundle is None:
            raise FileNotFoundError(
                "s2_species requires checkpoint; train with python -m src.training.train_t0_r4_s2_species"
            )
        return rollout_s2_species_series(
            data, phys_cfg, bio_cfg, device, bundle, nucleation_hops=nucleation_hops
        )

    if step_s == "s3_temporal":
        from src.core_physics.t0_r4_s3_temporal import load_s3_bundle, rollout_s3_species_series

        bundle = load_s3_bundle()
        if bundle is None:
            raise FileNotFoundError(
                "s3_temporal requires checkpoint; train with python -m src.training.train_t0_r4_s3_temporal"
            )
        return rollout_s3_species_series(
            data, phys_cfg, bio_cfg, device, bundle, nucleation_hops=nucleation_hops
        )

    if step_s in ("s4_band_ml", "s4_band_gnn"):
        from src.core_physics.t0_r4_s4_band_ml import load_s4_bundle, rollout_s4_species_series

        bundle = load_s4_bundle()
        if bundle is None:
            raise FileNotFoundError(
                "s4_band_ml requires checkpoint; train with python -m src.training.train_t0_r4_s4_band_ml"
            )
        return rollout_s4_species_series(
            data, phys_cfg, bio_cfg, device, bundle, nucleation_hops=nucleation_hops
        )

    if step_s in ("s5_gnode_fimat", "s5_fimat"):
        from src.core_physics.t0_r4_s5_fimat import load_s5_bundle, rollout_s5_species_series

        bundle = load_s5_bundle()
        if bundle is None:
            raise FileNotFoundError(
                "s5_gnode_fimat requires checkpoint; train with scripts/go_t0_rung4_s5.ps1"
            )
        return rollout_s5_species_series(
            data, phys_cfg, bio_cfg, device, bundle, nucleation_hops=nucleation_hops
        )

    _sweep_step = step_s not in ("s_star_g0_rules",) and (
        step_s.startswith("s_star") or step_s.startswith("s4_")
    )
    if _sweep_step:
        from src.core_physics.t0_r4_sweep import load_sweep_bundle, rollout_sweep_species_series

        bundle = load_sweep_bundle()
        if bundle is None:
            raise FileNotFoundError(
                f"{step_s} requires sweep checkpoint; set T0_R4_SWEEP_CKPT or train sweep leg"
            )
        return rollout_sweep_species_series(
            data, phys_cfg, bio_cfg, device, bundle, nucleation_hops=nucleation_hops
        )

    out = data.y.clone().to(device=device)
    commits_prev: torch.Tensor | None = None

    from src.core_physics.t0_mu_physics import predict_clot_phi_at_time
    from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env

    with t0_rung2_env():
        for t in range(int(data.y.shape[0])):
            sp, _ = build_rung4_species_log_nd_at_time(
                data,
                t,
                device,
                phys_cfg,
                bio_cfg,
                commits_prev=commits_prev,
                step=step_s,
                nucleation_hops=nucleation_hops,
            )
            out[t, :, 4:16] = sp
            phi_raw, _ = predict_clot_phi_at_time(
                data,
                t,
                phys_cfg,
                bio_cfg,
                device,
                gamma_mode=RUNG2_GAMMA_MODE,
                flow_source="gt",
                pred_species_series=out,
            )
            commits_prev = (phi_raw.reshape(-1) >= 0.5).bool()
    return out


@torch.no_grad()
def rollout_rung4_phi_trajectory(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    step: str | None = None,
    nucleation_hops: int = 1,
) -> dict[int, torch.Tensor]:
    """Rollout clot phi per step (species path or s0_phi_rule bypass)."""
    from src.core_physics.clot_nucleation_mask import project_phi_with_nucleation
    from src.core_physics.t0_mu_physics import predict_clot_phi_at_time, rollout_t0_clot_phi
    from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env

    step_s = (step or rung4_step_from_env()).strip().lower()

    if step_s == "s0_phi_rule":
        n_steps = int(data.y.shape[0])
        phi_prev: torch.Tensor | None = None
        commits_prev: torch.Tensor | None = None
        out: dict[int, torch.Tensor] = {}
        with t0_rung2_env():
            for t in range(n_steps):
                phi_raw, _score = predict_phi_prior_rule(
                    data, device, bio_cfg, t_in=t
                )
                elig = resolve_nucleation_eligibility(
                    data,
                    t,
                    device,
                    phys_cfg,
                    bio_cfg,
                    commits_prev=commits_prev,
                    growth_seed="pred",
                    nucleation_hops=nucleation_hops,
                )
                phi = project_phi_with_nucleation(phi_raw.reshape(-1), phi_prev, elig)
                out[t] = phi
                commits_prev = (phi.reshape(-1) >= 0.5).bool()
                phi_prev = phi.detach()
        return out

    if step_s == "s1_mlp_phi":
        from src.core_physics.t0_r4_s1_mlp_phi import load_s1_bundle, rollout_s1_phi_trajectory

        bundle = load_s1_bundle()
        if bundle is None:
            raise FileNotFoundError(
                "s1_mlp_phi requires checkpoint; train with "
                "python -m src.training.train_t0_r4_s1_mlp_phi"
            )
        return rollout_s1_phi_trajectory(
            data, phys_cfg, bio_cfg, device, bundle, nucleation_hops=nucleation_hops
        )

    if step_s in ("species_gnn", "s34_gnn"):
        from src.core_physics.species_gnn_clot_rollout import (
            load_species_gnn_rollout_bundle,
            prepare_species_gnn_rollout_static,
            rollout_species_gnn_phi_trajectory,
            species_gnn_rollout_ckpt,
        )

        flow_raw = (os.environ.get("T0_R4_FLOW_SOURCE") or "gt").strip().lower()
        ckpt = species_gnn_rollout_ckpt()
        bundle = load_species_gnn_rollout_bundle(ckpt, device=device)
        if bundle is None:
            raise FileNotFoundError(f"species_gnn requires checkpoint: {ckpt}")
        static = prepare_species_gnn_rollout_static(data, device=device)
        return rollout_species_gnn_phi_trajectory(
            data,
            bundle,
            static,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            flow_source=flow_raw,
        )

    pred_series = rollout_rung4_species_series(
        data, phys_cfg, bio_cfg, device, step=step_s, nucleation_hops=nucleation_hops
    )
    flow_raw = (os.environ.get("T0_R4_FLOW_SOURCE") or "gt").strip().lower()
    flow = "kinematics" if flow_raw in ("pred", "kine", "kinematics", "deq", "gino") else "gt"
    gel_beta = None
    if os.environ.get("SPECIES_VISCOSITY_CALIB", "").strip().lower() in ("1", "true", "yes", "on"):
        from src.core_physics.species_viscosity_calibration import (
            load_viscosity_calibration,
            viscosity_calibration_dir,
        )
        from pathlib import Path

        cal_path = os.environ.get("SPECIES_VISCOSITY_CALIB_PATH") or str(
            viscosity_calibration_dir() / "beta.pth"
        )
        if Path(cal_path).is_file():
            cal, _ = load_viscosity_calibration(cal_path, device=device)
            gel_beta = cal.beta
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data,
            phys_cfg,
            bio_cfg,
            device,
            gamma_mode=RUNG2_GAMMA_MODE,
            flow_source=flow,
            pred_species_series=pred_series,
            nucleation=True,
            nucleation_hops=nucleation_hops,
            gelation_beta=gel_beta,
        )
    return {t: v["phi"] for t, v in traj.items()}


S0_RULE_ENV_KEYS: tuple[str, ...] = (
    "T0_R4_S0_SPATIAL_TOP_FRAC",
    "T0_R4_S0_ONSET_TAU_START",
    "T0_R4_S0_ONSET_TAU_END",
    "T0_R4_S0_FI_MAT_GAIN",
    "T0_R4_S0_RISK_NORM_PCTL",
    "T0_R4_S0_SPREAD_HOPS",
    "T0_R4_S0_SPREAD_DECAY",
    "T0_R4_S0_ONSET_POWER",
)


@torch.no_grad()
def eval_rung4_step_clot(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    step: str = "s0",
    times: list[int] | None = None,
) -> dict:
    """Deploy-faithful clot metrics: nucleation-projected phi trajectory + health."""
    from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
    from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health
    from src.training.train_clot_phi_simple import _clot_metrics

    n_steps = int(data.y.shape[0])
    times = sorted(set(times or [0, n_steps - 1]))
    phi_traj = rollout_rung4_phi_trajectory(
        data, phys_cfg, bio_cfg, device, step=step,
    )
    mask = torch.ones(int(data.num_nodes), device=device, dtype=torch.bool)
    rows: list[dict] = []
    for t in times:
        phi_gt = gt_clot_phi_at_time(data, t, phys_cfg, device)
        m = _clot_metrics(
            phi_traj[int(t)].reshape(-1),
            phi_gt.reshape(-1),
            mask,
        )
        rows.append({"time": int(t), **m})
    health = compute_rung4_rollout_health(
        phi_traj, data, phys_cfg, bio_cfg, device, times=times,
    )
    return {
        "step": str(step),
        "clot": rows,
        "rollout_health": {k: v for k, v in health.items() if k != "timeline"},
        "health_timeline": health.get("timeline", []),
    }


@dataclass(frozen=True)
class Rung4StepInfo:
    step: str
    deploy: bool
    uses_gt_species: bool
    description: str


def describe_rung4_step(step: str | None = None) -> Rung4StepInfo:
    s = (step or rung4_step_from_env()).strip().lower()
    m = rung4_step_meta(s)
    return Rung4StepInfo(
        step=s,
        deploy=bool(m.get("deploy", False)),
        uses_gt_species=bool(m.get("uses_gt_species", False)),
        description=str(m.get("desc", "")),
    )
