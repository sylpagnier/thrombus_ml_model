"""Apply clot-phi training env vars from a saved checkpoint config."""

from __future__ import annotations

import os
from typing import Any


def apply_clot_phi_config_from_checkpoint(cfg: dict[str, Any]) -> None:
    """Mirror train/viz env so eval matches the training run."""
    if not cfg:
        return
    mapping = {
        "mu_cap_si": "CLOT_PHI_MU_CAP_SI",
        "mu_thresh_si": "CLOT_PHI_THRESH_SI",
        "oracle_mu": "CLOT_PHI_ORACLE_MU",
        "species_features": "CLOT_PHI_SPECIES_FEATURES",
        "joint_bio": "CLOT_PHI_JOINT_BIO",
        "use_prior_features": "CLOT_PHI_USE_PRIOR_FEATURES",
        "prior_n": "CLOT_PHI_PRIOR_N",
        "hybrid": "CLOT_PHI_HYBRID",
        "minimal_features": "CLOT_PHI_MINIMAL_FEATURES",
        "dropout": "CLOT_PHI_DROPOUT",
        "mlp_depth": "CLOT_PHI_MLP_DEPTH",
        "mu_log_lambda": "CLOT_PHI_MU_LOG_LAMBDA",
        "bio_lambda": "CLOT_PHI_BIO_LAMBDA",
        "physics_blend_alpha": "CLOT_PHI_PHYSICS_BLEND_ALPHA",
    }
    for key, env_name in mapping.items():
        if key in cfg:
            val = cfg[key]
            if isinstance(val, bool):
                os.environ[env_name] = "1" if val else "0"
            else:
                os.environ[env_name] = str(val)
    if cfg.get("anchor_dir"):
        os.environ["CLOT_PHI_ANCHOR_DIR"] = str(cfg["anchor_dir"])
    if "regression_only" in cfg:
        os.environ["CLOT_PHI_REGRESSION_ONLY"] = "1" if cfg["regression_only"] else "0"
    if cfg.get("physics_blend"):
        os.environ["CLOT_PHI_PHYSICS_BLEND"] = "1"
    elif "physics_blend" in cfg:
        os.environ["CLOT_PHI_PHYSICS_BLEND"] = "0"
    if cfg.get("clot_trigger_star"):
        os.environ["CLOT_TRIGGER_STAR"] = str(cfg["clot_trigger_star"])
    if "rollout" in cfg:
        os.environ["CLOT_PHI_ROLLOUT"] = "1" if cfg["rollout"] else "0"
    if cfg.get("rollout_vel_source"):
        os.environ["CLOT_PHI_VEL_SOURCE"] = str(cfg["rollout_vel_source"])
    if "rollout_carry_phi" in cfg:
        os.environ["CLOT_PHI_CARRY_PHI"] = "1" if cfg["rollout_carry_phi"] else "0"
    if "rollout_carry_log_mu" in cfg:
        os.environ["CLOT_PHI_CARRY_LOG_MU"] = "1" if cfg["rollout_carry_log_mu"] else "0"
    if "rollout_detach" in cfg:
        os.environ["CLOT_PHI_ROLLOUT_DETACH"] = "1" if cfg["rollout_detach"] else "0"
    if "fixed_mu_from_phi" in cfg:
        os.environ["CLOT_PHI_FIXED_MU_FROM_PHI"] = "1" if cfg["fixed_mu_from_phi"] else "0"
    if cfg.get("mu_solid_si") is not None:
        os.environ["CLOT_PHI_MU_SOLID_SI"] = str(cfg["mu_solid_si"])
    if cfg.get("mesh_aux_lambda") is not None:
        os.environ["CLOT_PHI_MESH_AUX_LAMBDA"] = str(cfg["mesh_aux_lambda"])
    if cfg.get("mesh_bulk_lambda") is not None:
        os.environ["CLOT_PHI_MESH_BULK_LAMBDA"] = str(cfg["mesh_bulk_lambda"])
    if "shape_use_t_out_mu" in cfg:
        os.environ["CLOT_PHI_SHAPE_USE_T_OUT"] = "1" if cfg["shape_use_t_out_mu"] else "0"
    if cfg.get("dgamma_feature_time"):
        os.environ["CLOT_PHI_DGAMMA_FEATURE_TIME"] = str(cfg["dgamma_feature_time"])
    if cfg.get("forecast_mask"):
        os.environ["CLOT_FORECAST_MASK"] = str(cfg["forecast_mask"])
    if cfg.get("forecast_one_step"):
        os.environ["CLOT_FORECAST_MODE"] = "one_step"
    if "forecast_input_mu" in cfg:
        os.environ["CLOT_FORECAST_INPUT_MU"] = "1" if cfg["forecast_input_mu"] else "0"
    if "forecast_mu_carry" in cfg:
        os.environ["CLOT_FORECAST_MU_CARRY"] = "1" if cfg["forecast_mu_carry"] else "0"
    if "forecast_mu_carry_detach" in cfg:
        os.environ["CLOT_FORECAST_MU_CARRY_DETACH"] = "1" if cfg["forecast_mu_carry_detach"] else "0"
    if cfg.get("forecast_mu_init"):
        os.environ["CLOT_FORECAST_MU_INIT"] = str(cfg["forecast_mu_init"])
    if cfg.get("forecast_pair_schedule"):
        os.environ["CLOT_FORECAST_PAIR_SCHEDULE"] = str(cfg["forecast_pair_schedule"])
    if cfg.get("forecast_pair_stride") is not None:
        os.environ["CLOT_FORECAST_PAIR_STRIDE"] = str(cfg["forecast_pair_stride"])
    if "hard_support_projection" in cfg:
        os.environ["CLOT_PHI_HARD_SUPPORT_PROJECTION"] = "1" if cfg["hard_support_projection"] else "0"
    if cfg.get("support_band"):
        os.environ["CLOT_PHI_SUPPORT_BAND"] = str(cfg["support_band"])
    if cfg.get("ceiling_hops") is not None:
        os.environ["CLOT_PHI_CEILING_HOPS"] = str(cfg["ceiling_hops"])
    if cfg.get("growth_seed"):
        os.environ["CLOT_PHI_GROWTH_SEED"] = str(cfg["growth_seed"])
    if cfg.get("loss_scope"):
        os.environ["CLOT_PHI_LOSS_SCOPE"] = str(cfg["loss_scope"])
    if "trigger_rollout" in cfg:
        os.environ["CLOT_PHI_TRIGGER_ROLLOUT"] = "1" if cfg["trigger_rollout"] else "0"
    if cfg.get("forward_seed"):
        os.environ["CLOT_TRIGGER_FORWARD_SEED"] = str(cfg["forward_seed"])
    if "nucleation" in cfg:
        os.environ["CLOT_TRIGGER_NUCLEATION"] = "1" if cfg["nucleation"] else "0"
    if cfg.get("ic_phi_zero") is not None:
        os.environ["CLOT_TRIGGER_IC_PHI_ZERO"] = "1" if cfg["ic_phi_zero"] else "0"
    from src.core_physics.clot_phi_rollout import sync_rollout_env_from_checkpoint

    sync_rollout_env_from_checkpoint(cfg)


def apply_clot_phi_eval_defaults() -> None:
    """Env from ``go_clot_phi_from_anchor_dir`` / ``_clot_phi_shared_env`` not stored in ckpt.

    Uses setdefault so checkpoint config and explicit env win.
    """
    defaults = {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_WALL_HOPS": "1",
        "CLOT_PHI_CLOT_HOPS": "2",
        "CLOT_PHI_CLOT_TOUCH_HOPS": "1",
        "CLOT_PHI_CENTER_EXCLUDE_FRAC": "0.10",
        "CLOT_PHI_DGAMMA_SLICE": "1",
        "CLOT_PHI_DGAMMA_WALL_MIN_SI": "100",
        "CLOT_PHI_DGAMMA_OFFWALL_PCT": "80",
        "CLOT_PHI_SHEAR_MIN_FRAC": "0",
        "CLOT_PHI_SOFT_LABELS": "1",
        "CLOT_PHI_BALANCED": "1",
        "CLOT_PHI_JOINT_USE_PRED_SPECIES": "1",
        "CLOT_PHI_PHYSICS_GELATION_GATE": "1",
        "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
        "CLOT_PHI_PHYSICS_BLEND_ALPHA": "0.75",
        "CLOT_PHI_ANCHOR_BALANCED": "1",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)
    if not (os.environ.get("CLOT_PHI_DGAMMA_FEATURE_TIME") or "").strip():
        os.environ.setdefault("CLOT_PHI_DGAMMA_FEATURE_TIME", "ref")
