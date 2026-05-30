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
