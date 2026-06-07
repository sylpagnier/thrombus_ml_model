"""Canonical Leg B deploy (no GT clot mask) env for closed-loop MLP mu map."""

from __future__ import annotations

import os
from typing import Any

from pathlib import Path

# Default: neighbor commit inside GT supervision band @ COMSOL t=0 (wall, 1-hop, dgamma
# via CLOT_PHI_* from clot checkpoint). Vision grows 1-hop only after pred clot inside mask.
# Commit dgamma slice stays OFF (BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE=0); dgamma in clot-phi only.
DEPLOY_MU_MAP_ENV: dict[str, str] = {
    "BIOCHEM_MLP_MU_MAP": "1",
    "BIOCHEM_MLP_MU_MAP_PHI_GATE": "1",
    "BIOCHEM_MLP_MU_MAP_MASK": "neighbor",
    "BIOCHEM_MLP_MU_MAP_BULK": "cap_low_shear",
    "BIOCHEM_MLP_MU_MAP_GAMMA_THRESH_ND": "0.01",
    "BIOCHEM_MLP_MU_MAP_GEO_CAP": "0",
    "BIOCHEM_MLP_NEIGHBOR_SEED": "pred_clot",
    "BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI": "1",
    "BIOCHEM_MLP_MU_MAP_PHI_THRESH": "0.5",
    "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "0",
    "BIOCHEM_MLP_DEPLOY_PHI_Q": "0",
    "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "0",
    "BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI": "0",
    "BIOCHEM_MLP_DEPLOY_VISION_RESTRICT": "1",
    "BIOCHEM_MLP_DEPLOY_VISION_INIT": "comsol_t0",
    "BIOCHEM_MLP_DEPLOY_VISION_GROW": "1",
    "BIOCHEM_MLP_DEPLOY_VISION_GROW_HOPS": "1",
    "BIOCHEM_MLP_DEPLOY_NO_COMMIT_T0": "1",
    "CLOT_SHAPE_MU_THRESH_SI": "0.055",
}

# Same t=0 supervision vision; commit = allowed & phi (no neighbor flood).
SEED_GROWTH_MU_MAP_ENV: dict[str, str] = {
    **DEPLOY_MU_MAP_ENV,
    "BIOCHEM_MLP_MU_MAP_MASK": "seed_growth",
    "BIOCHEM_MLP_SEED_GROWTH_INIT": "comsol_t0",
    "BIOCHEM_MLP_SEED_GROWTH_HOPS": "1",
    "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "0",
    "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "0",
    "BIOCHEM_MLP_DEPLOY_PHI_Q": "0",
    "BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI": "0",
    "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "0",
}


def apply_deploy_mu_map_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Set deploy Leg B env; returns merged dict applied."""
    merged = dict(DEPLOY_MU_MAP_ENV)
    if overrides:
        merged.update({k: str(v) for k, v in overrides.items()})
    for k, v in merged.items():
        os.environ[k] = str(v)
    return merged


def apply_seed_growth_mu_map_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Set seed-growth deploy env (GT t=0 vision + pred-clot 1-hop expansion)."""
    merged = dict(SEED_GROWTH_MU_MAP_ENV)
    if overrides:
        merged.update({k: str(v) for k, v in overrides.items()})
    clear_oracle_mu_map_env()
    for k, v in merged.items():
        os.environ[k] = str(v)
    return merged


def clear_oracle_mu_map_env() -> None:
    """Remove gt_clot-only knobs before applying deploy env."""
    for key in (
        "BIOCHEM_MLP_CLOT_INJECT",
        "BIOCHEM_MU_NEIGHBOR_WALL_ONLY",
        "BIOCHEM_MLP_CLOT_REGION",
    ):
        os.environ.pop(key, None)


# Wired deploy: MLP mu commits inside GT supervision vision (t=0) when phi+mu_mlp pass.
# Closer to offline clot-map readout than neighbor wall nucleation; still no gt_clot labels.
WIRED_DEPLOY_MU_MAP_ENV: dict[str, str] = {
    **DEPLOY_MU_MAP_ENV,
    "BIOCHEM_MLP_MU_MAP_MASK": "mlp_band",
    "BIOCHEM_MLP_NEIGHBOR_SEED": "pred_clot",
    "BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI": "1",
    "BIOCHEM_MLP_MU_MAP_PHI_THRESH": "0.5",
    "BIOCHEM_MLP_DEPLOY_VISION_RESTRICT": "1",
    "BIOCHEM_MLP_DEPLOY_VISION_INIT": "comsol_t0",
    "BIOCHEM_MLP_DEPLOY_VISION_GROW": "1",
    "BIOCHEM_MLP_DEPLOY_VISION_GROW_HOPS": "1",
    "BIOCHEM_MLP_DEPLOY_NO_COMMIT_T0": "1",
}


def apply_wired_deploy_mu_map_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Leg B wired: closed-loop MLP mu map with ``mlp_band`` commit inside deploy vision."""
    merged = dict(WIRED_DEPLOY_MU_MAP_ENV)
    if overrides:
        merged.update({k: str(v) for k, v in overrides.items()})
    clear_oracle_mu_map_env()
    for k, v in merged.items():
        os.environ[k] = str(v)
    return merged


def wire_deploy_mu_map(
    *,
    clot_ckpt: str | Path | None = None,
    wired: bool = True,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Apply deploy env + optional clot ckpt path for closed-loop MLP mu coupling."""
    apply_fn = apply_wired_deploy_mu_map_env if wired else apply_deploy_mu_map_env
    merged = apply_fn(overrides)
    if clot_ckpt is not None:
        os.environ["BIOCHEM_MLP_CLOT_CKPT"] = str(clot_ckpt)
    elif not (os.environ.get("BIOCHEM_MLP_CLOT_CKPT") or "").strip():
        from src.core_physics.clot_phi_mu_inject import resolve_mlp_clot_ckpt

        resolved = resolve_mlp_clot_ckpt(None)
        if resolved is not None:
            os.environ["BIOCHEM_MLP_CLOT_CKPT"] = str(resolved)
    return merged


def deploy_env_for_manifest() -> dict[str, Any]:
    return {"deploy_mu_map_env": dict(DEPLOY_MU_MAP_ENV), "leg": "B_deploy"}


def deploy_env_for_manifest_wired() -> dict[str, Any]:
    return {"deploy_mu_map_env": dict(WIRED_DEPLOY_MU_MAP_ENV), "leg": "B_wired"}
