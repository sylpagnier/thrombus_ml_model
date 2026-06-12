"""T0 physics trigger sweep legs (env presets for ~15 min LOAO search)."""

from __future__ import annotations

import os
from typing import Any

# Keys managed by sweep (cleared between legs).
_SWEEP_ENV_KEYS = (
    "CLOT_PHI_PHYSICS_MU_BASE",
    "CLOT_PHI_PHYSICS_MU_RATIO_MAX",
    "CLOT_PHI_PHYSICS_HARD_STEP",
    "CLOT_PHI_PHYSICS_GELATION_GATE",
    "CLOT_PHI_PHYSICS_WALL_MAT_ONLY",
    "CLOT_PHI_PHYSICS_GELATION_ONSET_FRAC",
    "CLOT_PHI_PHYSICS_MU2_CAP",
    "CLOT_PHI_PHYSICS_GAMMA_MODE",
    "CLOT_PHI_THRESH_SI",
    "CLOT_TRIGGER_GT_MU_ORACLE",
)


def clear_physics_sweep_env() -> None:
    for key in _SWEEP_ENV_KEYS:
        os.environ.pop(key, None)


def apply_physics_sweep_leg(leg: dict[str, Any]) -> None:
    clear_physics_sweep_env()
    env = dict(leg.get("env") or {})
    for key, val in env.items():
        os.environ[str(key)] = str(val)
    if leg.get("gt_mu_oracle"):
        os.environ["CLOT_TRIGGER_GT_MU_ORACLE"] = "1"


def physics_sweep_legs() -> list[dict[str, Any]]:
    """Ordered legs for T0 physics baseline search."""
    return [
        {
            "id": "A_comsol_carreau_max",
            "note": "COMSOL spf.mu: gel-scaled Carreau, gamma=max(graph,poiseuille,|u|/width)",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol_carreau",
                "CLOT_PHI_PHYSICS_GAMMA_MODE": "max",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
            },
        },
        {
            "id": "A_legacy_carreau",
            "note": "Legacy: fixed Carreau * (1+gel) delta above t0 mu",
            "env": {"CLOT_PHI_PHYSICS_MU_BASE": "carreau", "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4"},
        },
        {
            "id": "B_comsol_soft_r80",
            "note": "COMSOL mu_b*(mu1+mu2), soft steps, ratio=80",
            "env": {"CLOT_PHI_PHYSICS_MU_BASE": "comsol", "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "80"},
        },
        {
            "id": "C_comsol_soft_r4",
            "note": "COMSOL formula, ratio=4 (deploy-scale gelation)",
            "env": {"CLOT_PHI_PHYSICS_MU_BASE": "comsol", "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4"},
        },
        {
            "id": "D_blood_additive",
            "note": "mu_inf*(1+gel), ratio=4",
            "env": {"CLOT_PHI_PHYSICS_MU_BASE": "blood", "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4"},
        },
        {
            "id": "E_comsol_hard_r80",
            "note": "COMSOL hard steps, ratio=80",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "80",
                "CLOT_PHI_PHYSICS_HARD_STEP": "1",
            },
        },
        {
            "id": "F_prior_gate_carreau",
            "note": "Carreau + shear/dgamma prior gate on gelation",
            "env": {"CLOT_PHI_PHYSICS_GELATION_GATE": "1"},
        },
        {
            "id": "G_comsol_prior_r4",
            "note": "COMSOL r4 + prior gate",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
                "CLOT_PHI_PHYSICS_GELATION_GATE": "1",
            },
        },
        {
            "id": "H_comsol_wall_mat",
            "note": "COMSOL r4; mu1(Mat) only on geometry wall",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
            },
        },
        {
            "id": "I_comsol_onset_022",
            "note": "COMSOL r4 + step2t-like onset (tau~0.22)",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
                "CLOT_PHI_PHYSICS_GELATION_ONSET_FRAC": "0.22",
            },
        },
        {
            "id": "J_comsol_onset_gate",
            "note": "COMSOL r4 + onset + prior gate",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
                "CLOT_PHI_PHYSICS_GELATION_ONSET_FRAC": "0.22",
                "CLOT_PHI_PHYSICS_GELATION_GATE": "1",
            },
        },
        {
            "id": "K_comsol_mu2_cap1",
            "note": "COMSOL r4; cap fibrin leg",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
                "CLOT_PHI_PHYSICS_MU2_CAP": "1",
            },
        },
        {
            "id": "L_comsol_thresh_070",
            "note": "COMSOL r4; higher clot mu threshold",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
                "CLOT_PHI_THRESH_SI": "0.070",
            },
        },
        {
            "id": "M_wall_onset_gate",
            "note": "COMSOL r4 + wall Mat + onset + gate",
            "env": {
                "CLOT_PHI_PHYSICS_MU_BASE": "comsol",
                "CLOT_PHI_PHYSICS_MU_RATIO_MAX": "4",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_PHYSICS_GELATION_ONSET_FRAC": "0.22",
                "CLOT_PHI_PHYSICS_GELATION_GATE": "1",
            },
        },
        {
            "id": "A_legacy_carreau_no_t0",
            "note": "Legacy: no t0 mu subtract (ablation only)",
            "env": {
                "CLOT_PHI_PHYSICS_SUBTRACT_T0_MU": "0",
                "CLOT_TRIGGER_IC_PHI_ZERO": "0",
            },
        },
        {
            "id": "N_subtract_t0_mu",
            "note": "Alias: explicit t0 mu subtract (same as default A)",
            "env": {"CLOT_PHI_PHYSICS_SUBTRACT_T0_MU": "1", "CLOT_TRIGGER_IC_PHI_ZERO": "1"},
        },
        {
            "id": "Z_gt_mu_oracle",
            "note": "Upper bound: phi from GT mu_eff (not physics)",
            "env": {},
            "gt_mu_oracle": True,
        },
    ]


def sweep_score(row: dict[str, float]) -> float:
    """Rank legs: full-mesh F1 primary, penalize lumen FP."""
    full = float(row.get("mean_full_mesh_f1", float("nan")))
    lumen = float(row.get("mean_lumen_fp_deploy", float("nan")))
    sup = float(row.get("mean_support_f1", float("nan")))
    if not (full == full):  # nan
        return float("-inf")
    lumen_pen = lumen if lumen == lumen else 0.0
    return full - 0.35 * lumen_pen + 0.05 * (sup if sup == sup else 0.0)
