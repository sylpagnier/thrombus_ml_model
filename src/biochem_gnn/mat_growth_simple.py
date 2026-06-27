"""Minimal Mat-only single-head pushforward recipe (wall+3hop band).

Predicts per-step Mat log-delta on the wall-band subgraph; clot eval uses analytical
``mu1(Mat)`` gelation (no FI channel, no dual spatial/magnitude heads).

Compare against ``triangle6_wall3hop_20260624`` (fi_mat dual-head baseline).

Ladder legs (``go_mat_growth_ladder.ps1``):

  A_random   - fresh random init (simplest baseline)
  B_backbone - SAGE conv warm-start from triangle6 ``species/best.pth`` (fresh readout)
  C_geom          - fresh random init + static geometry feats (``SPECIES_GEOM_FEATS``)
  D_parity_single - baseline-like dynamics, but Mat-only + single-head
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.biochem_gnn.config import GLOBAL_TRAIN_RECIPE, apply_train_recipe_env, global_ckpt_path
from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL

# Overrides on top of the locked deploy recipe (triangle6 / wall+3hop topology).
# Precision-first defaults: checkpoint selection uses relaxed precision (not recall-heavy
# deploy_mat_f1 alone). Legs inherit unless they explicitly override a knob.
MAT_GROWTH_SIMPLE_RECIPE: dict[str, str] = {
    "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
    "SPECIES_CONTINUOUS_DUAL_HEAD": "0",
    "SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS": "1",
    "SPECIES_CONTINUOUS_PHYSICS_READOUT": "0",
    "SPECIES_CONTINUOUS_SATURATION_GATE": "0",
    "SPECIES_CONTINUOUS_DELTA_RESIDUAL": "0",
    "SPECIES_CONTINUOUS_TEMPORAL_OFFSET": "0",
    "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "8.0",
    "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "2.0",
    "SPECIES_CONTINUOUS_FP_WEIGHT": "16",
    "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "2.0",
    "SPECIES_CONTINUOUS_SPEED_FP_WEIGHT": "6.0",
    "SPECIES_CONTINUOUS_GATE_FP_WEIGHT": "4.0",
    "SPECIES_CONTINUOUS_CLOUT_SCORE": "relaxed_prec_floor",
    "SPECIES_CLOUT_PREC_REC_FLOOR": "0.35",
    "SPECIES_CONTINUOUS_SCORE_CLOUT_W": "0.75",
    "SPECIES_MAT_GROWTH_PRECISION_SELECT": "1",
    "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
    "SPECIES_VISCOSITY_CALIB": "0",
}

BASELINE_COMPARE_ID = "triangle6_wall3hop_20260624"
DEFAULT_OUT_DIR = "outputs/biochem/biochem_gnn/mat_growth_simple"
DEFAULT_CKPT = f"{DEFAULT_OUT_DIR}/best.pth"
LADDER_ROOT = "outputs/biochem/biochem_gnn/mat_growth_ladder"
LADDER_LEG_ORDER: tuple[str, ...] = (
    "A_random",
    "B_backbone",
    "C_geom",
    "D_parity_single",
    "E_dual_mat",
    "F_single_fimat",
    "G_dual_mat_neighbor_gate",
    "H_dual_mat_crit_focus",
    "I_dual_fimat_fi_aux",
    "J_dual_mat_neighbor_crit",
    "K_fimat_neighbor_gate",
    "L_fimat_geom_rich",
    "M_fimat_neighbor_geom_rich",
    "N_mat_geom_rich",
    "O_mat_neighbor_geom_rich",
    "P_mat_plain",
    "Q_mat_gate_sharp_fp",
    "R_mat_geom_gate_sharp_fp",
    "U_mat_frontier_only",
    "V_mat_frontier_geom",
    "W_mat_flow_stagnation",
    "X_mat_flow_seedfront",
    "Y_mat_tight_seed",
    "AB_mat_gelation_aux",
    "S_mat_frontier_nuc",
    "T_mat_frontier_sharp",
)


@dataclass(frozen=True)
class MatGrowthLegSpec:
    code: str
    label: str
    no_init: bool
    init_ckpt: str
    init_mode: str  # full | backbone | mat_readout
    env_overrides: dict[str, str]


def mat_growth_leg_spec(leg: str) -> MatGrowthLegSpec:
    code = leg.strip()
    init_default = str(global_ckpt_path()).replace("\\", "/")
    specs: dict[str, MatGrowthLegSpec] = {
        "A_random": MatGrowthLegSpec(
            code="A_random",
            label="random init (Mat-only single-head)",
            no_init=True,
            init_ckpt="",
            init_mode="full",
            env_overrides={},
        ),
        "B_backbone": MatGrowthLegSpec(
            code="B_backbone",
            label="backbone warm-start from triangle6 species/best.pth",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={},
        ),
        "C_geom": MatGrowthLegSpec(
            code="C_geom",
            label="random init + static geometry feats",
            no_init=True,
            init_ckpt="",
            init_mode="full",
            env_overrides={"SPECIES_GEOM_FEATS": "1"},
        ),
        "D_parity_single": MatGrowthLegSpec(
            code="D_parity_single",
            label="baseline-like dynamics, single-head Mat-only",
            no_init=False,
            init_ckpt=init_default,
            init_mode="mat_readout",
            env_overrides={
                # Keep baseline dynamics as much as possible while forcing single-head Mat-only.
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
            },
        ),
        "E_dual_mat": MatGrowthLegSpec(
            code="E_dual_mat",
            label="baseline-like dynamics, dual-head Mat-only",
            no_init=False,
            init_ckpt=init_default,
            init_mode="mat_readout",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
            },
        ),
        "F_single_fimat": MatGrowthLegSpec(
            code="F_single_fimat",
            label="baseline-like dynamics, single-head fi_mat",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "0",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "fi_mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                # restore baseline-like per-channel weighting for fi_mat.
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "4.0",
            },
        ),
        "G_dual_mat_neighbor_gate": MatGrowthLegSpec(
            code="G_dual_mat_neighbor_gate",
            label="dual-head Mat-only + neighbor commit-aware spatial gate",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
            },
        ),
        "H_dual_mat_crit_focus": MatGrowthLegSpec(
            code="H_dual_mat_crit_focus",
            label="dual-head Mat-only + crit-focused loss weighting",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_DELTA_THRESH_MAT": "2.5e-6",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "5.0",
                "SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT": "0.5",
            },
        ),
        "I_dual_fimat_fi_aux": MatGrowthLegSpec(
            code="I_dual_fimat_fi_aux",
            label="dual-head fi_mat with FI as light auxiliary target",
            no_init=False,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "fi_mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_FI": "0.15",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "8.0",
            },
        ),
        "J_dual_mat_neighbor_crit": MatGrowthLegSpec(
            code="J_dual_mat_neighbor_crit",
            label="dual-head Mat-only + neighbor gate + crit-focused loss (G+H)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_CONTINUOUS_DELTA_THRESH_MAT": "2.5e-6",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "5.0",
                "SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT": "0.5",
            },
        ),
        # ---- Precision sweep (in-training levers; vs baseline_fast dual fi_mat) ----
        # Hypothesis (docs/SPECIES_LEARNING_STRATEGY.md s6.13): geometry+kine is near its
        # deployable ranking ceiling, so the remaining gains come from (a) keeping the
        # autocatalytic neighbour coupling but on the *dual fi_mat* head (not Mat-only), and
        # (b) enriching the static geometry context with the proven 2-hop commit-vs-eligible
        # discriminators. Each leg flips exactly one of these on the fi_mat baseline so the
        # delta is attributable.
        "K_fimat_neighbor_gate": MatGrowthLegSpec(
            code="K_fimat_neighbor_gate",
            label="dual fi_mat + neighbor commit gate (autocatalysis on the full head)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "fi_mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_FI": "0.15",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "8.0",
            },
        ),
        "L_fimat_geom_rich": MatGrowthLegSpec(
            code="L_fimat_geom_rich",
            label="dual fi_mat + enriched geometry (2-hop expansion / curvature)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "fi_mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_GEOM_FEATS_RICH": "1",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_FI": "0.15",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "8.0",
            },
        ),
        "M_fimat_neighbor_geom_rich": MatGrowthLegSpec(
            code="M_fimat_neighbor_geom_rich",
            label="dual fi_mat + neighbor gate + enriched geometry (combined surviving levers)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "fi_mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_GEOM_FEATS_RICH": "1",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_FI": "0.15",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "8.0",
            },
        ),
        "N_mat_geom_rich": MatGrowthLegSpec(
            code="N_mat_geom_rich",
            label="dual Mat-only + enriched geometry (leg C scope at 2-hop geometry)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_GEOM_FEATS_RICH": "1",
            },
        ),
        "O_mat_neighbor_geom_rich": MatGrowthLegSpec(
            code="O_mat_neighbor_geom_rich",
            label="dual Mat-only + neighbor gate + rich geometry (N + G)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_GEOM_FEATS_RICH": "1",
            },
        ),
        # ---- 6h precision ladder (6.16): attribution control + gate-precision levers ----
        # Diagnosis 6.16: precision lives in the SPATIAL GATE, not the rate head; the failure is a
        # ranking/over-paint problem on the wall, amplified by monotone autocatalytic lock-in. So
        # the new levers all act on the gate: sharpen it (temperature), pressure gate-positives on
        # zero-growth nodes (spatial focal weight + gamma), on the proven Mat-only dual head.
        "P_mat_plain": MatGrowthLegSpec(
            code="P_mat_plain",
            label="dual Mat-only, NO gate / NO geom (pure-scope attribution control)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
            },
        ),
        "Q_mat_gate_sharp_fp": MatGrowthLegSpec(
            code="Q_mat_gate_sharp_fp",
            label="dual Mat-only + neighbor gate + SHARP gate (temp 0.5) + spatial FP pressure",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                # gate-precision levers:
                "SPECIES_CONTINUOUS_GATE_TEMP": "0.5",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "3.0",
                "SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT": "3.0",
            },
        ),
        "R_mat_geom_gate_sharp_fp": MatGrowthLegSpec(
            code="R_mat_geom_gate_sharp_fp",
            label="Q + rich geometry (all surviving gate-precision levers stacked)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_GEOM_FEATS_RICH": "1",
                "SPECIES_CONTINUOUS_GATE_TEMP": "0.5",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "3.0",
                "SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT": "3.0",
            },
        ),
        # ---- SeedFrontMat pivot (deployable: committed mask from PREDICTED state, seed from
        # model gate logits; NO GT clot mask at train or eval). U/V isolate structure vs geom. ----
        "U_mat_frontier_only": MatGrowthLegSpec(
            code="U_mat_frontier_only",
            label="SeedFrontMat structural pivot: sparse nucleation + 1-hop front only",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_FRONTIER_HOPS": "1",
                "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "0.05",
            },
        ),
        "V_mat_frontier_geom": MatGrowthLegSpec(
            code="V_mat_frontier_geom",
            label="SeedFrontMat + rich geometry (no neighbor gate)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_GEOM_FEATS_RICH": "1",
                "SPECIES_CONTINUOUS_FRONTIER_HOPS": "1",
                "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "0.05",
            },
        ),
        # ---- Physically guided heads (deployable flow / gelation priors). ----
        "W_mat_flow_stagnation": MatGrowthLegSpec(
            code="W_mat_flow_stagnation",
            label="Mat-only + low-shear/stagnation flow features (nucleation pocket prior)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
            },
        ),
        "X_mat_flow_seedfront": MatGrowthLegSpec(
            code="X_mat_flow_seedfront",
            label="Stagnation flow prior + SeedFront structural pivot (U + flow)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_CONTINUOUS_FRONTIER_HOPS": "1",
                "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "0.05",
            },
        ),
        "Y_mat_tight_seed": MatGrowthLegSpec(
            code="Y_mat_tight_seed",
            label="SeedFront with tighter top-2% nucleation (vs default 5%)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_FRONTIER_HOPS": "1",
                "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "0.02",
            },
        ),
        "AB_mat_gelation_aux": MatGrowthLegSpec(
            code="AB_mat_gelation_aux",
            label="Plain Mat + differentiable gelation readout aux (mu1(Mat) physics head)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "0.5",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.15",
            },
        ),
        "S_mat_frontier_nuc": MatGrowthLegSpec(
            code="S_mat_frontier_nuc",
            label="SeedFrontMat_v0: gate + geom + sparse nucleation + 1-hop slow front",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_GEOM_FEATS_RICH": "1",
                # nucleation + slow front (front advance = 1 hop / macro step):
                "SPECIES_CONTINUOUS_FRONTIER_HOPS": "1",
                "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "0.05",
            },
        ),
        "T_mat_frontier_sharp": MatGrowthLegSpec(
            code="T_mat_frontier_sharp",
            label="S + sharp gate (temp 0.5) + spatial FP pressure (max-precision nucleation front)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_GEOM_FEATS_RICH": "1",
                "SPECIES_CONTINUOUS_FRONTIER_HOPS": "1",
                "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "0.05",
                "SPECIES_CONTINUOUS_GATE_TEMP": "0.5",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "3.0",
                "SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT": "3.0",
            },
        ),
    }
    if code not in specs:
        raise ValueError(f"unknown mat growth leg {leg!r}; use {list(specs)}")
    return specs[code]


def leg_out_ckpt(leg: str, *, ladder: bool = True) -> str:
    if ladder:
        return f"{LADDER_ROOT}/{leg}/species/best.pth"
    return DEFAULT_CKPT


def apply_mat_growth_simple_recipe_env(
    *,
    overrides: dict[str, str] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Apply triangle6 wall+3hop defaults, then Mat-only single-head overrides."""
    merged = apply_train_recipe_env(force=force)
    for key, val in MAT_GROWTH_SIMPLE_RECIPE.items():
        if force or not str(os.environ.get(key, "")).strip():
            os.environ[key] = str(val)
    if overrides:
        for key, val in overrides.items():
            os.environ[key] = str(val)
        merged.update({k: str(v) for k, v in overrides.items()})
    return merged


def mat_growth_precision_selection_enabled() -> bool:
    raw = (os.environ.get("SPECIES_MAT_GROWTH_PRECISION_SELECT") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def apply_mat_growth_leg_env(leg: str, *, force: bool = True) -> dict[str, str]:
    spec = mat_growth_leg_spec(leg)
    return apply_mat_growth_simple_recipe_env(overrides=spec.env_overrides, force=force)


def recipe_fingerprint() -> dict[str, Any]:
    """Serializable knob set for baseline JSON / train meta."""
    keys = sorted(set(GLOBAL_TRAIN_RECIPE) | set(MAT_GROWTH_SIMPLE_RECIPE))
    return {k: os.environ.get(k, GLOBAL_TRAIN_RECIPE.get(k, MAT_GROWTH_SIMPLE_RECIPE.get(k, ""))) for k in keys}


def _fimat_mat_row_index() -> int:
    """Output row for Mat in fi_mat dual-head checkpoints (FI=0, Mat=1)."""
    return 1 if FI_CHANNEL < MAT_CHANNEL else 0


def init_mat_single_from_fimat_ckpt(
    model: nn.Module,
    ckpt_path: Path | str,
    *,
    device: torch.device,
    mode: str = "backbone",
    quiet: bool = False,
) -> int:
    """Warm-start Mat-only/single-head variants from a fi_mat dual-head checkpoint."""
    from src.core_physics.species_pushforward_continuous import (
        SpeciesDualHeadContinuousGNN,
        load_continuous_bundle,
        load_pushforward_state_dict_partial,
    )

    path = Path(ckpt_path)
    if not path.is_file():
        if not quiet:
            print(f"[WARN] mat warm-start missing: {path}")
        return 0
    mode_n = (mode or "backbone").strip().lower()
    if mode_n == "full":
        bundle = load_continuous_bundle(path, device=device, quiet=True, architecture="dual", apply_meta_env=False)
        if bundle is None:
            return 0
        return load_pushforward_state_dict_partial(model, bundle.model.state_dict(), quiet=quiet)

    bundle = load_continuous_bundle(path, device=device, quiet=True, architecture="dual", apply_meta_env=False)
    if bundle is None or not isinstance(bundle.model, SpeciesDualHeadContinuousGNN):
        if not quiet:
            print(f"[WARN] mat warm-start: expected dual-head ckpt at {path}")
        return 0
    src = bundle.model.state_dict()
    dst = dict(model.state_dict())
    copied = 0
    for key in (
        "conv1.lin_l.weight",
        "conv1.lin_l.bias",
        "conv1.lin_r.weight",
        "conv2.lin_l.weight",
        "conv2.lin_l.bias",
        "conv2.lin_r.weight",
        "conv3.lin_l.weight",
        "conv3.lin_l.bias",
        "conv3.lin_r.weight",
    ):
        if key in src and key in dst and src[key].shape == dst[key].shape:
            dst[key] = src[key].to(device=dst[key].device, dtype=dst[key].dtype)
            copied += 1
    if "log_vel_decay_mat" in src and "log_vel_decay_mat" in dst:
        dst["log_vel_decay_mat"] = src["log_vel_decay_mat"].to(
            device=dst["log_vel_decay_mat"].device,
            dtype=dst["log_vel_decay_mat"].dtype,
        )
        copied += 1

    if mode_n == "mat_readout":
        mat_row = _fimat_mat_row_index()
        if "readout.0.weight" in dst:
            # single-head target
            for prefix in ("spatial_head", "magnitude_head"):
                for suffix in (".0.weight", ".0.bias"):
                    sk = f"{prefix}{suffix}"
                    dk = f"readout{suffix}"
                    if sk not in src or dk not in dst:
                        continue
                    s, t = src[sk], dst[dk]
                    if s.shape == t.shape:
                        dst[dk] = s.to(device=t.device, dtype=t.dtype)
                        copied += 1
                    elif suffix == ".0.weight" and s.ndim == 2 and t.ndim == 2 and s.shape[0] == t.shape[0]:
                        in_c = min(int(s.shape[1]), int(t.shape[1]))
                        dst[dk][:, :in_c] = s[:, :in_c].to(device=t.device, dtype=t.dtype)
                        copied += 1
            mw = src.get("magnitude_head.2.weight")
            mb = src.get("magnitude_head.2.bias")
            rw = dst.get("readout.2.weight")
            rb = dst.get("readout.2.bias")
            if mw is not None and rw is not None and mw.ndim == 2 and rw.ndim == 2:
                if int(mw.shape[0]) > mat_row:
                    dst["readout.2.weight"][0] = mw[mat_row].to(device=rw.device, dtype=rw.dtype)
                    copied += 1
            if mb is not None and rb is not None and mb.ndim == 1 and rb.ndim == 1:
                if int(mb.shape[0]) > mat_row:
                    dst["readout.2.bias"][0] = mb[mat_row].to(device=rb.device, dtype=rb.dtype)
                    copied += 1
        else:
            # dual-head target with out_dim=1: map Mat row from source dual heads.
            for head in ("spatial_head", "magnitude_head"):
                w_src = src.get(f"{head}.2.weight")
                b_src = src.get(f"{head}.2.bias")
                w_dst = dst.get(f"{head}.2.weight")
                b_dst = dst.get(f"{head}.2.bias")
                if w_src is not None and w_dst is not None and w_src.ndim == 2 and w_dst.ndim == 2:
                    if int(w_src.shape[0]) > mat_row and int(w_dst.shape[0]) >= 1:
                        dst[f"{head}.2.weight"][0] = w_src[mat_row].to(
                            device=w_dst.device, dtype=w_dst.dtype
                        )
                        copied += 1
                if b_src is not None and b_dst is not None and b_src.ndim == 1 and b_dst.ndim == 1:
                    if int(b_src.shape[0]) > mat_row and int(b_dst.shape[0]) >= 1:
                        dst[f"{head}.2.bias"][0] = b_src[mat_row].to(
                            device=b_dst.device, dtype=b_dst.dtype
                        )
                        copied += 1

    model.load_state_dict(dst)
    if not quiet:
        print(f"[OK] mat warm-start mode={mode_n} from {path} ({copied} tensors)", flush=True)
    return copied
