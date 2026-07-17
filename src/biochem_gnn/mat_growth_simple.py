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
    "SPECIES_FLOW_FEATS_DROP_XY": "0",
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
    # W + COMSOL-physics channel extensions (go_mat_physics_triage.ps1)
    "WA_mat_flow_neighbor_gate",
    "WB_mat_flow_geom_rich",
    "WC_mat_flow_dynamic",
    "WD_mat_flow_frontier",
    "WE_mat_flow_thrombin",
    "WF_mat_flow_fg",
    "WG_mat_flow_neighbor_crit",
    "WH_mat_flow_gelation_light",
    "WI_mat_flow_neighbor_geom",
    "WJ_mat_flow_stack",
    "WK_mat_flow_dropxy",
    "WL_mat_flow_dropxy_tightfp",
    "WM_mat_flow_seedfront_tightfp",
    "WC_mat_everywhere",
    "WC_mat_dynamic_frontier",
    "WC_mat_3hop",
    "WC_pivot1_skiphop",
    "WC_pivot2_sheargate",
    "WC_pivot3_occlusion",
    "WC_pivot4_frontier",
    "WC_pivots_combined",
    "WC_canonical_v2",
    "WC_v7_fresh_canonical",
    "WC_v7_clot_phi_mse",
    "WC_v7_high_precision",
    "WC_v2_baseline",
    "WC_v2_convection",
    "WC_v2_longrange",
    "WC_v2_label_smooth",
    "WC_v2_dilation",
    "WC_v2_longrange_smooth",
    # ---- Off-wall supervision v3 sweep (CLOT_PHI_PHYSICS_WALL_MAT_ONLY=0, NUCLEATION_HOPS=3) ----
    "WC_v3_baseline",
    "WC_v3_widenet",
    "WC_v3_focal_offwall",
    "WC_v3_neighbor_offwall",
    "WC_v3_widenet_focal",
    "WC_v3_convection_offwall",
    # ---- Off-wall split saturation v4 sweep (2026-07-06) ----
    "WC_v4_offwall_sat15",
    "WC_v4_offwall_sat30",
    "WC_v4_offwall_sat50",
    "WC_v4_offwall_nuc4_sat15",
    # ---- Off-wall sweep v5 (architectural pivots for off-wall growth, 2026-07-06) ----
    "WC_v5_offwall_multiscale",
    "WC_v5_offwall_phys_nuc",
    "WC_v5_offwall_convection",
    "WC_v5_offwall_all_pivots",
    "WC_v5_skiphop",
    "WC_v5_blind_loss",
    "WC_v5_phys_gating",
    "WC_v5_closed_loop",
    "WC_v5_two_model",
    # ---- Off-wall sweep v6 (2026-07-07) ----
    "WC_v6_closed_loop_eval",
    "WC_v6_skiphop_multiscale",
    "WC_v6_blind_loss",
    "WC_v6_sdf_gating",
    "WC_v6_latent_dropout",
    "WC_v6_spatial_heads",
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
        # ---- W base + targeted COMSOL physics channels (physics triage 2026-06) ----
        # Shared W core: Mat-only dual head + stagnation flow feats (sr<lss proxy).
        "WA_mat_flow_neighbor_gate": MatGrowthLegSpec(
            code="WA_mat_flow_neighbor_gate",
            label="W + neighbor commit gate (autocatalytic k_aa·Mas·AP proxy on spatial head)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
            },
        ),
        "WB_mat_flow_geom_rich": MatGrowthLegSpec(
            code="WB_mat_flow_geom_rich",
            label="W + rich geometry (width/expansion/curvature deposition context)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_GEOM_FEATS_RICH": "1",
            },
        ),
        "WC_mat_flow_dynamic": MatGrowthLegSpec(
            code="WC_mat_flow_dynamic",
            label="W + per-step dynamic flow (clot-diverted velocity during rollout)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
            },
        ),
        "WD_mat_flow_frontier": MatGrowthLegSpec(
            code="WD_mat_flow_frontier",
            label="W + 1-hop committed frontier only (growth topology; no top-k seed mask)",
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
                "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "0",
            },
        ),
        "WE_mat_flow_thrombin": MatGrowthLegSpec(
            code="WE_mat_flow_thrombin",
            label="W + Mat+thrombin co-state (deployable AP/activation pathway proxy)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS": "11,5",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "8.0",
            },
        ),
        "WF_mat_flow_fg": MatGrowthLegSpec(
            code="WF_mat_flow_fg",
            label="W + Mat+FG co-state (reaction-active precursor marker; strategy Mat+FG)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS": "11,7",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "8.0",
            },
        ),
        "WG_mat_flow_neighbor_crit": MatGrowthLegSpec(
            code="WG_mat_flow_neighbor_crit",
            label="W + neighbor gate + underpred/crit focus (autocat + deposition boost)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "5.0",
                "SPECIES_CONTINUOUS_DELTA_THRESH_MAT": "2.5e-6",
            },
        ),
        "WH_mat_flow_gelation_light": MatGrowthLegSpec(
            code="WH_mat_flow_gelation_light",
            label="W + light differentiable gelation aux (mu1(Mat) train feedback; low overpaint risk)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "0.25",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.05",
            },
        ),
        "WI_mat_flow_neighbor_geom": MatGrowthLegSpec(
            code="WI_mat_flow_neighbor_geom",
            label="W + neighbor gate + rich geom (stagnation + autocat + vessel shape)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_GEOM_FEATS_RICH": "1",
            },
        ),
        "WJ_mat_flow_stack": MatGrowthLegSpec(
            code="WJ_mat_flow_stack",
            label="W stack: neighbor gate + rich geom + dynamic flow (max deployable physics bundle)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_GEOM_FEATS_RICH": "1",
            },
        ),
        "WK_mat_flow_dropxy": MatGrowthLegSpec(
            code="WK_mat_flow_dropxy",
            label="W with flow x/y ablated (speed+shear+div only; reduce spatial memorization)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
            },
        ),
        "WL_mat_flow_dropxy_tightfp": MatGrowthLegSpec(
            code="WL_mat_flow_dropxy_tightfp",
            label="WK + stronger all-node FP pressure (gate+spatial) for early inlet ring suppression",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_CONTINUOUS_SPEED_FP_WEIGHT": "8.0",
                "SPECIES_CONTINUOUS_GATE_FP_WEIGHT": "8.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "3.0",
            },
        ),
        "WM_mat_flow_seedfront_tightfp": MatGrowthLegSpec(
            code="WM_mat_flow_seedfront_tightfp",
            label="W + top-k seed/frontier + neighbor gate + tighter FP terms (middle ground vs WD cold-start)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_CONTINUOUS_FRONTIER_HOPS": "1",
                "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "0.03",
                "SPECIES_CONTINUOUS_SPEED_FP_WEIGHT": "8.0",
                "SPECIES_CONTINUOUS_GATE_FP_WEIGHT": "8.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "3.0",
            },
        ),
        "WC_mat_everywhere": MatGrowthLegSpec(
            code="WC_mat_everywhere",
            label="WC foundation with Mat predicted everywhere (full graph)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "99",
            },
        ),
        "WC_mat_dynamic_frontier": MatGrowthLegSpec(
            code="WC_mat_dynamic_frontier",
            label="WC foundation with Mat predicted only on dynamic wall + 1-hop frontier",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "99",
                "SPECIES_CONTINUOUS_DYNAMIC_FRONTIER_MASK": "1",
            },
        ),
        "WC_mat_3hop": MatGrowthLegSpec(
            code="WC_mat_3hop",
            label="WC foundation with Mat predicted only in 3-hop wall subgraph (canonical match)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
            },
        ),
        "WC_pivot1_skiphop": MatGrowthLegSpec(
            code="WC_pivot1_skiphop",
            label="WC 3-hop with Pivot 1 Decoupled Linear-Subgraph Message Passing (Skip-Hop GNN)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "SPECIES_SKIP_HOP_GNN": "1",
            },
        ),
        "WC_pivot2_sheargate": MatGrowthLegSpec(
            code="WC_pivot2_sheargate",
            label="WC 3-hop with Pivot 2 Differentiable Readout Shear Gate",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "SPECIES_SHEAR_READOUT_GATE": "1",
            },
        ),
        "WC_pivot3_occlusion": MatGrowthLegSpec(
            code="WC_pivot3_occlusion",
            label="WC 3-hop with Pivot 3 Dynamic Geometry Occlusion Loop (Flow Re-Solving)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
            },
        ),
        "WC_pivot4_frontier": MatGrowthLegSpec(
            code="WC_pivot4_frontier",
            label="WC 3-hop with Pivot 4 Autocatalytic Dynamic Frontier Growth kinetics",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "SPECIES_FRONTIER_KINETICS": "1",
                "SPECIES_FRONTIER_K_AP": "0.5",
                "SPECIES_FRONTIER_K_T": "0.5",
            },
        ),
        "WC_pivots_combined": MatGrowthLegSpec(
            code="WC_pivots_combined",
            label="WC 3-hop with all 4 architectural pivots combined",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "SPECIES_SKIP_HOP_GNN": "1",
                "SPECIES_SHEAR_READOUT_GATE": "1",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_FRONTIER_KINETICS": "1",
                "SPECIES_FRONTIER_K_AP": "0.5",
                "SPECIES_FRONTIER_K_T": "0.5",
            },
        ),
        "WC_canonical_v2": MatGrowthLegSpec(
            code="WC_canonical_v2",
            label="WC 3-hop + Dynamic Occlusion (Pivot 3 folded into canonical; WC_canonical_v2)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "BIOCHEM_ROLLOUT_DYNAMIC_OCCLUSION": "1",  # alias kept for clarity
                "SPECIES_DYNAMIC_OCCLUSION": "1",
            },
        ),
        "WC_v7_fresh_canonical": MatGrowthLegSpec(
            code="WC_v7_fresh_canonical",
            label="WC v7 Fresh Canonical: 4-hop lumen, dynamic occlusion, closed-loop corrector coupling, zero GT leak",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "BIOCHEM_ROLLOUT_DYNAMIC_OCCLUSION": "1",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "4",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "BIOCHEM_CORRECTOR_COUPLING": "1",
                "SPECIES_ROLLOUT_VEL_SOURCE": "coupled",
                "SPECIES_CONTINUOUS_CLOUT_SCORE": "guiding",
                "CLOT_GUIDE_RELAX_HOPS": "3",
                "SPECIES_CONTINUOUS_SCORE_CLOUT_W": "0.75",
                "SPECIES_CLOUT_PREC_REC_FLOOR": "0.30",
                "SPECIES_ROLLOUT_DEPLOY_FAITHFUL": "1",
                "SPECIES_ROLLOUT_IC_SOURCE": "resting",
                "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT": "1",
                "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.02",
                "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.08",
                "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.25",
                "SPECIES_CONTINUOUS_TBPTT_TAIL": "5",
                "SPECIES_CONTINUOUS_CLOSED_LOOP_INIT": "0.45",
            },
        ),
        "WC_v7_clot_phi_mse": MatGrowthLegSpec(
            code="WC_v7_clot_phi_mse",
            label="WC v7 Clot Phi MSE: Guiding loss is GT clot prediction - model clot prediction using MSE",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "BIOCHEM_ROLLOUT_DYNAMIC_OCCLUSION": "1",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "4",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "BIOCHEM_CORRECTOR_COUPLING": "1",
                "SPECIES_ROLLOUT_VEL_SOURCE": "coupled",
                "SPECIES_CONTINUOUS_CLOUT_SCORE": "guiding",
                "CLOT_GUIDE_RELAX_HOPS": "3",
                "SPECIES_CONTINUOUS_SCORE_CLOUT_W": "0.75",
                "SPECIES_CLOUT_PREC_REC_FLOOR": "0.30",
                "SPECIES_ROLLOUT_DEPLOY_FAITHFUL": "1",
                "SPECIES_ROLLOUT_IC_SOURCE": "resting",
                "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT": "1",
                "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.02",
                "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.08",
                "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.25",
                "SPECIES_CONTINUOUS_TBPTT_TAIL": "5",
                "SPECIES_CONTINUOUS_CLOSED_LOOP_INIT": "0.45",
                # --- Clot Phi MSE Loss Overrides ---
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "20.0",
                "SPECIES_GELATION_PHI_LOSS_TYPE": "mse",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.0",
                "SPECIES_CONTINUOUS_LOSS_SCALE": "0.1",
            },
        ),
        "WC_v7_high_precision": MatGrowthLegSpec(
            code="WC_v7_high_precision",
            label="WC v7 High Precision: Penalize false positives heavily to encourage small precise steps",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "BIOCHEM_ROLLOUT_DYNAMIC_OCCLUSION": "1",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "4",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "BIOCHEM_CORRECTOR_COUPLING": "1",
                "SPECIES_ROLLOUT_VEL_SOURCE": "coupled",
                "SPECIES_CONTINUOUS_CLOUT_SCORE": "guiding",
                "CLOT_GUIDE_RELAX_HOPS": "3",
                "SPECIES_CONTINUOUS_SCORE_CLOUT_W": "0.75",
                "SPECIES_CLOUT_PREC_REC_FLOOR": "0.30",
                "SPECIES_ROLLOUT_DEPLOY_FAITHFUL": "1",
                "SPECIES_ROLLOUT_IC_SOURCE": "resting",
                "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT": "1",
                "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.02",
                "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.08",
                "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.25",
                "SPECIES_CONTINUOUS_TBPTT_TAIL": "5",
                "SPECIES_CONTINUOUS_CLOSED_LOOP_INIT": "0.45",
                # --- High Precision FP Penalty Overrides ---
                "SPECIES_CONTINUOUS_FP_WEIGHT": "96.0",
                "SPECIES_CONTINUOUS_GATE_FP_WEIGHT": "16.0",
                "SPECIES_CONTINUOUS_SPEED_FP_WEIGHT": "24.0",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "0.5",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "8.0",
            },
        ),
        "WC_v2_baseline": MatGrowthLegSpec(
            code="WC_v2_baseline",
            label="WC v2 Baseline (reference for v2 sweep)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
            },
        ),
        "WC_v2_convection": MatGrowthLegSpec(
            code="WC_v2_convection",
            label="WC v2 + Arch 1: Convection-Aware Upwind Feature",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_CONVECTION_AGGR": "1",
                "SPECIES_CONVECTION_ALPHA": "0.5",
            },
        ),
        "WC_v2_longrange": MatGrowthLegSpec(
            code="WC_v2_longrange",
            label="WC v2 + Arch 2: Long-Range Skip Edges",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_LONGRANGE_EDGES": "1",
                "SPECIES_LONGRANGE_DIST_MULT": "2.5",
            },
        ),
        "WC_v2_label_smooth": MatGrowthLegSpec(
            code="WC_v2_label_smooth",
            label="WC v2 + Arch 3: Hop-1 Label Smoothing",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_HOP1_SMOOTH": "1",
                "SPECIES_HOP1_SMOOTH_ALPHA": "0.4",
            },
        ),
        "WC_v2_dilation": MatGrowthLegSpec(
            code="WC_v2_dilation",
            label="WC v2 + Arch 4: 2-Hop Growth Dilation",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_GROWTH_DILATION": "2",
                "CLOT_V2_NUCLEATION_HOPS": "2",
                "CLOT_PHI_CEILING_HOPS": "5",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
            },
        ),
        "WC_v2_longrange_smooth": MatGrowthLegSpec(
            code="WC_v2_longrange_smooth",
            label="WC v2 + Arch 2+3 Combined (Long-Range + Label Smooth)",
            no_init=False,
            init_ckpt=init_default,
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_LONGRANGE_EDGES": "1",
                "SPECIES_LONGRANGE_DIST_MULT": "2.5",
                "SPECIES_HOP1_SMOOTH": "1",
                "SPECIES_HOP1_SMOOTH_ALPHA": "0.4",
            },
        ),
        # ---- Off-wall supervision v3 sweep (2026-07-06) ----
        # All v3 legs share the core off-wall unlock:
        #   CLOT_PHI_PHYSICS_WALL_MAT_ONLY=0   - full gelation at all hops
        #   CLOT_V2_NUCLEATION_HOPS=3           - 3-hop front advance per step
        #   CLOT_PHI_CEILING_HOPS=6             - allow up to Hop 6
        #   SPECIES_DYNAMIC_OCCLUSION=1         - Pivot 3 (best structural pivot)
        #   SPECIES_FLOW_FEATS/DYNAMIC=1        - stagnation + per-step flow
        # All legs init from WC_v2_dilation (the only prior ckpt with off-wall gradients).
        # Each leg adds exactly one physically-motivated change to isolate its contribution.
        "WC_v3_baseline": MatGrowthLegSpec(
            code="WC_v3_baseline",
            label="V3 clean baseline: v2_baseline recipe + full off-wall supervision",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v2_dilation/species/best.pth",
            init_mode="backbone",
            env_overrides={
                # Core off-wall unlock (identical to v2_baseline except wall_mat_only + nuc_hops)
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                # v3 off-wall unlock:
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
            },
        ),
        "WC_v3_widenet": MatGrowthLegSpec(
            code="WC_v3_widenet",
            label="V3 + wider GNN band (Hop 5) + recall-biased loss to reach deeper interior nodes",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v2_dilation/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                # Wider training band (includes Hop 4-5 nodes in loss):
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                # Recall-biased loss (down-weights FP pressure, boosts underpred signal):
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
            },
        ),
        "WC_v3_focal_offwall": MatGrowthLegSpec(
            code="WC_v3_focal_offwall",
            label="V3 + strong focal loss (gamma=5) + high alpha for rare off-wall class",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v2_dilation/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                # Strong focal loss: down-weights easy non-clot nodes, forces focus on
                # rare off-wall clot nodes that the model currently misses:
                "SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT": "5.0",
                "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "0.97",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "5.0",
            },
        ),
        "WC_v3_neighbor_offwall": MatGrowthLegSpec(
            code="WC_v3_neighbor_offwall",
            label="V3 + autocatalytic neighbor commit gate (biochemical chain-reaction propagation)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v2_dilation/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                # Neighbor commit gate: once a node commits, its neighbors become
                # more likely to commit (thrombin amplification chain-reaction):
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "3.0",
            },
        ),
        "WC_v3_widenet_focal": MatGrowthLegSpec(
            code="WC_v3_widenet_focal",
            label="V3 kitchen-sink: wide band + focal + neighbor gate + aggressive nucleation (Hop 4 / Ceiling 8)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v2_dilation/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                # More aggressive nucleation to test: 4-hop front, wider ceiling:
                "CLOT_V2_NUCLEATION_HOPS": "4",
                "CLOT_PHI_CEILING_HOPS": "8",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "1",
                "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "0.8",
                "SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT": "5.0",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "2.0",
            },
        ),
        "WC_v3_convection_offwall": MatGrowthLegSpec(
            code="WC_v3_convection_offwall",
            label="V3 + convection-aware upwind feature (physically key for interior deposition pockets)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v2_dilation/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                # Convection-aware upwind aggregation: flow-directed message passing
                # identifies low-velocity recirculation pockets = interior deposition sites:
                "SPECIES_CONVECTION_AGGR": "1",
                "SPECIES_CONVECTION_ALPHA": "0.5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "3.0",
            },
        ),
        "WC_v4_offwall_sat15": MatGrowthLegSpec(
            code="WC_v4_offwall_sat15",
            label="V4 + widenet baseline + soft off-wall saturation clamp (scale_offwall=15.0)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "15.0",
            },
        ),
        "WC_v4_offwall_sat30": MatGrowthLegSpec(
            code="WC_v4_offwall_sat30",
            label="V4 + widenet baseline + soft off-wall saturation clamp (scale_offwall=30.0)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
            },
        ),
        "WC_v4_offwall_sat50": MatGrowthLegSpec(
            code="WC_v4_offwall_sat50",
            label="V4 + widenet baseline + soft off-wall saturation clamp (scale_offwall=50.0)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "50.0",
            },
        ),
        "WC_v4_offwall_nuc4_sat15": MatGrowthLegSpec(
            code="WC_v4_offwall_nuc4_sat15",
            label="V4 + widenet baseline + soft off-wall saturation clamp (scale_offwall=15.0) + 4-hop nucleation",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "4",
                "CLOT_PHI_CEILING_HOPS": "8",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "15.0",
            },
        ),
        "WC_v5_offwall_multiscale": MatGrowthLegSpec(
            code="WC_v5_offwall_multiscale",
            label="V5 + sat30 baseline + multiscale skip-hop messaging",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v4_offwall_sat30/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_MULTISCALE_SKIP_HOP": "1",
                "SPECIES_MULTISCALE_SKIP_HOP_MULT": "3.0",
                "SPECIES_MULTISCALE_SKIP_HOP_SCALE": "0.5",
            },
        ),
        "WC_v5_offwall_phys_nuc": MatGrowthLegSpec(
            code="WC_v5_offwall_phys_nuc",
            label="V5 + sat30 baseline + physics-inspired nucleation prior",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v4_offwall_sat30/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_PHYSICS_NUCLEATION": "1",
                "SPECIES_PHYSICS_NUC_SPEED_THRESH": "0.15",
                "SPECIES_PHYSICS_NUC_SHEAR_THRESH": "0.20",
            },
        ),
        "WC_v5_offwall_convection": MatGrowthLegSpec(
            code="WC_v5_offwall_convection",
            label="V5 + sat30 baseline + convective upwind messaging",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v4_offwall_sat30/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_CONVECTIVE_UPWIND": "1",
            },
        ),
        "WC_v5_offwall_all_pivots": MatGrowthLegSpec(
            code="WC_v5_offwall_all_pivots",
            label="V5 + sat30 baseline + all 3 pivots combined",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v4_offwall_sat30/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_MULTISCALE_SKIP_HOP": "1",
                "SPECIES_MULTISCALE_SKIP_HOP_MULT": "3.0",
                "SPECIES_MULTISCALE_SKIP_HOP_SCALE": "0.5",
                "SPECIES_PHYSICS_NUCLEATION": "1",
                "SPECIES_PHYSICS_NUC_SPEED_THRESH": "0.15",
                "SPECIES_PHYSICS_NUC_SHEAR_THRESH": "0.20",
                "SPECIES_CONVECTIVE_UPWIND": "1",
            },
        ),
        "WC_v5_skiphop": MatGrowthLegSpec(
            code="WC_v5_skiphop",
            label="V5 + skiphop bipartite GNN",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_SKIP_HOP_GNN": "1",
            },
        ),
        "WC_v5_blind_loss": MatGrowthLegSpec(
            code="WC_v5_blind_loss",
            label="V5 + midside-blind loss masking",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_MIDSIDE_BLIND_LOSS": "1",
            },
        ),
        "WC_v5_phys_gating": MatGrowthLegSpec(
            code="WC_v5_phys_gating",
            label="V5 + physical FP gating",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_PHYSICAL_FP_GATING": "1",
            },
        ),
        "WC_v5_closed_loop": MatGrowthLegSpec(
            code="WC_v5_closed_loop",
            label="V5 + step closed loop coupling",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
            },
        ),
        "WC_v5_two_model": MatGrowthLegSpec(
            code="WC_v5_two_model",
            label="V5 + two model wall/offwall blend",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_TWO_MODEL_MODE": "1",
                "SPECIES_OFFWALL_MODEL_CKPT": "outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            },
        ),
        "WC_v6_closed_loop_eval": MatGrowthLegSpec(
            code="WC_v6_closed_loop_eval",
            label="V6 closed loop baseline (align F1)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
            },
        ),
        "WC_v6_skiphop_multiscale": MatGrowthLegSpec(
            code="WC_v6_skiphop_multiscale",
            label="V6 skiphop multiscale skip connections",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "SPECIES_MULTISCALE_SKIP_HOP": "1",
                "SPECIES_MULTISCALE_SKIP_HOP_MULT": "3.0",
                "SPECIES_MULTISCALE_SKIP_HOP_SCALE": "0.5",
            },
        ),
        "WC_v6_blind_loss": MatGrowthLegSpec(
            code="WC_v6_blind_loss",
            label="V6 closed loop + midside blind loss",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "SPECIES_MIDSIDE_BLIND_LOSS": "1",
            },
        ),
        "WC_v6_sdf_gating": MatGrowthLegSpec(
            code="WC_v6_sdf_gating",
            label="V6 closed loop + SDF weighted FP gating",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "SPECIES_SDF_FP_GATING": "1",
                "SPECIES_SDF_FP_DECAY_SCALE": "0.015",
                "SPECIES_SDF_FP_MIN": "0.1",
            },
        ),
        "WC_v6_latent_dropout": MatGrowthLegSpec(
            code="WC_v6_latent_dropout",
            label="V6 closed loop + Latent Dropout 0.5",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "SPECIES_LATENT_DROPOUT": "0.5",
            },
        ),
        "WC_v6_spatial_heads": MatGrowthLegSpec(
            code="WC_v6_spatial_heads",
            label="V6 spatially gated heads + isolated offwall loss scaling",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/mat_growth_ladder/WC_pivot3_occlusion/species/best.pth",
            init_mode="backbone",
            env_overrides={
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
                "SPECIES_VISCOSITY_CALIB": "1",
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "0",
                "CLOT_V2_NUCLEATION_HOPS": "3",
                "CLOT_PHI_CEILING_HOPS": "6",
                "SPECIES_DYNAMIC_OCCLUSION": "1",
                "SPECIES_SNAPSHOT_WALL_HOPS": "5",
                "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
                "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
                "SPECIES_CONTINUOUS_SATURATION_SCALE": "80.0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "SPECIES_SPATIAL_GATE_HEADS": "1",
                "SPECIES_GATE_SDF_CRIT": "0.012",
                "SPECIES_GATE_SDF_TEMP": "0.003",
                "SPECIES_ISOLATE_OFFWALL_LOSS": "1",
                "SPECIES_OFFWALL_LOSS_SCALE": "2.0",
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
