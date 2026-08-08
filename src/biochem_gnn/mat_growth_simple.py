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
from dataclasses import dataclass, field
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
    # Train static/dynamic flow block on COMSOL GT velocity (not a second GINO-DEQ solve).
    "SPECIES_FLOW_FEATS_SOURCE": "gt",
    "SPECIES_CONTINUOUS_CLOUT_SCORE": "relaxed_prec_floor",
    "SPECIES_CLOUT_PREC_REC_FLOOR": "0.35",
    "SPECIES_CONTINUOUS_SCORE_CLOUT_W": "0.75",
    "SPECIES_MAT_GROWTH_PRECISION_SELECT": "1",
    "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
    "SPECIES_VISCOSITY_CALIB": "0",
    "SPECIES_CONTINUOUS_VEL_DECAY": "1",
    "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY": "1",
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
    # ---- Firewall-break sequence on WC_v7 (2026-07-21) ----
    "WC_v7_fw1_blind_sat",
    "WC_v7_fw1_blind",
    "WC_v7_fw1_smooth",
    "WC_v7_fw1_sat30",
    "WC_v7_fw3_isolate",
    "WC_v7_fw3_skiphop",
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
    "WG_sched_sample",
    "WG_noise_boost",
    "WG_long_tbptt",
    "WG_dynamics_all",
    "WG_mirror_y",
    "WG_geom_rich",
    "WG_flux_stag",
    "WG_full_stack",
    "WG_sweep_v3_01",
    "WG_sweep_v3_02",
    "WG_sweep_v3_03",
    "WG_sweep_v3_04",
    "WG_sweep_v3_05",
    "WG_sweep_v3_06",
    "WG_sweep_v3_07",
    "WG_sweep_v3_08",
    # ---- Feat-dim fix re-run of phase1 geom/flux arms (pack cache + in_dim) ----
    "WG_featfix_01",
    "WG_featfix_02",
    "WG_featfix_03",
    "WG_featfix_04",
    # ---- Clot-rich N+ LOAO (featfix_03 stack, expand sites, hold out patient020) ----
    "WG_clotrich_nplus",
    "WG_clotrich_nplus_v2",
    # ---- Multi-hop flow features (2026-08-05 root cause; plan s2.2) ----
    "WG_multihop",
    "WG_multihop_ctrl",
    # ---- Small-cohort precision iteration (fix objective mismatch before N+) ----
    "WG_prec_iter",
    "WG_prec_mirror",
    "WG_prec_sites",
    "WG_prec_mid",
    "WG_prec_ft",
    "WG_prec_loao",
    "WG_prec_loao_freeze",
    # ---- Train-time sparse commitment (seed-then-frontier; not eval-only masking) ----
    "WG_prec_seed",
    "WG_prec_seed_fh2",
    "WG_prec_seed_tk02",
    "WG_prec_seed_aux",
    # ---- Front/recall FT from prec_iter (no hard mask; seed_aux off) ----
    "WG_prec_front",
    # ---- Precision FT from floor (not seed/front): phys FP gate OR closed-loop ----
    "WG_prec_physfp",
    "WG_prec_cloop",
    # ---- Multi-pocket exclusive contrast (wrong-pocket soft penalty; no hard mask) ----
    "WG_prec_pocket",
    # ---- Physics-biased GAT trunk on featfix_03 feature stack ----
    "WG_physgat_01",
    "WG_physgat_ctrl",
    # ---- Flow-source A/B gate (before phase1_sweep_v3) ----
    "FS_ab_gt",
    "FS_ab_kine",
    "FS_ab_coupled",
    # ---- Stenosis/aneurysm sub-cohort recall fine-tune (WALL_MODEL_PLAN.md s9) ----
    "WG_stenosis_subcohort_ft",
    "WG_stenosis_subcohort_ft_v2",
    "WG_stenosis_subcohort_ft_v3",
    "WG_stenosis_subcohort_ft_v4",
    "WG_stenosis_subcohort_ft_v5",
    "WG_stenosis_subcohort_ft_v6",
    "WG_stenosis_subcohort_ft_v7",
    "WG_stenosis_subcohort_ft_v8",
    "WG_stenosis_subcohort_ft_v9",
    "WG_stenosis_subcohort_ft_v10",
    "WG_phase1_baseline",
    "WG_phase2a_decomp",
    "WG_phase2b_nobrake",
    "WG_phase3a_closedloop",
    "WG_phase3b_zkin_ablate",
)

# Full-length clot-rich anchors (T=201, off-wall >=30%) from docs/GENERALIZATION_PLAN.md EDA.
# Includes 2026-08-03 batch (012,040-044); excludes half-finished 039 and empty 027.
# Inventory / sealed roles: data/reference/generalization_new_vessels.json (§1b).
WALL_GEN_CLOT_RICH_ANCHORS: tuple[str, ...] = (
    "patient001",
    "patient005",
    "patient006",
    "patient007",
    "patient010",
    "patient012",
    "patient013",
    "patient016",
    "patient020",
    "patient021",
    "patient029",
    "patient032",
    "patient035",
    "patient037",
    "patient040",
    "patient041",
    "patient042",
    "patient043",
    "patient044",
)

# 2026-08-03 batch: sealed geometry challenge (never in default N+ train).
# 043 = aneurysm holdout; 044 = stenosis holdout. Primary gate remains patient020.
WALL_GEN_BATCH_1B_TRAIN: tuple[str, ...] = (
    "patient012",
    "patient040",
    "patient041",
    "patient042",
)
WALL_GEN_BATCH_1B_CHALLENGE: tuple[str, ...] = (
    "patient043",
    "patient044",
)
WALL_GEN_BATCH_1B_NEG_CONTROL: tuple[str, ...] = ("patient027",)
WALL_GEN_BATCH_1B_EXCLUDE: tuple[str, ...] = ("patient039",)

# Warm-start for clot-rich N+ (geom+flux SAGE; must not include the holdout in its train set).
WG_FEATFIX_03_CKPT = "outputs/biochem/eda/wall_gen_featfix/WG_featfix_03/best.pth"

# Best F1 on record (0.500 on patient020). Warm-start base for the multi-hop flow legs;
# its train set excludes patient020. See docs/WALL_MODEL_PLAN.md s2.
WG_CLOTRICH_NPLUS_CKPT = (
    "outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth"
)

# Small clot-rich iteration cohort (no 023/002 junk; hold out patient020 separately).
WALL_GEN_SMALL_TRAIN_ANCHORS: tuple[str, ...] = (
    "patient005",
    "patient006",
    "patient010",
)
# Controlled mid expand: small + 3 extra clot-rich (not full N+).
WALL_GEN_MID_TRAIN_ANCHORS: tuple[str, ...] = (
    "patient005",
    "patient006",
    "patient010",
    "patient001",
    "patient007",
    "patient012",
)
WG_PREC_ITER_CKPT = "outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth"


def wall_gen_clot_rich_train_anchors(
    *,
    holdout: str = "patient020",
    exclude_sealed_challenge: bool = True,
) -> list[str]:
    """Clot-rich train list for N+ LOAO (excludes holdout; never includes 023/002).

    By default also drops ``WALL_GEN_BATCH_1B_CHALLENGE`` (043 aneurysm / 044 stenosis)
    so the sealed geometry challenge stays clean. Primary holdout gate remains
    ``patient020`` (report challenge vessels separately; do not average into the gate).
    """
    h = str(holdout or "").strip()
    if not h:
        raise ValueError("holdout must be a non-empty anchor id")
    if h not in WALL_GEN_CLOT_RICH_ANCHORS:
        raise ValueError(
            f"holdout={h!r} is not in WALL_GEN_CLOT_RICH_ANCHORS; "
            f"gate only on clot-rich vessels"
        )
    sealed = set(WALL_GEN_BATCH_1B_CHALLENGE) if exclude_sealed_challenge else set()
    out = [a for a in WALL_GEN_CLOT_RICH_ANCHORS if a != h and a not in sealed]
    if not out:
        raise ValueError(f"holdout={h!r} left an empty clot-rich train set")
    return list(out)


# ---------------------------------------------------------------------------
# Stenosis/aneurysm sub-cohort pivot (WALL_MODEL_PLAN.md s9, 2026-08-05).
#
# Deliberately DIFFERENT from the sealed WALL_GEN_BATCH_1B_* split above:
#   - includes patient039 (excluded there, and from WALL_GEN_CLOT_RICH_ANCHORS entirely --
#     half-finished sim, T=92; the commit-order probe found only 29 GT nodes / 3 TP
#     components on it, the thinnest signal of any vessel probed).
#   - trains on patient044 (there: sealed challenge, held out together with 043).
#   - holds out ONLY patient043 (there: both 043 and 044 are held out).
# This answers a narrower question -- "can we generalize within this 6-vessel family" --
# not a replacement for the sealed protocol, which stays intact for the eventual
# all-vessel evaluation. Training on 044 here spends one of its two sealed challenge
# points: patient043 is the only vessel left sealed for BOTH this sub-study and the
# original wall-gen plan once this leg is trained.
WALL_GEN_STENOSIS_SUBCOHORT: tuple[str, ...] = (
    "patient039",
    "patient040",
    "patient041",
    "patient042",
    "patient043",
    "patient044",
)


def wall_gen_stenosis_subcohort_train_anchors(*, holdout: str = "patient043") -> list[str]:
    """Train list for the stenosis/aneurysm sub-cohort pivot (cohort minus holdout).

    See ``WALL_GEN_STENOSIS_SUBCOHORT`` above for how this split differs from the sealed
    ``WALL_GEN_BATCH_1B_*`` protocol. Zero-shot (``WG_clotrich_nplus`` + flow gate pct=25,
    no cohort-specific training at all) already scores ``deploy_clot_f1=0.650`` on
    ``patient043`` -- ``WG_stenosis_subcohort_ft`` is a light fine-tune from that floor,
    not a from-scratch train.
    """
    h = str(holdout or "").strip()
    if h not in WALL_GEN_STENOSIS_SUBCOHORT:
        raise ValueError(
            f"holdout={h!r} is not in WALL_GEN_STENOSIS_SUBCOHORT={WALL_GEN_STENOSIS_SUBCOHORT}"
        )
    return [a for a in WALL_GEN_STENOSIS_SUBCOHORT if a != h]


@dataclass(frozen=True)
class MatGrowthLegSpec:
    """One mat-growth sweep / ladder leg.

    * ``config_kwargs`` -- typed ``PushforwardConfig`` architecture / loss / feature knobs
    * ``runtime_kwargs`` -- typed ``BiochemRuntimeConfig`` flat fields (coupling, rollout,
      scoring, gelation, off-wall)
    * ``env_overrides`` -- deprecated residual unknowns only; do not add new keys here
    """

    code: str
    label: str
    no_init: bool
    init_ckpt: str
    init_mode: str  # "backbone", "full", "mat_readout"
    config_kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_kwargs: dict[str, Any] = field(default_factory=dict)
    env_overrides: dict[str, str] = field(default_factory=dict)


def mat_growth_leg_spec(leg: str) -> MatGrowthLegSpec:
    code = leg.strip()
    init_default = str(global_ckpt_path()).replace("\\", "/")
    
    # Typed architecture kwargs (PushforwardConfig fields).
    wc_v7_config: dict[str, Any] = {
        "dual_head": True,
        "species_scope": "mat",
        "saturation_gate": True,
        "flow_feats": True,
        "flow_feats_dynamic": True,
        "mature_fp_exempt": True,
        "teacher_noise": 0.02,
        "teacher_fp_frac": 0.08,
        "teacher_blur": 0.25,
        "tbptt_tail": 5,
        "closed_loop_init": 0.45,
        "physics_readout": True,
        "loss_scale": 0.1,
        "score_clout_w": 0.75,
    }
    # Typed runtime kwargs (BiochemRuntimeConfig flat fields).
    wc_v7_runtime: dict[str, Any] = {
        "viscosity_calib": True,
        "wall_hops": 3,
        "dynamic_occlusion": True,
        "wall_mat_only": False,
        "nucleation_hops": 4,
        "ceiling_hops": 4,
        "closed_loop_coupling": True,
        "corrector_coupling": True,
        "rollout_vel_source": "coupled",
        "clout_score_mode": "guiding",
        "guide_relax_hops": 3,
        "clout_prec_rec_floor": 0.30,
        "deploy_faithful": True,
        "rollout_ic_source": "resting",
        "phi_loss_weight": 20.0,
        "phi_loss_type": "mse",
        "mu_loss_weight": 0.0,
        "kine_resolve_on_clot": False,
    }
    v3_config: dict[str, Any] = {
        **wc_v7_config,
        "flow_feats_drop_xy": True,
        # Deploy-faithful train/eval: predicted kine + corrector override (FS_ab_coupled).
        "flow_feats_source": "auto",
        "scheduled_sampling": False,
    }
    v3_runtime: dict[str, Any] = {
        **wc_v7_runtime,
        "train_vel_source": "coupled",
        "rollout_vel_source": "coupled",
        "corrector_coupling": True,
        "closed_loop_coupling": True,
        "train_deploy_eval_flow": "auto",
    }
    # Promoted wall-gen baseline warm-start (see data/reference/mat_wall_gen_baseline.json).
    wall_gen_init = "outputs/biochem/biochem_gnn/wall_gen_baseline/species/best.pth"
    # Legacy combined env dicts (materialize_leg_spec splits these for older legs).
    wc_v7_base_env = {
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
        "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
        "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "20.0",
        "SPECIES_GELATION_PHI_LOSS_TYPE": "mse",
        "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.0",
        "SPECIES_CONTINUOUS_LOSS_SCALE": "0.1",
        "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
    }
    v3_base = {
        **wc_v7_base_env,
        "SPECIES_FLOW_FEATS_DROP_XY": "1",
        "SPECIES_FLOW_FEATS_SOURCE": "auto",
        "SPECIES_TRAIN_VEL_SOURCE": "coupled",
        "SPECIES_ROLLOUT_VEL_SOURCE": "coupled",
        "SPECIES_CLOSED_LOOP_COUPLING": "1",
        "BIOCHEM_CORRECTOR_COUPLING": "1",
        "SPECIES_SCHEDULED_SAMPLING": "0",
    }
    # Shared prec-iter stack: bind train loss to cold deploy before N+/mirror.
    prec_config: dict[str, Any] = {
        **v3_config,
        "geom_feats": True,
        "geom_feats_rich": True,
        "flux_stag_feat": True,
        "mature_fp_exempt": False,
        "gate_fp_weight": 6.0,
        "teacher_fp_frac": 0.0,
        "teacher_noise": 0.02,
        "underpred_weight": 1.0,
        "closed_loop_init": 0.55,
        "step_mass_penalty": 0.75,
        "step_prec_fp_penalty": 0.5,
        "final_mass_penalty": 1.5,
        "final_mass_target": 1.2,
        "final_prec_fp_penalty": 1.0,
        "freeze_backbone": False,
    }
    prec_runtime: dict[str, Any] = {
        **v3_runtime,
        "deploy_horizon": 40,
        "deploy_eval_full": True,
        "deploy_horizon_all_packs": False,
        "deploy_horizon_aux_cap": 40,
        "select_clot_score_weight": 0.90,
        "select_mat_f1_weight": 0.10,
        "select_mass_soft_lambda": 0.20,
        "select_mass_soft_target": 1.2,
        "select_mass_hard_max": 3.0,
        "select_overpaint_lambda": 0.30,
        "select_overpaint_frac_target": 0.08,
    }

    # Shared selection/rollout runtime for the stenosis sub-cohort legs from v3 on. v3..v6 each
    # spell this out verbatim (kept that way as historical record); v7+ share it so the ladder
    # differs ONLY in config_kwargs and each step stays a clean single-variable test.
    # select_mass_hard_max stays at 1.5: it is correct as *selection* policy. It was never
    # correct as *retention* policy, and that is fixed in the trainer (best_salvage.pth,
    # WALL_MODEL_PLAN.md s12.5 item 1), not by loosening the gate here.
    _SUBCOHORT_RUNTIME_V3PLUS: dict[str, Any] = {
        **v3_runtime,
        "select_clot_f1_weight": 0.70,
        "select_clot_score_weight": 0.30,
        "select_mat_f1_weight": 0.0,
        "select_front_speed_target_lambda": 0.15,
        "select_fp_fn_imbalance_lambda": 0.15,
        "select_mass_hard_min": 0.5,
        "select_mass_hard_max": 1.5,
        "select_f1_min_hard_floor": 0.30,
        "deploy_eval_time_fracs": "0.65,1.0",
        "deploy_horizon": 40,
        "deploy_eval_full": True,
        "deploy_horizon_all_packs": False,
        "deploy_horizon_aux_cap": 40,
    }

    # Phase-0e corrected selection window (WALL_MODEL_PLAN.md 20.2). Empirically derived, not
    # chosen: for each held-out vessel we swept the number of committed nodes and found the count
    # that maximises F1 at the ranking quality actually achievable. Mean 3.04x n_true, median
    # 3.00x, range 0.16-5.70. At AUC ~0.78 with 2-18% positives, over-committing buys recall
    # faster than it loses precision, so mass_ratio ~3 IS the F1-optimal operating point.
    #
    # _SUBCOHORT_RUNTIME_V3PLUS gates at [0.5, 1.5], which rejected EVERY best-scoring epoch this
    # project ever produced -- v2 ep5 (2.077), v3 ep5 (2.768), v6 ep3 (3.200), v10 ep12 (1.674),
    # v10 ep14 (2.021). Those legs were finding the right operating point and selection threw it
    # away, which is the mechanical explanation for sections 9-13.
    #
    # V3PLUS is left untouched so v3-v10 stay reproducible; new legs use this.
    _SUBCOHORT_RUNTIME_V11PLUS: dict[str, Any] = {
        **_SUBCOHORT_RUNTIME_V3PLUS,
        "select_mass_hard_min": 1.2,
        "select_mass_hard_max": 4.5,
        "select_mass_soft_target": 3.0,
        "select_mass_soft_lambda": 0.15,
    }

    specs: dict[str, MatGrowthLegSpec] = {
        # Phase1 v3: single-factor tweaks on promoted FS_ab_coupled wall-gen baseline.
        "WG_sweep_v3_01": MatGrowthLegSpec(
            code="WG_sweep_v3_01",
            label="Control: FS_ab_coupled wall-gen baseline (auto+coupled, drop-xy)",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={**v3_config},
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        "WG_sweep_v3_02": MatGrowthLegSpec(
            code="WG_sweep_v3_02",
            label="Geom Feats (+rich)",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={**v3_config, "geom_feats": True, "geom_feats_rich": True},
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        "WG_sweep_v3_03": MatGrowthLegSpec(
            code="WG_sweep_v3_03",
            label="Flux / stagnation feat",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={**v3_config, "flux_stag_feat": True},
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        "WG_sweep_v3_04": MatGrowthLegSpec(
            code="WG_sweep_v3_04",
            label="Mirror-Y augmentation",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={**v3_config},
            runtime_kwargs={**v3_runtime, "augment_mirror_y": True},
            env_overrides={},
        ),
        "WG_sweep_v3_05": MatGrowthLegSpec(
            code="WG_sweep_v3_05",
            label="Geom + Flux combo",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
            },
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        "WG_sweep_v3_06": MatGrowthLegSpec(
            code="WG_sweep_v3_06",
            label="Geom + Flux + Mirror-Y",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
            },
            runtime_kwargs={**v3_runtime, "augment_mirror_y": True},
            env_overrides={},
        ),
        "WG_sweep_v3_07": MatGrowthLegSpec(
            code="WG_sweep_v3_07",
            label="Teacher noise off (0.0)",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={**v3_config, "teacher_noise": 0.0, "teacher_fp_frac": 0.0, "teacher_blur": 0.0},
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        "WG_sweep_v3_08": MatGrowthLegSpec(
            code="WG_sweep_v3_08",
            label="Teacher noise boost (0.04)",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={**v3_config, "teacher_noise": 0.04},
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        # Feat-dim fix re-run: same ablations as WG_sweep_v3_02/03/05/06 after pack-cache
        # fingerprint + continuous_feature_dim band extras + warm-start input widen.
        "WG_featfix_01": MatGrowthLegSpec(
            code="WG_featfix_01",
            label="Featfix: Geom Feats (+rich)",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={**v3_config, "geom_feats": True, "geom_feats_rich": True},
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        "WG_featfix_02": MatGrowthLegSpec(
            code="WG_featfix_02",
            label="Featfix: Flux / stagnation feat",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={**v3_config, "flux_stag_feat": True},
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        "WG_featfix_03": MatGrowthLegSpec(
            code="WG_featfix_03",
            label="Featfix: Geom + Flux combo",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
            },
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        "WG_featfix_04": MatGrowthLegSpec(
            code="WG_featfix_04",
            label="Featfix: Geom + Flux + Mirror-Y",
            no_init=False,
            init_ckpt=wall_gen_init,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
            },
            runtime_kwargs={**v3_runtime, "augment_mirror_y": True},
            env_overrides={},
        ),
        # Same feature/runtime stack as WG_featfix_03; data axis = clot-rich N+ LOAO.
        # Warm-start from featfix_03 (holdout patient020 was never in that ckpt's train set).
        "WG_clotrich_nplus": MatGrowthLegSpec(
            code="WG_clotrich_nplus",
            label="Clot-rich N+: featfix_03 stack, expand sites, hold out patient020",
            no_init=False,
            init_ckpt=WG_FEATFIX_03_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
            },
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        # Multi-hop flow: adds hop-2/hop-3 neighbourhood mean speed as a trailing block.
        # 97% of patient020's FPs are a *distant* wrong pocket (median 56 hops), and the
        # label lives on wall nodes where u=v=0 by no-slip -- so 1-hop flow separates
        # TP/FP at AUC 0.41 while hop-2 separates at 0.94. Warm-start widens conv1 with
        # zero columns, so this starts as an exact functional copy of the N+ checkpoint.
        "WG_multihop": MatGrowthLegSpec(
            code="WG_multihop",
            label="Multi-hop flow: hop2/hop3 neighbourhood speed, warm from N+ (0.500)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "flow_feats_multihop": True,
            },
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        # Identical in every respect except the new feature -- isolates it from the
        # warm-start / lr / cohort changes. Without this arm a gain is unattributable.
        "WG_multihop_ctrl": MatGrowthLegSpec(
            code="WG_multihop_ctrl",
            label="Control: same stack + warm start, multi-hop OFF",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "flow_feats_multihop": False,
            },
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        # N+ v2: same data/stack, but selection + train aligned to cold deploy score.
        # Soft mass/overpaint gate, deploy_horizon aux, mature FP on, heads-only FT.
        "WG_clotrich_nplus_v2": MatGrowthLegSpec(
            code="WG_clotrich_nplus_v2",
            label="Clot-rich N+ v2: mass-gated select + deploy_horizon + light FT",
            no_init=False,
            init_ckpt=WG_FEATFIX_03_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                # Precision tilt (warm spray teachers, not cold under-seed).
                "mature_fp_exempt": False,
                "gate_fp_weight": 4.0,
                "teacher_fp_frac": 0.02,
                # Train–deploy: modest closed-loop init bump (not +unroll at once).
                "closed_loop_init": 0.55,
                # Soft rolled-state mass / FP (cheap differentiable signal).
                "final_mass_penalty": 0.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 0.35,
                # Light fine-tune: freeze SAGE, train heads/gates.
                "freeze_backbone": True,
            },
            runtime_kwargs={
                **v3_runtime,
                # Aux on a few packs only (12-vessel full aux is too heavy on 4GB).
                "deploy_horizon": 40,
                "deploy_eval_full": True,
                "deploy_horizon_all_packs": False,
                "deploy_horizon_aux_cap": 40,
                # Soft mass-gated deploy_only; hard reject catastrophe spray.
                "select_clot_score_weight": 0.90,
                "select_mat_f1_weight": 0.10,
                "select_mass_soft_lambda": 0.15,
                "select_mass_soft_target": 1.2,
                "select_mass_hard_max": 3.5,
                "select_overpaint_lambda": 0.25,
                "select_overpaint_frac_target": 0.08,
            },
            env_overrides={},
        ),
        # Stenosis/aneurysm sub-cohort recall fine-tune (WALL_MODEL_PLAN.md s9, 2026-08-05).
        # Zero-shot WG_clotrich_nplus + flow gate pct=25 already scores deploy_clot_f1=0.650
        # on patient043 with NO training on this cohort -- purity/precision there is already
        # near its selection ceiling (0.650 of an oracle 0.697). The diagnosed gap is
        # under-seeding (mass_ratio 0.653, mat_front_speed_ratio 0.862, FN=44 vs FP=11), not
        # spurious pockets, so this leg is the mirror image of WG_prec_iter/WG_clotrich_nplus_v2:
        # it turns the underpred:fp loss ratio UP (2.0:8.0 -> 4.0:4.0) instead of down, and
        # selects checkpoints on front-growth completeness instead of precision.  Backbone
        # frozen -- 5-vessel cohort, light fine-tune only, so the warm-start's broader (all
        # -vessel) behaviour for the follow-on phase (s9) is not overwritten.
        "WG_stenosis_subcohort_ft": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft",
            label="Stenosis/aneurysm sub-cohort: underpred-tilted loss + front-growth select, "
                  "warm from N+ (0.650 zero-shot on patient043)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                # Recall tilt: was underpred=2.0/fp=8.0 (4x precision-favoured) in the N+
                # warm start. Flip toward 1:1 -- do not overshoot into the opposite failure
                # mode (spurious pockets) that motivated the original ratio elsewhere.
                "underpred_weight": 4.0,
                "fp_weight": 4.0,
                # Light fine-tune only: 5 train vessels, warm-started from a checkpoint that
                # already clears 0.60 zero-shot -- freeze the trunk, train heads/gates only.
                "freeze_backbone": True,
            },
            runtime_kwargs={
                **v3_runtime,
                # Primary = strict deploy_clot_f1 (wall-gen gate convention), soft clout score
                # as tiebreak only -- avoid selecting a precision-mirage checkpoint on a
                # 5-vessel cohort where mat_f1 alone would be noisy.
                "select_clot_f1_weight": 0.70,
                "select_clot_score_weight": 0.30,
                "select_mat_f1_weight": 0.0,
                # Reward front-growth completeness / penalize FN-heavy underseed -- the two
                # diagnosed gaps (mat_front_speed_ratio=0.862, FN=44 vs FP=11 on patient043).
                "select_front_speed_lambda": 0.20,
                "select_fn_fp_lambda": 0.20,
                # Guardrail: never promote a checkpoint that under-seeds MORE than today's
                # zero-shot floor (mass_ratio 0.653) -- this leg's whole point is growing that
                # number toward 1.0, not shrinking it further via a precision-mirage.
                "select_mass_hard_min": 0.5,
                # Light aux (small-cohort GPU budget), matching WG_clotrich_nplus_v2 precedent.
                "deploy_horizon": 40,
                "deploy_eval_full": True,
                "deploy_horizon_all_packs": False,
                "deploy_horizon_aux_cap": 40,
            },
            env_overrides={},
        ),
        # v2 -- v1 regressed (deploy_clot_f1 0.650 -> 0.522) by overshooting past balance into
        # the opposite failure (mass 0.653 -> 2.59, front_speed 0.862 -> 2.99, FP 11 -> 157).
        # Root causes, each fixed here (WALL_MODEL_PLAN.md s9.8-s9.9):
        #  1. Loss-ratio move was too large for a frozen-trunk FT (2:8 -> 4:4, full parity in
        #     one step). v2 moves it half as far (2:8 -> 3:6), matching WG_prec_front's more
        #     moderate single-notch precedent instead of jumping straight to 1:1.
        #  2. select_mass_hard_min guarded against MORE under-seeding but nothing guarded
        #     against over-seeding -- v1's mass ballooned to 2.59 and nothing could reject it.
        #     v2 adds a symmetric select_mass_hard_max.
        #  3. Checkpoint selection graded WITHOUT the pocket gate (CLOT_POCKET_GATE_PCT was
        #     never set during training -- only the standalone post-training eval set it), so
        #     selection picked the best checkpoint under conditions that don't match how the
        #     checkpoint is actually deployed. v2 sets the gate for the whole training run via
        #     env_overrides so selection sees exactly what the final deploy eval will show.
        #  4. Training windows never START past t0=132 of a ~200-step timeline (the legacy
        #     per-vessel formula) -- the last third of the horizon was only ever seen as a
        #     continuation of an earlier window, never as a fresh rollout start. Late-forming
        #     clot (this cohort's whole diagnosis) gets structurally under-sampled. v2 raises
        #     train_t0_coverage_frac so windows can start almost anywhere in the timeline.
        #  5. Selection graded a single point (t_final) only -- a checkpoint that looks fine at
        #     t=200 could already have gone wrong earlier and nothing would catch it. v2 grades
        #     at two sliding points (t=0.65*last, t=last) and adds a hard floor on the WORSE of
        #     the two, so a checkpoint must hold up across the horizon, not just at the end.
        # Costs ~2x the per-epoch deploy-eval wall-clock of v1 (two full rollouts graded per
        # epoch instead of one) -- budget accordingly.
        "WG_stenosis_subcohort_ft_v2": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v2",
            label="Stenosis/aneurysm sub-cohort v2: moderate recall tilt + symmetric mass guard "
                  "+ gated + full-horizon sliding-window selection",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                # (1) Half the v1 move: 2.0/8.0 -> 3.0/6.0, not 4.0/4.0.
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                # (4) 85% of each vessel's own timeline, per-vessel (train_t0_per_vessel stays
                # True via v3_runtime) -- clamped by TRAIN_T0_COVERAGE_MIN_RUNWAY so a window
                # starting near the end still has room for the curriculum's largest unroll.
                "train_t0_coverage_frac": 0.85,
            },
            runtime_kwargs={
                **v3_runtime,
                "select_clot_f1_weight": 0.70,
                "select_clot_score_weight": 0.30,
                "select_mat_f1_weight": 0.0,
                "select_front_speed_lambda": 0.20,
                "select_fn_fp_lambda": 0.20,
                # (2) Symmetric guard. 1.5 sits above every off-gate mass seen pre-finetune
                # across the cohort (039-043 ranged 0.97-2.03), so it doesn't reject legitimate
                # recall gains, but stops the kind of runaway spray v1 produced (2.59, and
                # climbing toward 4+ pre-gate) from ever being promoted.
                "select_mass_hard_min": 0.5,
                "select_mass_hard_max": 1.5,
                # (5) Sliding-window selection: t=0.65*last (roughly where the old t0 cap sat)
                # and t=last (final). Mean drives the primary score; deploy_clot_f1_min (the
                # worse of the two) is hard-floored below so a checkpoint can't pass on the
                # strength of the final point alone. 0.30 is well below the v1 floor's own
                # 0.650 zero-shot -- generous on this first corrected attempt, tightenable once
                # v2 establishes a working baseline.
                "select_f1_min_hard_floor": 0.30,
                "deploy_eval_time_fracs": "0.65,1.0",
                # Light aux (small-cohort GPU budget), matching WG_clotrich_nplus_v2 precedent.
                "deploy_horizon": 40,
                "deploy_eval_full": True,
                "deploy_horizon_all_packs": False,
                "deploy_horizon_aux_cap": 40,
            },
            # (3) Not a typed PushforwardConfig/BiochemRuntimeConfig field -- CLOT_POCKET_GATE_PCT
            # is a raw eval-time env toggle (src/evaluation/pocket_gate.py), read fresh on every
            # grading call and untouched by canonical_deploy_clot_metrics' env snapshot/restore
            # (not in _PROTOCOL_ENV_KEYS/_NOISE_ENV_KEYS), so setting it once here holds for the
            # whole run. Must match the pct the launcher passes to the final standalone eval.
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # v3 -- EXACTLY v2's config plus one mechanism: the GT-relative, time-resolved growth
        # brake. Deliberately a single-mechanism A/B against v2 (WALL_MODEL_PLAN.md s9.11).
        #
        # The growth-arrest probe (s9.10, scripts/probe_growth_arrest.py, zero-shot warm-start
        # across the s9.4 cohort) found the real defect, and it is NOT "no arrest":
        #   * The model's clot ONSET is anti-correlated with the truth, perfectly monotone (n=5):
        #       deep mass  0 ->  GT onset t=55, model t=18  (-37, EARLY)   patient039
        #       deep mass  8 ->  GT onset t=60, model t=20  (-40, EARLY)   patient040
        #       deep mass  9 ->  GT onset t=60, model t=20  (-40, EARLY)   patient043 (holdout)
        #       deep mass 68 ->  GT onset t=20, model t=80  (+60, LATE )   patient042
        #       deep mass 74 ->  GT onset t=20, model t=60  (+40, LATE )   patient041
        #     i.e. vessels that clot early AND thick are exactly the ones it starts latest on.
        #     This also supplies the mechanism behind s9.5's deep-mass/coverage correlation:
        #     that was never a coverage problem, it is a phase error.
        #   * On the holdout the LOCATION is already right (precision 0.96 at t=80, 0.83 at
        #     t_final; the 11 nodes it fires early at t=40 are all TP by t=80, FP drops to 1).
        #     The whole t_final deficit is FN=42 / recall 0.558 / mass 0.674 -- it needs MORE
        #     growth, correctly timed, not braking.
        #
        # Why the brake is still the right single change: rolled_final_mass_fp_penalty is
        # GT-RELATIVE at every unroll step (n_gt clamps to 1, so while GT is still empty a
        # premature commit of N nodes yields mass_ratio=N and softplus(N-1.2) fires hard). So it
        # is a PREMATURE-FIRING suppressor, not the late-overgrowth suppressor s9.10 first
        # called it -- and being GT-relative it stays silent on patient041/042 where the model
        # is behind GT. Correct behaviour on both halves of a cohort that splits early/late.
        # That also explains v1/v2 mechanically: raising underpred_weight increases growth
        # UNIFORMLY, including where GT is still zero, and 200 autoregressive steps compound it
        # into mass 4.0. The brake is what makes recall pressure safe by making it time-aware.
        #
        # Consequently v3 KEEPS v2's recall pressure (underpred 3.0 / fp 6.0) rather than
        # lowering it: 4 of 5 vessels incl. the holdout are under-grown, the brake is silent
        # below target so there was nothing to protect them from, and holding the ratio fixed
        # is what makes this a clean attribution of the brake itself. Backbone stays FROZEN for
        # the same reason plus a diagnosed one -- the holdout's location is already correct, so
        # the defect is rate/onset, which the readout heads govern; unfreezing would add a
        # second uncontrolled variable with no diagnosed need. If v3 under-grows on the LATE
        # vessels (041/042), that is the arm where unfreezing (v3b) earns its place, since
        # their precision is genuinely poor (0.46/0.60) and location IS wrong there.
        "WG_stenosis_subcohort_ft_v3": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v3",
            label="Stenosis/aneurysm sub-cohort v3: v2 + GT-relative time-resolved growth brake "
                  "(single-mechanism A/B vs v2)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                # --- identical to v2 from here ---
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                # --- the single mechanism v3 adds (values = WG_prec_iter's own, not new) ---
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
            },
            runtime_kwargs={
                # Identical to v2 (v3_runtime base, same weights/bounds/eval grid) EXCEPT the
                # two selection terms below. Selection does not enter the training gradient --
                # it only picks among epochs -- so changing it does not confound the brake A/B.
                **v3_runtime,
                "select_clot_f1_weight": 0.70,
                "select_clot_score_weight": 0.30,
                "select_mat_f1_weight": 0.0,
                # Symmetric replacements for v1/v2's confirmed-dead terms (s9.10): the old
                # select_front_speed_lambda rewards min(front_speed, 1.5), so it saturated to a
                # flat +0.30 every epoch (front_speed ran 2.5-5.06) AND rewarded overshoot on
                # the way there; the old select_fn_fp_lambda only fires FN-heavy, so it read
                # 0.000 every epoch once the regime turned FP-heavy. These penalize DEVIATION
                # from front_speed=1.0 / FN-FP imbalance in either direction. The originals stay
                # untouched and at 0.0 here (other legs rely on their exact existing formula).
                "select_front_speed_target_lambda": 0.15,
                "select_fp_fn_imbalance_lambda": 0.15,
                # Same bounds as v2 -- now anchored to t_final (s9.10 fix), not the sliding
                # window mean, so this guards the exact quantity that blew up on patient043.
                "select_mass_hard_min": 0.5,
                "select_mass_hard_max": 1.5,
                "select_f1_min_hard_floor": 0.30,
                "deploy_eval_time_fracs": "0.65,1.0",
                "deploy_horizon": 40,
                "deploy_eval_full": True,
                "deploy_horizon_all_packs": False,
                "deploy_horizon_aux_cap": 40,
            },
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # v5 -- THE PREREQUISITE EXPERIMENT (WALL_MODEL_PLAN.md s11.3 change B).
        # v1-v4 all failed for reasons unrelated to the knob being tuned, and s9.12/s9.14
        # found why: the training objective supervises 5-10 step TBPTT windows while deploy
        # is a 200-step free rollout, so loss moves 0.2% while deploy F1 swings 0.37->0.61.
        # The FP term provably never fires (v3 vs v4 bit-identical) and the rolled-state
        # brake moves the rollout ~1%. The ONLY loss term that evaluates a rolled-out state
        # is the deploy_horizon aux -- and it is capped at 40 of ~200 steps, i.e. it never
        # sees the regime where over-painting actually accumulates (GT saturates by t~100,
        # the model keeps depositing to t=200).
        #
        # v5 = v3 with ONE change: deploy_horizon/aux_cap 40 -> 150. tbptt_tail=5 still
        # bounds gradient memory (activations beyond the tail are detached), so the extra
        # cost is forward-only. This tests the claim that gates all further training work:
        # does making the objective SEE the deploy horizon make decreasing loss actually
        # track deploy score? v3-vs-v5 is a clean single-variable test of exactly that.
        "WG_stenosis_subcohort_ft_v5": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v5",
            label="Stenosis/aneurysm sub-cohort v5: v3 + full-horizon deploy aux (40 -> 150) "
                  "-- tests whether loss can be made to track deploy score",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
            },
            runtime_kwargs={
                **v3_runtime,
                "select_clot_f1_weight": 0.70,
                "select_clot_score_weight": 0.30,
                "select_mat_f1_weight": 0.0,
                "select_front_speed_target_lambda": 0.15,
                "select_fp_fn_imbalance_lambda": 0.15,
                "select_mass_hard_min": 0.5,
                "select_mass_hard_max": 1.5,
                "select_f1_min_hard_floor": 0.30,
                "deploy_eval_time_fracs": "0.65,1.0",
                # THE single change from v3 (40 -> 150 on both).
                "deploy_horizon": 150,
                "deploy_eval_full": True,
                "deploy_horizon_all_packs": False,
                "deploy_horizon_aux_cap": 150,
            },
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # v6 -- s11.3 change B, RE-SPECIFIED TWICE (see s12.2, s12.3).
        # v5 lengthened the deploy_horizon aux 40 -> 150 and changed NOTHING (bit-identical to
        # v3). Census: the aux gets its OWN opt.zero_grad/backward/step, so it is 1 optimizer
        # step out of 757; and grad_clip=1.0 clips the update, so scaling its loss magnitude is
        # neutered too. COUNT is the only lever -- but making the aux 20% of steps needs ~190
        # rollouts of 150 steps per epoch (~28k model evals vs the main loop's 3.8k), which is
        # computationally infeasible. And --max-windows cannot rebalance it either: it
        # truncates to `windows[:N]`, i.e. the EARLIEST t0 only, which would bias training to
        # early times and defeat train_t0_coverage_frac (s9.9 change 4).
        #
        # So v6 attacks the same problem from the other side: instead of making ONE term
        # horizon-aware, make EVERY term horizon-aware.
        #   curriculum_unroll: True -> False   (curriculum pins unroll to 5 through epoch 10,
        #                                       overriding the configured value)
        #   unroll:              10 -> 25      (every one of the ~756 main windows now carries
        #                                       a 25-step rollout signal, not 5)
        #   deploy_horizon_all_packs: -> True  (aux on all train packs; val pack still excluded)
        # tbptt_tail=5 still bounds gradient memory, so the extra cost is forward-only:
        # ~756 x 25 = 18.9k evals/epoch vs v3's 3.8k, i.e. ~5x (~35 min/epoch). No t0 bias.
        #
        # The question is unchanged and is NOT "does v6 score higher": does decreasing loss now
        # TRACK deploy F1? v3 and v5 both sat at Spearman(loss, F1) = +0.314 -- weakly POSITIVE,
        # i.e. lower loss trended toward WORSE deploy score. If v6 turns that negative the
        # objective is finally aligned and further training work becomes interpretable. If a 5x
        # deeper rollout signal on every update still does not, then the per-step delta loss is
        # structurally misaligned with the thresholded-rollout metric, and s11 should move to
        # change D (explicit autocatalysis) rather than reweighting this objective further.
        #
        # V6 RAN (2026-08-06, 5100s) AND IS A CLEAN NEGATIVE -- s11.3 change B is CLOSED (s12.3).
        # Mechanism verified engaged (cur_unroll=25 all 6 epochs, 751 windows, all_packs=True) and
        # the objective really did change: loss 74.44-74.93 vs v3's 61.36-61.47, spread 0.66% vs
        # 0.18%. NOTHING downstream moved -- same fp=292 attractor 5/6 epochs, same mass ~4.03,
        # every epoch mass-rejected, no checkpoint. Spearman(loss, deploy_clot_score) did read
        # -0.406, but that is NOISE, not a win: exact permutation p=0.217, dropping one epoch
        # flips it to +0.05, and 5 of 6 score values lie within 0.002 of each other (ep2 and ep4
        # are identical to 9 d.p. on all six deploy metrics). The statistic that DOES resolve it:
        # deploy score is bimodal (one epoch at 0.44-0.50, the rest at ~0.26) and the loss of the
        # good epoch sits at z = +0.22 against the bad ones -- v3/v5 at unroll 5 were z = -0.30,
        # so 5x the rollout depth moved alignment the WRONG way. |z| < 0.5 in every leg: the loss
        # cannot see a near-doubling of deploy score. DO NOT re-specify change B a fourth time.
        # The bigger finding is in s12.4: a two-state attractor (35/41 epochs saturated at
        # fp>=292, 6/41 excursions, score strictly monotone in fp), and select_mass_hard_max=1.5
        # discarding every excursion -- v2..v6 wrote ZERO best.pth between them.
        "WG_stenosis_subcohort_ft_v6": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v6",
            label="Stenosis/aneurysm sub-cohort v6: unroll 5->25 on EVERY window + aux on all "
                  "packs -- makes the whole objective horizon-aware (s11.3 change B, retry 2)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
                # THE change: every main window becomes a 25-step rollout.
                "curriculum_unroll": False,
                "unroll": 25,
            },
            runtime_kwargs={
                **v3_runtime,
                "select_clot_f1_weight": 0.70,
                "select_clot_score_weight": 0.30,
                "select_mat_f1_weight": 0.0,
                "select_front_speed_target_lambda": 0.15,
                "select_fp_fn_imbalance_lambda": 0.15,
                "select_mass_hard_min": 0.5,
                "select_mass_hard_max": 1.5,
                "select_f1_min_hard_floor": 0.30,
                "deploy_eval_time_fracs": "0.65,1.0",
                "deploy_horizon": 150,
                "deploy_eval_full": True,
                "deploy_horizon_all_packs": True,
                "deploy_horizon_aux_cap": 150,
            },
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # =====================================================================================
        # v7/v8/v9 -- the post-change-B ladder (s12.6). Built on a NEW root cause that v6's
        # post-mortem turned up and that invalidates the premise of v3/v4/v5/v6 alike:
        #
        #   THE ROLLED-STATE LOSS TERMS WERE NUMERICALLY DEAD, NOT WEAK.
        #   soft occupancy is sigmoid(soft_k * (pred - thr)) with soft_k=40 and the actual Mat
        #   commit threshold thr=1e-4. That returns 0.4990 for an EMPTY node and 0.5000 at
        #   threshold -- the "soft committed set" is a constant 0.5 everywhere. Measured:
        #     rolled_final_mass_fp_penalty  29.11500 -> 29.15093  (0.12%)
        #     as the rollout goes from EMPTY to 12x OVER-PAINTED,
        #   and its soft mass_ratio reads 19.96 in every case instead of 0.00 -> 12.00.
        #   This is the mechanical cause of s9.12's "the brake moved the rollout ~1%".
        #   Fix: rolled_soft_k_relative=True makes the argument k*(pred-thr)/thr, i.e. a
        #   scale-free RELATIVE deviation. Dynamic range 0.12% -> 97.7%.
        #   The flag defaults False, so v1..v6 stay bit-reproducible.
        #
        # Second finding, same family: fp_weight compares per-step pred_delta against
        # fp_thresh=2e-5 while predicted deltas run ~1e-7, so the FP branch never selects any
        # node. That is the mechanical cause of s9.14's v3-vs-v4 bit-identity. NOT changed in
        # this ladder (one variable at a time); the soft-F_beta term below applies FP pressure
        # on the rolled state, which is the better place for it anyway.
        #
        # v7 = v3 with ONE change: rolled_soft_k_relative=True. This does not add a term; it
        # brings v3's OWN brake to life for the first time. v3-vs-v7 therefore finally tests
        # what s9.11 believed it was testing.
        "WG_stenosis_subcohort_ft_v7": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v7",
            label="Stenosis/aneurysm sub-cohort v7: v3 with the soft-occupancy scale fixed "
                  "(rolled_soft_k_relative) -- the first run in which the brake is not dead",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
                # THE single change from v3.
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
            },
            runtime_kwargs={**_SUBCOHORT_RUNTIME_V3PLUS},
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # v10 -- s13.6, step 4a of the architecture ladder. ONE variable off v7:
        # latent_dropout 0.0 -> 0.30. s11.2.1's largest structural imbalance is that z_kin is
        # 256 of a 287-dim input (89%) and is a FROZEN, off-task kinematics latent. Dropout on
        # it is the cheapest possible probe of "is the readout leaning on the off-task latent
        # instead of the geometry/flow channels" -- config-only, no new code.
        # Base is v7 (= v3 + the s12.6.1 occupancy fix), NOT v9: v9 bundles three changes and
        # would make this unattributable, which is the exact mistake that cost v1-v6.
        # 14 epochs, per s12.8.2's floor -- 6 would be unreadable.
        "WG_stenosis_subcohort_ft_v10": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v10",
            label="Stenosis/aneurysm sub-cohort v10: v7 + latent_dropout 0.30 "
                  "(s13 step 4a -- cheapest architecture probe, single variable)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
                # inherited from v7 (already tested in isolation, s12.6.6)
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
            },
            runtime_kwargs={
                **_SUBCOHORT_RUNTIME_V3PLUS,
                # THE single change from v7.
                "latent_dropout": 0.30,
            },
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # =================================================================================
        # WG_phase1_baseline -- the section 21.3 RE-BASELINE. Not an A/B: it switches on the
        # whole Phase-0 foundation at once, deliberately, and becomes the reference point that
        # every later single-variable leg is measured against. Nothing is attributable ACROSS
        # it, and no number before it is comparable with any number after it.
        #
        # What changes vs v7-v10, all together and only here:
        #   cohort        5 ad-hoc vessels -> WALL_COHORT_V2_TRAIN (27), sealed 8 held out (21.1)
        #   priors        stored (leaked CFD) -> analytic, the Z2 legal contract (16.1, 17)
        #   labels        fixed 1e-4 -> rel_max @ 10% of each vessel's peak Mat (20.3, 21.2)
        #   selection     mass window [0.5,1.5] -> [1.2,4.5] around the MEASURED F1 optimum
        #                 of 3.04x n_true (20.2) -- the old window rejected every good epoch
        #   objective     soft-F_beta surrogate live, occupancy scale fixed (12.6)
        # Backbone stays frozen and the growth law stays generic: change D / change E
        # attribution comes AFTER this, one variable at a time.
        "WG_phase1_baseline": MatGrowthLegSpec(
            code="WG_phase1_baseline",
            label="Phase 1 re-baseline: cohort v2 + analytic priors + rel_max labels + "
                  "measured mass window (s21.3). Reference point, not an A/B.",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                # Follows the measured optimum (20.2), not physical mass matching.
                "final_mass_target": 3.0,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
                "rolled_soft_f1_weight": 120.0,
                "rolled_soft_f1_beta": 1.0,
                "step_soft_f1_weight": 40.0,
                "mat_label_thresh_mode": "rel_max",
                "mat_label_rel_frac": 0.10,
            },
            runtime_kwargs={
                **_SUBCOHORT_RUNTIME_V11PLUS,
                "prior_source": "analytic",
            },
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # WG_phase2a_decomp -- IDENTICAL training config to WG_phase1_baseline. The only
        # difference is instrumentation (per-term loss accounting), which never enters the
        # optimizer. Purpose (s23): Phase 1 gave the first trustworthy misalignment measurement
        # -- 10 distinct deploy states, stable jackknife, Spearman(loss, score) = +0.564 -- but
        # a total says nothing about WHICH of ~8 terms fights the metric, and guessing that is
        # exactly how v1-v10 were spent. This run correlates each term with deploy score
        # separately, and doubles as a reproducibility check on Phase 1.
        "WG_phase2a_decomp": MatGrowthLegSpec(
            code="WG_phase2a_decomp",
            label="Phase 2a: Phase-1 config + per-term loss decomposition (instrumentation only)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 3.0,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
                "rolled_soft_f1_weight": 120.0,
                "rolled_soft_f1_beta": 1.0,
                "step_soft_f1_weight": 40.0,
                "mat_label_thresh_mode": "rel_max",
                "mat_label_rel_frac": 0.10,
            },
            runtime_kwargs={**_SUBCOHORT_RUNTIME_V11PLUS, "prior_source": "analytic"},
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # WG_phase2b_nobrake -- Phase 2a's decomposition made the objective fix TARGETED for the
        # first time (s23.7). Measured over 8 epochs, config byte-identical to Phase 1:
        #
        #   term            share of loss   rho(term, deploy_clot_score)
        #   final_mass_fp        39%              +0.464   <- minimising it HURTS the score
        #   step_mass_fp         16%              +0.429   <- same
        #   final_soft_f1        15%              -0.321   <- correct direction
        #   step_soft_f1          5%              -0.393   <- correct direction
        #
        # The mass-brake family is 54% of the loss and points the WRONG way; the soft-F_beta
        # surrogate is 20% and points the right way. They pull against each other 2.8:1 in
        # favour of the brake -- which is precisely the mechanism of Phase 1's mass collapse to
        # 0.43-0.96 when the score optimum is ~2.0-2.4 (23.3).
        #
        # ONE conceptual change from Phase 1: switch the brake family OFF, leaving soft-F_beta
        # as the rolled-state objective. This is not another reweighting guess -- it removes the
        # single term family measured to be anti-correlated with the metric.
        #
        # `final_mass_target` also drops 3.0 -> 2.2. That is NOT a second variable: with the
        # brake at zero the target is inert (it only parameterises the brake). It is set to the
        # measured score-optimum so the constant is not left stale if the brake is restored.
        "WG_phase2b_nobrake": MatGrowthLegSpec(
            code="WG_phase2b_nobrake",
            label="Phase 2b: Phase-1 minus the mass-brake family (s23.7 -- the terms measured "
                  "anti-correlated with deploy_clot_score)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                # THE change: the anti-correlated family, off.
                "step_mass_penalty": 0.0,
                "step_prec_fp_penalty": 0.0,
                "final_mass_penalty": 0.0,
                "final_prec_fp_penalty": 0.0,
                "final_mass_target": 2.2,
                "mature_fp_exempt": False,
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
                "rolled_soft_f1_weight": 120.0,
                "rolled_soft_f1_beta": 1.0,
                "step_soft_f1_weight": 40.0,
                "mat_label_thresh_mode": "rel_max",
                "mat_label_rel_frac": 0.10,
            },
            runtime_kwargs={**_SUBCOHORT_RUNTIME_V11PLUS, "prior_source": "analytic"},
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # =================================================================================
        # s24 fix legs. Effectively SINGLE-VARIABLE despite touching several knobs, because
        # Phase 2b measured the mass terms to be inert (removing them moved mass 0.028) and
        # fixes 2/5 are accounting and scale repair. The one BEHAVIOURAL variable in 3a is
        # closed_loop_init 0.45 -> 1.0.
        #
        # NB a correction to s24: `closed_loop_init` is NOT dead -- it is consumed at
        # train_species_pushforward_continuous.py:1334 and already starts 45%% of windows from
        # `rollout_prefix_log_state`, i.e. the model's own free-running state. The claim that
        # "every window starts from GT" was wrong. Raising it to 1.0 makes EVERY window start
        # from the model's own drifted state, which is the only state deploy ever sees.
        "WG_phase3a_closedloop": MatGrowthLegSpec(
            code="WG_phase3a_closedloop",
            label="s24 fixes + closed_loop_init 0.45->1.0 (every window from the model's own state)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                # (3) mass OUT of the loss entirely. Phase 2b proved these terms are inert
                # (removing them moved mass by 0.028), and deploy_clot_score is relaxed
                # PRECISION gated by a recall floor -- a mass target optimises a quantity the
                # metric does not reward. Mass survives only as a SELECTION guard.
                "step_mass_penalty": 0.0,
                "step_prec_fp_penalty": 0.0,
                "final_mass_penalty": 0.0,
                "final_prec_fp_penalty": 0.0,
                "mature_fp_exempt": False,
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
                "rolled_soft_f1_weight": 120.0,
                # (3b) beta 1.0 -> 0.5. F1 weights precision and recall equally, but the metric
                # is precision-dominated once recall clears the floor. The surrogate meant to
                # track the metric was itself mis-specified against it.
                "rolled_soft_f1_beta": 0.5,
                "step_soft_f1_weight": 40.0,
                # (5) lift the final-state Huber onto the growth loss's value scale (1.5e9x).
                "final_state_value_scaled": True,
                "mat_label_thresh_mode": "rel_max",
                "mat_label_rel_frac": 0.10,
                # THE behavioural variable.
                "closed_loop_init": 1.0,
            },
            runtime_kwargs={**_SUBCOHORT_RUNTIME_V11PLUS, "prior_source": "analytic"},
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # (4) z_kin ablation. Z1 measured the entire flow channel at 0.041 AUC while z_kin is
        # 256 of 287 input dims. Hard ablation (train AND eval) so the two match; `in_dim`
        # stays 287, sidestepping the s13.8 warm-start blocker that killed the shrink.
        "WG_phase3b_zkin_ablate": MatGrowthLegSpec(
            code="WG_phase3b_zkin_ablate",
            label="s24 fix 4: hard z_kin ablation (zeroed at train AND eval), in_dim unchanged",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                # (3) mass OUT of the loss entirely. Phase 2b proved these terms are inert
                # (removing them moved mass by 0.028), and deploy_clot_score is relaxed
                # PRECISION gated by a recall floor -- a mass target optimises a quantity the
                # metric does not reward. Mass survives only as a SELECTION guard.
                "step_mass_penalty": 0.0,
                "step_prec_fp_penalty": 0.0,
                "final_mass_penalty": 0.0,
                "final_prec_fp_penalty": 0.0,
                "mature_fp_exempt": False,
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
                "rolled_soft_f1_weight": 120.0,
                # (3b) beta 1.0 -> 0.5. F1 weights precision and recall equally, but the metric
                # is precision-dominated once recall clears the floor. The surrogate meant to
                # track the metric was itself mis-specified against it.
                "rolled_soft_f1_beta": 0.5,
                "step_soft_f1_weight": 40.0,
                # (5) lift the final-state Huber onto the growth loss's value scale (1.5e9x).
                "final_state_value_scaled": True,
                "mat_label_thresh_mode": "rel_max",
                "mat_label_rel_frac": 0.10,
                "closed_loop_init": 1.0,
                # THE variable vs 3a.
                "latent_ablate": True,
            },
            runtime_kwargs={**_SUBCOHORT_RUNTIME_V11PLUS, "prior_source": "analytic"},
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # v8 = v7 + the soft-F_beta rolled-state surrogate (s12.5 change E). The brake, even
        # alive, is still only a SUPPRESSOR: final_mass_penalty is softplus(mass_ratio-target),
        # identically zero below target, and final_prec_fp_penalty is an FP fraction. Neither
        # has a TP numerator, so neither is monotone in F1. This term is the deploy metric
        # itself, softened -- 1 - soft_F_beta over the rolled committed set -- and is the direct
        # answer to "make training loss track deploy score", which is what change B failed to
        # achieve by reweighting a per-step regression (s12.3).
        # beta=1.0: the cohort over-paints 4x during training (fp=292), so this is not the place
        # for a recall tilt; the deploy-time under-seeding is a separate, gate-side problem.
        "WG_stenosis_subcohort_ft_v8": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v8",
            label="Stenosis/aneurysm sub-cohort v8: v7 + soft-F_beta rolled-state surrogate "
                  "(s12.5 change E -- the metric itself, softened)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
                # THE single change from v7. Weights are sized against the OBSERVED loss scale,
                # not picked by feel -- this is the exact trap v5 fell into. loss_scale=0.1
                # multiplies both rolled-state terms and the total loss runs ~61, so the shipped
                # -looking weight of 4.0 would cap the term at 0.4, i.e. 0.65% of the objective:
                # unable to steer anything, for the same reason the deploy_horizon aux could not
                # (s12.2). 120.0 caps it at 12.0, ~20% of the objective, which can.
                "rolled_soft_f1_weight": 120.0,
                "rolled_soft_f1_beta": 1.0,
                "step_soft_f1_weight": 40.0,
            },
            runtime_kwargs={**_SUBCOHORT_RUNTIME_V3PLUS},
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # v9 = v8 + s11.3 change D, the explicit gated-autocatalytic growth law. COMSOL grows
        # Mat as ~(Mas/Minf)*k_aa*AP -- rate proportional to material already committed locally,
        # ~90% of real Mat growth. The generic delta head has no such term, so it can propagate
        # but cannot ignite, which is exactly what s12.4's two-basin attractor looks like (35/41
        # epochs pinned at fp>=292, 6/41 excursions, nothing steering between them).
        # magnitude is multiplied by (k_dep + k_auto * local_committed_frac); both learnable and
        # log-parameterised. At the k=1 init an isolated node keeps its current rate, so the
        # warm start is undisturbed at t=0 and only committed neighbourhoods accelerate.
        # NOTE: log_k_dep/log_k_auto are added to freeze_growth_backbone's head allowlist --
        # they ARE the growth law, and freezing them would silently disable the mechanism.
        "WG_stenosis_subcohort_ft_v9": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v9",
            label="Stenosis/aneurysm sub-cohort v9: v8 + explicit gated-autocatalytic growth "
                  "(s11.3 change D) -- adds the ignition term the two-basin attractor lacks",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                "fp_weight": 6.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
                "rolled_soft_k_relative": True,
                "rolled_soft_f1_k": 10.0,
                "rolled_soft_f1_weight": 120.0,
                "rolled_soft_f1_beta": 1.0,
                "step_soft_f1_weight": 40.0,
                # THE single change from v8.
                "autocatalytic_growth": True,
                "autocat_k_dep_init": 1.0,
                "autocat_k_auto_init": 1.0,
                "autocat_alpha": 0.8,
            },
            runtime_kwargs={**_SUBCOHORT_RUNTIME_V3PLUS},
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # v4 -- v3's brake moved the rollout ~1% on a model 400% off target (front_speed
        # 4.545 -> 4.605, t_final mass 4.02 -> 4.03). Comparing all four legs against their
        # OBSERVED mass on patient043 finally isolates the actual driver, and it is not any
        # knob v1/v2/v3 were tuning (WALL_MODEL_PLAN.md s9.12):
        #
        #     leg                 underpred   fp    t_final mass
        #     WG_clotrich_nplus       2.0    16.0   0.674   <- warm start, no FT
        #     WG_prec_iter            1.0    16.0   1.109   <- controls mass on p020
        #     v1                      4.0     4.0   4.200
        #     v2                      3.0     6.0  ~4.02
        #     v3 (+brake)             3.0     6.0   4.032
        #
        # underpred 4.0 -> 3.0 (a 33% cut) moves mass by 4%: underpred is nearly inert here.
        # fp_weight splits the table perfectly: every leg at 16.0 controls mass, every leg
        # that blew up had fp_weight CUT to 4-6. v1 cut it and v2/v3 inherited the cut.
        #
        # Root cause of the cut: fp_weight is not set by the geom/flux feature stack these
        # legs inherit, so it takes MAT_GROWTH_SIMPLE_RECIPE's 16.0 baseline -- but it was
        # documented as PushforwardConfig's bare 8.0 dataclass default, so "6.0" was designed
        # as a mild reduction when it was really a 2.7x cut. See the s9.10 correction.
        #
        # v4 is therefore v3 with ONE value changed: fp_weight 6.0 -> 16.0, restoring the
        # warm-start's own anti-FP pressure. Everything else -- including the brake, which
        # stays so its effect can still be read against v2 -- is byte-identical to v3, so
        # v3-vs-v4 is a clean single-variable test of fp_weight itself.
        "WG_stenosis_subcohort_ft_v4": MatGrowthLegSpec(
            code="WG_stenosis_subcohort_ft_v4",
            label="Stenosis/aneurysm sub-cohort v4: v3 + fp_weight restored to the warm-start's "
                  "16.0 (single-variable fp test; v1 cut it to 4.0 and v2/v3 inherited the cut)",
            no_init=False,
            init_ckpt=WG_CLOTRICH_NPLUS_CKPT,
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "underpred_weight": 3.0,
                # THE single change from v3. 16.0 = MAT_GROWTH_SIMPLE_RECIPE's baseline, i.e.
                # exactly what WG_clotrich_nplus (mass 0.674) and WG_prec_iter (mass 1.109)
                # both actually train at. Explicit, not inherited, so it cannot drift again.
                "fp_weight": 16.0,
                "freeze_backbone": True,
                "train_t0_coverage_frac": 0.85,
                "step_mass_penalty": 0.75,
                "step_prec_fp_penalty": 0.5,
                "final_mass_penalty": 1.5,
                "final_mass_target": 1.2,
                "final_prec_fp_penalty": 1.0,
                "mature_fp_exempt": False,
            },
            runtime_kwargs={
                **v3_runtime,
                "select_clot_f1_weight": 0.70,
                "select_clot_score_weight": 0.30,
                "select_mat_f1_weight": 0.0,
                "select_front_speed_target_lambda": 0.15,
                "select_fp_fn_imbalance_lambda": 0.15,
                "select_mass_hard_min": 0.5,
                "select_mass_hard_max": 1.5,
                "select_f1_min_hard_floor": 0.30,
                "deploy_eval_time_fracs": "0.65,1.0",
                "deploy_horizon": 40,
                "deploy_eval_full": True,
                "deploy_horizon_all_packs": False,
                "deploy_horizon_aux_cap": 40,
            },
            env_overrides={"CLOT_POCKET_GATE_PCT": "25"},
        ),
        # Small-cohort precision iteration: fix train–deploy mismatch before revisiting N+.
        # Stronger per-step + final mass/FP; no freeze; no teacher FP; mass-gated select.
        "WG_prec_iter": MatGrowthLegSpec(
            code="WG_prec_iter",
            label="Prec-iter: featfix_03 + step/final mass-FP + mass-gated select (small cohort)",
            no_init=False,
            init_ckpt=WG_FEATFIX_03_CKPT,
            init_mode="full",
            config_kwargs={**prec_config},
            runtime_kwargs={**prec_runtime},
            env_overrides={},
        ),
        # More shapes (exact N-S y-mirror), same small cohort + prec loss.
        "WG_prec_mirror": MatGrowthLegSpec(
            code="WG_prec_mirror",
            label="Prec-iter + Mirror-Y (more shapes, not more sites)",
            no_init=False,
            init_ckpt=WG_FEATFIX_03_CKPT,
            init_mode="full",
            config_kwargs={**prec_config},
            runtime_kwargs={**prec_runtime, "augment_mirror_y": True},
            env_overrides={},
        ),
        # Re-test N+/sites with the fixed prec objective (not the spray-prone v1/v2 recipe).
        "WG_prec_sites": MatGrowthLegSpec(
            code="WG_prec_sites",
            label="Prec loss + clot-rich sites expand (revisit N+ after objective fix)",
            no_init=False,
            init_ckpt=WG_FEATFIX_03_CKPT,
            init_mode="full",
            config_kwargs={**prec_config},
            runtime_kwargs={**prec_runtime},
            env_overrides={},
        ),
        # Mid expand (6 vessels): same prec loss; warm-start from prec_iter (not full N+).
        "WG_prec_mid": MatGrowthLegSpec(
            code="WG_prec_mid",
            label="Prec loss + mid cohort (6 vessels) from prec_iter",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={**prec_config},
            runtime_kwargs={**prec_runtime},
            env_overrides={},
        ),
        # Tight FT on small cohort: stronger mass/FP from prec_iter (no more sites).
        "WG_prec_ft": MatGrowthLegSpec(
            code="WG_prec_ft",
            label="Tight FT: stronger mass/FP from prec_iter (small cohort)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "step_mass_penalty": 1.0,
                "step_prec_fp_penalty": 0.75,
                "final_mass_penalty": 2.0,
                "final_prec_fp_penalty": 1.25,
                "gate_fp_weight": 8.0,
                "closed_loop_init": 0.60,
            },
            runtime_kwargs={
                **prec_runtime,
                "select_mass_soft_lambda": 0.25,
                "select_overpaint_lambda": 0.40,
            },
            env_overrides={},
        ),
        # Clot-rich LOAO: tight prec-FT recipe + init from best small-cohort ckpt.
        # Full N+/mid sprayed with lighter prec_iter loss; use stronger mass/FP here.
        # Launcher should pass --init to best of prec_iter/mirror/ft; sealed 043/044 excluded.
        "WG_prec_loao": MatGrowthLegSpec(
            code="WG_prec_loao",
            label="Clot-rich LOAO: tight mass/FP from best small-cohort (hold out 020)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "step_mass_penalty": 1.25,
                "step_prec_fp_penalty": 1.0,
                "final_mass_penalty": 2.5,
                "final_mass_target": 1.15,
                "final_prec_fp_penalty": 1.5,
                "gate_fp_weight": 8.0,
                "closed_loop_init": 0.60,
                "freeze_backbone": False,
            },
            runtime_kwargs={
                **prec_runtime,
                "select_mass_soft_lambda": 0.30,
                "select_mass_soft_target": 1.15,
                "select_mass_hard_max": 2.5,
                "select_overpaint_lambda": 0.45,
                "select_overpaint_frac_target": 0.06,
                "deploy_horizon_aux_cap": 30,
            },
            env_overrides={},
        ),
        # Fallback if full LOAO sprays: freeze SAGE, adapt heads only under tight mass.
        "WG_prec_loao_freeze": MatGrowthLegSpec(
            code="WG_prec_loao_freeze",
            label="Clot-rich LOAO freeze-backbone + tight mass/FP (spray fallback)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "step_mass_penalty": 1.25,
                "step_prec_fp_penalty": 1.0,
                "final_mass_penalty": 2.5,
                "final_mass_target": 1.15,
                "final_prec_fp_penalty": 1.5,
                "gate_fp_weight": 8.0,
                "closed_loop_init": 0.60,
                "freeze_backbone": True,
            },
            runtime_kwargs={
                **prec_runtime,
                "select_mass_soft_lambda": 0.30,
                "select_mass_soft_target": 1.15,
                "select_mass_hard_max": 2.5,
                "select_overpaint_lambda": 0.45,
                "select_overpaint_frac_target": 0.06,
                "deploy_horizon_aux_cap": 30,
            },
            env_overrides={},
        ),
        # Train WITH sparse commitment on (same weights as prec_iter; mask is behavioral).
        # Post-hoc eval masking stalled the front; these legs teach seed-then-grow under the gate.
        # Do NOT flip neighbor_commit_gate here -- that widens spatial_head and breaks warm-start.
        "WG_prec_seed": MatGrowthLegSpec(
            code="WG_prec_seed",
            label="Prec-iter + train-time frontier_hops=1 / nucleation_topk=0.05 (primary seed path)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "frontier_hops": 1,
                "nucleation_topk": 0.05,
            },
            runtime_kwargs={**prec_runtime},
            env_overrides={},
        ),
        "WG_prec_seed_fh2": MatGrowthLegSpec(
            code="WG_prec_seed_fh2",
            label="Prec-iter + train-time frontier_hops=2 / nucleation_topk=0.05",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "frontier_hops": 2,
                "nucleation_topk": 0.05,
            },
            runtime_kwargs={**prec_runtime},
            env_overrides={},
        ),
        "WG_prec_seed_tk02": MatGrowthLegSpec(
            code="WG_prec_seed_tk02",
            label="Prec-iter + train-time frontier_hops=1 / nucleation_topk=0.02 (tighter seed)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "frontier_hops": 1,
                "nucleation_topk": 0.02,
            },
            runtime_kwargs={**prec_runtime},
            env_overrides={},
        ),
        # Seed-location aux on prec stack (no hard frontier). Differentiable early pocket BCE
        # + light compactness; keep mass/FP primary. Warm-start WG_prec_iter.
        "WG_prec_seed_aux": MatGrowthLegSpec(
            code="WG_prec_seed_aux",
            label="Prec-iter + early seed-location aux (fh=0; small weight; select seed panel)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "frontier_hops": 0,
                "nucleation_topk": 0.0,
                "seed_aux_weight": 0.15,
                "seed_aux_early_steps": 3,
                "seed_aux_compact_weight": 0.05,
                "seed_aux_pos_weight": 4.0,
            },
            runtime_kwargs={
                **prec_runtime,
                "select_seed_prec_lambda": 0.10,
                "select_front_speed_lambda": 0.05,
                "select_fn_fp_lambda": 0.05,
            },
            env_overrides={},
        ),
        # Front/recall FT: seed_p was already ~1 on 020; FN + stalled front are the ceiling.
        # Raise underpred, ease gate/step FP; no hard frontier; seed_aux off. Select front+FN.
        "WG_prec_front": MatGrowthLegSpec(
            code="WG_prec_front",
            label="Prec-iter front/recall FT (underpred up, gate FP down; select front+FN)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "frontier_hops": 0,
                "nucleation_topk": 0.0,
                "seed_aux_weight": 0.0,
                "underpred_weight": 3.0,
                "gate_fp_weight": 3.0,
                "step_prec_fp_penalty": 0.35,
                "final_prec_fp_penalty": 0.75,
            },
            runtime_kwargs={
                **prec_runtime,
                "select_seed_prec_lambda": 0.0,
                "select_front_speed_lambda": 0.10,
                "select_fn_fp_lambda": 0.10,
            },
            env_overrides={},
        ),
        # Wall-gen gate FT: same prec_iter loss; physical FP gating only (no hard mask / seed_aux /
        # underpred bump). Punish high-speed/shear FPs; keep stagnant-pocket growth. F1-primary select.
        "WG_prec_physfp": MatGrowthLegSpec(
            code="WG_prec_physfp",
            label="Prec-iter + physical_fp_gating (distant-FP precision FT; F1-primary gate)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "frontier_hops": 0,
                "nucleation_topk": 0.0,
                "seed_aux_weight": 0.0,
                "physical_fp_gating": True,
            },
            runtime_kwargs={
                **prec_runtime,
                # Locked gate: primary F1, reject starvation/spray, FN must not rise vs floor~67.
                "select_clot_f1_weight": 0.75,
                "select_clot_score_weight": 0.15,
                "select_mat_f1_weight": 0.10,
                "select_mass_hard_min": 0.5,
                "select_mass_hard_max": 1.5,
                "select_mass_soft_lambda": 0.25,
                "select_mass_soft_target": 1.1,
                "select_fn_hard_max": 80.0,
                "select_seed_prec_lambda": 0.0,
                "select_front_speed_lambda": 0.0,
                "select_fn_fp_lambda": 0.0,
            },
            env_overrides={},
        ),
        # Alternate FT when FP geography is adjacent overpaint: deepen closed-loop exposure only.
        "WG_prec_cloop": MatGrowthLegSpec(
            code="WG_prec_cloop",
            label="Prec-iter closed-loop FT (cl_init 0.85, tbptt 12; no new loss; F1-primary gate)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "frontier_hops": 0,
                "nucleation_topk": 0.0,
                "seed_aux_weight": 0.0,
                "physical_fp_gating": False,
                "closed_loop_init": 0.85,
                "tbptt_tail": 12,
                "scheduled_sampling": False,
            },
            runtime_kwargs={
                **prec_runtime,
                "select_clot_f1_weight": 0.75,
                "select_clot_score_weight": 0.15,
                "select_mat_f1_weight": 0.10,
                "select_mass_hard_min": 0.5,
                "select_mass_hard_max": 1.5,
                "select_mass_soft_lambda": 0.25,
                "select_mass_soft_target": 1.1,
                "select_fn_hard_max": 80.0,
                "select_seed_prec_lambda": 0.0,
                "select_front_speed_lambda": 0.0,
                "select_fn_fp_lambda": 0.0,
            },
            env_overrides={},
        ),
        # Multi-pocket selection: soft-penalize Mat outside k-hop of GT first-seed.
        # Not hard frontier masking; growth inside the true pocket stays free. Park physfp.
        "WG_prec_pocket": MatGrowthLegSpec(
            code="WG_prec_pocket",
            label="Prec-iter + pocket-contrast (exclusive wrong-pocket soft loss; F1-primary gate)",
            no_init=False,
            init_ckpt=WG_PREC_ITER_CKPT,
            init_mode="full",
            config_kwargs={
                **prec_config,
                "frontier_hops": 0,
                "nucleation_topk": 0.0,
                "seed_aux_weight": 0.0,
                "physical_fp_gating": False,
                "pocket_contrast_weight": 0.35,
                "pocket_contrast_hops": 4,
                "pocket_contrast_early_steps": 8,
                "pocket_contrast_inside_weight": 0.05,
            },
            runtime_kwargs={
                **prec_runtime,
                "select_clot_f1_weight": 0.75,
                "select_clot_score_weight": 0.15,
                "select_mat_f1_weight": 0.10,
                "select_mass_hard_min": 0.5,
                "select_mass_hard_max": 1.5,
                "select_mass_soft_lambda": 0.25,
                "select_mass_soft_target": 1.1,
                "select_fn_hard_max": 80.0,
                "select_seed_prec_lambda": 0.0,
                "select_front_speed_lambda": 0.0,
                "select_fn_fp_lambda": 0.0,
            },
            env_overrides={},
        ),
        # Physics-GAT: Geom+Flux + Stage-A PM-GAT trunk (mesh normals/SDF).
        # Random init (no warm-start) — trunk keys never matched SAGE baseline anyway.
        # Soft prior_scale + identity edge_proj gate so wall mods do not wipe content
        # attention at init (unscaled mods caused ~2.8x mass spray).
        "WG_physgat_01": MatGrowthLegSpec(
            code="WG_physgat_01",
            label="Physics-GAT: Geom+Flux + physics_gat trunk (soft priors)",
            no_init=True,
            init_ckpt="",
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "arch": "physics_gat",
                "physics_gat_prior_scale": 0.05,
            },
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        # Fair random-init SAGE control (same feats as physgat; featfix_03 was warm-started).
        "WG_physgat_ctrl": MatGrowthLegSpec(
            code="WG_physgat_ctrl",
            label="Control: Geom+Flux SAGE random init (fair vs physgat)",
            no_init=True,
            init_ckpt="",
            init_mode="full",
            config_kwargs={
                **v3_config,
                "geom_feats": True,
                "geom_feats_rich": True,
                "flux_stag_feat": True,
                "arch": "sage",
            },
            runtime_kwargs={**v3_runtime},
            env_overrides={},
        ),
        # ---- Flow-source A/B (GT crutch vs RGP-DEQ / local-tiling deploy path) ----
        # Shared stack = WG_sweep_v3_01 (drop-xy, WC_v7 dynamics). Eval always forces
        # deploy-faithful coupling in eval_mat_growth_simple._apply_ckpt_recipe.
        "FS_ab_gt": MatGrowthLegSpec(
            code="FS_ab_gt",
            label="Flow A/B: GT train crutch (COMSOL flow feats + train vel)",
            no_init=True,
            init_ckpt="",
            init_mode="full",
            config_kwargs={**v3_config, "flow_feats_source": "gt"},
            runtime_kwargs={
                **v3_runtime,
                "train_vel_source": "gt",
                "rollout_vel_source": "gt",
                "corrector_coupling": False,
                "closed_loop_coupling": False,
                "train_deploy_eval_flow": "auto",
            },
            env_overrides={},
        ),
        "FS_ab_kine": MatGrowthLegSpec(
            code="FS_ab_kine",
            label="Flow A/B: clot-blind RGP-DEQ base (kine feats, no tiling in train)",
            no_init=True,
            init_ckpt="",
            init_mode="full",
            config_kwargs={**v3_config, "flow_feats_source": "kine"},
            runtime_kwargs={
                **v3_runtime,
                "train_vel_source": "kinematics",
                "rollout_vel_source": "kinematics",
                "corrector_coupling": False,
                "closed_loop_coupling": False,
                "train_deploy_eval_flow": "auto",
            },
            env_overrides={},
        ),
        "FS_ab_coupled": MatGrowthLegSpec(
            code="FS_ab_coupled",
            label="Flow A/B: deploy-faithful RGP-DEQ + local tiling (train=coupled)",
            no_init=True,
            init_ckpt="",
            init_mode="full",
            config_kwargs={**v3_config, "flow_feats_source": "auto"},
            runtime_kwargs={
                **v3_runtime,
                "train_vel_source": "coupled",
                "rollout_vel_source": "coupled",
                "corrector_coupling": True,
                "closed_loop_coupling": True,
                "train_deploy_eval_flow": "auto",
            },
            env_overrides={},
        ),
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
        # Hypothesis (docs/archive/SPECIES_LEARNING_STRATEGY.md s6.13): geometry+kine is near its
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
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
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
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
            },
        ),
        # Warm-start from locked WC_v7; short finetune legs for firewall sequence.
        "WC_v7_fw1_blind_sat": MatGrowthLegSpec(
            code="WC_v7_fw1_blind_sat",
            label="WC_v7 + midside-blind + hop1-smooth + sat_offwall=30 (firewall step 1 package)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
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
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "20.0",
                "SPECIES_GELATION_PHI_LOSS_TYPE": "mse",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.0",
                "SPECIES_CONTINUOUS_LOSS_SCALE": "0.1",
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
                "SPECIES_MIDSIDE_BLIND_LOSS": "1",
                "SPECIES_HOP1_SMOOTH": "1",
                "SPECIES_HOP1_SMOOTH_ALPHA": "0.4",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
            },
        ),
        "WC_v7_fw1_blind": MatGrowthLegSpec(
            code="WC_v7_fw1_blind",
            label="WC_v7 + midside-blind only (firewall step 1 ablation)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
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
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "20.0",
                "SPECIES_GELATION_PHI_LOSS_TYPE": "mse",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.0",
                "SPECIES_CONTINUOUS_LOSS_SCALE": "0.1",
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
                "SPECIES_MIDSIDE_BLIND_LOSS": "1",
            },
        ),
        "WC_v7_fw1_smooth": MatGrowthLegSpec(
            code="WC_v7_fw1_smooth",
            label="WC_v7 + hop1 label smooth only (firewall step 1 ablation)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
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
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "20.0",
                "SPECIES_GELATION_PHI_LOSS_TYPE": "mse",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.0",
                "SPECIES_CONTINUOUS_LOSS_SCALE": "0.1",
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
                "SPECIES_HOP1_SMOOTH": "1",
                "SPECIES_HOP1_SMOOTH_ALPHA": "0.4",
            },
        ),
        "WC_v7_fw1_sat30": MatGrowthLegSpec(
            code="WC_v7_fw1_sat30",
            label="WC_v7 + sat_offwall=30 only (firewall step 1 ablation)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
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
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "20.0",
                "SPECIES_GELATION_PHI_LOSS_TYPE": "mse",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.0",
                "SPECIES_CONTINUOUS_LOSS_SCALE": "0.1",
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
            },
        ),
        "WC_v7_fw3_isolate": MatGrowthLegSpec(
            code="WC_v7_fw3_isolate",
            label="WC_v7 + isolate offwall loss scale=3 (firewall step 3)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
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
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "20.0",
                "SPECIES_GELATION_PHI_LOSS_TYPE": "mse",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.0",
                "SPECIES_CONTINUOUS_LOSS_SCALE": "0.1",
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
                "SPECIES_MIDSIDE_BLIND_LOSS": "1",
                "SPECIES_HOP1_SMOOTH": "1",
                "SPECIES_HOP1_SMOOTH_ALPHA": "0.4",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_ISOLATE_OFFWALL_LOSS": "1",
                "SPECIES_OFFWALL_LOSS_SCALE": "3.0",
            },
        ),
        "WC_v7_fw3_skiphop": MatGrowthLegSpec(
            code="WC_v7_fw3_skiphop",
            label="WC_v7 + skiphop + blind/smooth/sat30 (firewall step 3 controlled)",
            no_init=False,
            init_ckpt="outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
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
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "20.0",
                "SPECIES_GELATION_PHI_LOSS_TYPE": "mse",
                "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "0.0",
                "SPECIES_CONTINUOUS_LOSS_SCALE": "0.1",
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
                "SPECIES_MIDSIDE_BLIND_LOSS": "1",
                "SPECIES_HOP1_SMOOTH": "1",
                "SPECIES_HOP1_SMOOTH_ALPHA": "0.4",
                "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "30.0",
                "SPECIES_SKIP_HOP_GNN": "1",
                "SPECIES_CONTINUOUS_FP_WEIGHT": "24",
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
                "BIOCHEM_KINE_RESOLVE_ON_CLOT": "0",
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
        "WG_sched_sample": MatGrowthLegSpec(
            code="WG_sched_sample",
            label="Wall-gen: scheduled sampling (noisy-GT anchoring ramps down)",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_FLOW_FEATS_SOURCE": "kine",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_SCHEDULED_SAMPLING": "1",
                "SPECIES_SS_TARGET_PROB": "0.5",
                "SPECIES_SS_WARMUP_EPOCHS": "3",
                "SPECIES_SS_ANCHOR_STRIDE": "10",
                "SPECIES_SS_NOISY": "1",
            },
        ),
        "WG_noise_boost": MatGrowthLegSpec(
            code="WG_noise_boost",
            label="Wall-gen: amplified per-step noise + teacher blur",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_FLOW_FEATS_SOURCE": "kine",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.10",
                "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.06",
                "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.40",
                "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.12",
            },
        ),
        "WG_long_tbptt": MatGrowthLegSpec(
            code="WG_long_tbptt",
            label="Wall-gen: longer TBPTT tail + higher max unroll",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_FLOW_FEATS_SOURCE": "kine",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_CONTINUOUS_TBPTT_TAIL": "15",
                "SPECIES_PUSHFORWARD_MAX_UNROLL": "120",
            },
        ),
        "WG_dynamics_all": MatGrowthLegSpec(
            code="WG_dynamics_all",
            label="Wall-gen: sched-sample + noise-boost + long-TBPTT combined",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_FLOW_FEATS_SOURCE": "kine",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_SCHEDULED_SAMPLING": "1",
                "SPECIES_SS_TARGET_PROB": "0.5",
                "SPECIES_SS_WARMUP_EPOCHS": "3",
                "SPECIES_SS_ANCHOR_STRIDE": "10",
                "SPECIES_SS_NOISY": "1",
                "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.10",
                "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.06",
                "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.40",
                "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.12",
                "SPECIES_CONTINUOUS_TBPTT_TAIL": "15",
                "SPECIES_PUSHFORWARD_MAX_UNROLL": "120",
            },
        ),
        "WG_mirror_y": MatGrowthLegSpec(
            code="WG_mirror_y",
            label="Wall-gen: y-axis mirror augmentation (exact N-S symmetry)",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_FLOW_FEATS_SOURCE": "kine",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_AUGMENT_MIRROR_Y": "1",
            },
        ),
        "WG_geom_rich": MatGrowthLegSpec(
            code="WG_geom_rich",
            label="Wall-gen: static 2-hop geometry discriminators",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_FLOW_FEATS_SOURCE": "kine",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_GEOM_FEATS_RICH": "1",
            },
        ),
        "WG_flux_stag": MatGrowthLegSpec(
            code="WG_flux_stag",
            label="Wall-gen: flux-stag nucleation prior channel",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_FLOW_FEATS_SOURCE": "kine",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_FLUX_STAG_FEAT": "1",
            },
        ),
        "WG_full_stack": MatGrowthLegSpec(
            code="WG_full_stack",
            label="Wall-gen: dynamics + conditioning + mirror full stack",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides={
                "SPECIES_FLOW_FEATS": "1",
                "SPECIES_FLOW_FEATS_DROP_XY": "1",
                "SPECIES_FLOW_FEATS_DYNAMIC": "1",
                "SPECIES_FLOW_FEATS_SOURCE": "kine",
                "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
                "SPECIES_CLOSED_LOOP_COUPLING": "1",
                "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1",
                "CLOT_PHI_CEILING_HOPS": "4",
                "SPECIES_SNAPSHOT_WALL_HOPS": "3",
                "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
                "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
                "SPECIES_SCHEDULED_SAMPLING": "1",
                "SPECIES_SS_TARGET_PROB": "0.5",
                "SPECIES_SS_WARMUP_EPOCHS": "3",
                "SPECIES_SS_ANCHOR_STRIDE": "10",
                "SPECIES_SS_NOISY": "1",
                "SPECIES_PUSHFORWARD_INPUT_NOISE": "0.10",
                "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.06",
                "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.40",
                "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.12",
                "SPECIES_CONTINUOUS_TBPTT_TAIL": "15",
                "SPECIES_PUSHFORWARD_MAX_UNROLL": "120",
                "SPECIES_AUGMENT_MIRROR_Y": "1",
                "SPECIES_GEOM_FEATS_RICH": "1",
                "SPECIES_FLUX_STAG_FEAT": "1",
            },
        ),
    }

    # Phase 1 Sweep Grid (Ablation on Safe Baseline)
    base_env = {
        "SPECIES_FLOW_FEATS_DROP_XY": "1",
        "SPECIES_FLOW_FEATS_SOURCE": "auto",
        "SPECIES_CLOSED_LOOP_COUPLING": "1",
        "BIOCHEM_CORRECTOR_COUPLING": "1",
        "SPECIES_SCHEDULED_SAMPLING": "0",
    }
    
    leg_configs = [
        # Leg 1: Pure standard discrete GNN
        (1, "Safe Baseline", {}),
        # Leg 2: Geom Feats
        (2, "Geom Feats", {"SPECIES_GEOM_FEATS_RICH": "1"}),
        # Leg 3: Flux Feat
        (3, "Flux Feat", {"SPECIES_FLUX_STAG_FEAT": "1"}),
        # Leg 4: Just Mat
        (4, "Just Mat", {"BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat"}),
        # Leg 5: Wall Loss
        (5, "Wall Loss", {"CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1"}),
        # Leg 6: Continuous
        (6, "Continuous", {"SPECIES_CONTINUOUS_PHYSICS_READOUT": "1"}),
        # Leg 7: Dual Head
        (7, "Dual Head", {"SPECIES_CONTINUOUS_PHYSICS_READOUT": "1", "SPECIES_CONTINUOUS_DUAL_HEAD": "1"}),
        # Leg 8: Teacher Noise
        (8, "Teacher Noise", {"SPECIES_CONTINUOUS_PHYSICS_READOUT": "1", "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.1"}),
        # Leg 9: mat_growth_simple combo (but with auto flow)
        (9, "mat_growth_simple combo", {
            "SPECIES_CONTINUOUS_PHYSICS_READOUT": "1",
            "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
            "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "mat",
            "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1"
        }),
        # Leg 10: Geom + Flux Combo
        (10, "Geom + Flux Combo", {"SPECIES_GEOM_FEATS_RICH": "1", "SPECIES_FLUX_STAG_FEAT": "1"})
    ]

    for idx, label, overrides in leg_configs:
        leg_name = f"WG_sweep_{idx:02d}"
        env_overrides = base_env.copy()
        env_overrides.update(overrides)
        
        specs[leg_name] = MatGrowthLegSpec(
            code=leg_name,
            label=f"{leg_name}: {label}",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides=env_overrides,
        )


    val_baseline_configs = [
        ("01_wcv7_fresh", "Raw WC_v7_clot_phi_mse cold", {}),
        ("02_wcv7_dropxy", "WC_v7_clot_phi_mse + drop-xy", {"SPECIES_FLOW_FEATS_DROP_XY": "1"}),
        ("03_wcv7_dropxy_kine", "WC_v7_clot_phi_mse + drop-xy + kine flow", {"SPECIES_FLOW_FEATS_DROP_XY": "1", "SPECIES_FLOW_FEATS_SOURCE": "kine"}),
        ("04_wcv7_dropxy_kine_wallonly", "WC_v7_clot_phi_mse + drop-xy + kine flow + wall mat only", {"SPECIES_FLOW_FEATS_DROP_XY": "1", "SPECIES_FLOW_FEATS_SOURCE": "kine", "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "1"}),
    ]

    for suffix, label, overrides in val_baseline_configs:
        leg_name = f"VAL_baseline_{suffix}"
        env_overrides = specs["WC_v7_clot_phi_mse"].env_overrides.copy()
        env_overrides.update(overrides)
        specs[leg_name] = MatGrowthLegSpec(
            code=leg_name,
            label=f"{leg_name}: {label}",
            no_init=True,
            init_ckpt=init_default,
            init_mode="full",
            env_overrides=env_overrides,
        )

    if code not in specs:
        raise ValueError(f"unknown mat growth leg {leg!r}; use {list(specs)}")
    return materialize_leg_spec(specs[code])


def leg_out_ckpt(leg: str, *, ladder: bool = True) -> str:
    if ladder:
        return f"{LADDER_ROOT}/{leg}/species/best.pth"
    return DEFAULT_CKPT


def apply_mat_growth_simple_recipe_env(
    *,
    overrides: dict[str, str] | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Bind typed mat-growth recipe; write only residual unknown env keys."""
    from dataclasses import replace as _replace

    from src.architecture.pushforward_config import (
        PushforwardConfig,
        split_legacy_env_overrides,
    )
    from src.architecture.runtime_config import (
        BiochemRuntimeConfig,
        split_legacy_runtime_env,
    )
    from src.biochem_gnn.config import _IO_ENV_KEYS, _bind_typed_configs

    merged = dict(GLOBAL_TRAIN_RECIPE)
    merged.update(MAT_GROWTH_SIMPLE_RECIPE)
    if overrides:
        merged.update({k: str(v) for k, v in overrides.items()})

    pf_kw, rem = split_legacy_env_overrides(merged)
    rt_kw, rem2 = split_legacy_runtime_env(rem)
    pf = _replace(PushforwardConfig(), **pf_kw) if pf_kw else PushforwardConfig()
    rt = BiochemRuntimeConfig.from_kwargs(rt_kw) if rt_kw else BiochemRuntimeConfig()
    _bind_typed_configs(pf, rt)

    # Residual unknown + process/IO keys only (never architecture/runtime control plane).
    for key, val in rem2.items():
        if key in _IO_ENV_KEYS or key not in (
            set(GLOBAL_TRAIN_RECIPE) | set(MAT_GROWTH_SIMPLE_RECIPE)
        ):
            existing = os.environ.get(key)
            if force or not str(existing or "").strip():
                os.environ[key] = str(val)
    for key in _IO_ENV_KEYS:
        if key in merged and (force or not str(os.environ.get(key, "")).strip()):
            os.environ[key] = str(merged[key])
    return merged


def mat_growth_precision_selection_enabled() -> bool:
    try:
        from src.architecture.runtime_config import get_active_runtime

        rt = get_active_runtime()
        if rt is not None:
            return bool(rt.scoring.precision_select)
    except Exception:
        pass
    raw = (os.environ.get("SPECIES_MAT_GROWTH_PRECISION_SELECT") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")



def materialize_leg_spec(spec: MatGrowthLegSpec) -> MatGrowthLegSpec:
    """Split legacy env_overrides into typed config_kwargs + runtime_kwargs."""
    from src.architecture.pushforward_config import split_legacy_env_overrides, validate_config_kwargs
    from src.architecture.runtime_config import split_legacy_runtime_env, validate_runtime_kwargs

    auto_cfg, rem1 = split_legacy_env_overrides(spec.env_overrides)
    auto_rt, rem2 = split_legacy_runtime_env(rem1)
    merged_cfg = {**auto_cfg, **dict(spec.config_kwargs)}
    merged_rt = {**auto_rt, **dict(spec.runtime_kwargs)}
    validate_config_kwargs(merged_cfg)
    validate_runtime_kwargs(merged_rt)
    if (
        merged_cfg == spec.config_kwargs
        and merged_rt == spec.runtime_kwargs
        and rem2 == spec.env_overrides
    ):
        return spec
    return MatGrowthLegSpec(
        code=spec.code,
        label=spec.label,
        no_init=spec.no_init,
        init_ckpt=spec.init_ckpt,
        init_mode=spec.init_mode,
        config_kwargs=merged_cfg,
        runtime_kwargs=merged_rt,
        env_overrides=rem2,
    )


def get_mat_growth_config_kwargs(leg: str) -> dict[str, Any]:
    """Typed PushforwardConfig overrides for ``leg`` (does not touch os.environ)."""
    return dict(mat_growth_leg_spec(leg).config_kwargs)


def get_mat_growth_runtime_kwargs(leg: str) -> dict[str, Any]:
    """Typed BiochemRuntimeConfig flat overrides for ``leg`` (does not touch os.environ)."""
    return dict(mat_growth_leg_spec(leg).runtime_kwargs)


def get_mat_growth_runtime_env(leg: str) -> dict[str, str]:
    """Deprecated residual unknown env knobs only. Prefer get_mat_growth_runtime_kwargs."""
    return dict(mat_growth_leg_spec(leg).env_overrides)


def apply_mat_growth_leg_env(leg: str, *, force: bool = True) -> dict[str, str]:
    """Bind recipe + leg typed configs; write residual unknown env only.

    Architecture -> ``get_mat_growth_config_kwargs`` / PushforwardConfig.
    Runtime policy -> ``get_mat_growth_runtime_kwargs`` / BiochemRuntimeConfig.
    Do not add new architecture or runtime knobs to env_overrides.
    """
    from src.architecture.pushforward_config import get_active_config
    from src.architecture.runtime_config import get_active_runtime
    from src.biochem_gnn.config import _bind_typed_configs

    spec = mat_growth_leg_spec(leg)
    merged = apply_mat_growth_simple_recipe_env(force=force)
    pf = get_active_config()
    rt = get_active_runtime()
    if pf is None or rt is None:
        raise RuntimeError("mat-growth recipe failed to bind typed configs")
    if spec.config_kwargs:
        pf = pf.with_overrides(**spec.config_kwargs)
    if spec.runtime_kwargs:
        rt = rt.with_overrides(**spec.runtime_kwargs)
    _bind_typed_configs(pf, rt)
    for key, val in spec.env_overrides.items():
        if force or not str(os.environ.get(key, "")).strip():
            os.environ[key] = str(val)
        merged[key] = str(val)
    return merged


def recipe_fingerprint() -> dict[str, Any]:
    """Serializable knob set for baseline JSON / train meta."""
    from src.architecture.pushforward_config import get_active_config
    from src.architecture.runtime_config import get_active_runtime

    keys = sorted(set(GLOBAL_TRAIN_RECIPE) | set(MAT_GROWTH_SIMPLE_RECIPE))
    pf = get_active_config()
    rt = get_active_runtime()
    out: dict[str, Any] = {}
    if pf is not None:
        out["config_kwargs"] = pf.to_meta_dict()
    if rt is not None:
        out["runtime_kwargs"] = rt.to_flat_dict()
    # Legacy flat env view for older compare tools.
    for k in keys:
        out[k] = os.environ.get(k, GLOBAL_TRAIN_RECIPE.get(k, MAT_GROWTH_SIMPLE_RECIPE.get(k, "")))
    return out


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

# =====================================================================================
# COHORT SPLIT v2 (WALL_MODEL_PLAN.md 21). Supersedes the ad-hoc 5-vessel sub-cohort split.
#
# Built from measured per-vessel descriptors, not by hand: positive rate, best-feature AUC,
# nucleation fraction, and width-profile skew (which cleanly recovers the geometry class --
# aneurysms 039/040/043 skew +1.05/+0.69/+0.65, stenoses 041/042/044 skew -0.58/-0.57/-0.51).
#
# Holdout construction rules, applied programmatically:
#   * every holdout vessel is INTERIOR on all four descriptor axes (never a global min/max),
#     so the sealed set tests interpolation rather than extrapolation;
#   * T >= 150, so no severely truncated simulation is in the holdout;
#   * class balance ~proportional to the pool (2 aneurysm / 6 stenosis vs the pool's 7/28);
#   * patient042 (median stenosis) and patient043 (the long-sealed aneurysm) are the two
#     dev-set holdouts, keeping continuity with every number in sections 9-20.
#
# Coverage: train spans the holdout's range on pos%, nucleation% and skew. The single
# exception is the AUC upper bound -- holdout 0.944 (043) vs train 0.941 (040), a 0.003
# tie -- which is immaterial.
#
# patient002 is excluded on the project's existing data-quality call (see the
# 'no 023/002 junk' note above); patient023/026/027/030/033/034/017/022 have no clot.
# Truncated sims (T<150) ARE kept in train: training consumes WINDOWS, whose per-step
# GT deltas are valid regardless of run length. T>=150 is a HOLDOUT rule, because there
# the vessel's final map is the target and 14.6 showed nucleation runs into the last
# quartile -- a T=29 'final' state is simply a different quantity.
#
# 039-044 remains the DEV cohort; 039/040/041/044 are dev-train, 042/043 are dev-holdout.
WALL_COHORT_V2_TRAIN: tuple[str, ...] = (
    "patient003", "patient004", "patient005",
    "patient006", "patient008", "patient009", "patient011",
    "patient012", "patient015", "patient016", "patient018",
    "patient019", "patient020", "patient021", "patient024",
    "patient025", "patient028", "patient029", "patient032",
    "patient035", "patient036", "patient037", "patient039",
    "patient040", "patient041", "patient044",
)

# SEALED. Do not train on these, and do not tune against them. They exist to be spent once.
WALL_COHORT_V2_GENERALIZATION: tuple[str, ...] = (
    "patient001", "patient007", "patient010", "patient013",
    "patient014", "patient031", "patient042", "patient043",
)

WALL_COHORT_V2_DEV: tuple[str, ...] = (
    "patient039", "patient040", "patient041",
    "patient042", "patient043", "patient044",
)
WALL_COHORT_V2_DEV_HOLDOUT: tuple[str, ...] = ("patient042", "patient043")
