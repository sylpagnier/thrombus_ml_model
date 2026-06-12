"""Time-varying rule-based clot phi: progressive commit, incubation, neighbor growth."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_forecast import build_clot_forecast_pair_step, iter_forecast_pairs
from src.core_physics.clot_growth_masks import graph_dilate_hops, resolve_ceiling_mask
from src.core_physics.clot_phi_simple import (
    ClotPriorRuleConfig,
    _anchor_flow_props,
    _hop_distance_from_seed,
    _top_frac_mask,
    _wall_mask_from_data,
    clot_prior_score_flat,
    log_blend_mu_eff_si,
    predict_phi_prior_rule,
    prior_rule_config_from_env,
    project_deploy_mu_with_support,
    sdf_nd_from_data,
)
from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields
from src.evaluation.clot_shape_score import compute_clot_shape_metrics
from src.core_physics.clot_localized_spatial import (
    LocalizedSpatialConfig,
    blend_species_into_risk,
    build_eligible_pool,
    build_localized_static_support,
    normalize_risk_per_wall_half,
    resolve_species_time_index,
    segment_topk_mask,
)
from src.training.train_clot_phi_simple import _clot_metrics

_temporal_pred_uv: tuple[torch.Tensor, torch.Tensor] | None = None
_temporal_pred_uv_key: int | None = None
_temporal_kine_model = None


def temporal_vel_source() -> str:
    """``gt`` = COMSOL [u,v] on anchor; ``kinematics`` = steady GINO-DEQ on mesh."""
    raw = (
        os.environ.get("CLOT_TEMPORAL_VEL_SOURCE")
        or os.environ.get("CLOT_PHI_VEL_SOURCE")
        or "gt"
    ).strip().lower()
    if raw in ("kin", "kinematics", "deq", "gino", "pred"):
        return "kinematics"
    if raw in ("coupled", "mu_coupled", "feedback", "5b"):
        return "coupled"
    return "gt"


def reset_temporal_kinematics_cache() -> None:
    """Clear cached steady GINO-DEQ uv (tests / multi-anchor sweeps)."""
    global _temporal_pred_uv, _temporal_pred_uv_key, _temporal_kine_model
    _temporal_pred_uv = None
    _temporal_pred_uv_key = None
    _temporal_kine_model = None


def _temporal_graph_cache_key(data) -> tuple[int, int, int]:
    """Stable cache key per graph (``id(data)`` alone can collide after GC)."""
    n = int(data.num_nodes)
    e = int(data.edge_index.shape[1])
    ptr = 0
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.numel() > 0:
        ptr = int(data.x.untyped_storage().data_ptr())
    return (n, e, ptr)


def _resolve_uv_for_temporal_risk(
    data,
    t_in: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow [u,v] ND for shear-risk features (steady pred when deploy mode)."""
    ti = max(0, min(int(t_in), int(data.y.shape[0]) - 1))
    y = data.y[ti].to(device=device, dtype=torch.float32)
    u_gt = y[:, 0]
    v_gt = y[:, 1]
    if temporal_vel_source() == "gt":
        return u_gt, v_gt

    if temporal_vel_source() == "coupled":
        from src.core_physics.clot_coupled_rollout import get_coupled_uv

        coupled = get_coupled_uv(data, device)
        if coupled is not None:
            return coupled

    global _temporal_pred_uv, _temporal_pred_uv_key, _temporal_kine_model
    key = _temporal_graph_cache_key(data)
    if _temporal_pred_uv is None or _temporal_pred_uv_key != key:
        from src.core_physics.clot_phi_rollout import clot_phi_kine_teacher_forcing
        from src.utils.kinematics_inference import (
            load_kinematics_predictor,
            predict_kinematics,
            resolve_kinematics_checkpoint,
        )

        ckpt = (os.environ.get("CLOT_PHI_KINE_CKPT") or "").strip()
        if not ckpt:
            ckpt = str(resolve_kinematics_checkpoint())
        if _temporal_kine_model is None:
            _temporal_kine_model = load_kinematics_predictor(
                ckpt,
                device,
                phys_cfg=PhysicsConfig(phase="kinematics"),
            )
        batch = data.to(device)
        with torch.no_grad():
            pred = predict_kinematics(_temporal_kine_model, batch)
        u_p = pred[:, 0]
        v_p = pred[:, 1]
        tf = clot_phi_kine_teacher_forcing()
        if tf >= 1.0:
            u_p, v_p = u_gt, v_gt
        elif tf > 0.0:
            u_p = (1.0 - tf) * u_p + tf * u_gt
            v_p = (1.0 - tf) * v_p + tf * v_gt
        _temporal_pred_uv = (u_p, v_p)
        _temporal_pred_uv_key = key
    return _temporal_pred_uv


@dataclass(frozen=True)
class TemporalGrowthRuleConfig:
    """Composable temporal commit policy on top of spatial risk."""

    name: str
    kind: str
    spatial_rule: ClotPriorRuleConfig | None = None
    localized: LocalizedSpatialConfig | None = None
    risk_flow_time: int = 0
    start_frac: float = 0.05
    end_frac: float = 0.22
    power: float = 1.5
    onset_spread: float = 0.55
    min_onset_frac: float = 0.05
    seed_frac: float = 0.08
    hop_per_step: int = 1
    risk_floor_quantile: float = 0.45
    neighbor_risk_q: float = 0.40
    global_onset_frac: float = 0.0
    promotion_boost: float = 1.0
    accum_gain: float = 0.25
    accum_threshold: float = 1.2
    accum_split_wall: float = 0.80
    accum_split_lumen: float = 0.03

    def describe(self) -> str:
        parts = [self.kind]
        if self.kind == "progressive_topk":
            hi = min(0.95, float(self.end_frac) * float(self.promotion_boost))
            parts.append(f"{self.start_frac:.2f}->{hi:.2f}^{self.power:.1f}")
        elif self.kind == "threshold_accum":
            parts.append(
                f"g={self.accum_gain:.2f}_Y={self.accum_threshold:.2f}"
                f"_sw={self.accum_split_wall:.2f}_sl={self.accum_split_lumen:.2f}"
            )
        elif self.kind == "ranked_onset":
            parts.append(f"spread={self.onset_spread:.2f}")
        elif self.kind == "hop_growth":
            parts.append(f"seed={self.seed_frac:.2f}")
        elif self.kind == "neighbor_ac":
            parts.append(f"seed={self.seed_frac:.2f}_nb={self.neighbor_risk_q:.2f}")
        elif self.kind == "static_spatial":
            parts.append("instant")
        if self.localized is not None:
            parts.append(f"loc={self.localized.describe()}")
        if self.global_onset_frac > 0:
            parts.append(f"offset={self.global_onset_frac:.2f}")
        if float(self.promotion_boost) > 1.0 + 1e-6:
            parts.append(f"boost={self.promotion_boost:.2f}")
        return "|".join(parts)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def temporal_rule_config_from_env() -> TemporalGrowthRuleConfig:
    """Load temporal rule from CLOT_TEMPORAL_* env (after spatial prior env)."""
    kind = os.environ.get("CLOT_TEMPORAL_RULE_KIND", "progressive_topk").strip()
    name = os.environ.get("CLOT_TEMPORAL_RULE_NAME", kind).strip() or kind
    loc = localized_config_from_env()
    return TemporalGrowthRuleConfig(
        name=name,
        kind=kind,
        spatial_rule=prior_rule_config_from_env(),
        localized=loc,
        start_frac=_env_float("CLOT_TEMPORAL_START_FRAC", 0.05),
        end_frac=_env_float("CLOT_TEMPORAL_END_FRAC", 0.22),
        power=_env_float("CLOT_TEMPORAL_POWER", 1.5),
        seed_frac=_env_float("CLOT_TEMPORAL_SEED_FRAC", 0.08),
        onset_spread=_env_float("CLOT_TEMPORAL_ONSET_SPREAD", 0.55),
        min_onset_frac=_env_float("CLOT_TEMPORAL_MIN_ONSET", 0.08),
        global_onset_frac=_env_float("CLOT_TEMPORAL_GLOBAL_ONSET", 0.0),
        promotion_boost=_env_float("CLOT_TEMPORAL_PROMOTION_BOOST", 1.0),
        accum_gain=_env_float("CLOT_TEMPORAL_ACCUM_GAIN", 0.25),
        accum_threshold=_env_float("CLOT_TEMPORAL_ACCUM_THRESHOLD", 1.2),
        accum_split_wall=_env_float("CLOT_TEMPORAL_ACCUM_SPLIT_WALL", 0.80),
        accum_split_lumen=_env_float("CLOT_TEMPORAL_ACCUM_SPLIT_LUMEN", 0.03),
        neighbor_risk_q=_env_float("CLOT_TEMPORAL_NEIGHBOR_RISK_Q", 0.38),
        risk_floor_quantile=_env_float("CLOT_TEMPORAL_RISK_FLOOR_Q", 0.42),
    )


def localized_config_from_env() -> LocalizedSpatialConfig | None:
    mode = os.environ.get("CLOT_LOCALIZED_MODE", "").strip()
    if not mode:
        return None
    sp_q = _env_float("CLOT_LOCALIZED_SPECIES_GT_Q", 0.0)
    sp_w = _env_float("CLOT_LOCALIZED_SPECIES_WEIGHT", 0.0)
    return LocalizedSpatialConfig(
        mode=mode,
        segment_top_frac=_env_float("CLOT_LOCALIZED_TOP_FRAC", 0.25),
        skip_wall_arc_frac=_env_float("CLOT_LOCALIZED_SKIP_ARC", 0.15),
        n_arc_bins=int(_env_float("CLOT_LOCALIZED_ARC_BINS", 4)),
        recess_gate=os.environ.get("CLOT_LOCALIZED_RECESS", "0").strip() in ("1", "true", "yes"),
        recess_width_d2_q=_env_float("CLOT_LOCALIZED_RECESS_D2_Q", 0.55),
        recess_y_within_half_q=_env_float("CLOT_LOCALIZED_RECESS_Y_Q", 0.0),
        species_gt_top_q=sp_q,
        species_gt_time=os.environ.get("CLOT_LOCALIZED_SPECIES_TIME", "t_out").strip(),
        species_risk_weight=sp_w,
        normalize_risk_per_half=os.environ.get("CLOT_LOCALIZED_PER_HALF_NORM", "1").strip()
        not in ("0", "false", "no"),
        neg_dx_risk_weight=_env_float(
            "CLOT_LOCALIZED_NEG_DX_WEIGHT",
            _env_float("CLOT_SHEAR_W_NEG_DX", 0.45),
        ),
        sep_stream_risk_weight=_env_float("CLOT_SHEAR_W_SEP", 0.0),
        stasis_risk_weight=_env_float("CLOT_SHEAR_W_STASIS", 0.0),
        low_grad_risk_weight=_env_float("CLOT_SHEAR_W_LGRAD", 0.0),
        low_shear_thresh_si=_env_float("CLOT_SHEAR_LSS_SI", 0.0),
        low_grad_thresh_si=_env_float("CLOT_SHEAR_LGRAD_THRESH", 10.0),
        aneurysm_size_mode=os.environ.get("CLOT_SHEAR_SIZE_MODE", "").strip(),
    )


def default_temporal_rule_grid() -> list[TemporalGrowthRuleConfig]:
    spatial = prior_rule_config_from_env()
    return [
        TemporalGrowthRuleConfig(name="static_spatial", kind="static_spatial", spatial_rule=spatial),
        TemporalGrowthRuleConfig(
            name="progressive_mild",
            kind="progressive_topk",
            spatial_rule=spatial,
            start_frac=0.04,
            end_frac=0.18,
            power=1.2,
        ),
        TemporalGrowthRuleConfig(
            name="progressive_std",
            kind="progressive_topk",
            spatial_rule=spatial,
            start_frac=0.05,
            end_frac=0.22,
            power=1.5,
        ),
        TemporalGrowthRuleConfig(
            name="progressive_late",
            kind="progressive_topk",
            spatial_rule=spatial,
            start_frac=0.02,
            end_frac=0.25,
            power=2.0,
        ),
        TemporalGrowthRuleConfig(
            name="ranked_onset_std",
            kind="ranked_onset",
            spatial_rule=spatial,
            onset_spread=0.55,
            min_onset_frac=0.08,
        ),
        TemporalGrowthRuleConfig(
            name="ranked_onset_wide",
            kind="ranked_onset",
            spatial_rule=spatial,
            onset_spread=0.70,
            min_onset_frac=0.05,
        ),
        TemporalGrowthRuleConfig(
            name="hop_growth_std",
            kind="hop_growth",
            spatial_rule=spatial,
            seed_frac=0.08,
            hop_per_step=1,
            risk_floor_quantile=0.42,
        ),
        TemporalGrowthRuleConfig(
            name="hop_growth_slow",
            kind="hop_growth",
            spatial_rule=spatial,
            seed_frac=0.05,
            hop_per_step=1,
            risk_floor_quantile=0.50,
        ),
        TemporalGrowthRuleConfig(
            name="neighbor_ac_std",
            kind="neighbor_ac",
            spatial_rule=spatial,
            seed_frac=0.06,
            neighbor_risk_q=0.38,
        ),
        TemporalGrowthRuleConfig(
            name="neighbor_ac_aggressive",
            kind="neighbor_ac",
            spatial_rule=spatial,
            seed_frac=0.08,
            neighbor_risk_q=0.30,
        ),
        TemporalGrowthRuleConfig(
            name="prog_plus_gonset",
            kind="progressive_topk",
            spatial_rule=spatial,
            start_frac=0.05,
            end_frac=0.20,
            power=1.4,
            global_onset_frac=0.12,
        ),
        TemporalGrowthRuleConfig(
            name="ranked_plus_gonset",
            kind="ranked_onset",
            spatial_rule=spatial,
            onset_spread=0.50,
            global_onset_frac=0.10,
        ),
    ]


def default_localized_rule_grid() -> list[TemporalGrowthRuleConfig]:
    """Localized segment sweep: wall-half / arc bins, recess, skip arc, species oracle."""
    spatial = prior_rule_config_from_env()
    skip15 = LocalizedSpatialConfig(skip_wall_arc_frac=0.15)
    loc_base = dict(spatial_rule=spatial, onset_spread=0.55, min_onset_frac=0.08)

    def _loc(name: str, kind: str, loc: LocalizedSpatialConfig, **kw) -> TemporalGrowthRuleConfig:
        return TemporalGrowthRuleConfig(name=name, kind=kind, localized=loc, **loc_base, **kw)

    grid: list[TemporalGrowthRuleConfig] = [
        TemporalGrowthRuleConfig(
            name="ranked_onset_std",
            kind="ranked_onset",
            spatial_rule=spatial,
            onset_spread=0.55,
            min_onset_frac=0.08,
        ),
        _loc(
            "loc_half_top25_skip15",
            "ranked_onset",
            LocalizedSpatialConfig(mode="wall_half", segment_top_frac=0.25, skip_wall_arc_frac=0.15),
        ),
        _loc(
            "loc_half_top30_skip15",
            "ranked_onset",
            LocalizedSpatialConfig(mode="wall_half", segment_top_frac=0.30, skip_wall_arc_frac=0.15),
        ),
        _loc(
            "loc_half_top20_skip15",
            "ranked_onset",
            LocalizedSpatialConfig(mode="wall_half", segment_top_frac=0.20, skip_wall_arc_frac=0.15),
        ),
        _loc(
            "loc_arc4_top25_skip15",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="arc_bins", segment_top_frac=0.25, skip_wall_arc_frac=0.15, n_arc_bins=4
            ),
        ),
        _loc(
            "loc_arc6_top25_skip15",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="arc_bins", segment_top_frac=0.25, skip_wall_arc_frac=0.15, n_arc_bins=6
            ),
        ),
        _loc(
            "loc_half_top25_recess_skip15",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="wall_half",
                segment_top_frac=0.25,
                skip_wall_arc_frac=0.15,
                recess_gate=True,
                recess_width_d2_q=0.55,
                recess_y_within_half_q=0.45,
            ),
        ),
        _loc(
            "loc_half_top30_recess_skip15",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="wall_half",
                segment_top_frac=0.30,
                skip_wall_arc_frac=0.15,
                recess_gate=True,
                recess_width_d2_q=0.50,
                recess_y_within_half_q=0.40,
            ),
        ),
        _loc(
            "loc_prog_half_top25_skip15",
            "progressive_topk",
            LocalizedSpatialConfig(mode="wall_half", segment_top_frac=0.25, skip_wall_arc_frac=0.15),
            start_frac=0.05,
            end_frac=0.22,
            power=1.5,
        ),
        _loc(
            "loc_rank_half25_sp_gt60",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="wall_half",
                segment_top_frac=0.25,
                skip_wall_arc_frac=0.15,
                species_gt_top_q=0.60,
                species_gt_time="t_out",
            ),
        ),
        _loc(
            "loc_rank_half25_sp_gt70",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="wall_half",
                segment_top_frac=0.25,
                skip_wall_arc_frac=0.15,
                species_gt_top_q=0.70,
                species_gt_time="t_out",
            ),
        ),
        _loc(
            "loc_rank_recess_sp_gt60",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="wall_half",
                segment_top_frac=0.25,
                skip_wall_arc_frac=0.15,
                recess_gate=True,
                recess_width_d2_q=0.55,
                recess_y_within_half_q=0.45,
                species_gt_top_q=0.60,
                species_gt_time="t_out",
            ),
        ),
        _loc(
            "loc_half_top25_sp_t0_w25",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="wall_half",
                segment_top_frac=0.25,
                skip_wall_arc_frac=0.15,
                species_risk_weight=0.25,
                species_gt_time="t0",
            ),
        ),
        _loc(
            "loc_half_top25_sp_t0_w40",
            "ranked_onset",
            LocalizedSpatialConfig(
                mode="wall_half",
                segment_top_frac=0.25,
                skip_wall_arc_frac=0.15,
                species_risk_weight=0.40,
                species_gt_time="t0",
            ),
        ),
    ]
    return grid


def comprehensive_rule_architecture_grid(*, include_oracle: bool = False) -> list[TemporalGrowthRuleConfig]:
    """Curated grid: global baselines + localized segment/temporal combos."""
    spatial = prior_rule_config_from_env()
    base_onset = dict(spatial_rule=spatial, onset_spread=0.55, min_onset_frac=0.08)
    base_prog = dict(spatial_rule=spatial, start_frac=0.05, end_frac=0.22, power=1.5)

    grid: list[TemporalGrowthRuleConfig] = [
        TemporalGrowthRuleConfig(name="static_global", kind="static_spatial", spatial_rule=spatial),
        TemporalGrowthRuleConfig(name="ranked_onset_global", kind="ranked_onset", **base_onset),
        TemporalGrowthRuleConfig(
            name="ranked_onset_global_wide",
            kind="ranked_onset",
            spatial_rule=spatial,
            onset_spread=0.70,
            min_onset_frac=0.05,
        ),
        TemporalGrowthRuleConfig(name="prog_global_std", kind="progressive_topk", **base_prog),
        TemporalGrowthRuleConfig(
            name="prog_global_late",
            kind="progressive_topk",
            spatial_rule=spatial,
            start_frac=0.02,
            end_frac=0.25,
            power=2.0,
        ),
        TemporalGrowthRuleConfig(
            name="hop_growth_global",
            kind="hop_growth",
            spatial_rule=spatial,
            seed_frac=0.08,
            hop_per_step=1,
            risk_floor_quantile=0.42,
        ),
        TemporalGrowthRuleConfig(
            name="neighbor_ac_global",
            kind="neighbor_ac",
            spatial_rule=spatial,
            seed_frac=0.06,
            neighbor_risk_q=0.38,
        ),
    ]

    def _loc(
        name: str,
        kind: str,
        loc: LocalizedSpatialConfig,
        **kw,
    ) -> TemporalGrowthRuleConfig:
        d = dict(base_onset if kind == "ranked_onset" else base_prog)
        d.update(kw)
        return TemporalGrowthRuleConfig(name=name, kind=kind, localized=loc, **d)

    for skip in (0.0, 0.15, 0.20):
        for top in (0.20, 0.25, 0.30):
            for walls in (("lower", "upper"), ("lower",)):
                for ndx_w in (0.25, 0.45):
                    wtag = "both" if len(walls) > 1 else "lower"
                    st = int(skip * 100)
                    loc = LocalizedSpatialConfig(
                        mode="wall_half",
                        segment_top_frac=float(top),
                        skip_wall_arc_frac=float(skip),
                        neg_dx_risk_weight=float(ndx_w),
                        wall_halves=walls,
                        normalize_risk_per_half=True,
                    )
                    for kind in ("ranked_onset", "progressive_topk"):
                        ktag = "rank" if kind == "ranked_onset" else "prog"
                        name = f"loc_{ktag}_{wtag}_t{int(top * 100)}_s{st}_ndx{int(ndx_w * 100)}"
                        grid.append(_loc(name, kind, loc))

    for top in (0.25, 0.30):
        loc = LocalizedSpatialConfig(
            mode="arc_bins",
            segment_top_frac=float(top),
            skip_wall_arc_frac=0.15,
            n_arc_bins=4,
            neg_dx_risk_weight=0.45,
            wall_halves=("lower", "upper"),
        )
        grid.append(_loc(f"loc_rank_arc4_t{int(top * 100)}", "ranked_onset", loc))
        grid.append(_loc(f"loc_prog_arc4_t{int(top * 100)}", "progressive_topk", loc))

    if include_oracle:
        loc = LocalizedSpatialConfig(
            mode="wall_half",
            segment_top_frac=0.25,
            skip_wall_arc_frac=0.15,
            species_gt_top_q=0.65,
            species_gt_time="t_out",
        )
        grid.append(_loc("oracle_rank_sp_gt65", "ranked_onset", loc))

    return grid


def curated_p007_rule_grid(*, include_hybrid_species: bool = True) -> list[TemporalGrowthRuleConfig]:
    """~35 high-signal rules for fast sweeps (p007-focused selection)."""
    want = {
        "static_global",
        "ranked_onset_global",
        "ranked_onset_global_wide",
        "prog_global_std",
        "prog_global_late",
        "hop_growth_global",
        "neighbor_ac_global",
        "loc_rank_both_t25_s15_ndx45",
        "loc_rank_both_t25_s15_ndx25",
        "loc_rank_both_t25_s0_ndx45",
        "loc_rank_both_t20_s0_ndx25",
        "loc_prog_both_t20_s0_ndx25",
        "loc_prog_both_t25_s15_ndx45",
        "loc_prog_both_t25_s0_ndx45",
        "loc_rank_lower_t25_s15_ndx45",
        "loc_rank_lower_t25_s15_ndx25",
        "loc_rank_lower_t20_s0_ndx25",
        "loc_rank_arc4_t25",
        "loc_rank_arc4_t30",
        "loc_prog_arc4_t25",
    }
    if include_hybrid_species:
        want.update(
            {
                "hyb_rank_both_t25_s15_ndx45_sp0",
                "hyb_rank_both_t25_s15_ndx45_sp20",
                "hyb_rank_both_t25_s15_ndx45_sp30",
                "hyb_rank_both_t25_s15_ndx45_sp40",
                "hyb_rank_both_t25_s0_ndx45_sp20",
                "hyb_rank_both_t25_s0_ndx45_sp30",
                "hyb_rank_both_t25_s0_ndx45_sp40",
                "hyb_rank_lower_t25_s15_ndx45_sp30",
                "hyb_prog_both_t25_s15_ndx45_sp40",
            }
        )
    pools = comprehensive_rule_architecture_grid()
    if include_hybrid_species:
        pools = pools + hybrid_teacher_species_rule_grid()
    by_name = {r.name: r for r in pools}
    out = [by_name[n] for n in sorted(want) if n in by_name]
    return out or pools[:20]


def incubation_rule_grid() -> list[TemporalGrowthRuleConfig]:
    """Localized templates + incubation gate (no phi until t_frac >= threshold).

    Reuses ``global_onset_frac`` in ``predict_phi_temporal_at_time``.
    """
    base_names = (
        "loc_prog_both_t20_s0_ndx25",
        "loc_rank_both_t25_s15_ndx45",
        "loc_rank_both_t20_s0_ndx25",
        "loc_prog_both_t25_s15_ndx45",
        "loc_rank_both_t25_s0_ndx45",
    )
    incub_fracs = (0.20, 0.30, 0.40, 0.50)
    pools = comprehensive_rule_architecture_grid()
    by_name = {r.name: r for r in pools}
    grid: list[TemporalGrowthRuleConfig] = []
    for bname in base_names:
        base = by_name.get(bname)
        if base is None:
            continue
        grid.append(base)
        for inc in incub_fracs:
            tag = int(round(float(inc) * 100))
            grid.append(
                replace(
                    base,
                    name=f"{bname}_inc{tag}",
                    global_onset_frac=float(inc),
                )
            )
    return grid


def _localized_prog_template() -> TemporalGrowthRuleConfig:
    pools = comprehensive_rule_architecture_grid()
    by_name = {r.name: r for r in pools}
    base = by_name.get("loc_prog_both_t20_s0_ndx25")
    if base is None:
        spatial = prior_rule_config_from_env()
        loc = LocalizedSpatialConfig(
            mode="wall_half",
            segment_top_frac=0.20,
            skip_wall_arc_frac=0.0,
            neg_dx_risk_weight=0.25,
            wall_halves=("lower", "upper"),
            normalize_risk_per_half=True,
        )
        base = TemporalGrowthRuleConfig(
            name="loc_prog_both_t20_s0_ndx25",
            kind="progressive_topk",
            spatial_rule=spatial,
            localized=loc,
            start_frac=0.05,
            end_frac=0.22,
            power=1.5,
        )
    return base


def offset_ramp_rule_grid() -> list[TemporalGrowthRuleConfig]:
    """Time offset before growth + optional promotion boost to fill shortened window."""
    base = _localized_prog_template()
    grid: list[TemporalGrowthRuleConfig] = [
        base,
        replace(base, name="loc_prog_both_t20_s0_ndx25_inc40", global_onset_frac=0.40),
    ]
    offsets = (0.25, 0.35, 0.45)
    boosts = (1.0, 1.35, 1.70)
    for off in offsets:
        for boost in boosts:
            grid.append(
                replace(
                    base,
                    name=f"offramp_off{int(off * 100)}_b{int(boost * 10)}",
                    global_onset_frac=float(off),
                    promotion_boost=float(boost),
                )
            )
    return grid


def threshold_accum_rule_grid() -> list[TemporalGrowthRuleConfig]:
    """Risk points accumulate per step; promote at threshold Y; split bonus to neighbors."""
    base = _localized_prog_template()
    spatial = base.spatial_rule or prior_rule_config_from_env()
    loc = base.localized
    combos: list[tuple[float, float, float, float]] = [
        (1.0, 0.20, 0.80, 0.03),
        (1.2, 0.25, 0.80, 0.03),
        (1.5, 0.25, 0.75, 0.05),
        (1.0, 0.30, 0.85, 0.02),
        (2.0, 0.20, 0.70, 0.05),
        (1.5, 0.35, 0.80, 0.03),
    ]
    onsets = (0.0, 0.35)
    grid: list[TemporalGrowthRuleConfig] = []
    for onset in onsets:
        for thr, gain, sw, sl in combos:
            tag = f"thr{int(thr * 10)}_g{int(gain * 100)}_sw{int(sw * 100)}"
            if onset > 0:
                tag = f"inc{int(onset * 100)}_{tag}"
            grid.append(
                TemporalGrowthRuleConfig(
                    name=f"accum_{tag}",
                    kind="threshold_accum",
                    spatial_rule=spatial,
                    localized=loc,
                    global_onset_frac=float(onset),
                    accum_threshold=float(thr),
                    accum_gain=float(gain),
                    accum_split_wall=float(sw),
                    accum_split_lumen=float(sl),
                )
            )
    return grid


def ideas_rule_grid() -> list[TemporalGrowthRuleConfig]:
    """Offset-ramp + threshold-accumulation probe grids."""
    return offset_ramp_rule_grid() + threshold_accum_rule_grid()


def shear_risk_rule_grid() -> list[TemporalGrowthRuleConfig]:
    """Shear-gradient risk blends on inc40 progressive template (~15m sweep budget)."""
    base = _localized_prog_template()
    spatial = base.spatial_rule or prior_rule_config_from_env()
    base_loc = base.localized
    if base_loc is None:
        base_loc = LocalizedSpatialConfig(
            mode="wall_half",
            segment_top_frac=0.20,
            skip_wall_arc_frac=0.0,
            neg_dx_risk_weight=0.25,
            wall_halves=("lower", "upper"),
            normalize_risk_per_half=True,
        )

    # (tag, neg_dx, sep, stasis, lgrad, lss_si, lgrad_thr, size_mode)
    variants: list[tuple[str, float, float, float, float, float, float, str]] = [
        ("base", 0.25, 0.0, 0.0, 0.0, 0.0, 10.0, ""),
        ("neg55", 0.55, 0.15, 0.15, 0.15, 0.0, 10.0, ""),
        ("neg70", 0.70, 0.10, 0.10, 0.10, 0.0, 10.0, ""),
        ("sep40", 0.25, 0.40, 0.20, 0.15, 0.0, 10.0, ""),
        ("stag40", 0.20, 0.15, 0.40, 0.25, 10.0, 10.0, ""),
        ("lgrad35", 0.25, 0.15, 0.25, 0.35, 0.0, 10.0, ""),
        ("lss10", 0.20, 0.15, 0.45, 0.20, 10.0, 10.0, ""),
        ("lss8", 0.20, 0.15, 0.45, 0.20, 8.0, 10.0, ""),
        ("combo", 0.40, 0.25, 0.20, 0.15, 10.0, 10.0, ""),
        ("auto", 0.30, 0.20, 0.30, 0.20, 10.0, 10.0, "auto"),
        ("sm_neg", 0.50, 0.20, 0.15, 0.15, 10.0, 10.0, "small_neg_dx"),
        ("lg_stag", 0.20, 0.15, 0.50, 0.15, 10.0, 10.0, "large_stasis"),
        ("lgrad5", 0.25, 0.15, 0.25, 0.35, 0.0, 5.0, ""),
        ("lgrad15", 0.25, 0.15, 0.25, 0.35, 0.0, 15.0, ""),
        ("sep55", 0.20, 0.55, 0.15, 0.10, 0.0, 10.0, ""),
        ("neg45_lss10", 0.45, 0.20, 0.20, 0.15, 10.0, 10.0, "auto"),
    ]

    grid: list[TemporalGrowthRuleConfig] = []
    for tag, ndx, sep, stag, lgrad, lss, lg_thr, sz in variants:
        loc = replace(
            base_loc,
            neg_dx_risk_weight=float(ndx),
            sep_stream_risk_weight=float(sep),
            stasis_risk_weight=float(stag),
            low_grad_risk_weight=float(lgrad),
            low_shear_thresh_si=float(lss),
            low_grad_thresh_si=float(lg_thr),
            aneurysm_size_mode=str(sz),
        )
        grid.append(
            replace(
                base,
                name=f"sh_{tag}_inc40",
                kind="progressive_topk",
                localized=loc,
                spatial_rule=spatial,
                global_onset_frac=0.40,
            )
        )
    return grid


def hybrid_teacher_species_rule_grid() -> list[TemporalGrowthRuleConfig]:
    """Kinematic localized templates + teacher Fi/Mat risk blend (y ch from baked anchors).

    ``blend_species_into_risk`` reads FI/Mat from ``data.y``; after
    ``dump_teacher_species_to_anchors.py`` those channels are model predictions.
    """
    spatial = prior_rule_config_from_env()
    base_onset = dict(spatial_rule=spatial, onset_spread=0.55, min_onset_frac=0.08)
    base_prog = dict(spatial_rule=spatial, start_frac=0.05, end_frac=0.22, power=1.5)

    templates: list[tuple[str, str, float, float, tuple[str, ...], float]] = [
        ("both", "ranked_onset", 0.25, 0.15, ("lower", "upper"), 0.45),
        ("both", "ranked_onset", 0.25, 0.0, ("lower", "upper"), 0.45),
        ("both", "progressive_topk", 0.25, 0.15, ("lower", "upper"), 0.45),
        ("lower", "ranked_onset", 0.25, 0.15, ("lower",), 0.45),
        ("both", "ranked_onset", 0.30, 0.15, ("lower", "upper"), 0.45),
    ]
    sp_weights = (0.0, 0.20, 0.30, 0.40)

    grid: list[TemporalGrowthRuleConfig] = []
    for wtag, kind, top, skip, walls, ndx_w in templates:
        for sp_w in sp_weights:
            loc = LocalizedSpatialConfig(
                mode="wall_half",
                segment_top_frac=float(top),
                skip_wall_arc_frac=float(skip),
                neg_dx_risk_weight=float(ndx_w),
                wall_halves=walls,
                normalize_risk_per_half=True,
                species_risk_weight=float(sp_w),
                species_gt_time="t_out",
            )
            ktag = "rank" if kind == "ranked_onset" else "prog"
            st = int(skip * 100)
            wlabel = "both" if len(walls) > 1 else "lower"
            sp_tag = f"sp{int(sp_w * 100)}"
            name = f"hyb_{ktag}_{wlabel}_t{int(top * 100)}_s{st}_ndx{int(ndx_w * 100)}_{sp_tag}"
            base = base_onset if kind == "ranked_onset" else base_prog
            grid.append(TemporalGrowthRuleConfig(name=name, kind=kind, localized=loc, **base))
    return grid


def _time_frac_at_index(data, time_index: int) -> float:
    from src.core_physics.clot_continuous_time import time_frac_for_rollout

    return time_frac_for_rollout(data, int(time_index), clamp_unit=False)


def compute_spatial_risk_score(
    data,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    t_in: int,
    ceiling: torch.Tensor,
    spatial_rule: ClotPriorRuleConfig | None = None,
) -> torch.Tensor:
    n = int(data.num_nodes)
    u, v = _resolve_uv_for_temporal_risk(data, t_in, device)
    props = _anchor_flow_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior = clot_prior_score_flat(data, u, v, bio_cfg, props).reshape(-1)
    stag = fields.flux_stag.reshape(-1)
    neg_dx = (-fields.dgamma_dx_phys).clamp(min=0.0).reshape(-1)
    wall = _wall_mask_from_data(data, device, n)
    hop = _hop_distance_from_seed(wall, data.edge_index.to(device)).float()
    dx = fields.flux_path_dx_raw.reshape(-1)

    pool = ceiling.reshape(-1).bool()
    rule = spatial_rule
    if rule and rule.rank_sdf_max_nd is not None:
        sdf = sdf_nd_from_data(data, device, n)
        pool = pool & (sdf <= float(rule.rank_sdf_max_nd))
    if rule and rule.skip_inlet_quantile is not None:
        if hasattr(data, "mask_inlet") and data.mask_inlet is not None:
            inlet = data.mask_inlet.view(-1).to(device).bool()
            if int(inlet.numel()) == n and bool(inlet.any().item()):
                hin = _hop_distance_from_seed(inlet, data.edge_index.to(device)).float()
                eligible = pool & (hin > 0)
                if bool(eligible.any().item()):
                    thr = torch.quantile(hin[eligible], float(rule.skip_inlet_quantile))
                    pool = pool & (hin >= thr)

    def _norm(v: torch.Tensor) -> torch.Tensor:
        if not bool(pool.any().item()):
            return torch.zeros(n, device=device)
        return (v - v[pool].min()) / (v[pool].max() - v[pool].min() + 1e-12)

    score = 0.40 * _norm(prior) + 0.35 * _norm(stag) + 0.25 * _norm(neg_dx)
    if rule and rule.rank_tie_break:
        score = score + 1e-6 * (_norm(dx) + _norm(-hop))
    return score.clamp(0, 1) * pool.float()


def _vessel_span_nd(data) -> float:
    if hasattr(data, "x") and torch.is_tensor(data.x) and data.x.dim() == 2 and data.x.shape[1] >= 2:
        xy = data.x[:, :2].detach().float()
        span = xy.max(dim=0).values - xy.min(dim=0).values
        return float(torch.linalg.vector_norm(span))
    return 0.01


def _effective_low_shear_thresh_si(bio_cfg: BiochemConfig, loc: LocalizedSpatialConfig) -> float:
    if float(loc.low_shear_thresh_si) > 0:
        return float(loc.low_shear_thresh_si)
    return float(bio_cfg.lss)


def compute_localized_risk_score(
    data,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    t_in: int,
    ceiling: torch.Tensor,
    pool: torch.Tensor,
    spatial_rule: ClotPriorRuleConfig | None,
    loc: LocalizedSpatialConfig,
) -> torch.Tensor:
    """Risk with tunable shear channels, then optional per-half renormalization."""
    n = int(data.num_nodes)
    u, v = _resolve_uv_for_temporal_risk(data, t_in, device)
    props = _anchor_flow_props(data, device)
    fields = compute_clot_kinematics_fields(data, u, v, bio_cfg, props)
    prior = clot_prior_score_flat(data, u, v, bio_cfg, props).reshape(-1)
    stag_legacy = fields.flux_stag.reshape(-1)
    neg_dx = (-fields.dgamma_dx_phys).clamp(min=0.0).reshape(-1)
    sep_stream = fields.flux_path_stream.reshape(-1)
    grad_mag = torch.sqrt(
        fields.dgamma_dx_phys.reshape(-1) ** 2 + fields.dgamma_dy_phys.reshape(-1) ** 2
    )
    lss = _effective_low_shear_thresh_si(bio_cfg, loc)
    T_ls = max(float(bio_cfg.soft_step_T_low_shear) * float(bio_cfg.soft_step_T_scale), 1e-6)
    low_shear = torch.sigmoid(((lss - fields.gamma_si.reshape(-1)) / T_ls).clamp(-50.0, 50.0))
    vel_mag_si = torch.sqrt(u ** 2 + v ** 2) * props["u_ref"].to(device=device).reshape(-1)
    u_ref_safe = props["u_ref"].to(device=device).reshape(-1).clamp(min=1e-8)
    residence = torch.exp(-(vel_mag_si / u_ref_safe).clamp(min=0.0, max=50.0))
    stasis = low_shear * (1.0 + 0.5 * residence)
    lgrad_thr = max(float(loc.low_grad_thresh_si), 1e-6)
    T_gr = max(float(bio_cfg.soft_step_T_grad) * float(bio_cfg.soft_step_T_scale), 1e-6)
    low_grad_zone = torch.sigmoid(((lgrad_thr - grad_mag) / T_gr).clamp(-50.0, 50.0))

    w_ndx = max(float(loc.neg_dx_risk_weight), 0.0)
    w_sep = max(float(loc.sep_stream_risk_weight), 0.0)
    w_stag = max(float(loc.stasis_risk_weight), 0.0)
    w_lgrad = max(float(loc.low_grad_risk_weight), 0.0)
    shear_mode = (w_sep + w_stag + w_lgrad) > 1e-6 or bool(str(loc.aneurysm_size_mode).strip())

    sz_mode = str(loc.aneurysm_size_mode or "").strip().lower()
    if sz_mode == "auto":
        large = _vessel_span_nd(data) >= 0.018
        if large:
            w_stag *= 1.25
        else:
            w_ndx *= 1.25
    elif sz_mode == "small_neg_dx":
        w_ndx *= 1.35
    elif sz_mode == "large_stasis":
        w_stag *= 1.35

    def _norm(v: torch.Tensor) -> torch.Tensor:
        if not bool(pool.any().item()):
            return torch.zeros(n, device=device)
        return (v - v[pool].min()) / (v[pool].max() - v[pool].min() + 1e-12)

    if not shear_mode:
        w_rem = max(1.0 - w_ndx, 0.0)
        w_prior = 0.40 * w_rem / 0.75 if w_rem > 0 else 0.0
        w_stag_legacy = 0.35 * w_rem / 0.75 if w_rem > 0 else 0.0
        score = w_prior * _norm(prior) + w_stag_legacy * _norm(stag_legacy) + w_ndx * _norm(neg_dx)
    else:
        w_sum = w_ndx + w_sep + w_stag + w_lgrad
        if w_sum < 1e-6:
            w_ndx, w_sep, w_stag, w_lgrad = 0.35, 0.25, 0.25, 0.15
            w_sum = 1.0
        inv = 1.0 / w_sum
        score = (
            (w_ndx * inv) * _norm(neg_dx)
            + (w_sep * inv) * _norm(sep_stream)
            + (w_stag * inv) * _norm(stasis)
            + (w_lgrad * inv) * _norm(low_grad_zone)
        )

    rule = spatial_rule
    if rule and rule.rank_tie_break:
        dx = fields.flux_path_dx_raw.reshape(-1)
        wall = _wall_mask_from_data(data, device, n)
        hop = _hop_distance_from_seed(wall, data.edge_index.to(device)).float()
        score = score + 1e-6 * (_norm(dx) + _norm(-hop))
    score = score.clamp(0, 1) * pool.float()
    if loc.normalize_risk_per_half:
        score = normalize_risk_per_wall_half(score, data, device, pool, loc)
    return score


def _resolve_pool_risk(
    data,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    ceiling: torch.Tensor,
    cfg: TemporalGrowthRuleConfig,
    t_out: int,
    t_in: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = cfg.spatial_rule or prior_rule_config_from_env()
    if cfg.localized is not None:
        pool = build_eligible_pool(data, device, ceiling, spatial, cfg.localized)
        risk = compute_localized_risk_score(
            data,
            device=device,
            bio_cfg=bio_cfg,
            t_in=cfg.risk_flow_time,
            ceiling=ceiling,
            pool=pool,
            spatial_rule=spatial,
            loc=cfg.localized,
        )
    else:
        pool = ceiling.reshape(-1).bool()
        risk = compute_spatial_risk_score(
            data,
            device=device,
            bio_cfg=bio_cfg,
            t_in=cfg.risk_flow_time,
            ceiling=ceiling,
            spatial_rule=spatial,
        )
    if cfg.localized is not None and cfg.localized.species_risk_weight > 0:
        ti = resolve_species_time_index(data, cfg.localized.species_gt_time, t_out, t_in)
        risk = blend_species_into_risk(risk, data, device, pool, cfg.localized, ti)
    elif cfg.localized is None:
        risk = risk * pool.float()
    return pool, risk


def _localized_static_support(
    data,
    *,
    device: torch.device,
    pool: torch.Tensor,
    risk: torch.Tensor,
    cfg: TemporalGrowthRuleConfig,
    t_out: int,
    t_in: int = 0,
) -> torch.Tensor:
    loc = cfg.localized
    if loc is None:
        raise ValueError("localized static support requires cfg.localized")
    return build_localized_static_support(
        risk, data, device, pool, loc, t_out=t_out, t_in=t_in
    )


def _growth_time_frac(t_frac: float, onset_frac: float) -> float:
    onset = max(float(onset_frac), 0.0)
    if onset > 0.0 and t_frac < onset:
        return 0.0
    if onset > 0.0:
        return (float(t_frac) - onset) / max(1.0 - onset, 1e-6)
    return float(t_frac)


def _progressive_frac_from_growth_u(
    cfg: TemporalGrowthRuleConfig,
    u_grow: float,
    *,
    extrap: bool = False,
    sim_end_scale: float = 1.0,
) -> float:
    from src.core_physics.clot_continuous_time import extrap_frac_headroom

    ug = max(0.0, float(u_grow))
    lo = float(cfg.start_frac)
    hi = min(0.95, float(cfg.end_frac) * max(float(cfg.promotion_boost), 1.0))
    power = max(float(cfg.power), 0.1)
    if extrap and ug > 1.0 + 1e-6:
        hi_ex = min(0.95, hi + extrap_frac_headroom())
        scale = max(float(sim_end_scale), 1.0 + 1e-6)
        t_extra = min((ug - 1.0) / max(scale - 1.0, 1e-6), 1.0)
        return hi + (hi_ex - hi) * t_extra
    ug = min(ug, 1.0)
    return lo + (hi - lo) * (ug ** power)


def _progressive_frac(cfg: TemporalGrowthRuleConfig, t_out: int, t_final: int) -> float:
    tf = max(int(t_final), 1)
    u = float(t_out) / tf
    return _progressive_frac_from_growth_u(cfg, u)


def predict_phi_temporal_at_time(
    data,
    t_out: int,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    cfg: TemporalGrowthRuleConfig,
    ceiling: torch.Tensor,
    risk: torch.Tensor,
    phi_prev: torch.Tensor | None,
    t_final: int,
    use_provided_risk: bool = False,
    onset_override: float | None = None,
    sim_end_scale: float | None = None,
) -> torch.Tensor:
    from src.core_physics.clot_continuous_time import (
        continuous_extrap_growth_enabled,
        feature_time_index,
        growth_time_frac,
        growth_u_from_t_frac,
        sim_end_scale_from_env,
    )

    n = int(data.num_nodes)
    t_virt = max(0, int(t_out))
    t_feat = feature_time_index(data, t_virt)
    scale = float(sim_end_scale if sim_end_scale is not None else sim_end_scale_from_env())
    extrap = continuous_extrap_growth_enabled()
    t_frac = growth_time_frac(data, t_virt, bio_cfg=bio_cfg)
    spatial = cfg.spatial_rule or prior_rule_config_from_env()
    onset_eff = (
        float(onset_override) if onset_override is not None else float(cfg.global_onset_frac)
    )

    if onset_eff > 0 and t_frac < onset_eff and t_virt <= t_final:
        return torch.zeros(n, device=device)

    pool, risk_hand = _resolve_pool_risk(
        data, device=device, bio_cfg=bio_cfg, ceiling=ceiling, cfg=cfg, t_out=t_feat
    )
    if use_provided_risk:
        risk_eff = risk.reshape(-1).to(device=device, dtype=risk_hand.dtype)
    else:
        risk_eff = risk_hand

    if cfg.kind == "static_spatial":
        if cfg.localized is not None:
            return _localized_static_support(
                data, device=device, pool=pool, risk=risk_eff, cfg=cfg, t_out=t_out
            ).float()
        phi, _ = predict_phi_prior_rule(
            data, device, bio_cfg, rule=spatial, t_in=cfg.risk_flow_time, ceiling_hops=2
        )
        return phi.reshape(-1).float()

    if cfg.kind == "progressive_topk":
        u_grow = growth_u_from_t_frac(
            t_frac,
            onset_eff,
            extrap=extrap,
            sim_end_scale=scale,
            tau_comsol_end=1.0,
        )
        frac = min(
            _progressive_frac_from_growth_u(
                cfg, u_grow, extrap=extrap, sim_end_scale=scale
            ),
            0.95,
        )
        if cfg.localized is not None:
            loc = cfg.localized
            loc_scale = frac / max(float(cfg.end_frac), 0.01)
            top_base = float(loc.segment_top_frac)
            if extrap and loc_scale > 1.0 + 1e-6:
                from src.core_physics.clot_continuous_time import extrap_frac_headroom

                top_cap = min(0.95, top_base + extrap_frac_headroom())
                eff_top = min(top_base * loc_scale, top_cap)
            else:
                eff_top = min(top_base * loc_scale, top_base)
            eff_top = max(eff_top, 0.01)
            flag = segment_topk_mask(risk_eff, data, device, pool, loc, top_frac_override=eff_top)
        else:
            flag = _top_frac_mask(risk_eff, pool, frac)
        out = flag.float()
        if phi_prev is not None:
            out = torch.maximum(out, phi_prev.reshape(-1).float())
        return out

    if cfg.kind == "ranked_onset":
        if cfg.localized is not None:
            static = _localized_static_support(
                data, device=device, pool=pool, risk=risk_eff, cfg=cfg, t_out=t_out
            )
        else:
            static, _ = predict_phi_prior_rule(
                data, device, bio_cfg, rule=spatial, t_in=cfg.risk_flow_time, ceiling_hops=2
            )
            static = static.reshape(-1).bool()
        static = static.reshape(-1).bool() & pool
        if not bool(static.any().item()):
            return torch.zeros(n, device=device)
        r_static = risk_eff[static]
        rmin = float(r_static.min())
        rmax = float(r_static.max()) + 1e-12
        rel = (r_static - rmin) / (rmax - rmin)
        onset = torch.full((n,), 1.0, device=device)
        idx = static.nonzero(as_tuple=False).reshape(-1)
        onset[idx] = float(cfg.min_onset_frac) + float(cfg.onset_spread) * (1.0 - rel)
        return static.float() * (t_frac >= onset).float()

    if cfg.kind == "hop_growth":
        seed = (
            segment_topk_mask(risk_eff, data, device, pool, cfg.localized)
            if cfg.localized is not None
            else _top_frac_mask(risk_eff, pool, max(float(cfg.seed_frac), 0.01))
        )
        committed = seed.clone()
        ei = data.edge_index.to(device=device)
        thr = torch.quantile(risk_eff[pool], float(cfg.risk_floor_quantile)) if bool(pool.any()) else risk_eff.median()
        extra_hops = max(0, t_virt - t_final) if extrap else 0
        for _ in range(max(int(t_feat), 0) + extra_hops):
            frontier = graph_dilate_hops(committed, ei, max(int(cfg.hop_per_step), 1))
            committed = committed | (frontier & pool & (risk_eff >= thr))
        return committed.float()

    if cfg.kind == "neighbor_ac":
        seed = (
            segment_topk_mask(risk_eff, data, device, pool, cfg.localized)
            if cfg.localized is not None
            else _top_frac_mask(risk_eff, pool, max(float(cfg.seed_frac), 0.01))
        )
        committed = seed.clone()
        ei = data.edge_index.to(device=device)
        src, dst = ei[0], ei[1]
        extra_hops = max(0, t_virt - t_final) if extrap else 0
        for step in range(max(int(t_feat), 0) + extra_hops):
            if bool(committed.any().item()):
                nb = torch.zeros(n, device=device)
                nb.scatter_add_(0, src, committed[dst].float())
                nb.scatter_add_(0, dst, committed[src].float())
                deg = torch.zeros(n, device=device)
                deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
                deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
                nb_frac = nb / deg.clamp(min=1.0)
            else:
                nb_frac = torch.zeros(n, device=device)
            rq = torch.quantile(risk_eff[pool], float(cfg.neighbor_risk_q)) if bool(pool.any()) else risk_eff.median()
            catalytic = (nb_frac >= 0.34) & pool & (risk_eff >= rq)
            prog_k = min(_progressive_frac(cfg, step + 1, t_final), 0.95)
            if cfg.localized is not None:
                prog = segment_topk_mask(risk_eff, data, device, pool, cfg.localized) & (
                    risk_eff >= torch.quantile(risk_eff[pool], 1.0 - prog_k)
                )
            else:
                prog = _top_frac_mask(risk_eff, pool, prog_k)
            committed = committed | catalytic | prog
        return committed.float()

    if cfg.kind == "threshold_accum":
        raise ValueError("threshold_accum is stateful; use rollout_temporal_phi")

    raise ValueError(f"unknown temporal rule kind {cfg.kind}")


def _rollout_threshold_accum(
    data,
    cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    bio_cfg: BiochemConfig,
    time_stride: int = 1,
) -> dict[int, torch.Tensor]:
    n = int(data.num_nodes)
    n_times = int(data.y.shape[0])
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    ei = data.edge_index.to(device=device)
    src, dst = ei[0], ei[1]

    accum = torch.zeros(n, device=device)
    committed = torch.zeros(n, dtype=torch.bool, device=device)
    phi_by_t: dict[int, torch.Tensor] = {}

    gain = float(cfg.accum_gain)
    thr = max(float(cfg.accum_threshold), 1e-6)
    sw = float(cfg.accum_split_wall)
    sl = float(cfg.accum_split_lumen)
    onset = float(cfg.global_onset_frac)

    for t_out in range(0, n_times, max(int(time_stride), 1)):
        t_frac = _time_frac_at_index(data, t_out)
        if onset > 0.0 and t_frac < onset:
            phi_by_t[int(t_out)] = committed.float()
            continue

        pool, risk_eff = _resolve_pool_risk(
            data,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            cfg=cfg,
            t_out=t_out,
        )
        active = pool & (~committed)
        if bool(active.any().item()) and bool(pool.any().item()):
            rp = risk_eff[pool]
            rmin = float(rp.min())
            rmax = float(rp.max()) + 1e-12
            rnorm = torch.zeros_like(risk_eff)
            rnorm[pool] = (risk_eff[pool] - rmin) / (rmax - rmin)
            accum[active] = accum[active] + rnorm[active] * gain

        newly = active & (accum >= thr)
        if bool(newly.any().item()):
            committed = committed | newly
            accum[newly] = 0.0
            budget_w = thr * sw
            budget_l = thr * sl
            new_src = newly[src]
            wall_e = new_src & pool[dst]
            lumen_e = new_src & (~pool[dst])
            if bool(wall_e.any().item()):
                deg = torch.zeros(n, device=device)
                deg.scatter_add_(0, src[wall_e], torch.ones_like(src[wall_e], dtype=torch.float32))
                w_amt = budget_w / deg[src[wall_e]].clamp(min=1.0)
                accum.scatter_add_(0, dst[wall_e], w_amt)
            if bool(lumen_e.any().item()):
                deg = torch.zeros(n, device=device)
                deg.scatter_add_(0, src[lumen_e], torch.ones_like(src[lumen_e], dtype=torch.float32))
                l_amt = budget_l / deg[src[lumen_e]].clamp(min=1.0)
                accum.scatter_add_(0, dst[lumen_e], l_amt)

        phi_by_t[int(t_out)] = committed.float()
    return phi_by_t


def rollout_temporal_phi(
    data,
    cfg: TemporalGrowthRuleConfig,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    time_stride: int = 1,
    sim_end_scale: float | None = None,
) -> dict[int, torch.Tensor]:
    del phys_cfg
    if cfg.kind == "threshold_accum":
        return _rollout_threshold_accum(
            data, cfg, device=device, bio_cfg=bio_cfg, time_stride=time_stride
        )
    from src.core_physics.clot_continuous_time import feature_time_index, rollout_time_indices

    n_times = int(data.y.shape[0])
    t_indices = rollout_time_indices(data, time_stride=time_stride, sim_end_scale=sim_end_scale)
    t_final = n_times - 1
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    phi_by_t: dict[int, torch.Tensor] = {}
    phi_prev: torch.Tensor | None = None
    scale = float(sim_end_scale if sim_end_scale is not None else 1.0)
    for t_out in t_indices:
        pool, risk = _resolve_pool_risk(
            data,
            device=device,
            bio_cfg=bio_cfg,
            ceiling=ceiling,
            cfg=cfg,
            t_out=feature_time_index(data, int(t_out)),
        )
        phi = predict_phi_temporal_at_time(
            data,
            t_out,
            device=device,
            bio_cfg=bio_cfg,
            cfg=cfg,
            ceiling=ceiling,
            risk=risk,
            phi_prev=phi_prev,
            t_final=t_final,
            sim_end_scale=scale,
        )
        phi_by_t[int(t_out)] = phi
        phi_prev = phi
    return phi_by_t


def _shape_from_phi_at_time(
    data,
    phi: torch.Tensor,
    t_in: int,
    t_out: int,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict[str, float]:
    step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
    mu = log_blend_mu_eff_si(step.mu_c_si, phi.reshape(-1))
    mu = project_deploy_mu_with_support(
        data=data,
        step=step,
        mu_pred=mu,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        forecast_one_step=True,
        time_index=t_out,
        bulk_time_index=t_out,
        phi_pred_by_time=None,
    )
    y_sl = data.y[int(t_out)].to(device=device, dtype=torch.float32)
    pred_state = y_sl.clone()
    pred_state[:, STATE_CHANNEL_MU_EFF_ND] = phys_cfg.viscosity_si_to_nd(mu.reshape(-1))
    shape_mask: torch.Tensor | None = None
    eval_mask = os.environ.get("CLOT_SHAPE_EVAL_MASK", "ceiling").strip().lower()
    if eval_mask not in ("", "0", "false", "full", "none", "off"):
        shape_mask = resolve_ceiling_mask(data, device, bio_cfg)
    shape = compute_clot_shape_metrics(
        pred_state=pred_state,
        gt_state=y_sl,
        edge_index=data.edge_index.to(device),
        phys_cfg=phys_cfg,
        node_mask=shape_mask,
    )
    return {k: float(v) for k, v in shape.items() if isinstance(v, (int, float))}


def eval_temporal_rule_on_anchor(
    data,
    cfg: TemporalGrowthRuleConfig,
    *,
    stem: str,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    pair_stride: int = 1,
) -> dict[str, Any]:
    pairs = iter_forecast_pairs(int(data.y.shape[0]), time_stride=1, pair_stride=pair_stride)
    if not pairs:
        return {"anchor": stem, "rule": cfg.describe(), "n_pairs": 0}

    phi_by_t = rollout_temporal_phi(data, cfg, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg)
    ceiling = resolve_ceiling_mask(data, device, bio_cfg)
    t_final = int(data.y.shape[0]) - 1
    rows: list[dict[str, float]] = []
    for t_in, t_out in pairs:
        phi = phi_by_t.get(int(t_out))
        if phi is None:
            pool, risk = _resolve_pool_risk(
                data,
                device=device,
                bio_cfg=bio_cfg,
                ceiling=ceiling,
                cfg=cfg,
                t_out=t_out,
                t_in=t_in,
            )
            phi = predict_phi_temporal_at_time(
                data,
                t_out,
                device=device,
                bio_cfg=bio_cfg,
                cfg=cfg,
                ceiling=ceiling,
                risk=risk,
                phi_prev=phi_by_t.get(int(t_in)),
                t_final=t_final,
            )
        step = build_clot_forecast_pair_step(data, t_in, t_out, phys_cfg, bio_cfg, device)
        band = _clot_metrics(phi, step.phi_gt, step.loss_mask)
        rows.append(
            {
                "t_in": float(t_in),
                "t_out": float(t_out),
                "t_frac": _time_frac_at_index(data, t_out),
                **{k: float(v) for k, v in band.items()},
            }
        )

    tfinal = rows[-1]
    early = [r for r in rows if r["t_frac"] <= 0.35]
    late = [r for r in rows if r["t_frac"] >= 0.65]

    def _mean(key: str, subset: list[dict]) -> float:
        if not subset:
            return float("nan")
        return float(sum(r[key] for r in subset) / len(subset))

    t_final_idx = int(tfinal["t_out"])
    t_in_final = int(pairs[-1][0])
    phi_final = phi_by_t.get(t_final_idx)
    if phi_final is None:
        phi_final = torch.zeros(int(data.num_nodes), device=device)
    shape_final = _shape_from_phi_at_time(
        data, phi_final, t_in_final, t_final_idx, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
    )

    shape_timeline: list[float] = []
    shape_timeline_bal: list[float] = []
    shape_late: list[float] = []
    for t_in, t_out in pairs:
        phi_t = phi_by_t.get(int(t_out))
        if phi_t is None:
            continue
        t_frac = _time_frac_at_index(data, int(t_out))
        sm = _shape_from_phi_at_time(
            data, phi_t, int(t_in), int(t_out), device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg
        )
        sh = sm.get("clot_shape", float("nan"))
        if sh == sh:
            shape_timeline.append(float(sh))
            if t_frac >= 0.35:
                shape_late.append(float(sh))
        bal = sm.get("clot_shape_balanced", float("nan"))
        if bal == bal:
            shape_timeline_bal.append(float(bal))

    mean_shape = sum(shape_timeline) / len(shape_timeline) if shape_timeline else float("nan")
    mean_shape_bal = (
        sum(shape_timeline_bal) / len(shape_timeline_bal) if shape_timeline_bal else float("nan")
    )
    mean_shape_late = sum(shape_late) / len(shape_late) if shape_late else float("nan")

    return {
        "anchor": stem,
        "rule": cfg.name,
        "rule_desc": cfg.describe(),
        "n_pairs": len(rows),
        "mean_band_f1": _mean("clot_f1", rows),
        "mean_band_prec": _mean("clot_prec", rows),
        "mean_band_rec": _mean("clot_rec", rows),
        "mean_band_pred_frac": _mean("pred_pos_frac", rows),
        "early_mean_f1": _mean("clot_f1", early),
        "early_mean_pred_frac": _mean("pred_pos_frac", early),
        "late_mean_f1": _mean("clot_f1", late),
        "tfinal_band_f1": float(tfinal["clot_f1"]),
        "tfinal_band_pred_frac": float(tfinal["pred_pos_frac"]),
        "tfinal_gt_pos_frac": float(tfinal["gt_pos_frac"]),
        "early_mean_gt_pos_frac": _mean("gt_pos_frac", early),
        "tfinal_band_rec": float(tfinal["clot_rec"]),
        "tfinal_clot_shape": float(shape_final.get("clot_shape", float("nan"))),
        "tfinal_clot_shape_eff": float(shape_final.get("clot_shape_efficiency", float("nan"))),
        "tfinal_clot_shape_bal": float(shape_final.get("clot_shape_balanced", float("nan"))),
        "tfinal_clot_recall": float(shape_final.get("clot_recall", float("nan"))),
        "tfinal_clot_f1": float(shape_final.get("clot_f1", float("nan"))),
        "tfinal_clot_pred_frac": float(shape_final.get("clot_pred_frac", float("nan"))),
        "tfinal_flow_ok": bool(shape_final.get("flow_ok", False)),
        "mean_clot_shape": float(mean_shape),
        "mean_clot_shape_bal": float(mean_shape_bal),
        "mean_clot_shape_late": float(mean_shape_late),
        "n_timeline_shape": len(shape_timeline),
    }


def symmetric_coverage_match(
    pred_frac: float,
    gt_frac: float,
    *,
    scale: float | None = None,
) -> float:
    """1 when pred_frac == gt_frac; linear decay to 0 at |pred-gt| >= scale."""
    pred = max(0.0, float(pred_frac) if pred_frac == pred_frac else 0.0)
    gt = max(0.0, float(gt_frac) if gt_frac == gt_frac else 0.0)
    ref = float(scale) if scale is not None else max(gt, 0.12)
    ref = max(ref, 1e-6)
    return max(0.0, 1.0 - abs(pred - gt) / ref)


def compute_deploy_score(
    *,
    p007_tfinal_shape: float,
    p007_early_pred: float,
    p007_tfinal_bal: float,
    p007_pred: float,
    tfinal_band_f1: float = float("nan"),
    tfinal_gt_pos_frac: float = float("nan"),
    early_gt_pos_frac: float = float("nan"),
) -> float:
    """Model-select score for clot ML ladder (per-anchor eval row).

    Primary: band F1 at t_final (phi>0.5 commits vs GT). Secondary: symmetric
    match of pred+ vs gt+ at t_final (under- and over-predict equally bad).
    Mu-shape terms are down-weighted and gated by band F1 so zero-commit models
    cannot score high. Early times only penalize over-prediction vs GT early rate.

    Legacy arg names (p007_*) are historical; values are per-anchor, not p007-only.
    """
    tf = float(p007_tfinal_shape) if p007_tfinal_shape == p007_tfinal_shape else 0.0
    bal = float(p007_tfinal_bal) if p007_tfinal_bal == p007_tfinal_bal else tf
    early_pred = max(0.0, float(p007_early_pred) if p007_early_pred == p007_early_pred else 0.0)
    pred = max(0.0, float(p007_pred) if p007_pred == p007_pred else 0.0)
    gt = max(0.0, float(tfinal_gt_pos_frac) if tfinal_gt_pos_frac == tfinal_gt_pos_frac else 0.0)
    early_gt = max(0.0, float(early_gt_pos_frac) if early_gt_pos_frac == early_gt_pos_frac else 0.0)
    band_f1 = float(tfinal_band_f1) if tfinal_band_f1 == tfinal_band_f1 else 0.0

    cover = symmetric_coverage_match(pred, gt)
    early_over = max(0.0, early_pred - early_gt)
    early_ok = 1.0 - min(early_over / 0.35, 1.0)

    # Shape only counts when commits exist (avoids Step-7-style mu-only inflation).
    commit_gate = min(1.0, max(band_f1, pred, gt) / 0.08)
    shape_w = tf * commit_gate
    bal_w = bal * commit_gate

    return (
        0.40 * band_f1
        + 0.25 * cover
        + 0.18 * shape_w
        + 0.12 * bal_w
        + 0.05 * early_ok
    )


def deploy_score_from_eval_row(row: dict[str, Any]) -> float:
    """Build deploy_score from a standard per-anchor ladder eval dict."""
    return compute_deploy_score(
        p007_tfinal_shape=float(row.get("tfinal_clot_shape", float("nan"))),
        p007_early_pred=float(row.get("early_mean_pred_frac", float("nan"))),
        p007_tfinal_bal=float(row.get("tfinal_clot_shape_bal", float("nan"))),
        p007_pred=float(row.get("tfinal_band_pred_frac", float("nan"))),
        tfinal_band_f1=float(row.get("tfinal_band_f1", float("nan"))),
        tfinal_gt_pos_frac=float(row.get("tfinal_gt_pos_frac", float("nan"))),
        early_gt_pos_frac=float(row.get("early_mean_gt_pos_frac", float("nan"))),
    )


def aggregate_temporal_rule_sweep(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool: dict[str, list[dict]] = {}
    for row in results:
        if row.get("n_pairs", 0) < 1:
            continue
        pool.setdefault(row["rule"], []).append(row)
    agg: list[dict[str, Any]] = []
    for rule, rows in pool.items():
        early_vals = [r["early_mean_f1"] for r in rows if r["early_mean_f1"] == r["early_mean_f1"]]
        mean_f1 = sum(r["mean_band_f1"] for r in rows) / len(rows)
        early_f1 = sum(early_vals) / len(early_vals) if early_vals else float("nan")
        tfinal_f1 = sum(r["tfinal_band_f1"] for r in rows) / len(rows)
        agg.append(
            {
                "rule": rule,
                "rule_desc": rows[0].get("rule_desc", rule),
                "n_anchors": len(rows),
                "mean_band_f1": mean_f1,
                "early_mean_f1": early_f1,
                "late_mean_f1": sum(r["late_mean_f1"] for r in rows if r["late_mean_f1"] == r["late_mean_f1"])
                / max(len([r for r in rows if r["late_mean_f1"] == r["late_mean_f1"]]), 1),
                "tfinal_mean_f1": tfinal_f1,
                "tfinal_mean_pred_frac": sum(r["tfinal_band_pred_frac"] for r in rows) / len(rows),
                "p007_tfinal_f1": next(
                    (r["tfinal_band_f1"] for r in rows if r["anchor"] == "patient007"), float("nan")
                ),
                "balance_score": (mean_f1 + 0.35 * early_f1 + 0.65 * tfinal_f1) / 2.0
                if early_f1 == early_f1
                else mean_f1,
            }
        )
    agg.sort(key=lambda x: (-x["balance_score"], -x["mean_band_f1"]))
    return agg


def aggregate_architecture_sweep(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate with deploy_score ranking (tfinal shape + early timing + anti-paint)."""
    pool: dict[str, list[dict]] = {}
    for row in results:
        if row.get("n_pairs", 0) < 1:
            continue
        pool.setdefault(row["rule"], []).append(row)
    agg: list[dict[str, Any]] = []
    for rule, rows in pool.items():
        tfinal_shape = sum(r.get("tfinal_clot_shape", float("nan")) for r in rows) / len(rows)
        mean_shape = sum(r.get("mean_clot_shape", float("nan")) for r in rows) / len(rows)
        mean_shape_bal = sum(r.get("mean_clot_shape_bal", float("nan")) for r in rows) / len(rows)
        mean_f1 = sum(r["mean_band_f1"] for r in rows) / len(rows)
        tfinal_f1 = sum(r["tfinal_band_f1"] for r in rows) / len(rows)
        pred_frac = sum(r["tfinal_band_pred_frac"] for r in rows) / len(rows)
        p007_timeline_shape = next(
            (r.get("mean_clot_shape", float("nan")) for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        p007_tfinal_shape = next(
            (r.get("tfinal_clot_shape", float("nan")) for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        p007_timeline_bal = next(
            (r.get("mean_clot_shape_bal", float("nan")) for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        p007_band = next(
            (r["tfinal_band_f1"] for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        p007_pred = next(
            (r["tfinal_band_pred_frac"] for r in rows if r["anchor"] == "patient007"),
            pred_frac,
        )
        p007_shape_eff = next(
            (r.get("tfinal_clot_shape_eff", float("nan")) for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        p007_shape_bal = next(
            (r.get("tfinal_clot_shape_bal", float("nan")) for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        p007_early_pred = next(
            (r.get("early_mean_pred_frac", float("nan")) for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        tfinal_shape_eff = sum(r.get("tfinal_clot_shape_eff", float("nan")) for r in rows) / len(rows)
        tfinal_shape_bal = sum(r.get("tfinal_clot_shape_bal", float("nan")) for r in rows) / len(rows)
        p007_gt = next(
            (r.get("tfinal_gt_pos_frac", float("nan")) for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        p007_early_gt = next(
            (r.get("early_mean_gt_pos_frac", float("nan")) for r in rows if r["anchor"] == "patient007"),
            float("nan"),
        )
        deploy = compute_deploy_score(
            p007_tfinal_shape=float(p007_tfinal_shape),
            p007_early_pred=float(p007_early_pred),
            p007_tfinal_bal=float(p007_shape_bal),
            p007_pred=float(p007_pred),
            tfinal_band_f1=float(p007_band),
            tfinal_gt_pos_frac=float(p007_gt),
            early_gt_pos_frac=float(p007_early_gt),
        )
        agg.append(
            {
                "rule": rule,
                "rule_desc": rows[0].get("rule_desc", rule),
                "n_anchors": len(rows),
                "deploy_score": deploy,
                "composite_score": deploy,
                "tfinal_mean_clot_shape": tfinal_shape,
                "tfinal_mean_clot_shape_eff": tfinal_shape_eff,
                "tfinal_mean_clot_shape_bal": tfinal_shape_bal,
                "mean_clot_shape": mean_shape,
                "mean_timeline_clot_shape_bal": mean_shape_bal,
                "mean_band_f1": mean_f1,
                "tfinal_mean_f1": tfinal_f1,
                "tfinal_mean_pred_frac": pred_frac,
                "p007_clot_shape": p007_timeline_shape,
                "p007_timeline_clot_shape": p007_timeline_shape,
                "p007_tfinal_clot_shape": p007_tfinal_shape,
                "p007_clot_shape_eff": p007_shape_eff,
                "p007_clot_shape_bal": p007_timeline_bal,
                "p007_tfinal_clot_shape_bal": p007_shape_bal,
                "p007_early_pred_frac": p007_early_pred,
                "p007_tfinal_f1": p007_band,
                "balance_score": deploy,
            }
        )
    agg.sort(
        key=lambda x: (
            -x.get("deploy_score", float("nan"))
            if x.get("deploy_score") == x.get("deploy_score")
            else 0.0,
            -x.get("p007_tfinal_clot_shape", float("nan"))
            if x.get("p007_tfinal_clot_shape") == x.get("p007_tfinal_clot_shape")
            else 0.0,
            x.get("p007_early_pred_frac", 1.0)
            if x.get("p007_early_pred_frac") == x.get("p007_early_pred_frac")
            else 1.0,
        )
    )
    return agg


def pick_architecture_winner(agg: list[dict[str, Any]]) -> dict[str, Any] | None:
    return agg[0] if agg else None


def pick_architecture_winner_shape(agg: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best p007 tfinal hop-graded clot_shape (end-state geometry)."""
    if not agg:
        return None
    return max(
        agg,
        key=lambda x: (
            x.get("p007_tfinal_clot_shape", float("nan"))
            if x.get("p007_tfinal_clot_shape") == x.get("p007_tfinal_clot_shape")
            else float("-inf"),
            x.get("p007_timeline_clot_shape", x.get("p007_clot_shape", float("nan")))
            if x.get("p007_timeline_clot_shape", x.get("p007_clot_shape")) == x.get(
                "p007_timeline_clot_shape", x.get("p007_clot_shape")
            )
            else float("-inf"),
        ),
    )


def pick_architecture_winner_balanced(agg: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best p007 tfinal clot_shape_balanced (end-state, anti-painting)."""
    if not agg:
        return None
    return max(
        agg,
        key=lambda x: (
            x.get("p007_tfinal_clot_shape_bal", float("nan"))
            if x.get("p007_tfinal_clot_shape_bal") == x.get("p007_tfinal_clot_shape_bal")
            else float("-inf"),
            x.get("p007_tfinal_clot_shape", float("nan"))
            if x.get("p007_tfinal_clot_shape") == x.get("p007_tfinal_clot_shape")
            else float("-inf"),
        ),
    )


def pick_architecture_winner_deploy(agg: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best visual-calibrated deploy score (tfinal shape + early timing + anti-paint)."""
    if not agg:
        return None
    return max(
        agg,
        key=lambda x: (
            x.get("deploy_score", float("nan"))
            if x.get("deploy_score") == x.get("deploy_score")
            else float("-inf"),
            x.get("p007_tfinal_clot_shape", float("nan"))
            if x.get("p007_tfinal_clot_shape") == x.get("p007_tfinal_clot_shape")
            else float("-inf"),
            -float(x.get("p007_early_pred_frac", 1.0))
            if x.get("p007_early_pred_frac") == x.get("p007_early_pred_frac")
            else float("-inf"),
        ),
    )


def pick_architecture_winner_incubation(agg: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best incubation rule by deploy_score among offset variants."""
    if not agg:
        return None
    inc = [
        r
        for r in agg
        if "_inc" in str(r.get("rule", "")) or str(r.get("rule", "")).startswith("offramp_")
    ]
    if not inc:
        return pick_architecture_winner_deploy(agg)
    return max(
        inc,
        key=lambda x: (
            x.get("deploy_score", float("nan"))
            if x.get("deploy_score") == x.get("deploy_score")
            else float("-inf"),
            x.get("p007_tfinal_clot_shape", float("nan"))
            if x.get("p007_tfinal_clot_shape") == x.get("p007_tfinal_clot_shape")
            else float("-inf"),
            -float(x.get("p007_early_pred_frac", 1.0))
            if x.get("p007_early_pred_frac") == x.get("p007_early_pred_frac")
            else float("-inf"),
        ),
    )


def pick_growing_winner(agg: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best non-static rule (rules that change phi over time)."""
    growing = [r for r in agg if r["rule"] != "static_spatial"]
    return growing[0] if growing else None


def pick_localized_deploy_winner(agg: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best deployable localized rule (no GT species oracle), precision-aware."""
    cands = [
        r
        for r in agg
        if r["rule"].startswith("loc_") and "sp_gt" not in r["rule"] and "recess" not in r["rule"]
    ]
    if not cands:
        return None

    def _score(row: dict[str, Any]) -> float:
        p007 = row.get("p007_tfinal_f1", float("nan"))
        pred = row.get("tfinal_mean_pred_frac", 1.0)
        if p007 != p007:
            p007 = row.get("tfinal_mean_f1", 0.0)
        return 0.55 * p007 + 0.25 * row.get("mean_band_f1", 0.0) + 0.20 * (1.0 - pred)

    return max(cands, key=_score)
