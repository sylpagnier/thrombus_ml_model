"""Typed runtime / deploy configuration for the biochem stack.

Architecture knobs live in ``PushforwardConfig``. Everything else that used to be
an ``os.environ`` string (coupling, rollout policy, scoring, gelation aux) belongs
here as frozen dataclasses.

Policy
------
* Prefer ``dataclasses.replace(cfg, **kwargs)`` and explicit args.
* Never mutate ``os.environ`` to toggle features, sweeps, or deploy policy.
* Process/IO only (checkpoint paths, tqdm, CUDA) may stay as CLI/env.
* Use ``use_biochem_runtime(cfg)`` so helpers resolve typed values without globals.
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterator, Mapping


@dataclass(frozen=True)
class CouplingConfig:
    """Local-corrector / closed-loop flow coupling."""

    corrector_coupling: bool = True
    closed_loop_coupling: bool = False
    kine_resolve_on_clot: bool = False
    corrector_num_hops: int = 4
    corrector_mu_thresh: float = 1e-3
    corrector_max_delta_mu: float = 3.0
    corrector_ckpt: str = ""
    local_clusters: bool = True
    cluster_radius_nd: float = 0.12
    cluster_max_nodes: int = 64
    growth_factor: float = 1.05
    kine_resolve_min_clot_nodes: int = 40
    kine_resolve_min_band_frac: float = 0.0
    kine_resolve_growth_factor: float = 1.5


@dataclass(frozen=True)
class RolloutDeployConfig:
    """Deploy-faithful rollout IC / velocity / occlusion / band policy."""

    deploy_faithful: bool = True
    rollout_vel_source: str = "kinematics"  # gt | kinematics | coupled
    train_vel_source: str = "gt"
    rollout_ic_source: str = "resting"
    rollout_pin_other: str = "rest"
    dynamic_occlusion: bool = False
    wall_hops: int = 3
    nucleation_hops: int = 3
    ceiling_hops: int = 3
    kin_per_vessel_norm: bool = True
    wall_mat_only: bool = True  # CLOT_PHI_PHYSICS_WALL_MAT_ONLY
    augment_mirror_y: bool = False
    latent_dropout: float = 0.0
    deploy_horizon: int = 0
    deploy_eval_full: bool = True
    deploy_horizon_all_packs: bool = True
    deploy_horizon_aux_cap: int = 72
    train_deploy_eval_flow: str = "auto"  # auto | gt | kinematics | coupled
    t0_flow_source: str = "auto"
    # --- s17 Z2/Z3: what goes into data.x[UV_PRIOR|MU_PRIOR|WSS_PRIOR] ---
    # The packs ship these bit-identical to the converged clot-free CFD field y[0] (s16.1), and
    # the RGP-DEQ consumes them as inputs. The deployment contract is geometry + IC/BC ONLY, so
    # "stored" trains on information that will not exist at deploy.
    #   stored   -- as shipped. LEAKED. Kept only to reproduce legs v1-v10.
    #   analytic -- Poiseuille magnitude + potential-flow direction from geometry+BC. LEGAL.
    #   zero     -- prior block zeroed; the Z1 ablation floor.
    prior_source: str = "stored"


@dataclass(frozen=True)
class ScoringConfig:
    """Checkpoint selection / clot-score policy."""

    clout_score_mode: str = "guiding"  # guiding | relaxed_prec_floor | relaxed_f05 | ...
    clout_prec_rec_floor: float = 0.30
    guide_relax_hops: int = 2
    precision_select: bool = False
    speed_fp_weight: float = 4.0
    guide_f_beta: float = 0.5
    guide_iou_w: float = 0.5
    guide_f05_w: float = 0.5
    empty_gt_fp_tol: float = 8.0
    deploy_eval_dual: bool = False
    deploy_dual_full_w: float = 0.5
    # Comma-separated fractions of the full timeline (e.g. "0.5,0.75,1.0") -- sliding-window
    # coverage of checkpoint-selection deploy grading across the WHOLE horizon, not just the
    # final step. Takes priority over deploy_eval_dual when set. "" = legacy dual/single
    # behaviour, unchanged. See WALL_MODEL_PLAN.md s9.8 -- late-forming clot needs the
    # selection metric to see how the rollout holds up well before t_final, not only at it.
    deploy_eval_time_fracs: str = ""
    # Held-out deploy_only selection (soft mass / overpaint; hard catastrophe reject).
    # Defaults preserve legacy 0.70*score + 0.30*mat_f1 with no mass term.
    select_clot_score_weight: float = 0.70
    select_mat_f1_weight: float = 0.30
    # Wall-gen gate: primary F1 (0 => legacy score+mat only). See passes_wall_gen_gate.
    select_clot_f1_weight: float = 0.0
    select_mass_soft_lambda: float = 0.0
    select_mass_soft_target: float = 1.2
    select_mass_hard_max: float = 0.0  # <=0 disables hard reject
    select_mass_hard_min: float = 0.0  # <=0 disables; reject starvation / precision mirage
    select_overpaint_lambda: float = 0.0
    select_overpaint_frac_target: float = 0.08
    # Soft bonuses from deploy seed/front panel (held-out selection only).
    select_seed_prec_lambda: float = 0.0
    select_front_speed_lambda: float = 0.0
    select_fn_fp_lambda: float = 0.0
    select_fn_hard_max: float = 0.0  # <=0 disables; reject FN rise vs floor
    # s9.10: select_front_speed_lambda rewards min(front_speed, 1.5) -- monotonic in front_speed,
    # so it saturates to a CONSTANT (no discrimination) and actively rewards overshoot once
    # front_speed exceeds 1.0 (confirmed dead on WG_stenosis_subcohort_ft_v2: front_speed was
    # 2.5-5.06 every epoch, term contributed a flat +0.30). This is a new, separately-named
    # knob (not a change to the existing one, which WG_prec_front etc already rely on) that
    # penalizes DEVIATION from front_speed=1.0 in either direction instead of rewarding more.
    select_front_speed_target_lambda: float = 0.0
    # s9.10: select_fn_fp_lambda only penalizes FN-heavy (max(0, fn-fp)) -- zero signal in an
    # FP-heavy (overspray) regime, which is exactly what WG_stenosis_subcohort_ft_v2 hit
    # (fn-fp term was 0.000 every epoch while FP ran 110-294). New, separately-named,
    # symmetric knob: penalizes |fn-fp| imbalance in either direction.
    select_fp_fn_imbalance_lambda: float = 0.0
    # <=0 disables; reject if deploy_clot_f1_min (worst sliding-window point, s9.8) falls below
    # this floor -- catches a checkpoint that only looks good at t_final.
    select_f1_min_hard_floor: float = 0.0


@dataclass(frozen=True)
class GelationAuxConfig:
    """Gelation / viscosity calibration aux losses."""

    viscosity_calib: bool = False
    phi_loss_weight: float = 1.0
    mu_loss_weight: float = 0.25
    phi_loss_type: str = "mse"  # mse | bce
    gelation_temp_scale: float = 1.0
    beta_min: float = 0.1
    beta_max: float = 2.0
    beta_override: str = ""
    frontier_boost: float = 2.0
    footprint_tversky: bool = False
    footprint_tversky_alpha: float = 0.7
    footprint_tversky_beta: float = 0.3
    footprint_wall_fp_w: float = 2.0
    footprint_lumen_fn_w: float = 2.0
    footprint_bce_blend: float = 0.25


@dataclass(frozen=True)
class OffwallConfig:
    """Compound / off-wall specialist routing and loss pivots."""

    two_model_mode: bool = False
    offwall_model_ckpt: str = ""
    two_model_route: str = "frontier"  # wall | frontier | frontier_offwall
    two_model_frontier_hops: float = 1.0
    frontier_hops_map: str = ""
    frontier_hops_anchor: str = ""
    isolate_offwall_loss: bool = False
    offwall_loss_scale: float = 1.0
    spatial_gate_heads: bool = False
    skip_hop_gnn: bool = False
    physics_nucleation: bool = False
    lumen_shape_fn_w: float = 0.0
    lumen_shape_fp_w: float = 0.0


# Flat field -> (subconfig_attr, field_name)
_SUBCONFIG_FIELDS: dict[str, tuple[str, str]] = {}
_FIELD_TYPES: dict[str, type] = {}


def _register_sub(name: str, cls: type) -> None:
    for f in fields(cls):
        _SUBCONFIG_FIELDS[f.name] = (name, f.name)
        ann = f.type
        if ann in ("bool", bool):
            _FIELD_TYPES[f.name] = bool
        elif ann in ("int", int):
            _FIELD_TYPES[f.name] = int
        elif ann in ("float", float):
            _FIELD_TYPES[f.name] = float
        else:
            _FIELD_TYPES[f.name] = str


_register_sub("coupling", CouplingConfig)
_register_sub("rollout", RolloutDeployConfig)
_register_sub("scoring", ScoringConfig)
_register_sub("gelation", GelationAuxConfig)
_register_sub("offwall", OffwallConfig)


RUNTIME_ENV_TO_FIELD: dict[str, str] = {
    "BIOCHEM_CORRECTOR_COUPLING": "corrector_coupling",
    "SPECIES_CLOSED_LOOP_COUPLING": "closed_loop_coupling",
    "BIOCHEM_KINE_RESOLVE_ON_CLOT": "kine_resolve_on_clot",
    "BIOCHEM_CORRECTOR_NUM_HOPS": "corrector_num_hops",
    "BIOCHEM_CORRECTOR_MU_THRESH": "corrector_mu_thresh",
    "BIOCHEM_CORRECTOR_MAX_DELTA_MU": "corrector_max_delta_mu",
    "BIOCHEM_CORRECTOR_CKPT": "corrector_ckpt",
    "BIOCHEM_CORRECTOR_LOCAL_CLUSTERS": "local_clusters",
    "BIOCHEM_CORRECTOR_CLUSTER_RADIUS_ND": "cluster_radius_nd",
    "BIOCHEM_CORRECTOR_CLUSTER_MAX_NODES": "cluster_max_nodes",
    "BIOCHEM_CORRECTOR_GROWTH_FACTOR": "growth_factor",
    "BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES": "kine_resolve_min_clot_nodes",
    "BIOCHEM_KINE_RESOLVE_MIN_BAND_FRAC": "kine_resolve_min_band_frac",
    "BIOCHEM_KINE_RESOLVE_GROWTH_FACTOR": "kine_resolve_growth_factor",
    "SPECIES_ROLLOUT_DEPLOY_FAITHFUL": "deploy_faithful",
    "SPECIES_ROLLOUT_VEL_SOURCE": "rollout_vel_source",
    "SPECIES_TRAIN_VEL_SOURCE": "train_vel_source",
    "SPECIES_ROLLOUT_IC_SOURCE": "rollout_ic_source",
    "SPECIES_ROLLOUT_PIN_OTHER": "rollout_pin_other",
    "SPECIES_DYNAMIC_OCCLUSION": "dynamic_occlusion",
    "BIOCHEM_ROLLOUT_DYNAMIC_OCCLUSION": "dynamic_occlusion",
    "SPECIES_SNAPSHOT_WALL_HOPS": "wall_hops",
    "CLOT_V2_NUCLEATION_HOPS": "nucleation_hops",
    "CLOT_PHI_CEILING_HOPS": "ceiling_hops",
    "SPECIES_KIN_PER_VESSEL_NORM": "kin_per_vessel_norm",
    "CLOT_PHI_PHYSICS_WALL_MAT_ONLY": "wall_mat_only",
    "SPECIES_AUGMENT_MIRROR_Y": "augment_mirror_y",
    "SPECIES_LATENT_DROPOUT": "latent_dropout",
    "SPECIES_CONTINUOUS_DEPLOY_HORIZON": "deploy_horizon",
    "SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL": "deploy_eval_full",
    "SPECIES_DEPLOY_HORIZON_ALL_PACKS": "deploy_horizon_all_packs",
    "SPECIES_DEPLOY_HORIZON_AUX_CAP": "deploy_horizon_aux_cap",
    "SPECIES_TRAIN_DEPLOY_EVAL_FLOW": "train_deploy_eval_flow",
    "T0_R4_FLOW_SOURCE": "t0_flow_source",
    "SPECIES_PRIOR_SOURCE": "prior_source",
    "SPECIES_CONTINUOUS_CLOUT_SCORE": "clout_score_mode",
    "SPECIES_CLOUT_PREC_REC_FLOOR": "clout_prec_rec_floor",
    "CLOT_GUIDE_RELAX_HOPS": "guide_relax_hops",
    "SPECIES_MAT_GROWTH_PRECISION_SELECT": "precision_select",
    "SPECIES_CONTINUOUS_SPEED_FP_WEIGHT": "speed_fp_weight",
    "CLOT_GUIDE_F_BETA": "guide_f_beta",
    "CLOT_GUIDE_IOU_W": "guide_iou_w",
    "CLOT_GUIDE_F05_W": "guide_f05_w",
    "CLOT_EMPTY_GT_FP_TOL": "empty_gt_fp_tol",
    "SPECIES_CONTINUOUS_DEPLOY_EVAL_DUAL": "deploy_eval_dual",
    "SPECIES_CONTINUOUS_DEPLOY_DUAL_FULL_W": "deploy_dual_full_w",
    "SPECIES_CONTINUOUS_DEPLOY_EVAL_TIME_FRACS": "deploy_eval_time_fracs",
    "SPECIES_SELECT_CLOT_SCORE_WEIGHT": "select_clot_score_weight",
    "SPECIES_SELECT_MAT_F1_WEIGHT": "select_mat_f1_weight",
    "SPECIES_SELECT_CLOT_F1_WEIGHT": "select_clot_f1_weight",
    "SPECIES_SELECT_MASS_SOFT_LAMBDA": "select_mass_soft_lambda",
    "SPECIES_SELECT_MASS_SOFT_TARGET": "select_mass_soft_target",
    "SPECIES_SELECT_MASS_HARD_MAX": "select_mass_hard_max",
    "SPECIES_SELECT_MASS_HARD_MIN": "select_mass_hard_min",
    "SPECIES_SELECT_OVERPAINT_LAMBDA": "select_overpaint_lambda",
    "SPECIES_SELECT_OVERPAINT_FRAC_TARGET": "select_overpaint_frac_target",
    "SPECIES_SELECT_SEED_PREC_LAMBDA": "select_seed_prec_lambda",
    "SPECIES_SELECT_FRONT_SPEED_LAMBDA": "select_front_speed_lambda",
    "SPECIES_SELECT_FN_FP_LAMBDA": "select_fn_fp_lambda",
    "SPECIES_SELECT_FN_HARD_MAX": "select_fn_hard_max",
    "SPECIES_SELECT_F1_MIN_HARD_FLOOR": "select_f1_min_hard_floor",
    "SPECIES_SELECT_FRONT_SPEED_TARGET_LAMBDA": "select_front_speed_target_lambda",
    "SPECIES_SELECT_FP_FN_IMBALANCE_LAMBDA": "select_fp_fn_imbalance_lambda",
    "SPECIES_VISCOSITY_CALIB": "viscosity_calib",
    "SPECIES_CONTINUOUS_PHI_LOSS_WEIGHT": "phi_loss_weight",
    "SPECIES_CONTINUOUS_MU_LOSS_WEIGHT": "mu_loss_weight",
    "SPECIES_GELATION_PHI_LOSS_TYPE": "phi_loss_type",
    "SPECIES_GELATION_TEMP_SCALE": "gelation_temp_scale",
    "SPECIES_VISCOSITY_BETA_MIN": "beta_min",
    "SPECIES_VISCOSITY_BETA_MAX": "beta_max",
    "SPECIES_GELATION_BETA_OVERRIDE": "beta_override",
    "SPECIES_GELATION_FRONTIER_BOOST": "frontier_boost",
    "SPECIES_FOOTPRINT_TVERSKY": "footprint_tversky",
    "SPECIES_FOOTPRINT_TVERSKY_ALPHA": "footprint_tversky_alpha",
    "SPECIES_FOOTPRINT_TVERSKY_BETA": "footprint_tversky_beta",
    "SPECIES_FOOTPRINT_WALL_FP_W": "footprint_wall_fp_w",
    "SPECIES_FOOTPRINT_LUMEN_FN_W": "footprint_lumen_fn_w",
    "SPECIES_FOOTPRINT_BCE_BLEND": "footprint_bce_blend",
    "SPECIES_TWO_MODEL_MODE": "two_model_mode",
    "SPECIES_OFFWALL_MODEL_CKPT": "offwall_model_ckpt",
    "SPECIES_TWO_MODEL_ROUTE": "two_model_route",
    "SPECIES_TWO_MODEL_FRONTIER_HOPS": "two_model_frontier_hops",
    "SPECIES_TWO_MODEL_FRONTIER_HOPS_MAP": "frontier_hops_map",
    "SPECIES_TWO_MODEL_FRONTIER_HOPS_ANCHOR": "frontier_hops_anchor",
    "SPECIES_ISOLATE_OFFWALL_LOSS": "isolate_offwall_loss",
    "SPECIES_OFFWALL_LOSS_SCALE": "offwall_loss_scale",
    "SPECIES_SPATIAL_GATE_HEADS": "spatial_gate_heads",
    "SPECIES_SKIP_HOP_GNN": "skip_hop_gnn",
    "SPECIES_PHYSICS_NUCLEATION": "physics_nucleation",
    "SPECIES_LUMEN_SHAPE_FN_W": "lumen_shape_fn_w",
    "SPECIES_LUMEN_SHAPE_FP_W": "lumen_shape_fp_w",
}


def _coerce(field_name: str, raw: Any) -> Any:
    ftype = _FIELD_TYPES.get(field_name)
    if ftype is None or raw is None:
        return raw
    if ftype is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if ftype is int:
        return int(float(raw))
    if ftype is float:
        return float(raw)
    return str(raw)


def split_legacy_runtime_env(
    env: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Split legacy env dict into (runtime_kwargs, unknown_remaining)."""
    kwargs: dict[str, Any] = {}
    remaining: dict[str, str] = {}
    if not env:
        return kwargs, remaining
    for key, val in env.items():
        field_name = RUNTIME_ENV_TO_FIELD.get(str(key))
        if field_name is None:
            remaining[str(key)] = str(val)
            continue
        kwargs[field_name] = _coerce(field_name, val)
    return kwargs, remaining


def validate_runtime_kwargs(kwargs: Mapping[str, Any]) -> None:
    bad = sorted(k for k in kwargs if k not in _SUBCONFIG_FIELDS)
    if bad:
        raise TypeError(f"Unknown BiochemRuntimeConfig fields: {bad}")


@dataclass(frozen=True)
class BiochemRuntimeConfig:
    """Composed runtime policy for train / eval / deploy (non-architecture)."""

    coupling: CouplingConfig = CouplingConfig()
    rollout: RolloutDeployConfig = RolloutDeployConfig()
    scoring: ScoringConfig = ScoringConfig()
    gelation: GelationAuxConfig = GelationAuxConfig()
    offwall: OffwallConfig = OffwallConfig()

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any] | None) -> "BiochemRuntimeConfig":
        if not kwargs:
            return cls()
        validate_runtime_kwargs(kwargs)
        buckets: dict[str, dict[str, Any]] = {
            "coupling": {},
            "rollout": {},
            "scoring": {},
            "gelation": {},
            "offwall": {},
        }
        for k, v in kwargs.items():
            sub, field_name = _SUBCONFIG_FIELDS[k]
            buckets[sub][field_name] = _coerce(k, v)
        return cls(
            coupling=replace(CouplingConfig(), **buckets["coupling"]) if buckets["coupling"] else CouplingConfig(),
            rollout=replace(RolloutDeployConfig(), **buckets["rollout"]) if buckets["rollout"] else RolloutDeployConfig(),
            scoring=replace(ScoringConfig(), **buckets["scoring"]) if buckets["scoring"] else ScoringConfig(),
            gelation=replace(GelationAuxConfig(), **buckets["gelation"]) if buckets["gelation"] else GelationAuxConfig(),
            offwall=replace(OffwallConfig(), **buckets["offwall"]) if buckets["offwall"] else OffwallConfig(),
        )

    @classmethod
    def from_env(cls) -> "BiochemRuntimeConfig":
        raw = {k: os.environ[k] for k in RUNTIME_ENV_TO_FIELD if k in os.environ}
        kwargs, _ = split_legacy_runtime_env(raw)
        return cls.from_kwargs(kwargs)

    @classmethod
    def from_meta(cls, meta: Mapping[str, Any] | None) -> "BiochemRuntimeConfig":
        if not meta:
            return cls()
        typed = meta.get("runtime_kwargs")
        if isinstance(typed, dict) and typed:
            return cls.from_kwargs(typed)
        env_ov = meta.get("env_overrides")
        if isinstance(env_ov, dict) and env_ov:
            kwargs, _ = split_legacy_runtime_env(env_ov)
            return cls.from_kwargs(kwargs)
        return cls()

    def with_overrides(self, **kwargs: Any) -> "BiochemRuntimeConfig":
        validate_runtime_kwargs(kwargs)
        base = self.to_flat_dict()
        base.update({k: _coerce(k, v) for k, v in kwargs.items()})
        return type(self).from_kwargs(base)

    def to_flat_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out.update(asdict(self.coupling))
        out.update(asdict(self.rollout))
        out.update(asdict(self.scoring))
        out.update(asdict(self.gelation))
        out.update(asdict(self.offwall))
        return out

    def to_meta_dict(self) -> dict[str, Any]:
        return {"runtime_kwargs": self.to_flat_dict()}


_ACTIVE_RUNTIME: contextvars.ContextVar[BiochemRuntimeConfig | None] = contextvars.ContextVar(
    "biochem_active_runtime",
    default=None,
)


def get_active_runtime() -> BiochemRuntimeConfig | None:
    return _ACTIVE_RUNTIME.get()


def resolve_runtime(explicit: BiochemRuntimeConfig | None = None) -> BiochemRuntimeConfig | None:
    return explicit if explicit is not None else get_active_runtime()


@contextmanager
def use_biochem_runtime(config: BiochemRuntimeConfig | None) -> Iterator[BiochemRuntimeConfig | None]:
    token = _ACTIVE_RUNTIME.set(config)
    try:
        yield config
    finally:
        _ACTIVE_RUNTIME.reset(token)
