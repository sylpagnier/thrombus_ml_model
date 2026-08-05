"""Configuration dataclass for biochem pushforward architecture.

Replaces scattered os.environ.get() calls with a centralized typed configuration.
Architecture / sweep knobs must flow through this dataclass -- never via os.environ
mutation (see AGENTS.md Configuration Architecture Guardrail).
"""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterator, Mapping

from src.core_physics.constants import *

# Active config for call sites that still omit an explicit PushforwardConfig arg.
_ACTIVE_CONFIG: contextvars.ContextVar[PushforwardConfig | None] = contextvars.ContextVar(
    "pushforward_active_config",
    default=None,
)

# Legacy env key -> PushforwardConfig field (architecture / train knobs that belong here).
ENV_KEY_TO_FIELD: dict[str, str] = {
    "SPECIES_PUSHFORWARD_ARCH": "arch",
    "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE": "species_scope",
    "BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS": "channels",
    "SPECIES_CONTINUOUS_DUAL_HEAD": "dual_head",
    "SPECIES_PUSHFORWARD_CKPT": "ckpt",
    "SPECIES_GEOM_FEATS": "geom_feats",
    "SPECIES_GEOM_FEATS_RICH": "geom_feats_rich",
    "SPECIES_FLOW_FEATS": "flow_feats",
    "SPECIES_FLOW_FEATS_SOURCE": "flow_feats_source",
    "SPECIES_FLOW_FEATS_DYNAMIC": "flow_feats_dynamic",
    "SPECIES_FLOW_FEATS_DROP_XY": "flow_feats_drop_xy",
    "SPECIES_FLOW_FEATS_TIME": "flow_feats_time",
    "SPECIES_FLOW_FEATS_ABLATE": "flow_feats_ablate",
    "SPECIES_FLOW_FEATS_MULTIHOP": "flow_feats_multihop",
    "SPECIES_FLUX_STAG_FEAT": "flux_stag_feat",
    "SPECIES_STAGNATION_FEATS": "stagnation_feats",
    "SPECIES_CONTINUOUS_SATURATION_GATE": "saturation_gate",
    "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE": "neighbor_commit_gate",
    "SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA": "neighbor_commit_alpha",
    "SPECIES_CONTINUOUS_GATE_TEMP": "gate_temp",
    "SPECIES_CONTINUOUS_FRONTIER_HOPS": "frontier_hops",
    "SPECIES_CONTINUOUS_NUCLEATION_TOPK": "nucleation_topk",
    "SPECIES_SHEAR_READOUT_GATE": "shear_readout_gate",
    "SPECIES_FRONTIER_KINETICS": "frontier_kinetics",
    "SPECIES_FRONTIER_K_AP": "frontier_k_ap",
    "SPECIES_FRONTIER_K_T": "frontier_k_t",
    "SPECIES_CONTINUOUS_TIME_CONTEXT": "time_context",
    "SPECIES_CONTINUOUS_TIME_REF_S": "time_ref_s",
    "SPECIES_CONTINUOUS_TIME_FOURIER_FREQS": "time_fourier_freqs",
    "SPECIES_CONTINUOUS_PHYSICS_READOUT": "physics_readout",
    "SPECIES_CONTINUOUS_VEL_DECAY": "vel_decay",
    "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY": "vel_decay_wall_only",
    "SPECIES_CONVECTION_AGGR": "convection_aggr",
    "SPECIES_CONVECTION_ALPHA": "convection_alpha",
    "SPECIES_CONVECTIVE_UPWIND": "convective_upwind",
    "SPECIES_LONGRANGE_EDGES": "longrange_edges",
    "SPECIES_LONGRANGE_DIST_MULT": "longrange_dist_mult",
    "SPECIES_MULTISCALE_SKIP_HOP": "multiscale_skip_hop",
    "SPECIES_MULTISCALE_SKIP_HOP_MULT": "multiscale_skip_hop_mult",
    "SPECIES_MULTISCALE_SKIP_HOP_SCALE": "multiscale_skip_hop_scale",
    "SPECIES_HOP1_SMOOTH": "hop1_smooth",
    "SPECIES_HOP1_SMOOTH_ALPHA": "hop1_smooth_alpha",
    "SPECIES_MIDSIDE_BLIND_LOSS": "midside_blind_loss",
    "SPECIES_CONTINUOUS_DYNAMIC_FRONTIER_MASK": "dynamic_frontier_mask",
    "SPECIES_GROWTH_DILATION": "growth_dilation",
    "SPECIES_PUSHFORWARD_FOCAL_ALPHA_FI": "focal_alpha_fi",
    "SPECIES_PUSHFORWARD_FOCAL_ALPHA_MAT": "focal_alpha_mat",
    "SPECIES_PUSHFORWARD_FOCAL_GAMMA_FI": "focal_gamma_fi",
    "SPECIES_PUSHFORWARD_FOCAL_GAMMA_MAT": "focal_gamma_mat",
    "SPECIES_PHYSICAL_FP_GATING": "physical_fp_gating",
    "SPECIES_PHYSICAL_FP_SPEED_CRIT": "physical_fp_speed_crit",
    "SPECIES_PHYSICAL_FP_SPEED_WIDTH": "physical_fp_speed_width",
    "SPECIES_PHYSICAL_FP_SHEAR_CRIT": "physical_fp_shear_crit",
    "SPECIES_PHYSICAL_FP_SHEAR_WIDTH": "physical_fp_shear_width",
    "SPECIES_PHYSICAL_FP_MIN_WEIGHT": "physical_fp_min_weight",
    "SPECIES_SDF_FP_GATING": "sdf_fp_gating",
    "SPECIES_SDF_FP_DECAY_SCALE": "sdf_fp_decay_scale",
    "SPECIES_SDF_FP_MIN": "sdf_fp_min",
    "SPECIES_GATE_SDF_CRIT": "gate_sdf_crit",
    "SPECIES_GATE_SDF_TEMP": "gate_sdf_temp",
    "SPECIES_PHYSICS_NUC_SPEED_THRESH": "physics_nuc_speed_thresh",
    "SPECIES_PHYSICS_NUC_SHEAR_THRESH": "physics_nuc_shear_thresh",
    "SPECIES_CONTINUOUS_DELTA_OUT_SCALE": "delta_out_scale",
    "SPECIES_CONTINUOUS_DELTA_SOFTPLUS_BETA": "delta_softplus_beta",
    "SPECIES_CONTINUOUS_MAT_COMMIT_THRESH": "mat_commit_thresh",
    "SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS": "growth_only_loss",
    "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "spatial_loss_weight",
    "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "underpred_weight",
    "SPECIES_CONTINUOUS_FP_WEIGHT": "fp_weight",
    "SPECIES_CONTINUOUS_GATE_FP_WEIGHT": "gate_fp_weight",
    "SPECIES_CONTINUOUS_FP_THRESH": "fp_thresh",
    "SPECIES_CONTINUOUS_LOSS_SCALE": "loss_scale",
    "SPECIES_CONTINUOUS_HUBER_BETA": "huber_beta",
    "SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT": "final_state_weight",
    "SPECIES_CONTINUOUS_FINAL_MASS_PENALTY": "final_mass_penalty",
    "SPECIES_CONTINUOUS_FINAL_MASS_TARGET": "final_mass_target",
    "SPECIES_CONTINUOUS_FINAL_PREC_FP_PENALTY": "final_prec_fp_penalty",
    "SPECIES_CONTINUOUS_STEP_MASS_PENALTY": "step_mass_penalty",
    "SPECIES_CONTINUOUS_STEP_PREC_FP_PENALTY": "step_prec_fp_penalty",
    "SPECIES_CONTINUOUS_FREEZE_BACKBONE": "freeze_backbone",
    "SPECIES_CONTINUOUS_SCORE_CLOUT_W": "score_clout_w",
    "SPECIES_PUSHFORWARD_SCORE_GROWTH_W": "score_growth_w",
    "SPECIES_PUSHFORWARD_SCORE_STATE_W": "score_state_w",
    "SPECIES_PUSHFORWARD_STEP_LOSS": "step_loss",
    "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_FI": "channel_weight_fi",
    "SPECIES_CONTINUOUS_CHANNEL_WEIGHT_MAT": "channel_weight_mat",
    "SPECIES_CONTINUOUS_DELTA_THRESH": "delta_thresh",
    "SPECIES_CONTINUOUS_DELTA_THRESH_FI": "delta_thresh_fi",
    "SPECIES_CONTINUOUS_DELTA_THRESH_MAT": "delta_thresh_mat",
    "SPECIES_CONTINUOUS_DELTA_VALUE_SCALE": "delta_value_scale",
    "SPECIES_CONTINUOUS_MAX_SAT_LOG_FI": "max_sat_log_fi",
    "SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT": "max_sat_log_mat",
    "SPECIES_CONTINUOUS_STATE_SCALE": "state_scale",
    "SPECIES_CONTINUOUS_SATURATION_SCALE": "saturation_scale",
    "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL": "saturation_scale_offwall",
    "SPECIES_CONTINUOUS_MATURE_FRAC": "mature_frac",
    "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT": "mature_fp_exempt",
    "SPECIES_CONTINUOUS_DELTA_RESIDUAL": "delta_residual",
    "SPECIES_CONTINUOUS_DELTA_RESIDUAL_ALPHA": "delta_residual_alpha",
    "SPECIES_CONTINUOUS_TEMPORAL_OFFSET": "temporal_offset",
    "SPECIES_CONTINUOUS_TEMPORAL_OFFSET_SCALE": "temporal_offset_scale",
    "SPECIES_PUSHFORWARD_TRAIN_T0_PER_VESSEL": "train_t0_per_vessel",
    "SPECIES_PUSHFORWARD_UNROLL": "unroll",
    "SPECIES_PUSHFORWARD_MAX_UNROLL": "max_unroll",
    "SPECIES_PUSHFORWARD_STEP_STRIDE": "step_stride",
    "SPECIES_PUSHFORWARD_INPUT_NOISE": "input_noise",
    "SPECIES_PUSHFORWARD_TRAIN_T0_MAX": "train_t0_max",
    "SPECIES_PUSHFORWARD_TRAIN_T0_MIN": "train_t0_min",
    "SPECIES_PUSHFORWARD_TRAIN_T0_COVERAGE_FRAC": "train_t0_coverage_frac",
    "SPECIES_PUSHFORWARD_TAU_CENTER": "tau_center",
    "SPECIES_PUSHFORWARD_TAU_SIGMA": "tau_sigma",
    "SPECIES_CONTINUOUS_TBPTT_TAIL": "tbptt_tail",
    "SPECIES_CONTINUOUS_CLOSED_LOOP_INIT": "closed_loop_init",
    "SPECIES_CONTINUOUS_CURRICULUM_UNROLL": "curriculum_unroll",
    "SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND": "final_state_all_band",
    "SPECIES_CONTINUOUS_TEACHER_NOISE": "teacher_noise",
    "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "teacher_fp_frac",
    "SPECIES_CONTINUOUS_TEACHER_BLUR": "teacher_blur",
    "SPECIES_SCHEDULED_SAMPLING": "scheduled_sampling",
    "SPECIES_SS_TARGET_PROB": "ss_target_prob",
    "SPECIES_SS_WARMUP_EPOCHS": "ss_warmup_epochs",
    "SPECIES_SS_ANCHOR_STRIDE": "ss_anchor_stride",
    "SPECIES_SS_NOISY": "ss_noisy",
}

# Populated after PushforwardConfig is defined.
_FIELD_TYPES: dict[str, type] = {}


def get_active_config() -> PushforwardConfig | None:
    return _ACTIVE_CONFIG.get()


def resolve_config(explicit: PushforwardConfig | None = None) -> PushforwardConfig | None:
    """Prefer an explicit arg, else the contextvar set by use_pushforward_config()."""
    return explicit if explicit is not None else get_active_config()


@contextmanager
def use_pushforward_config(config: PushforwardConfig | None) -> Iterator[PushforwardConfig | None]:
    token = _ACTIVE_CONFIG.set(config)
    try:
        yield config
    finally:
        _ACTIVE_CONFIG.reset(token)


def coerce_config_value(field_name: str, raw: Any) -> Any:
    """Coerce a legacy string / JSON-ish value to the PushforwardConfig field type."""
    if field_name not in _FIELD_TYPES:
        return raw
    ftype = _FIELD_TYPES[field_name]
    if raw is None:
        return raw
    if ftype is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if ftype is int:
        return int(float(raw))
    if ftype is float:
        return float(raw)
    if ftype is str:
        return str(raw)
    if ftype is tuple:
        if isinstance(raw, tuple):
            return tuple(int(c) for c in raw)
        if isinstance(raw, list):
            return tuple(int(c) for c in raw)
        text = str(raw).strip()
        if not text:
            return tuple()
        return tuple(int(c) for c in text.split(","))
    return raw


def split_legacy_env_overrides(
    env: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Split a legacy env_overrides dict into (config_kwargs, remaining_runtime_env)."""
    config_kwargs: dict[str, Any] = {}
    remaining: dict[str, str] = {}
    if not env:
        return config_kwargs, remaining
    for key, val in env.items():
        field_name = ENV_KEY_TO_FIELD.get(str(key))
        if field_name is None:
            remaining[str(key)] = str(val)
            continue
        config_kwargs[field_name] = coerce_config_value(field_name, val)
    return config_kwargs, remaining


def validate_config_kwargs(kwargs: Mapping[str, Any]) -> None:
    """Raise TypeError if any key is not a PushforwardConfig field."""
    known = set(_FIELD_TYPES)
    bad = sorted(k for k in kwargs if k not in known)
    if bad:
        raise TypeError(f"Unknown PushforwardConfig fields: {bad}")


@dataclass(frozen=True)
class PushforwardConfig:
    # --- Architecture and scope ---
    arch: str = "sage"
    species_scope: str = "mat"
    channels: tuple[int, ...] = (11,)
    dual_head: bool = True
    ckpt: str = "outputs/biochem/biochem_gnn/species/best.pth"

    # --- Feature flags ---
    geom_feats: bool = False
    geom_feats_rich: bool = False
    flow_feats: bool = False
    flow_feats_source: str = "auto"
    flow_feats_dynamic: bool = False
    flow_feats_drop_xy: bool = False
    flow_feats_time: float = -1.0
    flow_feats_ablate: bool = False
    # Multi-hop flow aggregation. The graded label lives on wall nodes where u=v=0 by
    # no-slip, so 1-hop flow stats barely separate the true clot pocket from a wrong one
    # (AUC 0.41); hop-2 separates at 0.94. See docs/WALL_MODEL_PLAN.md s2.2.
    flow_feats_multihop: bool = False
    flux_stag_feat: bool = False
    stagnation_feats: bool = False

    # --- Gates and logic ---
    saturation_gate: bool = True
    neighbor_commit_gate: bool = False
    neighbor_commit_alpha: float = 0.8
    gate_temp: float = 1.0
    frontier_hops: int = 0
    nucleation_topk: float = 0.0
    shear_readout_gate: bool = False
    frontier_kinetics: bool = False
    frontier_k_ap: float = 0.5
    frontier_k_t: float = 0.5

    # --- Time context ---
    time_context: bool = True
    time_ref_s: float = DEFAULT_TIME_REF_S
    time_fourier_freqs: int = 8

    # --- Deploy Safety & Physics ---
    physics_readout: bool = False
    vel_decay: bool = True
    vel_decay_wall_only: bool = True
    convection_aggr: bool = False
    convection_alpha: float = 0.5
    convective_upwind: bool = False
    # Soft scale on PM-GAT wall mods (rheo/adv/curve). At random init, unscaled mods
    # are O(1-7) while content logits are O(0.03); without this, attention ignores features.
    physics_gat_prior_scale: float = 0.05
    longrange_edges: bool = False
    longrange_dist_mult: float = 2.5
    multiscale_skip_hop: bool = False
    multiscale_skip_hop_mult: float = 3.0
    multiscale_skip_hop_scale: float = 0.5
    hop1_smooth: bool = False
    hop1_smooth_alpha: float = 0.4
    midside_blind_loss: bool = False
    dynamic_frontier_mask: bool = False
    growth_dilation: int = 1
    physics_nuc_speed_thresh: float = 0.15
    physics_nuc_shear_thresh: float = 0.20
    gate_sdf_crit: float = 0.012
    gate_sdf_temp: float = 0.003

    # --- Loss Weights & Scales ---
    growth_only_loss: bool = True
    spatial_loss_weight: float = 1.0
    underpred_weight: float = 2.0
    fp_weight: float = 8.0
    gate_fp_weight: float = 0.0
    fp_thresh: float = 2e-5
    loss_scale: float = 1.0
    huber_beta: float = 1e-4
    final_state_weight: float = 0.35
    final_state_all_band: bool = True
    # Soft mass / FP penalties on the rolled final state (train–deploy alignment).
    # Differentiable occupancy vs GT; 0 disables (legacy behavior).
    final_mass_penalty: float = 0.0
    final_mass_target: float = 1.2
    final_prec_fp_penalty: float = 0.0
    # Per-step soft mass / FP on rolled state vs GT at that time (binds TBPTT, not just final).
    step_mass_penalty: float = 0.0
    step_prec_fp_penalty: float = 0.0
    # Early seed-location aux (not a mass term): soft first-new / early Mat vs GT early pocket.
    # Keep weight small so prec mass/FP stays primary; 0 disables.
    seed_aux_weight: float = 0.0
    seed_aux_early_steps: int = 3
    seed_aux_compact_weight: float = 0.0
    seed_aux_pos_weight: float = 4.0
    # Pocket-contrast (exclusive wrong-pocket): soft-penalize Mat outside k-hop of GT first-seed.
    # Train-only; does NOT hard-mask the forward (growth inside the true pocket stays free).
    pocket_contrast_weight: float = 0.0
    pocket_contrast_hops: int = 4
    pocket_contrast_early_steps: int = 8
    pocket_contrast_inside_weight: float = 0.0  # optional mild under-recall inside allowed
    # Freeze SAGE/conv trunk; train spatial/magnitude heads (+gates) only.
    freeze_backbone: bool = False
    score_clout_w: float = 0.0
    score_growth_w: float = 0.75
    score_state_w: float = 0.25
    step_loss: str = "linear"
    focal_alpha_fi: float = 0.95
    focal_alpha_mat: float = 0.92
    focal_gamma_fi: float = 2.0
    focal_gamma_mat: float = 2.0
    physical_fp_gating: bool = False
    physical_fp_speed_crit: float = 0.05
    physical_fp_speed_width: float = 0.01
    physical_fp_shear_crit: float = 10.0
    physical_fp_shear_width: float = 2.0
    physical_fp_min_weight: float = 0.1
    sdf_fp_gating: bool = False
    sdf_fp_decay_scale: float = 0.015
    sdf_fp_min: float = 0.1

    # --- Channel specific params ---
    channel_weight_fi: float = 1.0
    channel_weight_mat: float = 4.0
    delta_thresh: float = DEFAULT_DELTA_THRESH
    delta_thresh_fi: float = DEFAULT_DELTA_THRESH
    delta_thresh_mat: float = DEFAULT_DELTA_THRESH
    delta_value_scale: float = DEFAULT_DELTA_VALUE_SCALE
    delta_out_scale: float = 1e-5
    delta_softplus_beta: float = 20.0
    mat_commit_thresh: float = -1.0  # <0 => use snapshot_active_log_nd()
    max_sat_log_fi: float = DEFAULT_MAX_SAT_LOG
    max_sat_log_mat: float = DEFAULT_MAX_SAT_LOG
    state_scale: float = DEFAULT_STATE_SCALE

    # --- Saturation & Maturation ---
    saturation_scale: float = DEFAULT_SATURATION_SCALE
    saturation_scale_offwall: float = DEFAULT_SATURATION_SCALE
    mature_frac: float = 0.95
    mature_fp_exempt: bool = True

    # --- Residuals & Offsets ---
    delta_residual: bool = False
    delta_residual_alpha: float = 0.35
    temporal_offset: bool = False
    temporal_offset_scale: float = 0.15

    # --- Training Unroll & Curriculum ---
    train_t0_per_vessel: bool = True
    unroll: int = 10
    max_unroll: int = 200
    step_stride: int = 1
    input_noise: float = 0.05
    train_t0_max: int = 40
    train_t0_min: int = 0
    # 0.0 = legacy per-vessel formula (train_t0_max_for_n_times: ~132/200 steps -- leaves the
    # last third of a full-length timeline unsampled as a window START). >0.0 overrides that
    # formula with `coverage_frac * last_step` (clamped to leave a fixed runway for the unroll
    # length), so windows can start later in the horizon -- for cohorts/vessels where clot forms
    # late and training must see it as a fresh rollout start, not only as a continuation of an
    # earlier window. See WALL_MODEL_PLAN.md s9.8.
    train_t0_coverage_frac: float = 0.0
    tau_center: float = DEFAULT_TAU_CENTER
    tau_sigma: float = DEFAULT_TAU_SIGMA
    tbptt_tail: int = 5
    closed_loop_init: float = 0.45
    curriculum_unroll: bool = True

    # --- Teacher Noise & Blurring ---
    teacher_noise: float = 0.02
    teacher_fp_frac: float = 0.08
    teacher_blur: float = 0.25

    # --- Scheduled Sampling ---
    scheduled_sampling: bool = False
    ss_target_prob: float = 0.5
    ss_warmup_epochs: int = 3
    ss_anchor_stride: int = 10
    ss_noisy: bool = True

    @classmethod
    def from_meta(cls, meta: dict | None) -> "PushforwardConfig":
        """Build config from checkpoint meta (typed fields + legacy env_overrides)."""
        if not meta:
            return cls()

        # Start from defaults, overlay legacy env_overrides, then explicit typed keys.
        cfg = cls()
        env_ov = meta.get("env_overrides")
        if isinstance(env_ov, dict) and env_ov:
            kwargs, _ = split_legacy_env_overrides(env_ov)
            if kwargs:
                validate_config_kwargs(kwargs)
                cfg = replace(cfg, **kwargs)

        typed = meta.get("config_kwargs")
        if isinstance(typed, dict) and typed:
            clean = {k: coerce_config_value(k, v) for k, v in typed.items() if k in _FIELD_TYPES}
            if clean:
                cfg = replace(cfg, **clean)

        def _bool(key: str, default: bool) -> bool:
            val = meta.get(key)
            if val is None:
                return default
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("1", "true", "yes", "on")

        def _float(key: str, default: float) -> float:
            val = meta.get(key)
            if val is None:
                return float(default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return float(default)

        def _int(key: str, default: int) -> int:
            val = meta.get(key)
            if val is None:
                return int(default)
            try:
                return int(float(val))
            except (TypeError, ValueError):
                return int(default)

        arch = str(meta.get("arch") or meta.get("pushforward_arch") or cfg.arch).strip().lower()
        scope = str(
            meta.get("pushforward_species_scope")
            or meta.get("species_scope")
            or cfg.species_scope
        ).strip().lower()

        channels_raw = meta.get("pushforward_species_channels") or meta.get("species_channels")
        if isinstance(channels_raw, (list, tuple)):
            channels = tuple(int(c) for c in channels_raw)
        elif channels_raw:
            channels = tuple(int(c) for c in str(channels_raw).split(","))
        else:
            channels = cfg.channels

        return replace(
            cfg,
            arch=arch,
            species_scope=scope,
            channels=channels,
            dual_head=_bool("dual_head", cfg.dual_head),
            geom_feats=_bool("geom_feats", cfg.geom_feats),
            geom_feats_rich=_bool("geom_feats_rich", cfg.geom_feats_rich),
            flux_stag_feat=_bool("flux_stag_feat", cfg.flux_stag_feat),
            flow_feats=_bool("flow_feats", cfg.flow_feats),
            flow_feats_dynamic=_bool("flow_dynamic", cfg.flow_feats_dynamic)
            if "flow_dynamic" in meta
            else _bool("flow_feats_dynamic", cfg.flow_feats_dynamic),
            flow_feats_drop_xy=_bool("flow_drop_xy", cfg.flow_feats_drop_xy)
            if "flow_drop_xy" in meta
            else _bool("flow_feats_drop_xy", cfg.flow_feats_drop_xy),
            flow_feats_multihop=_bool("flow_multihop", cfg.flow_feats_multihop)
            if "flow_multihop" in meta
            else _bool("flow_feats_multihop", cfg.flow_feats_multihop),
            saturation_gate=_bool("saturation_gate", cfg.saturation_gate),
            neighbor_commit_gate=_bool("neighbor_commit_gate", cfg.neighbor_commit_gate),
            neighbor_commit_alpha=_float("neighbor_commit_alpha", cfg.neighbor_commit_alpha),
            gate_temp=_float("gate_temp", cfg.gate_temp),
            frontier_hops=_int("frontier_hops", cfg.frontier_hops),
            nucleation_topk=_float("nucleation_topk", cfg.nucleation_topk),
            vel_decay=_bool("vel_decay", cfg.vel_decay),
            vel_decay_wall_only=_bool("vel_decay_wall_only", cfg.vel_decay_wall_only),
            delta_residual=_bool("delta_residual", cfg.delta_residual),
            temporal_offset=_bool("temporal_offset", cfg.temporal_offset),
            mature_fp_exempt=_bool("mature_fp_exempt", cfg.mature_fp_exempt),
            physics_readout=_bool("physics_readout", cfg.physics_readout),
        )

    @classmethod
    def from_env(cls) -> "PushforwardConfig":
        """Build config from process env (legacy bridge). Prefer use_pushforward_config."""
        raw = {k: os.environ[k] for k in ENV_KEY_TO_FIELD if k in os.environ}
        kwargs, _ = split_legacy_env_overrides(raw)
        if not kwargs:
            return cls()
        validate_config_kwargs(kwargs)
        return replace(cls(), **kwargs)

    def validate(self) -> None:
        """Raise ValueError on invalid combinations."""
        if self.arch not in ("sage", "gnode", "gat", "physics_gat"):
            # Allow unknown arch strings for forward-compat; gnode requires dual head.
            pass
        if self.arch == "gnode" and not self.dual_head:
            raise ValueError("gnode pushforward arch requires dual_head=True")
        if not self.channels:
            raise ValueError("channels must be non-empty")

    def with_overrides(self, **kwargs: Any) -> "PushforwardConfig":
        validate_config_kwargs(kwargs)
        return replace(self, **kwargs)

    def to_meta_dict(self) -> dict[str, Any]:
        """Typed snapshot for checkpoint meta (preferred over env_overrides)."""
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    def to_env(self) -> dict[str, str]:
        """Export field values as strings for logging only -- do not inject into os.environ."""
        return {k: str(v) for k, v in asdict(self).items()}


# Populate field type map now that the class exists.
_FIELD_TYPES.clear()
for _f in fields(PushforwardConfig):
    # annotations are strings under from __future__ annotations
    ann = _f.type
    if ann in ("bool", bool):
        _FIELD_TYPES[_f.name] = bool
    elif ann in ("int", int):
        _FIELD_TYPES[_f.name] = int
    elif ann in ("float", float):
        _FIELD_TYPES[_f.name] = float
    elif ann in ("str", str):
        _FIELD_TYPES[_f.name] = str
    elif "tuple" in str(ann):
        _FIELD_TYPES[_f.name] = tuple
    else:
        _FIELD_TYPES[_f.name] = str
