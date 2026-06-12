"""Deprecated alias -- use ``src.core_physics.t0_rung4_ladder``."""

from __future__ import annotations

from src.core_physics.t0_rung4_ladder import (  # noqa: F401
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    build_rung4_species_log_nd_at_time as _build_rung4_species,
    describe_rung4_step,
    resting_species_log_nd,
    rollout_rung4_species_series as _rollout_rung4_species,
    rung4_step_from_env as rules_species_mode_from_env,
    rung4_step_uses_gt_species as rules_species_is_oracle,
    species_log_mae_in_mask,
)


def build_rules_species_log_nd_at_time(*args, mode: str | None = None, step: str | None = None, **kwargs):
    """Legacy ``mode=`` kwarg maps to ``step=``."""
    step_s = step if step is not None else mode
    return _build_rung4_species(*args, step=step_s, **kwargs)


def rollout_t0_rules_species_series(*args, mode: str | None = None, step: str | None = None, **kwargs):
    """Legacy ``mode=`` kwarg maps to ``step=``."""
    step_s = step if step is not None else mode
    return _rollout_rung4_species(*args, step=step_s, **kwargs)
