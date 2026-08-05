"""s9.8 (WALL_MODEL_PLAN.md): train-window coverage + sliding-window deploy selection.

WG_stenosis_subcohort_ft (v1) regressed because two structural gaps let it: training windows
never started past ~66% of the timeline (late-forming clot under-sampled), and checkpoint
selection graded a single point (t_final) so a mid-rollout collapse could hide behind a decent
final score. Both are opt-in overrides -- every default here must reproduce the exact legacy
behaviour so no existing leg's numbers move.
"""

from __future__ import annotations

import os

import pytest

from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
from src.architecture.runtime_config import (
    BiochemRuntimeConfig,
    ScoringConfig,
    use_biochem_runtime,
)
from src.core_physics.species_pushforward_continuous import (
    TRAIN_T0_COVERAGE_MIN_RUNWAY,
    deploy_eval_clot_times,
    deploy_eval_time_fracs,
    pushforward_train_t0_coverage_frac,
    train_t0_max_for_n_times,
)


@pytest.fixture(autouse=True)
def _clean_env():
    keys = ("SPECIES_PUSHFORWARD_TRAIN_T0_COVERAGE_FRAC", "SPECIES_CONTINUOUS_DEPLOY_EVAL_TIME_FRACS")
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def test_train_t0_coverage_frac_default_is_unchanged_legacy_formula():
    assert pushforward_train_t0_coverage_frac() == 0.0
    assert train_t0_max_for_n_times(201) == 132  # legacy formula, byte-identical
    assert train_t0_max_for_n_times(92) == 60


def test_train_t0_coverage_frac_override_widens_the_cap():
    pf = PushforwardConfig(train_t0_coverage_frac=0.85)
    with use_pushforward_config(pf):
        cap = train_t0_max_for_n_times(201)
        assert cap == round(0.85 * 200)
        assert cap > 132  # strictly covers more of the horizon than the legacy formula


def test_train_t0_coverage_frac_never_eats_the_unroll_runway():
    """Even coverage_frac=1.0 must leave TRAIN_T0_COVERAGE_MIN_RUNWAY steps of room."""
    pf = PushforwardConfig(train_t0_coverage_frac=1.0)
    with use_pushforward_config(pf):
        last = 200
        cap = train_t0_max_for_n_times(201)
        assert cap <= last - TRAIN_T0_COVERAGE_MIN_RUNWAY
        # Also true on a short (patient039-like) vessel.
        cap_short = train_t0_max_for_n_times(92)
        assert cap_short <= 91 - TRAIN_T0_COVERAGE_MIN_RUNWAY


def test_deploy_eval_time_fracs_default_empty_preserves_legacy_single_point():
    assert deploy_eval_time_fracs() == []
    assert deploy_eval_clot_times(201) == [200]


def test_deploy_eval_time_fracs_sliding_window_resolves_to_expected_indices():
    rt = BiochemRuntimeConfig(scoring=ScoringConfig(deploy_eval_time_fracs="0.65,1.0"))
    with use_biochem_runtime(rt):
        assert deploy_eval_time_fracs() == [0.65, 1.0]
        assert deploy_eval_clot_times(201) == [130, 200]


def test_deploy_eval_time_fracs_takes_priority_over_legacy_dual():
    """Sliding-window fracs must win even when the old dual flag is also (accidentally) set."""
    rt = BiochemRuntimeConfig(
        scoring=ScoringConfig(deploy_eval_dual=True, deploy_eval_time_fracs="0.5")
    )
    with use_biochem_runtime(rt):
        assert deploy_eval_clot_times(201) == [100]


def test_deploy_eval_time_fracs_clamped_and_deduped():
    rt = BiochemRuntimeConfig(scoring=ScoringConfig(deploy_eval_time_fracs="-1,0.5,0.5,2.0"))
    with use_biochem_runtime(rt):
        # -1 -> 0.0, 2.0 -> 1.0, duplicate 0.5 collapses.
        assert deploy_eval_clot_times(201) == [0, 100, 200]
