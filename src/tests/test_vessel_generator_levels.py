import numpy as np

from src.data_gen.lib.vessel_generator import (
    _sample_params,
    cohort_levels,
    default_level_mix,
    parse_level_mix,
    resolve_bend_sign_mode,
)
from src.config import VesselConfig


def test_default_level_mix_sums_to_n():
    mix = default_level_mix(100)
    assert sum(mix.values()) == 100
    assert mix[2] >= 1


def test_parse_level_mix():
    assert parse_level_mix("10,20,5", 35) == {0: 10, 1: 20, 2: 5}


def test_cohort_levels_mixed_shuffle():
    rng = np.random.default_rng(0)
    levels = cohort_levels(6, level=0, level_mix={0: 2, 1: 2, 2: 2}, rng=rng)
    assert sorted(levels) == [0, 0, 1, 1, 2, 2]
    assert levels != [0, 0, 1, 1, 2, 2]


def test_sample_params_level2_avoids_straight_centerline():
    cfg = VesselConfig(phase="kinematics")
    rng = np.random.default_rng(42)
    for i in range(50):
        p = _sample_params(i, 2, cfg, rng)
        assert p["curve_type"] != "straight"
        assert p["v_type"] in ("stenosis", "aneurysm")


def test_sample_params_level1_arc_has_both_bend_signs():
    import os

    os.environ["KINEMATICS_BEND_SIGN_MODE"] = "bidirectional"
    cfg = VesselConfig(phase="kinematics")
    rng = np.random.default_rng(7)
    signs = set()
    for i in range(200):
        p = _sample_params(i, 1, cfg, rng)
        if p["curve_type"] in ("arc", "hook"):
            signs.add(p["bend_sign"])
        if signs == {-1.0, 1.0}:
            break
    assert signs == {-1.0, 1.0}


def test_sample_params_level1_down_only_fixed_sign():
    import os

    os.environ["KINEMATICS_BEND_SIGN_MODE"] = "down_only"
    cfg = VesselConfig(phase="kinematics")
    rng = np.random.default_rng(99)
    for i in range(80):
        p = _sample_params(i, 1, cfg, rng)
        if p["curve_type"] in ("arc", "hook"):
            assert p["bend_sign"] == 1.0
        if p["curve_type"] == "s_curve":
            assert p["amplitude"] >= 0.0
    assert resolve_bend_sign_mode() == "down_only"
