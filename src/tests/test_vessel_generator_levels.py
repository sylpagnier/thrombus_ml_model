import numpy as np
import pytest

from src.data_gen.lib.vessel_generator import (
    _sample_params,
    cohort_levels,
    default_level_mix,
    normalize_pathology_mode,
    parse_level_mix,
    resolve_bend_sign_mode,
    stenosis_wall_offset_for_occlusion,
)
from src.config import VesselConfig
from src.data_gen.lib.vessel_geometry import compute_geometry_from_params


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


def test_normalize_pathology_mode_aliases():
    assert normalize_pathology_mode("random") is None
    assert normalize_pathology_mode("max-stenosis") == "max_stenosis"
    assert normalize_pathology_mode("max_aneurysm") == "max_aneurysm"


def test_sample_params_max_stenosis_targets_occlusion():
    cfg = VesselConfig(phase="biochem")
    gen_cfg = {
        "num_ctrl_pts": cfg.num_ctrl_pts,
        "base_length": cfg.base_length,
        "min_lumen_width_fraction": cfg.min_lumen_width_fraction,
        "unit": "m",
    }
    rng = np.random.default_rng(0)
    p = _sample_params(0, 1, cfg, rng, pathology_mode="max_stenosis")
    assert p["v_type"] == "stenosis"
    assert p["path_loc"] == 2
    geom = compute_geometry_from_params(p, gen_cfg)
    widths = np.linalg.norm(geom.top_coords - geom.bot_coords, axis=1)
    peak_lumen = float(np.min(widths))
    nominal = float(p["width"])
    occlusion = 1.0 - (peak_lumen / nominal)
    assert occlusion == pytest.approx(cfg.max_stenosis_diameter_occlusion, abs=0.03)


def test_sample_params_max_aneurysm_uses_config_cap():
    cfg = VesselConfig(phase="biochem")
    rng = np.random.default_rng(1)
    p = _sample_params(0, 2, cfg, rng, pathology_mode="max_aneurysm")
    assert p["v_type"] == "aneurysm"
    offsets = np.asarray(p["offsets"], dtype=float)
    width = float(p["width"])
    expected_peak = cfg.max_aneurysm_wall_offset(width, pro_thrombotic=True)
    assert float(np.max(offsets)) == pytest.approx(expected_peak, rel=0.02)


def test_max_aneurysm_factor_is_double_nominal_cap():
    cfg = VesselConfig(phase="biochem")
    assert cfg.max_aneurysm_factor == pytest.approx(2.0 * cfg.aneurysm_factor_max)


def test_stenosis_wall_offset_for_occlusion_math():
    cfg = VesselConfig(phase="kinematics")
    width = 0.01
    mag = cfg.max_stenosis_wall_offset(width)
    assert mag == pytest.approx(-0.00375)
    assert width + 2.0 * mag == pytest.approx(0.25 * width)
    assert stenosis_wall_offset_for_occlusion(width, cfg) == mag
