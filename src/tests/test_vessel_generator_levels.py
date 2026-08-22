import numpy as np
import pytest

from src.data_gen.lib.vessel_generator import (
    _sample_params,
    cohort_levels,
    default_level_mix,
    normalize_aneurysm_wall_mode,
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
    assert normalize_pathology_mode("straight-max") == "straight_max"
    assert normalize_pathology_mode("max_straight") == "straight_max"


def test_normalize_aneurysm_wall_mode_aliases():
    assert normalize_aneurysm_wall_mode(None) == "one"
    assert normalize_aneurysm_wall_mode("both") == "mirrored"
    assert normalize_aneurysm_wall_mode("one-wall") == "one"
    assert normalize_aneurysm_wall_mode("single") == "one"


def _gen_cfg(cfg: VesselConfig) -> dict:
    return {
        "num_ctrl_pts": cfg.num_ctrl_pts,
        "base_length": cfg.base_length,
        "min_lumen_width_fraction": cfg.min_lumen_width_fraction,
        "unit": "m",
    }


def test_sample_params_max_stenosis_targets_occlusion():
    cfg = VesselConfig(phase="biochem")
    rng = np.random.default_rng(0)
    p = _sample_params(0, 1, cfg, rng, pathology_mode="max_stenosis")
    assert p["v_type"] == "stenosis"
    assert p["path_loc"] == 2
    geom = compute_geometry_from_params(p, _gen_cfg(cfg))
    widths = np.linalg.norm(geom.top_coords - geom.bot_coords, axis=1)
    peak_lumen = float(np.min(widths))
    nominal = float(p["width"])
    occlusion = 1.0 - (peak_lumen / nominal)
    assert occlusion == pytest.approx(cfg.max_stenosis_diameter_occlusion, abs=0.03)


def test_sample_params_max_aneurysm_mirrored_uses_config_cap():
    cfg = VesselConfig(phase="biochem")
    rng = np.random.default_rng(1)
    p = _sample_params(
        0, 2, cfg, rng, pathology_mode="max_aneurysm", aneurysm_wall_mode="mirrored"
    )
    assert p["v_type"] == "aneurysm"
    assert p["path_loc"] == 2
    assert p["aneurysm_wall_mode"] == "mirrored"
    offsets = np.asarray(p["offsets"], dtype=float)
    width = float(p["width"])
    expected_peak = cfg.max_aneurysm_wall_offset(width, pro_thrombotic=True)
    assert float(np.max(offsets)) == pytest.approx(expected_peak, rel=0.02)
    geom = compute_geometry_from_params(p, _gen_cfg(cfg))
    widths = np.linalg.norm(geom.top_coords - geom.bot_coords, axis=1)
    peak_lumen = float(np.max(widths))
    assert peak_lumen / width == pytest.approx(cfg.max_aneurysm_width_scale, rel=0.02)


def test_sample_params_max_aneurysm_one_wall_is_default():
    cfg = VesselConfig(phase="biochem")
    rng = np.random.default_rng(5)
    seen_walls = set()
    for i in range(30):
        p = _sample_params(i, 1, cfg, rng, pathology_mode="max_aneurysm")
        assert p["v_type"] == "aneurysm"
        assert p["aneurysm_wall_mode"] == "one"
        assert p["path_loc"] in (0, 1)
        seen_walls.add(p["path_loc"])
        width = float(p["width"])
        offsets = np.asarray(p["offsets"], dtype=float)
        expected_peak = cfg.max_aneurysm_wall_offset(width)
        assert float(np.max(offsets)) == pytest.approx(expected_peak, rel=0.02)
        # One-wall: only one side gets the offset -> peak lumen = inlet + max factor*inlet.
        top = offsets if p["path_loc"] in (0, 2) else np.zeros_like(offsets)
        bot = offsets if p["path_loc"] in (1, 2) else np.zeros_like(offsets)
        peak_lumen = float(np.max(width + top + bot))
        assert peak_lumen / width == pytest.approx(1.0 + float(cfg.max_aneurysm_factor), rel=0.02)
    assert seen_walls == {0, 1}


def test_sample_params_straight_max_one_wall_aneurysm():
    cfg = VesselConfig(phase="biochem")
    rng = np.random.default_rng(9)
    aneur_count = 0
    for i in range(60):
        p = _sample_params(i, 1, cfg, rng, pathology_mode="straight_max")
        assert p["curve_type"] == "straight"
        if p["v_type"] == "aneurysm":
            aneur_count += 1
            assert p["path_loc"] in (0, 1)
            assert p["aneurysm_wall_mode"] == "one"
        else:
            assert p["path_loc"] == 2  # max stenosis stays both-wall
    assert aneur_count >= 1


def test_max_aneurysm_factor_targets_triple_inlet_width():
    cfg = VesselConfig(phase="biochem")
    assert cfg.max_aneurysm_factor == pytest.approx(1.0)
    assert cfg.max_aneurysm_width_scale == pytest.approx(3.0)
    assert cfg.aneurysm_factor_max == pytest.approx(cfg.max_aneurysm_factor)


def test_stenosis_wall_offset_for_occlusion_math():
    cfg = VesselConfig(phase="kinematics")
    width = 0.01
    mag = cfg.max_stenosis_wall_offset(width)
    assert cfg.max_stenosis_diameter_occlusion == pytest.approx(0.80)
    assert mag == pytest.approx(-0.004)
    assert width + 2.0 * mag == pytest.approx(0.20 * width)
    assert stenosis_wall_offset_for_occlusion(width, cfg) == mag


def test_sample_params_straight_max_is_straight_extreme():
    cfg = VesselConfig(phase="biochem")
    rng = np.random.default_rng(3)
    seen = set()
    for i in range(40):
        p = _sample_params(i, 1, cfg, rng, pathology_mode="straight_max")
        assert p["curve_type"] == "straight"
        assert p["angle_span"] == 0.0
        assert p["amplitude"] == 0.0
        assert p["v_type"] in ("stenosis", "aneurysm")
        assert all(abs(x) < 1e-15 for x in p["tortuosity"])
        assert all(abs(x) < 1e-15 for x in p["noise_top"])
        assert all(abs(x) < 1e-15 for x in p["noise_bot"])
        seen.add(p["v_type"])
        nominal = float(p["width"])
        offsets = np.asarray(p["offsets"], dtype=float)
        if p["v_type"] == "stenosis":
            assert p["path_loc"] == 2
            geom = compute_geometry_from_params(p, _gen_cfg(cfg))
            widths = np.linalg.norm(geom.top_coords - geom.bot_coords, axis=1)
            occlusion = 1.0 - (float(np.min(widths)) / nominal)
            assert occlusion == pytest.approx(cfg.max_stenosis_diameter_occlusion, abs=0.03)
        else:
            assert p["path_loc"] in (0, 1)
            assert p["aneurysm_wall_mode"] == "one"
            top = offsets if p["path_loc"] in (0, 2) else np.zeros_like(offsets)
            bot = offsets if p["path_loc"] in (1, 2) else np.zeros_like(offsets)
            peak_lumen = float(np.max(nominal + top + bot))
            assert peak_lumen / nominal == pytest.approx(
                1.0 + float(cfg.max_aneurysm_factor), rel=0.02
            )
    assert seen == {"stenosis", "aneurysm"}


def test_random_sampling_can_reach_configured_maxes():
    cfg = VesselConfig(phase="kinematics", pathology_max_hit_prob=1.0)
    rng = np.random.default_rng(11)
    stenosis_hits = 0
    aneurysm_hits = 0
    for i in range(80):
        p = _sample_params(i, 0, cfg, rng)
        if p["v_type"] == "stenosis":
            geom = compute_geometry_from_params(p, _gen_cfg(cfg))
            widths = np.linalg.norm(geom.top_coords - geom.bot_coords, axis=1)
            occlusion = 1.0 - (float(np.min(widths)) / float(p["width"]))
            if abs(occlusion - cfg.max_stenosis_diameter_occlusion) < 0.03:
                stenosis_hits += 1
        elif p["v_type"] == "aneurysm":
            geom = compute_geometry_from_params(p, _gen_cfg(cfg))
            widths = np.linalg.norm(geom.top_coords - geom.bot_coords, axis=1)
            scale = float(np.max(widths)) / float(p["width"])
            if abs(scale - cfg.max_aneurysm_width_scale) < 0.05:
                aneurysm_hits += 1
    assert stenosis_hits >= 1
    assert aneurysm_hits >= 1


def test_wound_at_pathology_centers_on_stenosis_peak():
    cfg = VesselConfig(phase="biochem", wound_pathology_jitter_frac=0.0)
    rng = np.random.default_rng(0)
    for i in range(16):
        p = _sample_params(
            i,
            1,
            cfg,
            rng,
            pathology_mode="max_stenosis",
            wound_probability=1.0,
            wound_at_pathology=True,
        )
        assert p["v_type"] == "stenosis"
        assert p["wound_at_pathology"] is True
        peak = p["pathology_peak_frac"]
        assert peak is not None
        assert len(p["wound_sites"]) == 1
        assert p["wound_sites"][0]["center_frac"] == pytest.approx(peak, abs=1e-9)
        geom = compute_geometry_from_params(p, _gen_cfg(cfg))
        assert geom.meta["pathology_peak_frac"] == pytest.approx(peak)
        assert geom.meta["wound_at_pathology"] is True


def test_wound_at_pathology_centers_on_aneurysm_peak():
    cfg = VesselConfig(phase="biochem", wound_pathology_jitter_frac=0.0)
    rng = np.random.default_rng(4)
    p = _sample_params(
        0,
        1,
        cfg,
        rng,
        pathology_mode="max_aneurysm",
        wound_probability=1.0,
        wound_at_pathology=True,
    )
    assert p["v_type"] == "aneurysm"
    assert p["wound_sites"][0]["center_frac"] == pytest.approx(
        p["pathology_peak_frac"], abs=1e-9
    )


def test_wound_at_pathology_stays_within_jitter():
    jitter = 0.04
    cfg = VesselConfig(phase="biochem", wound_pathology_jitter_frac=jitter)
    rng = np.random.default_rng(8)
    for i in range(24):
        p = _sample_params(
            i,
            1,
            cfg,
            rng,
            pathology_mode="max_stenosis",
            wound_probability=1.0,
            wound_at_pathology=True,
        )
        center = p["wound_sites"][0]["center_frac"]
        peak = p["pathology_peak_frac"]
        assert abs(center - peak) <= jitter + 1e-9


def test_wound_at_pathology_falls_back_on_straight():
    cfg = VesselConfig(phase="kinematics", wound_pathology_jitter_frac=0.0)
    rng = np.random.default_rng(1)
    found = 0
    lo, hi = cfg.wound_center_frac_range
    for i in range(200):
        p = _sample_params(
            i, 0, cfg, rng, wound_probability=1.0, wound_at_pathology=True
        )
        if p["v_type"] != "straight":
            continue
        found += 1
        assert p["pathology_peak_frac"] is None
        assert len(p["wound_sites"]) == 1
        center = p["wound_sites"][0]["center_frac"]
        assert lo - 0.05 <= center <= hi + 0.05
    assert found >= 5


def test_wound_random_placement_is_not_forced_to_peak():
    cfg = VesselConfig(phase="biochem")
    rng = np.random.default_rng(2)
    far = 0
    for i in range(40):
        p = _sample_params(
            i,
            1,
            cfg,
            rng,
            pathology_mode="max_stenosis",
            wound_probability=1.0,
            wound_at_pathology=False,
        )
        d = abs(p["wound_sites"][0]["center_frac"] - p["pathology_peak_frac"])
        if d > 0.1:
            far += 1
    assert far >= 1


def test_wound_at_pathology_without_wounds_is_a_noop():
    cfg = VesselConfig(phase="biochem")
    rng = np.random.default_rng(3)
    p = _sample_params(
        0,
        1,
        cfg,
        rng,
        pathology_mode="max_stenosis",
        wound_probability=0.0,
        wound_at_pathology=True,
    )
    assert p["wound_sites"] == []
    assert p["pathology_peak_frac"] is not None
