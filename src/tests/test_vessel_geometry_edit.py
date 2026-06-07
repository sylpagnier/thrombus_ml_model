"""Tests for vessel wall geometry editing (no GUI)."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from src.config import NodeFeat, VesselConfig
from src.data_gen.lib.vessel_generator import build_vessel_mesh, make_vessel_params
from src.data_gen.lib.vessel_geometry import (
    GeometryValidationError,
    apply_wall_handle_drag,
    compute_geometry_from_params,
    compute_geometry_from_walls,
    default_max_wall_displacement_m,
    geometry_to_params_override,
    relax_wall_coords,
    smooth_wall_curve,
    subsample_handle_indices,
    validate_geometry,
)
from src.utils.vessel_drag_editor import WallControlPointEditor


def _gen_cfg(cfg: VesselConfig | None = None) -> dict:
    cfg = cfg or VesselConfig(phase="kinematics")
    return {
        "num_ctrl_pts": cfg.num_ctrl_pts,
        "base_length": cfg.base_length,
        "mesh_lc": cfg.mesh_lc * 2.0,
        "mesh_size_factor": cfg.mesh_size_factor,
        "width_min": cfg.width_min,
        "width_max": cfg.width_max,
        "stenosis_factor_min": cfg.stenosis_factor_min,
        "stenosis_factor_max": cfg.stenosis_factor_max,
        "min_lumen_width_fraction": cfg.min_lumen_width_fraction,
        "aneurysm_factor_min": cfg.aneurysm_factor_min,
        "aneurysm_factor_max": cfg.aneurysm_factor_max,
        "TAGS": dict(cfg.TAGS),
        "unit": "m",
    }


@pytest.mark.skipif(importlib.util.find_spec("gmsh") is None, reason="gmsh not installed")
def test_compute_from_params_matches_legacy_meta_keys(tmp_path):
    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(0))
    geom = compute_geometry_from_params(params, gen_cfg)
    idx, ok, err = build_vessel_mesh(params, gen_cfg, tmp_path)
    assert ok, err
    legacy = json.loads((tmp_path / f"vessel_{idx}.json").read_text(encoding="utf-8"))
    for key in ("d_bar", "d_inlet", "centerline_pts", "centerline_tangents", "unit"):
        assert key in geom.meta
        assert key in legacy
    assert geom.meta["centerline_pts"]
    assert len(geom.meta["centerline_pts"]) == len(legacy["centerline_pts"])
    assert geom.d_bar == pytest.approx(float(legacy["d_bar"]), rel=1e-6)
    assert geom.d_inlet == pytest.approx(float(legacy["d_inlet"]), rel=1e-6)


def test_smooth_wall_curve_has_more_samples():
    pts = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.005], [0.03, 0.01]], dtype=float)
    dense = smooth_wall_curve(pts, n_dense=80)
    assert dense.shape[0] >= 80
    assert dense.shape[1] == 2


def test_apply_wall_handle_drag_respects_fixed_interior():
    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(9))
    geom = compute_geometry_from_params(params, gen_cfg)
    top0 = geom.top_coords.copy()
    i = 15
    fixed = frozenset({0, geom.n - 1, i})
    top, _ = apply_wall_handle_drag(
        geom.top_coords,
        geom.bot_coords,
        index=i,
        side="top",
        xy=top0[i] + np.array([0.0, 0.005]),
        fixed_top=fixed,
        fixed_bot=frozenset({0, geom.n - 1}),
    )
    assert np.allclose(top, top0)


def test_relax_wall_coords_preserves_fixed():
    pts = np.array([[0.0, 0.0], [0.01, 0.005], [0.02, -0.003], [0.03, 0.0]], dtype=float)
    fixed = frozenset({0, 2, 3})
    out = relax_wall_coords(pts, fixed, n_iters=5, omega=0.5)
    assert np.allclose(out[0], pts[0])
    assert np.allclose(out[2], pts[2])
    assert np.allclose(out[3], pts[3])
    assert out[1, 1] != pts[1, 1]


def test_apply_wall_handle_drag_spreads_neighbors():
    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(8))
    geom = compute_geometry_from_params(params, gen_cfg)
    top0 = geom.top_coords.copy()
    i = 15
    new_xy = top0[i] + np.array([0.0, 0.003])
    top, bot = apply_wall_handle_drag(
        geom.top_coords, geom.bot_coords, index=i, side="top", xy=new_xy, sigma_stations=7.0
    )
    assert top[i, 1] > top0[i, 1]
    assert top[i + 1, 1] > top0[i + 1, 1]
    assert top[0, 0] == pytest.approx(top0[0, 0])
    assert np.allclose(bot, geom.bot_coords)


def test_default_max_wall_displacement_generous():
    cfg = VesselConfig(phase="kinematics")
    lim = default_max_wall_displacement_m(0.005, {"base_length": cfg.base_length})
    assert lim > 0.5 * 0.005


def test_compute_from_walls_midline():
    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(1))
    base = compute_geometry_from_params(params, gen_cfg)
    top = base.top_coords.copy()
    bot = base.bot_coords.copy()
    top[10, 1] += 0.002
    geom = compute_geometry_from_walls(top, bot, idx=0, base_length=cfg.base_length)
    assert geom.top_coords[10, 1] == pytest.approx(top[10, 1])
    assert geom.bot_coords[10, 1] == pytest.approx(bot[10, 1])
    mid = 0.5 * (top + bot)
    assert np.allclose(geom.centerline_pts, mid)
    assert geom.d_bar > 0


def test_validate_rejects_collapsed_lumen():
    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(2))
    geom = compute_geometry_from_params(params, gen_cfg)
    geom.top_coords = geom.bot_coords.copy()
    with pytest.raises(GeometryValidationError, match="narrow|Degenerate"):
        validate_geometry(geom, gen_cfg, reference_width=float(params["width"]))


def test_validate_rejects_outlet_curl():
    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(3))
    geom = compute_geometry_from_params(params, gen_cfg)
    geom.top_coords[-1, 0] = 0.0
    geom.bot_coords[-1, 0] = 0.0
    with pytest.raises(GeometryValidationError, match="L/3"):
        validate_geometry(geom, gen_cfg, reference_width=float(params["width"]))


def test_pinned_ends_unchanged_after_drag():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(4))
    geom = compute_geometry_from_params(params, gen_cfg)
    hi = subsample_handle_indices(geom.n)
    assert 0 not in hi
    assert geom.n - 1 not in hi

    fig, ax = plt.subplots()
    editor = WallControlPointEditor(fig, ax, geom, cfg_dict=gen_cfg, on_change=lambda _g: None)
    top0 = geom.top_coords[0].copy()
    botn = geom.bot_coords[-1].copy()
    editor._apply_drag(0, "top", (999.0, 999.0))
    editor._apply_drag(geom.n - 1, "bot", (999.0, 999.0))
    assert np.allclose(geom.top_coords[0], top0)
    assert np.allclose(geom.bot_coords[-1], botn)
    editor.disconnect()
    plt.close(fig)


def test_subsample_handle_indices_excludes_ends():
    hi = subsample_handle_indices(50, stride=5)
    assert 0 not in hi
    assert 49 not in hi
    assert hi[0] >= 2


@pytest.mark.skipif(importlib.util.find_spec("gmsh") is None, reason="gmsh not installed")
def test_edited_walls_build_produces_msh(tmp_path):
    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(5))
    geom = compute_geometry_from_params(params, gen_cfg)
    override = geometry_to_params_override(geom)
    idx, ok, err = build_vessel_mesh(override, gen_cfg, tmp_path)
    assert ok, err
    assert (tmp_path / f"vessel_{idx}.msh").exists()
    assert (tmp_path / f"vessel_{idx}.json").exists()


@pytest.mark.skipif(importlib.util.find_spec("gmsh") is None, reason="gmsh not installed")
def test_process_mesh_accepts_edited_meta(tmp_path):
    import meshio

    from src.data_gen.lib.mesh_to_graph import MeshToGraph

    cfg = VesselConfig(phase="kinematics")
    gen_cfg = _gen_cfg(cfg)
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(6))
    geom = compute_geometry_from_params(params, gen_cfg)
    override = geometry_to_params_override(geom)
    idx, ok, err = build_vessel_mesh(override, gen_cfg, tmp_path)
    assert ok, err
    mesh = meshio.read(tmp_path / f"vessel_{idx}.msh")
    meta = json.loads((tmp_path / f"vessel_{idx}.json").read_text(encoding="utf-8"))
    builder = MeshToGraph(phase="kinematics", rheology="carreau", proc_dir=tmp_path / "graphs")
    data = builder.process_mesh(mesh, meta, stem="edited")
    assert data is not None
    assert data.x.shape[1] == NodeFeat.WIDTH_D2.stop
