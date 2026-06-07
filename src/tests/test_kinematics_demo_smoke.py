"""Smoke tests for kinematics flow demo (no GUI display)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.config import NodeFeat, VesselConfig
from src.data_gen.lib.mesh_to_graph import MeshToGraph
from src.data_gen.lib.vessel_generator import build_vessel_mesh, make_vessel_params
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import resolve_checkpoint

_REPO = Path(__file__).resolve().parents[2]


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _gmsh_available(), reason="gmsh not installed")
def test_build_vessel_mesh_writes_sidecar(tmp_path):
    cfg = VesselConfig(phase="kinematics")
    gen_cfg = {
        "num_ctrl_pts": cfg.num_ctrl_pts,
        "base_length": cfg.base_length,
        "mesh_lc": cfg.mesh_lc,
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
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(0))
    idx, ok, err = build_vessel_mesh(params, gen_cfg, tmp_path)
    assert ok, err
    assert (tmp_path / f"vessel_{idx}.msh").exists()
    meta_path = tmp_path / f"vessel_{idx}.json"
    assert meta_path.exists()
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "centerline_pts" in meta
    assert len(meta["centerline_pts"]) >= 2


@pytest.mark.skipif(not _gmsh_available(), reason="gmsh not installed")
def test_process_mesh_minimal_contract(tmp_path):
    import json
    import meshio

    cfg = VesselConfig(phase="kinematics")
    gen_cfg = {
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
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(1))
    idx, ok, err = build_vessel_mesh(params, gen_cfg, tmp_path)
    assert ok, err
    mesh = meshio.read(tmp_path / f"vessel_{idx}.msh")
    meta = json.loads((tmp_path / f"vessel_{idx}.json").read_text(encoding="utf-8"))

    builder = MeshToGraph(phase="kinematics", rheology="carreau", proc_dir=tmp_path / "graphs")
    data = builder.process_mesh(mesh, meta, stem="demo")
    assert data is not None
    assert data.x.shape[1] == NodeFeat.WIDTH_D2.stop
    assert hasattr(data, "mask_inlet") and hasattr(data, "mask_wall")
    assert data.G_x.is_sparse and data.G_y.is_sparse


def test_load_predictor_smoke():
    ckpt_env = os.environ.get("KIN_DEMO_CKPT")
    if ckpt_env:
        ckpt = Path(ckpt_env)
    else:
        try:
            ckpt = resolve_kinematics_checkpoint()
        except FileNotFoundError:
            pytest.skip("no kinematics checkpoint on disk")
    if not ckpt.exists():
        pytest.skip(f"checkpoint missing: {ckpt}")
    model = load_kinematics_predictor(ckpt, "cpu", max_iters=5)
    assert model is not None


def test_demo_no_gui_runs(tmp_path):
    ckpt_env = os.environ.get("KIN_DEMO_CKPT")
    if ckpt_env:
        ckpt = Path(ckpt_env)
    else:
        candidate = resolve_checkpoint("a", "kinematics_best.pth")
        if not candidate.exists():
            pytest.skip("no kinematics checkpoint for --no-gui smoke")
        ckpt = candidate

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    cmd = [
        sys.executable,
        "-m",
        "src.tools.demo_kinematics_flow",
        "--no-gui",
        "--mesh-coarse",
        "--max-iters",
        "8",
        "--checkpoint",
        str(ckpt),
    ]
    if not _gmsh_available():
        pytest.skip("gmsh not installed")
    proc = subprocess.run(cmd, cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(not _gmsh_available(), reason="gmsh not installed")
def test_demo_edited_walls_no_gui(tmp_path):
    import json

    from src.data_gen.lib.vessel_geometry import compute_geometry_from_params, geometry_to_params_override

    ckpt_env = os.environ.get("KIN_DEMO_CKPT")
    if ckpt_env:
        ckpt = Path(ckpt_env)
    else:
        candidate = resolve_checkpoint("a", "kinematics_best.pth")
        if not candidate.exists():
            pytest.skip("no kinematics checkpoint for edited-walls smoke")
        ckpt = candidate

    cfg = VesselConfig(phase="kinematics")
    gen_cfg = {
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
    params = make_vessel_params(idx=0, level=0, cfg=cfg, rng=np.random.default_rng(7))
    geom = compute_geometry_from_params(params, gen_cfg)
    fixture = tmp_path / "edited_walls.json"
    payload = geometry_to_params_override(geom)
    payload["unit"] = "m"
    fixture.write_text(json.dumps(payload), encoding="utf-8")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    cmd = [
        sys.executable,
        "-m",
        "src.tools.demo_kinematics_flow",
        "--no-gui",
        "--mesh-coarse",
        "--max-iters",
        "8",
        "--geometry-mode",
        "edited_walls",
        "--load-geometry",
        str(fixture),
        "--checkpoint",
        str(ckpt),
    ]
    proc = subprocess.run(cmd, cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "[OK]" in proc.stdout
