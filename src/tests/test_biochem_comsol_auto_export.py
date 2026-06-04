"""Tests for biochem COMSOL auto-export helpers (no LiveLink required)."""

from __future__ import annotations

import numpy as np

from src.data_gen.lib.biochem_comsol_auto_export import (
    DOMAIN_FIELD_NAMES,
    phase2_nowound_mph_name_for_stem,
    resolve_biochem_comsol_model_path,
    write_boundary_txt_from_mesh,
    write_wide_domain_txt,
)
from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor


def test_phase2_nowound_mph_name_for_patient_stem():
    assert phase2_nowound_mph_name_for_stem("patient007") == "phase2_nowound_007.mph"
    assert phase2_nowound_mph_name_for_stem("patient7") == "phase2_nowound_007.mph"
    assert phase2_nowound_mph_name_for_stem("vessel_001") is None


def test_resolve_patient_stem_to_phase2_nowound_mph(tmp_path, monkeypatch):
    models = tmp_path / "comsol_models"
    models.mkdir()
    (models / "phase2_nowound_003.mph").write_bytes(b"stub")
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_auto_export.comsol_models_dir",
        lambda: models,
    )
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_auto_export.data_root",
        lambda: tmp_path,
    )
    got = resolve_biochem_comsol_model_path("patient003")
    assert got == (models / "phase2_nowound_003.mph").resolve()
    assert resolve_biochem_comsol_model_path("patient999") is None


def test_write_wide_domain_txt_roundtrip_with_extractor(tmp_path):
    stem = "stub"
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float64)
    times = [0.0, 10.0]
    rng = np.random.default_rng(1)
    fields_by_time = {
        float(t): rng.standard_normal((coords.shape[0], len(DOMAIN_FIELD_NAMES)))
        for t in times
    }
    fp = tmp_path / f"{stem}.txt"
    write_wide_domain_txt(fp, times_s=times, coords_xy_cm=coords, fields_by_time=fields_by_time)

    ext = PatientDataExtractor(phase="biochem_anchors", raw_dir=tmp_path, label_dir=tmp_path, proc_dir=tmp_path)
    blocks = ext.load_comsol_trajectory(fp)
    assert set(blocks.keys()) == {0.0, 10.0}
    for t in times:
        df = blocks[float(t)]
        assert list(df.columns)[:6] == ["x", "y", "u", "v", "p", "mu_effective"]
        np.testing.assert_allclose(df["x"].values, coords[:, 0], rtol=0, atol=1e-6)


def test_write_boundary_txt_from_mesh_minimal_square(tmp_path):
    import meshio

    # Unit square with tagged edges (Gmsh 2 format).
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    line_tags = np.array([101, 102, 103, 103], dtype=np.int32)
    tri_tags = np.array([201, 201], dtype=np.int32)
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", tri), ("line", lines)],
        cell_data={"gmsh:physical": [tri_tags, line_tags]},
    )
    msh = tmp_path / "sq.msh"
    meshio.write(msh, mesh, file_format="gmsh22", binary=False)

    inlet_p, outlet_p, wall_p = write_boundary_txt_from_mesh(msh, tmp_path, "sq")
    assert inlet_p.is_file() and outlet_p.is_file() and wall_p.is_file()
    assert "0 0" in inlet_p.read_text(encoding="utf-8")
