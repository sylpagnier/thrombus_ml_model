"""Tests for biochem COMSOL auto-export helpers (no LiveLink required)."""

from __future__ import annotations

import numpy as np

from src.data_gen.lib.biochem_comsol_auto_export import (
    DOMAIN_FIELD_NAMES,
    apply_variant_to_stem,
    collect_biochem_extract_stems,
    parse_biochem_extract_stem,
    patient_stem_from_phase2_mph,
    phase2_mph_name_for_stem,
    phase2_wound_mph_name_for_stem,
    resolve_biochem_comsol_model_path,
    resolve_stem_selection,
    stems_from_phase2_mph,
    stems_from_phase2_nowound_mph,
    stems_from_phase2_wound_mph,
    write_boundary_txt_from_mesh,
    write_wide_domain_txt,
)
from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor


def test_phase2_wound_mph_name_for_patient_stem():
    assert phase2_wound_mph_name_for_stem("patient007") == "phase2_wound_007.mph"
    assert phase2_wound_mph_name_for_stem("patient7") == "phase2_wound_007.mph"
    assert phase2_wound_mph_name_for_stem("wound_patient007") == "phase2_wound_007.mph"
    assert phase2_wound_mph_name_for_stem("vessel_001") is None


def test_parse_biochem_extract_stem_aliases():
    nowound = parse_biochem_extract_stem("patient007")
    assert nowound is not None
    assert nowound.stem == "patient007"
    assert nowound.variant == "nowound"
    assert nowound.mph_name == "phase2_nowound_007.mph"
    assert parse_biochem_extract_stem("patient007_nowound").stem == "patient007"

    wound = parse_biochem_extract_stem("wound_patient007")
    assert wound is not None
    assert wound.stem == "wound_patient007"
    assert wound.variant == "wound"
    assert wound.mph_name == "phase2_wound_007.mph"
    assert parse_biochem_extract_stem("patient007_wound").stem == "wound_patient007"
    assert apply_variant_to_stem("patient007", "wound") == "wound_patient007"
    assert phase2_mph_name_for_stem("patient007") == "phase2_nowound_007.mph"
    assert phase2_mph_name_for_stem("wound_patient007") == "phase2_wound_007.mph"


def test_stems_from_phase2_mph_keeps_variants_apart(tmp_path, monkeypatch):
    models = tmp_path / "comsol_models"
    models.mkdir()
    (models / "phase2_wound_008.mph").write_bytes(b"a")
    (models / "phase2_nowound_008.mph").write_bytes(b"c")
    (models / "phase2_template_nowound.mph").write_bytes(b"b")
    (models / "phase2_template_wound.mph").write_bytes(b"d")
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_auto_export.comsol_models_dir",
        lambda: models,
    )
    assert stems_from_phase2_wound_mph() == ["wound_patient008"]
    assert stems_from_phase2_nowound_mph() == ["patient008"]
    assert stems_from_phase2_mph() == ["patient008", "wound_patient008"]
    assert patient_stem_from_phase2_mph(models / "phase2_wound_011.mph") == "wound_patient011"
    assert patient_stem_from_phase2_mph(models / "phase2_nowound_011.mph") == "patient011"
    assert patient_stem_from_phase2_mph(models / "phase2_template_wound.mph") is None


def test_resolve_patient_stem_does_not_cross_variants(tmp_path, monkeypatch):
    models = tmp_path / "comsol_models"
    models.mkdir()
    (models / "phase2_wound_003.mph").write_bytes(b"stub")
    (models / "phase2_nowound_004.mph").write_bytes(b"stub")
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_auto_export.comsol_models_dir",
        lambda: models,
    )
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_auto_export.data_root",
        lambda: tmp_path,
    )
    assert resolve_biochem_comsol_model_path("patient003") is None
    assert resolve_biochem_comsol_model_path("wound_patient003") == (
        models / "phase2_wound_003.mph"
    ).resolve()
    assert resolve_biochem_comsol_model_path("patient003_wound") == (
        models / "phase2_wound_003.mph"
    ).resolve()
    assert resolve_biochem_comsol_model_path("patient004") == (
        models / "phase2_nowound_004.mph"
    ).resolve()
    assert resolve_biochem_comsol_model_path("wound_patient004") is None
    assert resolve_biochem_comsol_model_path("patient999") is None


def test_collect_stems_keeps_domain_wound_files_apart(tmp_path, monkeypatch):
    models = tmp_path / "comsol_models"
    models.mkdir()
    (models / "phase2_nowound_007.mph").write_bytes(b"a")
    (models / "phase2_wound_007.mph").write_bytes(b"b")
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_auto_export.comsol_models_dir",
        lambda: models,
    )
    raw = tmp_path / "raw"
    label = tmp_path / "label"
    raw.mkdir()
    label.mkdir()
    (label / "patient007.txt").write_text("domain\n", encoding="utf-8")
    (label / "patient007_wall.txt").write_text("wall\n", encoding="utf-8")
    (label / "wound_patient007.txt").write_text("domain\n", encoding="utf-8")
    (label / "wound_patient007_wound.txt").write_text("wound-bc\n", encoding="utf-8")
    stems = collect_biochem_extract_stems(raw, label)
    assert "patient007" in stems
    assert "wound_patient007" in stems
    assert "patient007_wall" not in stems
    assert "wound_patient007_wound" not in stems


def test_resolve_stem_selection_accepts_wound_aliases():
    table = ["patient005", "wound_patient007", "patient008"]
    assert resolve_stem_selection("patient007_wound", table) == ["wound_patient007"]
    assert resolve_stem_selection("patient007", table, variant="wound") == ["wound_patient007"]
    assert resolve_stem_selection("2", table) == ["wound_patient007"]


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
    assert not (tmp_path / "sq_wound.txt").is_file()


def test_write_boundary_txt_from_mesh_writes_optional_wound(tmp_path):
    import meshio

    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    line_tags = np.array([101, 102, 104, 103], dtype=np.int32)
    tri_tags = np.array([201, 201], dtype=np.int32)
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", tri), ("line", lines)],
        cell_data={"gmsh:physical": [tri_tags, line_tags]},
    )
    msh = tmp_path / "sqw.msh"
    meshio.write(msh, mesh, file_format="gmsh22", binary=False)
    write_boundary_txt_from_mesh(msh, tmp_path, "sqw")
    wound_p = tmp_path / "sqw_wound.txt"
    assert wound_p.is_file()
    assert "0 0" in wound_p.read_text(encoding="utf-8")


def test_write_boundary_txt_from_mesh_writes_wound_even_if_wall_exists(tmp_path):
    import meshio

    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    line_tags = np.array([101, 102, 104, 103], dtype=np.int32)
    tri_tags = np.array([201, 201], dtype=np.int32)
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", tri), ("line", lines)],
        cell_data={"gmsh:physical": [tri_tags, line_tags]},
    )
    msh = tmp_path / "sqw2.msh"
    meshio.write(msh, mesh, file_format="gmsh22", binary=False)
    (tmp_path / "sqw2_inlet.txt").write_text("% x  y\n0 0 0.0 0.0\n", encoding="utf-8")
    (tmp_path / "sqw2_outlet.txt").write_text("% x  y\n0 0 1.0 0.0\n", encoding="utf-8")
    (tmp_path / "sqw2_wall.txt").write_text("% x  y\n0 0 0.0 1.0\n", encoding="utf-8")
    write_boundary_txt_from_mesh(msh, tmp_path, "sqw2")
    assert (tmp_path / "sqw2_wound.txt").is_file()
    # Did not clobber the existing wall file.
    assert "0.0 1.0" in (tmp_path / "sqw2_wall.txt").read_text(encoding="utf-8")


def test_ensure_boundary_txt_files_writes_wound_when_wall_already_exists(tmp_path):
    import meshio
    import numpy as np

    from src.config import VesselConfig
    from src.data_gen.lib.biochem_comsol_mesh_export import ensure_boundary_txt_files

    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    tri = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    line_tags = np.array([101, 102, 104, 103], dtype=np.int32)
    tri_tags = np.array([201, 201], dtype=np.int32)
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", tri), ("line", lines)],
        cell_data={"gmsh:physical": [tri_tags, line_tags]},
    )
    msh = tmp_path / "sqw3.msh"
    meshio.write(msh, mesh, file_format="gmsh22", binary=False)
    for name in ("inlet", "outlet", "wall"):
        (tmp_path / f"sqw3_{name}.txt").write_text("% x  y\n0 0 0.0 0.0\n", encoding="utf-8")
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    ensure_boundary_txt_files(
        None,
        coords,
        msh,
        tmp_path,
        "sqw3",
        vessel_cfg=VesselConfig(phase="biochem_anchors"),
        force_boundary=False,
    )
    assert (tmp_path / "sqw3_wound.txt").is_file()


def test_ensure_wound_from_comsol_selection_mask(tmp_path, monkeypatch):
    import numpy as np

    from src.data_gen.lib.biochem_comsol_mesh_export import ensure_wound_boundary_txt

    coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float64)

    def _fake_candidates(_model, bname):
        assert bname == "wound"
        return ["wound(x,y)"]

    def _fake_eval(_model, coords_cm, expr, *, dataset_tag="dset1"):
        del _model, expr, dataset_tag
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_mesh_export.boundary_mask_expr_candidates",
        _fake_candidates,
    )
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_mesh_export._evaluate_boundary_mask",
        _fake_eval,
    )
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_mesh_export.resolve_boundary_datasets",
        lambda _m: {},
    )
    ok = ensure_wound_boundary_txt(
        object(),
        coords,
        None,
        tmp_path,
        "wound_patient001",
        force=True,
    )
    assert ok
    text = (tmp_path / "wound_patient001_wound.txt").read_text(encoding="utf-8")
    assert "0.5000000000 0.0000000000" in text
    assert "1.0000000000 0.0000000000" not in text


def test_ensure_wound_from_geometry_selection_snap(tmp_path, monkeypatch):
    import numpy as np

    from src.data_gen.lib.biochem_comsol_mesh_export import ensure_wound_boundary_txt

    coords = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.5, 0.5]],
        dtype=np.float64,
    )
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_mesh_export.resolve_boundary_datasets",
        lambda _m: {},
    )
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_mesh_export.discover_boundary_selection_tags",
        lambda _m: {"wound": "sel1"},
    )
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_mesh_export.sample_coords_from_named_selection",
        lambda _m, _tag, *, parent_dataset="dset1": np.array([[0.50, 0.0], [0.49, 0.0]]),
    )
    monkeypatch.setenv("BIOCHEM_BOUNDARY_SNAP_CM", "0.05")
    ok = ensure_wound_boundary_txt(
        object(),
        coords,
        None,
        tmp_path,
        "wound_patient001",
        force=True,
    )
    assert ok
    text = (tmp_path / "wound_patient001_wound.txt").read_text(encoding="utf-8")
    assert "selection 'sel1'" in text
    assert "0.5000000000 0.0000000000" in text
    assert "1.0000000000 0.0000000000" not in text
    assert "0.5000000000 0.5000000000" not in text


def test_wound_snap_retries_meter_to_cm(tmp_path, monkeypatch):
    import numpy as np

    from src.data_gen.lib.biochem_comsol_mesh_export import _write_wound_snap_txt

    pts = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    ref_m = np.array([[0.02, 0.0]], dtype=np.float64)
    monkeypatch.setenv("BIOCHEM_BOUNDARY_SNAP_CM", "0.05")
    ok = _write_wound_snap_txt(
        tmp_path / "w.txt",
        pts,
        ref_m,
        stem="wound_patient001",
        source="selection 'sel1'",
    )
    assert ok
    text = (tmp_path / "w.txt").read_text(encoding="utf-8")
    assert "*100" in text
    assert "0 0 2.0000000000 0.0000000000" in text
    assert "0 0 0.0000000000 0.0000000000" not in text


def test_boundary_txt_has_coords_ignores_header_only(tmp_path):
    from src.data_gen.lib.biochem_comsol_mesh_export import boundary_txt_has_coords

    empty = tmp_path / "empty_wound.txt"
    empty.write_text("% Model: COMSOL mask (wound(x,y))\n% x  y\n", encoding="utf-8")
    assert boundary_txt_has_coords(empty) is False
    empty.write_text("% x  y\n0 0 1.0 2.0\n", encoding="utf-8")
    assert boundary_txt_has_coords(empty) is True


def test_comsol_steady_kine_builder_does_not_import_removed_gt_prior():
    import inspect

    from src.data_gen.lib.kinematics_graph_builder import build_kinematics_graph_from_comsol_steady

    source = inspect.getsource(build_kinematics_graph_from_comsol_steady)
    assert "apply_gt_flow_priors_to_kine_x" not in source
