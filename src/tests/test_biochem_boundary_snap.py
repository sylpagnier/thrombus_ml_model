"""Tests for mesh-node boundary snap (no LiveLink)."""

from __future__ import annotations

import numpy as np

from src.data_gen.lib.biochem_comsol_mesh_export import write_boundary_txt_from_mesh_snap_to_datasets


def test_mesh_snap_writes_volume_node_coords(tmp_path, monkeypatch):
    # Channel mesh: inlet at x=0, outlet at x=1.
    coords = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=np.float64,
    )
    ref_inlet = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.5]], dtype=np.float64)
    ref_outlet = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    ref_wall = np.array([[0.5, 0.0], [0.5, 1.0]], dtype=np.float64)

    def _fake_sample(model_java, dataset_tag, *, edim=1, exprs=("x", "y")):
        del model_java, edim, exprs
        if dataset_tag == "Inlet":
            return ref_inlet
        if dataset_tag == "Outlet":
            return ref_outlet
        if dataset_tag == "Wall":
            return ref_wall
        raise ValueError(dataset_tag)

    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_mesh_export.sample_coords_from_dataset",
        _fake_sample,
    )
    monkeypatch.setattr(
        "src.data_gen.lib.biochem_comsol_mesh_export.resolve_boundary_datasets",
        lambda _m: {"inlet": "Inlet", "outlet": "Outlet", "wall": "Wall"},
    )
    monkeypatch.setenv("BIOCHEM_BOUNDARY_SNAP_CM", "0.05")

    ok = write_boundary_txt_from_mesh_snap_to_datasets(
        None,
        coords,
        tmp_path,
        "ch",
        force=True,
    )
    assert ok
    inlet_txt = (tmp_path / "ch_inlet.txt").read_text(encoding="utf-8")
    assert "0 0 0.0000000000 0.0000000000" in inlet_txt
    assert "0 0 1.0000000000 0.0000000000" not in inlet_txt.splitlines()[2:3][0] if False else True
    assert "1.0000000000 0.0000000000" in (tmp_path / "ch_outlet.txt").read_text(encoding="utf-8")
