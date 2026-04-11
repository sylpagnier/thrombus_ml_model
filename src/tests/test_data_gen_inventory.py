"""Inventory helpers for ``anchor_generator`` and ``vessel_generator`` (mesh/NPZ paths, next index)."""

from src.data_pipeline.anchor_generator import (
    list_anchor_candidate_json_paths,
    summarize_anchor_inventory,
)
from src.data_pipeline.vessel_generator import summarize_vessel_mesh_inventory


def test_summarize_anchor_inventory(tmp_path):
    mesh = tmp_path / "mesh"
    out = tmp_path / "out"
    mesh.mkdir()
    out.mkdir()
    (mesh / "vessel_1.json").write_text("{}")
    (mesh / "vessel_1.nas").write_text("x" * 10)
    (mesh / "vessel_1.msh").write_text("msh")
    (mesh / "vessel_2.json").write_text("{}")
    (mesh / "vessel_2.nas").write_text("y" * 10)
    (mesh / "vessel_2.msh").write_text("msh")
    (out / "vessel_1.npz").write_bytes(b"fake")
    inv = summarize_anchor_inventory(mesh, out)
    assert inv["existing_npz"] == 1
    assert inv["mesh_json_with_valid_nas"] == 2
    assert inv["pending_missing_npz"] == 1
    assert inv["candidate_pool_ready"] == 1
    assert inv["candidate_pool_including_npz"] == 2


def test_list_anchor_candidate_json_paths_requires_msh(tmp_path):
    mesh = tmp_path / "mesh"
    out = tmp_path / "out"
    mesh.mkdir()
    out.mkdir()
    (mesh / "vessel_0.json").write_text("{}")
    (mesh / "vessel_0.nas").write_text("nas")
    assert list_anchor_candidate_json_paths(mesh, out) == []
    (mesh / "vessel_0.msh").write_text("msh")
    assert len(list_anchor_candidate_json_paths(mesh, out)) == 1
    (out / "vessel_0.npz").write_bytes(b"x")
    assert len(list_anchor_candidate_json_paths(mesh, out)) == 0
    assert len(list_anchor_candidate_json_paths(mesh, out, include_existing_npz=True)) == 1


def test_summarize_vessel_mesh_inventory_empty(tmp_path):
    inv = summarize_vessel_mesh_inventory(tmp_path / "missing")
    assert inv["count"] == 0
    assert inv["next_idx"] == 0


def test_summarize_vessel_mesh_inventory_next_idx(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    (d / "vessel_10.json").write_text("{}")
    (d / "vessel_10.msh").write_text("x")
    inv = summarize_vessel_mesh_inventory(d)
    assert inv["count"] == 1
    assert inv["max_idx"] == 10
    assert inv["next_idx"] == 11
