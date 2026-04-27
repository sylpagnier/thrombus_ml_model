"""Inventory helpers for ``anchor_generator`` and ``vessel_generator`` (mesh/NPZ paths, next index)."""

from src.config import PhysicsConfig
from src.data_gen.pipeline_kinematics import _purge_anchor_npz_outputs
from src.data_gen.lib.anchor_generator import AnchorGenerator
from src.data_gen.lib.anchor_generator import (
    list_anchor_candidate_json_paths,
    summarize_anchor_inventory,
)
from src.data_gen.lib.mesh_to_graph import MeshToGraph
from src.data_gen.lib.vessel_generator import summarize_vessel_mesh_inventory


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


def test_kinematics_physicsconfig_rheology_mode_switches():
    newt = PhysicsConfig(phase="kinematics", rheology="newtonian")
    carr = PhysicsConfig(phase="kinematics", rheology="carreau")
    assert newt.viscosity_model == "newtonian"
    assert newt.n == 1.0
    assert carr.viscosity_model == "carreau"
    assert carr.n < 1.0


def test_anchor_generator_infers_rheology_from_output_dir(tmp_path):
    out_newt = tmp_path / "cfd_results_kinematics" / "newtonian"
    out_carr = tmp_path / "cfd_results_kinematics" / "carreau"
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir(parents=True)
    out_newt.mkdir(parents=True)
    out_carr.mkdir(parents=True)
    (tmp_path / "template.mph").write_text("dummy")

    a_newt = AnchorGenerator(
        phase="kinematics",
        output_dir=out_newt,
        mesh_dir=mesh_dir,
        template_path=tmp_path / "template.mph",
    )
    a_carr = AnchorGenerator(
        phase="kinematics",
        output_dir=out_carr,
        mesh_dir=mesh_dir,
        template_path=tmp_path / "template.mph",
    )
    assert a_newt.phys_cfg.viscosity_model == "newtonian"
    assert a_carr.phys_cfg.viscosity_model == "carreau"


def test_mesh_to_graph_infers_rheology_from_subdir():
    g_newt = MeshToGraph(phase="kinematics", n_subdir="newtonian")
    g_carr = MeshToGraph(phase="kinematics", n_subdir="carreau")
    assert g_newt.phys_cfg.viscosity_model == "newtonian"
    assert g_carr.phys_cfg.viscosity_model == "carreau"


def test_purge_anchor_npz_outputs_removes_only_vessel_npz(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "vessel_1.npz").write_bytes(b"x")
    (out / "vessel_2.npz").write_bytes(b"x")
    (out / "keep.txt").write_text("keep")

    removed = _purge_anchor_npz_outputs(out)

    assert removed == 2
    assert not (out / "vessel_1.npz").exists()
    assert not (out / "vessel_2.npz").exists()
    assert (out / "keep.txt").exists()
