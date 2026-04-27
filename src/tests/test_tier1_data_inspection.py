"""
Inspection tests for generated Kinematics data (raw meshes, COMSOL ``.npz``, ``graphs_kinematics`` ``.pt``).

These tests are **offline** checks on your ``data/`` tree. They skip with a clear message if
paths are missing or counts are below optional environment thresholds.

Environment (all optional):

* ``KINEMATICS_INSPECT_MIN_MSH`` — minimum distinct vessel indices in ``data/raw/kinematics`` (default ``1``).
* ``KINEMATICS_INSPECT_MAX_GRAPHS`` — max ``vessel_*.pt`` files to fully load per test (default ``400``).
* ``KINEMATICS_INSPECT_MIN_GRAPHS`` — if set, assert at least this many ``.pt`` graphs exist.
* ``KINEMATICS_INSPECT_MIN_NPZ`` — if set, assert at least this many ``vessel_*.npz`` under CFD output.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.config import NodeFeat, VesselConfig
from src.data_gen.lib.anchor_generator import summarize_anchor_inventory
from src.data_gen.lib.vessel_generator import summarize_vessel_mesh_inventory
from src.utils.anchor_mask import graph_has_anchor


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _kinematics_paths() -> Dict[str, Path]:
    cfg = VesselConfig(phase="kinematics")
    return {
        "raw_meshes": cfg.mesh_input_dir,
        "cfd_npz": cfg.output_dir,
        "graphs": cfg.graph_output_dir,
    }


def _skip_if_below_min_msh(inv: Dict[str, Any], min_msh: int) -> None:
    if int(inv.get("count", 0)) < min_msh:
        p = _kinematics_paths()["raw_meshes"]
        pytest.skip(f"Kinematics raw mesh inventory count {inv.get('count')} < {min_msh} ({p}).")


@pytest.fixture(scope="module")
def kinematics_layout() -> Dict[str, Any]:
    p = _kinematics_paths()
    cfg = VesselConfig(phase="kinematics")
    inv_mesh = summarize_vessel_mesh_inventory(p["raw_meshes"])
    inv_anchor = summarize_anchor_inventory(p["raw_meshes"], p["cfd_npz"])
    graph_dir = p["graphs"]
    n_pt = len(list(graph_dir.glob("vessel_*.pt"))) if graph_dir.is_dir() else 0
    return {
        "paths": p,
        "vessel_cfg": cfg,
        "mesh_inventory": inv_mesh,
        "anchor_inventory": inv_anchor,
        "graph_pt_count": n_pt,
    }


def test_kinematics_paths_report(kinematics_layout: Dict[str, Any]) -> None:
    """Print canonical Kinematics directories (always runs; does not require data files)."""
    paths = kinematics_layout["paths"]
    for key in ("raw_meshes", "cfd_npz", "graphs"):
        assert key in paths and isinstance(paths[key], Path)
    # Readable summary for CI logs
    print(
        "\n[Phase1 paths]\n"
        f"  raw_meshes: {paths['raw_meshes']}\n"
        f"  cfd_npz:    {paths['cfd_npz']}\n"
        f"  graphs:     {paths['graphs']}\n"
    )


def test_kinematics_raw_mesh_inventory(kinematics_layout: Dict[str, Any]) -> None:
    min_msh = _env_int("KINEMATICS_INSPECT_MIN_MSH", 1)
    inv = kinematics_layout["mesh_inventory"]
    if int(inv.get("count", 0)) < min_msh:
        pytest.skip(
            f"Need >= {min_msh} vessel mesh stems in {_kinematics_paths()['raw_meshes']}; "
            f"found count={inv.get('count')}."
        )
    assert inv["max_idx"] >= 0
    assert inv["next_idx"] == inv["max_idx"] + 1
    assert inv["count"] >= 1


def test_kinematics_cfd_npz_inventory(kinematics_layout: Dict[str, Any]) -> None:
    """Cross-check ``.npz`` counts vs mesh pool (same paths as anchor batch)."""
    min_msh = _env_int("KINEMATICS_INSPECT_MIN_MSH", 1)
    inv_m = kinematics_layout["mesh_inventory"]
    _skip_if_below_min_msh(inv_m, min_msh)

    inv = kinematics_layout["anchor_inventory"]
    paths = kinematics_layout["paths"]
    assert inv["existing_npz"] >= 0
    assert inv["mesh_json_with_valid_nas"] >= 1
    # Every mesh with valid NAS should either have or be missing npz; pool sizes are consistent
    assert inv["candidate_pool_including_npz"] >= inv["candidate_pool_ready"]

    min_npz = os.environ.get("KINEMATICS_INSPECT_MIN_NPZ", "").strip()
    if min_npz:
        assert inv["existing_npz"] >= int(min_npz), (
            f"Expected at least {min_npz} vessel_*.npz under {paths['cfd_npz']}, "
            f"found {inv['existing_npz']}."
        )


def _iter_graph_paths(graph_dir: Path, limit: int) -> List[Path]:
    files = sorted(graph_dir.glob("vessel_*.pt"))
    return files[:limit]


def _rheology_dirs(base_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for name in ("newtonian", "carreau"):
        d = base_dir / name
        if d.is_dir():
            out[name] = d
    return out


def _extract_idx(path: Path) -> int | None:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def _assert_graph_invariants(data: Data, path: Path) -> None:
    assert isinstance(data, Data)
    n = int(data.num_nodes)
    assert n == data.x.shape[0]
    assert data.x.shape[1] == NodeFeat.WIDTH_D2.stop, (
        f"{path.name}: x second dim {data.x.shape[1]} != {NodeFeat.WIDTH_D2.stop}"
    )
    assert torch.isfinite(data.x).all(), f"{path.name}: non-finite node features"

    assert data.edge_index.dim() == 2 and data.edge_index.shape[0] == 2
    ei = data.edge_index
    assert int(ei.max()) < n, f"{path.name}: edge_index max >= num_nodes"
    assert int(ei.min()) >= 0

    for name in ("mask_wall", "mask_inlet", "mask_outlet"):
        m = getattr(data, name)
        assert m.shape == (n,) or m.shape == (n, 1)
        m = m.view(-1)
        assert torch.isfinite(m).all()
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0 + 1e-5

    assert data.y.shape == (n, 5), f"{path.name}: y shape {data.y.shape}"
    assert torch.isfinite(data.y).all(), f"{path.name}: non-finite labels"

    if graph_has_anchor(data):
        assert bool(data.is_anchor.any().item()) if hasattr(data.is_anchor, "any") else bool(data.is_anchor)

    for op_name in ("V", "W", "M_inv", "G_x", "G_y"):
        op = getattr(data, op_name)
        assert op is not None
        if op_name in ("G_x", "G_y"):
            assert hasattr(op, "coalesce")


def test_kinematics_saved_graphs_schema(kinematics_layout: Dict[str, Any]) -> None:
    paths = kinematics_layout["paths"]
    graph_dir = paths["graphs"]
    if not graph_dir.is_dir():
        pytest.skip(f"No graph directory: {graph_dir}")

    n_all = len(list(graph_dir.glob("vessel_*.pt")))
    min_g = os.environ.get("KINEMATICS_INSPECT_MIN_GRAPHS", "").strip()
    if min_g:
        assert n_all >= int(min_g), f"Expected >= {min_g} graphs, found {n_all}"

    if n_all == 0:
        pytest.skip(f"No vessel_*.pt under {graph_dir}")

    max_load = _env_int("KINEMATICS_INSPECT_MAX_GRAPHS", 400)
    to_check = _iter_graph_paths(graph_dir, max_load)

    anchors = 0
    for path in to_check:
        data = torch.load(path, map_location="cpu", weights_only=False)
        _assert_graph_invariants(data, path)
        if graph_has_anchor(data):
            anchors += 1

    print(
        f"\n[Phase1 graphs] scanned {len(to_check)}/{n_all} files; "
        f"anchors among scanned: {anchors}\n"
    )


def test_kinematics_anchor_flag_prevalence(kinematics_layout: Dict[str, Any]) -> None:
    """Ensure at least one COMSOL anchor graph exists when npz inventory is non-zero."""
    inv_npz = kinematics_layout["anchor_inventory"]["existing_npz"]
    paths = kinematics_layout["paths"]
    graph_dir = paths["graphs"]
    if inv_npz <= 0 or not graph_dir.is_dir():
        pytest.skip("No CFD npz or no graph dir; anchor prevalence not applicable.")

    found_anchor = False
    for path in sorted(graph_dir.glob("vessel_*.pt"))[: _env_int("KINEMATICS_INSPECT_MAX_GRAPHS", 400)]:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if graph_has_anchor(data):
            found_anchor = True
            break
    assert found_anchor, (
        f"Found {inv_npz} npz under {paths['cfd_npz']} but no graph with is_anchor in sampled .pt "
        f"(check mesh_to_graph / label paths)."
    )


def test_kinematics_rheology_applied_in_anchor_groups(kinematics_layout: Dict[str, Any]) -> None:
    """Validate that generated anchor ``mu`` behavior matches selected rheology folders."""
    cfd_base = kinematics_layout["paths"]["cfd_npz"]
    rheo_dirs = _rheology_dirs(cfd_base)
    if not rheo_dirs:
        pytest.skip(f"No rheology subdirectories under {cfd_base}")

    max_npz = _env_int("KINEMATICS_INSPECT_MAX_NPZ", 200)
    for rheology, d in rheo_dirs.items():
        files = sorted(d.glob("vessel_*.npz"))[:max_npz]
        if not files:
            pytest.skip(f"No vessel_*.npz files under {d}")
        mu_ranges: list[float] = []
        for p in files:
            with np.load(p) as npz:
                if "mu" not in npz:
                    continue
                mu = np.asarray(npz["mu"]).reshape(-1)
                if mu.size == 0:
                    continue
                mu_ranges.append(float(np.nanmax(mu) - np.nanmin(mu)))
        if not mu_ranges:
            pytest.skip(f"No usable mu arrays found in {d}")

        q50 = float(np.quantile(np.asarray(mu_ranges), 0.5))
        q90 = float(np.quantile(np.asarray(mu_ranges), 0.9))
        if rheology == "newtonian":
            assert q90 < 1e-5, (
                f"Newtonian anchors should be near-constant viscosity in {d}; "
                f"observed mu range quantiles: q50={q50:.3e}, q90={q90:.3e}"
            )
        elif rheology == "carreau":
            assert q50 > 1e-4, (
                f"Carreau anchors should show variable viscosity in {d}; "
                f"observed mu range quantiles: q50={q50:.3e}, q90={q90:.3e}"
            )


def test_kinematics_anchor_graph_node_count_compatibility(kinematics_layout: Dict[str, Any]) -> None:
    """For labeled anchor graphs, per-index node counts must match anchor ``.npz`` nodes."""
    cfd_base = kinematics_layout["paths"]["cfd_npz"]
    graph_base = kinematics_layout["paths"]["graphs"]
    cfd_rheos = _rheology_dirs(cfd_base)
    graph_rheos = _rheology_dirs(graph_base)
    common = sorted(set(cfd_rheos).intersection(graph_rheos))
    if not common:
        pytest.skip(f"No common rheology subdirs between {cfd_base} and {graph_base}")

    max_pairs = _env_int("KINEMATICS_INSPECT_MAX_MATCHED", 300)
    for rheology in common:
        npz_files = sorted(cfd_rheos[rheology].glob("vessel_*.npz"))
        pt_files = {p.stem: p for p in sorted(graph_rheos[rheology].glob("vessel_*.pt"))}
        pairs = [(p, pt_files[p.stem]) for p in npz_files if p.stem in pt_files][:max_pairs]
        if not pairs:
            pytest.skip(f"No paired vessel_*.npz / vessel_*.pt files for {rheology}")

        checked = 0
        for npz_path, pt_path in pairs:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            if not graph_has_anchor(data):
                continue
            with np.load(npz_path) as npz:
                n_npz = int(np.asarray(npz["x"]).reshape(-1).shape[0])
            n_pt = int(data.num_nodes)
            checked += 1
            assert n_npz == n_pt, (
                f"{rheology}:{npz_path.stem} node-count mismatch "
                f"(npz={n_npz}, graph={n_pt})"
            )
        if checked == 0:
            pytest.skip(f"No labeled anchor graphs found in sampled pairs for {rheology}")
