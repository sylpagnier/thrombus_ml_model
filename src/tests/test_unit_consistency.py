"""Guardrails that the data-generation pipelines never silently mix unit systems.

The contract (see ``src/utils/units.py`` docstring):

* synthetic / kinematics meshes are written and stored in **meters**;
* patient / COMSOL meshes are written in **centimeters** and the extractor
  converts every CGS field back to **SI** before saving.

These tests check both the helper that enforces this contract and that the
graph builders + the COMSOL extractor actually call it. They also assert that
the centralized ``CGS_to_SI`` multipliers stay in lock-step with
``PhysicsConfig`` and that the on-disk synthetic meshes really do carry a
``unit='m'`` declaration with a plausible SI ``d_bar``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.config import PhysicsConfig
from src.utils.units import (
    CGS_to_SI,
    MESH_UNIT_CM,
    MESH_UNIT_M,
    MeshUnitMismatchError,
    SUPPORTED_MESH_UNITS,
    assert_mesh_unit,
    d_bar_si_from_sidecar,
    length_in_meters,
    read_mesh_length_unit,
)


# --- assert_mesh_unit helper ---------------------------------------------------


def test_assert_mesh_unit_passes_when_match():
    assert assert_mesh_unit(
        {"unit": "m"}, MESH_UNIT_M, stem="vessel_0", builder="UnitTest"
    ) == "m"
    assert assert_mesh_unit(
        {"unit": "cm"}, MESH_UNIT_CM, stem="vessel_0", builder="UnitTest"
    ) == "cm"


def test_assert_mesh_unit_is_case_insensitive():
    assert assert_mesh_unit(
        {"unit": "M"}, MESH_UNIT_M, stem="vessel_0", builder="UnitTest"
    ) == "m"


def test_assert_mesh_unit_raises_on_cm_to_m_mismatch():
    with pytest.raises(MeshUnitMismatchError, match="unit='cm'"):
        assert_mesh_unit(
            {"unit": "cm"}, MESH_UNIT_M, stem="vessel_0", builder="MeshToGraphPhase3"
        )


def test_assert_mesh_unit_raises_on_m_to_cm_mismatch():
    with pytest.raises(MeshUnitMismatchError, match="unit='m'"):
        assert_mesh_unit(
            {"unit": "m"}, MESH_UNIT_CM, stem="vessel_0", builder="PatientDataExtractor"
        )


def test_assert_mesh_unit_raises_on_unsupported_unit():
    with pytest.raises(MeshUnitMismatchError, match="unsupported"):
        assert_mesh_unit(
            {"unit": "km"}, MESH_UNIT_M, stem="vessel_0", builder="UnitTest"
        )


def test_assert_mesh_unit_meta_none_returns_expected_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assert_mesh_unit(
            None, MESH_UNIT_M, stem="vessel_0", builder="UnitTest"
        ) == "m"


def test_assert_mesh_unit_missing_unit_field_warns_and_returns_expected():
    with pytest.warns(UserWarning, match="no 'unit' field"):
        out = assert_mesh_unit(
            {"d_bar": 0.012}, MESH_UNIT_M, stem="vessel_0", builder="UnitTest"
        )
    assert out == "m"


def test_assert_mesh_unit_unsupported_expected_raises_value_error():
    with pytest.raises(ValueError, match="unsupported expected"):
        assert_mesh_unit(
            {"unit": "m"}, "km", stem="vessel_0", builder="UnitTest"
        )


def test_supported_mesh_units_match_expected_set():
    assert set(SUPPORTED_MESH_UNITS) == {"m", "cm"}


def test_read_mesh_length_unit_cm_and_m():
    assert read_mesh_length_unit(
        {"unit": "m"}, stem="vessel_0", builder="UnitTest"
    ) == "m"
    assert read_mesh_length_unit(
        {"unit": "cm"}, stem="vessel_0", builder="UnitTest"
    ) == "cm"


def test_length_in_meters_cm_to_si():
    assert length_in_meters(1.2, "cm") == pytest.approx(0.012)
    assert length_in_meters(0.012, "m") == pytest.approx(0.012)


def test_d_bar_si_from_sidecar_cm():
    d_si, unit = d_bar_si_from_sidecar(
        {"unit": "cm", "d_bar": 1.2}, stem="vessel_0", builder="UnitTest"
    )
    assert unit == "cm"
    assert d_si == pytest.approx(0.012)


def test_d_bar_si_from_sidecar_m():
    d_si, unit = d_bar_si_from_sidecar(
        {"unit": "m", "d_bar": 0.012}, stem="vessel_0", builder="UnitTest"
    )
    assert unit == "m"
    assert d_si == pytest.approx(0.012)


def test_d_bar_si_from_sidecar_missing_d_bar_raises():
    with pytest.raises(KeyError, match="d_bar"):
        d_bar_si_from_sidecar({"unit": "m"}, stem="vessel_0", builder="UnitTest")


# --- CGS_to_SI vs PhysicsConfig coherence --------------------------------------


def test_cgs_to_si_length_matches_phys_cfg_cm_to_m():
    """CGS_to_SI.LENGTH and PhysicsConfig.cm_to_m must stay numerically in sync."""
    cfg = PhysicsConfig(phase="biochem")
    assert CGS_to_SI.LENGTH == pytest.approx(cfg.cm_to_m)
    assert CGS_to_SI.PRESSURE == pytest.approx(cfg.cgs_p_to_pa)
    assert CGS_to_SI.VISCOSITY == pytest.approx(cfg.cgs_mu_to_pa_s)


# --- Mesh-builder integration: cm-declared mesh must be rejected ---------------


def _write_minimal_triangle_msh(path: Path) -> None:
    """Write a 1-triangle gmsh22 mesh that meshio can round-trip."""
    import meshio

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0e-3, 0.0, 0.0],
            [0.0, 1.0e-3, 0.0],
        ],
        dtype=float,
    )
    cells = [("triangle", np.array([[0, 1, 2]], dtype=int))]
    mesh = meshio.Mesh(points, cells)
    mesh.write(str(path), file_format="gmsh22")


def _write_sidecar_json(path: Path, *, unit: str, d_bar: float = 1.0e-3) -> None:
    payload = {
        "id": 0,
        "unit": unit,
        "d_bar": d_bar,
        "centerline_pts": [[0.0, 0.0], [0.5, 0.0]],
        "centerline_tangents": [[1.0, 0.0], [1.0, 0.0]],
        "top_wall_pts": [[0.0, 0.5], [0.5, 0.5]],
        "bot_wall_pts": [[0.0, -0.5], [0.5, -0.5]],
        "top_wall_tangents": [[1.0, 0.0], [1.0, 0.0]],
        "bot_wall_tangents": [[1.0, 0.0], [1.0, 0.0]],
        "top_wall_normals": [[0.0, 1.0], [0.0, 1.0]],
        "bot_wall_normals": [[0.0, -1.0], [0.0, -1.0]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_mesh_to_graph_phase3_rejects_cm_sidecar(tmp_path):
    """Synthetic biochem builder must refuse a sidecar declaring unit='cm'."""
    from src.data_gen.lib.mesh_to_graph_biochem import MeshToGraphPhase3

    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    _write_minimal_triangle_msh(raw / "vessel_0.msh")
    _write_sidecar_json(raw / "vessel_0.json", unit="cm")

    builder = MeshToGraphPhase3(raw_dir=raw, label_dir=raw, proc_dir=out)
    with pytest.raises(MeshUnitMismatchError, match="MeshToGraphPhase3"):
        builder.process_file("vessel_0.msh")


def test_patient_data_extractor_rejects_m_sidecar(tmp_path):
    """Anchor extractor must refuse a sidecar declaring unit='m' (it expects CGS cm)."""
    from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor

    raw = tmp_path / "raw"
    label = tmp_path / "label"
    out = tmp_path / "out"
    raw.mkdir()
    label.mkdir()
    out.mkdir()
    _write_minimal_triangle_msh(raw / "vessel_0.msh")
    _write_sidecar_json(raw / "vessel_0.json", unit="m")

    extractor = PatientDataExtractor(
        phase="biochem", raw_dir=raw, label_dir=label, proc_dir=out
    )
    with pytest.raises(MeshUnitMismatchError, match="PatientDataExtractor"):
        extractor.process_patient("vessel_0")


# --- On-disk synthetic mesh: declared unit + plausible SI d_bar ----------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SYNTHETIC_MESH_DIR = _REPO_ROOT / "data" / "raw" / "biochem"


@pytest.mark.skipif(
    not (_SYNTHETIC_MESH_DIR / "vessel_0.json").exists(),
    reason="No synthetic biochem meshes present (run pipeline_biochem track 1 first).",
)
def test_existing_synthetic_meshes_declare_meters_with_si_d_bar():
    """Every synthetic biochem sidecar on disk must declare unit='m' and d_bar in SI."""
    sidecars = sorted(_SYNTHETIC_MESH_DIR.glob("vessel_*.json"))
    assert sidecars, "expected at least one sidecar JSON in data/raw/biochem"

    for sc in sidecars:
        meta = json.loads(sc.read_text(encoding="utf-8"))
        unit = str(meta.get("unit", "")).lower()
        d_bar = float(meta.get("d_bar", 0.0))
        assert unit == "m", f"{sc.name}: expected unit='m', got {unit!r}"
        # Plausible vessel diameter in SI: 0.5 mm .. 5 cm.
        assert 5.0e-4 < d_bar < 5.0e-2, (
            f"{sc.name}: d_bar={d_bar} m is outside the plausible "
            "SI vessel range (5e-4, 5e-2). Likely a unit mismatch."
        )


# --- Boundary mapping health: _load_spatial_mask three-tier policy -------------


@pytest.fixture
def _kdtree_mesh_pair():
    """A simple square mesh with vertices at the integer grid (in metres)."""
    from scipy.spatial import cKDTree

    pts = np.array(
        [[0.0, 0.0], [1e-3, 0.0], [2e-3, 0.0], [3e-3, 0.0], [4e-3, 0.0]],
        dtype=float,
    )
    return pts, cKDTree(pts)


def _make_extractor(tmp_path):
    from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor

    raw = tmp_path / "raw"
    label = tmp_path / "label"
    out = tmp_path / "out"
    for d in (raw, label, out):
        d.mkdir()
    return PatientDataExtractor(
        phase="biochem", raw_dir=raw, label_dir=label, proc_dir=out
    ), label


def _write_boundary_csv(path: Path, coords_cm: np.ndarray) -> None:
    """Write a minimal COMSOL-shaped boundary export. Two leading data cols are dummies."""
    lines = [
        "% Model: test.mph",
        "% x  y",
    ]
    for x, y in coords_cm:
        lines.append(f"0 0 {x:.10f} {y:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_spatial_mask_pure_p1_export_is_silent(tmp_path, _kdtree_mesh_pair, capsys):
    """All exported coords sit on mesh vertices -> no warning, vertex_hit_rate == 1."""
    pts, tree = _kdtree_mesh_pair
    extractor, label = _make_extractor(tmp_path)

    # Export every vertex (in cm to match the cm-to-m conversion inside the loader).
    coords_cm = pts * 100.0
    f = label / "stem_inlet.txt"
    _write_boundary_csv(f, coords_cm)

    mask, diag = extractor._load_spatial_mask(
        f, tree, num_nodes=len(pts), mesh_edge_scale_m=1e-3
    )
    captured = capsys.readouterr().out

    assert mask.sum().item() == len(pts)
    assert diag["vertex_hit_rate"] == pytest.approx(1.0)
    assert diag["unmapped_ratio"] == pytest.approx(0.0)
    assert diag["d_median_m"] < 1e-9
    assert diag["p2_inferred"] is False
    assert "Warning" not in captured and "P2 export" not in captured


def test_load_spatial_mask_p2_export_is_inferred_and_quiet(tmp_path, _kdtree_mesh_pair, capsys):
    """Vertex + mid-edge interleaved -> p2_inferred True, info note (not warning)."""
    pts, tree = _kdtree_mesh_pair
    extractor, label = _make_extractor(tmp_path)

    edge_scale_m = 1e-3  # exactly the spacing in the fixture
    midpoints_m = (pts[:-1] + pts[1:]) / 2.0
    combined_m = np.vstack([pts, midpoints_m])
    f = label / "stem_wall.txt"
    _write_boundary_csv(f, combined_m * 100.0)

    mask, diag = extractor._load_spatial_mask(
        f, tree, num_nodes=len(pts), mesh_edge_scale_m=edge_scale_m
    )
    captured = capsys.readouterr().out

    assert mask.sum().item() == len(pts)  # every real vertex flagged
    assert 0.30 <= diag["vertex_hit_rate"] <= 1.0
    assert diag["p2_inferred"] is True
    assert "P2 export inferred" in captured
    assert "⚠️" not in captured  # no warning escalation when P2 explains it


def test_load_spatial_mask_unit_bug_raises(tmp_path, _kdtree_mesh_pair):
    """Coords given in cm-as-if-they-were-m (100x too far) -> hard error, not warning."""
    pts, tree = _kdtree_mesh_pair
    extractor, label = _make_extractor(tmp_path)

    # If COMSOL exported in mm but the loader treats it as cm, distances explode.
    # Simulate by writing coords whose .cm * 0.01 lands at metre-scale offsets.
    coords_cm = (pts + 0.1) * 100.0  # +10 cm offset before unit conversion
    f = label / "stem_inlet.txt"
    _write_boundary_csv(f, coords_cm)

    with pytest.raises(ValueError, match="median nearest-vertex distance"):
        extractor._load_spatial_mask(
            f, tree, num_nodes=len(pts), mesh_edge_scale_m=1e-3
        )


def test_load_spatial_mask_low_coverage_warns_loudly(tmp_path, _kdtree_mesh_pair, capsys):
    """vertex_hit_rate < 0.30 triggers the loud mesh-coverage warning."""
    pts, tree = _kdtree_mesh_pair
    extractor, label = _make_extractor(tmp_path)

    # 1 vertex hit + 9 stray-but-near (e.g. 50um off) coords -> 1/10 = 10% hit.
    near_real = pts[:1]
    strays = pts[:1] + np.array([[5e-5, 0.0]] * 9) + np.arange(9).reshape(-1, 1) * 1e-7
    combined = np.vstack([near_real, strays])
    f = label / "stem_wall.txt"
    _write_boundary_csv(f, combined * 100.0)

    mask, diag = extractor._load_spatial_mask(
        f, tree, num_nodes=len(pts), mesh_edge_scale_m=1e-3
    )
    captured = capsys.readouterr().out

    assert diag["vertex_hit_rate"] < 0.30
    assert "⚠️" in captured
    assert "matched a mesh vertex" in captured


# --- On-disk anchor metadata sanity floor --------------------------------------


_ANCHOR_GRAPH_DIR = _REPO_ROOT / "data" / "processed" / "graphs_biochem_anchors"


# --- Biochem ADR kernel: fast/slow accumulators must remain independent --------


def test_biochem_adr_residual_fast_slow_are_distinct_tensors():
    """Regression for the silent ``adr_losses_fast = adr_losses_slow = z`` aliasing.

    The original kernel initialised both accumulators from the *same* zero-grad tensor
    and used ``+=`` to accumulate per-species losses. ``+=`` is in-place on tensors,
    so every species' loss landed in both buckets and the kernel returned two
    references to the same storage. That neutralised the fast/slow Kendall weighting
    in training and produced bit-identical ``ADR_fast`` / ``ADR_slow`` in tests.

    This test fabricates a tiny graph in which only one fast species (RP) carries
    a non-zero spatial gradient. Independent accumulators must give
    ``adr_fast > 0`` and ``adr_slow == 0``; an aliased pair would inflate both equally.
    """
    import torch
    from torch_geometric.data import Data

    from src.config import (
        BiochemConfig,
        BulkSpecies,
        PhysicsConfig,
    )
    from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
    from src.core_physics.physics_kernels import PhysicsKernels

    bio_cfg = BiochemConfig(phase="biochem")
    phys_cfg = PhysicsConfig(phase="biochem")
    core = PhysicsKernels(phys_cfg=phys_cfg)
    kernels = BiochemPhysicsKernels(biochem_cfg=bio_cfg, core_physics_kernels=core)

    num_nodes = 4
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 1, 0, 2, 1, 3, 2], [1, 0, 2, 1, 3, 2, 0, 1, 1, 2, 2, 3]],
        dtype=torch.long,
    )
    eye = torch.eye(num_nodes, dtype=torch.float32)
    sparse_eye = eye.to_sparse_coo()

    data = Data(
        edge_index=edge_index,
        G_x=sparse_eye,
        G_y=sparse_eye,
        Laplacian=sparse_eye,
        mask_inlet=torch.zeros(num_nodes, dtype=torch.bool),
        mask_outlet=torch.zeros(num_nodes, dtype=torch.bool),
        mask_wall=torch.zeros(num_nodes, dtype=torch.bool),
    )

    species_preds = torch.zeros((num_nodes, 9), dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        species_preds[:, BulkSpecies.RP.value] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    velocity_field = torch.ones((num_nodes, 2), dtype=torch.float32)
    spatial_props = {
        "u_ref": torch.full((num_nodes,), 0.1, dtype=torch.float32),
        "d_bar": torch.full((num_nodes,), 0.01, dtype=torch.float32),
    }

    adr_fast, adr_slow = kernels.biochem_adr_residual(
        species_preds, velocity_field, spatial_props, data, d_pred_dt=None
    )

    assert adr_fast.data_ptr() != adr_slow.data_ptr(), (
        "biochem_adr_residual returned two views of the same tensor; the in-place "
        "`+=` accumulators inside the kernel will silently equate fast and slow "
        "losses and neutralise their Kendall weighting downstream."
    )
    assert float(adr_fast.item()) > 0.0, "Expected non-zero fast loss with RP gradient set."
    assert float(adr_slow.item()) == pytest.approx(0.0, abs=1e-30), (
        "Expected zero slow loss with no slow species perturbation; non-zero "
        "indicates the fast-bucket contribution leaked into the slow bucket."
    )


def test_biochem_adr_residual_kernel_pattern_is_not_aliased_assignment():
    """Source-level guard: the buggy ``adr_losses_fast = adr_losses_slow = z`` pattern
    must not reappear (a future refactor could reintroduce it). The fix initialises
    each accumulator from a fresh ``species_preds.sum() * 0.0`` so they have distinct
    storage.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "core_physics"
        / "biochem_physics_kernels.py"
    ).read_text(encoding="utf-8")
    assert "adr_losses_fast = adr_losses_slow" not in src, (
        "Found the historical aliasing pattern in biochem_physics_kernels.py. "
        "Initialise adr_losses_fast and adr_losses_slow from independent expressions."
    )


@pytest.mark.skipif(
    not _ANCHOR_GRAPH_DIR.exists()
    or not any(_ANCHOR_GRAPH_DIR.glob("*_metadata.json")),
    reason="No anchor metadata present (run pipeline_biochem track 2 first).",
)
def test_anchor_metadata_boundary_diagnostics_within_unit_floor():
    """Every saved anchor must record a sub-mm median residual on every boundary.

    A median > 1 mm cannot be caused by COMSOL P2 mid-edge nodes (those sit at
    ~½ mesh edge ~150 um); it can only be produced by a unit / coordinate-frame
    mismatch. This test makes that regression fail fast in CI.
    """
    metas = sorted(_ANCHOR_GRAPH_DIR.glob("*_metadata.json"))
    assert metas, "expected at least one anchor metadata file"

    floor_m = 1.0e-3
    for mp in metas:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        boundaries = meta.get("quality", {}).get("boundaries")
        if boundaries is None:
            # Pre-policy metadata; remind the user to re-extract.
            pytest.skip(
                f"{mp.name}: no per-boundary diagnostics (re-run extractor)."
            )
        for name, diag in boundaries.items():
            status = diag.get("status")
            d_med = diag.get("d_median_m")
            if status != "ok" or d_med is None or np.isnan(d_med):
                continue
            assert d_med < floor_m, (
                f"{mp.name}/{name}: d_median_m={d_med:.3e} m exceeds the "
                f"{floor_m} m unit-bug floor; the export and mesh likely "
                "disagree on units or coordinate frame."
            )
