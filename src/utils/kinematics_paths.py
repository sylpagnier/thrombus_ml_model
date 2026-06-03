"""Paths and rheology conventions for Stage-A kinematics graphs."""

from __future__ import annotations

from pathlib import Path

import torch

from src.utils.paths import data_root, get_project_root

# Biochem COMSOL anchors always use Carreau physics; steady kine sidecars match.
BIOCHEM_ANCHOR_KINE_RHEOLOGY = "carreau"


def kinematics_anchor_graph_dir(
    *,
    rheology: str | None = None,
    root: Path | None = None,
) -> Path:
    """Directory for steady ``KINE_*`` graphs extracted from biochem anchors."""
    r = (rheology or BIOCHEM_ANCHOR_KINE_RHEOLOGY).strip().lower()
    base = root if root is not None else get_project_root()
    return base / "data/processed/graphs_kinematics_anchors" / r


def resolve_kinematics_anchor_graph(stem: str, *, rheology: str | None = None) -> Path | None:
    """Return existing anchor kine graph path (prefers Carreau, falls back to legacy newtonian)."""
    stem = str(stem).strip()
    primary = kinematics_anchor_graph_dir(rheology=rheology) / f"{stem}.pt"
    if primary.is_file():
        return primary
    legacy = kinematics_anchor_graph_dir(rheology="newtonian") / f"{stem}.pt"
    if legacy.is_file():
        return legacy
    return None


def kinematics_training_graph_dir(*, rheology: str = "carreau", root: Path | None = None) -> Path:
    dr = data_root() if root is None else root
    return dr / "processed/graphs_kinematics" / rheology.strip().lower()


def kinematics_graph_rheology_dir(rheology: str, *, root: Path | None = None) -> Path:
    """Alias used by trainer / viz (``graphs_kinematics/<rheology>/``)."""
    return kinematics_training_graph_dir(rheology=rheology, root=root)


def iter_patient_kine_anchor_paths(*, rheology: str | None = None) -> list[Path]:
    """Sorted ``patient*.pt`` steady kine sidecars (Carreau by default)."""
    anchor_dir = kinematics_anchor_graph_dir(rheology=rheology)
    if not anchor_dir.is_dir():
        return []
    return sorted(anchor_dir.glob("patient*.pt"))


def load_patient_kine_anchor_graphs(
    *,
    rheology: str | None = None,
    attach_geometry: bool = True,
) -> list:
    """Load clinical COMSOL steady kine graphs for Stage-A finetune / eval."""
    from src.utils.channel_schema import assert_graph_schema, infer_missing_schema
    from src.utils.kinematics_geometry import attach_geometry_metadata
    from src.config import VesselConfig
    from src.utils.channel_schema import KINE_Y_SCHEMA

    paths = iter_patient_kine_anchor_paths(rheology=rheology)
    if not paths:
        return []
    cfg = VesselConfig(phase="biochem_anchors")
    out = []
    for f in paths:
        data = torch.load(f, map_location="cpu", weights_only=False)
        data = infer_missing_schema(data, phase_hint="kinematics")
        assert_graph_schema(data, expected_y_schema=(KINE_Y_SCHEMA,))
        data.graph_stem = f.stem
        data.is_clinical_anchor = True
        if attach_geometry:
            attach_geometry_metadata(data, mesh_input_dir=cfg.mesh_input_dir, stem=f.stem)
        out.append(data)
    return out
