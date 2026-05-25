"""Path overrides for kinematics graph training / conversion A/B runs."""

from __future__ import annotations

import os
from pathlib import Path

from src.config import VesselConfig


def kinematics_graph_rheology_dir(rheology: str) -> Path:
    """
    Directory of ``vessel_*.pt`` for one rheology pass.

    Override with env ``KINEMATICS_GRAPH_RHEOLOGY_DIR`` (absolute or project-relative)
    for bend-sign A/B without touching the main ``graphs_kinematics/newtonian`` tree.
    """
    override = os.environ.get("KINEMATICS_GRAPH_RHEOLOGY_DIR", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = VesselConfig(phase="kinematics").project_root / p
        return p
    cfg = VesselConfig(phase="kinematics")
    return cfg.graph_output_dir / str(rheology).lower()
