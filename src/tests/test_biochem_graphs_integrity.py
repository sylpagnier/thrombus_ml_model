"""Pytest suite for validating the physical and mathematical integrity of biochem graphs."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.utils.paths import get_project_root
from src.data_gen.lib.extract_biochem_comsol_data import validate_graph_physical_integrity

# Locate biochem graph directory
GRAPH_DIR = get_project_root() / "data" / "processed" / "graphs_biochem_anchors"
PT_FILES = sorted(list(GRAPH_DIR.glob("patient*.pt")))

def discover_test_cases():
    return [fp.stem for fp in PT_FILES]

@pytest.mark.skipif(not PT_FILES, reason="No biochem patient graphs found in data/processed/graphs_biochem_anchors/")
@pytest.mark.parametrize("stem", discover_test_cases())
def test_biochem_graph_physical_integrity(stem):
    """Load each patient graph and check that all physical properties are correct."""
    graph_path = GRAPH_DIR / f"{stem}.pt"
    meta_path = GRAPH_DIR / f"{stem}_metadata.json"
    
    # 1. Load PyG Graph Data
    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    
    # 2. Extract Mass Flux Imbalance from Metadata JSON if available
    avg_flux_imbalance = 0.0
    if meta_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                avg_flux_imbalance = float(meta.get("quality", {}).get("mass_flux_imbalance", 0.0))
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            pass
            
    # 3. Assert Graph schema/integrity and run validations
    validate_graph_physical_integrity(data, stem, avg_flux_imbalance)
