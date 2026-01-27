"""
Tests for src.io.parser: GRID/CTRIA3/CQUAD4 extraction and .npy export.

Uses .nas files from data/raw/. Skip if none present:
    python scripts/move_nas_files.py "C:\\Users\\pgssy\\Downloads" --copy
"""

from pathlib import Path

import numpy as np
import pytest

from src.io.parser import count_grid_cards, extract_mesh, process_patient

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def _nas_files():
    if not RAW_DIR.exists():
        return []
    return sorted(RAW_DIR.glob("*.nas"))


@pytest.fixture
def nas_path():
    files = _nas_files()
    if not files:
        pytest.skip("No .nas files in data/raw/ — add patient meshes to run parser tests")
    return files[0]


@pytest.fixture
def tmp_processed(tmp_path):
    return tmp_path / "processed"


def test_grid_count_matches_extracted_nodes(nas_path):
    """Number of nodes extracted matches the GRID count in the original .nas file."""
    grid_count = count_grid_cards(nas_path)
    coords, info = extract_mesh(nas_path)
    n_extracted = coords.shape[0]
    assert n_extracted == grid_count, (
        f"Extracted {n_extracted} nodes but file has {grid_count} GRID cards"
    )
    assert len(info["node_ids"]) == grid_count


def test_process_patient_node_count_matches_grid(nas_path, tmp_processed):
    """process_patient saves .npy whose row count matches GRID count."""
    grid_count = count_grid_cards(nas_path)
    out_path, n_nodes = process_patient(nas_path, tmp_processed)
    assert n_nodes == grid_count
    arr = np.load(out_path)
    assert arr.ndim == 2 and arr.shape[1] == 2
    assert arr.shape[0] == grid_count
