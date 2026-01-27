"""
Parse .nas (Nastran BDF) mesh files and extract GRID coordinates and CTRIA3/CQUAD4 connectivity.

Outputs clean 2D NumPy arrays of shape (N, 2) with (x, y) coordinates, saved as .npy files.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
from pyNastran.bdf.bdf import read_bdf
from pyNastran.bdf.errors import MissingDeckSections
from pyNastran.bdf.utils import get_xyz_cid0_dict


def _read_bdf_robust(path: Path):
    """Read BDF, supporting bulk-only files (e.g. COMSOL export) via temp prepended Exec+Case."""
    p = Path(path)
    try:
        return read_bdf(str(p), xref=True)
    except MissingDeckSections:
        text = p.read_text(encoding="utf-8", errors="replace")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".nas", delete=False, encoding="utf-8"
        ) as f:
            f.write("SOL 101\nCEND\n")
            f.write(text)
            tmp = f.name
        try:
            return read_bdf(tmp, xref=True)
        finally:
            Path(tmp).unlink(missing_ok=True)


def count_grid_cards(nas_path: str | Path) -> int:
    """Count GRID cards in a Nastran BDF file by scanning the raw text."""
    path = Path(nas_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    count = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("$") or s.startswith("+"):
            continue
        if re.match(r"^GRID\*?\s", s, re.IGNORECASE):
            count += 1
    return count


def extract_mesh(nas_path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Extract GRID coordinates and CTRIA3/CQUAD4 connectivity from a .nas file.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 2) array of (x, y) coordinates, one row per GRID, ordered by node ID.
    info : dict
        'node_ids', 'connectivity', 'element_types'.
    """
    path = Path(nas_path)
    model = _read_bdf_robust(path)

    xyz_dict = get_xyz_cid0_dict(model)
    node_ids = sorted(xyz_dict.keys())
    coords_3d = np.array([xyz_dict[nid] for nid in node_ids], dtype=np.float64)
    coords = coords_3d[:, :2]

    connectivity = []
    element_types = []

    def _node_ids(elem):
        if hasattr(elem, "node_ids") and elem.node_ids is not None:
            return list(elem.node_ids)
        return [n.nid for n in elem.nodes]

    for _eid, elem in model.elements.items():
        t = elem.type
        if t == "CTRIA3":
            nids = _node_ids(elem)
            if len(nids) >= 3:
                connectivity.append(np.array(nids[:3], dtype=np.int64))
                element_types.append("CTRIA3")
        elif t == "CQUAD4":
            nids = _node_ids(elem)
            if len(nids) >= 4:
                connectivity.append(np.array(nids[:4], dtype=np.int64))
                element_types.append("CQUAD4")

    info = {"node_ids": node_ids, "connectivity": connectivity, "element_types": element_types}
    return coords, info


def process_patient(nas_path: str | Path, out_dir: str | Path) -> tuple[Path, int]:
    """
    Load a patient .nas, extract (N, 2) coordinates, and save as .npy.

    Returns (out_path, n_nodes).
    """
    path = Path(nas_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    coords, _ = extract_mesh(path)
    n_nodes = coords.shape[0]
    out_path = out / f"{path.stem}.npy"
    np.save(out_path, coords)
    return out_path, n_nodes


def process_all_nas(input_dir: str | Path, output_dir: str | Path) -> list[tuple[Path, int]]:
    """Process all .nas in input_dir and save .npy to output_dir. Returns [(out_path, n_nodes), ...]."""
    inp = Path(input_dir)
    return [process_patient(f, output_dir) for f in sorted(inp.glob("*.nas"))]


if __name__ == "__main__":
    import argparse

    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Extract GRID (x,y) from .nas → .npy")
    ap.add_argument("--input", default=root / "data" / "raw", type=Path, help="Input dir with .nas")
    ap.add_argument("--output", default=root / "data" / "processed", type=Path, help="Output dir for .npy")
    args = ap.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input dir not found: {args.input}")
    for out_path, n in process_all_nas(args.input, args.output):
        print(f"  {out_path.name}  ({n} nodes)")
