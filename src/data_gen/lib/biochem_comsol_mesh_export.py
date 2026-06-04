"""Export anchor FE mesh + boundary masks from a solved biochem COMSOL ``.mph``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import meshio
import numpy as np

from src.data_gen.lib.biochem_comsol_auto_export import _evaluate_at_coords_and_time
from src.data_gen.lib.centerline_utils import resolve_anchor_mesh_path
from src.tools.prepare_biochem_anchors import scaffold_anchor_sidecars

logger = logging.getLogger(__name__)

_BOUNDARY_SPECS: tuple[tuple[str, str], ...] = (
    ("inlet", "is_inlet"),
    ("outlet", "is_outlet"),
    ("wall", "is_wall"),
)


def find_comsol_mesh_tag(model_java) -> str:
    """Pick the mesh sequence with the most vertices (typical solved comp1 mesh)."""
    tags = [str(t) for t in model_java.mesh().tags()]
    if not tags:
        raise RuntimeError("COMSOL model has no mesh sequences.")

    best_tag = tags[0]
    best_n = -1
    for tag in tags:
        try:
            n = int(model_java.mesh(tag).getNumVertex())
        except Exception:
            continue
        if n > best_n:
            best_n = n
            best_tag = tag

    if best_n < 10:
        raise RuntimeError(f"COMSOL mesh '{best_tag}' has too few vertices ({best_n}).")
    logger.info("[i] Using COMSOL mesh tag '%s' (%d vertices)", best_tag, best_n)
    return best_tag


def mesh_has_gmsh_boundary_tags(mesh_path: Path) -> bool:
    try:
        mesh = meshio.read(mesh_path)
        line_cells = mesh.get_cells_type("line")
        if len(line_cells) == 0:
            return False
        mesh.get_cell_data("gmsh:physical", "line")
        return True
    except Exception:
        return False


def write_boundary_txt_from_comsol_masks(
    model_java,
    coords_cm: np.ndarray,
    label_dir: Path,
    stem: str,
    *,
    dataset_tag: str = "dset1",
    threshold: float = 0.5,
    force: bool = False,
) -> None:
    """Write inlet/outlet/wall ``.txt`` using COMSOL selection variables on mesh nodes."""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    for bname, expr in _BOUNDARY_SPECS:
        out_path = label_dir / f"{stem}_{bname}.txt"
        if out_path.is_file() and not force:
            continue

        vals = _evaluate_at_coords_and_time(
            model_java,
            coords_cm,
            [expr],
            dataset_tag=dataset_tag,
            time_value=0.0,
        ).reshape(-1)
        mask = vals > threshold
        if not np.any(mask):
            logger.warning("[WARN] %s: no nodes matched %s for %s boundary.", stem, expr, bname)
            coords = np.zeros((0, 2), dtype=np.float64)
        else:
            coords = np.unique(coords_cm[mask, :2], axis=0)

        lines = ["% Model: COMSOL selection masks", "% x  y"]
        for x, y in coords:
            lines.append(f"0 0 {x:.10f} {y:.10f}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("[OK] %s: %s boundary (%d unique coords)", stem, bname, len(coords))


def ensure_anchor_mesh_from_comsol(
    model_java,
    stem: str,
    raw_dir: Path,
    *,
    force: bool = False,
    mesh_tag: str | None = None,
) -> tuple[Path, bool]:
    """Export ``<stem>.nas`` (+ ``.msh``) from COMSOL when missing.

    Returns ``(mesh_path, exported_now)``.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    msh_path = raw_dir / f"{stem}.msh"
    nas_path = raw_dir / f"{stem}.nas"

    existing = resolve_anchor_mesh_path(raw_dir, stem)
    if existing is not None and not force:
        return existing, False

    tag = mesh_tag or find_comsol_mesh_tag(model_java)
    safe_nas = str(nas_path.resolve()).replace("\\", "/")
    logger.info("[NEW] %s: exporting COMSOL mesh -> %s", stem, nas_path.name)
    model_java.mesh(tag).export(safe_nas)

    if not nas_path.is_file() or nas_path.stat().st_size == 0:
        raise RuntimeError(f"COMSOL mesh export did not create {nas_path}")

    try:
        m = meshio.read(nas_path)
        if int(m.points.shape[0]) >= 3 and int(m.points.shape[1]) >= 2:
            meshio.write(msh_path, m)
            logger.info("[OK] %s: wrote %s from NASTRAN export", stem, msh_path.name)
    except Exception as exc:
        logger.warning("[WARN] %s: .nas -> .msh conversion failed (%s); using .nas", stem, exc)

    scaffold_anchor_sidecars(raw_dir, unit="cm", force=False, dry_run=False)
    sidecar = raw_dir / f"{stem}.json"
    if not sidecar.is_file():
        sidecar.write_text(json.dumps({"unit": "cm", "level": 2}), encoding="utf-8")

    out = msh_path if msh_path.is_file() else nas_path
    return out, True
