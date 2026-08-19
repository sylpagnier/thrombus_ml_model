"""Export anchor FE mesh + boundary masks from a solved biochem COMSOL ``.mph``."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import meshio
import numpy as np
from scipy.spatial import cKDTree

from src.data_gen.lib.biochem_comsol_auto_export import _evaluate_at_coords_and_time
from src.data_gen.lib.biochem_comsol_datasets import resolve_boundary_datasets, sample_coords_from_dataset
from src.data_gen.lib.centerline_utils import resolve_anchor_mesh_path
from src.tools.prepare_biochem_anchors import scaffold_anchor_sidecars

logger = logging.getLogger(__name__)

_BOUNDARY_SPECS: tuple[tuple[str, str], ...] = (
    ("inlet", "is_inlet"),
    ("outlet", "is_outlet"),
    ("wall", "is_wall"),
)

# Phase-2 anchors (phase2_nowound_008 / phase2_wound_008): box selections labeled inlet/outlet/wall.
# Older templates: is_inlet=sel1(x,y) or box1/box2/dif1 tags.
_STATIC_BOUNDARY_EXPRS: dict[str, tuple[str, ...]] = {
    "inlet": ("inlet(x,y)", "is_inlet", "sel1(x,y)", "box1(x,y)"),
    "outlet": ("outlet(x,y)", "is_outlet", "sel2(x,y)", "box2(x,y)"),
    "wall": ("wall(x,y)", "is_wall", "sel3(x,y)", "dif1(x,y)"),
    "wound": ("wound(x,y)", "is_wound", "sel4(x,y)"),
}


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def discover_boundary_mask_exprs(model_java) -> dict[str, str]:
    """Map inlet/outlet/wall -> ``<selection_tag>(x,y)`` from COMSOL Definitions > Selections."""
    found: dict[str, str] = {}
    s_root = model_java.selection()
    try:
        tags = [str(t) for t in s_root.tags()]
    except Exception:
        return found

    for bname in ("inlet", "outlet", "wall", "wound"):
        for sid in tags:
            low = sid.lower()
            if "nastran" in low or low.startswith("imp"):
                continue
            if low == bname:
                found[bname] = f"{sid}(x,y)"
                break

    for sid in tags:
        low = sid.lower()
        if "nastran" in low or low.startswith("imp"):
            continue
        try:
            label = str(s_root.get(sid).name()).lower()
        except Exception:
            label = low
        expr = f"{sid}(x,y)"
        for bname, keys in (
            ("inlet", ("inlet",)),
            ("outlet", ("outlet",)),
            ("wall", ("wall",)),
            ("wound", ("wound",)),
        ):
            if bname in found:
                continue
            if label == bname or low == bname:
                found[bname] = expr
                continue
            if any(k in label for k in keys) or any(k in low for k in keys):
                found[bname] = expr
    return found


def boundary_mask_expr_candidates(model_java, bname: str) -> list[str]:
    """Ordered COMSOL expressions to probe for a boundary mask (deduped)."""
    discovered = discover_boundary_mask_exprs(model_java)
    out: list[str] = []
    seen: set[str] = set()

    def _add(expr: str) -> None:
        key = expr.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)

    if bname in discovered:
        _add(discovered[bname])
    _add(f"{bname}(x,y)")
    for expr in _STATIC_BOUNDARY_EXPRS.get(bname, ()):
        _add(expr)
    return out


def _evaluate_boundary_mask(
    model_java,
    coords_cm: np.ndarray,
    expr: str,
    *,
    dataset_tag: str,
) -> np.ndarray:
    vals = _evaluate_at_coords_and_time(
        model_java,
        coords_cm,
        [expr],
        dataset_tag=dataset_tag,
        time_value=0.0,
    ).reshape(-1)
    if vals.size != coords_cm.shape[0]:
        raise ValueError(f"Boundary mask {expr!r}: length {vals.size} != {coords_cm.shape[0]}")
    return vals


def boundary_txt_has_coords(path: Path) -> bool:
    """True when a boundary txt exists and has at least one non-comment coordinate row."""
    path = Path(path)
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("%"):
            return True
    return False


def _write_xy_boundary_txt(path: Path, coords: np.ndarray, header: str) -> None:
    pts = np.unique(np.asarray(coords, dtype=np.float64).reshape(-1, 2), axis=0)
    lines = [header, "% x  y"]
    for x, y in pts:
        lines.append(f"0 0 {x:.10f} {y:.10f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mesh_coords_cm(mesh_path: Path | None) -> np.ndarray | None:
    if mesh_path is None or not Path(mesh_path).is_file():
        return None
    try:
        mesh = meshio.read(mesh_path)
        pts = np.asarray(mesh.points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] < 2:
            return None
        return pts[:, :2]
    except Exception:
        return None


def write_wound_txt_from_comsol_mask(
    model_java,
    coords_cm: np.ndarray,
    label_dir: Path,
    stem: str,
    *,
    dataset_tag: str = "dset1",
    threshold: float = 0.5,
    force: bool = False,
) -> bool:
    """Evaluate the COMSOL ``wound`` selection on mesh nodes and write ``*_wound.txt``.

    Returns True when the file has coordinates.
    """
    wound_path = Path(label_dir) / f"{stem}_wound.txt"
    if boundary_txt_has_coords(wound_path) and not force:
        return True
    coords_cm = np.asarray(coords_cm, dtype=np.float64)
    if coords_cm.ndim != 2 or coords_cm.shape[0] < 1:
        return False
    wound_mask = None
    wound_expr = None
    for expr in boundary_mask_expr_candidates(model_java, "wound"):
        try:
            vals = _evaluate_boundary_mask(
                model_java,
                coords_cm,
                expr,
                dataset_tag=dataset_tag,
            )
            wound_mask = vals > threshold
            wound_expr = expr
            break
        except Exception:
            continue
    if wound_mask is None or not np.any(wound_mask):
        return False
    _write_xy_boundary_txt(
        wound_path,
        coords_cm[wound_mask, :2],
        f"% Model: COMSOL mask ({wound_expr})",
    )
    logger.info(
        "[OK] %s: wound boundary via %s (%d unique coords)",
        stem,
        wound_expr,
        int(np.unique(coords_cm[wound_mask, :2], axis=0).shape[0]),
    )
    return True


def ensure_wound_boundary_txt(
    model_java,
    coords_cm: np.ndarray | None,
    mesh_path: Path | None,
    label_dir: Path,
    stem: str,
    *,
    vessel_cfg=None,
    dataset_tag: str = "dset1",
    force: bool = False,
) -> bool:
    """Write ``*_wound.txt`` from Gmsh tags, a Wound dataset, or the COMSOL wound selection.

    Inlet/outlet/wall already existing does not skip this. Nowound stems stay silent
    when no wound selection exists. Returns True when the file has coordinates.
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    wound_path = label_dir / f"{stem}_wound.txt"
    if boundary_txt_has_coords(wound_path) and not force:
        return True

    mesh_path = Path(mesh_path) if mesh_path is not None else None
    if mesh_path is not None and mesh_has_gmsh_boundary_tags(mesh_path):
        from src.data_gen.lib.biochem_comsol_auto_export import write_boundary_txt_from_mesh

        write_boundary_txt_from_mesh(
            mesh_path,
            label_dir,
            stem,
            vessel_cfg=vessel_cfg,
            force=force,
        )
        if boundary_txt_has_coords(wound_path) and not force:
            return True

    pts = None if coords_cm is None else np.asarray(coords_cm, dtype=np.float64)
    if pts is None:
        pts = _mesh_coords_cm(mesh_path)

    if model_java is not None and pts is not None:
        try:
            bmap = resolve_boundary_datasets(model_java)
        except Exception:
            bmap = {}
        dset = bmap.get("wound")
        if dset:
            try:
                ref = None
                for edim in (1, None):
                    try:
                        ref = sample_coords_from_dataset(model_java, dset, edim=edim)
                        break
                    except Exception:
                        continue
                if ref is not None and ref.size:
                    tol_cm = _boundary_snap_tol_cm(pts)
                    tree = cKDTree(np.asarray(ref, dtype=np.float64)[:, :2])
                    dist, _ = tree.query(pts[:, :2])
                    mask = dist <= tol_cm
                    if np.any(mask):
                        _write_xy_boundary_txt(
                            wound_path,
                            pts[mask, :2],
                            f"% Model: mesh nodes snapped to dataset {dset} (tol={tol_cm:.4g} cm)",
                        )
                        logger.info(
                            "[OK] %s: wound = %d mesh nodes (snap to '%s')",
                            stem,
                            int(np.unique(pts[mask, :2], axis=0).shape[0]),
                            dset,
                        )
                        if not force:
                            return True
            except Exception as exc:
                logger.debug("[i] %s: wound dataset snap failed (%s)", stem, exc)

        if write_wound_txt_from_comsol_mask(
            model_java,
            pts,
            label_dir,
            stem,
            dataset_tag=dataset_tag,
            force=force,
        ):
            return True

    from src.data_gen.lib.biochem_comsol_auto_export import parse_biochem_extract_stem

    ref = parse_biochem_extract_stem(stem)
    if ref is not None and ref.variant == "wound" and not boundary_txt_has_coords(wound_path):
        logger.warning(
            "[WARN] %s: wound variant but no wound nodes written "
            "(no Gmsh Wound tag, no wound_nodes export, and wound(x,y)/sel1 did not match).",
            stem,
        )
    return boundary_txt_has_coords(wound_path)


def write_boundary_txt_from_axis_extents(
    coords_cm: np.ndarray,
    label_dir: Path,
    stem: str,
    *,
    axis: int = 0,
    band_frac: float = 0.02,
    force: bool = False,
) -> None:
    """Heuristic inlet/outlet/wall from mesh bbox (flow along ``axis``, walls on the other axis)."""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(coords_cm[:, :2], dtype=np.float64)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    tol = np.maximum(span * float(band_frac), 1e-6)

    inlet_m = pts[:, axis] <= lo[axis] + tol[axis]
    outlet_m = pts[:, axis] >= hi[axis] - tol[axis]
    other = 1 - axis
    wall_m = (
        (pts[:, other] <= lo[other] + tol[other]) | (pts[:, other] >= hi[other] - tol[other])
    ) & ~(inlet_m | outlet_m)

    specs = (("inlet", inlet_m), ("outlet", outlet_m), ("wall", wall_m))
    for bname, mask in specs:
        out_path = label_dir / f"{stem}_{bname}.txt"
        if out_path.is_file() and not force:
            continue
        coords = np.unique(pts[mask, :2], axis=0) if np.any(mask) else np.zeros((0, 2))
        lines = ["% Model: mesh axis-extent heuristic", "% x  y"]
        for x, y in coords:
            lines.append(f"0 0 {x:.10f} {y:.10f}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("[OK] %s: %s boundary heuristic (%d unique coords)", stem, bname, len(coords))


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


def _boundary_snap_tol_cm(coords_cm: np.ndarray) -> float:
    """Distance (cm) to snap volume mesh nodes onto a COMSOL boundary dataset."""
    raw = (os.environ.get("BIOCHEM_BOUNDARY_SNAP_CM") or "").strip()
    if raw:
        return float(raw)
    pts = np.asarray(coords_cm[:, :2], dtype=np.float64)
    if pts.shape[0] < 4:
        return 0.01
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=2)
    nn = np.asarray(dist[:, 1], dtype=np.float64)
    return max(0.002, 0.35 * float(np.median(nn)))


def write_boundary_txt_from_mesh_snap_to_datasets(
    model_java,
    coords_cm: np.ndarray,
    label_dir: Path,
    stem: str,
    *,
    boundary_datasets: dict[str, str] | None = None,
    force: bool = False,
) -> bool:
    """Mark volume mesh nodes near Inlet/Outlet/Wall datasets; write their exact mesh coords.

    Avoids the edg* vs NASTRAN node mismatch: boundary txt uses the same nodes as domain export.
    """
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    bmap = boundary_datasets if boundary_datasets is not None else resolve_boundary_datasets(model_java)
    if not all(k in bmap for k in ("inlet", "outlet", "wall")):
        return False

    pts_vol = np.asarray(coords_cm[:, :2], dtype=np.float64)
    tol_cm = _boundary_snap_tol_cm(coords_cm)
    ok_all = True

    for bname in ("inlet", "outlet", "wall"):
        out_path = label_dir / f"{stem}_{bname}.txt"
        if out_path.is_file() and not force:
            continue
        dset_tag = bmap[bname]
        ref: np.ndarray | None = None
        for edim in (1, None):
            try:
                ref = sample_coords_from_dataset(model_java, dset_tag, edim=edim)
                break
            except Exception as exc:
                logger.debug("[i] %s: dataset %s edim=%s failed: %s", stem, dset_tag, edim, exc)
        if ref is None or ref.size == 0:
            ok_all = False
            continue

        tree = cKDTree(ref[:, :2])
        dist, _ = tree.query(pts_vol)
        mask = dist <= tol_cm
        coords = np.unique(pts_vol[mask], axis=0) if np.any(mask) else np.zeros((0, 2))
        lines = [
            f"% Model: mesh nodes snapped to dataset {dset_tag} (tol={tol_cm:.4g} cm)",
            "% x  y",
        ]
        for x, y in coords:
            lines.append(f"0 0 {x:.10f} {y:.10f}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(
            "[OK] %s: %s = %d mesh nodes (snap to '%s', %d ref pts, tol=%.4g cm)",
            stem,
            bname,
            len(coords),
            dset_tag,
            ref.shape[0],
            tol_cm,
        )

    return ok_all


def write_boundary_txt_from_boundary_datasets(
    model_java,
    label_dir: Path,
    stem: str,
    *,
    boundary_datasets: dict[str, str] | None = None,
    force: bool = False,
) -> bool:
    """Write boundary txt from Results datasets Inlet/Outlet/Wall (same as inlet_nodes export)."""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    bmap = boundary_datasets if boundary_datasets is not None else resolve_boundary_datasets(model_java)
    if not all(k in bmap for k in ("inlet", "outlet", "wall")):
        missing = [k for k in ("inlet", "outlet", "wall") if k not in bmap]
        logger.debug("[i] %s: boundary datasets missing %s", stem, missing)
        return False

    for bname in ("inlet", "outlet", "wall"):
        out_path = label_dir / f"{stem}_{bname}.txt"
        if out_path.is_file() and not force:
            continue
        dset_tag = bmap[bname]
        coords = sample_coords_from_dataset(model_java, dset_tag)
        unique = np.unique(coords[:, :2], axis=0) if coords.size else np.zeros((0, 2))
        lines = [f"% Model: COMSOL dataset {dset_tag}", "% x  y"]
        for x, y in unique:
            lines.append(f"0 0 {x:.10f} {y:.10f}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(
            "[OK] %s: %s boundary from dataset '%s' (%d unique coords)",
            stem,
            bname,
            dset_tag,
            len(unique),
        )
    return True


def write_boundary_txt_from_comsol_masks(
    model_java,
    coords_cm: np.ndarray,
    label_dir: Path,
    stem: str,
    *,
    dataset_tag: str = "dset1",
    threshold: float = 0.5,
    force: bool = False,
    preserve_existing_on_failure: bool = True,
) -> None:
    """Write inlet/outlet/wall ``.txt`` using COMSOL selection variables on mesh nodes."""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for bname, _default_expr in _BOUNDARY_SPECS:
        out_path = label_dir / f"{stem}_{bname}.txt"
        if out_path.is_file() and not force:
            continue

        expr_used: str | None = None
        mask: np.ndarray | None = None
        last_exc: Exception | None = None
        for expr in boundary_mask_expr_candidates(model_java, bname):
            try:
                vals = _evaluate_boundary_mask(
                    model_java,
                    coords_cm,
                    expr,
                    dataset_tag=dataset_tag,
                )
                mask = vals > threshold
                expr_used = expr
                break
            except Exception as exc:
                last_exc = exc
                logger.debug("[i] %s: boundary %s expr %s failed: %s", stem, bname, expr, exc)

        if mask is None:
            if preserve_existing_on_failure and out_path.is_file():
                logger.warning(
                    "[WARN] %s: COMSOL boundary %s failed (%s); keeping existing %s.",
                    stem,
                    bname,
                    last_exc,
                    out_path.name,
                )
                continue
            failures.append(f"{bname} ({last_exc})")
            continue

        if not np.any(mask):
            logger.warning(
                "[WARN] %s: no nodes matched %s for %s boundary.",
                stem,
                expr_used,
                bname,
            )
            coords = np.zeros((0, 2), dtype=np.float64)
        else:
            coords = np.unique(coords_cm[mask, :2], axis=0)

        lines = [f"% Model: COMSOL mask ({expr_used})", "% x  y"]
        for x, y in coords:
            lines.append(f"0 0 {x:.10f} {y:.10f}")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("[OK] %s: %s boundary via %s (%d unique coords)", stem, bname, expr_used, len(coords))

    write_wound_txt_from_comsol_mask(
        model_java,
        coords_cm,
        label_dir,
        stem,
        dataset_tag=dataset_tag,
        threshold=threshold,
        force=force,
    )

    if failures:
        raise RuntimeError(
            f"{stem}: COMSOL boundary mask export failed for: {', '.join(failures)}. "
            "Re-save the .mph with is_inlet/is_outlet/is_wall (sel1-sel3) or named selections "
            "box1/box2/dif1, or keep manual *_inlet/outlet/wall.txt exports."
        )


def ensure_boundary_txt_files(
    model_java,
    coords_cm: np.ndarray,
    mesh_path: Path,
    label_dir: Path,
    stem: str,
    *,
    vessel_cfg,
    dataset_tag: str = "dset1",
    force_boundary: bool = False,
) -> None:
    """Write missing boundary txt from Gmsh tags, COMSOL masks, or mesh-extent heuristic.

    Wound identity is always attempted afterwards, even when inlet/outlet/wall already exist.
    """
    from src.data_gen.lib.biochem_comsol_auto_export import write_boundary_txt_from_mesh

    label_dir = Path(label_dir)
    paths = tuple(label_dir / f"{stem}_{b}.txt" for b in ("inlet", "outlet", "wall"))
    try:
        if not (all(p.is_file() for p in paths) and not force_boundary):
            if mesh_has_gmsh_boundary_tags(mesh_path):
                write_boundary_txt_from_mesh(
                    mesh_path,
                    label_dir,
                    stem,
                    vessel_cfg=vessel_cfg,
                    force=force_boundary,
                )
            else:
                wrote = False
                try:
                    wrote = write_boundary_txt_from_mesh_snap_to_datasets(
                        model_java,
                        coords_cm,
                        label_dir,
                        stem,
                        force=force_boundary,
                    )
                except Exception as exc:
                    logger.warning(
                        "[WARN] %s: mesh snap to boundary datasets failed (%s); trying raw dataset coords.",
                        stem,
                        exc,
                    )
                if not wrote:
                    try:
                        wrote = write_boundary_txt_from_boundary_datasets(
                            model_java,
                            label_dir,
                            stem,
                            force=force_boundary,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[WARN] %s: COMSOL boundary datasets failed (%s); trying mask expressions.",
                            stem,
                            exc,
                        )
                if not wrote:
                    try:
                        write_boundary_txt_from_comsol_masks(
                            model_java,
                            coords_cm,
                            label_dir,
                            stem,
                            dataset_tag=dataset_tag,
                            force=force_boundary,
                            preserve_existing_on_failure=not force_boundary,
                        )
                    except Exception as exc:
                        if all(p.is_file() for p in paths):
                            logger.warning(
                                "[WARN] %s: COMSOL boundary masks unavailable (%s); using existing boundary txt.",
                                stem,
                                exc,
                            )
                        else:
                            logger.warning(
                                "[WARN] %s: COMSOL boundary masks failed (%s); trying mesh-extent heuristic.",
                                stem,
                                exc,
                            )
                            write_boundary_txt_from_axis_extents(
                                coords_cm,
                                label_dir,
                                stem,
                                force=force_boundary,
                            )
    finally:
        ensure_wound_boundary_txt(
            model_java,
            coords_cm,
            mesh_path,
            label_dir,
            stem,
            vessel_cfg=vessel_cfg,
            dataset_tag=dataset_tag,
            force=force_boundary,
        )



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

    from src.data_gen.lib.biochem_comsol_mph_export import find_comp1_mesh_tag

    tag = mesh_tag or find_comp1_mesh_tag(model_java)
    safe_nas = str(nas_path.resolve()).replace("\\", "/")
    logger.info("[NEW] %s: exporting COMSOL mesh -> %s", stem, nas_path.name)
    model_java.mesh(tag).export(safe_nas)

    if not nas_path.is_file() or nas_path.stat().st_size == 0:
        raise RuntimeError(f"COMSOL mesh export did not create {nas_path}")

    try:
        m = meshio.read(nas_path)
        if int(m.points.shape[0]) >= 3 and int(m.points.shape[1]) >= 2:
            meshio.write(msh_path, m, file_format="gmsh")
            logger.info("[OK] %s: wrote %s from NASTRAN export", stem, msh_path.name)
    except Exception as exc:
        logger.warning("[WARN] %s: .nas -> .msh conversion failed (%s); using .nas", stem, exc)
        if msh_path.is_file():
            try:
                msh_path.unlink()
            except Exception:
                pass

    scaffold_anchor_sidecars(raw_dir, unit="cm", force=False, dry_run=False)
    sidecar = raw_dir / f"{stem}.json"
    if not sidecar.is_file():
        sidecar.write_text(json.dumps({"unit": "cm", "level": 2}), encoding="utf-8")

    out = msh_path if msh_path.is_file() else nas_path
    return out, True
