"""Run preconfigured COMSOL Results > Export nodes (sol_data, inlet_nodes, ...)."""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_EXPORT_TAGS: dict[str, str] = {
    "domain": "sol_data",
    "inlet": "inlet_nodes",
    "outlet": "outlet_nodes",
    "wall": "wall_nodes",
}

_ENV_EXPORT_KEYS = {
    "domain": "BIOCHEM_COMSOL_EXPORT_DOMAIN",
    "inlet": "BIOCHEM_COMSOL_EXPORT_INLET",
    "outlet": "BIOCHEM_COMSOL_EXPORT_OUTLET",
    "wall": "BIOCHEM_COMSOL_EXPORT_WALL",
}


def _env_flag(name: str, default_true: bool = True) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default_true
    return raw in ("1", "true", "yes", "on")


def use_mph_result_exports() -> bool:
    return _env_flag("BIOCHEM_COMSOL_USE_MPH_EXPORTS", default_true=True)


def resolve_export_tags() -> dict[str, str]:
    """Map logical roles -> COMSOL Export node tags in the .mph."""
    out = dict(_DEFAULT_EXPORT_TAGS)
    for role, env_name in _ENV_EXPORT_KEYS.items():
        val = (os.environ.get(env_name) or "").strip()
        if val:
            out[role] = val
    return out


def ensure_biochem_extract_dirs(
    raw_dir: Path,
    label_dir: Path,
    proc_dir: Path | None = None,
) -> None:
    """Create anchor mesh / COMSOL txt / graph output folders if missing."""
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    if proc_dir is not None:
        Path(proc_dir).mkdir(parents=True, exist_ok=True)


def list_result_export_tags(model_java) -> list[str]:
    try:
        return [str(t) for t in model_java.result().export().tags()]
    except Exception:
        return []


def _resolve_export_tag(model_java, preferred: str) -> str | None:
    tags = list_result_export_tags(model_java)
    if preferred in tags:
        return preferred
    low = preferred.lower()
    for tag in tags:
        if tag.lower() == low:
            return tag
    for tag in tags:
        if low in tag.lower():
            return tag
    return None


def find_comp1_mesh_tag(model_java) -> str:
    """Prefer ``comp1`` mesh sequence ``mesh1`` (same as GUI Mesh 1 under Component 1)."""
    preferred = (os.environ.get("BIOCHEM_COMSOL_MESH_TAG") or "").strip()
    if preferred:
        return preferred

    try:
        comp = model_java.component("comp1")
        comp_mesh = comp.mesh()
        for tag in comp_mesh.tags():
            tag_s = str(tag)
            if tag_s.lower() in ("mesh1", "mesh"):
                try:
                    n = int(comp_mesh(tag_s).getNumVertex())
                    if n >= 10:
                        logger.info("[i] Using comp1 mesh tag '%s' (%d vertices)", tag_s, n)
                        return tag_s
                except Exception:
                    continue
    except Exception:
        pass

    mesh_root = model_java.mesh()
    for tag in mesh_root.tags():
        tag_s = str(tag)
        if tag_s.lower() in ("mesh1", "mesh"):
            try:
                n = int(mesh_root(tag_s).getNumVertex())
                if n >= 10:
                    logger.info("[i] Using COMSOL mesh tag '%s' (%d vertices)", tag_s, n)
                    return tag_s
            except Exception:
                continue

    from src.data_gen.lib.biochem_comsol_mesh_export import find_comsol_mesh_tag

    return find_comsol_mesh_tag(model_java)


def _safe_set(export_node, key: str, value) -> None:
    try:
        export_node.set(key, value)
    except Exception:
        pass


def run_comsol_data_export(model_java, export_tag: str, dest_path: Path) -> None:
    """Run a Results > Export > Data node to ``dest_path``."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file():
        dest_path.unlink()

    resolved = _resolve_export_tag(model_java, export_tag)
    if resolved is None:
        raise KeyError(
            f"Export node '{export_tag}' not found. Available: {', '.join(list_result_export_tags(model_java))}"
        )

    exp = model_java.result().export(resolved)
    out_path = str(dest_path.resolve()).replace("\\", "/")
    _safe_set(exp, "filename", out_path)
    _safe_set(exp, "alwaysask", False)
    _safe_set(exp, "alwaysaskfilename", False)
    logger.info("[NEW] COMSOL export '%s' -> %s", resolved, dest_path.name)
    exp.run()

    if not dest_path.is_file() or dest_path.stat().st_size == 0:
        raise RuntimeError(f"COMSOL export '{resolved}' did not create {dest_path}")


def _normalize_boundary_export(src: Path, dest: Path) -> None:
    """Ensure boundary txt matches extractor format: ``%`` header + ``0 0 x y`` rows."""
    text = src.read_text(encoding="utf-8", errors="replace")
    lines_out = ["% Model: COMSOL Data export", "% x  y"]
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        parts = re.split(r"\s+", s)
        nums = [float(p) for p in parts if _is_float(p)]
        if len(nums) >= 2:
            x, y = nums[-2], nums[-1]
            lines_out.append(f"0 0 {x:.10f} {y:.10f}")
    if len(lines_out) <= 2:
        raise ValueError(f"Boundary export {src.name} has no coordinate rows.")
    dest.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _validate_domain_export(path: Path) -> None:
    head = path.read_text(encoding="utf-8", errors="replace")[:8000]
    if "@ t=" not in head and "% x" not in head:
        raise ValueError(
            f"{path.name}: does not look like a wide COMSOL spreadsheet export (missing '% x' / '@ t=')."
        )


def pull_exports_via_mph_nodes(
    model_java,
    stem: str,
    *,
    label_dir: Path,
    raw_dir: Path,
    force: bool = False,
    mesh_tag: str | None = None,
) -> bool:
    """Export mesh + sol_data + inlet/outlet/wall_nodes into biochem anchor folders."""
    label_dir = Path(label_dir)
    raw_dir = Path(raw_dir)
    ensure_biochem_extract_dirs(raw_dir, label_dir)

    tags = resolve_export_tags()
    domain_p = label_dir / f"{stem}.txt"
    boundary_paths = {
        "inlet": label_dir / f"{stem}_inlet.txt",
        "outlet": label_dir / f"{stem}_outlet.txt",
        "wall": label_dir / f"{stem}_wall.txt",
    }

    need_domain = force or not domain_p.is_file()
    need_b = {k: force or not p.is_file() for k, p in boundary_paths.items()}

    nas_path = raw_dir / f"{stem}.nas"
    need_mesh = force or not nas_path.is_file()
    if need_mesh or os.environ.get("BIOCHEM_COMSOL_FORCE_MESH", "").strip().lower() in ("1", "true", "yes"):
        tag = mesh_tag or find_comp1_mesh_tag(model_java)
        safe_nas = str(nas_path.resolve()).replace("\\", "/")
        logger.info("[NEW] %s: exporting comp1 mesh -> %s", stem, nas_path.name)
        try:
            model_java.component("comp1").mesh(tag).export(safe_nas)
        except Exception:
            model_java.mesh(tag).export(safe_nas)
        if not nas_path.is_file():
            raise RuntimeError(f"Mesh export failed for {stem}")

        msh_path = raw_dir / f"{stem}.msh"
        try:
            import meshio

            m = meshio.read(nas_path)
            if m.points.shape[0] >= 3:
                meshio.write(msh_path, m)
        except Exception as exc:
            logger.warning("[WARN] %s: .nas -> .msh failed (%s)", stem, exc)

    tmp_dir = label_dir / "_mph_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        if need_domain:
            tmp_dom = tmp_dir / f"{stem}_domain.txt"
            run_comsol_data_export(model_java, tags["domain"], tmp_dom)
            _validate_domain_export(tmp_dom)
            shutil.copy2(tmp_dom, domain_p)
            logger.info("[OK] %s: domain from export '%s'", stem, tags["domain"])

        for bname, dest in boundary_paths.items():
            if not need_b[bname]:
                continue
            tmp_b = tmp_dir / f"{stem}_{bname}.txt"
            run_comsol_data_export(model_java, tags[bname], tmp_b)
            _normalize_boundary_export(tmp_b, dest)
            logger.info("[OK] %s: %s from export '%s'", stem, bname, tags[bname])
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return True
