"""Discover COMSOL result datasets (sol1 domain + Inlet/Outlet/Wall boundaries)."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_BOUNDARY_NAMES = ("inlet", "outlet", "wall")
_SKIP_DOMAIN_LABEL_PARTS = (
    "inlet",
    "outlet",
    "wall",
    "oracle",
    "filtro",
    "filter",
    "cut",
    "revolve",
    "mirror",
)


def _safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _dataset_label(ds) -> str:
    for getter in (lambda: ds.label(), lambda: ds.name()):
        val = _safe_call(getter, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _dataset_solution_tag(ds) -> str | None:
    for prop in ("solution", "sol"):
        val = _safe_call(lambda p=prop: ds.getString(p), None)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def list_comsol_datasets(model_java) -> list[dict[str, Any]]:
    """Return metadata for each Results > Dataset node."""
    root = model_java.result().dataset()
    try:
        tag_list = [str(t) for t in root.tags()]
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for tag in tag_list:
        tag_s = str(tag)
        ds = _safe_call(lambda t=tag_s: root.get(t), None)
        if ds is None:
            continue
        out.append(
            {
                "tag": tag_s,
                "label": _dataset_label(ds),
                "solution": _dataset_solution_tag(ds),
                "type": _safe_call(lambda: ds.getDatasetType(), None),
            }
        )
    return out


def _label_is_boundary_dataset(label: str, tag: str) -> str | None:
    ll = label.lower().strip()
    tl = tag.lower().strip()
    for bname in _BOUNDARY_NAMES:
        if ll == bname or tl == bname:
            return bname
        if ll.endswith(f" {bname}") or f"_{bname}" in tl:
            return bname
    return None


def _label_is_domain_candidate(label: str) -> bool:
    ll = label.lower()
    return not any(part in ll for part in _SKIP_DOMAIN_LABEL_PARTS)


def resolve_solution_dataset(
    model_java,
    sol_tag: str,
    *,
    explicit: str | None = None,
) -> str:
    """Pick the Results dataset tied to ``sol_tag`` (default biochem: ``sol1`` / Study 1)."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()

    env = (os.environ.get("BIOCHEM_COMSOL_DATASET_TAG") or "").strip()
    if env:
        return env

    sol_tag = str(sol_tag).strip()
    datasets = list_comsol_datasets(model_java)
    if not datasets:
        logger.warning("[WARN] No COMSOL datasets found; falling back to dset1.")
        return "dset1"

    for row in datasets:
        if row.get("solution") == sol_tag and _label_is_domain_candidate(str(row.get("label", ""))):
            logger.info(
                "[i] Domain dataset '%s' (%s) <- solution %s",
                row["tag"],
                row.get("label") or row["tag"],
                sol_tag,
            )
            return str(row["tag"])

    for row in datasets:
        label = str(row.get("label", ""))
        ll = label.lower()
        if not _label_is_domain_candidate(label):
            continue
        if "study 1" in ll and sol_tag.lower() in ll:
            logger.info("[i] Domain dataset '%s' (%s) <- label match Study 1", row["tag"], label)
            return str(row["tag"])
        if "biochem" in ll and "solution 1" in ll:
            logger.info("[i] Domain dataset '%s' (%s) <- biochem Study 1", row["tag"], label)
            return str(row["tag"])

    non_boundary = [r for r in datasets if _label_is_boundary_dataset(str(r.get("label", "")), str(r["tag"])) is None]
    if len(non_boundary) == 1:
        row = non_boundary[0]
        logger.warning(
            "[WARN] Using sole non-boundary dataset '%s' (%s) for sol %s.",
            row["tag"],
            row.get("label"),
            sol_tag,
        )
        return str(row["tag"])

    logger.warning(
        "[WARN] Could not resolve dataset for %s; using dset1. Available: %s",
        sol_tag,
        ", ".join(f"{r['tag']}({r.get('label')})" for r in datasets),
    )
    return "dset1"


def _boundary_dataset_score(bname: str, label: str, tag: str) -> int:
    """Higher = better match. Deprioritize stray edge datasets, not ``edg*`` named Inlet/Outlet/Wall."""
    ll = label.lower().strip()
    tl = tag.lower().strip()
    if ll == bname or tl == bname:
        if tl.startswith("edg"):
            return 90
        return 100
    # phase2_template_nowound selections: box1=inlet, box2=outlet, dif1=wall
    legacy_tag = {"inlet": "box1", "outlet": "box2", "wall": "dif1"}.get(bname)
    if legacy_tag and tl == legacy_tag:
        return 95
    if tl.startswith("edg") and bname in ll:
        return 85
    if (tl.startswith("edg") or ll.startswith("edge")) and bname not in ll:
        return -100
    if ll.endswith(f" {bname}") or f"_{bname}" in tl:
        return 50
    if bname in ll.split():
        return 40
    if bname in ll:
        return 10
    return -1


def resolve_boundary_datasets(model_java) -> dict[str, str]:
    """Map inlet/outlet/wall -> dataset tags (prefer Results ``Inlet``/``Outlet``/``Wall``)."""
    env_map = {
        "inlet": (os.environ.get("BIOCHEM_COMSOL_INLET_DATASET") or "").strip(),
        "outlet": (os.environ.get("BIOCHEM_COMSOL_OUTLET_DATASET") or "").strip(),
        "wall": (os.environ.get("BIOCHEM_COMSOL_WALL_DATASET") or "").strip(),
    }
    found: dict[str, str] = {k: v for k, v in env_map.items() if v}
    best_score: dict[str, int] = {k: 10_000 for k in found}

    for row in list_comsol_datasets(model_java):
        label = str(row.get("label", ""))
        tag = str(row["tag"])
        for bname in _BOUNDARY_NAMES:
            sc = _boundary_dataset_score(bname, label, tag)
            if sc < 0:
                continue
            prev = best_score.get(bname, -10_000)
            if sc > prev:
                best_score[bname] = sc
                found[bname] = tag
    return found


def sample_coords_from_dataset(
    model_java,
    dataset_tag: str,
    *,
    exprs: tuple[str, ...] = ("x", "y"),
    edim: int | None = 1,
) -> np.ndarray:
    """Sample coordinates on a boundary dataset (Points: From dataset), like inlet_nodes export."""
    results = model_java.result()
    tag = "py_bnd_dset"
    try:
        interp = results.numerical().create(tag, "Interp")
        interp.set("data", dataset_tag)
        interp.set("expr", list(exprs))
        if edim is not None:
            try:
                interp.set("edim", int(edim))
            except Exception:
                pass
        raw = interp.getData()
        if len(raw) < 2:
            raise ValueError(f"Dataset {dataset_tag}: expected x,y from Interp, got {len(raw)} arrays.")
        x = np.asarray(raw[0], dtype=np.float64).reshape(-1)
        y = np.asarray(raw[1], dtype=np.float64).reshape(-1)
        if x.size != y.size:
            raise ValueError(f"Dataset {dataset_tag}: x/y length mismatch {x.size} vs {y.size}.")
        coords = np.column_stack([x, y])
        try:
            crd = np.asarray(interp.getCoordinates(), dtype=np.float64)
            if crd.ndim == 2 and crd.shape[0] == 2 and crd.shape[1] == coords.shape[0]:
                coords = crd.T
        except Exception:
            pass
        return coords
    finally:
        try:
            results.numerical().remove(tag)
        except Exception:
            pass
