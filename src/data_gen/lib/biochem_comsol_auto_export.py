"""Pull biochem anchor fields from a solved COMSOL ``.mph`` via LiveLink (``mph``).

Writes the same ``.txt`` layout that ``PatientDataExtractor`` expects under
``data/processed/cfd_results_biochem/``. Boundary ``*_inlet/outlet/wall.txt``
files are generated from Gmsh mesh tags (no COMSOL boundary export needed).

Typical workflow::

    1. Run the biochem study in COMSOL; save as ``comsol_models/phase2_nowound_XXX.mph``.
    2. Mesh ``.nas``/``.msh`` is auto-exported from the ``.mph`` when missing (no manual mesh export).
       (maps to anchor stem ``patientXXX``) or set ``BIOCHEM_COMSOL_MODEL``.
    3. ``python -m src.tools.extract_biochem_comsol --stem <stem> --from-comsol``

Requires: COMSOL 6.x + ``pip install mph`` (same as kinematics ``AnchorGenerator``).

``--force`` rewrites domain field txt only; existing mesh and boundary txt are kept unless
``BIOCHEM_COMSOL_FORCE_MESH=1`` or ``BIOCHEM_COMSOL_FORCE_BOUNDARY=1``.

Domain fields use the Results dataset linked to ``sol1`` (Study 1 biochemistry), not ``sol2``.
Boundaries prefer Results datasets ``Inlet`` / ``Outlet`` / ``Wall`` (same as inlet_nodes export).
Override: ``BIOCHEM_COMSOL_DATASET_TAG``, ``BIOCHEM_COMSOL_INLET_DATASET``, etc.
Boundary txt uses volume mesh nodes snapped to Inlet/Outlet/Wall datasets (``BIOCHEM_BOUNDARY_SNAP_CM``).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import meshio
import numpy as np

from src.config import PhysicsConfig, VesselConfig, biochem_comsol_time_cap_s
from src.data_gen.lib.biochem_comsol_datasets import resolve_solution_dataset
from src.utils.paths import comsol_models_dir, data_root

logger = logging.getLogger(__name__)

_BOUNDARY_SUFFIXES = ("_inlet", "_outlet", "_wall")


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")

# Internal column order (no x,y) written as COMSOL wide export fields.
DOMAIN_FIELD_NAMES: tuple[str, ...] = (
    "u",
    "v",
    "p",
    "mu_effective",
    "rp",
    "ap",
    "apr",
    "aps",
    "PT",
    "th",
    "at",
    "fg",
    "fi",
    "M",
    "Mas",
    "Mat",
)

_DEFAULT_DOMAIN_EXPRS: tuple[str, ...] = (
    "u",
    "v",
    "p",
    "mu_b*(mu1(Mat)+mu2(FI))",
    "rp",
    "ap",
    "apr",
    "aps",
    "PT",
    "th",
    "at",
    "fg",
    "fi",
    "M",
    "Mas",
    "Mat",
)


def _parse_expr_list(raw: str | None) -> tuple[str, ...]:
    if not raw or not str(raw).strip():
        return _DEFAULT_DOMAIN_EXPRS
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) != len(DOMAIN_FIELD_NAMES):
        raise ValueError(
            f"BIOCHEM_COMSOL_DOMAIN_EXPRS must have {len(DOMAIN_FIELD_NAMES)} comma-separated "
            f"expressions (got {len(parts)})."
        )
    return tuple(parts)


_PATIENT_STEM_RE = re.compile(r"^patient(\d+)$", re.IGNORECASE)
_PHASE2_NOWOUND_MPH_RE = re.compile(r"^phase2_nowound_(\d+)\.mph$", re.IGNORECASE)


def patient_stem_from_phase2_mph(path: Path) -> str | None:
    """Map ``phase2_nowound_008.mph`` -> ``patient008``."""
    m = _PHASE2_NOWOUND_MPH_RE.match(path.name)
    if not m:
        return None
    return f"patient{int(m.group(1)):03d}"


def resolve_stem_selection(
    raw: str,
    statuses: Sequence[tuple[str, ...]] | Sequence[object],
    *,
    stem_attr: str = "stem",
) -> list[str]:
    """Parse ``5,8-10``, ``patient005,patient008``, or ``5 8 9`` into anchor stem names.

    ``statuses`` is the status table order (1-based indices match the printed ``#`` column).
    """
    if not raw or not str(raw).strip():
        return []

    def _stem_at(i: int) -> str:
        row = statuses[i - 1]
        if isinstance(row, str):
            return row
        return str(getattr(row, stem_attr))

    seen: set[str] = set()
    out: list[str] = []

    def _add_stem(stem: str) -> None:
        if stem not in seen:
            seen.add(stem)
            out.append(stem)

    def _add_index(idx: int) -> None:
        if idx < 1 or idx > len(statuses):
            raise ValueError(f"Index {idx} out of range (1-{len(statuses)}).")
        _add_stem(_stem_at(idx))

    tokens = re.split(r"[\s,;]+", raw.strip())
    for token in tokens:
        if not token:
            continue
        m_patient = _PATIENT_STEM_RE.match(token.strip())
        if m_patient:
            stem_name = f"patient{int(m_patient.group(1)):03d}"
            table_hit = next(
                (_stem_at(i + 1) for i in range(len(statuses)) if _stem_at(i + 1).lower() == stem_name),
                stem_name,
            )
            _add_stem(table_hit)
            continue
        if re.fullmatch(r"\d+\s*-\s*\d+", token):
            a_str, b_str = re.split(r"\s*-\s*", token)
            a_i, b_i = int(a_str), int(b_str)
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            for idx in range(a_i, b_i + 1):
                _add_index(idx)
            continue
        if token.isdigit():
            _add_index(int(token))
            continue
        raise ValueError(f"Unrecognized stem token: {token!r}")

    return out


def collect_biochem_extract_stems(raw_dir: Path, label_dir: Path) -> list[str]:
    """Union of mesh stems, domain export stems, and ``phase2_nowound_*.mph`` (sorted)."""
    seen: set[str] = set()
    out: list[str] = []
    if raw_dir.is_dir():
        for ext in (".msh", ".nas"):
            for p in sorted(raw_dir.glob(f"*{ext}")):
                if p.stem not in seen:
                    seen.add(p.stem)
                    out.append(p.stem)
    if label_dir.is_dir():
        for p in sorted(label_dir.glob("*.txt")):
            stem = p.stem
            if any(stem.endswith(suf) for suf in _BOUNDARY_SUFFIXES):
                continue
            if stem not in seen:
                seen.add(stem)
                out.append(stem)
    for stem in stems_from_phase2_nowound_mph():
        if stem not in seen:
            seen.add(stem)
            out.append(stem)
    return sorted(out)


def stems_from_phase2_nowound_mph(models_dir: Path | None = None) -> list[str]:
    """Anchor stems implied by ``comsol_models/phase2_nowound_*.mph`` files."""
    root = models_dir if models_dir is not None else comsol_models_dir()
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.glob("phase2_nowound_*.mph")):
        stem = patient_stem_from_phase2_mph(p)
        if stem:
            out.append(stem)
    return out


def phase2_nowound_mph_name_for_stem(stem: str) -> str | None:
    """Map anchor stem ``patient007`` -> COMSOL filename ``phase2_nowound_007.mph``."""
    m = _PATIENT_STEM_RE.match(stem.strip())
    if not m:
        return None
    return f"phase2_nowound_{int(m.group(1)):03d}.mph"


def resolve_biochem_comsol_model_path(stem: str, explicit: Path | None = None) -> Path | None:
    """Return first existing ``.mph`` for ``stem`` (explicit path, env, then search dirs).

    Anchor stems ``patientXXX`` resolve to ``comsol_models/phase2_nowound_XXX.mph``
    (3-digit index, e.g. ``patient7`` -> ``phase2_nowound_007.mph``).
    """
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"COMSOL model not found: {p}")

    env = (os.environ.get("BIOCHEM_COMSOL_MODEL") or "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p.resolve()

    dr = data_root()
    models_dir = comsol_models_dir()
    candidates: list[Path] = [
        dr / "raw" / "biochem_anchors" / f"{stem}.mph",
        models_dir / "runs" / f"{stem}.mph",
        models_dir / f"{stem}.mph",
    ]
    mapped = phase2_nowound_mph_name_for_stem(stem)
    if mapped:
        candidates.append(models_dir / mapped)

    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def write_boundary_txt_from_mesh(
    mesh_path: Path,
    label_dir: Path,
    stem: str,
    *,
    vessel_cfg: VesselConfig | None = None,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Write inlet/outlet/wall ``.txt`` from Gmsh physical line tags on ``mesh_path``."""
    vessel_cfg = vessel_cfg or VesselConfig(phase="biochem_anchors")
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    inlet_p = label_dir / f"{stem}_inlet.txt"
    outlet_p = label_dir / f"{stem}_outlet.txt"
    wall_p = label_dir / f"{stem}_wall.txt"
    if not force and inlet_p.is_file() and outlet_p.is_file() and wall_p.is_file():
        return inlet_p, outlet_p, wall_p

    mesh = meshio.read(mesh_path)
    pts = np.asarray(mesh.points, dtype=np.float64)
    if pts.shape[1] < 2:
        raise ValueError(f"{mesh_path.name}: expected 2D point coordinates.")
    n_nodes = pts.shape[0]

    inlet_tag = vessel_cfg.TAGS["Inlet"]
    wall_tag = vessel_cfg.TAGS["Walls"]
    outlet_tags = {tag_id for name, tag_id in vessel_cfg.TAGS.items() if "Outlet" in name}

    masks = {
        "inlet": np.zeros(n_nodes, dtype=bool),
        "outlet": np.zeros(n_nodes, dtype=bool),
        "wall": np.zeros(n_nodes, dtype=bool),
    }
    try:
        line_cells = mesh.get_cells_type("line")
        line_tags = mesh.get_cell_data("gmsh:physical", "line")
    except Exception as exc:
        raise RuntimeError(
            f"{mesh_path.name}: cannot read Gmsh line tags for boundaries ({exc}). "
            "Re-export mesh with inlet/outlet/wall physical groups."
        ) from exc

    for j, tag_arr in enumerate(line_tags):
        if j >= len(line_cells):
            break
        tag = int(np.asarray(tag_arr).reshape(-1)[0])
        nodes = np.asarray(line_cells[j], dtype=np.int64).reshape(-1)
        nodes = nodes[(nodes >= 0) & (nodes < n_nodes)]
        if nodes.size == 0:
            continue
        if tag == inlet_tag:
            masks["inlet"][nodes] = True
        elif tag in outlet_tags:
            masks["outlet"][nodes] = True
        elif tag == wall_tag:
            masks["wall"][nodes] = True

    def _write_one(path: Path, node_mask: np.ndarray) -> None:
        coords = np.unique(pts[node_mask, :2], axis=0)
        lines = ["% Model: mesh-derived boundary", "% x  y"]
        for x, y in coords:
            lines.append(f"0 0 {x:.10f} {y:.10f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_one(inlet_p, masks["inlet"])
    _write_one(outlet_p, masks["outlet"])
    _write_one(wall_p, masks["wall"])
    logger.info(
        "[OK] %s: boundary txt from mesh (inlet=%d outlet=%d wall=%d unique coords)",
        stem,
        int(masks["inlet"].sum()),
        int(masks["outlet"].sum()),
        int(masks["wall"].sum()),
    )
    return inlet_p, outlet_p, wall_p


def write_wide_domain_txt(
    path: Path,
    *,
    times_s: Sequence[float],
    coords_xy_cm: np.ndarray,
    fields_by_time: dict[float, np.ndarray],
) -> None:
    """Write COMSOL spreadsheet-style domain export consumed by ``load_comsol_trajectory``."""
    n_rows = int(coords_xy_cm.shape[0])
    n_times = len(times_s)
    if n_times == 0:
        raise ValueError("times_s must not be empty.")
    for t in times_s:
        arr = fields_by_time[float(t)]
        if arr.shape != (n_rows, len(DOMAIN_FIELD_NAMES)):
            raise ValueError(
                f"time {t}: expected fields shape ({n_rows}, {len(DOMAIN_FIELD_NAMES)}), got {arr.shape}"
            )

    header_parts = ["% x"]
    for t in times_s:
        names = " ".join(["x", "y", *DOMAIN_FIELD_NAMES])
        header_parts.append(f"{names} @ t={t}")
    header = " ".join(header_parts) + "\n"

    body_lines: list[str] = []
    for i in range(n_rows):
        # COMSOL wide exports reserve two leading numeric columns before the first @ t= block.
        row_vals: list[float] = [0.0, 0.0]
        for t in times_s:
            row_vals.append(float(coords_xy_cm[i, 0]))
            row_vals.append(float(coords_xy_cm[i, 1]))
            row_vals.extend(float(x) for x in fields_by_time[float(t)][i, :])
        body_lines.append(" ".join(f"{v:.8g}" for v in row_vals) + "\n")

    path.write_text(header + "".join(body_lines), encoding="utf-8")


def _cap_times(times: Iterable[float]) -> list[float]:
    arr = sorted({float(t) for t in times})
    t_cap = biochem_comsol_time_cap_s()
    if t_cap is None:
        return arr
    kept = [t for t in arr if t <= float(t_cap)]
    if not kept:
        raise ValueError(f"No solution times <= BIOCHEM time cap {t_cap} s.")
    if len(kept) < len(arr):
        logger.info(
            "[i] truncating COMSOL times to t <= %.1f s (kept %d/%d)",
            float(t_cap),
            len(kept),
            len(arr),
        )
    return kept


def _discover_solution_times(model_java, sol_tag: str) -> list[float]:
    sol = model_java.sol(sol_tag)
    times: list[float] = []
    try:
        for p in sol.getPVals():
            times.append(float(p))
    except Exception:
        pass
    if times:
        return _cap_times(times)

    # Fallback: parse % Time comment lines from a stored table export (single-time steady).
    logger.warning(
        "[WARN] Could not read transient times from sol('%s'); using t=0 only. "
        "Set BIOCHEM_COMSOL_SOL_TAG if your solution uses another tag.",
        sol_tag,
    )
    return [0.0]


def _evaluate_at_coords_and_time(
    model_java,
    coords_cm: np.ndarray,
    exprs: Sequence[str],
    *,
    dataset_tag: str,
    time_value: float,
) -> np.ndarray:
    """Evaluate COMSOL expressions at 2D points (cm) for one output time."""
    import mph  # noqa: F401 - ensures LiveLink path is importable

    coords_T = np.asarray(coords_cm, dtype=np.float64).T
    n_pts = coords_T.shape[1]
    results = model_java.result()
    tag = "py_biochem_interp"
    try:
        interp = results.numerical().create(tag, "Interp")
        interp.set("data", dataset_tag)
        interp.set("expr", list(exprs))
        interp.setInterpolationCoordinates(coords_T.tolist())
        try:
            interp.set("t", float(time_value))
        except Exception:
            try:
                interp.set("looplevelinput", [["t", f"{time_value}"]])
            except Exception:
                pass
        raw = interp.getData()
        if len(raw) < len(exprs):
            raise ValueError(f"COMSOL Interp returned {len(raw)} fields, expected {len(exprs)}.")
        out = np.column_stack([np.asarray(raw[i], dtype=np.float64).reshape(-1) for i in range(len(exprs))])
        if out.shape[0] != n_pts:
            raise ValueError(f"Interp row count {out.shape[0]} != {n_pts} coordinates.")
        return out
    finally:
        try:
            results.numerical().remove(tag)
        except Exception:
            pass


class BiochemComsolAutoExporter:
    """Connect to COMSOL, sample solved fields on the anchor mesh, write ``cfd_results_biochem`` txt."""

    def __init__(
        self,
        *,
        label_dir: Path | None = None,
        raw_dir: Path | None = None,
        phys_cfg: PhysicsConfig | None = None,
        vessel_cfg: VesselConfig | None = None,
        domain_exprs: Sequence[str] | None = None,
        sol_tag: str | None = None,
        dataset_tag: str | None = None,
    ):
        self.vessel_cfg = vessel_cfg or VesselConfig(phase="biochem_anchors")
        self.phys_cfg = phys_cfg or PhysicsConfig(phase="biochem_anchors")
        dr = data_root()
        self.raw_dir = Path(raw_dir) if raw_dir else dr / "raw" / "biochem_anchors"
        self.label_dir = Path(label_dir) if label_dir else dr / "processed" / "cfd_results_biochem"
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.domain_exprs = tuple(domain_exprs) if domain_exprs else _parse_expr_list(
            os.environ.get("BIOCHEM_COMSOL_DOMAIN_EXPRS")
        )
        self.sol_tag = (sol_tag or os.environ.get("BIOCHEM_COMSOL_SOL_TAG") or "sol1").strip()
        self._dataset_tag_explicit = (dataset_tag or os.environ.get("BIOCHEM_COMSOL_DATASET_TAG") or "").strip()
        self.dataset_tag = self._dataset_tag_explicit or "dset1"
        self._client = None
        self._model = None

    def _resolve_mesh_path(self, stem: str) -> Path:
        msh = self.raw_dir / f"{stem}.msh"
        nas = self.raw_dir / f"{stem}.nas"
        if msh.is_file():
            return msh
        if nas.is_file():
            logger.warning(
                "[WARN] %s: using .nas for coordinates; prefer .msh for Gmsh boundary tags.",
                stem,
            )
            return nas
        raise FileNotFoundError(
            f"No mesh for {stem} under {self.raw_dir} (.msh/.nas required for node coordinates)."
        )

    def __enter__(self) -> BiochemComsolAutoExporter:
        import mph

        model_path = getattr(self, "_model_path", None)
        if model_path is None:
            raise RuntimeError("Set _model_path before entering context.")
        logger.info("[i] Connecting to COMSOL for %s", model_path.name)
        self._client = mph.start()
        self._model = self._client.load(str(model_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client is not None:
            logger.info("[i] Disconnecting from COMSOL.")
            self._client.clear()
        self._client = None
        self._model = None

    def export_stem(
        self,
        stem: str,
        *,
        model_path: Path | None = None,
        force: bool = False,
        boundary_from_mesh: bool = True,
    ) -> Path:
        """Write ``<stem>.txt`` (+ boundaries) from the solved ``.mph``; return domain txt path."""
        resolved = resolve_biochem_comsol_model_path(stem, model_path)
        if resolved is None:
            raise FileNotFoundError(
                f"No .mph for stem '{stem}'. Save the solved model as "
                f"{self.raw_dir / f'{stem}.mph'} or set BIOCHEM_COMSOL_MODEL."
            )

        domain_path = self.label_dir / f"{stem}.txt"
        skip_fields = domain_path.is_file() and not force

        self._model_path = resolved
        with self:
            assert self._model is not None
            self.dataset_tag = resolve_solution_dataset(
                self._model.java,
                self.sol_tag,
                explicit=self._dataset_tag_explicit or None,
            )
            from src.data_gen.lib.biochem_comsol_mesh_export import (
                ensure_anchor_mesh_from_comsol,
                ensure_boundary_txt_files,
            )
            from src.data_gen.lib.centerline_utils import resolve_anchor_mesh_path

            force_mesh = force and _env_flag("BIOCHEM_COMSOL_FORCE_MESH")
            mesh_path, _mesh_exported = ensure_anchor_mesh_from_comsol(
                self._model.java,
                stem,
                self.raw_dir,
                force=force_mesh,
            )
            mesh = meshio.read(mesh_path)
            coords_cm = np.asarray(mesh.points[:, :2], dtype=np.float64)

            if boundary_from_mesh:
                force_boundary = force and _env_flag("BIOCHEM_COMSOL_FORCE_BOUNDARY")
                ensure_boundary_txt_files(
                    self._model.java,
                    coords_cm,
                    mesh_path,
                    self.label_dir,
                    stem,
                    vessel_cfg=self.vessel_cfg,
                    dataset_tag=self.dataset_tag,
                    force_boundary=force_boundary,
                )

            if skip_fields:
                logger.info("[skip] %s exists (use force=True to rewrite fields)", domain_path.name)
                return domain_path

            if resolve_anchor_mesh_path(self.raw_dir, stem) is None:
                raise RuntimeError(f"{stem}: mesh export failed under {self.raw_dir}")
            times = _discover_solution_times(self._model.java, self.sol_tag)
            logger.info(
                "[i] %s: evaluating %d time step(s) on %d mesh nodes (dataset=%s, sol=%s)",
                stem,
                len(times),
                len(coords_cm),
                self.dataset_tag,
                self.sol_tag,
            )

            fields_by_time: dict[float, np.ndarray] = {}
            for t in times:
                block = _evaluate_at_coords_and_time(
                    self._model.java,
                    coords_cm,
                    self.domain_exprs,
                    dataset_tag=self.dataset_tag,
                    time_value=float(t),
                )
                if block.shape[1] != len(DOMAIN_FIELD_NAMES):
                    raise ValueError(f"Field width mismatch at t={t}: {block.shape[1]}")
                fields_by_time[float(t)] = block
                nan_frac = float(np.isnan(block).mean())
                if nan_frac > 0.01:
                    logger.warning(
                        "[WARN] %s @ t=%.4g: %.1f%% NaN in sampled fields (check mesh/solution alignment).",
                        stem,
                        t,
                        100.0 * nan_frac,
                    )

        write_wide_domain_txt(
            domain_path,
            times_s=times,
            coords_xy_cm=coords_cm,
            fields_by_time=fields_by_time,
        )
        logger.info("[OK] Wrote domain export %s (%d times, %d nodes)", domain_path, len(times), len(coords_cm))
        return domain_path


def pull_biochem_comsol_exports(
    stem: str,
    *,
    label_dir: Path | None = None,
    raw_dir: Path | None = None,
    model_path: Path | None = None,
    force: bool = False,
) -> Path:
    """One-shot: COMSOL -> ``cfd_results_biochem/<stem>.txt`` (+ boundary txt from mesh)."""
    exporter = BiochemComsolAutoExporter(label_dir=label_dir, raw_dir=raw_dir)
    return exporter.export_stem(stem, model_path=model_path, force=force)
