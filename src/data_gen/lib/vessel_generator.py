"""
vessel_generator.py
-------------------
Generates 2D vessel meshes with parametric pathologies using Gmsh.

Performance design
~~~~~~~~~~~~~~~~~~
Gmsh holds global C++ state and is not thread-safe, so parallelism must be
achieved via *processes*, not threads.  The strategy is:

  1. The main process pre-samples ALL random parameters for every vessel and
     packages them as plain dicts (fully picklable, no Gmsh state).
  2. Worker processes each receive a *chunk* of those dicts.
  3. Every worker calls gmsh.initialize() once, iterates over its chunk, and
     calls gmsh.finalize() before exiting — keeping Gmsh init overhead at
     O(num_workers) rather than O(n).
  4. ProcessPoolExecutor dispatches chunks; tqdm tracks completion.

The result is near-linear scaling up to the physical core count.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from tqdm import tqdm

from src.config import VesselConfig
from src.utils.paths import get_project_root, migrate_legacy_vessel_meshes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def summarize_vessel_mesh_inventory(output_dir: Path) -> Dict[str, Any]:
    """Scan ``output_dir`` for ``vessel_*.json`` (or ``.msh``) and report append state.

    New runs default to **append**: indices continue after ``max_idx`` so existing meshes
    are not overwritten. To regenerate from index 0, pass ``start_idx=0`` explicitly.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return {"count": 0, "max_idx": -1, "next_idx": 0}
    indices: List[int] = []
    for pat in ("vessel_*.json", "vessel_*.msh"):
        for p in output_dir.glob(pat):
            try:
                indices.append(int(p.stem.split("_")[1]))
            except (ValueError, IndexError):
                continue
    uniq = sorted(set(indices))
    if not uniq:
        return {"count": 0, "max_idx": -1, "next_idx": 0}
    mx = uniq[-1]
    return {
        "count": len(uniq),
        "max_idx": mx,
        "next_idx": mx + 1,
    }


def _next_vessel_index(output_dir: Path) -> int:
    return int(summarize_vessel_mesh_inventory(output_dir)["next_idx"])


def _params_by_idx(params_list: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(p["idx"]): p for p in params_list}


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Centerline generators
# All accept pre-sampled scalar parameters so they are deterministic given
# a params dict — no random calls inside.
# ---------------------------------------------------------------------------

def _centerline_straight(
    n: int, length: float, jitter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Slightly jittered straight line along +X."""
    x = np.linspace(0.0, length, n)
    y = np.zeros(n)
    y[2 : n - 2] = jitter
    pts = np.column_stack([x, y])
    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return pts, tangents / np.maximum(norms, 1e-9)


def _centerline_arc(
    n: int,
    length: float,
    angle_span: float,
    *,
    bend_sign: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Circular arc: starts at (0,0) pointing +X, sweeps by ``angle_span`` in the x–y plane.

    Default ``bend_sign=1`` matches the historical clockwise sweep (negative y at the tip for
    ``angle_span > 0``). Use ``bend_sign=-1`` to mirror across the x-axis (opposite vertical offset).
    ``radius = length / angle_span`` so arc length ~= ``length`` for small angles.
    """
    radius = length / max(angle_span, 1e-3)
    theta = np.linspace(0.0, angle_span, n)
    pts = np.column_stack([
        radius * np.sin(theta),
        radius * (np.cos(theta) - 1.0),
    ])
    bs = float(bend_sign)
    pts[:, 1] *= bs
    tangents = np.column_stack([np.cos(theta), -bs * np.sin(theta)])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return pts, tangents / np.maximum(norms, 1e-9)


def _centerline_s_curve(
    n: int, length: float, amplitude: float
) -> Tuple[np.ndarray, np.ndarray]:
    """S-shaped: one full sine period transverse to flow."""
    t = np.linspace(0.0, 1.0, n)
    pts = np.column_stack([t * length, amplitude * np.sin(2.0 * np.pi * t)])
    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return pts, tangents / np.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# Curve-type weight tables
# ---------------------------------------------------------------------------

_CURVE_WEIGHTS: Dict[int, Dict[str, float]] = {
    0: {"straight": 0.60, "arc": 0.30, "s_curve": 0.10, "hook": 0.00},
    1: {"straight": 0.15, "arc": 0.45, "s_curve": 0.20, "hook": 0.20},
}


def resolve_bend_sign_mode(explicit: Optional[str] = None) -> str:
    """
    ``down_only``: historical arcs (bend_sign=+1) and non-flipped S-curves (Apr-2026 style).
    ``bidirectional``: random mirror for L1+ arc/hook and signed S-curve (default since May 2026).
    """
    raw = (explicit or os.environ.get("KINEMATICS_BEND_SIGN_MODE", "bidirectional")).strip().lower()
    if raw in ("down_only", "down", "legacy", "historical", "apr26"):
        return "down_only"
    if raw in ("bidirectional", "both", "up_down", "default", "may26"):
        return "bidirectional"
    raise ValueError(
        f"Unknown bend sign mode {raw!r}; use 'down_only' or 'bidirectional' "
        "(or env KINEMATICS_BEND_SIGN_MODE)."
    )


def default_level_mix(n: int) -> Dict[int, int]:
    """Default kinematics cohort: mostly L0/L1, ~20% high-thrombus (L2)."""
    n = max(1, int(n))
    n2 = max(1, round(n * 0.2))
    rem = n - n2
    n0 = rem // 2
    n1 = rem - n0
    return {0: n0, 1: n1, 2: n2}


def parse_level_mix(spec: str, n: int) -> Dict[int, int]:
    """Parse ``n0,n1,n2`` counts; must sum to ``n``."""
    parts = [int(x.strip()) for x in str(spec).split(",")]
    if len(parts) != 3:
        raise ValueError("level_mix must have exactly three comma-separated integers (L0,L1,L2)")
    mix = {0: parts[0], 1: parts[1], 2: parts[2]}
    total = sum(mix.values())
    if total != n:
        raise ValueError(f"level_mix counts sum to {total}, expected n={n}")
    return mix


def cohort_levels(
    n: int,
    level: int,
    level_mix: Optional[Dict[int, int]],
    rng: np.random.Generator,
) -> List[int]:
    """Per-vessel geometry levels for one run (shuffled when ``level_mix`` is set)."""
    if level_mix is None:
        return [int(level)] * n
    total = sum(int(v) for v in level_mix.values())
    if total != n:
        raise ValueError(f"level_mix counts sum to {total}, expected n={n}")
    out: List[int] = []
    for lvl, cnt in sorted(level_mix.items()):
        out.extend([int(lvl)] * int(cnt))
    rng.shuffle(out)
    return out


# Target diameter occlusion at the stenosis peak (symmetric both-wall narrowing).
MAX_STENOSIS_DIAMETER_OCCLUSION = 0.75

_PATHOLOGY_MODE_CHOICES = ("random", "max_stenosis", "max_aneurysm")


def normalize_pathology_mode(mode: str | None) -> str | None:
    """Return a canonical pathology mode or ``None`` for default random sampling."""
    if mode is None:
        return None
    raw = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "": "random",
        "default": "random",
        "none": "random",
        "max_stenosis": "max_stenosis",
        "maxstenosis": "max_stenosis",
        "stenosis_max": "max_stenosis",
        "max_aneurysm": "max_aneurysm",
        "maxaneurysm": "max_aneurysm",
        "aneurysm_max": "max_aneurysm",
    }
    resolved = aliases.get(raw, raw)
    if resolved == "random":
        return None
    if resolved not in _PATHOLOGY_MODE_CHOICES[1:]:
        raise ValueError(
            f"Unknown pathology_mode {mode!r}; use one of: {', '.join(_PATHOLOGY_MODE_CHOICES)}"
        )
    return resolved


def stenosis_wall_offset_for_occlusion(
    width: float,
    occlusion_frac: float = MAX_STENOSIS_DIAMETER_OCCLUSION,
) -> float:
    """Negative wall offset magnitude (Gaussian peak, both walls) for diameter occlusion."""
    occlusion_frac = float(np.clip(occlusion_frac, 0.0, 0.95))
    lumen_frac = 1.0 - occlusion_frac
    return (lumen_frac - 1.0) * float(width) / 2.0


def _sample_params(
        idx: int,
        level: int,
        cfg: VesselConfig,
        rng: np.random.Generator,
        pathology_mode: str | None = None,
) -> Dict[str, Any]:
    """
    Draw ALL random numbers for one vessel and return a plain picklable dict.
    Level 2 triggers pro-thrombotic geometry (extreme expansions/stagnation zones).
    """
    # Level-driven mode: 2 => pro-thrombotic cohort shaping.
    pro_thrombotic = (level == 2)
    pathology_mode = normalize_pathology_mode(pathology_mode)

    if pro_thrombotic:
        # Eliminate straight vessels; favor sharp turns and hooks
        active = {"straight": 0.0, "arc": 0.20, "s_curve": 0.40, "hook": 0.40}
    else:
        weights_map = _CURVE_WEIGHTS.get(min(level, 1), _CURVE_WEIGHTS)
        active = {k: v for k, v in weights_map.items() if v > 0}

    keys = list(active.keys())
    probs = np.array(list(active.values()), dtype=float)
    probs /= probs.sum()
    curve_type = str(rng.choice(keys, p=probs))

    if pathology_mode == "max_stenosis":
        v_type = "stenosis"
    elif pathology_mode == "max_aneurysm":
        v_type = "aneurysm"
    elif pro_thrombotic:
        # Guarantee a pathology. Aneurysms (stagnation) and Stenosis (downstream deceleration)
        v_type = str(rng.choice(["stenosis", "aneurysm"], p=[0.3, 0.7]))
    else:
        v_type = str(rng.choice(["straight", "stenosis", "aneurysm"]))

    width = float(rng.uniform(cfg.width_min, cfg.width_max))
    n = cfg.num_ctrl_pts
    L = cfg.base_length
    t = np.linspace(0, 1, n)

    # 1. Main Clinical Pathology
    offsets = np.zeros(n)
    if v_type != "straight":
        if pathology_mode == "max_stenosis":
            mag = stenosis_wall_offset_for_occlusion(width)
        elif pathology_mode == "max_aneurysm":
            mult = 1.5 if pro_thrombotic else 1.0
            mag = float(cfg.aneurysm_factor_max * mult * width)
        elif v_type in ("stenosis", "occlusion"):
            mult = 1.2 if pro_thrombotic else 1.0  # Tighter stenosis
            mag = -float(
                rng.uniform(
                    cfg.stenosis_factor_min * width,
                    min(0.9, cfg.stenosis_factor_max * mult) * width,
                )
            )
        else:
            mult = 1.5 if pro_thrombotic else 1.0  # Deeper aneurysms
            mag = float(rng.uniform(cfg.aneurysm_factor_min * width, cfg.aneurysm_factor_max * mult * width))

        min_idx, max_idx = max(3, int(n * 0.2)), min(n - 4, int(n * 0.8))
        if pathology_mode in ("max_stenosis", "max_aneurysm"):
            peak = int(min_idx + 0.5 * (max_idx - min_idx))
        else:
            peak = int(rng.integers(min_idx, max_idx))

        if pathology_mode == "max_stenosis" or pro_thrombotic:
            # Sharper geometric transition to trigger sr_grad_flow < -750 (sgt)
            std_dev = float(rng.uniform(0.02 * n, 0.05 * n))
        else:
            std_dev = float(rng.uniform(0.04 * n, 0.10 * n))

        x_idx = np.arange(n)
        gauss = np.exp(-0.5 * ((x_idx - peak) / std_dev) ** 2)
        if pathology_mode in ("max_stenosis", "max_aneurysm"):
            skew = np.ones(n, dtype=float)
        else:
            skew_factor = float(rng.uniform(-0.3, 0.3))
            skew = 1.0 + skew_factor * ((x_idx - peak) / n)
        offsets = mag * gauss * skew

    if v_type == "straight":
        path_loc = 2
    elif pathology_mode == "max_stenosis":
        path_loc = 2  # symmetric both-wall narrowing for target occlusion
    else:
        path_loc = int(rng.choice([0, 1, 2]))

    # 2. Universal Centerline Tortuosity
    f1, f2 = rng.uniform(0.5, 1.5), rng.uniform(1.5, 2.5)
    meander = np.sin(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi)) + \
              0.5 * np.sin(2 * np.pi * f2 * t + rng.uniform(0, 2 * np.pi))

    # Higher tortuosity triggers separation on inner radii
    max_meander = (0.15 if pro_thrombotic else 0.10) * width
    meander = (meander / max(1e-9, float(np.max(np.abs(meander))))) * max_meander
    meander *= np.sin(np.pi * t)
    tortuosity = meander[2:n - 2].tolist()

    # 3. Independent Wall Roughness
    def get_wall_noise():
        if pro_thrombotic:
            # Higher frequency and amplitude to create local micro-cavities (sr < 25)
            f_h1, f_h2 = rng.uniform(2.0, 4.0), rng.uniform(4.0, 6.0)
            max_noise = 0.08 * width
        else:
            f_h1, f_h2 = rng.uniform(1.0, 2.5), rng.uniform(2.5, 4.0)
            max_noise = 0.05 * width

        noise = np.sin(2 * np.pi * f_h1 * t + rng.uniform(0, 2 * np.pi)) + \
                0.5 * np.sin(2 * np.pi * f_h2 * t + rng.uniform(0, 2 * np.pi))
        noise = (noise / max(1e-9, float(np.max(np.abs(noise))))) * max_noise
        noise *= np.sin(np.pi * t) ** 0.5
        return noise.tolist()

    noise_top = get_wall_noise()
    noise_bot = get_wall_noise()

    # 4. Safely Bound Parameters (Unchanged)
    max_half_width = (width / 2.0) + max(0, float(np.max(offsets))) + (0.08 * width)
    min_safe_radius = 1.6 * max_half_width
    max_safe_angle_span = L / min_safe_radius

    if curve_type == "straight":
        angle_span, amplitude = 0.0, 0.0
    elif curve_type in ("arc", "hook"):
        if curve_type == "arc":
            target_angle = float(rng.uniform(np.deg2rad(45), np.deg2rad(100)))
        else:
            target_angle = float(rng.uniform(np.deg2rad(100), np.deg2rad(125)))

        angle_span = min(target_angle, max_safe_angle_span)
        amplitude = 0.0
    else:  # s_curve
        angle_span = 0.0
        amp_mag = min(float(rng.uniform(0.003, 0.007)), L * 0.15)
        amplitude = amp_mag

    bend_mode = resolve_bend_sign_mode()
    bend_sign = 1.0
    if level >= 1:
        if curve_type in ("arc", "hook"):
            bend_sign = (
                1.0
                if bend_mode == "down_only"
                else float(rng.choice([-1.0, 1.0]))
            )
        elif curve_type == "s_curve":
            if bend_mode == "bidirectional":
                amplitude *= float(rng.choice([-1.0, 1.0]))
            else:
                amplitude = abs(float(amplitude))

    return {
        "idx": idx,
        "level": level,
        "v_type": v_type,
        "curve_type": curve_type,
        "width": width,
        "angle_span": angle_span,
        "amplitude": amplitude,
        "bend_sign": bend_sign,
        "bend_sign_mode": bend_mode,
        "jitter": [],
        "tortuosity": tortuosity,
        "noise_top": noise_top,
        "noise_bot": noise_bot,
        "offsets": offsets.tolist(),
        "path_loc": path_loc,
    }


def recompute_pathology_offsets(
    params: Dict[str, Any],
    cfg: VesselConfig,
    rng: np.random.Generator,
    *,
    strength: float = 1.0,
    path_loc_frac: float | None = None,
) -> Dict[str, Any]:
    """Recompute stenosis/aneurysm offsets when pathology type or strength changes."""
    out = dict(params)
    v_type = str(out.get("v_type", "straight"))
    width = float(out.get("width", cfg.width_min))
    level = int(out.get("level", 0))
    pro_thrombotic = level == 2
    n = cfg.num_ctrl_pts
    t = np.linspace(0, 1, n)
    strength = float(np.clip(strength, 0.0, 1.0))

    offsets = np.zeros(n)
    if v_type != "straight" and strength > 0.0:
        if v_type in ("stenosis", "occlusion"):
            mult = 1.2 if pro_thrombotic else 1.0
            mag = -float(
                rng.uniform(
                    cfg.stenosis_factor_min * width,
                    min(0.9, cfg.stenosis_factor_max * mult) * width,
                )
            )
        else:
            mult = 1.5 if pro_thrombotic else 1.0
            mag = float(
                rng.uniform(
                    cfg.aneurysm_factor_min * width,
                    cfg.aneurysm_factor_max * mult * width,
                )
            )
        mag *= strength

        min_idx, max_idx = max(3, int(n * 0.2)), min(n - 4, int(n * 0.8))
        if path_loc_frac is not None:
            peak = int(min_idx + float(np.clip(path_loc_frac, 0.0, 1.0)) * (max_idx - min_idx))
        else:
            peak = int(rng.integers(min_idx, max_idx))

        if pro_thrombotic:
            std_dev = float(rng.uniform(0.02 * n, 0.05 * n))
        else:
            std_dev = float(rng.uniform(0.04 * n, 0.10 * n))

        x_idx = np.arange(n)
        gauss = np.exp(-0.5 * ((x_idx - peak) / std_dev) ** 2)
        skew_factor = float(rng.uniform(-0.3, 0.3))
        skew = 1.0 + skew_factor * ((x_idx - peak) / n)
        offsets = mag * gauss * skew

    out["offsets"] = offsets.tolist()
    if v_type == "straight":
        out["path_loc"] = 2
    elif "path_loc" not in out:
        out["path_loc"] = int(rng.choice([0, 1, 2]))
    return out


def make_vessel_params(
    idx: int = 0,
    level: int = 0,
    cfg: VesselConfig | None = None,
    rng: np.random.Generator | None = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Sample one vessel parameter dict (same keys as ``_sample_params``)."""
    cfg = cfg or VesselConfig(phase="kinematics")
    rng = rng or np.random.default_rng(0)
    base = _sample_params(idx, level, cfg, rng)
    base.update(overrides)
    return base


def build_vessel_mesh(
    params: Dict[str, Any],
    cfg_dict: Dict[str, Any],
    output_dir: str | Path,
) -> Tuple[int, bool, str]:
    """Build and mesh one vessel via Gmsh; returns ``(idx, success, error_msg)``."""
    unit = cfg_dict.get("unit", "m")
    unit_scale = 100.0 if unit == "cm" else 1.0
    mesh_lc = cfg_dict["mesh_lc"] * unit_scale

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", cfg_dict["mesh_size_factor"])
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_lc)
        return _build_and_mesh(params, cfg_dict, str(output_dir))
    finally:
        gmsh.finalize()


def _build_and_mesh(
        params: Dict[str, Any],
        cfg_dict: Dict[str, Any],
        output_dir: str,
) -> Tuple[int, bool, str]:
    """Build, mesh, and save one vessel. Returns (idx, success, error_msg)."""
    from src.data_gen.lib.vessel_geometry import (
        GeometryValidationError,
        compute_geometry_from_params,
        compute_geometry_from_walls,
        validate_geometry,
    )

    idx = int(params["idx"])
    try:
        if str(params.get("geometry_mode", "parametric")) == "edited_walls":
            top = np.asarray(params["top_coords"], dtype=float)
            bot = np.asarray(params["bot_coords"], dtype=float)
            geom = compute_geometry_from_walls(
                top,
                bot,
                idx=idx,
                unit=str(cfg_dict.get("unit", "m")),
                params=params,
                base_length=float(cfg_dict["base_length"]),
            )
        else:
            geom = compute_geometry_from_params(params, cfg_dict)
        validate_geometry(geom, cfg_dict)
        return _mesh_geometry(geom, cfg_dict, output_dir)
    except GeometryValidationError as exc:
        return idx, False, str(exc)
    except Exception as exc:
        try:
            gmsh.model.remove()
        except Exception:
            pass
        return idx, False, str(exc)


def _mesh_geometry(
    geom,
    cfg_dict: Dict[str, Any],
    output_dir: str,
) -> Tuple[int, bool, str]:
    """Gmsh meshing + file write from a ``VesselGeometry``."""
    idx = int(geom.idx)
    out = Path(output_dir)
    lc = float(cfg_dict["mesh_lc"])
    unit = str(cfg_dict.get("unit", "m"))
    unit_scale = 100.0 if unit == "cm" else 1.0
    if unit_scale != 1.0:
        lc *= unit_scale

    top_coords = geom.top_coords
    bot_coords = geom.bot_coords

    try:
        gmsh.model.add(f"vessel_{idx}")

        top_tags = [gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0, lc) for p in top_coords]
        bot_tags = [gmsh.model.geo.addPoint(float(p[0]), float(p[1]), 0.0, lc) for p in bot_coords]

        s_top = gmsh.model.geo.addBSpline(top_tags)
        l_out = gmsh.model.geo.addLine(top_tags[-1], bot_tags[-1])
        s_bot = gmsh.model.geo.addBSpline(list(reversed(bot_tags)))
        l_in = gmsh.model.geo.addLine(bot_tags[0], top_tags[0])

        cl = gmsh.model.geo.addCurveLoop([s_top, l_out, s_bot, l_in])
        s = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        tags = cfg_dict["TAGS"]
        gmsh.model.addPhysicalGroup(1, [l_in], tags["Inlet"], name="Inlet")
        gmsh.model.addPhysicalGroup(1, [l_out], tags["Outlet_1"], name="Outlet_1")
        gmsh.model.addPhysicalGroup(1, [s_top, s_bot], tags["Walls"], name="Walls")
        gmsh.model.addPhysicalGroup(2, [s], tags["Fluid_Domain"], name="Fluid_Domain")

        gmsh.model.mesh.generate(2)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) < 50:
            raise RuntimeError(f"Too few nodes ({len(node_tags)})")

        gmsh.write(str(out / f"vessel_{idx}.msh"))
        gmsh.write(str(out / f"vessel_{idx}.nas"))
        gmsh.model.remove()

        with open(out / f"vessel_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(geom.meta, f, indent=4)

        return idx, True, ""
    except Exception as exc:
        try:
            gmsh.model.remove()
        except Exception:
            pass
        return idx, False, str(exc)


def _worker_run_chunk(
    chunk: List[Dict[str, Any]],
    cfg_dict: Dict[str, Any],
    output_dir: str,
) -> List[Tuple[int, bool, str]]:
    """
    Initialise Gmsh ONCE per process, process every sample in the chunk,
    then finalise.  Gmsh init/finalize is the heaviest fixed cost, so
    processing multiple samples per worker amortises it.
    """
    unit = cfg_dict.get("unit", "m")
    unit_scale = 100.0 if unit == "cm" else 1.0
    mesh_lc = cfg_dict["mesh_lc"] * unit_scale

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",          0)
    gmsh.option.setNumber("Mesh.Algorithm",            6)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Smoothing",            5)
    gmsh.option.setNumber("Mesh.MshFileVersion",       2.2)
    gmsh.option.setNumber("Mesh.Binary",               0)
    gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",    1)
    gmsh.option.setNumber("Mesh.SaveAll",              0)
    gmsh.option.setNumber("Mesh.MeshSizeFactor",       cfg_dict["mesh_size_factor"])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_lc)

    results = [_build_and_mesh(p, cfg_dict, output_dir) for p in chunk]

    gmsh.finalize()
    return results


class VesselGenerator:
    """Generates 2D vessel meshes with parametric pathologies using Gmsh."""

    def __init__(self, phase: str = "kinematics", output_dir: Optional[str | Path] = None) -> None:
        self.cfg          = VesselConfig(phase=phase)
        self.project_root = get_project_root()
        self.output_dir   = Path(output_dir) if output_dir else self.project_root / self.cfg.mesh_input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir is None:
            migrate_legacy_vessel_meshes(self.output_dir)

    def _cfg_dict(self) -> Dict[str, Any]:
        return {
            "num_ctrl_pts":       self.cfg.num_ctrl_pts,
            "base_length":        self.cfg.base_length,
            "mesh_lc":            self.cfg.mesh_lc,
            "mesh_size_factor":   self.cfg.mesh_size_factor,
            "width_min":          self.cfg.width_min,
            "width_max":          self.cfg.width_max,
            "stenosis_factor_min": self.cfg.stenosis_factor_min,
            "stenosis_factor_max": self.cfg.stenosis_factor_max,
            "min_lumen_width_fraction": self.cfg.min_lumen_width_fraction,
            "aneurysm_factor_min": self.cfg.aneurysm_factor_min,
            "aneurysm_factor_max": self.cfg.aneurysm_factor_max,
            "TAGS":               dict(self.cfg.TAGS),
        }

    # ------------------------------------------------------------------
    # Visualisation  (reads saved .msh files; main-process only)
    # ------------------------------------------------------------------

    def visualize_saved(self, indices: List[int], max_plots: int = 9) -> None:
        """Load and plot already-saved .msh files."""
        import meshio

        indices = indices[:max_plots]
        cols = min(3, len(indices))
        rows = math.ceil(len(indices) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for ax, idx in zip(axes, indices):
            path = self.output_dir / f"vessel_{idx}.msh"
            meta_path = self.output_dir / f"vessel_{idx}.json"
            if not path.exists():
                ax.set_visible(False)
                continue
            try:
                mesh  = meshio.read(str(path))
                nodes = mesh.points[:, :2]
                tris  = mesh.cells_dict.get("triangle")
                if tris is None:
                    ax.set_visible(False)
                    continue
                poly = PolyCollection(
                    nodes[tris], edgecolors="black", facecolors="lightblue", linewidths=0.1
                )
                ax.add_collection(poly)
                ax.autoscale_view()
                ax.set_aspect("equal")
                title = f"vessel_{idx}"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if "d_inlet" in meta:
                            inlet_diameter = float(meta["d_inlet"])
                            if str(meta.get("unit", "m")).lower() == "m":
                                inlet_diameter_cm = inlet_diameter * 100.0
                            else:
                                inlet_diameter_cm = inlet_diameter
                            title = f"{title} | d_inlet={inlet_diameter_cm:.2f} cm"
                    except Exception:
                        pass
                ax.set_title(title, fontsize=8)
            except Exception:
                ax.set_title(f"vessel_{idx} ERROR", fontsize=8)

        for ax in axes[len(indices):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        n: int = 50,
        level: int = 0,
        level_mix: Optional[Dict[int, int]] = None,
        max_retries: int = 3,
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        start_idx: Optional[int] = None,
        unit: str = "m",
        pathology_mode: str | None = None,
    ) -> None:
        """
        Parallel batch vessel generation.

        Parameters
        ----------
        n           : number of vessels to produce in this run (file indices ``start_idx`` …)
        level       : geometry complexity when ``level_mix`` is None
                      (0 = mostly straight, 1 = curved, 2 = pro-thrombotic / high-clot)
        level_mix   : optional per-level counts ``{0: n0, 1: n1, 2: n2}`` (must sum to ``n``);
                      shuffled across indices for a mixed cohort in one run
        max_retries : retry attempts for failed samples (same ``idx`` as the failure; new parameters)
        num_workers : worker processes (default: cpu_count - 1, min 1)
        chunk_size  : samples per worker chunk (default: auto-balanced)
        seed        : integer seed for reproducibility (None = random)
        start_idx   : first vessel index ``vessel_{idx}.*``. If ``None``, appends after the
                      highest existing index in ``output_dir`` (no overwrite). Pass ``0`` to
                      fill from the beginning (may overwrite existing files).
        pathology_mode : ``None``/``random`` (default), ``max_stenosis`` (~75% diameter
                         occlusion at peak), or ``max_aneurysm`` (maximum expansion).
        """
        pathology_mode = normalize_pathology_mode(pathology_mode)
        phys_cores  = os.cpu_count() or 1
        num_workers = max(1, phys_cores - 1) if num_workers is None else num_workers
        num_workers = min(num_workers, n)

        if start_idx is None:
            start_idx = _next_vessel_index(self.output_dir)

        if level_mix is not None:
            mix_msg = ", ".join(f"L{k}={v}" for k, v in sorted(level_mix.items()))
            logger.info(
                f"Generating {n} mixed-level vessels ({mix_msg}) → {self.output_dir} "
                f"[indices {start_idx}..{start_idx + n - 1}] "
                f"[{num_workers} workers / {phys_cores} logical cores]"
            )
        else:
            logger.info(
                f"Generating {n} Level-{level} vessels → {self.output_dir} "
                f"[indices {start_idx}..{start_idx + n - 1}] "
                f"[{num_workers} workers / {phys_cores} logical cores]"
            )
        if pathology_mode:
            logger.info("Pathology mode: %s", pathology_mode)

        cfg_d   = self._cfg_dict()
        cfg_d["unit"] = unit
        out_str = str(self.output_dir)
        rng     = np.random.default_rng(seed)

        # Pre-sample everything in the main process
        per_vessel_levels = cohort_levels(n, level, level_mix, rng)
        all_params = [
            _sample_params(
                start_idx + i,
                per_vessel_levels[i],
                self.cfg,
                rng,
                pathology_mode=pathology_mode,
            )
            for i in range(n)
        ]
        params_lookup = _params_by_idx(all_params)

        # Split into balanced chunks — larger chunks = less IPC overhead
        if chunk_size is None:
            chunk_size = max(1, math.ceil(n / num_workers))
        chunks = [all_params[i : i + chunk_size] for i in range(0, n, chunk_size)]

        # ---- Dispatch ----
        generated: int = 0
        failed_params: List[Dict[str, Any]] = []

        if num_workers == 1:
            logger.info("Executing sequentially in main process to avoid Windows spawn issues.")
            # We can just call the worker function directly
            results = _worker_run_chunk(all_params, cfg_d, out_str)

            # FIX: Initialize the progress bar (pbar) for the single-worker loop
            with tqdm(total=n, desc="Generating vessels", unit="vessel") as pbar:
                for idx, success, err in results:
                    pbar.update(1)
                    if success:
                        generated += 1
                    else:
                        logger.warning(f"[ {idx} ] failed: {err}")
                        if idx in params_lookup:
                            failed_params.append(params_lookup[idx])

        else:
            # Existing multiprocessing logic for num_workers > 1
            with mp.Pool(processes=num_workers) as pool:
                # Submit all chunks
                async_results = [
                    (pool.apply_async(_worker_run_chunk, (chunk, cfg_d, out_str)), chunk)
                    for chunk in chunks
                ]

                with tqdm(total=n, desc="Generating vessels", unit="vessel") as pbar:
                    for async_result, chunk in async_results:
                        try:
                            # Force a hard timeout (e.g., 60 seconds per chunk)
                            # Adjust time based on your average mesh generation speed
                            results = async_result.get(timeout=60)

                            for idx, success, err in results:
                                pbar.update(1)
                                if success:
                                    generated += 1
                                else:
                                    logger.warning(f"[ {idx} ] failed: {err}")
                                    if idx in params_lookup:
                                        failed_params.append(params_lookup[idx])

                        except mp.TimeoutError:
                            logger.error(f"Worker hung (Timeout) — {len(chunk)} samples queued for retry")
                            failed_params.extend(chunk)
                            pbar.update(len(chunk))
                            continue
                        except Exception as exc:
                            logger.error(f"Worker crash: {exc} — {len(chunk)} samples queued for retry")
                            failed_params.extend(chunk)
                            pbar.update(len(chunk))
                            continue

        # ---- Retry failed samples ----
        # Resample geometry parameters but keep the same vessel idx so outputs stay
        # vessel_{start_idx}..vessel_{start_idx+n-1} (replacement, no extra indices).
        for retry_round in range(1, max_retries + 1):
            if not failed_params:
                break
            logger.info(f"Retry {retry_round}/{max_retries}: {len(failed_params)} samples")

            retry_batch = [
                _sample_params(
                    int(failed_p["idx"]),
                    int(failed_p.get("level", level)),
                    self.cfg,
                    rng,
                    pathology_mode=pathology_mode,
                )
                for failed_p in failed_params
            ]

            still_failed: List[Dict[str, Any]] = []
            retry_chunks = [retry_batch[i: i + chunk_size] for i in
                            range(0, len(retry_batch), chunk_size)]

            # [FIX 1: Indented inside the loop]
            # [FIX 2: Swapped to mp.Pool to prevent C++ deadlocks during retries]
            with mp.Pool(processes=min(num_workers, len(retry_batch))) as pool:
                async_results = [
                    (pool.apply_async(_worker_run_chunk, (chunk, cfg_d, out_str)), chunk)
                    for chunk in retry_chunks
                ]

                for async_result, chunk in async_results:
                    try:
                        # Apply the same 60-second timeout here
                        results = async_result.get(timeout=60)

                        for idx, success, err in results:
                            if success:
                                generated += 1
                            else:
                                # Find the original param dict to retry again
                                failed_p = next((p for p in chunk if p["idx"] == idx), None)
                                if failed_p:
                                    still_failed.append(failed_p)

                    except mp.TimeoutError:
                        logger.error(f"Retry worker hung (Timeout) — {len(chunk)} samples failed")
                        still_failed.extend(chunk)
                    except Exception as exc:
                        logger.error(f"Retry worker crash: {exc}")
                        still_failed.extend(chunk)

            failed_params = still_failed

        if failed_params:
            logger.warning(
                f"{len(failed_params)} samples could not be generated after {max_retries} retries."
            )

        logger.info(f"Done. {generated}/{n} vessels saved.")


class VesselGeneratorPhase3(VesselGenerator):
    """Synthetic Phase-3 vessel cohort (same geometry pipeline as ``VesselGenerator(phase='biochem')``)."""

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        super().__init__(phase="biochem", output_dir=output_dir)

    def run_pipeline(
        self,
        n: int = 50,
        level: int = 0,
        max_retries: int = 3,
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        start_idx: Optional[int] = None,
        unit: str = "m",
        pathology_mode: str | None = None,
    ) -> None:
        if start_idx is None:
            start_idx = 0
        return super().run_pipeline(
            n=n,
            level=level,
            max_retries=max_retries,
            num_workers=num_workers,
            chunk_size=chunk_size,
            seed=seed,
            start_idx=start_idx,
            unit=unit,
            pathology_mode=pathology_mode,
        )


def _prompt_int_choice(label: str, allowed: Tuple[int, ...]) -> int:
    """Read an integer from stdin until it is one of ``allowed``."""
    allowed_str = "/".join(str(x) for x in allowed)
    while True:
        raw = input(f"{label} ({allowed_str}): ").strip()
        try:
            v = int(raw)
        except ValueError:
            print(f"  Enter an integer: {allowed_str}")
            continue
        if v in allowed:
            return v
        print(f"  Must be one of: {allowed_str}")


def _prompt_positive_int(label: str, default: int = 500) -> int:
    """Read a positive integer from stdin; empty input returns ``default``."""
    while True:
        raw = input(f"{label} (>=1) [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            print("  Enter a positive integer.")
            continue
        if v >= 1:
            return v
        print("  Must be at least 1.")


def _prompt_write_mode_vessel() -> bool:
    """Return True to overwrite from index 0, False to append with new indices."""
    while True:
        raw = input("Write mode [1=append new files / 2=overwrite from vessel_0] [1]: ").strip()
        if raw in ("", "1"):
            return False
        if raw == "2":
            return True
        print("  Enter 1 or 2.")


def _prompt_yes_no(label: str, default: bool = False) -> bool:
    """Read a yes/no answer; empty input returns ``default``."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{label} {suffix}: ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Enter y/yes or n/no.")


def _prompt_unit_choice(default: str = "m") -> str:
    """Read output unit from stdin; valid values are 'm' and 'cm'."""
    default = default.lower().strip()
    if default not in ("m", "cm"):
        default = "m"
    while True:
        raw = input(f"Mesh unit system [m/cm] [{default}]: ").strip().lower()
        if raw == "":
            return default
        if raw in ("m", "cm"):
            return raw
        print("  Enter m or cm.")


def _vessel_gen_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate 2D vessel meshes (Gmsh) for Kinematics phases.")
    p.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=None,
        help="Dataset (1=kinematics, 2=biochem; use with --level and -n)",
    )
    p.add_argument(
        "--level",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Geometry complexity (0=straight, 1=curved, 2=pro-clot)",
    )
    p.add_argument(
        "-n",
        "--num-vessels",
        type=int,
        default=None,
        metavar="N",
        help="How many vessels to generate (use with --phase and --level)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility. Omit for a fresh random draw each run (default).",
    )
    p.add_argument("--num-workers", type=int, default=None, help="Worker processes (default: auto)")
    p.add_argument("--chunk-size", type=int, default=None, help="Samples per worker chunk (default: auto)")
    p.add_argument(
        "--unit",
        type=str,
        choices=("m", "cm"),
        default="m",
        help="Mesh unit system (default: m).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Start vessel indices at 0. Default is to append after existing meshes.",
    )
    p.add_argument(
        "--show-vessel-plot",
        action="store_true",
        help="Show matplotlib preview of saved meshes (default: skip; avoids blocking on plot windows).",
    )
    p.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

    parser = _vessel_gen_arg_parser()
    args = parser.parse_args()

    trio = (args.phase is not None, args.level is not None, args.num_vessels is not None)
    if any(trio) and not all(trio):
        parser.error("Provide --phase, --level, and -n/--num-vessels together for non-interactive mode.")

    phase_map = {1: "kinematics", 2: "biochem"}

    if all(trio):
        phase = phase_map[int(args.phase)]
        level = args.level
        n_vessels = args.num_vessels
        start_idx = 0 if args.overwrite else None
        show_vessel_plot = bool(args.show_vessel_plot)
        unit_choice = str(args.unit).lower()
    else:
        phase_n = _prompt_int_choice("Dataset (1=kinematics, 2=biochem)", (1, 2))
        level = _prompt_int_choice("Level (0=straight, 1=curved, 2=pro-clot)", (0, 1, 2))
        phase = phase_map[phase_n]

        unit_choice = "m"
        if phase == "biochem" and level == 2:
            print("High-thrombus biochem generation detected.")
            print("Use 'cm' for thrombus CFD-compatible meshes, or keep 'm' for SI-scale meshes.")
            unit_choice = _prompt_unit_choice(default="cm")

        vg = VesselGenerator(phase=phase)
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        n_on_disk = int(inv["count"])
        max_idx = int(inv["max_idx"])
        index_span = max_idx + 1 if max_idx >= 0 else 0
        unused_slots = index_span - n_on_disk if max_idx >= 0 else 0
        print("\n--- Vessel mesh inventory ---")
        print(f"  Output: {vg.output_dir}")
        print(f"  Total number of phase vessels: {index_span}")
        print(f"  Number of vessel meshes already generated: {n_on_disk}")
        print(f"  Number of non-anchors remaining: {unused_slots}")
        print()
        if n_on_disk == 0:
            overwrite = True
            print("  No meshes on disk — starting indices at 0 (overwrite).\n")
        else:
            overwrite = True if args.overwrite else _prompt_write_mode_vessel()
        default_n = 50 if n_on_disk > 0 else 500
        n_vessels = _prompt_positive_int("How many vessels to generate", default_n)
        start_idx = 0 if overwrite else None
        show_vessel_plot = bool(args.show_vessel_plot) or _prompt_yes_no(
            "Show matplotlib preview of generated meshes after this run?",
            default=False,
        )

    if all(trio):
        vg = VesselGenerator(phase=phase)

    if args.seed is not None:
        logger.info("Using fixed RNG seed=%s", args.seed)
    else:
        logger.info("Using random RNG seed (each run draws a new cohort)")

    vg.run_pipeline(
        n=n_vessels,
        level=level,
        seed=args.seed,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        start_idx=start_idx,
        unit=unit_choice,
    )

    if show_vessel_plot:
        saved_indices = sorted(
            int(p.stem.split("_")[-1])
            for p in vg.output_dir.glob("vessel_*.msh")
        )[:9]
        if saved_indices:
            vg.visualize_saved(saved_indices)