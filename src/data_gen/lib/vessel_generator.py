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
from src.utils.paths import get_project_root

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


def _sample_params(
        idx: int,
        level: int,
        cfg: VesselConfig,
        rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Draw ALL random numbers for one vessel and return a plain picklable dict.
    Adds high-frequency biological roughness and global tortuosity.
    """
    weights_map = _CURVE_WEIGHTS.get(min(level, 1), _CURVE_WEIGHTS)
    active = {k: v for k, v in weights_map.items() if v > 0}
    keys = list(active.keys())
    probs = np.array(list(active.values()), dtype=float)
    probs /= probs.sum()
    curve_type = str(rng.choice(keys, p=probs))

    v_type = str(rng.choice(["straight", "stenosis", "aneurysm"]))
    width = float(rng.uniform(cfg.width_min, cfg.width_max))
    n = cfg.num_ctrl_pts
    L = cfg.base_length
    t = np.linspace(0, 1, n)

    # 1. Main Clinical Pathology (Gaussian Aneurysm/Stenosis)
    offsets = np.zeros(n)
    if v_type != "straight":
        if v_type in ("stenosis", "occlusion"):
            mag = -float(rng.uniform(cfg.stenosis_factor_min * width, cfg.stenosis_factor_max * width))
        else:
            mag = float(rng.uniform(cfg.aneurysm_factor_min * width, cfg.aneurysm_factor_max * width))

        min_idx, max_idx = max(3, int(n * 0.2)), min(n - 4, int(n * 0.8))
        peak = int(rng.integers(min_idx, max_idx))
        std_dev = float(rng.uniform(0.04 * n, 0.10 * n))
        x_idx = np.arange(n)

        gauss = np.exp(-0.5 * ((x_idx - peak) / std_dev) ** 2)
        skew_factor = float(rng.uniform(-0.3, 0.3))
        skew = 1.0 + skew_factor * ((x_idx - peak) / n)
        offsets = mag * gauss * skew

    path_loc = int(rng.choice([0, 1, 2])) if v_type != "straight" else 2

    # 2. Universal Centerline Tortuosity (Organic Meander)
    # Lowered frequencies: Prevents B-spline loops between control points
    f1, f2 = rng.uniform(0.5, 1.5), rng.uniform(1.5, 2.5)
    meander = np.sin(2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi)) + \
              0.5 * np.sin(2 * np.pi * f2 * t + rng.uniform(0, 2 * np.pi))
    max_meander = 0.10 * width  # Reduced to 10% of width to prevent acute angles
    meander = (meander / max(1e-9, float(np.max(np.abs(meander))))) * max_meander
    meander *= np.sin(np.pi * t)  # Taper exactly to 0 at inlet/outlet
    tortuosity = meander[2:n - 2].tolist()

    # 3. Independent Wall Roughness (Plaque/Irregularity)
    def get_wall_noise():
        # Lowered frequencies so the 50-point spline can smoothly resolve the waves
        f_h1, f_h2 = rng.uniform(1.0, 2.5), rng.uniform(2.5, 4.0)
        noise = np.sin(2 * np.pi * f_h1 * t + rng.uniform(0, 2 * np.pi)) + \
                0.5 * np.sin(2 * np.pi * f_h2 * t + rng.uniform(0, 2 * np.pi))
        max_noise = 0.05 * width  # Reduced to 5% of width to prevent pinching
        noise = (noise / max(1e-9, float(np.max(np.abs(noise))))) * max_noise
        noise *= np.sin(np.pi * t) ** 0.5  # Taper smoothly to ends
        return noise.tolist()

    noise_top = get_wall_noise()
    noise_bot = get_wall_noise()

    # 4. Safely Bound Parameters to prevent self-intersection
    max_half_width = (width / 2.0) + max(0, float(np.max(offsets))) + (0.08 * width)
    min_safe_radius = 1.6 * max_half_width  # Increased margin for high-frequency noise
    max_safe_angle_span = L / min_safe_radius

    if curve_type == "straight":
        angle_span, amplitude = 0.0, 0.0
    elif curve_type in ("arc", "hook"):
        if curve_type == "arc":
            target_angle = float(rng.uniform(np.deg2rad(45), np.deg2rad(100)))
        else:
            # Cap hook angle at 125 degrees to prevent the outlet from curling back past L/3
            target_angle = float(rng.uniform(np.deg2rad(100), np.deg2rad(125)))

        angle_span = min(target_angle, max_safe_angle_span)
        amplitude = 0.0
    else:  # s_curve
        angle_span = 0.0
        amplitude = min(float(rng.uniform(0.003, 0.007)), L * 0.15)

    return {
        "idx": idx,
        "level": level,
        "v_type": v_type,
        "curve_type": curve_type,
        "width": width,
        "angle_span": angle_span,
        "amplitude": amplitude,
        "jitter": [],  # Deprecated, handled by tortuosity now
        "tortuosity": tortuosity,
        "noise_top": noise_top,
        "noise_bot": noise_bot,
        "offsets": offsets.tolist(),
        "path_loc": path_loc,
    }


def _build_and_mesh(
        params: Dict[str, Any],
        cfg_dict: Dict[str, Any],
        output_dir: str,
) -> Tuple[int, bool, str]:
    """Build, mesh, and save one vessel. Returns (idx, success, error_msg)."""
    idx = params["idx"]
    out = Path(output_dir)

    try:
        n = cfg_dict["num_ctrl_pts"]
        L = cfg_dict["base_length"]
        lc = cfg_dict["mesh_lc"]
        min_lumen_frac = float(cfg_dict["min_lumen_width_fraction"])
        curve_type = params["curve_type"]
        v_type = params["v_type"]
        width = params["width"]
        path_loc = params["path_loc"]
        offsets = np.array(params["offsets"])

        # 1. Base Centerline Geometry (Perfect shapes)
        if curve_type == "straight":
            pts, tangents = _centerline_straight(n, L, np.zeros(n - 4))
        elif curve_type in ("arc", "hook"):
            pts, tangents = _centerline_arc(n, L, params["angle_span"])
        else:
            pts, tangents = _centerline_s_curve(n, L, params["amplitude"])

        # 2. Apply Global Tortuosity (Organic Centerline Meander)
        tortuosity = np.array(params.get("tortuosity", np.zeros(n - 4)))
        if np.any(tortuosity):
            normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
            pts[2:n - 2] += normals[2:n - 2] * tortuosity[:, np.newaxis]

            # Recompute tangents and normals after adding meander
            tangents = np.gradient(pts, axis=0)
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            tangents = tangents / np.maximum(norms, 1e-9)

        # Final normals for wall generation
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        # 3. Compile Wall Offsets (Main pathology + High-frequency biological noise)
        top_offsets = offsets if path_loc in (0, 2) else np.zeros(n)
        bot_offsets = offsets if path_loc in (1, 2) else np.zeros(n)

        top_offsets += np.array(params.get("noise_top", np.zeros(n)))
        bot_offsets += np.array(params.get("noise_bot", np.zeros(n)))

        top_dist = (width / 2.0) + top_offsets
        bot_dist = (width / 2.0) + bot_offsets
        cross_widths = top_dist + bot_dist
        d_bar = float(np.mean(cross_widths))

        if d_bar < 1e-5 or np.any(cross_widths < 0):
            raise ValueError(f"Degenerate geometry: d_bar={d_bar:.2e}")
        if np.any(cross_widths < (width * min_lumen_frac)):
            raise ValueError(f"Geometry too narrow at a control point.")

        # 4. Generate Final Coordinates
        top_coords = pts + normals * top_dist[:, np.newaxis]
        bot_coords = pts - normals * bot_dist[:, np.newaxis]

        # Safety Check for Self-Intersection
        for coords in (top_coords, bot_coords):
            step_vectors = np.diff(coords, axis=0)
            step_lengths = np.linalg.norm(step_vectors, axis=1)
            if np.any(step_lengths < (L / n) * 0.1):
                raise ValueError("Self-intersection detected: Boundary spline collapsed.")

            # --- NEW COMSOL OUTLET CHECK ---
            # Ensure the physical X-coordinates of the outlet nodes are strictly > L/3
            if top_coords[-1, 0] < (L / 3.0) or bot_coords[-1, 0] < (L / 3.0):
                raise ValueError("Geometry rejected: Outlet curled back past L/3.")

            # Gmsh geometry
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

        meta = {
            "id": idx,
            "type": f"{v_type}_{curve_type}",
            "curve": curve_type,
            "level": params["level"],
            "d_bar": d_bar,
            "num_outlets": 1,
            # Nondimensional centerline (same scaling as mesh graphs: nodes / d_bar) + unit tangents
            "centerline_pts": (pts / d_bar).tolist(),
            "centerline_tangents": tangents.tolist(),
        }
        with open(out / f"vessel_{idx}.json", "w") as f:
            json.dump(meta, f, indent=4)

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
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",          0)
    gmsh.option.setNumber("Mesh.Algorithm",            6)   # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Smoothing",            5)
    gmsh.option.setNumber("Mesh.MshFileVersion",       2.2)
    gmsh.option.setNumber("Mesh.Binary",               0)
    gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",    1)
    gmsh.option.setNumber("Mesh.SaveAll",              0)
    gmsh.option.setNumber("Mesh.MeshSizeFactor",       cfg_dict["mesh_size_factor"])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cfg_dict["mesh_lc"])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cfg_dict["mesh_lc"])

    results = [_build_and_mesh(p, cfg_dict, output_dir) for p in chunk]

    gmsh.finalize()
    return results


class VesselGenerator:
    """Generates 2D vessel meshes with parametric pathologies using Gmsh."""

    def __init__(self, tier: str = "tier1", output_dir: Optional[str | Path] = None) -> None:
        self.cfg          = VesselConfig(tier=tier)
        self.project_root = get_project_root()
        self.output_dir   = Path(output_dir) if output_dir else self.project_root / self.cfg.mesh_input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
                ax.set_title(f"vessel_{idx}", fontsize=8)
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
        max_retries: int = 3,
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        start_idx: Optional[int] = None,
    ) -> None:
        """
        Parallel batch vessel generation.

        Parameters
        ----------
        n           : number of vessels to produce in this run (file indices ``start_idx`` …)
        level       : geometry complexity (0 = mostly straight, 1 = curved)
        max_retries : retry attempts for failed samples (same ``idx`` as the failure; new parameters)
        num_workers : worker processes (default: cpu_count - 1, min 1)
        chunk_size  : samples per worker chunk (default: auto-balanced)
        seed        : integer seed for reproducibility (None = random)
        start_idx   : first vessel index ``vessel_{idx}.*``. If ``None``, appends after the
                      highest existing index in ``output_dir`` (no overwrite). Pass ``0`` to
                      fill from the beginning (may overwrite existing files).
        """
        phys_cores  = os.cpu_count() or 1
        num_workers = max(1, phys_cores - 1) if num_workers is None else num_workers
        num_workers = min(num_workers, n)

        if start_idx is None:
            start_idx = _next_vessel_index(self.output_dir)

        logger.info(
            f"Generating {n} Level-{level} vessels → {self.output_dir} "
            f"[indices {start_idx}..{start_idx + n - 1}] "
            f"[{num_workers} workers / {phys_cores} logical cores]"
        )

        cfg_d   = self._cfg_dict()
        out_str = str(self.output_dir)
        rng     = np.random.default_rng(seed)

        # Pre-sample everything in the main process
        all_params = [_sample_params(start_idx + i, level, self.cfg, rng) for i in range(n)]
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
                _sample_params(int(failed_p["idx"]), level, self.cfg, rng)
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


class VesselGeneratorTier3(VesselGenerator):
    """Synthetic Tier-3 vessel cohort (same geometry pipeline as ``VesselGenerator(tier='tier3')``)."""

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        super().__init__(tier="tier3", output_dir=output_dir)

    def run_pipeline(
        self,
        n: int = 50,
        level: int = 0,
        max_retries: int = 3,
        num_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        seed: Optional[int] = None,
        start_idx: Optional[int] = None,
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


def _vessel_gen_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate 2D vessel meshes (Gmsh) for Phase 1 tiers.")
    p.add_argument("--tier", type=int, choices=(1, 2), default=None, help="Tier (use with --level and -n)")
    p.add_argument("--level", type=int, choices=(0, 1), default=None, help="Geometry complexity (use with --tier and -n)")
    p.add_argument(
        "-n",
        "--num-vessels",
        type=int,
        default=None,
        metavar="N",
        help="How many vessels to generate (use with --tier and --level)",
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

    trio = (args.tier is not None, args.level is not None, args.num_vessels is not None)
    if any(trio) and not all(trio):
        parser.error("Provide --tier, --level, and -n/--num-vessels together for non-interactive mode.")

    if all(trio):
        tier = f"tier{args.tier}"
        level = args.level
        n_vessels = args.num_vessels
        start_idx = 0 if args.overwrite else None
        show_vessel_plot = bool(args.show_vessel_plot)
    else:
        tier_n = _prompt_int_choice("Tier", (1, 2))
        level = _prompt_int_choice("Level", (0, 1))
        tier = f"tier{tier_n}"

        vg = VesselGenerator(tier=tier)
        inv = summarize_vessel_mesh_inventory(vg.output_dir)
        n_on_disk = int(inv["count"])
        max_idx = int(inv["max_idx"])
        index_span = max_idx + 1 if max_idx >= 0 else 0
        unused_slots = index_span - n_on_disk if max_idx >= 0 else 0
        print("\n--- Vessel mesh inventory ---")
        print(f"  Output: {vg.output_dir}")
        print(f"  Total number of tier vessels: {index_span}")
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
        vg = VesselGenerator(tier=tier)

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
    )

    if show_vessel_plot:
        saved_indices = sorted(
            int(p.stem.split("_")[-1])
            for p in vg.output_dir.glob("vessel_*.msh")
        )[:9]
        if saved_indices:
            vg.visualize_saved(saved_indices)