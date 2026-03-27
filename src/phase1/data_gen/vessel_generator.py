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
    n: int, length: float, angle_span: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Circular arc: starts at (0,0) pointing +X, sweeps clockwise by angle_span.
    radius = length / angle_span  so arc length == length.
    """
    radius = length / max(angle_span, 1e-3)
    theta = np.linspace(0.0, angle_span, n)
    pts = np.column_stack([
        radius * np.sin(theta),
        radius * (np.cos(theta) - 1.0),
    ])
    tangents = np.column_stack([np.cos(theta), -np.sin(theta)])
    return pts, tangents


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
        if np.any(cross_widths < (width * 0.1)):
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

            # Inside the Safety Check for Self-Intersection loop:
            cross_distances = np.linalg.norm(top_coords - bot_coords, axis=1)
            if np.any(cross_distances < (width * 0.20)):  # Increased margin to 20%
                raise ValueError("Self-intersection detected: Walls pinched together.")

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
    ) -> None:
        """
        Parallel batch vessel generation.

        Parameters
        ----------
        n           : total number of vessels to produce
        level       : geometry complexity (0 = mostly straight, 1 = curved)
        max_retries : retry attempts for failed samples
        num_workers : worker processes (default: cpu_count - 1, min 1)
        chunk_size  : samples per worker chunk (default: auto-balanced)
        seed        : integer seed for reproducibility (None = random)
        """
        phys_cores  = os.cpu_count() or 1
        num_workers = max(1, phys_cores - 1) if num_workers is None else num_workers
        num_workers = min(num_workers, n)

        logger.info(
            f"Generating {n} Level-{level} vessels → {self.output_dir} "
            f"[{num_workers} workers / {phys_cores} logical cores]"
        )

        cfg_d   = self._cfg_dict()
        out_str = str(self.output_dir)
        rng     = np.random.default_rng(seed)

        # Pre-sample everything in the main process
        all_params = [_sample_params(i, level, self.cfg, rng) for i in range(n)]

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
                        failed_params.append(all_params[idx])

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
                                    failed_params.append(all_params[idx])

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
        for retry_round in range(1, max_retries + 1):
            if not failed_params:
                break
            logger.info(f"Retry {retry_round}/{max_retries}: {len(failed_params)} samples")

            # Give retried samples fresh indices AND roll BRAND NEW parameters
            base = n + (retry_round - 1) * len(failed_params)
            retry_batch = [_sample_params(base + i, level, self.cfg, rng) for i in
                           range(len(failed_params))]

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


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    # Run the generation
    vg = VesselGenerator(tier="tier1")
    vg.run_pipeline(n=100, level=1, seed=25, num_workers=8, chunk_size=1)

    # Call visualization here
    saved_indices = sorted(
        int(p.stem.split("_")[-1])
        for p in vg.output_dir.glob("vessel_*.msh")
    )[:9]
    if saved_indices:
        vg.visualize_saved(saved_indices)