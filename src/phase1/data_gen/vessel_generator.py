import json
import logging
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from src.utils.paths import get_project_root
from src.config import VesselConfig

# Configure logging for the data pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VesselGenerator:
    """Generates 2D vessel meshes with parametric pathologies using Gmsh."""

    def __init__(self, tier: str = "tier1", output_dir: Optional[str | Path] = None) -> None:
        self.cfg = VesselConfig(tier=tier)
        self.project_root = get_project_root()

        # Resolve output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.project_root / self.cfg.mesh_input_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._configure_gmsh_options()

    def _configure_gmsh_options(self) -> None:
        """Initializes and configures Gmsh global options."""
        if not gmsh.is_initialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", self.cfg.mesh_size_factor)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.cfg.mesh_lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.cfg.mesh_lc)

    def _get_mesh_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extracts nodes and triangles for matplotlib visualization."""
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            return None, None

        nodes = coords.reshape(-1, 3)[:, :2]
        node_dict = {tag: i for i, tag in enumerate(node_tags)}

        element_types, _, node_connectivity = gmsh.model.mesh.getElements(2)
        if not element_types or len(node_connectivity) == 0:
            return None, None

        tri_nodes = node_connectivity[0].reshape(-1, 3)
        triangles = np.array([[node_dict[tag] for tag in tri] for tri in tri_nodes])
        return nodes, triangles

    def visualize_sample(self, idx: int, ax: plt.Axes) -> None:
        """Visualizes the generated mesh on a Matplotlib axis."""
        nodes, triangles = self._get_mesh_data()
        if nodes is None or triangles is None:
            logger.warning(f"No mesh data found to visualize for sample {idx}.")
            return

        verts = nodes[triangles]
        poly = PolyCollection(verts, edgecolors='black', facecolors='lightblue', linewidths=0.1)
        ax.add_collection(poly)

        # Dynamic limit calculation
        max_expansion = self.cfg.width_max * self.cfg.aneurysm_factor_max
        max_y = self.cfg.curvature_amplitude + (self.cfg.width_max / 2) + max_expansion
        y_limit = max_y * 1.2
        x_pad = self.cfg.base_length * 0.1

        ax.set_xlim(-x_pad, self.cfg.base_length + x_pad)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_aspect('equal')
        ax.set_title(f"Sample {idx}")

    def _save_metadata(self, idx: int, d_bar: float, vessel_type: str, level: int, num_outlets: int) -> None:
        """Saves physical and categorical metadata for the dataset."""
        meta = {
            "id": idx,
            "type": vessel_type,
            "level": level,
            "d_bar": float(d_bar),
            "num_outlets": num_outlets,
        }
        meta_path = self.output_dir / f"vessel_{idx}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    def _calculate_pathology_offsets(self, v_type: str, width: float) -> np.ndarray:
        """Calculates localized width offsets for eccentric stenosis or aneurysm."""
        offsets = np.zeros(self.cfg.num_ctrl_pts)

        if v_type == 'straight':
            return offsets

        # Determine severity magnitude based on pathology type
        if v_type in ['stenosis', 'occlusion']:
            max_off = -random.uniform(self.cfg.stenosis_factor_min * width, self.cfg.stenosis_factor_max * width)
        elif v_type == 'aneurysm':
            max_off = random.uniform(self.cfg.aneurysm_factor_min * width, self.cfg.aneurysm_factor_max * width)
        else:
            max_off = 0.0

        # Randomize the peak location securely
        # Safeguard in case num_ctrl_pts is set too low in the future
        min_idx, max_idx = 3, max(4, self.cfg.num_ctrl_pts - 3)
        peak_idx = random.choice(range(min_idx, max_idx))

        # Apply asymmetric weights for an organic, eccentric shape
        left_weight = random.uniform(0.2, 0.8)
        right_weight = random.uniform(0.2, 0.8)

        offsets[peak_idx - 1] = max_off * left_weight
        offsets[peak_idx] = max_off
        offsets[peak_idx + 1] = max_off * right_weight

        return offsets

    def generate_spline_vessel(self, v_type: str, is_curved: bool) -> Tuple[List[int], Dict[str, List[int]], float]:
        """Generates the B-splines and boundaries for the vessel geometry."""
        width = random.uniform(self.cfg.width_min, self.cfg.width_max)
        length = self.cfg.base_length
        lc = self.cfg.mesh_lc

        # Vectorized x-coordinate generation
        x_vals = np.linspace(0, length, self.cfg.num_ctrl_pts)
        c_pts = []

        for i, x in enumerate(x_vals):
            if i < 3 or i > self.cfg.num_ctrl_pts - 4:
                y = 0.0
            else:
                jitter = random.uniform(-width * 0.03, width * 0.03)
                curvature = random.uniform(-self.cfg.curvature_amplitude,
                                           self.cfg.curvature_amplitude) if is_curved else 0.0
                y = curvature + jitter

            c_pts.append(np.array([x, y]))

        offsets = self._calculate_pathology_offsets(v_type, width)

        # Decide pathology location: 0 = Top, 1 = Bottom, 2 = Both
        path_loc = random.choice([0, 1, 2]) if v_type != 'straight' else 2

        # Calculate true effective diameter
        widths = [width + (o if path_loc in [0, 2] else 0) + (o if path_loc in [1, 2] else 0) for o in offsets]
        d_bar_true = float(np.mean(widths))

        top_tags, bot_tags = [], []
        for i in range(self.cfg.num_ctrl_pts):
            if i < self.cfg.num_ctrl_pts - 1:
                tangent = c_pts[i + 1] - c_pts[i]
            else:
                tangent = c_pts[i] - c_pts[i - 1]

            norm = np.array([-tangent[1], tangent[0]])
            norm = norm / (np.linalg.norm(norm) + 1e-9)

            # Apply offsets selectively to top, bottom, or both
            t_off = (width / 2) + (offsets[i] if path_loc in [0, 2] else 0)
            b_off = -(width / 2) - (offsets[i] if path_loc in [1, 2] else 0)

            p_top = c_pts[i] + norm * t_off
            p_bot = c_pts[i] + norm * b_off

            top_tags.append(gmsh.model.geo.addPoint(p_top[0], p_top[1], 0, lc))
            bot_tags.append(gmsh.model.geo.addPoint(p_bot[0], p_bot[1], 0, lc))

        # Build boundaries
        s_top = gmsh.model.geo.addBSpline(top_tags)
        l_out = gmsh.model.geo.addLine(top_tags[-1], bot_tags[-1])
        s_bot = gmsh.model.geo.addBSpline(bot_tags[::-1])
        l_in = gmsh.model.geo.addLine(bot_tags[0], top_tags[0])

        curves = [s_top, l_out, s_bot, l_in]
        groups = {
            "Inlet": [l_in],
            "Outlet_1": [l_out],
            "Outlet_2": [],
            "Walls": [s_top, s_bot]
        }

        return curves, groups, d_bar_true

    def generate(self, idx: int, level: int = 0, show_viz: bool = False, ax: Optional[plt.Axes] = None) -> None:
        """Executes the full generation, meshing, and saving pipeline for a single sample."""
        gmsh.model.add(f"vessel_{idx}")
        try:
            # Determine vessel characteristics based on dataset level
            v_type_choices = ['straight', 'stenosis', 'aneurysm']
            v_type = random.choice(v_type_choices)
            is_curved = random.choice([True, False]) if level > 0 else False

            curves, groups, d_bar = self.generate_spline_vessel(v_type, is_curved)
            vessel_label = f"{v_type}_{'curved' if is_curved else 'straight'}"
            num_outlets = 1

            # Define surface domain
            cl = gmsh.model.geo.addCurveLoop(curves)
            s = gmsh.model.geo.addPlaneSurface([cl])
            gmsh.model.geo.synchronize()

            # Assign Physical Groups for downstream CFD
            gmsh.model.addPhysicalGroup(1, groups["Inlet"], self.cfg.TAGS["Inlet"], name="Inlet")
            gmsh.model.addPhysicalGroup(1, groups["Outlet_1"], self.cfg.TAGS["Outlet_1"], name="Outlet_1")
            gmsh.model.addPhysicalGroup(1, groups["Walls"], self.cfg.TAGS["Walls"], name="Walls")
            gmsh.model.addPhysicalGroup(2, [s], self.cfg.TAGS["Fluid_Domain"], name="Fluid_Domain")

            # Generate uniform mesh
            gmsh.model.mesh.generate(2)

            if show_viz and ax is not None:
                self.visualize_sample(idx, ax)

            # Save outputs
            gmsh.write(str(self.output_dir / f"vessel_{idx}.msh"))
            gmsh.write(str(self.output_dir / f"vessel_{idx}.nas"))

            self._save_metadata(idx, d_bar, vessel_label, level, num_outlets)

        except Exception as e:
            logger.error(f"Error generating sample {idx}: {e}", exc_info=True)
        finally:
            gmsh.model.remove()

    def run_pipeline(self, n: int = 50, level: int = 0) -> None:
        """Runs batch generation of vessel meshes."""
        logger.info(f"Generating Level {level} Dataset (N={n}) in {self.output_dir}...")

        # Plot the first up to 9 samples
        viz_count = min(9, n)
        if viz_count > 0:
            fig, axes = plt.subplots(3, 3, figsize=(15, 8))
            axes = axes.flatten()
            for i in range(viz_count):
                self.generate(i, level=level, show_viz=True, ax=axes[i])
            plt.tight_layout()
            plt.show()

        # Generate the rest silently
        for i in range(viz_count, n):
            self.generate(i, level=level)
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{n} generated.")

        logger.info("Dataset generation complete.")


if __name__ == "__main__":
    active_tier = "tier1"

    vg = VesselGenerator(tier=active_tier)
    vg.run_pipeline(n=100, level=0)

    # Finalize Gmsh session safely
    if gmsh.is_initialized():
        gmsh.finalize()