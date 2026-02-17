import random
import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from src.utils.paths import get_project_root
from src.config import VesselConfig

# --- Context Manager for Safety ---
class GmshContext:
    """Ensures Gmsh is initialized and finalized properly."""

    def __enter__(self):
        if not gmsh.is_initialized():
            gmsh.initialize()
            # Suppress terminal output
            gmsh.option.setNumber("General.Terminal", 0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if gmsh.is_initialized():
            gmsh.finalize()


# --- Main Generator ---
class VesselGenerator:
    def __init__(self, config: VesselConfig):
        self.cfg = config
        self.output_dir = get_project_root() / self.cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._configure_gmsh_options()

    def _configure_gmsh_options(self):
        """Sets global GMSH meshing options."""
        if not gmsh.is_initialized():
            gmsh.initialize()

        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay 2D
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # Legacy for compatibility
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", self.cfg.mesh_size_factor)

    def _get_mesh_data(self):
        """Extracts nodes and triangles for visualization."""
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        if len(coords) == 0: return None, None

        nodes = coords.reshape(-1, 3)[:, :2]
        node_dict = {tag: i for i, tag in enumerate(node_tags)}

        element_types, _, node_connectivity = gmsh.model.mesh.getElements(2)
        if not element_types or len(node_connectivity) == 0:
            return None, None

        tri_nodes = node_connectivity[0].reshape(-1, 3)
        triangles = np.array([[node_dict[tag] for tag in tri] for tri in tri_nodes])
        return nodes, triangles

    def visualize_sample(self, idx, ax):
        """Visualizes the generated mesh."""
        nodes, triangles = self._get_mesh_data()
        if nodes is None: return

        verts = nodes[triangles]
        poly = PolyCollection(verts, edgecolors='black', facecolors='lightblue', linewidths=0.05)
        ax.add_collection(poly)

        # Dynamic limits based on config to avoid hardcoding
        lim = self.cfg.base_length * 1.2
        ax.set_xlim(-0.002, lim)
        ax.set_ylim(-lim / 2, lim / 2)
        ax.set_aspect('equal')
        ax.set_title(f"Sample {idx}")

    def _calculate_pathology_offsets(self, v_type, width):
        """Calculates offsets for Stenosis or Aneurysm."""
        offsets = np.zeros(self.cfg.num_ctrl_pts)
        peak_idx = self.cfg.num_ctrl_pts // 2

        if v_type in ['stenosis', 'occlusion']:
            max_allowed = self.cfg.stenosis_factor_max * width
            min_allowed = self.cfg.stenosis_factor_min * width
            max_off = -random.uniform(min_allowed, max_allowed)
            offsets[peak_idx - 1: peak_idx + 2] = [max_off * 0.6, max_off, max_off * 0.6]

        elif v_type == 'aneurysm':
            max_off = random.uniform(self.cfg.aneurysm_factor_min * width,
                                     self.cfg.aneurysm_factor_max * width)
            offsets[peak_idx - 1: peak_idx + 2] = [max_off * 0.6, max_off, max_off * 0.6]

        return offsets

    def generate_spline_vessel(self, v_type, is_curved):
        """Generates single-channel vessels (straight/curved/pathological)."""
        width = random.uniform(self.cfg.width_min, self.cfg.width_max)
        length = self.cfg.base_length
        lc = self.cfg.mesh_lc

        # 1. Define Centerline
        c_pts = []
        for i in range(self.cfg.num_ctrl_pts):
            x = (i / (self.cfg.num_ctrl_pts - 1)) * length
            y = random.uniform(-self.cfg.curvature_amplitude, self.cfg.curvature_amplitude) if is_curved else 0.0
            c_pts.append(np.array([x, y]))

        # 2. Calculate Offsets
        path_offsets = self._calculate_pathology_offsets(v_type, width)

        # 3. Build Walls
        top_tags, bot_tags = [], []
        for i in range(self.cfg.num_ctrl_pts):
            # Calculate Normal Vector
            if i < self.cfg.num_ctrl_pts - 1:
                tangent = c_pts[i + 1] - c_pts[i]
            else:
                tangent = c_pts[i] - c_pts[i - 1]

            norm = np.array([-tangent[1], tangent[0]])
            norm = norm / (np.linalg.norm(norm) + 1e-9)

            t_off = (width / 2) + path_offsets[i]
            b_off = -(width / 2)

            p_top = c_pts[i] + norm * t_off
            p_bot = c_pts[i] + norm * b_off

            top_tags.append(gmsh.model.geo.addPoint(p_top[0], p_top[1], 0, lc))
            bot_tags.append(gmsh.model.geo.addPoint(p_bot[0], p_bot[1], 0, lc))

        # 4. Connect
        s_top = gmsh.model.geo.addBSpline(top_tags)
        l_out = gmsh.model.geo.addLine(top_tags[-1], bot_tags[-1])
        s_bot = gmsh.model.geo.addBSpline(bot_tags[::-1])
        l_in = gmsh.model.geo.addLine(bot_tags[0], top_tags[0])

        curves = [s_top, l_out, s_bot, l_in]
        groups = {
            "Inlet": [l_in],
            "Outlet_1": [l_out],
            "Outlet_2": [],  # Empty for single vessel
            "Walls": [s_top, s_bot]
        }
        return curves, groups

    def generate_bifurcation(self):
        """Generates Y-shaped bifurcation."""
        width = random.uniform(self.cfg.width_min, self.cfg.width_max)
        angle = np.deg2rad(random.uniform(self.cfg.bifurcation_angle_min, self.cfg.bifurcation_angle_max))
        lc = self.cfg.mesh_lc

        L1, L2 = self.cfg.bifurcation_l1, self.cfg.bifurcation_l2
        f_rad = 0.0004  # Fillet radius estimate

        # Inlet Points
        p_in_t = gmsh.model.geo.addPoint(0, width / 2, 0, lc)
        p_in_b = gmsh.model.geo.addPoint(0, -width / 2, 0, lc)

        # Outlet 1 (Upper)
        o1_t_x, o1_t_y = L1 + L2 * np.cos(angle), width / 2 + L2 * np.sin(angle)
        p_o1_t = gmsh.model.geo.addPoint(o1_t_x, o1_t_y, 0, lc)
        p_o1_b = gmsh.model.geo.addPoint(o1_t_x + width * np.sin(angle), o1_t_y - width * np.cos(angle), 0, lc)

        # Outlet 2 (Lower)
        o2_b_x, o2_b_y = L1 + L2 * np.cos(angle), -width / 2 - L2 * np.sin(angle)
        p_o2_b = gmsh.model.geo.addPoint(o2_b_x, o2_b_y, 0, lc)
        p_o2_t = gmsh.model.geo.addPoint(o2_b_x + width * np.sin(angle), o2_b_y + width * np.cos(angle), 0, lc)

        # Geometry Construction
        l_in = gmsh.model.geo.addLine(p_in_b, p_in_t)
        mid_top = gmsh.model.geo.addPoint(L1 * 0.6, width / 2, 0, lc)
        l_wt = gmsh.model.geo.addBSpline([p_in_t, mid_top, p_o1_t])
        l_o1 = gmsh.model.geo.addLine(p_o1_t, p_o1_b)

        # Fork
        p_f_upper = gmsh.model.geo.addPoint(L1 + f_rad * 2, width * 0.1, 0, lc)
        p_f_center = gmsh.model.geo.addPoint(L1 + f_rad, 0, 0, lc)
        p_f_lower = gmsh.model.geo.addPoint(L1 + f_rad * 2, -width * 0.1, 0, lc)
        l_fork = gmsh.model.geo.addBSpline([p_o1_b, p_f_upper, p_f_center, p_f_lower, p_o2_t])

        l_o2 = gmsh.model.geo.addLine(p_o2_t, p_o2_b)
        mid_bot = gmsh.model.geo.addPoint(L1 * 0.6, -width / 2, 0, lc)
        l_wb = gmsh.model.geo.addBSpline([p_o2_b, mid_bot, p_in_b])

        curves = [l_in, l_wt, l_o1, l_fork, l_o2, l_wb]
        groups = {
            "Inlet": [l_in],
            "Outlet_1": [l_o1],
            "Outlet_2": [l_o2],
            "Walls": [l_wt, l_fork, l_wb]
        }
        return curves, groups

    def _determine_vessel_type(self, level):
        """Logic to decide which vessel type to generate based on complexity level."""
        is_bifurcation = False
        is_curved = False
        v_type = 'straight'

        if level == 1:
            v_type = random.choice(['straight', 'stenosis', 'aneurysm'])
        elif level == 2:
            v_type = random.choice(['straight', 'stenosis', 'aneurysm', 'curved'])
            is_curved = (v_type == 'curved') or (random.random() > 0.5)
        else:
            if random.random() > 0.7:
                is_bifurcation = True
            else:
                v_type = random.choice(['straight', 'stenosis', 'aneurysm', 'curved'])
                is_curved = (v_type == 'curved') or (random.random() > 0.5)

        return v_type, is_curved, is_bifurcation

    def generate(self, idx, level=1, show_viz=False, ax=None):
        """Orchestrates the generation of a single sample."""
        gmsh.model.add(f"vessel_{idx}")

        try:
            v_type, is_curved, is_bifurcation = self._determine_vessel_type(level)

            if is_bifurcation:
                curves, groups = self.generate_bifurcation()
            else:
                curves, groups = self.generate_spline_vessel(v_type, is_curved)

            # Surface
            cl = gmsh.model.geo.addCurveLoop(curves)
            s = gmsh.model.geo.addPlaneSurface([cl])
            gmsh.model.geo.synchronize()

            # Physical Groups
            gmsh.model.addPhysicalGroup(1, groups["Inlet"], self.cfg.TAGS["Inlet"], name="Inlet")
            gmsh.model.addPhysicalGroup(1, groups["Outlet_1"], self.cfg.TAGS["Outlet_1"], name="Outlet_1")

            # Logic for second outlet vs walls
            if groups["Outlet_2"]:
                gmsh.model.addPhysicalGroup(1, groups["Outlet_2"], self.cfg.TAGS["Outlet_2"], name="Outlet_2")
                gmsh.model.addPhysicalGroup(1, groups["Walls"], self.cfg.TAGS["Walls"], name="Walls")
            else:
                # If no second outlet, walls take the Walls tag
                gmsh.model.addPhysicalGroup(1, groups["Walls"], self.cfg.TAGS["Walls"], name="Walls")

            gmsh.model.addPhysicalGroup(2, [s], self.cfg.TAGS["Fluid_Domain"], name="Fluid_Domain")

            # Mesh
            gmsh.model.mesh.generate(2)

            if show_viz and ax:
                self.visualize_sample(idx, ax)

            # Write files
            gmsh.write(str(self.output_dir / f"vessel_{idx}.msh"))
            gmsh.write(str(self.output_dir / f"vessel_{idx}.nas"))

        except Exception as e:
            print(f"Error generating sample {idx}: {e}")
        finally:
            gmsh.model.remove()

    def run_pipeline(self, n=500, level=1):
        print(f"Generating Level {level} Dataset (N={n}) in {self.output_dir}...")

        # Batch 1: Visualization
        num_viz = 9
        if n > 0:
            fig, axes = plt.subplots(3, 3, figsize=(15, 8))
            axes = axes.flatten()
            for i in range(min(num_viz, n)):
                self.generate(i, level=level, show_viz=True, ax=axes[i])
            plt.tight_layout()
            plt.show()

        # Batch 2: Remainder
        for i in range(num_viz, n):
            self.generate(i, level=level)
            if i % 50 == 0: print(f"Progress: {i}/{n}")


if __name__ == "__main__":
    # Setup configuration
    config = VesselConfig()

    # Use Context Manager to safely handle Gmsh session
    with GmshContext():
        gen = VesselGenerator(config)
        gen.run_pipeline()