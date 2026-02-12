import os
import random
import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path


class VesselGenerator:
    def __init__(self, output_dir="data/raw/synthetic_v1"):
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent.parent.parent
        self.output_dir = project_root / output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # Legacy format for compatibility
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)  # Only save physical groups
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)

    def _get_mesh_data(self):
        """Extracts nodes and triangles for visualization."""
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)[:, :2]
        node_dict = {tag: i for i, tag in enumerate(node_tags)}
        element_types, _, node_connectivity = gmsh.model.mesh.getElements(2)
        if not element_types or len(node_connectivity) == 0: return None, None
        tri_nodes = node_connectivity[0].reshape(-1, 3)
        triangles = np.array([[node_dict[tag] for tag in tri] for tri in tri_nodes])
        return nodes, triangles

    def visualize_sample(self, idx, ax):
        """Visualizes the generated mesh on a Matplotlib axis."""
        nodes, triangles = self._get_mesh_data()
        if nodes is None: return
        verts = nodes[triangles]
        poly = PolyCollection(verts, edgecolors='black', facecolors='lightblue', linewidths=0.05)
        ax.add_collection(poly)
        # Fixed limits to show standard scale (Meters)
        ax.set_xlim(-0.002, 0.020)
        ax.set_ylim(-0.008, 0.008)
        ax.set_aspect('equal')
        ax.set_title(f"Sample {idx}")

    def generate_pathological_spline(self, v_type, is_curved, lc):
        """
        Generates Level 1 (Straight) and Level 2 (Curved) vessels with pathologies.
        Uses centerline splines + normal offsets to ensure consistent width.
        """
        L, W = 0.015, random.uniform(0.0012, 0.0018)
        num_ctrl_pts = 7

        # 1. Define Centerline
        # Level 1: Y=0. Level 2: Y varies slightly to create curves.
        c_pts = []
        for i in range(num_ctrl_pts):
            x = (i / (num_ctrl_pts - 1)) * L
            y = random.uniform(-0.0035, 0.0035) if is_curved else 0.0
            c_pts.append(np.array([x, y]))

        # 2. Define Local Pathology (Stenosis/Aneurysm)
        # Applied as an offset to the top wall thickness
        pathology_offsets = np.zeros(num_ctrl_pts)
        peak_idx = num_ctrl_pts // 2

        if v_type in ['stenosis', 'occlusion']:
            # SAFETY CONSTRAINT: Max constriction = 66% of Width (leaving 33% open)
            max_allowed_constriction = 0.66 * W
            # Random constriction between 30% and 66%
            max_off = -random.uniform(0.30 * W, max_allowed_constriction)
            pathology_offsets[peak_idx - 1: peak_idx + 2] = [max_off * 0.6, max_off, max_off * 0.6]

        elif v_type == 'aneurysm':
            # Aneurysms expand outward (safer for physics)
            max_off = random.uniform(0.4 * W, 0.8 * W)
            pathology_offsets[peak_idx - 1: peak_idx + 2] = [max_off * 0.6, max_off, max_off * 0.6]

        # 3. Calculate Wall Points via Normal Vectors
        top_tags, bot_tags = [], []
        for i in range(num_ctrl_pts):
            # Calculate Tangent & Normal
            if i < num_ctrl_pts - 1:
                tangent = c_pts[i + 1] - c_pts[i]
            else:
                tangent = c_pts[i] - c_pts[i - 1]

            # Rotate tangent 90 degrees to get normal
            norm = np.array([-tangent[1], tangent[0]])
            norm = norm / (np.linalg.norm(norm) + 1e-9)  # Epsilon for safety

            # Apply offsets
            t_off = (W / 2) + pathology_offsets[i]
            b_off = -(W / 2)

            p_top = c_pts[i] + norm * t_off
            p_bot = c_pts[i] + norm * b_off

            top_tags.append(gmsh.model.geo.addPoint(p_top[0], p_top[1], 0, lc))
            bot_tags.append(gmsh.model.geo.addPoint(p_bot[0], p_bot[1], 0, lc))

        # 4. Form Closed Loop (Head-to-Tail Connectivity)
        # Forward along Top
        s_top = gmsh.model.geo.addBSpline(top_tags)
        # Down at Outlet
        l_out = gmsh.model.geo.addLine(top_tags[-1], bot_tags[-1])
        # Backward along Bottom (Points reversed)
        s_bot = gmsh.model.geo.addBSpline(bot_tags[::-1])
        # Up at Inlet
        l_in = gmsh.model.geo.addLine(bot_tags[0], top_tags[0])

        curves = [s_top, l_out, s_bot, l_in]
        groups = {
            "Inlet": [l_in], "Outlet_1": [l_out], "Outlet_2": [], "Walls": [s_top, s_bot]
        }
        return curves, groups

    def generate_bifurcation(self, lc):
        """
        Generates Level 3 (Bifurcation) vessels.
        Fixed topology to ensure correct curve loop orientation.
        """
        W, angle = random.uniform(0.0014, 0.0018), np.deg2rad(random.uniform(20, 45))
        L1, L2 = 0.006, 0.009
        f_rad = 0.0004

        # Control Points
        p_in_t = gmsh.model.geo.addPoint(0, W / 2, 0, lc)
        p_in_b = gmsh.model.geo.addPoint(0, -W / 2, 0, lc)

        # Outlet 1 (Upper)
        o1_t_x, o1_t_y = L1 + L2 * np.cos(angle), W / 2 + L2 * np.sin(angle)
        p_o1_t = gmsh.model.geo.addPoint(o1_t_x, o1_t_y, 0, lc)
        p_o1_b = gmsh.model.geo.addPoint(o1_t_x + W * np.sin(angle), o1_t_y - W * np.cos(angle), 0, lc)

        # Outlet 2 (Lower)
        o2_b_x, o2_b_y = L1 + L2 * np.cos(angle), -W / 2 - L2 * np.sin(angle)
        p_o2_b = gmsh.model.geo.addPoint(o2_b_x, o2_b_y, 0, lc)
        p_o2_t = gmsh.model.geo.addPoint(o2_b_x + W * np.sin(angle), o2_b_y + W * np.cos(angle), 0, lc)

        # 5. Build Loop (Counter-Clockwise traversal)
        # Inlet (Bot->Top) -> Top Wall -> Out1 -> Fork -> Out2 -> Bot Wall
        l_in = gmsh.model.geo.addLine(p_in_b, p_in_t)

        # Top Wall (Spline for smooth transition)
        mid_top = gmsh.model.geo.addPoint(L1 * 0.6, W / 2, 0, lc)
        l_wt = gmsh.model.geo.addBSpline([p_in_t, mid_top, p_o1_t])

        l_o1 = gmsh.model.geo.addLine(p_o1_t, p_o1_b)

        # Fork (Crotch of bifurcation)
        p_f_upper = gmsh.model.geo.addPoint(L1 + f_rad * 2, W * 0.1, 0, lc)
        p_f_center = gmsh.model.geo.addPoint(L1 + f_rad, 0, 0, lc)
        p_f_lower = gmsh.model.geo.addPoint(L1 + f_rad * 2, -W * 0.1, 0, lc)
        l_fork = gmsh.model.geo.addBSpline([p_o1_b, p_f_upper, p_f_center, p_f_lower, p_o2_t])

        l_o2 = gmsh.model.geo.addLine(p_o2_t, p_o2_b)

        # Bottom Wall
        mid_bot = gmsh.model.geo.addPoint(L1 * 0.6, -W / 2, 0, lc)
        l_wb = gmsh.model.geo.addBSpline([p_o2_b, mid_bot, p_in_b])

        curves = [l_in, l_wt, l_o1, l_fork, l_o2, l_wb]
        groups = {
            "Inlet": [l_in],
            "Outlet_1": [l_o1],
            "Outlet_2": [l_o2],
            "Walls": [l_wt, l_fork, l_wb]
        }
        return curves, groups

    def generate(self, idx, level=1, show_viz=False, ax=None):
        """
        Main generation entry point.
        Level 1: Straight + Pathology
        Level 2: Curved + Pathology
        Level 3: Bifurcations + Curved + Pathology
        """
        gmsh.model.add(f"vessel_{idx}")
        lc = 0.0002

        # --- SELECTION LOGIC ---
        is_bifurcation = False
        is_curved = False
        v_type = 'straight'

        if level == 1:
            # Straight vessels with potential pathology
            v_type = random.choice(['straight', 'stenosis', 'aneurysm'])
            is_curved = False
        elif level == 2:
            # Curved or Straight vessels with pathology
            v_type = random.choice(['straight', 'stenosis', 'aneurysm', 'curved'])
            is_curved = (v_type == 'curved') or (random.random() > 0.5)
        else:
            # Level 3: All of the above + Bifurcations
            r = random.random()
            if r > 0.7:
                is_bifurcation = True
            else:
                v_type = random.choice(['straight', 'stenosis', 'aneurysm', 'curved'])
                is_curved = (v_type == 'curved') or (random.random() > 0.5)

        # --- GENERATION ---
        if is_bifurcation:
            curves, groups = self.generate_bifurcation(lc)
        else:
            curves, groups = self.generate_pathological_spline(v_type, is_curved, lc)

        # Create Surface
        try:
            cl = gmsh.model.geo.addCurveLoop(curves)
            s = gmsh.model.geo.addPlaneSurface([cl])
            gmsh.model.geo.synchronize()
        except Exception as e:
            print(f"Skipping bad geometry {idx}: {e}")
            gmsh.model.remove()
            return

        # Physical Groups (Standard IDs for mesh_to_graph.py)
        gmsh.model.addPhysicalGroup(1, groups["Inlet"], 101, name="Inlet")
        gmsh.model.addPhysicalGroup(1, groups["Outlet_1"], 102, name="Outlet_1")
        if groups["Outlet_2"]:
            gmsh.model.addPhysicalGroup(1, groups["Outlet_2"], 103, name="Outlet_2")
            gmsh.model.addPhysicalGroup(1, groups["Walls"], 104, name="Walls")
        else:
            gmsh.model.addPhysicalGroup(1, groups["Walls"], 103, name="Walls")  # 103 = Walls if no 2nd outlet

        gmsh.model.addPhysicalGroup(2, [s], 201, name="Fluid_Domain")

        # Mesh & Visualize
        gmsh.model.mesh.generate(2)
        if show_viz and ax:
            self.visualize_sample(idx, ax)

        # Export
        gmsh.write(str(self.output_dir / f"vessel_{idx}.msh"))
        gmsh.write(str(self.output_dir / f"vessel_{idx}.nas"))  # For COMSOL

        gmsh.model.remove()

    def run_pipeline(self, n=500, level=1):
        print(f"🚀 Generating Level {level} Dataset (N={n})...")

        # Visualize first 9 samples
        fig, axes = plt.subplots(3, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i in range(min(9, n)):
            self.generate(i, level=level, show_viz=True, ax=axes[i])
        plt.tight_layout()
        plt.show()

        # Run remainder
        for i in range(9, n):
            self.generate(i, level=level)
            if i % 50 == 0: print(f"Progress: {i}/{n}")


if __name__ == "__main__":
    gen = VesselGenerator()
    # Change level=1, 2, or 3 here to control complexity
    gen.run_pipeline(n=500, level=1)
    gmsh.finalize()