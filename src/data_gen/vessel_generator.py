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
        project_root = current_script_path.parent.parent.parent
        self.output_dir = project_root / output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)  # Only save physical groups
        gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1)

    def _get_mesh_data(self):
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)[:, :2]
        node_dict = {tag: i for i, tag in enumerate(node_tags)}
        element_types, _, node_connectivity = gmsh.model.mesh.getElements(2)
        if not element_types or len(node_connectivity) == 0: return None, None
        tri_nodes = node_connectivity[0].reshape(-1, 3)
        triangles = np.array([[node_dict[tag] for tag in tri] for tri in tri_nodes])
        return nodes, triangles

    def visualize_sample(self, idx, ax):
        nodes, triangles = self._get_mesh_data()
        if nodes is None: return
        verts = nodes[triangles]
        poly = PolyCollection(verts, edgecolors='black', facecolors='lightblue', linewidths=0.05)
        ax.add_collection(poly)
        # Scaled limits for Meters (0.015m length)
        ax.set_xlim(-0.002, 0.020)
        ax.set_ylim(-0.006, 0.006)
        ax.set_aspect('equal')
        ax.set_title(f"Sample {idx} (Meters)")

    def generate_smooth_pathology(self, idx, v_type, lc):
        # Scale: L=15mm, W=~1.5mm in Meters
        L, W = 0.015, random.uniform(0.0012, 0.0018)
        bottom_pts = []
        for x in np.linspace(0, L, 6):
            # Random jitter scaled to meters (0.05mm)
            pt = gmsh.model.geo.addPoint(x, -W / 2 + random.uniform(-0.00005, 0.00005), 0, lc)
            bottom_pts.append(pt)
        s_bottom = gmsh.model.geo.addBSpline(bottom_pts)

        p_outlet_top = gmsh.model.geo.addPoint(L, W / 2, 0, lc)
        l_outlet = gmsh.model.geo.addLine(bottom_pts[-1], p_outlet_top)
        p_inlet_top = gmsh.model.geo.addPoint(0, W / 2, 0, lc)

        if v_type != 'straight':
            # Offset scaled to meters
            offset = (W * random.uniform(0.6, 0.8) * -1 if v_type in ['stenosis', 'occlusion']
                      else random.uniform(0.0005, 0.0015))
            p_t1 = gmsh.model.geo.addPoint(L * 0.8, W / 2, 0, lc)
            p_peak = gmsh.model.geo.addPoint(L * 0.5, W / 2 + offset, 0, lc)
            p_t2 = gmsh.model.geo.addPoint(L * 0.2, W / 2, 0, lc)
            s_top = gmsh.model.geo.addBSpline([p_outlet_top, p_t1, p_peak, p_t2, p_inlet_top])
        else:
            s_top = gmsh.model.geo.addLine(p_outlet_top, p_inlet_top)

        l_inlet = gmsh.model.geo.addLine(p_inlet_top, bottom_pts[0])
        return [s_bottom, l_outlet, s_top, l_inlet]

    def generate_bifurcation(self, idx, lc):
        # Scale units to meters (W ~ 1.5mm, L ~ 6mm)
        W, angle = random.uniform(0.0014, 0.0018), np.deg2rad(random.uniform(20, 40))
        L1, L2 = 0.006, 0.009
        fillet_radius = 0.0004

        p_in_b = gmsh.model.geo.addPoint(0, -W / 2, 0, lc)
        p_in_t = gmsh.model.geo.addPoint(0, W / 2, 0, lc)

        o1_t_x, o1_t_y = L1 + L2 * np.cos(angle), W / 2 + L2 * np.sin(angle)
        p_o1_t = gmsh.model.geo.addPoint(o1_t_x, o1_t_y, 0, lc)
        p_o1_b = gmsh.model.geo.addPoint(o1_t_x + W * np.sin(angle), o1_t_y - W * np.cos(angle), 0, lc)

        o2_b_x, o2_b_y = L1 + L2 * np.cos(angle), -W / 2 - L2 * np.sin(angle)
        p_o2_b = gmsh.model.geo.addPoint(o2_b_x, o2_b_y, 0, lc)
        p_o2_t = gmsh.model.geo.addPoint(o2_b_x + W * np.sin(angle), o2_b_y + W * np.cos(angle), 0, lc)

        p_f_center = gmsh.model.geo.addPoint(L1 + fillet_radius, 0, 0, lc)
        p_f_upper = gmsh.model.geo.addPoint(L1 + fillet_radius * 2, W * 0.2, 0, lc)
        p_f_lower = gmsh.model.geo.addPoint(L1 + fillet_radius * 2, -W * 0.2, 0, lc)

        l_in = gmsh.model.geo.addLine(p_in_t, p_in_b)
        mid_bot = gmsh.model.geo.addPoint(L1 * 0.6, -W / 2, 0, lc)
        l_wb = gmsh.model.geo.addBSpline([p_in_b, mid_bot, p_o2_b])
        l_o2 = gmsh.model.geo.addLine(p_o2_b, p_o2_t)
        l_fork_smooth = gmsh.model.geo.addBSpline([p_o2_t, p_f_lower, p_f_center, p_f_upper, p_o1_b])
        l_o1 = gmsh.model.geo.addLine(p_o1_b, p_o1_t)
        mid_top = gmsh.model.geo.addPoint(L1 * 0.6, W / 2, 0, lc)
        l_wt = gmsh.model.geo.addBSpline([p_o1_t, mid_top, p_in_t])

        curves = [l_in, l_wb, l_o2, l_fork_smooth, l_o1, l_wt]
        groups = {
            "Inlet": [l_in],
            "Outlet_1": [l_o1],
            "Outlet_2": [l_o2],
            "Walls": [l_wb, l_fork_smooth, l_wt]
        }
        return curves, groups

    def generate(self, idx, show_viz=False, ax=None):
        gmsh.model.add(f"vessel_{idx}")
        lc = 0.0002
        v_type = random.choice(['straight', 'stenosis', 'aneurysm', 'occlusion', 'bifurcation'])

        if v_type == 'bifurcation':
            curves, groups = self.generate_bifurcation(idx, lc)
        else:
            curves = self.generate_smooth_pathology(idx, v_type, lc)
            groups = {
                "Inlet": [curves[3]],
                "Outlet_1": [curves[1]],
                "Outlet_2": [],
                "Walls": [curves[0], curves[2]]
            }

        cl = gmsh.model.geo.addCurveLoop(curves)
        s = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        # Physical groups are still useful for partitioning fluid vs walls in Gmsh
        gmsh.model.addPhysicalGroup(1, groups["Inlet"], 101, name="Inlet")
        gmsh.model.addPhysicalGroup(1, groups["Outlet_1"], 102, name="Outlet_1")
        if groups["Outlet_2"]:
            gmsh.model.addPhysicalGroup(1, groups["Outlet_2"], 103, name="Outlet_2")
            gmsh.model.addPhysicalGroup(1, groups["Walls"], 104, name="Walls")
        else:
            gmsh.model.addPhysicalGroup(1, groups["Walls"], 103, name="Walls")

        gmsh.model.addPhysicalGroup(2, [s], 201, name="Fluid_Domain")

        # Generate Mesh
        gmsh.model.mesh.generate(2)

        # Visualize if requested
        if show_viz and ax:
            self.visualize_sample(idx, ax)

        # --- EXPORT SECTION ---
        # 1. Export Gmsh format (.msh) for the mesh_to_graph.py script
        # Ensure only physical groups are saved to keep the graph clean
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.write(str(self.output_dir / f"vessel_{idx}.msh"))

        # 2. Export Nastran format (.nas) for COMSOL
        gmsh.write(str(self.output_dir / f"vessel_{idx}.nas"))

        # Clear the model for the next iteration to prevent memory bloat
        gmsh.model.remove()

    def run_pipeline(self, n=5000):
        # Visualize first 9 samples
        fig, axes = plt.subplots(3, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i in range(9):
            self.generate(i, show_viz=True, ax=axes[i])
        plt.tight_layout()
        plt.show()

        # Run remainder
        for i in range(9, n):
            self.generate(i)
            if i % 100 == 0: print(f"Progress: {i}/{n} complete.")


if __name__ == "__main__":
    gen = VesselGenerator()
    # Starting with a smaller batch to verify units
    gen.run_pipeline(n=500)
    gmsh.finalize()