import os
import random
import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


class VesselGenerator:
    def __init__(self, output_dir="data/raw/synthetic_v1"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)

    def _get_mesh_data(self):
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)[:, :2]
        node_dict = {tag: i for i, tag in enumerate(node_tags)}
        element_types, _, node_connectivity = gmsh.model.mesh.getElements(2)
        if not element_types: return None, None
        tri_nodes = node_connectivity[0].reshape(-1, 3)
        triangles = np.array([[node_dict[tag] for tag in tri] for tri in tri_nodes])
        return nodes, triangles

    def visualize_sample(self, idx, ax):
        nodes, triangles = self._get_mesh_data()
        if nodes is None: return
        verts = nodes[triangles]
        poly = PolyCollection(verts, edgecolors='black', facecolors='lightblue', linewidths=0.05)
        ax.add_collection(poly)
        ax.set_xlim(-2, 18)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.set_title(f"Smooth Sample {idx}")

    def generate_smooth_pathology(self, idx, v_type, lc):
        L, W = 15.0, random.uniform(1.2, 1.8)
        bottom_pts = []
        for x in np.linspace(0, L, 6):
            pt = gmsh.model.geo.addPoint(x, -W / 2 + random.uniform(-0.05, 0.05), 0, lc)
            bottom_pts.append(pt)
        s_bottom = gmsh.model.geo.addBSpline(bottom_pts)

        p_outlet_top = gmsh.model.geo.addPoint(L, W / 2, 0, lc)
        l_outlet = gmsh.model.geo.addLine(bottom_pts[-1], p_outlet_top)
        p_inlet_top = gmsh.model.geo.addPoint(0, W / 2, 0, lc)

        if v_type != 'straight':
            offset = (
                W * random.uniform(0.6, 0.8) * -1 if v_type in ['stenosis', 'occlusion'] else random.uniform(0.5, 1.5))
            p_t1 = gmsh.model.geo.addPoint(L * 0.8, W / 2, 0, lc)
            p_peak = gmsh.model.geo.addPoint(L * 0.5, W / 2 + offset, 0, lc)
            p_t2 = gmsh.model.geo.addPoint(L * 0.2, W / 2, 0, lc)
            s_top = gmsh.model.geo.addBSpline([p_outlet_top, p_t1, p_peak, p_t2, p_inlet_top])
        else:
            s_top = gmsh.model.geo.addLine(p_outlet_top, p_inlet_top)

        l_inlet = gmsh.model.geo.addLine(p_inlet_top, bottom_pts[0])
        return [s_bottom, l_outlet, s_top, l_inlet]

    def generate_bifurcation(self, idx, lc):
        W, angle = random.uniform(1.4, 1.8), np.deg2rad(random.uniform(20, 40))
        L1, L2 = 6.0, 9.0

        # Geometry constants for the "fillet" (smoothed crotch)
        fillet_radius = 0.4

        p_in_b = gmsh.model.geo.addPoint(0, -W / 2, 0, lc)
        p_in_t = gmsh.model.geo.addPoint(0, W / 2, 0, lc)

        o1_t_x, o1_t_y = L1 + L2 * np.cos(angle), W / 2 + L2 * np.sin(angle)
        p_o1_t = gmsh.model.geo.addPoint(o1_t_x, o1_t_y, 0, lc)
        p_o1_b = gmsh.model.geo.addPoint(o1_t_x + W * np.sin(angle), o1_t_y - W * np.cos(angle), 0, lc)

        o2_b_x, o2_b_y = L1 + L2 * np.cos(angle), -W / 2 - L2 * np.sin(angle)
        p_o2_b = gmsh.model.geo.addPoint(o2_b_x, o2_b_y, 0, lc)
        p_o2_t = gmsh.model.geo.addPoint(o2_b_x + W * np.sin(angle), o2_b_y + W * np.cos(angle), 0, lc)

        # Smoothed Apex Logic:
        # Create control points for a B-Spline curve at the fork
        p_f_center = gmsh.model.geo.addPoint(L1 + fillet_radius, 0, 0, lc)
        p_f_upper = gmsh.model.geo.addPoint(L1 + fillet_radius * 2, W * 0.2, 0, lc)
        p_f_lower = gmsh.model.geo.addPoint(L1 + fillet_radius * 2, -W * 0.2, 0, lc)

        l_in = gmsh.model.geo.addLine(p_in_t, p_in_b)

        mid_bot = gmsh.model.geo.addPoint(L1 * 0.6, -W / 2, 0, lc)
        l_wb = gmsh.model.geo.addBSpline([p_in_b, mid_bot, p_o2_b])
        l_o2 = gmsh.model.geo.addLine(p_o2_b, p_o2_t)

        # Fork is now a single B-Spline arc
        l_fork_smooth = gmsh.model.geo.addBSpline([p_o2_t, p_f_lower, p_f_center, p_f_upper, p_o1_b])

        l_o1 = gmsh.model.geo.addLine(p_o1_b, p_o1_t)
        mid_top = gmsh.model.geo.addPoint(L1 * 0.6, W / 2, 0, lc)
        l_wt = gmsh.model.geo.addBSpline([p_o1_t, mid_top, p_in_t])

        curves = [l_in, l_wb, l_o2, l_fork_smooth, l_o1, l_wt]
        groups = {
            "Inlet": [l_in],
            "Outlet": [l_o1, l_o2],
            "Walls": [l_wb, l_fork_smooth, l_wt]
        }
        return curves, groups

    def generate(self, idx, show_viz=False, ax=None):
        gmsh.model.add(f"vessel_{idx}")
        lc = 0.18
        v_type = random.choice(['straight', 'stenosis', 'aneurysm', 'occlusion', 'bifurcation'])
        if v_type == 'bifurcation':
            curves, groups = self.generate_bifurcation(idx, lc)
        else:
            curves = self.generate_smooth_pathology(idx, v_type, lc)
            groups = {"Inlet": [curves[3]], "Outlet": [curves[1]], "Walls": [curves[0], curves[2]]}

        cl = gmsh.model.geo.addCurveLoop(curves)
        s = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        for name, tags in groups.items():
            gmsh.model.addPhysicalGroup(1, tags, name=name)
        gmsh.model.addPhysicalGroup(2, [s], name="Fluid_Domain")
        gmsh.model.mesh.generate(2)
        if show_viz and ax: self.visualize_sample(idx, ax)
        gmsh.write(f"{self.output_dir}/vessel_{idx}.msh")
        gmsh.model.remove()

    def run_pipeline(self, n=5000):
        fig, axes = plt.subplots(3, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i in range(9): self.generate(i, show_viz=True, ax=axes[i])
        plt.tight_layout()
        plt.show()
        for i in range(9, n):
            self.generate(i)
            if i % 250 == 0: print(f"Progress: {i}/{n} complete.")


if __name__ == "__main__":
    gen = VesselGenerator()
    gen.run_pipeline(n=500)
    gmsh.finalize()