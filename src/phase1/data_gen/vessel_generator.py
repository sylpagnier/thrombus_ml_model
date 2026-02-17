import json
import random
import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from src.utils.paths import get_project_root
from src.config import VesselConfig


class VesselGenerator:
    def __init__(self):
        self.cfg = VesselConfig
        self.project_root = get_project_root()

        # Resolve output directory from Config
        self.output_dir = self.project_root / self.cfg.mesh_input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._configure_gmsh_options()

    def _configure_gmsh_options(self):
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

    def _get_mesh_data(self):
        """Extracts nodes and triangles for visualization."""
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)[:, :2]
        node_dict = {tag: i for i, tag in enumerate(node_tags)}
        element_types, _, node_connectivity = gmsh.model.mesh.getElements(2)
        if not element_types or len(node_connectivity) == 0:
            return None, None
        tri_nodes = node_connectivity[0].reshape(-1, 3)
        triangles = np.array([[node_dict[tag] for tag in tri] for tri in tri_nodes])
        return nodes, triangles

    def visualize_sample(self, idx, ax):
        """Visualizes the generated mesh on a Matplotlib axis."""
        nodes, triangles = self._get_mesh_data()
        if nodes is None:
            return
        verts = nodes[triangles]
        poly = PolyCollection(verts, edgecolors='black', facecolors='lightblue', linewidths=0.1)
        ax.add_collection(poly)
        # Fixed limits for standard scale (Meters)
        ax.set_xlim(-0.002, 0.020)
        ax.set_ylim(-0.008, 0.008)
        ax.set_aspect('equal')
        ax.set_title(f"Sample {idx}")

    def _save_metadata(self, idx, d_bar, vessel_type, level, num_outlets):
        meta = {
            "id": idx,
            "type": vessel_type,
            "level": level,
            "d_bar": float(d_bar),
            "num_outlets": num_outlets,
        }
        with open(self.output_dir / f"vessel_{idx}.json", "w") as f:
            json.dump(meta, f, indent=4)

    def _calculate_pathology_offsets(self, v_type, width):
        offsets = np.zeros(self.cfg.num_ctrl_pts)
        peak_idx = self.cfg.num_ctrl_pts // 2

        if v_type in ['stenosis', 'occlusion']:
            max_off = -random.uniform(self.cfg.stenosis_factor_min * width, self.cfg.stenosis_factor_max * width)
            offsets[peak_idx - 1: peak_idx + 2] = [max_off * 0.6, max_off, max_off * 0.6]
        elif v_type == 'aneurysm':
            max_off = random.uniform(self.cfg.aneurysm_factor_min * width, self.cfg.aneurysm_factor_max * width)
            offsets[peak_idx - 1: peak_idx + 2] = [max_off * 0.6, max_off, max_off * 0.6]
        return offsets

    def generate_bifurcation(self):
        w_parent = random.uniform(self.cfg.width_min, self.cfg.width_max)
        split_ratio = random.uniform(0.4, 0.8)
        w_d1 = w_parent * split_ratio
        w_d2 = w_parent * (1.2 - split_ratio)

        angle_1 = np.deg2rad(random.uniform(self.cfg.bifurcation_angle_min, self.cfg.bifurcation_angle_max))
        angle_2 = np.deg2rad(random.uniform(25, 60))

        lc = self.cfg.mesh_lc
        L1, L2 = self.cfg.bifurcation_l1, self.cfg.bifurcation_l2
        f_rad = 0.0004

        # Geometry Points
        p_in_t = gmsh.model.geo.addPoint(0, w_parent / 2, 0, lc)
        p_in_b = gmsh.model.geo.addPoint(0, -w_parent / 2, 0, lc)
        l_in = gmsh.model.geo.addLine(p_in_b, p_in_t)

        o1_x = L1 + L2 * np.cos(angle_1)
        o1_y = (w_parent / 2) + L2 * np.sin(angle_1)
        p_o1_t = gmsh.model.geo.addPoint(o1_x - (w_d1 / 2) * np.sin(angle_1), o1_y + (w_d1 / 2) * np.cos(angle_1), 0,
                                         lc)
        p_o1_b = gmsh.model.geo.addPoint(o1_x + (w_d1 / 2) * np.sin(angle_1), o1_y - (w_d1 / 2) * np.cos(angle_1), 0,
                                         lc)
        l_o1 = gmsh.model.geo.addLine(p_o1_t, p_o1_b)

        o2_x = L1 + L2 * np.cos(angle_2)
        o2_y = -(w_parent / 2) - L2 * np.sin(angle_2)
        p_o2_t = gmsh.model.geo.addPoint(o2_x + (w_d2 / 2) * np.sin(angle_2), o2_y + (w_d2 / 2) * np.cos(angle_2), 0,
                                         lc)
        p_o2_b = gmsh.model.geo.addPoint(o2_x - (w_d2 / 2) * np.sin(angle_2), o2_y - (w_d2 / 2) * np.cos(angle_2), 0,
                                         lc)
        l_o2 = gmsh.model.geo.addLine(p_o2_t, p_o2_b)

        mid_top = gmsh.model.geo.addPoint(L1 * 0.5, w_parent / 2, 0, lc)
        l_wt = gmsh.model.geo.addBSpline([p_in_t, mid_top, p_o1_t])

        mid_bot = gmsh.model.geo.addPoint(L1 * 0.5, -w_parent / 2, 0, lc)
        l_wb = gmsh.model.geo.addBSpline([p_o2_b, mid_bot, p_in_b])

        p_f_upper = gmsh.model.geo.addPoint(L1 + f_rad, w_parent * 0.1, 0, lc)
        p_f_center = gmsh.model.geo.addPoint(L1, 0, 0, lc)
        p_f_lower = gmsh.model.geo.addPoint(L1 + f_rad, -w_parent * 0.1, 0, lc)
        l_fork = gmsh.model.geo.addBSpline([p_o1_b, p_f_upper, p_f_center, p_f_lower, p_o2_t])

        curves = [l_in, l_wt, l_o1, l_fork, l_o2, l_wb]
        groups = {
            "Inlet": [l_in], "Outlet_1": [l_o1], "Outlet_2": [l_o2],
            "Walls": [l_wt, l_fork, l_wb]
        }
        return curves, groups, w_parent

    def generate_spline_vessel(self, v_type, is_curved):
        width = random.uniform(self.cfg.width_min, self.cfg.width_max)
        length = self.cfg.base_length
        lc = self.cfg.mesh_lc

        c_pts = []
        for i in range(self.cfg.num_ctrl_pts):
            x = (i / (self.cfg.num_ctrl_pts - 1)) * length

            # Strictly align the first point to y=0 for centered inlets
            if i == 0:
                y = 0.0
            else:
                y = random.uniform(-self.cfg.curvature_amplitude, self.cfg.curvature_amplitude) if is_curved else 0.0

            c_pts.append(np.array([x, y]))

        offsets = self._calculate_pathology_offsets(v_type, width)
        widths = [width + (2 * o) if o > 0 else width - (2 * abs(o)) for o in offsets]
        d_bar_true = np.mean(widths)

        top_tags, bot_tags = [], []
        for i in range(self.cfg.num_ctrl_pts):
            if i < self.cfg.num_ctrl_pts - 1:
                tangent = c_pts[i + 1] - c_pts[i]
            else:
                tangent = c_pts[i] - c_pts[i - 1]
            norm = np.array([-tangent[1], tangent[0]])
            norm = norm / (np.linalg.norm(norm) + 1e-9)

            t_off = (width / 2) + offsets[i]
            b_off = -(width / 2)

            p_top = c_pts[i] + norm * t_off
            p_bot = c_pts[i] + norm * b_off
            top_tags.append(gmsh.model.geo.addPoint(p_top[0], p_top[1], 0, lc))
            bot_tags.append(gmsh.model.geo.addPoint(p_bot[0], p_bot[1], 0, lc))

        s_top = gmsh.model.geo.addBSpline(top_tags)
        l_out = gmsh.model.geo.addLine(top_tags[-1], bot_tags[-1])
        s_bot = gmsh.model.geo.addBSpline(bot_tags[::-1])
        l_in = gmsh.model.geo.addLine(bot_tags[0], top_tags[0])

        curves = [s_top, l_out, s_bot, l_in]
        groups = {
            "Inlet": [l_in], "Outlet_1": [l_out], "Outlet_2": [],
            "Walls": [s_top, s_bot]
        }
        return curves, groups, d_bar_true

    def generate(self, idx, level=1, show_viz=False, ax=None):
        gmsh.model.add(f"vessel_{idx}")
        try:
            is_bifurcation = False
            is_curved = False
            v_type = 'straight'

            # 3-Tier Level Logic
            if level == 1:
                v_type = random.choice(['straight', 'stenosis', 'aneurysm'])
                is_curved = False
            elif level == 2:
                v_type = random.choice(['straight', 'stenosis', 'aneurysm'])
                is_curved = True
            else:  # Level 3
                if random.random() > 0.7:
                    is_bifurcation = True
                else:
                    v_type = random.choice(['straight', 'stenosis', 'aneurysm'])
                    is_curved = random.choice([True, False])

            if is_bifurcation:
                curves, groups, d_bar = self.generate_bifurcation()
                v_str = "bifurcation"
                num_outlets = 2
            else:
                curves, groups, d_bar = self.generate_spline_vessel(v_type, is_curved)
                v_str = f"{v_type}_{'curved' if is_curved else 'straight'}"
                num_outlets = 1

            cl = gmsh.model.geo.addCurveLoop(curves)
            s = gmsh.model.geo.addPlaneSurface([cl])
            gmsh.model.geo.synchronize()

            gmsh.model.addPhysicalGroup(1, groups["Inlet"], 101, name="Inlet")
            gmsh.model.addPhysicalGroup(1, groups["Outlet_1"], 102, name="Outlet_1")

            if is_bifurcation:
                # Outlet_2 gets 103
                gmsh.model.addPhysicalGroup(1, groups["Outlet_2"], 103, name="Outlet_2")
                # Walls get 104
                gmsh.model.addPhysicalGroup(1, groups["Walls"], 104, name="Walls")
            else:
                # Straight vessels: Walls get 103 (because there is no second outlet)
                gmsh.model.addPhysicalGroup(1, groups["Walls"], 103, name="Walls")

            gmsh.model.addPhysicalGroup(2, [s], 201, name="Fluid_Domain")

            gmsh.model.mesh.generate(2)

            if show_viz and ax:
                self.visualize_sample(idx, ax)

            gmsh.write(str(self.output_dir / f"vessel_{idx}.msh"))
            gmsh.write(str(self.output_dir / f"vessel_{idx}.nas"))

            self._save_metadata(idx, d_bar, v_str, level, num_outlets)

        except Exception as e:
            print(f"Error generating sample {idx}: {e}")
        finally:
            gmsh.model.remove()

    def run_pipeline(self, n=50, level=1):
        print(f" Generating Level {level} Dataset (N={n}) in {self.output_dir}...")

        # Visualize first 9 samples in a grid
        fig, axes = plt.subplots(3, 3, figsize=(15, 8))
        axes = axes.flatten()
        for i in range(min(9, n)):
            self.generate(i, level=level, show_viz=True, ax=axes[i])
        plt.tight_layout()
        plt.show()

        # Generate remainder
        for i in range(9, n):
            self.generate(i, level=level)
            if i % 50 == 0:
                print(f"Progress: {i}/{n}")


if __name__ == "__main__":
    config = VesselConfig()
    vg = VesselGenerator()
    vg.run_pipeline(n=50, level=1)  # Level 1=Straight, 2=Curved, 3=Bifurcations
    gmsh.finalize()