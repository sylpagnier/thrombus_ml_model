import os
import torch
import json
import numpy as np
import meshio
from pathlib import Path
from scipy.spatial import KDTree, cKDTree
from torch_geometric.data import Data
from tqdm import tqdm
from src.config import VesselConfig, PhysicsConfig
from src.utils.paths import get_project_root


class MeshToGraphComplete:
    def __init__(self):
        self.root = get_project_root()
        self.vessel_cfg = VesselConfig()
        self.phys_cfg = PhysicsConfig()

        self.raw_dir = self.root / self.vessel_cfg.mesh_input_dir
        self.label_dir = self.root / self.vessel_cfg.output_dir
        self.proc_dir = self.root / self.vessel_cfg.graph_output_dir

        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def _get_boundary_masks(self, mesh, num_nodes, num_outlets):
        mask_inlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_outlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_wall = torch.zeros(num_nodes, dtype=torch.bool)

        # 1. Initialize empty to avoid "Unresolved reference"
        line_cells = []
        line_tags = []

        # 2. Define standard tags
        T_IN = 101
        T_OUT1 = 102
        T_OUT2 = 103
        T_WALL = 104 if num_outlets == 2 else 103

        # 3. Populate if data exists
        if "line" in mesh.cells_dict and "gmsh:physical" in mesh.cell_data_dict:
            line_cells = mesh.cells_dict["line"]
            line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]

        # 4. Loop is now safe even if line_tags is empty
        for i, tag in enumerate(line_tags):
            nodes = line_cells[i]
            if tag == T_IN:
                mask_inlet[nodes] = True
            elif tag == T_OUT1 or (num_outlets == 2 and tag == T_OUT2):
                mask_outlet[nodes] = True
            elif tag == T_WALL:
                mask_wall[nodes] = True

        return mask_inlet, mask_outlet, mask_wall

    def process_file(self, filename):
        stem = Path(filename).stem
        msh_path = self.raw_dir / filename
        json_path = self.raw_dir / f"{stem}.json"
        label_path = self.label_dir / f"{stem}.npz"

        if not msh_path.exists(): return

        try:
            mesh = meshio.read(msh_path)
            nodes = mesh.points[:, :2]
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            return

        # Restored: Fallback for d_bar if JSON is missing
        d_bar = None
        if json_path.exists():
            with open(json_path, 'r') as f:
                meta = json.load(f)
                d_bar = meta.get('d_bar')

        mask_inlet, mask_outlet, mask_wall = self._get_boundary_masks(mesh, len(nodes))

        if d_bar is None:
            if mask_inlet.any():
                inlet_nodes = nodes[mask_inlet]
                d_bar = np.max(inlet_nodes[:, 1]) - np.min(inlet_nodes[:, 1])

        # Physics Scaling
        u_ref = (self.phys_cfg.re_target * self.phys_cfg.mu_newtonian) / (self.phys_cfg.rho * d_bar)
        p_ref_scale = self.phys_cfg.rho * (u_ref ** 2)

        # SDF and Gradients
        wall_node_indices = np.where(mask_wall.numpy())[0]
        if len(wall_node_indices) == 0: return
        wall_pts = nodes[wall_node_indices]
        tree = KDTree(wall_pts)
        dist_raw, indices = tree.query(nodes)

        nearest_wall_pts = wall_pts[indices]
        diff_vec = nodes - nearest_wall_pts
        sdf_grad = diff_vec / (dist_raw[:, None] + 1e-8)

        # Restored: Spatial Label Mapping via cKDTree
        y_labels = torch.zeros((len(nodes), 3), dtype=torch.float32)
        is_anchor = False
        if label_path.exists():
            try:
                cfd = np.load(label_path)
                # Map CFD coordinates to mesh nodes
                sol_points = np.stack([cfd['x'].flatten(), cfd['y'].flatten()], axis=-1)
                sol_tree = cKDTree(sol_points)
                _, idx = sol_tree.query(nodes)

                u_raw = torch.tensor(cfd['u'].flatten()[idx], dtype=torch.float32)
                v_raw = torch.tensor(cfd['v'].flatten()[idx], dtype=torch.float32)
                p_raw = torch.tensor(cfd['p'].flatten()[idx], dtype=torch.float32)

                # Normalize
                u_nd, v_nd = u_raw / u_ref, v_raw / u_ref
                outlet_idx = mask_outlet.nonzero(as_tuple=True)[0]
                p_offset = p_raw[outlet_idx].mean() if len(outlet_idx) > 0 else 0.0
                p_nd = (p_raw - p_offset) / p_ref_scale

                y_labels = torch.stack([u_nd, v_nd, p_nd], dim=1)
                is_anchor = True
            except Exception as e:
                print(f"Error mapping labels {filename}: {e}")

        # Inlet BC Calculation
        u_inlet_bc = torch.zeros((len(nodes), 2), dtype=torch.float32)
        if mask_inlet.any():
            inlet_indices = torch.where(mask_inlet)[0]
            y_coords = nodes[inlet_indices, 1]
            y_center = np.mean(y_coords)
            R = (np.max(y_coords) - np.min(y_coords)) / 2.0
            u_max = 1.5 * u_ref
            r = np.abs(y_coords - y_center)
            profile_mag = u_max * (1 - (r ** 2 / (R ** 2 + 1e-12)))
            u_inlet_bc[inlet_indices, 0] = torch.tensor(profile_mag / u_ref, dtype=torch.float32)

        # Graph Assembly
        nodes_nd = nodes / d_bar
        sdf_nd = dist_raw.reshape(-1, 1) / d_bar
        shear_pot = torch.log(1.0 + 1.0 / (torch.tensor(sdf_nd, dtype=torch.float32) + 1e-6))

        if "triangle" not in mesh.cells_dict: return
        tri_nodes = mesh.cells_dict["triangle"]
        edges = np.unique(np.sort(np.vstack([
            tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)

        node_type = torch.zeros((len(nodes), 4), dtype=torch.float32)

        # 1. Default all nodes to Internal (Index 0)
        node_type[:, 0] = 1.0

        # 2. Assign Inlet (Index 1)
        node_type[mask_inlet, 0] = 0.0
        node_type[mask_inlet, 1] = 1.0

        # 3. Assign Outlet (Index 2)
        node_type[mask_outlet, 0] = 0.0
        node_type[mask_outlet, 2] = 1.0

        # 4. Assign Wall (Index 3) - Walls override others if nodes overlap
        node_type[mask_wall, 0] = 0.0
        node_type[mask_wall, 3] = 1.0

        # Assembly: Concatenate the 4-channel one-hot features to the x_tensor
        x_tensor = torch.cat([
            torch.tensor(nodes_nd, dtype=torch.float32),
            torch.tensor(sdf_nd, dtype=torch.float32),
            shear_pot,
            torch.tensor(sdf_grad, dtype=torch.float32),
            node_type  # Added One-Hot Features
        ], dim=1)

        data = Data(
            x=x_tensor, edge_index=edge_index, y=y_labels,
            d_bar=torch.tensor([d_bar], dtype=torch.float32),
            u_ref=torch.tensor([u_ref], dtype=torch.float32),
            u_inlet_bc=u_inlet_bc,
            mask_inlet=mask_inlet, mask_outlet=mask_outlet, mask_wall=mask_wall,
            is_anchor=torch.tensor([is_anchor], dtype=torch.bool)
        )
        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self):
        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".msh")])
        for f in tqdm(files):
            self.process_file(f)


if __name__ == "__main__":
    MeshToGraphComplete().run()