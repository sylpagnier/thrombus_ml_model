import os
import torch
import json
import numpy as np
import meshio
from pathlib import Path
from scipy.spatial import KDTree
from torch_geometric.data import Data
from tqdm import tqdm
from src.config import VesselConfig, PhysicsConfig
from src.utils.paths import get_project_root


class MeshToGraphComplete:
    def __init__(self):
        self.root = get_project_root()
        self.vessel_cfg = VesselConfig()
        self.phys_cfg = PhysicsConfig()

        # Resolve paths via Config
        self.raw_dir = self.root / self.vessel_cfg.mesh_input_dir
        self.label_dir = self.root / self.vessel_cfg.output_dir
        self.proc_dir = self.root / self.vessel_cfg.graph_output_dir

        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def _get_boundary_masks(self, mesh, num_nodes):
        mask_inlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_outlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_wall = torch.zeros(num_nodes, dtype=torch.bool)

        T_IN = self.vessel_cfg.TAGS["Inlet"]
        T_OUT1 = self.vessel_cfg.TAGS["Outlet_1"]
        T_OUT2 = self.vessel_cfg.TAGS["Outlet_2"]
        T_WALL = self.vessel_cfg.TAGS["Walls"]

        if "line" in mesh.cells_dict and "gmsh:physical" in mesh.cell_data_dict:
            line_cells = mesh.cells_dict["line"]
            line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
            for i, tag in enumerate(line_tags):
                nodes = line_cells[i]
                if tag == T_IN:
                    mask_inlet[nodes] = True
                elif tag == T_OUT1:
                    mask_outlet[nodes] = True
                elif tag == T_OUT2:
                    mask_outlet[nodes] = True
                elif tag == T_WALL:
                    mask_wall[nodes] = True
        return mask_inlet, mask_outlet, mask_wall

    def process_file(self, filename):
        stem = Path(filename).stem
        msh_path = self.raw_dir / filename
        json_path = self.raw_dir / f"{stem}.json"
        label_path = self.label_dir / f"{stem}.npz"

        # Check inputs
        if not msh_path.exists() or not json_path.exists(): return

        try:
            mesh = meshio.read(msh_path)
            nodes = mesh.points[:, :2]
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            return

        with open(json_path, 'r') as f:
            meta = json.load(f)
            d_bar = meta.get('d_bar')
            num_outlets = meta.get('num_outlets', 1)

        # Physics Scaling (Re = Target)
        u_ref = (self.phys_cfg.re_target * self.phys_cfg.mu_newtonian) / (self.phys_cfg.rho * d_bar)
        p_ref_scale = self.phys_cfg.rho * (u_ref ** 2)

        # Geometry Features
        mask_inlet, mask_outlet, mask_wall = self._get_boundary_masks(mesh, len(nodes))

        # SDF Calculation
        wall_node_indices = np.where(mask_wall.numpy())[0]
        if len(wall_node_indices) == 0: return

        wall_pts = nodes[wall_node_indices]
        tree = KDTree(wall_pts)
        dist_raw, indices = tree.query(nodes)

        # SDF Gradient
        nearest_wall_pts = wall_pts[indices]
        diff_vec = nodes - nearest_wall_pts
        sdf_grad = diff_vec / (dist_raw[:, None] + 1e-8)

        # Initialize BC tensor (N, 2) - default is 0.0
        u_inlet_bc = torch.zeros((len(nodes), 2), dtype=torch.float32)

        if mask_inlet.any():
            # Extract inlet nodes (assuming vertical inlet at x=0, varying y)
            inlet_indices = torch.where(mask_inlet)[0]
            inlet_nodes = nodes[inlet_indices]

            # 1. Geometry of the inlet
            y_coords = inlet_nodes[:, 1]
            y_center = np.mean(y_coords)
            # Calculate half-width (radius) of the inlet
            R = (np.max(y_coords) - np.min(y_coords)) / 2.0

            # 2. Parabolic Profile Calculation (Poiseuille)
            # U(r) = U_max * (1 - (r/R)^2)
            # For 2D Planar Channels: U_max = 1.5 * U_avg
            # For 3D Pipes: U_max = 2.0 * U_avg
            # Using 1.5 since prompt specifies "Synthetic 2D channels"
            u_max = 1.5 * u_ref

            r = np.abs(y_coords - y_center)
            # Avoid divide by zero
            R_sq = R ** 2 + 1e-12

            # Calculate magnitude
            profile_mag = u_max * (1 - (r ** 2 / R_sq))

            # 3. Assign and Normalize
            # Assign to x-component (Index 0) assuming flow moves +x
            # Normalize by u_ref to match the network's output scale
            u_inlet_bc[inlet_indices, 0] = torch.tensor(profile_mag / u_ref, dtype=torch.float32)

        if d_bar is None or d_bar <= 0:
            print(f"Invalid d_bar for {filename}, skipping.")
            return

        # Normalize Inputs
        nodes_nd = nodes / d_bar
        sdf_nd = dist_raw.reshape(-1, 1) / d_bar
        shear_pot = torch.log(1.0 + 1.0 / (torch.tensor(sdf_nd, dtype=torch.float32) + 1e-6))

        # Connectivity
        if "triangle" not in mesh.cells_dict: return
        tri_nodes = mesh.cells_dict["triangle"]
        edges = np.unique(np.sort(np.vstack([
            tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)

        # Labels (CFD Results)
        y_labels = torch.zeros((len(nodes), 3), dtype=torch.float32)
        is_anchor = False
        if label_path.exists():
            try:
                cfd = np.load(label_path)
                u_raw = torch.tensor(cfd['u'], dtype=torch.float32)
                v_raw = torch.tensor(cfd['v'], dtype=torch.float32)
                p_raw = torch.tensor(cfd['p'], dtype=torch.float32)

                # Normalize Labels
                u_nd = u_raw / u_ref
                v_nd = v_raw / u_ref

                outlet_idx = mask_outlet.nonzero(as_tuple=True)[0]
                p_offset = p_raw[outlet_idx].mean() if len(outlet_idx) > 0 else 0.0
                p_nd = (p_raw - p_offset) / p_ref_scale

                y_labels = torch.stack([u_nd, v_nd, p_nd], dim=1)
                is_anchor = True
            except Exception as e:
                print(f"Error loading labels {filename}: {e}")

        # Data Object
        x_tensor = torch.cat([
            torch.tensor(nodes_nd, dtype=torch.float32),
            torch.tensor(sdf_nd, dtype=torch.float32),
            shear_pot,
            torch.tensor(sdf_grad, dtype=torch.float32)
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
        print(f"Processing {len(files)} meshes from {self.raw_dir}...")
        for f in tqdm(files):
            self.process_file(f)


if __name__ == "__main__":
    MeshToGraphComplete().run()