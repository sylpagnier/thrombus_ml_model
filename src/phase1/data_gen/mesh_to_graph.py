import os
import torch
import numpy as np
import meshio
from pathlib import Path
from scipy.spatial import KDTree, cKDTree
from torch_geometric.data import Data
from tqdm import tqdm
from src.utils.paths import get_project_root
from src.config import VesselConfig, PhysicsConfig

def _calculate_dynamic_dbar(nodes, mask_inlet):
    """
    Calculates characteristic length (Diameter) based strictly on Inlet.
    """
    # Ensure mask is numpy boolean for indexing
    if isinstance(mask_inlet, torch.Tensor):
        mask_np = mask_inlet.numpy().astype(bool)
    else:
        mask_np = mask_inlet.astype(bool)

    if mask_np.any():
        inlet_nodes = nodes[mask_np]
        # Diameter = Max Y - Min Y of the inlet
        d_bar = inlet_nodes[:, 1].max() - inlet_nodes[:, 1].min()
        return float(d_bar)

    # Fallback if inlet detection fails
    return 1.0


class MeshToGraphComplete:
    def __init__(self, raw_dir="data/raw/synthetic_v2", label_dir="data/raw/cfd_anchors",
                 proc_dir="data/processed/tier1_graphs_v2"):

        # Load Configs
        self.vessel_cfg = VesselConfig()
        self.physics_cfg = PhysicsConfig

        # Use your project root helper
        self.project_root = get_project_root()
        self.raw_dir = self.project_root / raw_dir
        self.label_dir = self.project_root / label_dir
        self.proc_dir = self.project_root / proc_dir
        self.proc_dir.mkdir(parents=True, exist_ok=True)

        # Standardized Tags (Must match VesselConfig in vessel_generator.py)
        self.TAG_INLET = self.vessel_cfg.TAGS["Inlet"]
        self.TAG_OUTLET_1 = self.vessel_cfg.TAGS["Outlet_1"]
        self.TAG_OUTLET_2 = self.vessel_cfg.TAGS["Outlet_2"]
        self.TAG_WALLS = self.vessel_cfg.TAGS["Walls"]

    def _get_boundary_masks(self, mesh, num_nodes):
        """
        Extracts boundary masks using the standardized Tag ID system.
        Returns explicit masks for Outlet 1 vs Outlet 2 for pressure pinning.
        """
        mask_inlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_outlet = torch.zeros(num_nodes, dtype=torch.bool)  # Combined Outlets
        mask_outlet_1 = torch.zeros(num_nodes, dtype=torch.bool)  # Primary Outlet Only
        mask_wall = torch.zeros(num_nodes, dtype=torch.bool)

        if "line" not in mesh.cells_dict:
            return mask_inlet, mask_outlet, mask_wall, mask_outlet_1

        try:
            line_cells = mesh.cells_dict["line"]
            line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
        except KeyError:
            print("Warning: Mesh does not contain line physical groups.")
            return mask_inlet, mask_outlet, mask_wall, mask_outlet_1

        for i, tag in enumerate(line_tags):
            nodes = line_cells[i]

            if tag == self.TAG_INLET:
                mask_inlet[nodes] = True
            elif tag == self.TAG_OUTLET_1:
                mask_outlet[nodes] = True
                mask_outlet_1[nodes] = True  # Pin Pressure Here
            elif tag == self.TAG_OUTLET_2:
                mask_outlet[nodes] = True
            elif tag == self.TAG_WALLS:
                mask_wall[nodes] = True

        return mask_inlet, mask_outlet, mask_wall, mask_outlet_1

    def process_file(self, filename):
        msh_path = self.raw_dir / filename
        stem = Path(filename).stem
        label_path = self.label_dir / f"{stem}.npz"

        # 1. Load Mesh
        try:
            mesh = meshio.read(msh_path)
        except Exception as e:
            print(f"Failed to read mesh {filename}: {e}")
            return

        nodes = mesh.points[:, :2]
        num_nodes = len(nodes)

        # 2. Geometry Processing & SDF
        mask_inlet, mask_outlet, mask_wall, mask_outlet_1 = self._get_boundary_masks(mesh, num_nodes)

        wall_pts = nodes[mask_wall.numpy()]
        if len(wall_pts) == 0:
            print(f"Skipping {filename}: No wall nodes found.")
            return

        # KDTree for SDF and Wall Normals
        tree = KDTree(wall_pts)
        dist, idx = tree.query(nodes)

        # --- PHYSICS FEATURE: WALL NORMALS ---
        # Vector pointing FROM Wall TO Node
        nearest_wall_coords = wall_pts[idx]
        wall_vecs = nodes - nearest_wall_coords

        # Normalize to unit vectors (Safe division)
        norms = np.linalg.norm(wall_vecs, axis=1, keepdims=True)
        wall_normals_nd = np.divide(wall_vecs, norms, out=np.zeros_like(wall_vecs), where=norms != 0)

        # SDF Dimensions
        sdf_dim = dist.reshape(-1, 1).astype(np.float32)

        # Calculate Characteristic Diameter (d_bar)
        d_bar = _calculate_dynamic_dbar(nodes, mask_inlet)
        d_bar_safe = d_bar if d_bar > 1e-9 else 1.0

        # --- PHYSICS SCALING ---
        rho = self.physics_cfg.rho
        mu_phase1 = self.physics_cfg.mu_newtonian
        re_target = self.physics_cfg.re_target

        u_ref = (re_target * mu_phase1) / (rho * d_bar_safe)
        u_ref_safe = u_ref if u_ref > 1e-6 else 1.0

        # Normalize Inputs
        nodes_nd = nodes / d_bar_safe
        sdf_nd = sdf_dim / d_bar_safe
        # Log-Shear Potential: High at wall (low SDF), Low at center
        shear_pot_nd = torch.log(1.0 + 1.0 / (torch.tensor(sdf_nd) + 1e-6))

        # 3. Connectivity (Edges)
        if "triangle" not in mesh.cells_dict:
            return
        tri_nodes = mesh.cells_dict["triangle"]
        edges = np.unique(np.sort(np.vstack([
            tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)

        # 4. Label Injection
        y_labels = None
        has_labels = False

        if label_path.exists():
            try:
                cfd = np.load(label_path)
                c_x, c_y = cfd['x'].flatten(), cfd['y'].flatten()
                c_u, c_v, c_p = cfd['u'].flatten(), cfd['v'].flatten(), cfd['p'].flatten()

                # Interpolate Ground Truth to Mesh Nodes
                sol_points = np.stack([c_x, c_y], axis=-1)
                sol_tree = cKDTree(sol_points)
                _, idx = sol_tree.query(nodes)

                mapped_u, mapped_v, mapped_p = c_u[idx], c_v[idx], c_p[idx]

                # --- PRESSURE PINNING ---
                # Pin Pressure to 0 at Outlet 1 specifically
                outlet_1_indices = mask_outlet_1.nonzero(as_tuple=True)[0].numpy()
                if len(outlet_1_indices) > 0:
                    p_ref = mapped_p[outlet_1_indices].mean()
                else:
                    # Fallback only if Outlet 1 is missing (e.g. error)
                    p_ref = mapped_p[mask_outlet.numpy().astype(bool)].mean()

                # Non-dimensionalize Labels
                u_nd_val = mapped_u / u_ref_safe
                v_nd_val = mapped_v / u_ref_safe
                dynamic_pressure = rho * (u_ref_safe ** 2)
                p_nd_val = (mapped_p - p_ref) / dynamic_pressure

                y_labels = torch.stack([
                    torch.tensor(u_nd_val, dtype=torch.float32),
                    torch.tensor(v_nd_val, dtype=torch.float32),
                    torch.tensor(p_nd_val, dtype=torch.float32)
                ], dim=1)
                has_labels = True
            except Exception as e:
                print(f"Error reading labels for {stem}: {e}")

        # Default labels for unsupervised samples
        if y_labels is None:
            y_labels = torch.zeros((num_nodes, 3), dtype=torch.float32)
            mask_supervised = torch.tensor([False], dtype=torch.bool)
        else:
            mask_supervised = torch.tensor([True], dtype=torch.bool)

        # 5. Save Data Object
        data = Data(
            x=torch.cat([
                torch.tensor(nodes_nd, dtype=torch.float32),  # [N, 2] Position
                torch.tensor(sdf_nd, dtype=torch.float32),  # [N, 1] SDF
                shear_pot_nd.reshape(-1, 1),  # [N, 1] Shear Pot
                torch.tensor(wall_normals_nd, dtype=torch.float32)  # [N, 2] Wall Normals
            ], dim=1),
            edge_index=edge_index,
            d_bar=torch.tensor([d_bar_safe], dtype=torch.float32),
            u_ref=torch.tensor([u_ref_safe], dtype=torch.float32),
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            y=y_labels,
            is_anchor=mask_supervised
        )
        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self):
        if not self.raw_dir.exists():
            print(f"Error: Raw directory not found: {self.raw_dir}")
            return

        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".msh")])
        print(f"Processing {len(files)} meshes from {self.raw_dir}...")

        for f in tqdm(files):
            self.process_file(f)


if __name__ == "__main__":
    converter = MeshToGraphComplete(
        raw_dir="data/raw/synthetic_v2",
        proc_dir="data/processed/tier1_graphs_v2"
    )
    converter.run()