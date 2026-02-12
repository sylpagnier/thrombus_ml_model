import os
import torch
import numpy as np
import meshio
from pathlib import Path
from scipy.spatial import KDTree, cKDTree
from torch_geometric.data import Data
from tqdm import tqdm


class MeshToGraphComplete:
    def __init__(self, raw_dir="data/raw/synthetic_v1", label_dir="data/raw/cfd_anchors",
                 proc_dir="data/processed/tier1_graphs"):
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent.parent
        self.raw_dir = project_root / raw_dir
        self.label_dir = project_root / label_dir
        self.proc_dir = project_root / proc_dir
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_dynamic_dbar(self, nodes, sdf):
        """Calculates characteristic length based on geometry thickness."""
        local_max_radius = np.percentile(sdf, 95)
        d_bar = 2.0 * local_max_radius
        if d_bar < 1e-7:
            return nodes[:, 1].max() - nodes[:, 1].min()
        return d_bar

    def _get_boundary_masks(self, mesh, num_nodes):
        mask_inlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_outlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_wall = torch.zeros(num_nodes, dtype=torch.bool)

        if "line" not in mesh.cells_dict:
            return mask_inlet, mask_outlet, mask_wall

        line_cells = mesh.cells_dict["line"]
        line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
        unique_tags = set(line_tags)
        is_bifurcation = 104 in unique_tags

        for i, tag in enumerate(line_tags):
            nodes = line_cells[i]
            if tag == 101:
                mask_inlet[nodes] = True
            elif tag == 102:
                mask_outlet[nodes] = True
            elif tag == 103:
                if is_bifurcation:
                    mask_outlet[nodes] = True
                else:
                    mask_wall[nodes] = True
            elif tag == 104:
                mask_wall[nodes] = True

        return mask_inlet, mask_outlet, mask_wall

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

        # 2. Geometry Processing (Masks & SDF)
        mask_inlet, mask_outlet, mask_wall = self._get_boundary_masks(mesh, num_nodes)

        wall_pts = nodes[mask_wall.numpy()]
        if len(wall_pts) == 0:
            wall_pts = nodes[nodes[:, 1].abs() > 0.001]

        tree = KDTree(wall_pts)
        dist, _ = tree.query(nodes)
        sdf_dim = dist.reshape(-1, 1).astype(np.float32)

        d_bar = self._calculate_dynamic_dbar(nodes, sdf_dim)
        d_bar_safe = d_bar if d_bar > 1e-9 else 1.0

        nodes_nd = nodes / d_bar_safe
        sdf_nd = sdf_dim / d_bar_safe
        shear_pot_nd = torch.clamp(1.0 / (torch.tensor(sdf_nd) + 0.05), max=10.0)

        # 3. Connectivity
        if "triangle" not in mesh.cells_dict:
            return
        tri_nodes = mesh.cells_dict["triangle"]
        edges = np.unique(np.sort(np.vstack([
            tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)

        # 4. Label Injection (Integrated)
        y_labels = None
        if label_path.exists():
            try:
                cfd = np.load(label_path)

                # --- ROBUST LOADING START ---
                # Use .flatten() to ensure we have simple 1D arrays of length 1690
                c_x = cfd['x'].flatten()
                c_y = cfd['y'].flatten()
                c_u = cfd['u'].flatten()
                c_v = cfd['v'].flatten()
                c_p = cfd['p'].flatten()

                # Use np.stack with axis=-1 to guarantee shape (1690, 2)
                sol_points = np.stack([c_x, c_y], axis=-1)
                # --- ROBUST LOADING END ---

                # Build Tree for fast lookup (Now correctly 2D)
                sol_tree = cKDTree(sol_points)

                # Query the nearest COMSOL point for every Mesh Node
                # nodes (686, 2) matched against sol_points (1690, 2)
                d, idx = sol_tree.query(nodes)

                # Map values (using the flat arrays)
                mapped_u = c_u[idx]
                mapped_v = c_v[idx]
                mapped_p = c_p[idx]

                # --- UNIT CORRECTION ---
                RHO = 1060.0
                vel_mag = np.sqrt(mapped_u ** 2 + mapped_v ** 2)
                u_ref = np.percentile(vel_mag, 99)
                if u_ref < 1e-6: u_ref = 1.0

                outlet_indices = mask_outlet.nonzero(as_tuple=True)[0].numpy()
                p_ref = mapped_p[outlet_indices].mean() if len(outlet_indices) > 0 else 0.0

                u_nd_val = mapped_u / u_ref
                v_nd_val = mapped_v / u_ref
                dynamic_pressure = RHO * (u_ref ** 2)
                p_nd_val = (mapped_p - p_ref) / dynamic_pressure

                u_tensor = torch.tensor(u_nd_val, dtype=torch.float32).reshape(-1, 1)
                v_tensor = torch.tensor(v_nd_val, dtype=torch.float32).reshape(-1, 1)
                p_tensor = torch.tensor(p_nd_val, dtype=torch.float32).reshape(-1, 1)

                y_labels = torch.cat([u_tensor, v_tensor, p_tensor], dim=1)
            except Exception as e:
                print(f"Error reading labels for {stem}: {e}")
        else:
            print(f"Warning: No labels found for {stem}. Saving graph without y.")

        # 5. Save Final Data
        data = Data(
            x=torch.tensor(nodes_nd, dtype=torch.float32),
            edge_index=edge_index,
            sdf=torch.tensor(sdf_nd, dtype=torch.float32),
            shear_pot=shear_pot_nd,
            d_bar=torch.tensor([d_bar_safe], dtype=torch.float32),
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            y=y_labels
        )
        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self):
        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".msh")])
        print(f"Processing {len(files)} meshes (Geometry + Labels)...")
        for f in tqdm(files):
            self.process_file(f)


if __name__ == "__main__":
    MeshToGraphComplete().run()