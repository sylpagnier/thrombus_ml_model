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
        project_root = current_script_path.parent.parent.parent.parent
        self.raw_dir = project_root / raw_dir
        self.label_dir = project_root / label_dir
        self.proc_dir = project_root / proc_dir
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_dynamic_dbar(self, nodes, sdf, mask_inlet):
        """
        Calculates characteristic length (Diameter) based strictly on Inlet.
        """
        # FIX: Convert boolean Tensor to Numpy for indexing
        mask_np = mask_inlet.numpy().astype(bool)

        if mask_np.any():
            # Get inlet nodes
            inlet_nodes = nodes[mask_np]
            # Diameter = Max Y - Min Y
            d_bar = inlet_nodes[:, 1].max() - inlet_nodes[:, 1].min()
            return d_bar

        # Fallback
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

        tree = KDTree(wall_pts)
        dist, _ = tree.query(nodes)
        sdf_dim = dist.reshape(-1, 1).astype(np.float32)

        d_bar = self._calculate_dynamic_dbar(nodes, sdf_dim, mask_inlet)
        d_bar_safe = d_bar if d_bar > 1e-9 else 1.0

        # --- MANDATORY PHYSICS SCALING ---
        # These constants must match your Phase 1 template exactly
        RHO = 1050.0
        MU_phase1 = 0.0035
        RE_TARGET = 150.0

        # Calculate u_ref for EVERY sample (not just anchors)
        u_ref = (RE_TARGET * MU_phase1) / (RHO * d_bar_safe)
        u_ref_safe = u_ref if u_ref > 1e-6 else 1.0

        nodes_nd = nodes / d_bar_safe
        sdf_nd = sdf_dim / d_bar_safe
        shear_pot_nd = torch.log(1.0 + 1.0 / (torch.tensor(sdf_nd) + 1e-6))

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
        has_labels = False  # Flag
        if label_path.exists():
            try:
                cfd = np.load(label_path)
                c_x, c_y = cfd['x'].flatten(), cfd['y'].flatten()
                c_u, c_v, c_p = cfd['u'].flatten(), cfd['v'].flatten(), cfd['p'].flatten()

                sol_points = np.stack([c_x, c_y], axis=-1)
                sol_tree = cKDTree(sol_points)
                _, idx = sol_tree.query(nodes)

                mapped_u, mapped_v, mapped_p = c_u[idx], c_v[idx], c_p[idx]

                outlet_indices = mask_outlet.nonzero(as_tuple=True)[0].numpy()
                p_ref = mapped_p[outlet_indices].mean() if len(outlet_indices) > 0 else 0.0

                u_nd_val = mapped_u / u_ref_safe
                v_nd_val = mapped_v / u_ref_safe
                dynamic_pressure = RHO * (u_ref_safe ** 2)
                p_nd_val = (mapped_p - p_ref) / dynamic_pressure

                y_labels = torch.stack([
                    torch.tensor(u_nd_val, dtype=torch.float32),
                    torch.tensor(v_nd_val, dtype=torch.float32),
                    torch.tensor(p_nd_val, dtype=torch.float32)
                ], dim=1)
                has_labels = True
            except Exception as e:
                print(f"Error reading labels for {stem}: {e}")
        else:
            print(f"Warning: No labels found for {stem}. Saving graph without y.")

        if y_labels is None:
            # Create dummy labels of shape [Num_Nodes, 3]
            y_labels = torch.zeros((num_nodes, 3), dtype=torch.float32)
            mask_supervised = torch.tensor([False], dtype=torch.bool)
        else:
            mask_supervised = torch.tensor([True], dtype=torch.bool)

        # 5. Save Final Data
        # We pack all features into data.x to simplify the GINO forward pass
        data = Data(
            x=torch.cat([
                torch.tensor(nodes_nd, dtype=torch.float32),
                torch.tensor(sdf_nd, dtype=torch.float32),
                shear_pot_nd.reshape(-1, 1)
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
        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".msh")])
        print(f"Processing {len(files)} meshes (Geometry + Labels)...")
        for f in tqdm(files):
            self.process_file(f)


if __name__ == "__main__":
    MeshToGraphComplete().run()