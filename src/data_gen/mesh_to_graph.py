import os
import torch
import numpy as np
import meshio
from pathlib import Path
from scipy.spatial import KDTree, cKDTree
from torch_geometric.data import Data
from tqdm import tqdm


class MeshToGraphConverter:
    def __init__(self, raw_dir="data/raw/synthetic_v1", proc_dir="data/processed/tier1_graphs"):
        current_script_path = Path(__file__).resolve()
        project_root = current_script_path.parent.parent.parent
        self.raw_dir = project_root / raw_dir
        self.proc_dir = project_root / proc_dir
        self.label_dir = project_root / "data" / "raw" / "cfd_anchors"
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def _calculate_dynamic_dbar(self, nodes, sdf):
        x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
        length = x_max - x_min
        if length < 1e-9: return 1.0

        samples = np.linspace(x_min + 0.10 * length, x_max - 0.10 * length, 20)
        slice_tol = 0.01 * length

        radii = []
        for s in samples:
            mask = np.abs(nodes[:, 0] - s) < slice_tol
            if mask.any():
                radii.append(sdf[mask].max())
        return 2.0 * np.mean(radii) if radii else (nodes[:, 1].max() - nodes[:, 1].min())

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

        try:
            mesh = meshio.read(msh_path)
        except Exception:
            return

        nodes = mesh.points[:, :2]
        num_nodes = len(nodes)

        # 1. Masks & SDF
        mask_inlet, mask_outlet, mask_wall = self._get_boundary_masks(mesh, num_nodes)

        wall_pts = nodes[mask_wall.numpy()]
        if len(wall_pts) == 0: wall_pts = nodes[nodes[:, 1].abs() > 0.001]

        tree = KDTree(wall_pts)
        dist, _ = tree.query(nodes)
        sdf_dim = dist.reshape(-1, 1).astype(np.float32)

        d_bar = self._calculate_dynamic_dbar(nodes, sdf_dim)
        d_bar_safe = d_bar if d_bar > 1e-9 else 1.0

        nodes_nd = nodes / d_bar_safe
        sdf_nd = sdf_dim / d_bar_safe
        shear_pot_nd = torch.clamp(1.0 / (torch.tensor(sdf_nd) + 0.05), max=10.0)

        # 2. Connectivity
        if "triangle" not in mesh.cells_dict: return
        tri_nodes = mesh.cells_dict["triangle"]
        edges = np.unique(np.sort(np.vstack([
            tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)

        # 3. Hybrid Labels (WITH SPATIAL MAPPING FIX)
        y_labels = None
        if label_path.exists():
            cfd = np.load(label_path)

            # Retrieve COMSOL coordinates and values
            c_x, c_y = cfd['x'], cfd['y']
            c_u, c_v, c_p = cfd['u'], cfd['v'], cfd['p']

            # Stack COMSOL points: [N_sol, 2]
            sol_points = np.column_stack((c_x, c_y))

            # Build Tree for fast lookup
            sol_tree = cKDTree(sol_points)

            # Query the nearest COMSOL point for every Mesh Node
            # k=1 returns distances (d) and indices (idx)
            d, idx = sol_tree.query(nodes)

            # Map values
            mapped_u = c_u[idx]
            mapped_v = c_v[idx]
            mapped_p = c_p[idx]

            # Outlet Pressure Pinning
            outlet_indices = mask_outlet.nonzero(as_tuple=True)[0].numpy()
            p_ref = mapped_p[outlet_indices].mean() if len(outlet_indices) > 0 else 0.0

            u_nd = torch.tensor(mapped_u, dtype=torch.float32).reshape(-1, 1)
            v_nd = torch.tensor(mapped_v, dtype=torch.float32).reshape(-1, 1)
            p_nd = torch.tensor((mapped_p - p_ref), dtype=torch.float32).reshape(-1, 1)

            y_labels = torch.cat([u_nd, v_nd, p_nd], dim=1)

        data = Data(
            x=torch.tensor(nodes_nd, dtype=torch.float32),
            edge_index=edge_index,
            sdf=torch.tensor(sdf_nd, dtype=torch.float32),
            shear_pot=shear_pot_nd,
            d_bar=torch.tensor([d_bar_safe]),
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            y=y_labels
        )
        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self):
        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".msh")])
        print(f"Converting {len(files)} meshes...")
        for f in tqdm(files):
            self.process_file(f)


if __name__ == "__main__":
    converter = MeshToGraphConverter()
    converter.run()