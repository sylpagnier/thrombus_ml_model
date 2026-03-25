import os
import json
import torch
import numpy as np
import pandas as pd
import meshio
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree, KDTree
from torch_geometric.data import Data
from tqdm import tqdm

from src.config import VesselConfig, PhysicsConfig
from src.utils.paths import get_project_root


class PatientDataExtractor:
    """
    Extracts and processes Eulerian node-wise COMSOL data into PyTorch Geometric Data objects.

    --- COMSOL Export Instructions ---
    To make this script work, export the exact node-wise data from COMSOL to match the .msh topology.

    1. In COMSOL: Go to Results > Export > Data.
    2. Main Domain: Export domain nodes to `data/processed/cfd_results_tier3_patients/<stem>.txt`
       Headers must map exactly to:
       x, y, u, v, p, mu_effective, rp, ap, apr, aps, PT, th, at, fg, fi, M, Mas, Mat
    3. Boundaries: Export Edge 2D coordinates (x, y) with "Time Selection: Last" to:
       - <stem>_inlet.txt
       - <stem>_outlet.txt
       - <stem>_wall.txt
    ----------------------------------
    """

    def __init__(self, tier="tier3_patients", raw_dir=None, label_dir=None, proc_dir=None, visualize=True):
        self.root = get_project_root()
        self.vessel_cfg = VesselConfig(tier=tier)
        self.phys_cfg = PhysicsConfig(tier=tier)
        self.visualize = visualize

        # Directory handling
        self.raw_dir = Path(raw_dir) if raw_dir else self.root / self.vessel_cfg.mesh_input_dir
        self.label_dir = Path(label_dir) if label_dir else self.root / self.vessel_cfg.output_dir
        self.proc_dir = Path(proc_dir) if proc_dir else self.root / self.vessel_cfg.graph_output_dir
        self.proc_dir.mkdir(parents=True, exist_ok=True)

        if self.visualize:
            self.vis_dir = self.proc_dir / "sanity_checks"
            self.vis_dir.mkdir(exist_ok=True)

        # Dictionary mapping exact COMSOL export names to standardized internal names
        self.species_map = {
            'rp': 'RP', 'ap': 'AP', 'apr': 'APR', 'aps': 'APS',
            'PT': 'PT', 'th': 'T', 'at': 'AT', 'fg': 'FG',
            'fi': 'FI', 'M': 'M', 'Mas': 'Mas', 'Mat': 'Mat'
        }

        self.csv_fields = ['x', 'y', 'u', 'v', 'p', 'mu_effective'] + list(self.species_map.keys())

    def _precompute_wls(self, edge_index, num_nodes, pos_tensor):
        """Computes the 2nd Order Polynomial Basis and WLS Inverse Matrix."""
        row, col = edge_index
        pos_diff = pos_tensor[col, :2] - pos_tensor[row, :2]
        dx, dy = pos_diff[:, 0], pos_diff[:, 1]

        dist_sq = dx ** 2 + dy ** 2 + 1e-8

        dx2 = 0.5 * dx ** 2
        dxy = dx * dy
        dy2 = 0.5 * dy ** 2

        V = torch.stack([dx, dy, dx2, dxy, dy2], dim=1)
        W = 1.0 / dist_sq

        V_unsqueezed = V.unsqueeze(2)
        V_T_unsqueezed = V.unsqueeze(1)
        M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)

        M_e_flat = M_e.view(-1, 25)
        out = torch.zeros((num_nodes, 25), dtype=M_e_flat.dtype, device=M_e_flat.device)
        row_exp = row.view(-1, 1).expand_as(M_e_flat)
        M_flat = out.scatter_add_(0, row_exp, M_e_flat)

        M = M_flat.view(num_nodes, 5, 5)
        epsilon = 1e-6
        I = torch.eye(5, dtype=M.dtype, device=M.device).unsqueeze(0).expand(num_nodes, 5, 5)
        M_reg = M + epsilon * I
        M_inv = torch.linalg.pinv(M_reg)

        return V, W, M_inv.squeeze(1)

    def _precompute_sparse_operators(self, edge_index, num_nodes, M_inv, V, W):
        """Converts WLS polynomial weights into global sparse matrices."""
        row, col = edge_index
        M_inv_edges = M_inv[row]
        WV = (W.unsqueeze(1) * V).unsqueeze(2)
        C = torch.bmm(M_inv_edges, WV).squeeze(2)

        Cx = C[:, 0]
        Cy = C[:, 1]
        C_laplacian = C[:, 2] + C[:, 4]

        def build_sparse_matrix(edge_weights):
            off_diag_indices = edge_index
            off_diag_values = edge_weights
            diag_values = torch.zeros(num_nodes, dtype=torch.float32, device=C.device)
            diag_values.scatter_add_(0, row, -edge_weights)
            diag_indices = torch.arange(num_nodes, device=C.device).repeat(2, 1)

            indices = torch.cat([off_diag_indices, diag_indices], dim=1)
            values = torch.cat([off_diag_values, diag_values])
            return torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()

        return build_sparse_matrix(Cx), build_sparse_matrix(Cy), build_sparse_matrix(C_laplacian)

    def _compute_gradient_wls(self, f_node, row, col, W, V, M_inv, num_nodes):
        """Generic WLS gradient computer for any scalar field f_node."""
        df = f_node[col] - f_node[row]
        sum_W_V_df = torch.zeros((num_nodes, 5), dtype=torch.float32, device=f_node.device)
        integrand = W.unsqueeze(1) * V * df.unsqueeze(1)
        sum_W_V_df.scatter_add_(0, row.unsqueeze(1).expand(-1, 5), integrand)
        grad_f = torch.bmm(M_inv, sum_W_V_df.unsqueeze(2)).squeeze(2)
        return grad_f[:, :2]

    def _load_spatial_mask(self, file_path, tree, num_nodes, tolerance=1e-5):
        """Robust KDTree spatial mapping for boundary nodes."""
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        if not file_path.exists():
            print(f"Warning: Boundary file missing: {file_path}")
            return mask

        # Load coords, grab last two columns (x, y), and drop duplicates
        bnd_df = pd.read_csv(file_path, comment='%', sep=r'\s+', header=None)
        bnd_coords = bnd_df.iloc[:, -2:].values
        bnd_coords = np.unique(bnd_coords, axis=0)

        # Query KDTree
        distances, indices = tree.query(bnd_coords)
        valid_matches = indices[distances < tolerance]
        mask[valid_matches] = True
        return mask

    def _plot_sanity_check(self, data, stem):
        """Generates a PNG visualization of the mapped graph data to verify correctness."""
        x = data.x[:, 0].numpy()
        y = data.x[:, 1].numpy()

        u_vel = data.y[:, 0].numpy()
        thrombin = data.y[:, 9].numpy()
        sdf = data.x[:, 2].numpy()

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Sanity Check: {stem}", fontsize=16)

        sc1 = axs[0, 0].scatter(x, y, c=u_vel, cmap='viridis', s=2)
        axs[0, 0].set_title("Mapped Velocity (u)")
        fig.colorbar(sc1, ax=axs[0, 0])

        sc2 = axs[0, 1].scatter(x, y, c=thrombin, cmap='plasma', s=2)
        axs[0, 1].set_title("Mapped Thrombin (th)")
        fig.colorbar(sc2, ax=axs[0, 1])

        sc3 = axs[1, 0].scatter(x, y, c=sdf, cmap='coolwarm', s=2)
        axs[1, 0].set_title("Computed Wall Distance (SDF)")
        fig.colorbar(sc3, ax=axs[1, 0])

        axs[1, 1].scatter(x, y, c='gray', s=1, alpha=0.1, label='Internal')
        axs[1, 1].scatter(x[data.mask_wall], y[data.mask_wall], c='black', s=5, label='Wall')
        axs[1, 1].scatter(x[data.mask_inlet], y[data.mask_inlet], c='blue', s=5, label='Inlet')
        axs[1, 1].scatter(x[data.mask_outlet], y[data.mask_outlet], c='red', s=5, label='Outlet')
        axs[1, 1].set_title("Boundary Masks")
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.axis('equal')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.vis_dir / f"{stem}_sanity.png", dpi=150, bbox_inches='tight')
        plt.close()

    def process_patient(self, stem):
        # Automatically detect .nas (from COMSOL) or .msh (from Gmsh)
        msh_path_nas = self.raw_dir / f"{stem}.nas"
        msh_path_msh = self.raw_dir / f"{stem}.msh"

        if msh_path_nas.exists():
            msh_path = msh_path_nas
        elif msh_path_msh.exists():
            msh_path = msh_path_msh
        else:
            print(f"\n❌ SKIPPING {stem}: Mesh file not found.")
            print(f"   Looked for: {msh_path_nas} OR {msh_path_msh}")
            return

        json_path = self.raw_dir / f"{stem}.json"

        # Paths for spatial mapping
        txt_path = self.label_dir / f"{stem}.txt"
        inlet_path = self.label_dir / f"{stem}_inlet.txt"
        outlet_path = self.label_dir / f"{stem}_outlet.txt"
        wall_path = self.label_dir / f"{stem}_wall.txt"

        # Explicitly check for the main text file
        if not txt_path.exists():
            print(f"\n❌ SKIPPING {stem}: COMSOL domain data missing.")
            print(f"   Looked for: {txt_path}")
            return

        print(f"\nProcessing patient: {stem} (Using mesh: {msh_path.name})...")

        # 1. Topology & Graph Extraction
        # meshio natively understands .nas and .msh, no extra flags needed!
        mesh = meshio.read(msh_path)
        mesh_nodes = mesh.points[:, :2]
        num_nodes = len(mesh_nodes)

        d_bar = 0.005
        if json_path.exists():
            with open(json_path, 'r') as f:
                d_bar = json.load(f).get('d_bar', d_bar)

        u_ref = self.phys_cfg.get_u_ref(d_bar)

        # Build KDTree on the mesh nodes for spatial boundary mapping
        mesh_tree = cKDTree(mesh_nodes)

        # Load exactly mapped PyTorch boolean tensors using our new robust method
        mask_inlet = self._load_spatial_mask(inlet_path, mesh_tree, num_nodes)
        mask_outlet = self._load_spatial_mask(outlet_path, mesh_tree, num_nodes)
        mask_wall = self._load_spatial_mask(wall_path, mesh_tree, num_nodes)

        if "triangle" in mesh.cells_dict:
            all_tris = mesh.cells_dict["triangle"]
        elif "triangle6" in mesh.cells_dict:
            all_tris = mesh.cells_dict["triangle6"][:, :3]
        else:
            print(f"❌ Warning: No valid triangles found in {stem}'s mesh")
            return

        if len(all_tris) == 0: return
        edges = np.unique(np.sort(np.vstack([
            all_tris[:, [0, 1]], all_tris[:, [1, 2]], all_tris[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)
        row, col = edge_index

        # 2. Eulerian Mapping (1:1 Constraint)
        df_csv = pd.read_csv(txt_path, sep=r'\s+', comment='%', header=None)

        df_csv.columns = [
            'x_orig', 'y_orig', 'x', 'y', 'u', 'v', 'p', 'mu_effective',
            'rp', 'ap', 'apr', 'aps', 'PT', 'th', 'at', 'fg', 'fi', 'M', 'Mas', 'Mat'
        ]

        expected_cols = ['x', 'y', 'u', 'v', 'p', 'mu_effective'] + list(self.species_map.keys())
        df_csv = df_csv[expected_cols]

        csv_coords = df_csv[['x', 'y']].values

        # Re-using a KDTree here just to align the domain CSV data perfectly to the mesh indices
        domain_tree = cKDTree(csv_coords)
        dists, match_indices = domain_tree.query(mesh_nodes)
        if np.max(dists) > 1e-2:
            print(f"⚠️ Warning: Spatial mismatch detected in {stem}. Max err: {np.max(dists)}")

        df_matched = df_csv.iloc[match_indices].reset_index(drop=True)

        # 3. Geometric Features (SDF & Wall Normals)
        wall_node_indices = np.where(mask_wall.numpy())[0]
        if len(wall_node_indices) > 0:
            wall_pts = mesh_nodes[wall_node_indices]
            tree_wall = KDTree(wall_pts)
            dist_raw, indices_wall = tree_wall.query(mesh_nodes)
            nearest_wall_pts = wall_pts[indices_wall]
            diff_vec = mesh_nodes - nearest_wall_pts
            norms = np.linalg.norm(diff_vec, axis=1, keepdims=True)
            wall_normal_vec = torch.tensor(diff_vec / (norms + 1e-12), dtype=torch.float32)
        else:
            print(f"⚠️ CRITICAL WARNING: No wall nodes found for {stem}. SDF will be zeroed out.")
            dist_raw = np.zeros(num_nodes)
            wall_normal_vec = torch.zeros((num_nodes, 2), dtype=torch.float32)

        nodes_nd = torch.tensor(mesh_nodes / d_bar, dtype=torch.float32)
        sdf_nd = torch.clamp(torch.tensor(dist_raw / d_bar, dtype=torch.float32).view(-1, 1), min=1e-6)

        edge_attr = torch.cat([
            nodes_nd[row] - nodes_nd[col],
            torch.linalg.norm(nodes_nd[row] - nodes_nd[col], dim=1, keepdim=True)
        ], dim=1)

        # 4. Feature Engineering: WLS Shear Rate Gradient
        V, W, M_inv = self._precompute_wls(edge_index, num_nodes, nodes_nd)
        G_x, G_y, Laplacian = self._precompute_sparse_operators(edge_index, num_nodes, M_inv, V, W)

        u = torch.tensor(df_matched['u'].values, dtype=torch.float32)
        v = torch.tensor(df_matched['v'].values, dtype=torch.float32)

        grad_u = self._compute_gradient_wls(u, row, col, W, V, M_inv, num_nodes)
        grad_v = self._compute_gradient_wls(v, row, col, W, V, M_inv, num_nodes)

        du_dx, du_dy = grad_u[:, 0], grad_u[:, 1]
        dv_dx, dv_dy = grad_v[:, 0], grad_v[:, 1]

        dot_gamma = torch.sqrt(2.0 * (du_dx ** 2 + dv_dy ** 2 + 0.5 * (du_dy + dv_dx) ** 2))
        grad_dot_gamma = self._compute_gradient_wls(dot_gamma, row, col, W, V, M_inv, num_nodes)

        x_tensor = torch.cat([
            nodes_nd,
            sdf_nd,
            wall_normal_vec,
            grad_dot_gamma
        ], dim=1)

        # 5. Assemble Target Tensor (y)
        p_raw = torch.tensor(df_matched['p'].values, dtype=torch.float32)
        if mask_outlet.any():
            p_outlet_mean = p_raw[mask_outlet].mean()
        else:
            p_outlet_mean = p_raw.min()

        p_relative = p_raw - p_outlet_mean
        mu_eff = torch.tensor(df_matched['mu_effective'].values, dtype=torch.float32)

        species_cols = list(self.species_map.keys())
        species = torch.tensor(df_matched[species_cols].values, dtype=torch.float32)

        y_tensor = torch.cat([
            u.unsqueeze(1),
            v.unsqueeze(1),
            p_relative.unsqueeze(1),
            mu_eff.unsqueeze(1),
            species
        ], dim=1)

        # 6. PyG Data Construction
        data = Data(
            x=x_tensor,
            y=y_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            is_anchor=torch.tensor([True], dtype=torch.bool),
            d_bar=torch.tensor([d_bar], dtype=torch.float32),
            u_ref=torch.tensor([u_ref], dtype=torch.float32),
            G_x=G_x,
            G_y=G_y,
            Laplacian=Laplacian,
            V=V,
            W=W,
            M_inv=M_inv
        )

        torch.save(data, self.proc_dir / f"{stem}.pt")

        if self.visualize:
            self._plot_sanity_check(data, stem)

    def run(self):
        # Look for the .nas files now!
        files = [f for f in os.listdir(self.raw_dir) if f.endswith(".nas") or f.endswith(".msh")]

        if len(files) == 0:
            print(f"CRITICAL ERROR: No .msh or .nas files found in {self.raw_dir}")
            return

        stems = sorted(list(set([Path(f).stem for f in files])))

        for stem in tqdm(stems, desc="Extracting Tier 3 Patient Data"):
            self.process_patient(stem)

if __name__ == "__main__":
    extractor = PatientDataExtractor(tier="tier3_patients", visualize=True)
    extractor.run()