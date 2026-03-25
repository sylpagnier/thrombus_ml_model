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
        """Computes the 2nd Order Polynomial Basis, WLS Inverse, and Condition Number."""
        row, col = edge_index
        pos_diff = pos_tensor[col, :2] - pos_tensor[row, :2]
        dx, dy = pos_diff[:, 0], pos_diff[:, 1]
        dist_sq = dx ** 2 + dy ** 2 + 1e-8

        V = torch.stack([dx, dy, 0.5 * dx ** 2, dx * dy, 0.5 * dy ** 2], dim=1)
        W = 1.0 / dist_sq

        V_unsqueezed = V.unsqueeze(2)
        V_T_unsqueezed = V.unsqueeze(1)
        M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)

        M_e_flat = M_e.view(-1, 25)
        out = torch.zeros((num_nodes, 25), dtype=M_e_flat.dtype, device=M_e_flat.device)
        M_flat = out.scatter_add_(0, row.view(-1, 1).expand_as(M_e_flat), M_e_flat)

        M = M_flat.view(num_nodes, 5, 5)
        epsilon = 1e-6
        I = torch.eye(5, device=M.device).unsqueeze(0).expand(num_nodes, 5, 5)
        M_reg = M + epsilon * I

        # --- NEW: Compute Max Condition Number for Stability Check ---
        cond_numbers = torch.linalg.cond(M_reg)
        max_cond = cond_numbers.max().item()

        M_inv = torch.linalg.pinv(M_reg)
        return V, W, M_inv.squeeze(1), max_cond

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
        """Robust KDTree mapping with unmapped coordinate reporting."""
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        unmapped_ratio = 0.0

        if not file_path.exists():
            return mask, 1.0  # 100% missing

        bnd_df = pd.read_csv(file_path, comment='%', sep=r'\s+', header=None)
        bnd_coords = np.unique(bnd_df.iloc[:, -2:].values, axis=0)

        distances, indices = tree.query(bnd_coords)
        valid_matches = indices[distances < tolerance]

        mask[valid_matches] = True
        unmapped_ratio = 1.0 - (len(np.unique(valid_matches)) / len(bnd_coords))

        return mask, unmapped_ratio

    def _plot_sanity_check(self, data, stem):
        """Generates a PNG visualization including Velocity, Thrombin, Viscosity, and Geometry."""
        x = data.x[:, 0].numpy()
        y = data.x[:, 1].numpy()

        # Extract fields from y_tensor indices
        u_vel = data.y[:, 0].numpy()      # Index 0: Velocity U
        p_rel = data.y[:, 2].numpy()      # Index 2: Pressure
        mu_eff = data.y[:, 3].numpy()     # Index 3: Effective Viscosity
        thrombin = data.y[:, 9].numpy()   # Index 9: Thrombin (log-scale)
        sdf = data.x[:, 2].numpy()        # x index 2: Signed Distance Function

        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f"Sanity Check: {stem}", fontsize=18, fontweight='bold')

        # 1. Mapped Velocity
        sc1 = axs[0, 0].scatter(x, y, c=u_vel, cmap='viridis', s=2)
        axs[0, 0].set_title("Ground Truth Velocity (u)")
        fig.colorbar(sc1, ax=axs[0, 0], label='m/s')

        # 2. Relative Pressure
        sc2 = axs[0, 1].scatter(x, y, c=p_rel, cmap='RdBu_r', s=2)
        axs[0, 1].set_title("Ground Truth Pressure (Relative)")
        fig.colorbar(sc2, ax=axs[0, 1], label='Pa')

        # 3. Effective Viscosity (The new plot)
        sc3 = axs[1, 0].scatter(x, y, c=mu_eff, cmap='magma', s=2)
        axs[1, 0].set_title("Effective Viscosity ($\mu_{eff}$)")
        fig.colorbar(sc3, ax=axs[1, 0], label='Pa·s')

        # 4. Thrombin Concentration
        sc4 = axs[1, 1].scatter(x, y, c=thrombin, cmap='plasma', s=2)
        axs[1, 1].set_title("Thrombin (Log Concentration)")
        fig.colorbar(sc4, ax=axs[1, 1], label='log(M)')

        # 5. Wall Distance (SDF)
        sc5 = axs[2, 0].scatter(x, y, c=sdf, cmap='coolwarm', s=2)
        axs[2, 0].set_title("Wall Distance (SDF)")
        fig.colorbar(sc5, ax=axs[2, 0])

        # 6. Boundary Masks
        axs[2, 1].scatter(x, y, c='gray', s=1, alpha=0.05, label='Internal')
        axs[2, 1].scatter(x[data.mask_wall], y[data.mask_wall], c='black', s=5, label='Wall')
        axs[2, 1].scatter(x[data.mask_inlet], y[data.mask_inlet], c='blue', s=8, label='Inlet')
        axs[2, 1].scatter(x[data.mask_outlet], y[data.mask_outlet], c='red', s=8, label='Outlet')
        axs[2, 1].set_title("Boundary Node Verification")
        axs[2, 1].legend(loc='upper right')

        # Formatting
        for ax in axs.flat:
            ax.axis('equal')
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.vis_dir / f"{stem}_sanity.png", dpi=200, bbox_inches='tight')
        plt.close()

    def process_patient(self, stem):
        """
        Full extraction pipeline with Physics-Informed Sanity Checks and
        Training Metadata generation.
        """
        # 1. Path Setup and Mesh Loading
        msh_path_nas = self.raw_dir / f"{stem}.nas"
        msh_path_msh = self.raw_dir / f"{stem}.msh"
        msh_path = msh_path_nas if msh_path_nas.exists() else msh_path_msh

        if not msh_path.exists():
            print(f"❌ Skipping {stem}: Mesh file (.nas/.msh) not found.")
            return

        json_path = self.raw_dir / f"{stem}.json"
        txt_path = self.label_dir / f"{stem}.txt"
        inlet_path = self.label_dir / f"{stem}_inlet.txt"
        outlet_path = self.label_dir / f"{stem}_outlet.txt"
        wall_path = self.label_dir / f"{stem}_wall.txt"

        if not txt_path.exists():
            print(f"❌ Skipping {stem}: COMSOL domain data (.txt) missing.")
            return

        # 2. Topology & Enhanced Boundary Mapping
        mesh = meshio.read(msh_path)
        mesh_nodes = mesh.points[:, :2]  # Extract 2D coords
        num_nodes = len(mesh_nodes)
        mesh_tree = cKDTree(mesh_nodes)

        # Get masks and mapping quality metrics (Requires updated _load_spatial_mask)
        mask_inlet, inlet_fail = self._load_spatial_mask(inlet_path, mesh_tree, num_nodes)
        mask_outlet, outlet_fail = self._load_spatial_mask(outlet_path, mesh_tree, num_nodes)
        mask_wall, wall_fail = self._load_spatial_mask(wall_path, mesh_tree, num_nodes)

        # 3. Scale Detection (d_bar)
        d_bar = 0.005  # Default 5mm
        if json_path.exists():
            with open(json_path, 'r') as f:
                d_bar = json.load(f).get('d_bar', d_bar)
        u_ref = self.phys_cfg.get_u_ref(d_bar)

        # 4. Connectivity and Edge Construction
        if "triangle" in mesh.cells_dict:
            all_tris = mesh.cells_dict["triangle"]
        elif "triangle6" in mesh.cells_dict:
            all_tris = mesh.cells_dict["triangle6"][:, :3]
        else:
            print(f"⚠️ {stem}: Unsupported cell type.")
            return

        edges = np.unique(np.sort(np.vstack([
            all_tris[:, [0, 1]], all_tris[:, [1, 2]], all_tris[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)
        row, col = edge_index

        # 5. Eulerian Field Mapping
        df_csv = pd.read_csv(txt_path, sep=r'\s+', comment='%', header=None)
        # Assuming the standard 20-column layout from your documentation
        df_csv.columns = [
            'x_orig', 'y_orig', 'x', 'y', 'u', 'v', 'p', 'mu_effective',
            'rp', 'ap', 'apr', 'aps', 'PT', 'th', 'at', 'fg', 'fi', 'M', 'Mas', 'Mat'
        ]
        csv_coords = df_csv[['x', 'y']].values
        domain_tree = cKDTree(csv_coords)
        _, match_indices = domain_tree.query(mesh_nodes)
        df_matched = df_csv.iloc[match_indices].reset_index(drop=True)

        nodes_nd = torch.tensor(mesh_nodes / d_bar, dtype=torch.float32)

        # 6. Gradients & Numerical Stability (Requires updated _precompute_wls)
        V, W, M_inv, max_cond = self._precompute_wls(edge_index, num_nodes, nodes_nd)
        G_x, G_y, Laplacian = self._precompute_sparse_operators(edge_index, num_nodes, M_inv, V, W)

        # 7. Mass Flux Proxy Calculation
        u = torch.tensor(df_matched['u'].values, dtype=torch.float32)
        v = torch.tensor(df_matched['v'].values, dtype=torch.float32)

        # We estimate flux by summing velocity magnitude on boundary nodes
        inlet_flux = torch.sqrt(u[mask_inlet] ** 2 + v[mask_inlet] ** 2).sum().item()
        outlet_flux = torch.sqrt(u[mask_outlet] ** 2 + v[mask_outlet] ** 2).sum().item()
        flux_imbalance = abs(inlet_flux - outlet_flux) / (inlet_flux + 1e-8)

        # 8. Feature Engineering (Pressure & Species)
        p_raw = torch.tensor(df_matched['p'].values, dtype=torch.float32)
        # Relative pressure (zeroed at outlet)
        p_relative = p_raw - (p_raw[mask_outlet].mean() if mask_outlet.any() else p_raw.min())

        mu_eff = torch.tensor(df_matched['mu_effective'].values, dtype=torch.float32)

        species_cols = list(self.species_map.keys())
        species = torch.tensor(df_matched[species_cols].values, dtype=torch.float32)
        species_log = torch.log(torch.clamp(species, min=1e-8))

        y_tensor = torch.cat([
            u.unsqueeze(1), v.unsqueeze(1), p_relative.unsqueeze(1),
            mu_eff.unsqueeze(1), species_log
        ], dim=1)

        # 9. SDF and Normals (The "Wall Knowledge")
        wall_coords = mesh_nodes[mask_wall.numpy()]
        if len(wall_coords) > 0:
            wall_tree = KDTree(wall_coords)
            dist, idx = wall_tree.query(mesh_nodes)
            sdf = torch.tensor(dist / d_bar, dtype=torch.float32).unsqueeze(1)

            # Normal vector calculation (pointing into domain)
            nearest_wall_points = wall_coords[idx]
            normals = mesh_nodes - nearest_wall_points
            norm_mag = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
            normals_unit = torch.tensor(normals / norm_mag, dtype=torch.float32)
        else:
            sdf = torch.zeros((num_nodes, 1))
            normals_unit = torch.zeros((num_nodes, 2))

        x_tensor = torch.cat([nodes_nd, sdf, normals_unit], dim=1)

        # 10. Metadata Export (Improved with Flux interpretation)
        metadata = {
            "stem": stem,
            "quality": {
                "max_wls_condition_number": max_cond,
                "mass_flux_magnitude_imbalance": flux_imbalance,  # Treat as "Scale Check"
                "boundary_unmapped_ratio": max(inlet_fail, outlet_fail, wall_fail)
            },
            "field_stats": {
                "u_max": u.max().item(),
                "p_range": [p_relative.min().item(), p_relative.max().item()],
                "mean_log_thrombin": species_log[:, 5].mean().item()
            }
        }
        with open(self.proc_dir / f"{stem}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # 11. Final PyG Data Save (REPAIRED FOR TRAINING COMPATIBILITY)
        # We add 'is_anchor' for the StratifiedAnchorSampler
        # We add 'mu_wall_bc' for the wall loss in physics_kernels
        data = Data(
            x=x_tensor,
            y=y_tensor,
            edge_index=edge_index,
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            is_anchor=torch.tensor([True], dtype=torch.bool),  # Required by Sampler
            d_bar=torch.tensor([d_bar], dtype=torch.float32),
            u_ref=torch.tensor([u_ref], dtype=torch.float32),
            G_x=G_x,
            G_y=G_y,
            Laplacian=Laplacian,
            V=V,  # Kept for potential on-the-fly gradients
            W=W,  # Kept for potential on-the-fly gradients
            M_inv=M_inv,
            u_inlet_bc=u.unsqueeze(1),
            mu_inlet_bc=mu_eff.unsqueeze(1),
            mu_wall_bc=mu_eff.unsqueeze(1)  # Required by BiochemPhysicsKernels
        )

        torch.save(data, self.proc_dir / f"{stem}.pt")
        print(f"✅ Saved {stem}: Imbalance: {flux_imbalance:.2%}, Max Cond: {max_cond:.1e}")

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