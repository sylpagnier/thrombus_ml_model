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

from src.config import VesselConfig, PhysicsConfig, BiochemConfig
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
        """Generates a PNG visualization with updated Log1p scaling labels."""
        x = data.x[:, 0].numpy()
        y = data.x[:, 1].numpy()

        u_vel = data.y[:, 0].numpy()
        p_rel = data.y[:, 2].numpy()
        mu_eff = data.y[:, 3].numpy()
        # Index 9 in y_tensor (4 kinematics + 5th species) is Thrombin
        thrombin_transformed = data.y[:, 9].numpy()
        sdf = data.x[:, 2].numpy()

        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f"Sanity Check: {stem}", fontsize=18, fontweight='bold')

        # 1. Mapped Velocity
        sc1 = axs[0, 0].scatter(x, y, c=u_vel, cmap='viridis', s=2)
        axs[0,0].set_title(r"Normalized Velocity ($u / u_{ref}$)")
        fig.colorbar(sc1, ax=axs[0,0], label='ND')

        # 2. Relative Pressure
        sc2 = axs[0, 1].scatter(x, y, c=p_rel, cmap='RdBu_r', s=2)
        axs[0, 1].set_title("Non-Dimensional Pressure (Relative)")
        fig.colorbar(sc2, ax=axs[0, 1], label='ND (p / p_ref)')

        # 3. Effective Viscosity (The new plot)
        sc3 = axs[1, 0].scatter(x, y, c=mu_eff, cmap='magma', s=2)
        axs[1, 0].set_title(r"Non-Dimensional Viscosity ($\mu_{eff} / \mu_{ref}$)")
        fig.colorbar(sc3, ax=axs[1, 0], label='ND Ratio'),

        # 4. Thrombin Concentration (Updated Labels)
        sc4 = axs[1, 1].scatter(x, y, c=thrombin_transformed, cmap='plasma', s=2)
        axs[1, 1].set_title(r"Thrombin $\ln(1 + \hat{T})$")
        fig.colorbar(sc4, ax=axs[1, 1], label='Transformed ND Units')

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

        txt_path = self.label_dir / f"{stem}.txt"
        inlet_path = self.label_dir / f"{stem}_inlet.txt"
        outlet_path = self.label_dir / f"{stem}_outlet.txt"
        wall_path = self.label_dir / f"{stem}_wall.txt"

        if not txt_path.exists():
            print(f"❌ Skipping {stem}: COMSOL domain data (.txt) missing.")
            return

        # 2. Topology & Enhanced Boundary Mapping
        mesh = meshio.read(msh_path)
        mesh_nodes = mesh.points[:, :2] * 0.01 # cm --> m
        num_nodes = len(mesh_nodes)
        mesh_tree = cKDTree(mesh_nodes)

        mask_inlet, inlet_fail = self._load_spatial_mask(inlet_path, mesh_tree, num_nodes)
        mask_outlet, outlet_fail = self._load_spatial_mask(outlet_path, mesh_tree, num_nodes)
        mask_wall, wall_fail = self._load_spatial_mask(wall_path, mesh_tree, num_nodes)

        # 3. AUTO-DETECT SCALE (d_bar) FROM INLET BOUNDARY
        inlet_coords = mesh_nodes[mask_inlet.numpy()]
        if len(inlet_coords) > 1:
            # Calculate max distance between any two nodes on the inlet
            d_bar = float(np.max(np.linalg.norm(inlet_coords[:, None] - inlet_coords, axis=-1)))
        else:
            d_bar = 0.0198  # Fallback to your COMSOL D_eff if mapping fails

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
        df_csv.columns = [
            'x_orig', 'y_orig', 'x', 'y', 'u', 'v', 'p', 'mu_effective',
            'rp', 'ap', 'apr', 'aps', 'PT', 'th', 'at', 'fg', 'fi', 'M', 'Mas', 'Mat'
        ]
        csv_coords = df_csv[['x', 'y']].values * 0.01 # cm --> m
        domain_tree = cKDTree(csv_coords)
        _, match_indices = domain_tree.query(mesh_nodes)
        df_matched = df_csv.iloc[match_indices].reset_index(drop=True)

        nodes_nd = torch.tensor(mesh_nodes / d_bar, dtype=torch.float32)

        edge_attr = torch.cat([
            nodes_nd[row] - nodes_nd[col],
            torch.linalg.norm(nodes_nd[row] - nodes_nd[col], dim=1, keepdim=True)
        ], dim=1)

        # 6. Gradients & Numerical Stability
        V, W, M_inv, max_cond = self._precompute_wls(edge_index, num_nodes, nodes_nd)
        G_x, G_y, Laplacian = self._precompute_sparse_operators(edge_index, num_nodes, M_inv, V, W)

        # 7. Mass Flux Calculation
        u_raw = torch.tensor(df_matched['u'].values, dtype=torch.float32) * 0.01
        v_raw = torch.tensor(df_matched['v'].values, dtype=torch.float32) * 0.01
        p_raw = torch.tensor(df_matched['p'].values, dtype=torch.float32)
        mu_eff = torch.tensor(df_matched['mu_effective'].values, dtype=torch.float32)

        inlet_flux = torch.sqrt(u_raw[mask_inlet] ** 2 + v_raw[mask_inlet] ** 2).sum().item()
        outlet_flux = torch.sqrt(u_raw[mask_outlet] ** 2 + v_raw[mask_outlet] ** 2).sum().item()
        flux_imbalance = abs(inlet_flux - outlet_flux) / (inlet_flux + 1e-8)

        # 8. Data-Driven ML Scaling
        # Compute U_ref from the 99th percentile to clamp data strictly to ~[-1.0, 1.0]
        velocity_mag = torch.sqrt(u_raw ** 2 + v_raw ** 2)
        u_ref_actual = torch.quantile(velocity_mag, 0.99).item()

        # Calculate the dynamic ML Reynolds number
        re_actual = self.phys_cfg.get_re(u_ref_actual, d_bar)

        # Pressure scaling based on the new ML velocity reference
        p_ref = self.phys_cfg.rho * (u_ref_actual ** 2)
        p_relative = p_raw - (p_raw[mask_outlet].mean() if mask_outlet.any() else p_raw.min())

        u_nd = u_raw / u_ref_actual
        v_nd = v_raw / u_ref_actual
        p_nd = p_relative / p_ref
        mu_nd = mu_eff / self.phys_cfg.mu_ref

        # Species Non-dimensionalization
        species_cols = list(self.species_map.keys())
        species = torch.tensor(df_matched[species_cols].values, dtype=torch.float32)
        species = torch.clamp(species, min=0.0)

        bio_cfg = BiochemConfig(tier=self.vessel_cfg.tier)
        scales = torch.tensor([
            bio_cfg.c_RP0, bio_cfg.c_RP0, bio_cfg.APRcrit, bio_cfg.APScrit,
            bio_cfg.c_pT0, bio_cfg.c_pT0, bio_cfg.cAT0, bio_cfg.c_Fg0,
            bio_cfg.c_Fg0, bio_cfg.Minf, bio_cfg.Minf, bio_cfg.Minf
        ], dtype=torch.float32)

        species_nd = species / scales
        species_transformed = torch.log1p(species_nd)

        y_tensor = torch.cat([
            u_nd.unsqueeze(1), v_nd.unsqueeze(1), p_nd.unsqueeze(1),
            mu_nd.unsqueeze(1), species_transformed
        ], dim=1)

        # 9. SDF and Normals
        wall_coords = mesh_nodes[mask_wall.numpy()]
        if len(wall_coords) > 0:
            wall_tree = KDTree(wall_coords)
            dist, idx = wall_tree.query(mesh_nodes)
            sdf = torch.tensor(dist / d_bar, dtype=torch.float32).unsqueeze(1)
            normals = mesh_nodes - wall_coords[idx]
            normals_unit = torch.tensor(normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9),
                                        dtype=torch.float32)
        else:
            sdf = torch.zeros((num_nodes, 1))
            normals_unit = torch.zeros((num_nodes, 2))

        x_tensor = torch.cat([nodes_nd, sdf, normals_unit], dim=1)

        # 10. Metadata Export
        metadata = {
            "stem": stem,
            "quality": {
                "max_wls_condition_number": max_cond,
                "mass_flux_imbalance": flux_imbalance,
                "boundary_unmapped_ratio": max(inlet_fail, outlet_fail, wall_fail)
            },
            "field_stats": {
                "u_max": u_raw.max().item(),
                "u_ref_ml": u_ref_actual,
                "re_ml": re_actual,
                "d_bar": d_bar
            }
        }
        with open(self.proc_dir / f"{stem}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # 11. Final PyG Data Save
        data = Data(
            x=x_tensor, y=y_tensor, edge_index=edge_index, edge_attr=edge_attr,
            mask_inlet=mask_inlet, mask_outlet=mask_outlet, mask_wall=mask_wall,
            is_anchor=torch.tensor([True], dtype=torch.bool),
            d_bar=torch.tensor([d_bar], dtype=torch.float32),
            u_ref=torch.tensor([u_ref_actual], dtype=torch.float32),
            re_actual=torch.tensor([re_actual], dtype=torch.float32),
            G_x=G_x, G_y=G_y, Laplacian=Laplacian, V=V, W=W, M_inv=M_inv,
            u_inlet_bc=u_nd.unsqueeze(1), mu_inlet_bc=mu_nd.unsqueeze(1), mu_wall_bc=mu_nd.unsqueeze(1)
        )

        torch.save(data, self.proc_dir / f"{stem}.pt")
        print(f"✅ Saved {stem}: D={d_bar * 1000:.1f}mm | Re_ML={re_actual:.0f} | Imbal={flux_imbalance:.2%}")

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