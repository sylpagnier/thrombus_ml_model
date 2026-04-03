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
import glob
import re
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

        self.csv_fields = [ 'x', 'y', 'u', 'v', 'p', 'mu_effective' ] + list(self.species_map.keys())

    def _precompute_wls(self, edge_index, num_nodes, pos_tensor):
        """Computes the 2nd Order Polynomial Basis, WLS Inverse, and Condition Number."""
        row, col = edge_index
        pos_diff = pos_tensor[ col, :2 ] - pos_tensor[ row, :2 ]
        dx, dy = pos_diff[ :, 0 ], pos_diff[ :, 1 ]
        dist_sq = dx ** 2 + dy ** 2 + 1e-8

        V = torch.stack([ dx, dy, 0.5 * dx ** 2, dx * dy, 0.5 * dy ** 2 ], dim=1)
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
        M_inv_edges = M_inv[ row ]
        WV = (W.unsqueeze(1) * V).unsqueeze(2)
        C = torch.bmm(M_inv_edges, WV).squeeze(2)

        Cx = C[ :, 0 ]
        Cy = C[ :, 1 ]
        C_laplacian = C[ :, 2 ] + C[ :, 4 ]

        def build_sparse_matrix(edge_weights):
            off_diag_indices = edge_index
            off_diag_values = edge_weights
            diag_values = torch.zeros(num_nodes, dtype=torch.float32, device=C.device)
            diag_values.scatter_add_(0, row, -edge_weights)
            diag_indices = torch.arange(num_nodes, device=C.device).repeat(2, 1)

            indices = torch.cat([ off_diag_indices, diag_indices ], dim=1)
            values = torch.cat([ off_diag_values, diag_values ])
            return torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()

        return build_sparse_matrix(Cx), build_sparse_matrix(Cy), build_sparse_matrix(C_laplacian)

    def _compute_gradient_wls(self, f_node, row, col, W, V, M_inv, num_nodes):
        """Generic WLS gradient computer for any scalar field f_node."""
        df = f_node[ col ] - f_node[ row ]
        sum_W_V_df = torch.zeros((num_nodes, 5), dtype=torch.float32, device=f_node.device)
        integrand = W.unsqueeze(1) * V * df.unsqueeze(1)
        sum_W_V_df.scatter_add_(0, row.unsqueeze(1).expand(-1, 5), integrand)
        grad_f = torch.bmm(M_inv, sum_W_V_df.unsqueeze(2)).squeeze(2)
        return grad_f[ :, :2 ]

    def _compute_boundary_normals(self, edge_index, boundary_mask, pos_tensor, num_nodes):
        """Computes geometric unit normals for any boundary mask using adjacent edges."""
        normals = torch.zeros((num_nodes, 2), dtype=torch.float32, device=pos_tensor.device)

        row, col = edge_index
        # Find edges where BOTH nodes are on the requested boundary
        b_edges = boundary_mask[ row ] & boundary_mask[ col ]

        r = row[ b_edges ]
        c = col[ b_edges ]

        # Compute edge vectors (dx, dy)
        edge_vecs = pos_tensor[ c ] - pos_tensor[ r ]

        # Perpendicular vector (-dy, dx)
        edge_normals = torch.stack([ -edge_vecs[ :, 1 ], edge_vecs[ :, 0 ] ], dim=1)

        # Accumulate normals at the respective nodes
        normals.scatter_add_(0, r.unsqueeze(1).expand(-1, 2), edge_normals)
        normals.scatter_add_(0, c.unsqueeze(1).expand(-1, 2), edge_normals)

        # Normalize to create unit vectors
        norm_mag = torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-9
        normals_unit = normals / norm_mag

        return normals_unit

    def _load_spatial_mask(self, file_path, tree, num_nodes, tolerance=1e-5):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        unmapped_ratio = 0.0

        if not file_path.exists():
            return mask, 1.0

        bnd_df = pd.read_csv(file_path, comment='%', sep=r'\s+', header=None)

        # USE CENTRALIZED SCALE
        bnd_coords = np.unique(bnd_df.iloc[ :, -2: ].values, axis=0) * self.phys_cfg.cm_to_m

        distances, indices = tree.query(bnd_coords)
        valid_matches = indices[ distances < tolerance ]

        if len(valid_matches) == 0 and len(bnd_coords) > 0:
            raise ValueError(
                f"\nCRITICAL ERROR: Zero boundary nodes mapped for {file_path.name}!\n"
                f"Attempted to map {len(bnd_coords)} nodes but none fell within the {tolerance} tolerance.\n"
                f"Please verify that your COMSOL boundary exports are in the same spatial units (cm) as your domain export."
            )

        mask[ valid_matches ] = True
        unmapped_ratio = 1.0 - (len(np.unique(valid_matches)) / len(bnd_coords))

        # Optional: Print a warning if the mask only partially maps
        if unmapped_ratio > 0.10:
            print(f"⚠️ Warning: {unmapped_ratio:.1%} of boundary nodes in {file_path.name} failed to map.")

        return mask, unmapped_ratio

    def load_comsol_trajectory(self, filepath):
        """Parses a single 'wide-format' COMSOL Spreadsheet export."""

        # 1. Read the header to extract the time steps dynamically
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header_line = ""
        for line in lines:
            if line.startswith('% x') and '@ t=' in line:
                header_line = line
                break

        if not header_line:
            raise ValueError(f"Could not find time-step header in {filepath.name}")

        import re
        # Find all unique time values in the header
        times = []
        for match in re.finditer(r't=([0-9.]+)', header_line):
            t_val = float(match.group(1))
            if t_val not in times:
                times.append(t_val)

        # 2. Load the numeric data (skipping comment lines)
        df_full = pd.read_csv(filepath, comment='%', sep=r'\s+', header=None)

        # 3. Slice the wide dataframe into time blocks
        time_blocks = {}
        vars_per_step = 18  # x, y, u, v, p, mu, + 12 species

        for i, t_val in enumerate(times):
            # Base coords take columns. Step 0 starts at col 2.
            start_col = 2 + (i * vars_per_step)
            end_col = start_col + vars_per_step

            # Extract the block
            df_step = df_full.iloc[ :, start_col:end_col ].copy()

            # Assign consistent internal column names
            df_step.columns = [
                'x', 'y', 'u', 'v', 'p', 'mu_effective',
                'rp', 'ap', 'apr', 'aps', 'PT', 'th', 'at', 'fg', 'fi', 'M', 'Mas', 'Mat'
            ]
            time_blocks[ t_val ] = df_step

        return time_blocks

    def _plot_sanity_check(self, data, stem):
        """Generates a PNG visualization with updated Log1p scaling labels."""
        x = data.x[ :, 0 ].numpy()
        y = data.x[ :, 1 ].numpy()

        # FIX: Slice the LAST time step [-1] from the [Time, Nodes, Features] tensor
        p_rel = data.y[ -1, :, 2 ].numpy()
        mu_eff = data.y[ -1, :, 3 ].numpy()
        # Index 9 in y_tensor (4 kinematics + 5th species) is Thrombin
        thrombin_transformed = data.y[ -1, :, 9 ].numpy()
        sdf = data.x[ :, 2 ].numpy()

        fig, axs = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f"Sanity Check (Final Time Step): {stem}", fontsize=18, fontweight='bold')

        # 1. Mapped Velocity
        # FIX: Slice the LAST time step [-1] for velocity components
        u_comp = data.y[ -1, :, 0 ].numpy()
        v_comp = data.y[ -1, :, 1 ].numpy()
        vel_mag = np.sqrt(u_comp ** 2 + v_comp ** 2)

        sc1 = axs[ 0, 0 ].scatter(x, y, c=vel_mag, cmap='viridis', s=2)
        axs[ 0, 0 ].set_title(r"Normalized Velocity ($|U| / u_{ref}$)")
        fig.colorbar(sc1, ax=axs[ 0, 0 ], label='ND')

        # 2. Relative Pressure
        sc2 = axs[ 0, 1 ].scatter(x, y, c=p_rel, cmap='RdBu_r', s=2)
        axs[ 0, 1 ].set_title("Non-Dimensional Pressure (Relative)")
        fig.colorbar(sc2, ax=axs[ 0, 1 ], label='ND (p / p_ref)')

        # 3. Effective Viscosity
        sc3 = axs[ 1, 0 ].scatter(x, y, c=mu_eff, cmap='magma', s=2)
        axs[ 1, 0 ].set_title(r"Non-Dimensional Viscosity ($\mu_{eff} / \mu_{ref}$)")
        fig.colorbar(sc3, ax=axs[ 1, 0 ], label='ND Ratio')

        # 4. Thrombin Concentration
        sc4 = axs[ 1, 1 ].scatter(x, y, c=thrombin_transformed, cmap='plasma', s=2)
        axs[ 1, 1 ].set_title(r"Thrombin $\ln(1 + \hat{T})$")
        fig.colorbar(sc4, ax=axs[ 1, 1 ], label='Transformed ND Units')

        # 5. Wall Distance (SDF)
        sc5 = axs[ 2, 0 ].scatter(x, y, c=sdf, cmap='coolwarm', s=2)
        axs[ 2, 0 ].set_title("Wall Distance (SDF)")
        fig.colorbar(sc5, ax=axs[ 2, 0 ])

        # 6. Boundary Masks
        axs[ 2, 1 ].scatter(x, y, c='gray', s=1, alpha=0.05, label='Internal')
        axs[ 2, 1 ].scatter(x[ data.mask_wall ], y[ data.mask_wall ], c='black', s=5, label='Wall')
        axs[ 2, 1 ].scatter(x[ data.mask_inlet ], y[ data.mask_inlet ], c='blue', s=8, label='Inlet')
        axs[ 2, 1 ].scatter(x[ data.mask_outlet ], y[ data.mask_outlet ], c='red', s=8, label='Outlet')
        axs[ 2, 1 ].set_title("Boundary Node Verification")
        axs[ 2, 1 ].legend(loc='upper right')

        # Formatting
        for ax in axs.flat:
            ax.axis('equal')
            ax.axis('off')

        plt.tight_layout(rect=[ 0, 0.03, 1, 0.95 ])
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
        mesh_nodes = mesh.points[ :, :2 ] * self.phys_cfg.cm_to_m
        num_nodes = len(mesh_nodes)
        mesh_tree = cKDTree(mesh_nodes)

        mask_inlet, inlet_fail = self._load_spatial_mask(inlet_path, mesh_tree, num_nodes)
        mask_outlet, outlet_fail = self._load_spatial_mask(outlet_path, mesh_tree, num_nodes)
        mask_wall, wall_fail = self._load_spatial_mask(wall_path, mesh_tree, num_nodes)

        # 3. AUTO-DETECT SCALE (d_bar) FROM INLET BOUNDARY
        inlet_coords = mesh_nodes[ mask_inlet.numpy() ]
        if len(inlet_coords) > 1:
            # Calculate max distance between any two nodes on the inlet
            d_bar = float(np.max(np.linalg.norm(inlet_coords[ :, None ] - inlet_coords, axis=-1)))
        else:
            d_bar = 0.0198  # Fallback to your COMSOL D_eff if mapping fails

        # 4. Connectivity and Edge Construction
        if "triangle" in mesh.cells_dict:
            all_tris = mesh.cells_dict[ "triangle" ]
        elif "triangle6" in mesh.cells_dict:
            all_tris = mesh.cells_dict[ "triangle6" ][ :, :3 ]
        else:
            print(f"⚠️ {stem}: Unsupported cell type.")
            return

        edges = np.unique(np.sort(np.vstack([
            all_tris[ :, [ 0, 1 ] ], all_tris[ :, [ 1, 2 ] ], all_tris[ :, [ 2, 0 ] ]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([ edges.T, edges[ :, [ 1, 0 ] ].T ]), dtype=torch.long)
        row, col = edge_index

        # --- 5. DYNAMIC EULERIAN FIELD MAPPING (TRAJECTORY EXTRACTION) ---
        trajectory_file = self.label_dir / f"{stem}.txt"

        if not trajectory_file.exists():
            print(f"❌ Skipping {stem}: Trajectory file not found.")
            return

        print(f"Parsing transient trajectory for {stem}...")
        time_blocks = self.load_comsol_trajectory(trajectory_file)

        eval_times = sorted(list(time_blocks.keys()))
        eval_times_tensor = torch.tensor(eval_times, dtype=torch.float32)

        # Pre-compute WLS operators once based on geometry
        nodes_nd = torch.tensor(mesh_nodes / d_bar, dtype=torch.float32)
        edge_attr = torch.cat([
            nodes_nd[ row ] - nodes_nd[ col ],
            torch.linalg.norm(nodes_nd[ row ] - nodes_nd[ col ], dim=1, keepdim=True)
        ], dim=1)

        V, W, M_inv, max_cond = self._precompute_wls(edge_index, num_nodes, nodes_nd)
        G_x, G_y, Laplacian = self._precompute_sparse_operators(edge_index, num_nodes, M_inv, V, W)

        y_trajectory = []
        u_raw_list = []

        # --- Pre-Compute KDTree Mapping ONCE outside the loop ---
        # Load just the first timestep block to establish the spatial mapping
        df_first = time_blocks[ eval_times[ 0 ] ]
        csv_coords_static = df_first[ [ 'x', 'y' ] ].values * self.phys_cfg.cm_to_m

        domain_tree = cKDTree(csv_coords_static)
        _, match_indices = domain_tree.query(mesh_nodes)

        # Iterate through the parsed time steps
        for t_idx, t_val in enumerate(eval_times):
            df_csv = time_blocks[t_val]
            df_matched = df_csv.iloc[match_indices].reset_index(drop=True)

            # --- USE CENTRALIZED SCALES ---
            u_raw = torch.tensor(df_matched['u'].values, dtype=torch.float32) * self.phys_cfg.cm_to_m
            v_raw = torch.tensor(df_matched['v'].values, dtype=torch.float32) * self.phys_cfg.cm_to_m
            p_raw = torch.tensor(df_matched['p'].values, dtype=torch.float32) * self.phys_cfg.cgs_p_to_pa
            mu_eff = torch.tensor(df_matched['mu_effective'].values, dtype=torch.float32) * self.phys_cfg.cgs_mu_to_pa_s

            u_ref_actual = self.phys_cfg.get_u_ref(d_bar)
            p_ref = self.phys_cfg.rho * (u_ref_actual ** 2)
            p_relative = p_raw - (p_raw[mask_outlet].mean() if mask_outlet.any() else p_raw.min())

            u_nd = u_raw / u_ref_actual
            v_nd = v_raw / u_ref_actual
            p_nd = p_relative / p_ref
            mu_nd = mu_eff / self.phys_cfg.mu_ref

            # --- USE CENTRALIZED BIOCHEM SCALES ---
            bio_cfg = BiochemConfig(tier=self.vessel_cfg.tier)
            species_cols = list(self.species_map.keys())
            raw_bulk_cgs = torch.tensor(df_matched[species_cols[:9]].values, dtype=torch.float32)
            raw_surf_cgs = torch.tensor(df_matched[species_cols[9:]].values, dtype=torch.float32)

            bulk_si = raw_bulk_cgs * bio_cfg.bulk_scale
            surf_si = raw_surf_cgs * bio_cfg.surface_scale
            species = torch.clamp(torch.cat([bulk_si, surf_si], dim=1), min=0.0)

            scales = bio_cfg.get_species_scales(device='cpu')
            species_nd = species / scales
            species_transformed = torch.log1p(species_nd)

            # Combine to [Nodes, 16]
            y_t = torch.cat([
                u_nd.unsqueeze(1), v_nd.unsqueeze(1), p_nd.unsqueeze(1),
                mu_nd.unsqueeze(1), species_transformed
            ], dim=1)

            y_trajectory.append(y_t)
            u_raw_list.append(u_raw)

            # Save the inlet/wall BCs explicitly from the FIRST timestep (t=0)
            if t_idx == 0:
                u_nd_0 = u_nd
                v_nd_0 = v_nd
                mu_nd_0 = mu_nd

        # Stack into shape: [Time, Nodes, 16]
        y_tensor_series = torch.stack(y_trajectory, dim=0)

        # 6. Gradients & Numerical Stability
        V, W, M_inv, max_cond = self._precompute_wls(edge_index, num_nodes, nodes_nd)
        G_x, G_y, Laplacian = self._precompute_sparse_operators(edge_index, num_nodes, M_inv, V, W)

        # 7. Mass Flux Calculation
        # Convert CGS (comsol) (cm/s) to SI (m/s)
        u_raw = torch.tensor(df_matched['u'].values, dtype=torch.float32) * 0.01
        v_raw = torch.tensor(df_matched['v'].values, dtype=torch.float32) * 0.01
        p_raw = torch.tensor(df_matched['p'].values, dtype=torch.float32) * 0.1
        mu_eff = torch.tensor(df_matched['mu_effective'].values, dtype=torch.float32) * 0.1

        # Calculate true normals for the inlets and outlets
        pos_tensor = torch.tensor(mesh_nodes, dtype=torch.float32)
        inlet_normals = self._compute_boundary_normals(edge_index, mask_inlet, pos_tensor, num_nodes)
        outlet_normals = self._compute_boundary_normals(edge_index, mask_outlet, pos_tensor, num_nodes)

        # Calculate True Mass Flux via Dot Product (U dot N)
        inlet_v = torch.stack([ u_raw[ mask_inlet ], v_raw[ mask_inlet ] ], dim=1)
        outlet_v = torch.stack([ u_raw[ mask_outlet ], v_raw[ mask_outlet ] ], dim=1)

        inlet_flux = torch.abs(torch.sum(inlet_v * inlet_normals[ mask_inlet ])).item()
        outlet_flux = torch.abs(torch.sum(outlet_v * outlet_normals[ mask_outlet ])).item()

        flux_imbalance = abs(inlet_flux - outlet_flux) / (inlet_flux + 1e-8)

        # 8. Data-Driven ML Scaling
        # Compute U_ref from the 99th percentile to clamp data strictly to ~[-1.0, 1.0]
        u_ref_actual = self.phys_cfg.get_u_ref(d_bar)  # Matches Tier 1/2 logic exactly
        re_actual = self.phys_cfg.re_target  # Locks the ML Re to your target

        # Ensure your bio_cfg scales match these new SI units
        bio_cfg = BiochemConfig(tier=self.vessel_cfg.tier)
        scales = torch.tensor([
            bio_cfg.c_RP0 * 1e6, bio_cfg.c_RP0 * 1e6, bio_cfg.APRcrit * 1e6, bio_cfg.APScrit * 1e6,
            bio_cfg.c_pT0 * 1e6, bio_cfg.c_pT0 * 1e6, bio_cfg.cAT0 * 1e6, bio_cfg.c_Fg0 * 1e6,
            bio_cfg.c_Fg0 * 1e6, bio_cfg.Minf * 1e4, bio_cfg.Minf * 1e4, bio_cfg.Minf * 1e4
        ], dtype=torch.float32)

        # 9. SDF and Normals
        wall_coords = mesh_nodes[ mask_wall.numpy() ]
        if len(wall_coords) > 0:
            wall_tree = KDTree(wall_coords)
            dist, idx = wall_tree.query(mesh_nodes)
            sdf = torch.tensor(dist / d_bar, dtype=torch.float32).unsqueeze(1)
            normals_unit = self._compute_boundary_normals(edge_index, mask_wall, pos_tensor, num_nodes)
        else:
            sdf = torch.zeros((num_nodes, 1))
            normals_unit = torch.zeros((num_nodes, 2))

        # Boundary Masks
        m_in = mask_inlet.float().unsqueeze(1)
        m_out = mask_outlet.float().unsqueeze(1)
        m_wall = mask_wall.float().unsqueeze(1)

        # Velocity BCs
        u_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)
        v_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)
        u_bc[ mask_inlet, 0 ] = u_nd_0[ mask_inlet ]  # FIX: Was u_nd
        v_bc[ mask_inlet, 0 ] = v_nd_0[ mask_inlet ]  # FIX: Was v_nd

        # Pressure BC
        p_bc = torch.zeros((num_nodes, 1), dtype=torch.float32)

        # Active BC Masks
        uv_mask = (mask_inlet | mask_wall).float().unsqueeze(1)
        p_mask = mask_outlet.float().unsqueeze(1)

        # Viscosity BC & Mask
        mu_bc = mu_nd_0.unsqueeze(1)  # FIX: Was mu_nd
        mu_mask = torch.ones((num_nodes, 1), dtype=torch.float32)

        # Assuming you had 3 initial guess channels of zeros to make 15 total, like Tier 1
        zero_init = torch.zeros((num_nodes, 3), dtype=torch.float32)

        # --- FIX: Ensure x_tensor follows the Tier 2 foundational layout ---
        x_tensor = torch.cat([
            nodes_nd,  # [0:2] - MUST be here for Fourier Encoding
            sdf,  # [2:3]
            normals_unit,  # [3:5]
            m_in,  #
            m_out,  #
            m_wall,  #
            u_bc,  #
            v_bc,  #
            p_bc,  #
            uv_mask,  #
            p_mask,  #
            mu_bc,  #
            mu_mask  #
        ], dim=1)

        # --- Use index assignment for the tensor ---
        inlet_species_si = torch.zeros(9, dtype=torch.float32)
        inlet_species_si[0] = bio_cfg.c_RP0 * bio_cfg.bulk_scale  # RP
        inlet_species_si[4] = bio_cfg.c_pT0 * bio_cfg.bulk_scale  # PT
        inlet_species_si[6] = bio_cfg.cAT0 * bio_cfg.bulk_scale  # AT
        inlet_species_si[7] = bio_cfg.c_Fg0 * bio_cfg.bulk_scale  # FG
        # Note: AP, APR, APS, T, FI remain 0.0 at the inlet

        # Scale and transform
        inlet_species_nd = inlet_species_si / scales[ :9 ]
        inlet_species_transformed = torch.log1p(inlet_species_nd)

        # Broadcast to all nodes
        bio_inlet_bc = inlet_species_transformed.unsqueeze(0).expand(num_nodes, -1)

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
        uv_inlet_bc = torch.cat([ u_nd_0.unsqueeze(1), v_nd_0.unsqueeze(1) ], dim=1)
        data = Data(
            x=x_tensor,
            y=y_tensor_series,
            t=eval_times_tensor,
            edge_index=edge_index, edge_attr=edge_attr,
            mask_inlet=mask_inlet, mask_outlet=mask_outlet, mask_wall=mask_wall,
            is_anchor=torch.tensor([ True ], dtype=torch.bool),
            d_bar=torch.tensor([ d_bar ], dtype=torch.float32),
            u_ref=torch.tensor([ u_ref_actual ], dtype=torch.float32),
            re_actual=torch.tensor([ re_actual ], dtype=torch.float32),
            G_x=G_x, G_y=G_y, Laplacian=Laplacian, V=V, W=W, M_inv=M_inv,
            u_inlet_bc=uv_inlet_bc,
            mu_inlet_bc=mu_nd_0.unsqueeze(1), mu_wall_bc=mu_nd_0.unsqueeze(1),
            bio_inlet_bc=bio_inlet_bc
        )

        torch.save(data, self.proc_dir / f"{stem}.pt")
        print(f"✅ Saved {stem}: D={d_bar * 1000:.1f}mm | Re_ML={re_actual:.0f} | Imbal={flux_imbalance:.2%}")

        if self.visualize:
            self._plot_sanity_check(data, stem)

    def run(self):
        # Look for the .nas files now!
        files = [ f for f in os.listdir(self.raw_dir) if f.endswith(".nas") or f.endswith(".msh") ]

        if len(files) == 0:
            print(f"CRITICAL ERROR: No .msh or .nas files found in {self.raw_dir}")
            return

        stems = sorted(list(set([ Path(f).stem for f in files ])))

        for stem in tqdm(stems, desc="Extracting Tier 3 Patient Data"):
            self.process_patient(stem)

if __name__ == "__main__":
    extractor = PatientDataExtractor(tier="tier3_patients", visualize=True)
    extractor.run()