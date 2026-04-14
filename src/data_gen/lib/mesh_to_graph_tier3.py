"""
Tier 3 synthetic mesh-to-graph conversion.

Scope: non-anchor Tier 3 samples (synthetic meshes / priors-only trajectory setup).
Anchor/patient samples with COMSOL trajectories are produced by:
``src.data_gen.lib.extract_tier3_comsol_data.PatientDataExtractor``.
"""

import os
import torch
import json
import numpy as np
import meshio
from pathlib import Path
from scipy.spatial import KDTree, cKDTree
from torch_geometric.data import Data
from tqdm import tqdm
from src.config import VesselConfig, PhysicsConfig, BiochemConfig
from .mesh_wls import gmsh_line_boundary_masks, precompute_wls_operators
from src.utils.paths import get_project_root
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

class MeshToGraphTier3:
    """Build Tier 3 non-anchor graphs from synthetic meshes."""

    def __init__(self, raw_dir=None, label_dir=None, proc_dir=None):
        tier = "tier3"
        self.root = get_project_root()
        self.vessel_cfg = VesselConfig(tier=tier)
        self.phys_cfg = PhysicsConfig(tier=tier)

        # Resolve Raw Dir
        if raw_dir:
            self.raw_dir = Path(raw_dir)
        else:
            self.raw_dir = self.root / self.vessel_cfg.mesh_input_dir

        # Resolve Label Dir
        if label_dir:
            self.label_dir = Path(label_dir)
        else:
            self.label_dir = self.root / self.vessel_cfg.output_dir

        # Resolve Processed Dir
        if proc_dir:
            self.proc_dir = Path(proc_dir)
        else:
            self.proc_dir = self.root / self.vessel_cfg.graph_output_dir

        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def _precompute_wls(self, edge_index, num_nodes, pos_tensor):
        """Modified to accept pos_tensor directly so it can run before x_tensor assembly."""
        return precompute_wls_operators(edge_index, num_nodes, pos_tensor)

    def _get_boundary_masks(self, mesh, num_nodes):
        return gmsh_line_boundary_masks(mesh, num_nodes, dict(self.vessel_cfg.TAGS))

    def _compute_outlet_normals(self, mesh, nodes, mask_outlet):
        """Compute unit normals on outlet nodes from Gmsh outlet line segments."""
        outlet_normal = np.zeros((len(nodes), 2), dtype=np.float32)
        t_out = self.vessel_cfg.TAGS["Outlet_1"]
        outlet_lines = []

        try:
            if "line" in mesh.cells_dict:
                l_cells = mesh.cells_dict["line"]
                l_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
            elif hasattr(mesh, "get_cells_type"):
                l_cells = mesh.get_cells_type("line")
                l_tags = mesh.get_cell_data("gmsh:physical", "line")
            else:
                l_cells, l_tags = [], []

            for i, tag in enumerate(l_tags):
                if tag == t_out:
                    outlet_lines.append(l_cells[i])
        except Exception:
            outlet_lines = []

        for line in outlet_lines:
            idx_a, idx_b = line[0], line[1]
            pt_a, pt_b = nodes[idx_a], nodes[idx_b]
            dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]
            n = np.array([dy, -dx], dtype=np.float32)
            n_norm = np.linalg.norm(n)
            if n_norm > 0:
                n = n / n_norm
                outlet_normal[idx_a] += n
                outlet_normal[idx_b] += n

        mag = np.linalg.norm(outlet_normal, axis=1, keepdims=True)
        outlet_normal = outlet_normal / (mag + 1e-12)
        outlet_normal[~mask_outlet.numpy()] = 0.0
        return torch.tensor(outlet_normal, dtype=torch.float32)

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

        # --- Element & Metadata Extraction ---
        all_tris = []
        if "triangle" in mesh.cells_dict:
            all_tris.append(mesh.cells_dict["triangle"])
        elif hasattr(mesh, "get_cells_type"):
            tc = mesh.get_cells_type("triangle")
            if len(tc) > 0: all_tris.append(tc)

        if not all_tris: return
        tri_nodes = np.vstack(all_tris)

        d_bar = None
        if json_path.exists():
            with open(json_path, 'r') as f:
                meta = json.load(f)
                d_bar = meta.get('d_bar')

        mask_inlet, mask_outlet, mask_wall = self._get_boundary_masks(mesh, len(nodes))
        outlet_normal = self._compute_outlet_normals(mesh, nodes, mask_outlet)

        # Scaling: u_ref uses mu_ref (Re); label mu_nd uses mu_viscosity_nd_scale (channel STATE_CHANNEL_MU_EFF_ND).
        mu_nd_scale = self.phys_cfg.mu_viscosity_nd_scale
        u_ref = self.phys_cfg.get_u_ref(d_bar)
        p_ref_scale = self.phys_cfg.get_p_ref(u_ref)

        # --- ROBUST WALL Normal & Distance Calculation ---
        wall_node_indices = np.where(mask_wall.numpy())[0]
        if len(wall_node_indices) == 0: return
        wall_pts = nodes[wall_node_indices]

        # 1. Standard distance from wall for interior nodes
        tree_wall = cKDTree(wall_pts)
        dist_raw, indices_wall = tree_wall.query(nodes)

        nearest_wall_pts = wall_pts[indices_wall]
        diff_vec = nodes - nearest_wall_pts  # Points FROM wall TO node (into the fluid)

        # 2. Exact Mathematical Normals for Wall Nodes using Gmsh Line Segments
        t_wall = self.vessel_cfg.TAGS["Walls"]
        wall_lines = []

        try:
            if "line" in mesh.cells_dict:
                l_cells = mesh.cells_dict["line"]
                l_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
            elif hasattr(mesh, "get_cells_type"):
                l_cells = mesh.get_cells_type("line")
                l_tags = mesh.get_cell_data("gmsh:physical", "line")

            for i, tag in enumerate(l_tags):
                if tag == t_wall:
                    wall_lines.append(l_cells[i])
        except Exception:
            pass

        if len(wall_lines) > 0:
            node_normals = np.zeros((len(nodes), 2))

            interior_mask = ~(mask_wall.numpy() | mask_inlet.numpy() | mask_outlet.numpy())
            interior_nodes = nodes[interior_mask]

            if len(interior_nodes) > 0:
                interior_tree = cKDTree(interior_nodes)
            else:
                interior_tree = None
                # Fallback if no interior nodes exist (highly unlikely)
                center_pt = np.mean(nodes, axis=0)

            for line in wall_lines:
                idx_a, idx_b = line[0], line[1]
                pt_a, pt_b = nodes[idx_a], nodes[idx_b]

                # Tangent vector
                dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]

                # Orthogonal normal vector (-dy, dx)
                n = np.array([-dy, dx])

                # Ensure the normal points towards the vessel interior
                midpoint = (pt_a + pt_b) / 2.0

                if interior_tree is not None:
                    # Find the strictly closest fluid node to this wall segment
                    _, nearest_idx = interior_tree.query(midpoint)
                    nearest_interior_pt = interior_nodes[nearest_idx]
                    inward_vec = nearest_interior_pt - midpoint
                else:
                    inward_vec = center_pt - midpoint

                # If the normal points away from the local inward vector, flip it
                if np.dot(n, inward_vec) < 0:
                    n = -n

                # Accumulate the normalized segment normal to the vertices
                n_norm = n / (np.linalg.norm(n) + 1e-12)
                node_normals[idx_a] += n_norm
                node_normals[idx_b] += n_norm

            # Replace KDTree vectors with the exact geometric normals for wall nodes
            diff_vec[wall_node_indices] = node_normals[wall_node_indices]

        # 3. Explicitly Normalize every vector to a magnitude of exactly 1.0
        norms = np.linalg.norm(diff_vec, axis=1, keepdims=True)
        wall_normal_vec = torch.tensor(diff_vec / (norms + 1e-12), dtype=torch.float32)

        # Non-dimensionalize
        nodes_nd = nodes / d_bar
        pos_nd_tensor = torch.tensor(nodes_nd, dtype=torch.float32)
        sdf_nd = dist_raw / d_bar
        sdf_tensor = torch.clamp(torch.tensor(sdf_nd, dtype=torch.float32).view(-1, 1), min=1e-6)

        # --- Graph Assembly & WLS Precomputation ---
        edges = np.unique(np.sort(np.vstack([
            tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)
        row, col = edge_index

        edge_attr = torch.cat([pos_nd_tensor[row] - pos_nd_tensor[col],
                               torch.linalg.norm(pos_nd_tensor[row] - pos_nd_tensor[col], dim=1, keepdim=True)], dim=1)

        V, W, M_inv = self._precompute_wls(edge_index, len(nodes), pos_nd_tensor)
        M_inv = M_inv.squeeze(1)

        # --- Ground Truth Mapping (WSS Calculation) ---
        y_labels = torch.zeros((len(nodes), 5), dtype=torch.float32)
        is_anchor = False

        if label_path.exists():
            try:
                cfd = np.load(label_path)
                sol_points = np.stack([cfd['x'].flatten(), cfd['y'].flatten()], axis=-1)
                sol_tree = cKDTree(sol_points)
                _, idx = sol_tree.query(nodes)

                # 1. Map raw values from CFD
                u_raw = torch.tensor(cfd['u'].flatten()[idx], dtype=torch.float32)
                v_raw = torch.tensor(cfd['v'].flatten()[idx], dtype=torch.float32)

                # 2. Hard-enforce No-Slip Condition
                # This prevents interpolation bleed from the interior fluid nodes
                u_raw[mask_wall] = 0.0
                v_raw[mask_wall] = 0.0

                # 3. Proceed with non-dimensionalization
                u_nd, v_nd = u_raw / u_ref, v_raw / u_ref
                p_nd = torch.tensor(cfd['p'].flatten()[idx] / p_ref_scale, dtype=torch.float32)
                mu_nd = torch.tensor(cfd['mu'].flatten()[idx] / mu_nd_scale,
                                     dtype=torch.float32) if 'mu' in cfd else torch.ones_like(u_nd)

                # WLS Gradients for WSS
                df_u, df_v = u_nd[col] - u_nd[row], v_nd[col] - v_nd[row]
                sum_W_V_du = torch.zeros((len(nodes), 5)).scatter_add_(0, row.unsqueeze(1).expand(-1, 5),
                                                                       W.unsqueeze(1) * V * df_u.unsqueeze(1))
                sum_W_V_dv = torch.zeros((len(nodes), 5)).scatter_add_(0, row.unsqueeze(1).expand(-1, 5),
                                                                       W.unsqueeze(1) * V * df_v.unsqueeze(1))

                grad_u, grad_v = torch.bmm(M_inv, sum_W_V_du.unsqueeze(2)).squeeze(), torch.bmm(M_inv,
                                                                                                sum_W_V_dv.unsqueeze(
                                                                                                    2)).squeeze()

                # Stress Tensor Components
                tau_xx = 2.0 * mu_nd * grad_u[:, 0]
                tau_yy = 2.0 * mu_nd * grad_v[:, 1]
                tau_xy = mu_nd * (grad_u[:, 1] + grad_v[:, 0])

                # Extract normal vector components
                n_x = wall_normal_vec[:, 0]
                n_y = wall_normal_vec[:, 1]

                # Project stress tensor onto the normal vector to get the traction vector (t = Tau * n)
                t_x = tau_xx * n_x + tau_xy * n_y
                t_y = tau_xy * n_x + tau_yy * n_y

                # True WSS magnitude is the magnitude of the traction vector at the wall
                wss_mag = torch.sqrt(t_x ** 2 + t_y ** 2) * mask_wall.float()
                y_labels = torch.stack([u_nd, v_nd, p_nd, mu_nd, wss_mag], dim=1)
                is_anchor = True
            except Exception as e:
                print(f"Error mapping labels: {e}")

        # --- Refurbished Analytic Prior (Geodesic & SDF-Based) ---
        # --------------------------------------------------------------------------
        #  Compute Explicit Sparse Gradient Matrices (G_x, G_y) for GINO
        # --------------------------------------------------------------------------
        W_V = W.unsqueeze(1) * V  # Shape: (E, 5)
        M_inv_row = M_inv[row]  # Shape: (E, 5, 5)

        # Calculate the gradient coefficients for each edge
        coeffs = torch.bmm(M_inv_row, W_V.unsqueeze(2)).squeeze(2)  # Shape: (E, 5)
        cx = coeffs[:, 0]
        cy = coeffs[:, 1]
        c_lap = coeffs[:, 2] + coeffs[:, 4]

        N = len(nodes)

        # The diagonal entries are the negative sum of the off-diagonal coefficients
        diag_cx = torch.zeros(N, dtype=torch.float32).scatter_add_(0, row, cx)
        diag_cy = torch.zeros(N, dtype=torch.float32).scatter_add_(0, row, cy)

        # Assemble indices for sparse tensors (Off-diagonals + Diagonals)
        diag_indices = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2, 1)

        idx_x = torch.cat([edge_index, diag_indices], dim=1)
        val_x = torch.cat([cx, -diag_cx], dim=0)

        idx_y = torch.cat([edge_index, diag_indices], dim=1)
        val_y = torch.cat([cy, -diag_cy], dim=0)

        diag_c_lap = torch.zeros(N, dtype=torch.float32).scatter_add_(0, row, c_lap)
        idx_lap = torch.cat([edge_index, diag_indices], dim=1)
        val_lap = torch.cat([c_lap, -diag_c_lap], dim=0)

        # Create the sparse tensors that GINO DEQ uses for derivatives
        G_x = torch.sparse_coo_tensor(idx_x, val_x, size=(N, N)).coalesce()
        G_y = torch.sparse_coo_tensor(idx_y, val_y, size=(N, N)).coalesce()
        Laplacian = torch.sparse_coo_tensor(idx_lap, val_lap, size=(N, N)).coalesce()

        # 1. Continuous SDF Profile (Fixes jagged viscosity seams in aneurysms)
        R_nd = 0.5
        r_nd = torch.clamp(R_nd - sdf_tensor.squeeze(), min=0.0)

        # 2. Velocity Magnitude Prior (Poiseuille)
        u_max_nd = 1.5
        u_prior_mag = u_max_nd * (1.0 - (r_nd ** 2 / (R_nd ** 2)))

        # 3. Derive Flow Direction (Geodesic Gradient)
        # Calculate the shortest path distance from the inlet using the mesh edges
        row_np, col_np = row.numpy(), col.numpy()
        dist_np = edge_attr[:, 2].numpy()  # Edge lengths

        adj = coo_matrix((dist_np, (row_np, col_np)), shape=(len(nodes), len(nodes)))
        inlet_idx = np.where(mask_inlet.numpy())[0]

        if len(inlet_idx) > 0:
            geodesic_dist = dijkstra(adj, directed=False, indices=inlet_idx).min(axis=0)
            phi = torch.tensor(geodesic_dist, dtype=torch.float32)

            # The gradient of the distance from the inlet points exactly downstream
            flow_dir_x = torch.sparse.mm(G_x, phi.unsqueeze(1)).squeeze()
            flow_dir_y = torch.sparse.mm(G_y, phi.unsqueeze(1)).squeeze()

            # Normalize the flow vectors
            flow_mag = torch.sqrt(flow_dir_x ** 2 + flow_dir_y ** 2) + 1e-12
            flow_dir_x /= flow_mag
            flow_dir_y /= flow_mag
        else:
            flow_dir_x = torch.ones(len(nodes))
            flow_dir_y = torch.zeros(len(nodes))

        # Project magnitude onto the exact downstream directions
        u_prior = u_prior_mag * flow_dir_x
        v_prior = u_prior_mag * flow_dir_y

        # 4. Viscosity Prior (Carreau)
        gamma_dot_prior = torch.abs(-2.0 * u_max_nd * r_nd / (R_nd ** 2))
        lambda_nd = self.phys_cfg.lam * (u_ref / d_bar)

        mu_prior = (self.phys_cfg.mu_inf / mu_nd_scale) + (
                (self.phys_cfg.mu_0 / mu_nd_scale) - (self.phys_cfg.mu_inf / mu_nd_scale)) * \
                   (1.0 + (lambda_nd * gamma_dot_prior) ** self.phys_cfg.a) ** (
                           (self.phys_cfg.n - 1.0) / self.phys_cfg.a)

        # 5. WSS Prior: MASKED to wall boundary
        wss_prior = (mu_prior * gamma_dot_prior) * mask_wall.float()

        # --- Final Assembly ---
        m_in = mask_inlet.float().unsqueeze(1)
        m_out = mask_outlet.float().unsqueeze(1)
        m_wall = mask_wall.float().unsqueeze(1)

        u_bc = torch.zeros((len(nodes), 1), dtype=torch.float32)
        v_bc = torch.zeros((len(nodes), 1), dtype=torch.float32)
        u_bc[mask_inlet, 0] = u_prior[mask_inlet]
        v_bc[mask_inlet, 0] = v_prior[mask_inlet]

        p_bc = torch.zeros((len(nodes), 1), dtype=torch.float32)
        uv_mask = (mask_inlet | mask_wall).float().unsqueeze(1)
        p_mask = mask_outlet.float().unsqueeze(1)
        mu_bc = mu_prior.view(-1, 1)
        mu_mask = torch.ones((len(nodes), 1), dtype=torch.float32)

        x_tensor = torch.cat([
            pos_nd_tensor,  # [0:2]
            sdf_tensor,  # [2:3]
            wall_normal_vec,  # [3:5]
            m_in,  # [5:6]
            m_out,  # [6:7]
            m_wall,  # [7:8]
            u_bc,  # [8:9]
            v_bc,  # [9:10]
            p_bc,  # [10:11]
            uv_mask,  # [11:12]
            p_mask,  # [12:13]
            mu_bc,  # [13:14]
            mu_mask,  # [14:15]
        ], dim=1)



        # --- TIER 3 NON-ANCHOR FORMAT: build transient-compatible dummy trajectory ---
        if self.vessel_cfg.tier == "tier3" and not is_anchor:
            bio_cfg = BiochemConfig(tier="tier3")

            # 1) Dummy time trajectory [T, N, 16] + time tensor
            num_times = bio_cfg.num_time_steps
            eval_times_tensor = torch.linspace(0.0, bio_cfg.t_final, num_times, dtype=torch.float32)
            y_tensor_series = torch.zeros((num_times, len(nodes), 16), dtype=torch.float32)

            # 2) Biochemical inlet BCs in transformed (log1p ND) space
            scales = bio_cfg.get_species_scales(device="cpu")
            inlet_species_si = torch.zeros(9, dtype=torch.float32)
            inlet_species_si[0] = bio_cfg.c_RP0 * bio_cfg.bulk_scale  # RP
            inlet_species_si[4] = bio_cfg.c_pT0 * bio_cfg.bulk_scale  # PT
            inlet_species_si[6] = bio_cfg.cAT0 * bio_cfg.bulk_scale   # AT
            inlet_species_si[7] = bio_cfg.c_Fg0 * bio_cfg.bulk_scale  # FG

            inlet_species_nd = inlet_species_si / scales[:9]
            inlet_species_transformed = torch.log1p(inlet_species_nd)
            bio_inlet_bc = inlet_species_transformed.unsqueeze(0).expand(len(nodes), -1)

            # 3) Build Tier 3-style Data object
            uv_inlet_bc = torch.cat([u_prior.view(-1, 1), v_prior.view(-1, 1)], dim=1)
            data = Data(
                x=x_tensor,
                y=y_tensor_series,
                t=eval_times_tensor,
                edge_index=edge_index,
                edge_attr=edge_attr,
                mask_inlet=mask_inlet,
                mask_outlet=mask_outlet,
                mask_wall=mask_wall,
                is_anchor=torch.zeros(len(nodes), dtype=torch.bool),
                d_bar=torch.tensor([d_bar], dtype=torch.float32),
                u_ref=torch.tensor([u_ref], dtype=torch.float32),
                re_actual=torch.tensor([self.phys_cfg.re_target], dtype=torch.float32),
                G_x=G_x,
                G_y=G_y,
                Laplacian=Laplacian,
                V=V,
                W=W,
                M_inv=M_inv,
                u_inlet_bc=uv_inlet_bc,
                mu_inlet_bc=mu_prior.view(-1, 1),
                bio_inlet_bc=bio_inlet_bc,
                outlet_normal=outlet_normal,
            )
        else:
            # Original Tier 1/2 format (steady labels + anchor flag)
            data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_labels,
                        mask_inlet=mask_inlet, mask_outlet=mask_outlet, mask_wall=mask_wall,
                        is_anchor=torch.tensor([is_anchor], dtype=torch.bool),
                        d_bar=torch.tensor([d_bar], dtype=torch.float32),
                        u_ref=torch.tensor([u_ref], dtype=torch.float32),
                        u_inlet_bc=u_prior.view(-1, 1),
                        mu_inlet_bc=mu_prior.view(-1, 1),
                        outlet_normal=outlet_normal,
                        V=V, W=W, M_inv=M_inv,
                        G_x=G_x, G_y=G_y, Laplacian=Laplacian)

        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self, max_files=None):
        files = sorted(self.raw_dir.glob("*.msh"))
        if max_files is not None:
            files = files[:max_files]
        for f in tqdm(files):
            self.process_file(f.name)


if __name__ == "__main__":
    def _prompt_optional_int(label):
        while True:
            raw = input(f"{label} [all]: ").strip()
            raw_l = raw.lower()
            if raw == "" or raw_l == "all":
                return None
            try:
                return int(raw)
            except ValueError:
                print("Invalid input. Enter an integer value, 'all', or leave blank for all.")

    print("\nTier 3 synthetic graph generation (non-anchor only)")
    print("Anchor/patient graphs are generated via extract_tier3_comsol_data.py")
    print("\nNumber of vessels:")
    print("  - Enter an integer to process only that many meshes")
    print("  - Enter 'all' (or leave blank) to process all meshes")

    max_files = _prompt_optional_int("Number of vessels")
    processor = MeshToGraphTier3()
    processor.run(max_files=max_files)
