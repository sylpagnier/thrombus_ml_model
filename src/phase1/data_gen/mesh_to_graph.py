import os
import torch
import json
import numpy as np
import meshio
from pathlib import Path
from scipy.spatial import KDTree, cKDTree
from torch_geometric.data import Data
from tqdm import tqdm
from src.config import VesselConfig, PhysicsConfig
from src.utils.paths import get_project_root


class MeshToGraphComplete:
    def __init__(self, tier="tier1", raw_dir=None, label_dir=None, proc_dir=None):
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
        row, col = edge_index
        pos_diff = pos_tensor[col, :2] - pos_tensor[row, :2]
        dx, dy = pos_diff[:, 0], pos_diff[:, 1]

        dist_sq = dx ** 2 + dy ** 2 + 1e-8

        # 2nd Order Polynomial Basis
        dx2 = 0.5 * dx ** 2
        dxy = dx * dy
        dy2 = 0.5 * dy ** 2

        V = torch.stack([dx, dy, dx2, dxy, dy2], dim=1)
        W = 1.0 / dist_sq

        V_unsqueezed = V.unsqueeze(2)
        V_T_unsqueezed = V.unsqueeze(1)
        M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)

        # Scatter sum equivalent
        M_e_flat = M_e.view(-1, 25)
        out = torch.zeros((num_nodes, 25), dtype=M_e_flat.dtype, device=M_e_flat.device)
        row_exp = row.view(-1, 1).expand_as(M_e_flat)
        M_flat = out.scatter_add_(0, row_exp, M_e_flat)

        M = M_flat.view(num_nodes, 5, 5)
        epsilon = 1e-6
        I = torch.eye(5, dtype=M.dtype, device=M.device).unsqueeze(0).expand(num_nodes, 5, 5)
        M_reg = M + epsilon * I
        M_inv = torch.linalg.pinv(M_reg)

        # FIX 1: Add a dummy dimension so PyG DataLoader concatenates properly
        # We will squeeze it during Data object construction
        return V, W, M_inv.unsqueeze(1)

    def _get_boundary_masks(self, mesh, num_nodes):
        mask_inlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_outlet = torch.zeros(num_nodes, dtype=torch.bool)
        mask_wall = torch.zeros(num_nodes, dtype=torch.bool)

        # Initialize empty
        line_cells = []
        line_tags = []

        # Define standard tags from Config
        tags = self.vessel_cfg.TAGS
        t_in = tags["Inlet"]
        t_out = tags["Outlet_1"]
        t_wall = tags["Walls"]

        # Populate if data exists
        try:
            if "line" in mesh.cells_dict:
                line_cells = mesh.cells_dict["line"]
                line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
            elif hasattr(mesh, "get_cells_type"):
                line_cells = mesh.get_cells_type("line")
                line_tags = mesh.get_cell_data("gmsh:physical", "line")
        except Exception:
            pass

        # Loop is safe even if line_tags is empty
        for i, tag in enumerate(line_tags):
            if isinstance(line_cells, list) and not isinstance(line_cells[0], (int, float, np.integer)):
                nodes = line_cells[i]
            else:
                nodes = line_cells[i]

            if tag == t_in:
                mask_inlet[nodes] = True
            elif tag == t_out:
                mask_outlet[nodes] = True
            elif tag == t_wall:
                mask_wall[nodes] = True

        mask_inlet = mask_inlet & (~mask_wall)
        mask_outlet = mask_outlet & (~mask_wall)

        return mask_inlet, mask_outlet, mask_wall

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

        # Scaling Factors
        ref_mu = self.phys_cfg.mu_ref
        u_ref = self.phys_cfg.get_u_ref(d_bar)
        p_ref_scale = self.phys_cfg.get_p_ref(u_ref)

        # --- ROBUST WALL Normal & Distance Calculation ---
        wall_node_indices = np.where(mask_wall.numpy())[0]
        if len(wall_node_indices) == 0: return
        wall_pts = nodes[wall_node_indices]

        # 1. Standard distance from wall for interior nodes
        tree_wall = KDTree(wall_pts)
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

            # Calculate vessel center to ensure normals point inward (into the fluid)
            interior_mask = ~(mask_wall.numpy() | mask_inlet.numpy() | mask_outlet.numpy())
            center_pt = np.mean(nodes[interior_mask], axis=0) if interior_mask.any() else np.mean(nodes, axis=0)

            for line in wall_lines:
                idx_a, idx_b = line[0], line[1]
                pt_a, pt_b = nodes[idx_a], nodes[idx_b]

                # Tangent vector
                dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]

                # Orthogonal normal vector (-dy, dx)
                n = np.array([-dy, dx])

                # Ensure the normal points towards the vessel interior
                midpoint = (pt_a + pt_b) / 2.0
                if np.dot(n, center_pt - midpoint) < 0:
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
                mu_nd = torch.tensor(cfd['mu'].flatten()[idx] / ref_mu,
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

        # --- Refurbished Analytic Prior (SDF-Based) ---
        # 1. Use SDF to determine r_nd (0 at center, R at wall)
        R_nd = 0.5  # Normalized Radius is always 0.5 if d_bar is the scaling factor
        r_nd = torch.abs(R_nd - sdf_tensor.squeeze())

        # 2. Velocity & Shear Rate Prior
        u_max_nd = 1.5
        u_prior_mag = torch.clamp(u_max_nd * (1.0 - (r_nd ** 2 / (R_nd ** 2 + 1e-12))), min=0.0)

        # 3. Viscosity Prior (Carreau)
        gamma_dot_prior = torch.abs(-2.0 * u_max_nd * r_nd / (R_nd ** 2 + 1e-12))
        lambda_nd = self.phys_cfg.lam * (u_ref / d_bar)
        mu_prior = (self.phys_cfg.mu_inf / ref_mu) + ((self.phys_cfg.mu_0 / ref_mu) - (self.phys_cfg.mu_inf / ref_mu)) * \
                   (1.0 + (lambda_nd * gamma_dot_prior) ** self.phys_cfg.a) ** (
                               (self.phys_cfg.n - 1.0) / self.phys_cfg.a)

        # 4. WSS Prior: MASKED to wall boundary
        wss_prior = (mu_prior * gamma_dot_prior) * mask_wall.float()

        # --- Final Assembly ---
        x_tensor = torch.cat([
            pos_nd_tensor, sdf_tensor, torch.abs(1.0 - 2.0 * sdf_tensor),  # Pos, SDF, ShearPot
            wall_normal_vec,
            torch.zeros((len(nodes), 4)),  # Node Type (Placeholder)
            torch.full((len(nodes), 1), 1.0 if self.phys_cfg.viscosity_model == "carreau" else 0.0),
            u_prior_mag.view(-1, 1), torch.zeros((len(nodes), 1)),  # UV Prior
            mu_prior.view(-1, 1), wss_prior.view(-1, 1)  # Mu and WSS Prior
        ], dim=1)

        data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_labels,
                    mask_inlet=mask_inlet, mask_outlet=mask_outlet, mask_wall=mask_wall,
                    is_anchor=torch.tensor([is_anchor], dtype=torch.bool),
                    d_bar=torch.tensor([d_bar], dtype=torch.float32),
                    u_ref=torch.tensor([u_ref], dtype=torch.float32),
                    u_inlet_bc=u_prior_mag.view(-1, 1),
                    mu_inlet_bc=mu_prior.view(-1, 1),
                    mu_wall_bc=mu_prior.view(-1, 1),
                    V=V, W=W, M_inv=M_inv)

        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self):
        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".msh")])
        for f in tqdm(files):
            self.process_file(f)


if __name__ == "__main__":
    active_tier = "tier2"
    processor = MeshToGraphComplete(tier=active_tier)
    processor.run()