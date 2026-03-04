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

        # --- Element Extraction ---
        all_tris = []
        if "triangle" in mesh.cells_dict:
            all_tris.append(mesh.cells_dict["triangle"])
        elif hasattr(mesh, "get_cells_type"):
            tc = mesh.get_cells_type("triangle")
            if len(tc) > 0: all_tris.append(tc)

        if "quad" in mesh.cells_dict:
            qn = mesh.cells_dict["quad"]
            all_tris.extend([qn[:, [0, 1, 2]], qn[:, [0, 2, 3]]])
        elif hasattr(mesh, "get_cells_type"):
            qc = mesh.get_cells_type("quad")
            if len(qc) > 0:
                all_tris.extend([qc[:, [0, 1, 2]], qc[:, [0, 2, 3]]])

        if not all_tris:
            return
        tri_nodes = np.vstack(all_tris)

        d_bar = None
        if json_path.exists():
            with open(json_path, 'r') as f:
                meta = json.load(f)
                d_bar = meta.get('d_bar')

        mask_inlet, mask_outlet, mask_wall = self._get_boundary_masks(mesh, len(nodes))

        if d_bar is None:
            if mask_inlet.any():
                inlet_nodes = nodes[mask_inlet]
                d_bar = np.max(np.linalg.norm(inlet_nodes - inlet_nodes.mean(axis=0), axis=1)) * 2

        ref_mu = self.phys_cfg.mu_ref
        u_ref = self.phys_cfg.get_u_ref(d_bar)
        p_ref_scale = self.phys_cfg.get_p_ref(u_ref)

        # --- WALL NORMAL CALCULATION ---
        wall_node_indices = np.where(mask_wall.numpy())[0]
        if len(wall_node_indices) == 0: return
        wall_pts = nodes[wall_node_indices]

        tree_wall = KDTree(wall_pts)
        dist_raw, indices_wall = tree_wall.query(nodes)

        nearest_wall_pts = wall_pts[indices_wall]
        diff_vec = nodes - nearest_wall_pts

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
            center_pt = np.mean(nodes[interior_mask], axis=0) if interior_mask.any() else np.mean(nodes, axis=0)

            for line in wall_lines:
                idx_a, idx_b = line[0], line[1]
                pt_a, pt_b = nodes[idx_a], nodes[idx_b]

                dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]
                n = np.array([-dy, dx])

                midpoint = (pt_a + pt_b) / 2.0
                flipped = False

                mask_a = np.any(tri_nodes == idx_a, axis=1)
                mask_b = np.any(tri_nodes == idx_b, axis=1)
                shared_tris = tri_nodes[mask_a & mask_b]

                if len(shared_tris) > 0:
                    tri = shared_tris[0]
                    idx_c = tri[(tri != idx_a) & (tri != idx_b)][0]
                    pt_c = nodes[idx_c]
                    interior_vec = pt_c - midpoint

                    if np.dot(n, interior_vec) < 0:
                        n = -n
                    flipped = True

                if not flipped:
                    if np.dot(n, center_pt - midpoint) < 0:
                        n = -n

                n_norm = n / (np.linalg.norm(n) + 1e-12)
                node_normals[idx_a] += n_norm
                node_normals[idx_b] += n_norm

            diff_vec[wall_node_indices] = node_normals[wall_node_indices]

        norms = np.linalg.norm(diff_vec, axis=1, keepdims=True)
        wall_normal_vec = diff_vec / (norms + 1e-12)

        # Normalize spatial nodes
        nodes_nd = nodes / d_bar
        pos_nd_tensor = torch.tensor(nodes_nd, dtype=torch.float32)

        # --- GRAPH EDGE ASSEMBLY ---
        edges = np.unique(np.sort(np.vstack([
            tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)
        row, col = edge_index

        # --- GINO ARCHITECTURE PREP: EDGE ATTRIBUTES ---
        edge_disp = pos_nd_tensor[row] - pos_nd_tensor[col]
        edge_dist = torch.linalg.norm(edge_disp, dim=1, keepdim=True)
        edge_attr = torch.cat([edge_disp, edge_dist], dim=1)

        # Precompute WLS operators using early pos_nd_tensor
        V, W, M_inv_expanded = self._precompute_wls(edge_index, len(nodes), pos_nd_tensor)

        # Squeeze the expanded M_inv down to (N, 5, 5) for PyG compatibility
        M_inv = M_inv_expanded.squeeze(1)

        # --- SPATIAL LABEL MAPPING & WSS TARGET ---
        y_labels = torch.zeros((len(nodes), 5), dtype=torch.float32)
        is_anchor = False

        if label_path.exists():
            try:
                cfd = np.load(label_path)
                sol_points = np.stack([cfd['x'].flatten(), cfd['y'].flatten()], axis=-1)
                sol_tree = cKDTree(sol_points)
                _, idx = sol_tree.query(nodes)

                u_raw = torch.tensor(cfd['u'].flatten()[idx], dtype=torch.float32)
                v_raw = torch.tensor(cfd['v'].flatten()[idx], dtype=torch.float32)
                p_raw = torch.tensor(cfd['p'].flatten()[idx], dtype=torch.float32)

                if 'mu' in cfd:
                    mu_raw = torch.tensor(cfd['mu'].flatten()[idx], dtype=torch.float32)
                else:
                    mu_raw = torch.full_like(u_raw, ref_mu)

                # Normalize labels
                u_nd, v_nd = u_raw / u_ref, v_raw / u_ref
                p_nd = p_raw / p_ref_scale
                mu_nd = mu_raw / ref_mu

                # ---- WSS CALCULATION USING PRECOMPUTED WLS GRADS ----
                df_u = u_nd[col] - u_nd[row]
                df_v = v_nd[col] - v_nd[row]

                W_V_du = W.unsqueeze(1) * V * df_u.unsqueeze(1)
                W_V_dv = W.unsqueeze(1) * V * df_v.unsqueeze(1)

                sum_W_V_du = torch.zeros((len(nodes), 5), dtype=torch.float32)
                sum_W_V_dv = torch.zeros((len(nodes), 5), dtype=torch.float32)

                sum_W_V_du.scatter_add_(0, row.unsqueeze(1).expand(-1, 5), W_V_du)
                sum_W_V_dv.scatter_add_(0, row.unsqueeze(1).expand(-1, 5), W_V_dv)

                # Execute WLS inversion to get velocity gradients
                grad_u = torch.bmm(M_inv, sum_W_V_du.unsqueeze(2)).squeeze(2)
                grad_v = torch.bmm(M_inv, sum_W_V_dv.unsqueeze(2)).squeeze(2)

                dudx, dudy = grad_u[:, 0], grad_u[:, 1]
                dvdx, dvdy = grad_v[:, 0], grad_v[:, 1]

                # Compute 2D Viscous Stress Tensor (tau)
                tau_xx = 2.0 * mu_nd * dudx
                tau_yy = 2.0 * mu_nd * dvdy
                tau_xy = mu_nd * (dudy + dvdx)

                # Project onto Wall Normals to find Traction (t = tau * n)
                nx = torch.tensor(wall_normal_vec[:, 0], dtype=torch.float32)
                ny = torch.tensor(wall_normal_vec[:, 1], dtype=torch.float32)

                tx = tau_xx * nx + tau_xy * ny
                ty = tau_xy * nx + tau_yy * ny

                # Tangential extraction (Wall Shear Stress magnitude)
                t_n = tx * nx + ty * ny
                wss_x = tx - t_n * nx
                wss_y = ty - t_n * ny
                wss_mag = torch.sqrt(wss_x ** 2 + wss_y ** 2 + 1e-8)

                # Mask WSS so it acts solely as a physical boundary constraint
                wss_mag = wss_mag * mask_wall.float()

                # Stack the updated 5 channels
                y_labels = torch.stack([u_nd, v_nd, p_nd, mu_nd, wss_mag], dim=1)
                is_anchor = True
            except Exception as e:
                print(f"Error mapping labels {filename}: {e}")

        # Inlet BC Calculation
        u_inlet_bc = torch.zeros((len(nodes), 2), dtype=torch.float32)
        mu_inlet_bc = torch.zeros(len(nodes), dtype=torch.float32)

        y_center = 0.0
        R = d_bar / 2.0

        if mask_inlet.any():
            inlet_indices = torch.where(mask_inlet)[0]
            y_coords = nodes[inlet_indices, 1]
            y_center = np.mean(y_coords)
            R = (np.max(y_coords) - np.min(y_coords)) / 2.0
            u_max = 1.5 * u_ref
            r = np.abs(y_coords - y_center)

            profile_mag = u_max * (1 - (r ** 2 / (R ** 2 + 1e-12)))
            u_inlet_bc[inlet_indices, 0] = torch.tensor(profile_mag / u_ref, dtype=torch.float32)

            if self.phys_cfg.viscosity_model == "carreau":
                gamma_dot = np.abs(-2.0 * u_max * r / (R ** 2 + 1e-12))
                gamma_dot_nd = gamma_dot / (u_ref / d_bar)
                lambda_nd = self.phys_cfg.lam * (u_ref / d_bar)
                a = self.phys_cfg.a
                n = self.phys_cfg.n

                shear_term = 1.0 + (lambda_nd * gamma_dot_nd) ** a
                power = (n - 1.0) / a

                mu_inf_nd = self.phys_cfg.mu_inf / self.phys_cfg.mu_ref
                mu_0_nd = self.phys_cfg.mu_0 / self.phys_cfg.mu_ref

                mu_profile = mu_inf_nd + (mu_0_nd - mu_inf_nd) * (shear_term ** power)
                mu_inlet_bc[inlet_indices] = torch.tensor(mu_profile, dtype=torch.float32)
            else:
                mu_inlet_bc[inlet_indices] = 1.0

        # --- Wall BC Calculation ---
        mu_wall_bc = torch.zeros(len(nodes), dtype=torch.float32)
        if mask_wall.any():
            if self.phys_cfg.viscosity_model == "carreau":
                mu_inf_nd = self.phys_cfg.mu_inf / self.phys_cfg.mu_ref
                mu_wall_bc[mask_wall] = torch.tensor(mu_inf_nd, dtype=torch.float32)
            else:
                mu_wall_bc[mask_wall] = 1.0

        # FIX 2: Clamp SDF to prevent NaN values in log calculations
        sdf_nd = dist_raw.reshape(-1, 1) / d_bar
        sdf_tensor = torch.tensor(sdf_nd, dtype=torch.float32)
        sdf_tensor = torch.clamp(sdf_tensor, min=1e-6)  # Added clamp
        shear_pot = torch.abs(1.0 - 2.0 * sdf_tensor)

        node_type = torch.zeros((len(nodes), 4), dtype=torch.float32)
        node_type[:, 0] = 1.0
        node_type[mask_inlet, 0] = 0.0
        node_type[mask_inlet, 1] = 1.0
        node_type[mask_outlet, 0] = 0.0
        node_type[mask_outlet, 2] = 1.0
        node_type[mask_wall, 0] = 0.0
        node_type[mask_wall, 3] = 1.0

        is_non_newt = 1.0 if self.phys_cfg.viscosity_model == "carreau" else 0.0

        u_max_nd = 1.5
        r_dist = torch.tensor(np.abs(nodes[:, 1] - y_center), dtype=torch.float32)
        r_nd = r_dist / d_bar
        R_nd = R / d_bar

        u_prior_mag = torch.clamp(u_max_nd * (1.0 - (r_nd ** 2 / (R_nd ** 2 + 1e-12))), min=0.0)

        # FIX 3: Pseudo-Huber regularization for gamma_dot_prior_nd to prevent NaN at 0
        eps = 1e-6
        raw_gamma = -2.0 * u_max_nd * r_nd / (R_nd ** 2 + 1e-12)
        gamma_dot_prior_nd = torch.sqrt(raw_gamma ** 2 + eps ** 2)
        gamma_dot_prior_nd[r_nd > R_nd] = 0.0

        lambda_nd = self.phys_cfg.lam * (u_ref / d_bar)
        shear_term = 1.0 + (lambda_nd * gamma_dot_prior_nd) ** self.phys_cfg.a
        power = (self.phys_cfg.n - 1.0) / self.phys_cfg.a

        mu_inf_nd = self.phys_cfg.mu_inf / self.phys_cfg.mu_ref
        mu_0_nd = self.phys_cfg.mu_0 / self.phys_cfg.mu_ref

        mu_prior = mu_inf_nd + (mu_0_nd - mu_inf_nd) * (shear_term ** power)
        mu_prior_tensor = mu_prior.view(-1, 1)

        u_prior_x = u_prior_mag.unsqueeze(1)
        v_prior_y = torch.zeros_like(u_prior_x)
        uv_prior = torch.cat([u_prior_x, v_prior_y], dim=1)

        wss_prior = mu_prior * gamma_dot_prior_nd
        wss_prior_tensor = wss_prior.view(-1, 1)

        # Assemble Final Graph Matrix
        x_tensor = torch.cat([
            pos_nd_tensor,
            sdf_tensor,
            shear_pot,
            torch.tensor(wall_normal_vec, dtype=torch.float32),
            node_type,
            torch.full((len(nodes), 1), is_non_newt, dtype=torch.float32),
            uv_prior,
            mu_prior_tensor,
            wss_prior_tensor
        ], dim=1)

        # Build PyG Data Object with edge_attr included
        data = Data(
            x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_labels,
            d_bar=torch.tensor([d_bar], dtype=torch.float32),
            u_ref=torch.tensor([u_ref], dtype=torch.float32),
            u_inlet_bc=u_inlet_bc,
            mu_inlet_bc=mu_inlet_bc,
            mu_wall_bc=mu_wall_bc,
            mask_inlet=mask_inlet, mask_outlet=mask_outlet, mask_wall=mask_wall,
            V=V, W=W, M_inv=M_inv,
            is_anchor=torch.tensor([is_anchor], dtype=torch.bool)
        )

        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self):
        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".msh")])
        for f in tqdm(files):
            self.process_file(f)


if __name__ == "__main__":
    active_tier = "tier2"
    processor = MeshToGraphComplete(tier=active_tier)
    processor.run()