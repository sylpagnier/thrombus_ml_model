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

    def _precompute_wls(self, edge_index, num_nodes, x_tensor):
        row, col = edge_index
        pos_diff = x_tensor[col, :2] - x_tensor[row, :2]
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
        M_inv = torch.linalg.pinv(M)  # Compute pseudo-inverse just once per graph

        return V, W, M_inv

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
            # Ensure we access the correct cell block if line_cells is a list of blocks
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

        # If a node is marked as a Wall, it cannot be an Inlet or Outlet
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

        # Initialize d_bar
        d_bar = None

        # Check json data
        if json_path.exists():
            with open(json_path, 'r') as f:
                meta = json.load(f)
                d_bar = meta.get('d_bar')

        mask_inlet, mask_outlet, mask_wall = self._get_boundary_masks(mesh, len(nodes))

        # d_bar Fallback
        if d_bar is None:
            if mask_inlet.any():
                inlet_nodes = nodes[mask_inlet]
                # Use Euclidean distance for non-vertical inlets
                d_bar = np.max(np.linalg.norm(inlet_nodes - inlet_nodes.mean(axis=0), axis=1)) * 2

        # Physics Scaling
        ref_mu = self.phys_cfg.mu_ref
        u_ref = self.phys_cfg.get_u_ref(d_bar)
        p_ref_scale = self.phys_cfg.get_p_ref(u_ref)

        # --- WALL NORMAL CALCULATION ---
        wall_node_indices = np.where(mask_wall.numpy())[0]
        if len(wall_node_indices) == 0: return
        wall_pts = nodes[wall_node_indices]

        # 1. Standard distance from wall for interior nodes
        tree_wall = KDTree(wall_pts)
        dist_raw, indices_wall = tree_wall.query(nodes)

        nearest_wall_pts = wall_pts[indices_wall]
        diff_vec = nodes - nearest_wall_pts

        try:
            if "triangle" in mesh.cells_dict:
                tri_nodes = mesh.cells_dict["triangle"]
            else:
                tri_nodes = mesh.get_cells_type("triangle")
        except Exception:
            tri_nodes = np.array([])

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

                # --- Use local mesh topology to orient normal ---
                midpoint = (pt_a + pt_b) / 2.0
                flipped = False

                if len(tri_nodes) > 0:
                    # Find the triangle sharing this boundary edge
                    mask_a = np.any(tri_nodes == idx_a, axis=1)
                    mask_b = np.any(tri_nodes == idx_b, axis=1)
                    shared_tris = tri_nodes[mask_a & mask_b]

                    if len(shared_tris) > 0:
                        tri = shared_tris[0]
                        # Get the 3rd node in the triangle (strictly in the fluid interior)
                        idx_c = tri[(tri != idx_a) & (tri != idx_b)][0]
                        pt_c = nodes[idx_c]
                        interior_vec = pt_c - midpoint

                        # If normal points away from the interior node, flip it
                        if np.dot(n, interior_vec) < 0:
                            n = -n
                        flipped = True

                # Fallback to centroid if topological search failed
                if not flipped:
                    if np.dot(n, center_pt - midpoint) < 0:
                        n = -n

                # Accumulate the normalized segment normal to the vertices
                n_norm = n / (np.linalg.norm(n) + 1e-12)
                node_normals[idx_a] += n_norm
                node_normals[idx_b] += n_norm

            # Replace KDTree vectors with the exact geometric normals for wall nodes
            diff_vec[wall_node_indices] = node_normals[wall_node_indices]

        # 3. Explicitly Normalize every vector to a magnitude of exactly 1.0
        # This is strictly required for the ModulatedGATConv attention dot-products
        norms = np.linalg.norm(diff_vec, axis=1, keepdims=True)
        wall_normal_vec = diff_vec / (norms + 1e-12)
        # --------------------------------------

        # Spatial Label Mapping via cKDTree
        y_labels = torch.zeros((len(nodes), 4), dtype=torch.float32)
        is_anchor = False
        if label_path.exists():
            try:
                cfd = np.load(label_path)
                # Map CFD coordinates to mesh nodes
                sol_points = np.stack([cfd['x'].flatten(), cfd['y'].flatten()], axis=-1)
                sol_tree = cKDTree(sol_points)
                _, idx = sol_tree.query(nodes)

                u_raw = torch.tensor(cfd['u'].flatten()[idx], dtype=torch.float32)
                v_raw = torch.tensor(cfd['v'].flatten()[idx], dtype=torch.float32)
                p_raw = torch.tensor(cfd['p'].flatten()[idx], dtype=torch.float32)

                # Robust extraction for mu with a Newtonian fallback for older datasets
                if 'mu' in cfd:
                    mu_raw = torch.tensor(cfd['mu'].flatten()[idx], dtype=torch.float32)
                else:
                    mu_raw = torch.full_like(u_raw, ref_mu)

                # Normalize
                u_nd, v_nd = u_raw / u_ref, v_raw / u_ref
                p_nd = p_raw / p_ref_scale

                # Non-dimensionalize viscosity by the reference viscosity
                mu_nd = mu_raw / ref_mu

                # Stack the 4 channels
                y_labels = torch.stack([u_nd, v_nd, p_nd, mu_nd], dim=1)
                is_anchor = True
            except Exception as e:
                print(f"Error mapping labels {filename}: {e}")

        # Inlet BC Calculation
        u_inlet_bc = torch.zeros((len(nodes), 2), dtype=torch.float32)
        mu_inlet_bc = torch.zeros(len(nodes), dtype=torch.float32)  # NEW

        if mask_inlet.any():
            inlet_indices = torch.where(mask_inlet)[0]
            y_coords = nodes[inlet_indices, 1]
            y_center = np.mean(y_coords)
            R = (np.max(y_coords) - np.min(y_coords)) / 2.0
            u_max = 1.5 * u_ref
            r = np.abs(y_coords - y_center)

            # 1. Kinematics (Velocity Profile)
            profile_mag = u_max * (1 - (r ** 2 / (R ** 2 + 1e-12)))
            u_inlet_bc[inlet_indices, 0] = torch.tensor(profile_mag / u_ref, dtype=torch.float32)

            # 2. Rheology (Analytical Viscosity Profile)
            if self.phys_cfg.viscosity_model == "carreau":
                # Analytical derivative of parabolic flow: |du/dy| = |-2 * u_max * r / R^2|
                gamma_dot = np.abs(-2.0 * u_max * r / (R ** 2 + 1e-12))

                # Non-dimensionalize shear rate
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
                mu_inlet_bc[inlet_indices] = 1.0  # Newtonian baseline

        # --- Wall BC Calculation (Viscosity) ---
        mu_wall_bc = torch.zeros(len(nodes), dtype=torch.float32)
        if mask_wall.any():
            if self.phys_cfg.viscosity_model == "carreau":
                # Non-dimensionalize mu_inf by mu_ref
                mu_inf_nd = self.phys_cfg.mu_inf / self.phys_cfg.mu_ref
                mu_wall_bc[mask_wall] = torch.tensor(mu_inf_nd, dtype=torch.float32)
            else:
                mu_wall_bc[mask_wall] = 1.0

        # Graph Assembly
        nodes_nd = nodes / d_bar
        sdf_nd = dist_raw.reshape(-1, 1) / d_bar
        shear_pot = torch.log(1.0 + 1.0 / (torch.tensor(sdf_nd, dtype=torch.float32) + 1e-6))

        if "triangle" not in mesh.cells_dict: return
        tri_nodes = mesh.cells_dict["triangle"]
        edges = np.unique(np.sort(np.vstack([
            tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]
        ]), axis=1), axis=0)
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)

        node_type = torch.zeros((len(nodes), 4), dtype=torch.float32)

        # Default all nodes to Internal (Index 0)
        node_type[:, 0] = 1.0

        # Assign Inlet (Index 1)
        node_type[mask_inlet, 0] = 0.0
        node_type[mask_inlet, 1] = 1.0

        # Assign Outlet (Index 2)
        node_type[mask_outlet, 0] = 0.0
        node_type[mask_outlet, 2] = 1.0

        # Assign Wall (Index 3)
        node_type[mask_wall, 0] = 0.0
        node_type[mask_wall, 3] = 1.0

        # Add a one-hot or scalar for the physics mode
        is_non_newt = 1.0 if self.phys_cfg.viscosity_model == "carreau" else 0.0

        # --- GENERALIZED POISSEUILLE PRIOR ---
        # Around line 258, where you calculate u_prior_mag
        u_max_nd = 1.5
        sdf_tensor = torch.tensor(sdf_nd, dtype=torch.float32)
        u_prior_mag = 4.0 * u_max_nd * sdf_tensor * (1.0 - sdf_tensor)

        # Derivative of the parabolic profile yields a linear shear rate prior
        # The domain radius R_nd is 0.5 (since sdf_nd goes from 0 to 0.5 to 0)
        r_nd = torch.abs(0.5 - sdf_tensor)
        gamma_dot_prior_nd = torch.abs(4.0 * u_max_nd * (1.0 - 2.0 * sdf_tensor))

        lambda_nd = self.phys_cfg.lam * (u_ref / d_bar)
        shear_term = 1.0 + (lambda_nd * gamma_dot_prior_nd.numpy()) ** self.phys_cfg.a
        power = (self.phys_cfg.n - 1.0) / self.phys_cfg.a

        mu_inf_nd = self.phys_cfg.mu_inf / self.phys_cfg.mu_ref
        mu_0_nd = self.phys_cfg.mu_0 / self.phys_cfg.mu_ref

        mu_prior = mu_inf_nd + (mu_0_nd - mu_inf_nd) * (shear_term ** power)
        mu_prior_tensor = torch.tensor(mu_prior, dtype=torch.float32)

        # Update x_tensor concatenation to include mu_prior_tensor

        # We assume the primary background flow is in the +x direction
        u_prior_x = torch.clamp(u_prior_mag, min=0.0)
        v_prior_y = torch.zeros_like(u_prior_x)

        # Stack into a 2D vector
        uv_prior = torch.cat([u_prior_x, v_prior_y], dim=1)
        # --------------------------------------

        # Assembly: Concatenate to the x_tensor
        x_tensor = torch.cat([
            torch.tensor(nodes_nd, dtype=torch.float32),
            torch.tensor(sdf_nd, dtype=torch.float32),
            shear_pot,
            torch.tensor(wall_normal_vec, dtype=torch.float32),
            node_type,
            torch.full((len(nodes), 1), is_non_newt, dtype=torch.float32),
            uv_prior,
            mu_prior_tensor
        ], dim=1)

        # Precompute WLS operators for rapid gradient calculations during training
        V, W, M_inv = self._precompute_wls(edge_index, len(nodes), x_tensor)

        data = Data(
            x=x_tensor, edge_index=edge_index, y=y_labels,
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