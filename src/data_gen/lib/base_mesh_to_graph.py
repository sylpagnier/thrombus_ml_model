import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import meshio
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from tqdm import tqdm

from src.config import PhysicsConfig, VesselConfig
from src.utils.paths import get_project_root

from .mesh_wls import gmsh_line_boundary_masks, precompute_wls_operators

logger = logging.getLogger(__name__)


class BaseMeshToGraph(ABC):
    """Shared mesh -> graph conversion pipeline used by Kinematics/2 and Biochem."""

    def __init__(self, phase: str, n_subdir: str = None, raw_dir=None, label_dir=None, proc_dir=None):
        """
        Base class for converting .npz + .msh into PyG graphs.
        :param phase: "kinematics" or "kinematics"
        :param n_subdir: Optional n-subdirectory (e.g., "n_0.800") used for
                         continuation/final Kinematics datasets; redirects label/proc paths.
        """
        self.root = get_project_root()
        self.cfg = VesselConfig(phase=phase)
        self.vessel_cfg = self.cfg
        self.phys_cfg = PhysicsConfig(phase=phase)
        self.phase = phase

        # Apply continuation subdir logic if provided.
        label_base = Path(label_dir) if label_dir else (self.root / self.vessel_cfg.output_dir)
        proc_base = Path(proc_dir) if proc_dir else (self.root / self.vessel_cfg.graph_output_dir)
        if n_subdir:
            label_base = label_base / n_subdir
            proc_base = proc_base / n_subdir

        self.raw_dir = Path(raw_dir) if raw_dir else (self.root / self.vessel_cfg.mesh_input_dir)
        self.label_dir = label_base
        self.proc_dir = proc_base
        self.proc_dir.mkdir(parents=True, exist_ok=True)

        # We need a reference geometry to define continuous boundary nodes.
        mesh_root = getattr(self.cfg, "vessel_mesh_dir", self.vessel_cfg.mesh_input_dir)
        mesh_root = Path(mesh_root)
        if not mesh_root.is_absolute():
            mesh_root = self.root / mesh_root
        try:
            self.ref_mesh = meshio.read(mesh_root / "vessel_0000.msh")
        except FileNotFoundError:
            self.ref_mesh = None
            logger.warning("vessel_0000.msh not found. Ensure it exists for node tagging.")

    def _precompute_wls(self, edge_index, num_nodes, pos_tensor):
        return precompute_wls_operators(edge_index, num_nodes, pos_tensor)

    def _get_boundary_masks(self, mesh, num_nodes):
        return gmsh_line_boundary_masks(mesh, num_nodes, dict(self.vessel_cfg.TAGS))

    def _get_mu_scale(self) -> float:
        return float(self.phys_cfg.mu_ref)

    def _extract_line_segments(self, mesh, tag: int):
        line_segments = []
        try:
            if "line" in mesh.cells_dict:
                l_cells = mesh.cells_dict["line"]
                l_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
            elif hasattr(mesh, "get_cells_type"):
                l_cells = mesh.get_cells_type("line")
                l_tags = mesh.get_cell_data("gmsh:physical", "line")
            else:
                l_cells, l_tags = [], []
            for i, line_tag in enumerate(l_tags):
                if line_tag == tag:
                    line_segments.append(l_cells[i])
        except Exception:
            return []
        return line_segments

    def _build_wall_orientation_context(self, nodes, mask_inlet, mask_outlet, mask_wall):
        interior_mask = ~(mask_wall.numpy() | mask_inlet.numpy() | mask_outlet.numpy())
        center_pt = np.mean(nodes[interior_mask], axis=0) if interior_mask.any() else np.mean(nodes, axis=0)
        return {"center_pt": center_pt}

    def _orient_wall_normal(self, normal, midpoint, orientation_context):
        center_pt = orientation_context["center_pt"]
        if np.dot(normal, center_pt - midpoint) < 0:
            normal = -normal
        return normal

    def _compute_outlet_normals(self, mesh, nodes, mask_outlet):
        return torch.zeros((len(nodes), 2), dtype=torch.float32)

    def _compute_wall_normals_and_sdf(self, mesh, nodes, mask_inlet, mask_outlet, mask_wall):
        wall_node_indices = np.where(mask_wall.numpy())[0]
        if len(wall_node_indices) == 0:
            return None, None

        wall_pts = nodes[wall_node_indices]
        wall_tree = cKDTree(wall_pts)
        dist_raw, indices_wall = wall_tree.query(nodes)
        nearest_wall_pts = wall_pts[indices_wall]
        diff_vec = nodes - nearest_wall_pts

        wall_tag = self.vessel_cfg.TAGS["Walls"]
        wall_lines = self._extract_line_segments(mesh, wall_tag)
        if wall_lines:
            node_normals = np.zeros((len(nodes), 2), dtype=np.float32)
            orient_ctx = self._build_wall_orientation_context(nodes, mask_inlet, mask_outlet, mask_wall)
            for line in wall_lines:
                idx_a, idx_b = line[0], line[1]
                pt_a, pt_b = nodes[idx_a], nodes[idx_b]
                dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]
                normal = np.array([-dy, dx], dtype=np.float32)
                midpoint = (pt_a + pt_b) / 2.0
                normal = self._orient_wall_normal(normal, midpoint, orient_ctx)
                normal = normal / (np.linalg.norm(normal) + 1e-12)
                node_normals[idx_a] += normal
                node_normals[idx_b] += normal
            diff_vec[wall_node_indices] = node_normals[wall_node_indices]

        norms = np.linalg.norm(diff_vec, axis=1, keepdims=True)
        wall_normal_vec = torch.tensor(diff_vec / (norms + 1e-12), dtype=torch.float32)
        return dist_raw, wall_normal_vec

    def _map_ground_truth(self, nodes, label_path, edge_index, wall_normal_vec, mask_wall, u_ref, p_ref_scale, mu_scale, V, W, M_inv):
        y_labels = torch.zeros((len(nodes), 5), dtype=torch.float32)
        is_anchor = False

        if not label_path.exists():
            return y_labels, is_anchor

        try:
            cfd = np.load(label_path)
            sol_points = np.stack([cfd["x"].flatten(), cfd["y"].flatten()], axis=-1)
            sol_tree = cKDTree(sol_points)
            _, idx = sol_tree.query(nodes)

            u_raw = torch.tensor(cfd["u"].flatten()[idx], dtype=torch.float32)
            v_raw = torch.tensor(cfd["v"].flatten()[idx], dtype=torch.float32)
            u_raw[mask_wall] = 0.0
            v_raw[mask_wall] = 0.0

            u_nd, v_nd = u_raw / u_ref, v_raw / u_ref
            p_nd = torch.tensor(cfd["p"].flatten()[idx] / p_ref_scale, dtype=torch.float32)
            if "mu" in cfd:
                mu_nd = torch.tensor(cfd["mu"].flatten()[idx] / mu_scale, dtype=torch.float32)
            else:
                mu_nd = torch.ones_like(u_nd)

            row, col = edge_index
            df_u, df_v = u_nd[col] - u_nd[row], v_nd[col] - v_nd[row]
            sum_w_v_du = torch.zeros((len(nodes), 5)).scatter_add_(
                0, row.unsqueeze(1).expand(-1, 5), W.unsqueeze(1) * V * df_u.unsqueeze(1)
            )
            sum_w_v_dv = torch.zeros((len(nodes), 5)).scatter_add_(
                0, row.unsqueeze(1).expand(-1, 5), W.unsqueeze(1) * V * df_v.unsqueeze(1)
            )

            grad_u = torch.bmm(M_inv, sum_w_v_du.unsqueeze(2)).squeeze()
            grad_v = torch.bmm(M_inv, sum_w_v_dv.unsqueeze(2)).squeeze()

            tau_xx = 2.0 * mu_nd * grad_u[:, 0]
            tau_yy = 2.0 * mu_nd * grad_v[:, 1]
            tau_xy = mu_nd * (grad_u[:, 1] + grad_v[:, 0])

            n_x, n_y = wall_normal_vec[:, 0], wall_normal_vec[:, 1]
            t_x = tau_xx * n_x + tau_xy * n_y
            t_y = tau_xy * n_x + tau_yy * n_y
            wss_mag = torch.sqrt(t_x ** 2 + t_y ** 2) * mask_wall.float()

            y_labels = torch.stack([u_nd, v_nd, p_nd, mu_nd, wss_mag], dim=1)
            is_anchor = True
        except Exception as e:
            print(f"Error mapping labels: {e}")

        return y_labels, is_anchor

    @abstractmethod
    def _build_priors(self, context):
        raise NotImplementedError

    @abstractmethod
    def _assemble_node_features(self, context, priors):
        raise NotImplementedError

    @abstractmethod
    def _build_data_object(self, context, priors, x_tensor, y_labels, is_anchor):
        raise NotImplementedError

    def process_file(self, filename):
        stem = Path(filename).stem
        msh_path = self.raw_dir / filename
        json_path = self.raw_dir / f"{stem}.json"
        label_path = self.label_dir / f"{stem}.npz"

        if not msh_path.exists():
            return

        try:
            mesh = meshio.read(msh_path)
            nodes = mesh.points[:, :2]
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            return

        all_tris = []
        if "triangle" in mesh.cells_dict:
            all_tris.append(mesh.cells_dict["triangle"])
        elif hasattr(mesh, "get_cells_type"):
            tc = mesh.get_cells_type("triangle")
            if len(tc) > 0:
                all_tris.append(tc)
        if not all_tris:
            return
        tri_nodes = np.vstack(all_tris)

        d_bar = None
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                d_bar = json.load(f).get("d_bar")
        if d_bar is None or float(d_bar) <= 0.0:
            print(f"Skipping {filename}: missing/invalid d_bar in metadata.")
            return

        mask_inlet, mask_outlet, mask_wall = self._get_boundary_masks(mesh, len(nodes))
        outlet_normal = self._compute_outlet_normals(mesh, nodes, mask_outlet)

        dist_raw, wall_normal_vec = self._compute_wall_normals_and_sdf(mesh, nodes, mask_inlet, mask_outlet, mask_wall)
        if dist_raw is None:
            return

        mu_scale = self._get_mu_scale()
        u_ref = self.phys_cfg.get_u_ref(d_bar)
        p_ref_scale = self.phys_cfg.get_p_ref(u_ref)

        nodes_nd = nodes / d_bar
        pos_nd_tensor = torch.tensor(nodes_nd, dtype=torch.float32)
        sdf_tensor = torch.clamp(torch.tensor(dist_raw / d_bar, dtype=torch.float32).view(-1, 1), min=1e-6)

        edges = np.unique(
            np.sort(np.vstack([tri_nodes[:, [0, 1]], tri_nodes[:, [1, 2]], tri_nodes[:, [2, 0]]]), axis=1), axis=0
        )
        edge_index = torch.tensor(np.hstack([edges.T, edges[:, [1, 0]].T]), dtype=torch.long)
        row, col = edge_index
        edge_delta = pos_nd_tensor[row] - pos_nd_tensor[col]
        edge_attr = torch.cat([edge_delta, torch.linalg.norm(edge_delta, dim=1, keepdim=True)], dim=1)

        V, W, M_inv = self._precompute_wls(edge_index, len(nodes), pos_nd_tensor)
        M_inv = M_inv.squeeze(1)
        y_labels, is_anchor = self._map_ground_truth(
            nodes,
            label_path,
            edge_index,
            wall_normal_vec,
            mask_wall,
            u_ref,
            p_ref_scale,
            mu_scale,
            V,
            W,
            M_inv,
        )

        context = {
            "stem": stem,
            "mesh": mesh,
            "nodes": nodes,
            "d_bar": d_bar,
            "u_ref": u_ref,
            "mu_scale": mu_scale,
            "mask_inlet": mask_inlet,
            "mask_outlet": mask_outlet,
            "mask_wall": mask_wall,
            "outlet_normal": outlet_normal,
            "pos_nd_tensor": pos_nd_tensor,
            "sdf_tensor": sdf_tensor,
            "wall_normal_vec": wall_normal_vec,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "row": row,
            "col": col,
            "V": V,
            "W": W,
            "M_inv": M_inv,
            "num_nodes": len(nodes),
        }
        priors = self._build_priors(context)
        x_tensor = self._assemble_node_features(context, priors)
        data = self._build_data_object(context, priors, x_tensor, y_labels, is_anchor)

        if not isinstance(data, Data):
            raise TypeError("Expected _build_data_object() to return torch_geometric.data.Data")
        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self, max_files=None, clear_existing=True):
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        if clear_existing:
            for stale in self.proc_dir.glob("*.pt"):
                try:
                    stale.unlink()
                except OSError as e:
                    print(f"Warning: could not remove {stale}: {e}")
        files = sorted([f for f in os.listdir(self.raw_dir) if f.endswith(".msh")])
        if max_files is not None:
            files = files[: int(max_files)]
        for f in tqdm(files):
            self.process_file(f)
