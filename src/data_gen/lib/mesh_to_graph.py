"""
Kinematics/2 mesh→graph pipeline: **2D planar** vessel lumen only (in-plane CFD / graphs).
Velocity priors follow 2D plane Poiseuille and mass scaling; 3D pipe flow is not modeled.
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
from typing import Optional
import meshio
from pathlib import Path
from scipy.spatial import KDTree, cKDTree
from torch_geometric.data import Data
from tqdm import tqdm

# Running `python .../mesh_to_graph.py` sets __package__ to None; ensure project root is importable.
if __name__ == "__main__":
    _proj = Path(__file__).resolve().parents[3]
    _ps = str(_proj)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from src.config import NodeFeat, PhysicsConfig, VesselConfig
from src.data_gen.lib.mesh_wls import gmsh_line_boundary_masks, precompute_wls_operators
from src.data_gen.lib.graph_velocity_priors import (
    mass_conserving_umax_nd,
    smooth_width_nd_on_edges,
    width_nd_to_radius_nd,
)
from src.utils.paths import (
    get_project_root,
    migrate_legacy_final_n_subdir,
    migrate_legacy_vessel_meshes,
)
from src.utils.channel_schema import (
    KINE_X_SCHEMA,
    KINE_Y_SCHEMA,
    attach_channel_metadata,
)
from src.utils.units import MESH_UNIT_M, assert_mesh_unit
from src.utils.kinematics_geometry import attach_geometry_metadata, vessel_index_from_stem


def _clip_wss_magnitude_quantile(
    wss_mag: torch.Tensor, mask_wall: torch.Tensor, q: float = 0.995
) -> torch.Tensor:
    """Cap wall WSS magnitudes at the q-quantile of wall values (robust to WLS gradient spikes)."""
    mw = mask_wall.view(-1).bool()
    if not mw.any():
        return wss_mag
    w = wss_mag[mw]
    if w.numel() == 0:
        return wss_mag
    cap = torch.quantile(w, q)
    out = wss_mag.clone()
    out[mw] = torch.clamp(wss_mag[mw], max=cap)
    return out


def assemble_kinematics_graph_data(
    *,
    x_tensor: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y_labels: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    mask_wall: torch.Tensor,
    is_anchor: bool,
    d_bar: float,
    u_ref: float,
    u_prior: torch.Tensor,
    mu_prior: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    M_inv: torch.Tensor,
    G_x: torch.Tensor,
    G_y: torch.Tensor,
) -> Data:
    """Build the Kinematics/2 ``Data`` object written to ``*.pt`` (single code path for tests + ``process_file``).

    Kinematics/2 graphs store sparse WLS gradient operators ``G_x`` / ``G_y`` for physics kernels.
    """
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_labels,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        mask_wall=mask_wall,
        is_anchor=torch.tensor([is_anchor], dtype=torch.bool),
        d_bar=torch.tensor([d_bar], dtype=torch.float32),
        u_ref=torch.tensor([u_ref], dtype=torch.float32),
        u_inlet_bc=u_prior.view(-1, 1),
        mu_inlet_bc=mu_prior.view(-1, 1),
        mu_wall_bc=mu_prior.view(-1, 1),
        V=V,
        W=W,
        M_inv=M_inv,
        G_x=G_x,
        G_y=G_y,
    )
    return attach_channel_metadata(
        data,
        x_schema=KINE_X_SCHEMA,
        y_schema=KINE_Y_SCHEMA,
        mask_wall=mask_wall,
    )


class MeshToGraphComplete:
    def __init__(
        self,
        phase="kinematics",
        n_subdir: str = None,
        raw_dir=None,
        label_dir=None,
        proc_dir=None,
        rheology: Optional[str] = None,
    ):
        self.root = get_project_root()
        self.vessel_cfg = VesselConfig(phase=phase)
        inferred_rheology = rheology
        if inferred_rheology is None:
            leaf = (n_subdir or "").strip().lower()
            if leaf in {"newtonian", "carreau"}:
                inferred_rheology = leaf
        self.phys_cfg = PhysicsConfig(phase=phase, rheology=inferred_rheology)

        # Resolve Raw Dir
        if raw_dir:
            self.raw_dir = Path(raw_dir)
        else:
            self.raw_dir = self.root / self.vessel_cfg.mesh_input_dir
            migrate_legacy_vessel_meshes(self.raw_dir)

        # Resolve Label Dir
        if label_dir:
            self.label_dir = Path(label_dir)
        else:
            self.label_dir = self.root / self.vessel_cfg.output_dir
        if n_subdir:
            self.label_dir = self.label_dir / n_subdir

        # Resolve Processed Dir
        if proc_dir:
            self.proc_dir = Path(proc_dir)
        else:
            self.proc_dir = self.root / self.vessel_cfg.graph_output_dir
        if n_subdir:
            self.proc_dir = self.proc_dir / n_subdir

        if phase == "kinematics":
            migrate_legacy_final_n_subdir(self.label_dir, n_value=self.phys_cfg.n, ext="npz")
            migrate_legacy_final_n_subdir(self.proc_dir, n_value=self.phys_cfg.n, ext="pt")

        self.proc_dir.mkdir(parents=True, exist_ok=True)


class MeshToGraph(MeshToGraphComplete):
    """
    Kinematics & Kinematics specific graph conversion logic.
    Computes kinematics and packages variables.
    """

    def __init__(
        self,
        phase: str,
        n_subdir: str = None,
        raw_dir=None,
        label_dir=None,
        proc_dir=None,
        rheology: Optional[str] = None,
    ):
        # Keep explicit path overrides for callers (benchmark/pipeline/tests) that
        # build temporary datasets outside default phase directories.
        super().__init__(
            phase=phase,
            n_subdir=n_subdir,
            raw_dir=raw_dir,
            label_dir=label_dir,
            proc_dir=proc_dir,
            rheology=rheology,
        )

    def _precompute_wls(self, edge_index, num_nodes, pos_tensor):
        """Modified to accept pos_tensor directly so it can run before x_tensor assembly."""
        return precompute_wls_operators(edge_index, num_nodes, pos_tensor)

    def _get_boundary_masks(self, mesh, num_nodes):
        return gmsh_line_boundary_masks(mesh, num_nodes, dict(self.vessel_cfg.TAGS))

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
        meta = None
        if json_path.exists():
            with open(json_path, 'r') as f:
                meta = json.load(f)
                d_bar = meta.get('d_bar')
        if d_bar is None or (isinstance(d_bar, (int, float)) and float(d_bar) <= 0):
            d_bar = float(np.max(np.ptp(nodes, axis=0)) + 1e-6)

        if meta is None:
            raise ValueError(
                f"{stem}: missing sidecar JSON {json_path}; flow priors and wall normal orientation require centerline metadata."
            )
        assert_mesh_unit(meta, MESH_UNIT_M, stem=stem, builder="MeshToGraph")
        spine_pts_nd = meta.get("centerline_pts")
        spine_tangents = meta.get("centerline_tangents")
        if spine_pts_nd is None or spine_tangents is None:
            raise ValueError(
                f"{stem}: JSON must define centerline_pts and centerline_tangents "
                "(regenerate meshes with the current vessel_generator)."
            )
        spine_pts_nd = np.asarray(spine_pts_nd, dtype=np.float64)
        spine_tangents = np.asarray(spine_tangents, dtype=np.float64)
        if not (
            spine_pts_nd.ndim == 2
            and spine_pts_nd.shape[1] == 2
            and spine_tangents.shape == spine_pts_nd.shape
            and spine_pts_nd.shape[0] > 0
        ):
            raise ValueError(
                f"{stem}: invalid centerline_pts / centerline_tangents "
                f"(got pts {getattr(spine_pts_nd, 'shape', None)}, tangents {getattr(spine_tangents, 'shape', None)})."
            )
        spine_pts = spine_pts_nd * float(d_bar)
        spine_tree_wall = cKDTree(spine_pts)

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

            for line in wall_lines:
                idx_a, idx_b = line[0], line[1]
                pt_a, pt_b = nodes[idx_a], nodes[idx_b]

                # Tangent vector
                dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]

                # Orthogonal normal vector (-dy, dx)
                n = np.array([-dy, dx])

                # Ensure the normal points towards the local lumen center (nearest centerline sample).
                midpoint = (pt_a + pt_b) / 2.0
                _, nearest_spine_idx = spine_tree_wall.query(midpoint)
                local_center = spine_pts[nearest_spine_idx]
                if np.dot(n, local_center - midpoint) < 0:
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
                wss_mag = _clip_wss_magnitude_quantile(wss_mag, mask_wall, q=0.995)
                y_labels = torch.stack([u_nd, v_nd, p_nd, mu_nd, wss_mag], dim=1)
                is_anchor = True
            except Exception as e:
                print(f"Error mapping labels: {e}")

        # --------------------------------------------------------------------------
        #  Sparse WLS: G_x, G_y — then hydraulic width and Poiseuille-style velocity priors
        # --------------------------------------------------------------------------
        W_V = W.unsqueeze(1) * V  # Shape: (E, 5)
        M_inv_row = M_inv[row]  # Shape: (E, 5, 5)

        coeffs = torch.bmm(M_inv_row, W_V.unsqueeze(2)).squeeze(2)  # Shape: (E, 5)
        cx = coeffs[:, 0]
        cy = coeffs[:, 1]

        N = len(nodes)

        diag_cx = torch.zeros(N, dtype=torch.float32).scatter_add_(0, row, cx)
        diag_cy = torch.zeros(N, dtype=torch.float32).scatter_add_(0, row, cy)

        diag_indices = torch.arange(N, dtype=torch.long).unsqueeze(0).repeat(2, 1)

        idx_x = torch.cat([edge_index, diag_indices], dim=1)
        val_x = torch.cat([cx, -diag_cx], dim=0)

        idx_y = torch.cat([edge_index, diag_indices], dim=1)
        val_y = torch.cat([cy, -diag_cy], dim=0)

        G_x = torch.sparse_coo_tensor(idx_x, val_x, size=(N, N)).coalesce()
        G_y = torch.sparse_coo_tensor(idx_y, val_y, size=(N, N)).coalesce()

        # --------------------------------------------------------------------------
        #  Local width D(x) (sphere tracing; full lumen width along inward normal ray)
        # --------------------------------------------------------------------------
        d_bar_f = float(d_bar)
        width_nd = torch.zeros(N, 1, dtype=torch.float32)
        t_march = sdf_tensor.clone() + 0.05
        active = torch.ones(N, dtype=torch.bool)
        for _ in range(30):
            if not active.any():
                break
            idx = torch.nonzero(active, as_tuple=False).view(-1)
            if idx.numel() == 0:
                break
            probe = pos_nd_tensor[idx] + t_march[idx] * wall_normal_vec[idx]
            probe_np = (probe * d_bar_f).detach().cpu().numpy()
            dist, _ = tree_wall.query(probe_np)
            dist_nd = torch.tensor(dist / d_bar_f, dtype=torch.float32, device=pos_nd_tensor.device).view(-1, 1)
            hit_mask = (dist_nd < 0.02).squeeze(-1)
            hit_idx = idx[hit_mask]
            width_nd[hit_idx] = sdf_tensor[hit_idx] + t_march[hit_idx]
            active[hit_idx] = False
            still_idx = idx[~hit_mask]
            if still_idx.numel() == 0:
                break
            dist_still = dist_nd[~hit_mask]
            t_march[still_idx] = t_march[still_idx] + torch.clamp(dist_still, min=0.01)

        width_nd[width_nd.squeeze(-1) < 1e-6] = 1.0
        width_nd = smooth_width_nd_on_edges(width_nd, edge_index, N)
        width_nd = torch.clamp(width_nd, min=1e-6)

        # Flow tangent from inward wall normals, oriented with vessel centerline (from mesh JSON).
        n_x = wall_normal_vec[:, 0]
        n_y = wall_normal_vec[:, 1]
        t_x_unoriented = -n_y
        t_y_unoriented = n_x

        spine_tree = cKDTree(spine_pts_nd)
        _, nearest_spine_idx = spine_tree.query(pos_nd_tensor.detach().cpu().numpy())
        local_flow_vec = torch.tensor(
            spine_tangents[nearest_spine_idx], dtype=torch.float32, device=pos_nd_tensor.device
        )

        dot_prod = t_x_unoriented * local_flow_vec[:, 0] + t_y_unoriented * local_flow_vec[:, 1]
        orientation = torch.sign(dot_prod)
        orientation = torch.where(orientation == 0, torch.ones_like(orientation), orientation)

        flow_dir_x = t_x_unoriented * orientation
        flow_dir_y = t_y_unoriented * orientation
        # Explicit unit direction: avoids FEM/interpolation drift from warping |u_prior| (mass balance).
        flow_norm = torch.sqrt(flow_dir_x ** 2 + flow_dir_y ** 2).clamp_min(1e-8)
        flow_dir_x = flow_dir_x / flow_norm
        flow_dir_y = flow_dir_y / flow_norm

        # Width-adaptive Poiseuille radius (R = full width / 2) and 2D mass-style u_max ~ 1/R
        R_nd = width_nd_to_radius_nd(width_nd)
        u_max_nd = mass_conserving_umax_nd(R_nd)
        # 2. Safely clamp SDF to prevent centerline dipping due to ray-tracing vs SDF misalignment
        safe_sdf = torch.minimum(sdf_tensor.squeeze().clamp_min(0.0), R_nd)
        r_nd = R_nd - safe_sdf
        u_prior_mag = torch.clamp(u_max_nd * (1.0 - (r_nd ** 2 / (R_nd ** 2 + 1e-12))), min=0.0)

        # Mass-conserving 2D Poiseuille-style profile along local flow tangent.
        u_prior = u_prior_mag * flow_dir_x
        v_prior = u_prior_mag * flow_dir_y

        # Viscosity prior + WSS (wall)
        gamma_dot_prior = torch.abs(-2.0 * u_max_nd * r_nd / (R_nd ** 2 + 1e-12))

        if self.phys_cfg.viscosity_model == "newtonian":
            mu_prior = torch.ones(N, dtype=torch.float32)
        else:
            # Carreau formulation for Kinematics / Biochem
            lambda_nd = self.phys_cfg.lam * (u_ref / d_bar)
            mu_prior = (self.phys_cfg.mu_inf / ref_mu) + (
                        (self.phys_cfg.mu_0 / ref_mu) - (self.phys_cfg.mu_inf / ref_mu)) * \
                       (1.0 + (lambda_nd * gamma_dot_prior) ** self.phys_cfg.a) ** (
                               (self.phys_cfg.n - 1.0) / self.phys_cfg.a)

        wss_prior = (mu_prior * gamma_dot_prior) * mask_wall.float()

        grad_w_x = torch.sparse.mm(G_x, width_nd)
        grad_w_y = torch.sparse.mm(G_y, width_nd)
        width_d1 = grad_w_x * flow_dir_x.unsqueeze(1) + grad_w_y * flow_dir_y.unsqueeze(1)
        grad2_w_x = torch.sparse.mm(G_x, width_d1)
        grad2_w_y = torch.sparse.mm(G_y, width_d1)
        width_d2 = grad2_w_x * flow_dir_x.unsqueeze(1) + grad2_w_y * flow_dir_y.unsqueeze(1)

        # --- Final Assembly ---
        x_tensor = torch.cat([
            pos_nd_tensor, sdf_tensor, torch.abs(1.0 - 2.0 * sdf_tensor),  # Pos, SDF, ShearPot
            wall_normal_vec,
            torch.zeros((len(nodes), 4)),  # Node Type (Placeholder)
            torch.full((len(nodes), 1), 1.0 if self.phys_cfg.viscosity_model == "carreau" else 0.0),
            u_prior.view(-1, 1), v_prior.view(-1, 1),  # Vectorized UV prior
            mu_prior.view(-1, 1), wss_prior.view(-1, 1),
            width_nd, width_d1, width_d2,
        ], dim=1)
        assert x_tensor.shape[1] == NodeFeat.WIDTH_D2.stop, "Phase-1 node feature width must match NodeFeat"

        data = assemble_kinematics_graph_data(
            x_tensor=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_labels=y_labels,
            mask_inlet=mask_inlet,
            mask_outlet=mask_outlet,
            mask_wall=mask_wall,
            is_anchor=is_anchor,
            d_bar=d_bar,
            u_ref=u_ref,
            u_prior=u_prior,
            mu_prior=mu_prior,
            V=V,
            W=W,
            M_inv=M_inv,
            G_x=G_x,
            G_y=G_y,
        )
        data.graph_stem = stem
        if meta is not None and meta.get("level") is not None:
            data.geometry_level = torch.tensor([int(meta["level"])], dtype=torch.int8)
        else:
            attach_geometry_metadata(data, mesh_input_dir=self.raw_dir, stem=stem)
        idx = vessel_index_from_stem(stem)
        if idx is not None:
            data.config_id = int(idx)

        torch.save(data, self.proc_dir / f"{stem}.pt")

    def run(self, max_files=None) -> None:
        """Convert ``.msh`` files under ``raw_dir`` to graph ``.pt`` (with labels if CFD ``.npz`` exists).

        Clears all existing ``*.pt`` files in ``proc_dir`` first so each run fully replaces
        graph outputs (no stale graphs if meshes were removed or indices changed).

        Args:
            max_files: If set, only the first ``max_files`` meshes (sorted by name) are converted.
        """
        self.proc_dir.mkdir(parents=True, exist_ok=True)
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


class MeshToGraphComplete(MeshToGraph):
    """Backward-compatible alias for callers still importing MeshToGraphComplete."""


def build_mesh_converter(
    phase: str = "kinematics",
    *,
    is_non_newtonian: Optional[bool] = None,
    **kwargs,
):
    """Return Kinematics/2 or Biochem graph builder (sparse operators + biochem tensors for Biochem).

    If ``is_non_newtonian`` is True, the Phase-3 pipeline is selected regardless of ``phase``.
    If False, Kinematics/2-style graphs are built. If None (default), ``phase`` alone decides
    (``biochem`` / ``biochem_anchors`` / ``biochem_patients`` / ``biochem_mix`` → Biochem).
    """
    t = (phase or "kinematics").lower()
    if is_non_newtonian is True:
        from src.data_gen.lib.mesh_to_graph_biochem import MeshToGraphPhase3

        return MeshToGraphPhase3(**kwargs)
    if is_non_newtonian is False:
        return MeshToGraph(phase=phase, **kwargs)
    if t in ("biochem", "biochem_anchors", "biochem_patients", "biochem_mix"):
        from src.data_gen.lib.mesh_to_graph_biochem import MeshToGraphPhase3

        return MeshToGraphPhase3(**kwargs)
    return MeshToGraph(phase=phase, **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert vessel meshes into graph tensors for kinematics/biochem pipelines."
    )
    parser.add_argument(
        "--phase",
        choices=("kinematics", "biochem"),
        default="kinematics",
        help="Pipeline target. Uses modern names (no numeric phase selection).",
    )
    parser.add_argument(
        "--rheology",
        choices=("newtonian", "carreau"),
        default=None,
        help="Kinematics-only viscosity model override (defaults to config behavior).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_kwargs = {}
    if args.phase == "kinematics" and args.rheology is not None:
        build_kwargs["rheology"] = args.rheology
    processor = build_mesh_converter(phase=args.phase, **build_kwargs)
    processor.run()