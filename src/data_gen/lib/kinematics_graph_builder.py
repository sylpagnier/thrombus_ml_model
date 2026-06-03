"""Build trainer-compatible kinematics graphs (``KINE_X_SCHEMA`` / ``KINE_Y_SCHEMA``).

Shared between synthetic ``MeshToGraph`` and COMSOL patient re-extraction so Stage-A
``train_kinematics_predictor`` sees the same node layout, priors, and label semantics.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import meshio
import numpy as np
import torch
from scipy.spatial import KDTree, cKDTree
from torch_geometric.data import Data

from src.config import NodeFeat, PhysicsConfig, VesselConfig
from src.data_gen.lib.centerline_utils import resolve_centerline_nd
from src.data_gen.lib.mesh_to_graph import _clip_wss_magnitude_quantile, assemble_kinematics_graph_data
from src.data_gen.lib.node_feature_assembly import build_kinematics_node_x_tensor
from src.utils.kinematics_geometry import attach_geometry_metadata
from src.utils.units import d_bar_si_from_sidecar


def wall_normals_and_sdf_mesh_to_graph_style(
    mesh: meshio.Mesh,
    nodes_si: np.ndarray,
    *,
    mask_wall: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    d_bar_si: float,
    centerline_pts_si: Optional[np.ndarray] = None,
    wall_tag: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gmsh line-segment wall normals + KD-tree SDF (matches ``MeshToGraph``)."""
    if wall_tag is None:
        wall_tag = int(VesselConfig(phase="kinematics").TAGS["Walls"])

    nodes = np.asarray(nodes_si, dtype=np.float64)
    wall_idx = np.where(mask_wall.detach().cpu().numpy())[0]
    if len(wall_idx) == 0:
        raise ValueError("wall_normals_and_sdf: empty mask_wall")

    wall_pts = nodes[wall_idx]
    tree_wall = KDTree(wall_pts)
    dist_raw, indices_wall = tree_wall.query(nodes)
    nearest_wall_pts = wall_pts[indices_wall]
    diff_vec = nodes - nearest_wall_pts

    wall_lines = []
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
            if int(tag) == int(wall_tag):
                wall_lines.append(l_cells[i])
    except Exception:
        wall_lines = []

    if wall_lines:
        node_normals = np.zeros((len(nodes), 2), dtype=np.float32)
        if centerline_pts_si is not None and len(centerline_pts_si) > 0:
            spine_tree = cKDTree(np.asarray(centerline_pts_si, dtype=np.float64))
        else:
            spine_tree = None
        interior = ~(mask_wall.numpy() | mask_inlet.numpy() | mask_outlet.numpy())
        center_pt = np.mean(nodes[interior], axis=0) if interior.any() else np.mean(nodes, axis=0)

        for line in wall_lines:
            idx_a, idx_b = int(line[0]), int(line[1])
            pt_a, pt_b = nodes[idx_a], nodes[idx_b]
            dx, dy = pt_b[0] - pt_a[0], pt_b[1] - pt_a[1]
            n = np.array([-dy, dx], dtype=np.float64)
            midpoint = (pt_a + pt_b) / 2.0
            if spine_tree is not None:
                _, nearest = spine_tree.query(midpoint)
                target = centerline_pts_si[nearest]
            else:
                target = center_pt
            if np.dot(n, target - midpoint) < 0:
                n = -n
            n_norm = n / (np.linalg.norm(n) + 1e-12)
            node_normals[idx_a] += n_norm.astype(np.float32)
            node_normals[idx_b] += n_norm.astype(np.float32)
        diff_vec[wall_idx] = node_normals[wall_idx]

    norms = np.linalg.norm(diff_vec, axis=1, keepdims=True)
    wall_normal_vec = torch.tensor(diff_vec / (norms + 1e-12), dtype=torch.float32)
    sdf_tensor = torch.clamp(
        torch.tensor(dist_raw / float(d_bar_si), dtype=torch.float32).view(-1, 1),
        min=1e-6,
    )
    return sdf_tensor, wall_normal_vec


def comsol_fields_to_kinematics_y(
    *,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    p_nd: torch.Tensor,
    mu_nd: torch.Tensor,
    wall_normal_vec: torch.Tensor,
    mask_wall: torch.Tensor,
    edge_index: torch.Tensor,
    M_inv: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    ref_mu: float,
) -> torch.Tensor:
    """``[N, 5]`` labels with WSS from COMSOL velocity (same recipe as ``MeshToGraph``)."""
    row, col = edge_index
    u_raw, v_raw = u_nd, v_nd
    df_u, df_v = u_raw[col] - u_raw[row], v_raw[col] - v_raw[row]
    sum_W_V_du = torch.zeros((u_nd.shape[0], 5)).scatter_add_(
        0, row.unsqueeze(1).expand(-1, 5), W.unsqueeze(1) * V * df_u.unsqueeze(1)
    )
    sum_W_V_dv = torch.zeros((u_nd.shape[0], 5)).scatter_add_(
        0, row.unsqueeze(1).expand(-1, 5), W.unsqueeze(1) * V * df_v.unsqueeze(1)
    )
    grad_u = torch.bmm(M_inv, sum_W_V_du.unsqueeze(2)).squeeze()
    grad_v = torch.bmm(M_inv, sum_W_V_dv.unsqueeze(2)).squeeze()

    tau_xx = 2.0 * mu_nd * grad_u[:, 0]
    tau_yy = 2.0 * mu_nd * grad_v[:, 1]
    tau_xy = mu_nd * (grad_u[:, 1] + grad_v[:, 0])
    n_x = wall_normal_vec[:, 0]
    n_y = wall_normal_vec[:, 1]
    t_x = tau_xx * n_x + tau_xy * n_y
    t_y = tau_xy * n_x + tau_yy * n_y
    wss_mag = torch.sqrt(t_x ** 2 + t_y ** 2) * mask_wall.float()
    wss_mag = _clip_wss_magnitude_quantile(wss_mag, mask_wall, q=0.995)
    return torch.stack([u_nd, v_nd, p_nd, mu_nd, wss_mag], dim=1)


def resolve_d_bar_si_from_sidecar_or_inlet(
    sidecar_meta: Optional[Mapping[str, Any]],
    *,
    stem: str,
    mesh_nodes_si: np.ndarray,
    mask_inlet: torch.Tensor,
) -> float:
    if sidecar_meta is not None and "d_bar" in sidecar_meta:
        d_bar, _ = d_bar_si_from_sidecar(sidecar_meta, stem=stem, builder="PatientDataExtractor")
        return float(d_bar)
    inlet_coords = mesh_nodes_si[mask_inlet.detach().cpu().numpy()]
    if len(inlet_coords) > 1:
        return float(np.max(np.linalg.norm(inlet_coords[:, None] - inlet_coords, axis=-1)))
    return 0.0198


def build_kinematics_graph_from_comsol_steady(
    *,
    mesh: meshio.Mesh,
    mesh_nodes_si: np.ndarray,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    mask_wall: torch.Tensor,
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    p_nd: torch.Tensor,
    mu_nd: torch.Tensor,
    d_bar_si: float,
    u_ref: float,
    sidecar_meta: Optional[Mapping[str, Any]],
    stem: str,
    G_x: torch.Tensor,
    G_y: torch.Tensor,
    V: torch.Tensor,
    W: torch.Tensor,
    M_inv: torch.Tensor,
    phys_cfg: Optional[PhysicsConfig] = None,
    raw_sidecar_dir: Optional[Any] = None,
    geometry_level: Optional[int] = None,
    prior_mode: str = "gt_flow",
) -> Data:
    """Assemble a single steady-state kinematics ``Data`` object (trainer schema).

    ``prior_mode``:
      - ``gt_flow``: COMSOL t=0 u,v,mu (+ WSS from GT) in prior channels (anchors).
      - ``analytic``: Poiseuille/Carreau priors only (synthetic meshes).
    """
    from src.data_gen.lib.node_feature_assembly import apply_gt_flow_priors_to_kine_x

    phys_cfg = phys_cfg or PhysicsConfig(phase="kinematics", rheology="carreau")
    ref_mu = float(phys_cfg.mu_ref)
    n = int(mesh_nodes_si.shape[0])
    nodes_nd = torch.tensor(mesh_nodes_si / float(d_bar_si), dtype=torch.float32)
    pos_nd_tensor = nodes_nd

    cl_pts_nd, cl_tan_nd, centerline_source = resolve_centerline_nd(
        pos_nd_tensor,
        mask_inlet,
        mask_outlet,
        edge_index=edge_index,
        mask_wall=mask_wall,
        stem=stem,
        raw_sidecar_dir=raw_sidecar_dir,
    )
    centerline_pts_si = np.asarray(cl_pts_nd, dtype=np.float64) * float(d_bar_si)

    sdf_tensor, wall_normal_vec = wall_normals_and_sdf_mesh_to_graph_style(
        mesh,
        mesh_nodes_si,
        mask_wall=mask_wall,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        d_bar_si=float(d_bar_si),
        centerline_pts_si=centerline_pts_si,
    )

    wall_coords_si = mesh_nodes_si[mask_wall.detach().cpu().numpy()]
    wall_tree = cKDTree(wall_coords_si if len(wall_coords_si) else mesh_nodes_si)

    u_bc = torch.zeros((n, 1), dtype=torch.float32)
    v_bc = torch.zeros((n, 1), dtype=torch.float32)
    u_bc[mask_inlet, 0] = u_nd[mask_inlet]
    v_bc[mask_inlet, 0] = v_nd[mask_inlet]

    x_tensor, u_prior, mu_prior = build_kinematics_node_x_tensor(
        pos_nd=pos_nd_tensor,
        sdf_nd=sdf_tensor,
        wall_normal=wall_normal_vec,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        mask_wall=mask_wall,
        d_bar_si=float(d_bar_si),
        u_ref=float(u_ref),
        phys_cfg=phys_cfg,
        wall_tree=wall_tree,
        edge_index=edge_index,
        G_x=G_x,
        G_y=G_y,
        centerline_pts_nd=cl_pts_nd,
        centerline_tangents_nd=cl_tan_nd,
        inlet_uv_nd=(u_bc.squeeze(-1), v_bc.squeeze(-1)),
        mu_nd_scale=phys_cfg.mu_viscosity_nd_scale,
    )

    mode = str(prior_mode or "analytic").strip().lower()
    if mode in ("gt", "gt_flow", "gt_t0"):
        x_tensor = apply_gt_flow_priors_to_kine_x(
            x_tensor,
            u_nd=u_nd,
            v_nd=v_nd,
            mu_nd=mu_nd,
            mask_wall=mask_wall,
            wall_normal=wall_normal_vec,
            edge_index=edge_index,
            M_inv=M_inv,
            V=V,
            W=W,
        )

    u_nd = u_nd.clone()
    v_nd = v_nd.clone()
    u_nd[mask_wall] = 0.0
    v_nd[mask_wall] = 0.0

    y_labels = comsol_fields_to_kinematics_y(
        u_nd=u_nd,
        v_nd=v_nd,
        p_nd=p_nd,
        mu_nd=mu_nd,
        wall_normal_vec=wall_normal_vec,
        mask_wall=mask_wall,
        edge_index=edge_index,
        M_inv=M_inv,
        V=V,
        W=W,
        ref_mu=ref_mu,
    )

    data = assemble_kinematics_graph_data(
        x_tensor=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_labels=y_labels,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        mask_wall=mask_wall,
        is_anchor=True,
        d_bar=float(d_bar_si),
        u_ref=float(u_ref),
        u_prior=u_prior,
        mu_prior=mu_prior,
        V=V,
        W=W,
        M_inv=M_inv,
        G_x=G_x,
        G_y=G_y,
    )
    data.graph_stem = stem
    if geometry_level is not None:
        data.geometry_level = torch.tensor([int(geometry_level)], dtype=torch.int8)
    elif sidecar_meta is not None and sidecar_meta.get("level") is not None:
        data.geometry_level = torch.tensor([int(sidecar_meta["level"])], dtype=torch.int8)
    else:
        attach_geometry_metadata(data, mesh_input_dir=VesselConfig(phase="biochem_anchors").mesh_input_dir, stem=stem)
    data.centerline_source = centerline_source
    data.u_prior = u_prior
    data.mu_prior = mu_prior
    return data
