"""Shared 2nd-order WLS operators and Gmsh boundary mask extraction for graph builders (DRY)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def precompute_wls_operators(edge_index: torch.Tensor, num_nodes: int, pos_tensor: torch.Tensor):
    """
    2nd-order polynomial WLS on edges; returns ``V``, ``W``, ``M_inv``.
    """
    row, col = edge_index
    pos_diff = pos_tensor[col, :2] - pos_tensor[row, :2]
    dx, dy = pos_diff[:, 0], pos_diff[:, 1]

    dist_sq = dx**2 + dy**2 + 1e-8

    dx2 = 0.5 * dx**2
    dxy = dx * dy
    dy2 = 0.5 * dy**2

    V = torch.stack([dx, dy, dx2, dxy, dy2], dim=1)
    W = 1.0 / dist_sq

    V_unsqueezed = V.unsqueeze(2)
    V_T_unsqueezed = V.unsqueeze(1)
    M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)

    M_e_flat = M_e.view(-1, 25)
    out = torch.zeros((num_nodes, 25), dtype=M_e_flat.dtype, device=M_e_flat.device)
    row_exp = row.view(-1, 1).expand_as(M_e_flat)
    M_flat = out.scatter_add_(0, row_exp, M_e_flat)

    M = M_flat.view(num_nodes, 5, 5)
    epsilon = 1e-6
    I = torch.eye(5, dtype=M.dtype, device=M.device).unsqueeze(0).expand(num_nodes, 5, 5)
    M_reg = M + epsilon * I
    M_inv = torch.linalg.pinv(M_reg, rcond=1e-5)

    return V, W, M_inv


def gmsh_line_boundary_masks(mesh, num_nodes: int, tags: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inlet / outlet / wall boolean masks from Gmsh line physical tags."""
    mask_inlet = torch.zeros(num_nodes, dtype=torch.bool)
    mask_outlet = torch.zeros(num_nodes, dtype=torch.bool)
    mask_wall = torch.zeros(num_nodes, dtype=torch.bool)

    line_cells = []
    line_tags = []

    t_in = tags["Inlet"]
    t_out = tags["Outlet_1"]
    t_wall = tags["Walls"]

    try:
        if "line" in mesh.cells_dict:
            line_cells = mesh.cells_dict["line"]
            line_tags = mesh.cell_data_dict["gmsh:physical"]["line"]
        elif hasattr(mesh, "get_cells_type"):
            line_cells = mesh.get_cells_type("line")
            line_tags = mesh.get_cell_data("gmsh:physical", "line")
    except Exception:
        pass

    if len(line_tags) == 0 or len(line_cells) == 0:
        raise ValueError(
            "gmsh_line_boundary_masks: no Gmsh line cells with physical tags were found "
            "(expected mesh.cells_dict['line'] and cell_data_dict['gmsh:physical']['line'], "
            "or meshio equivalents). Re-export the .msh with tagged inlet, outlet, and wall curves."
        )
    if len(line_tags) != len(line_cells):
        raise ValueError(
            f"gmsh_line_boundary_masks: line_tags length ({len(line_tags)}) != line_cells length "
            f"({len(line_cells)}); mesh file is inconsistent."
        )

    def _line_node_indices(cell) -> np.ndarray:
        arr = np.asarray(cell, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return arr
        if (arr < 0).any() or (arr >= num_nodes).any():
            bad = arr[(arr < 0) | (arr >= num_nodes)]
            raise ValueError(
                "gmsh_line_boundary_masks: line references node index outside "
                f"[0, {num_nodes - 1}] (bad values: {bad[:16]!r}{'...' if bad.size > 16 else ''}). "
                "Re-export or repair the mesh."
            )
        return arr

    for i, tag in enumerate(line_tags):
        if isinstance(line_cells, list) and not isinstance(line_cells[0], (int, float, np.integer)):
            nodes = line_cells[i]
        else:
            nodes = line_cells[i]
        idx = _line_node_indices(nodes)

        if tag == t_in:
            mask_inlet[idx] = True
        elif tag == t_out:
            mask_outlet[idx] = True
        elif tag == t_wall:
            mask_wall[idx] = True

    mask_inlet = mask_inlet & (~mask_wall)
    mask_outlet = mask_outlet & (~mask_wall)

    unique_tags = sorted({int(t) for t in line_tags})
    tag_msg = f"Unique gmsh:physical line tags present in mesh: {unique_tags}. " f"Expected Inlet={t_in}, Outlet_1={t_out}, Walls={t_wall}."

    if not bool(mask_inlet.any()):
        raise ValueError(
            "gmsh_line_boundary_masks: **no inlet nodes** matched VesselConfig.TAGS['Inlet']. "
            + tag_msg
            + " Fix Gmsh physical names/IDs or update TAGS, then re-export."
        )
    if not bool(mask_outlet.any()):
        raise ValueError(
            "gmsh_line_boundary_masks: **no outlet nodes** matched VesselConfig.TAGS['Outlet_1']. "
            + tag_msg
            + " Fix Gmsh physical names/IDs or update TAGS, then re-export."
        )
    if not bool(mask_wall.any()):
        raise ValueError(
            "gmsh_line_boundary_masks: **no wall nodes** matched VesselConfig.TAGS['Walls']. "
            + tag_msg
            + " Without wall tags, surface species and wall residuals are undefined. "
            "Re-export the mesh with wall boundary curves under the expected physical group."
        )
    if bool((mask_inlet & mask_outlet).any()):
        overlap = int((mask_inlet & mask_outlet).sum().item())
        raise ValueError(
            f"gmsh_line_boundary_masks: {overlap} node(s) are both inlet and outlet after wall "
            "carving (tags overlap on shared vertices). Fix boundary curve tagging in Gmsh."
        )

    return mask_inlet, mask_outlet, mask_wall


__all__ = ["precompute_wls_operators", "gmsh_line_boundary_masks"]
