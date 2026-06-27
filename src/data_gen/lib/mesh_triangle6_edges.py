"""P2 (triangle6) mesh edge construction for biochem graphs.

COMSOL quadratic triangles use six nodes per element: corners (0,1,2) and
mid-edge nodes (3 on 0-1, 4 on 1-2, 5 on 2-0).  Corner-only wiring leaves
~75% of nodes degree-0 and breaks wall-hop BFS on anchor meshes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Local edges for one quadratic triangle (Gmsh / meshio triangle6 ordering).
_TRIANGLE6_LOCAL_EDGES = np.array(
    [[0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0]],
    dtype=np.int64,
)
_LINEAR_LOCAL_EDGES = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)


def triangle6_undirected_edge_pairs(cells: np.ndarray) -> np.ndarray:
    """Build unique undirected edges from ``triangle6`` cells ``(M, 6)``."""
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 6:
        raise ValueError(f"triangle6 cells must be (M, 6); got {cells.shape}")
    local = _TRIANGLE6_LOCAL_EDGES
    pairs = cells[:, local[:, 0]].reshape(-1, 1)
    pairs = np.column_stack([pairs.reshape(-1), cells[:, local[:, 1]].reshape(-1)])
    return np.unique(np.sort(pairs, axis=1), axis=0)


def linear_triangle_undirected_edge_pairs(cells: np.ndarray) -> np.ndarray:
    """Build unique undirected edges from linear ``triangle`` cells ``(M, 3)``."""
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] < 3:
        raise ValueError(f"triangle cells must be (M, >=3); got {cells.shape}")
    tris = cells[:, :3]
    edges = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def mesh_undirected_edge_pairs(mesh) -> np.ndarray:
    """Return unique undirected edges from a meshio mesh (triangle or triangle6)."""
    if "triangle" in mesh.cells_dict:
        return linear_triangle_undirected_edge_pairs(mesh.cells_dict["triangle"])
    if "triangle6" in mesh.cells_dict:
        return triangle6_undirected_edge_pairs(mesh.cells_dict["triangle6"])
    raise ValueError(
        "mesh has no supported triangle cell type "
        f"(got {list(mesh.cells_dict.keys())})"
    )


def edge_pairs_to_bidirectional_index(edges: np.ndarray) -> torch.Tensor:
    """``(E, 2)`` undirected edges -> PyG ``edge_index`` ``(2, 2E)``."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    rev = edges[:, [1, 0]]
    stacked = np.hstack([edges.T, rev.T])
    return torch.tensor(stacked, dtype=torch.long)


def edge_index_from_mesh(mesh) -> torch.Tensor:
    """Full P2 (or linear) connectivity as bidirectional ``edge_index``."""
    return edge_pairs_to_bidirectional_index(mesh_undirected_edge_pairs(mesh))


def mesh_num_nodes(mesh) -> int:
    return int(len(mesh.points))


def edge_index_from_mesh_path(mesh_path: str | Path) -> torch.Tensor:
    """Load ``.nas`` / ``.msh`` and return bidirectional ``edge_index``."""
    import meshio

    mesh = meshio.read(str(mesh_path))
    return edge_index_from_mesh(mesh)


def edge_index_from_mesh_path_checked(
    mesh_path: str | Path,
    *,
    num_nodes: int,
    stem: str = "",
) -> torch.Tensor:
    """Like ``edge_index_from_mesh_path`` but verifies mesh node count matches the graph."""
    import meshio

    mesh = meshio.read(str(mesh_path))
    n_mesh = mesh_num_nodes(mesh)
    if int(num_nodes) != n_mesh:
        tag = stem or Path(mesh_path).stem
        raise ValueError(
            f"{tag}: mesh nodes ({n_mesh}) != graph nodes ({int(num_nodes)}) "
            f"for {Path(mesh_path).name}; need COMSOL triangle6 export matching the anchor graph"
        )
    return edge_index_from_mesh(mesh)
