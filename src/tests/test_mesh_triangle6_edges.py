"""Tests for P2 triangle6 edge construction."""
from __future__ import annotations

import numpy as np
import torch

from src.data_gen.lib.mesh_triangle6_edges import (
    edge_pairs_to_bidirectional_index,
    triangle6_undirected_edge_pairs,
)


def test_triangle6_local_edges_match_midpoints():
    # Synthetic right triangle in 2D; mid nodes at edge midpoints.
    pts = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    edges = triangle6_undirected_edge_pairs(cells)
    assert edges.shape == (6, 2)
    for a, b in edges:
        # Each edge should be short (corner-mid or mid-corner along triangle side).
        assert np.linalg.norm(pts[a] - pts[b]) <= 2.0 + 1e-9


def test_triangle6_bidirectional_index_shape():
    cells = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], dtype=np.int64)
    edges = triangle6_undirected_edge_pairs(cells)
    ei = edge_pairs_to_bidirectional_index(edges)
    assert ei.shape[0] == 2
    assert ei.shape[1] == 2 * edges.shape[0]
    assert ei.dtype == torch.long
