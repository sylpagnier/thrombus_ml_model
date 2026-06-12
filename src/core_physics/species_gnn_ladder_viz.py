"""Shared ladder time grid for species GNN clot / species comparison viz."""

from __future__ import annotations

import numpy as np


def ladder_viz_times(n_steps: int, *, max_frames: int = 10) -> list[int]:
    """Evenly spaced macro-step indices (same grid as ``viz_species_gnn_clot_ladder``)."""
    n = max(int(n_steps), 1)
    mf = int(max_frames)
    if mf <= 0 or n <= mf:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=mf, dtype=int)
    return sorted({int(i) for i in idx.tolist()})
