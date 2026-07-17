"""Shared ladder time grid and clot-ladder panel helpers for species GNN viz."""

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


# Hop-based error color coding (FPs = red/orange spectrum, FNs = blue/teal spectrum)
FP_COLORS = {
    0: "#800000",   # Wall: Dark Red
    1: "#d62728",   # Hop 1: Standard Red
    2: "#ff7f0e",   # Hop 2: Orange
    3: "#fd8d3c",   # Hop 3: Light Orange
    4: "#feb24c",   # Hop 4+: Yellow-Orange
}

FN_COLORS = {
    0: "#084594",   # Wall: Dark Blue
    1: "#1f77b4",   # Hop 1: Standard Blue
    2: "#4292c6",   # Hop 2: Sky Blue
    3: "#6baed6",   # Hop 3: Light Blue
    4: "#9ecae1",   # Hop 4+: Pale Blue
}


def scatter_clot_error_panel(
    ax,
    pos: np.ndarray,
    phi_gt: np.ndarray,
    phi_pred: np.ndarray,
    row_label: str,
    *,
    thresh: float = 0.5,
    s: float = 3.0,
    hop_distances: np.ndarray | None = None,
) -> dict[str, int]:
    """FP=red overpaint, FN=blue missed clot; gray bulk elsewhere. Can color-code by hop."""
    gt = phi_gt.reshape(-1) >= float(thresh)
    pr = phi_pred.reshape(-1) >= float(thresh)
    fp = pr & ~gt
    fn = ~pr & gt
    
    # Gray background
    ax.scatter(
        pos[:, 0], pos[:, 1], c="#d9d9d9", s=s, linewidths=0, alpha=0.55, zorder=1,
    )
    
    # FP coloring
    if fp.any():
        if hop_distances is not None:
            hops = hop_distances.reshape(-1)[fp]
            for hop_val in range(5):
                mask = (hops == hop_val) if hop_val < 4 else (hops >= 4)
                if mask.any():
                    color = FP_COLORS[hop_val]
                    ax.scatter(
                        pos[fp][mask, 0], pos[fp][mask, 1], c=color, s=max(s * 1.15, 2.5),
                        linewidths=0, alpha=0.92, zorder=3,
                    )
        else:
            ax.scatter(
                pos[fp, 0], pos[fp, 1], c="#d62728", s=max(s * 1.15, 2.5),
                linewidths=0, alpha=0.92, zorder=3,
            )
            
    # FN coloring
    if fn.any():
        if hop_distances is not None:
            hops = hop_distances.reshape(-1)[fn]
            for hop_val in range(5):
                mask = (hops == hop_val) if hop_val < 4 else (hops >= 4)
                if mask.any():
                    color = FN_COLORS[hop_val]
                    ax.scatter(
                        pos[fn][mask, 0], pos[fn][mask, 1], c=color, s=max(s * 1.15, 2.5),
                        linewidths=0, alpha=0.92, zorder=3,
                    )
        else:
            ax.scatter(
                pos[fn, 0], pos[fn, 1], c="#1f77b4", s=max(s * 1.15, 2.5),
                linewidths=0, alpha=0.92, zorder=3,
            )
            
    ax.set_aspect("equal")
    ax.axis("off")
    if row_label:
        ax.set_ylabel(row_label, fontsize=10, fontweight="bold", labelpad=8)
    return {
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "tp": int((pr & gt).sum()),
        "tn": int((~pr & ~gt).sum()),
    }
