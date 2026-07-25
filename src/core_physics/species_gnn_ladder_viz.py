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

# Clot presence colored by hop-from-wall (shared GT / pred rows).
CLOT_HOP_COLORS = {
    0: "#4a148c",   # Wall: deep purple
    1: "#7b1fa2",   # Hop 1
    2: "#e65100",   # Hop 2: orange (lumen near)
    3: "#f9a825",   # Hop 3: gold
    4: "#c0ca33",   # Hop 4+: lime
}
CLOT_HOP_BG = "#d9d9d9"


def scatter_clot_hop_panel(
    ax,
    pos: np.ndarray,
    phi: np.ndarray,
    hop_distances: np.ndarray,
    row_label: str,
    *,
    thresh: float = 0.5,
    s: float = 3.0,
) -> dict[str, int]:
    """Scatter clot nodes colored by hop distance from wall; non-clot = gray."""
    import matplotlib.colors as mcolors

    clot = phi.reshape(-1) >= float(thresh)
    hops = hop_distances.reshape(-1).astype(np.int64)
    n = int(pos.shape[0])
    codes = np.full(n, -1, dtype=np.int64)  # -1 = background
    if clot.any():
        h = np.clip(hops[clot], 0, 4)
        codes[clot] = h

    colors = [CLOT_HOP_BG] + [CLOT_HOP_COLORS[i] for i in range(5)]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-1.5, 5.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Background under clot so lumen hops stay visible.
    bg = codes < 0
    if bg.any():
        ax.scatter(
            pos[bg, 0],
            pos[bg, 1],
            c=CLOT_HOP_BG,
            s=max(s * 0.65, 1.0),
            linewidths=0,
            alpha=0.45,
            zorder=1,
        )
    if clot.any():
        ax.scatter(
            pos[clot, 0],
            pos[clot, 1],
            c=codes[clot],
            cmap=cmap,
            norm=norm,
            s=s,
            linewidths=0,
            alpha=0.95,
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    if row_label:
        ax.set_ylabel(row_label, fontsize=10, fontweight="bold", labelpad=8)

    counts = {f"hop{h}": int(((codes == h) & clot).sum()) for h in range(5)}
    counts["n_clot"] = int(clot.sum())
    return counts


def clot_hop_legend(fig) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=CLOT_HOP_BG, label="no clot")]
    for h in range(5):
        label = f"hop {h}" if h < 4 else "hop >=4"
        if h == 0:
            label = "hop 0 (wall)"
        handles.append(Patch(facecolor=CLOT_HOP_COLORS[h], label=label))
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=8,
        title="Clot hop from wall",
        title_fontsize=9,
    )


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
    """FP=red, FN=blue; gray elsewhere. Color-coded by hop distance from wall (-x: FN, +x: FP)."""
    import matplotlib.colors as mcolors
    gt = phi_gt.reshape(-1) >= float(thresh)
    pr = phi_pred.reshape(-1) >= float(thresh)
    fp = pr & ~gt
    fn = ~pr & gt
    
    n_nodes = pos.shape[0]
    error_vals = np.zeros(n_nodes, dtype=float)
    
    if hop_distances is not None:
        hops = hop_distances.reshape(-1)
        # FP: positive (hop + 1)
        error_vals[fp] = np.clip(hops[fp] + 1, 1, 5)
        # FN: negative -(hop + 1)
        error_vals[fn] = np.clip(-(hops[fn] + 1), -5, -1)
    else:
        error_vals[fp] = 2.0  # fallback to standard red (Hop 1 equivalent)
        error_vals[fn] = -2.0 # fallback to standard blue (Hop 1 equivalent)

    colors = [
        "#9ecae1", "#6baed6", "#4292c6", "#1f77b4", "#084594",
        "#d9d9d9",
        "#800000", "#d62728", "#ff7f0e", "#fd8d3c", "#feb24c"
    ]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-5.5, 6.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    ax.scatter(
        pos[:, 0], pos[:, 1], c=error_vals, cmap=cmap, norm=norm,
        s=s, linewidths=0, alpha=0.92, zorder=3
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
