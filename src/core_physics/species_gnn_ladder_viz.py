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


def scatter_clot_error_panel(
    ax,
    pos: np.ndarray,
    phi_gt: np.ndarray,
    phi_pred: np.ndarray,
    row_label: str,
    *,
    thresh: float = 0.5,
    s: float = 3.0,
) -> dict[str, int]:
    """FP=red overpaint, FN=blue missed clot; gray bulk elsewhere."""
    gt = phi_gt.reshape(-1) >= float(thresh)
    pr = phi_pred.reshape(-1) >= float(thresh)
    fp = pr & ~gt
    fn = ~pr & gt
    ax.scatter(
        pos[:, 0], pos[:, 1], c="#d9d9d9", s=s, linewidths=0, alpha=0.55, zorder=1,
    )
    if fp.any():
        ax.scatter(
            pos[fp, 0], pos[fp, 1], c="#d62728", s=max(s * 1.15, 2.5),
            linewidths=0, alpha=0.92, zorder=3,
        )
    if fn.any():
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
