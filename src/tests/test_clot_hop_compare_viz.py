"""Hop-colored clot compare panel helpers."""

from __future__ import annotations

import numpy as np


def test_scatter_clot_hop_panel_counts():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.core_physics.species_gnn_ladder_viz import scatter_clot_hop_panel

    pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)
    phi = np.array([1.0, 1.0, 0.0, 1.0])
    hops = np.array([0, 2, 1, 4])
    fig, ax = plt.subplots()
    counts = scatter_clot_hop_panel(ax, pos, phi, hops, "GT", s=5.0)
    plt.close(fig)
    assert counts["n_clot"] == 3
    assert counts["hop0"] == 1
    assert counts["hop2"] == 1
    assert counts["hop4"] == 1
    assert counts["hop1"] == 0
