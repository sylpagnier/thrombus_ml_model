"""Rollout macro index must map to COMSOL ``data.y`` time by physical time, not by ``i``."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.architecture.gnode_biochem import resolve_y_label_index_at_macro_step
from src.config import BiochemConfig
from src.utils.nondim import to_t_nd


def test_subsampled_viz_grid_maps_final_macro_to_last_comsol_label():
    """41 macro knots (viz fast) at t_final must use y[53], not y[40]."""
    n_y = 54
    times_si = torch.linspace(0.0, 7950.0, n_y)
    data = Data(
        x=torch.randn(10, 15),
        y=torch.randn(n_y, 10, 16),
        t=times_si,
        is_anchor=torch.tensor([True]),
    )
    bio = BiochemConfig(phase="biochem")
    idx = torch.linspace(0, n_y - 1, 41).round().long()
    eval_nd = to_t_nd(times_si[idx], bio.t_final)

    assert resolve_y_label_index_at_macro_step(data, eval_nd, 40, bio_cfg=bio) == 53
    assert resolve_y_label_index_at_macro_step(data, eval_nd, 40, bio_cfg=bio) != 40

    mid = resolve_y_label_index_at_macro_step(data, eval_nd, 20, bio_cfg=bio)
    t_mid_si = float(eval_nd[20].item()) * bio.t_final
    expected = int(torch.argmin(torch.abs(times_si - t_mid_si)).item())
    assert mid == expected


def test_contiguous_tbptt_slice_uses_start_offset():
    n_y = 20
    times_si = torch.linspace(0.0, 3000.0, n_y)
    data = Data(
        x=torch.randn(8, 15),
        y=torch.randn(n_y, 8, 16),
        t=times_si,
        is_anchor=torch.tensor([True]),
    )
    bio = BiochemConfig(phase="biochem")
    start = 5
    window = 4
    eval_nd = to_t_nd(times_si[start : start + window], bio.t_final)
    assert resolve_y_label_index_at_macro_step(
        data, eval_nd, 2, bio_cfg=bio, start_idx=start
    ) == start + 2
