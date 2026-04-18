"""Regression checks for mesh-to-graph ``Data`` contracts (aligned with ``process_file`` assemblers)."""

from __future__ import annotations

import torch

from src.config import BiochemConfig
from src.data_gen.lib.mesh_to_graph import assemble_tier12_graph_data
from src.data_gen.lib.mesh_to_graph_tier3 import (
    assemble_tier3_steady_graph_data,
    assemble_tier3_transient_graph_data,
    default_tier3_bio_inlet_bc,
)


def _base_context(num_nodes: int = 3):
    edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long)
    return {
        "edge_index": edge_index,
        "edge_attr": torch.zeros((edge_index.shape[1], 3), dtype=torch.float32),
        "mask_inlet": torch.tensor([True, False, False], dtype=torch.bool),
        "mask_outlet": torch.tensor([False, False, True], dtype=torch.bool),
        "mask_wall": torch.tensor([False, True, False], dtype=torch.bool),
        "d_bar": 1.0,
        "u_ref": 1.0,
        "V": torch.zeros((edge_index.shape[1], 5), dtype=torch.float32),
        "W": torch.ones(edge_index.shape[1], dtype=torch.float32),
        "M_inv": torch.eye(5, dtype=torch.float32).unsqueeze(0).repeat(num_nodes, 1, 1),
        "outlet_normal": torch.zeros((num_nodes, 2), dtype=torch.float32),
        "num_nodes": num_nodes,
    }


def _diag_sparse_grad(n: int) -> torch.Tensor:
    """Minimal valid coalesced sparse N×N operator (structure-only for regression tests)."""
    idx = torch.arange(n, dtype=torch.long)
    ii = torch.stack([idx, idx], dim=0)
    return torch.sparse_coo_tensor(ii, torch.ones(n, dtype=torch.float32), (n, n)).coalesce()


def test_tier12_saved_graph_includes_wls_and_sparse_gradients_without_laplacian(tmp_path):
    """Tier 1/2 ``*.pt`` graphs carry WLS blocks + ``G_x``/``G_y``; Laplacian is not serialized."""
    ctx = _base_context(num_nodes=3)
    priors = {
        "u_prior": torch.zeros(3, dtype=torch.float32),
        "mu_prior": torch.ones(3, dtype=torch.float32),
    }
    x_tensor = torch.zeros((3, 15), dtype=torch.float32)
    y_labels = torch.zeros((3, 5), dtype=torch.float32)
    gx = _diag_sparse_grad(3)
    gy = _diag_sparse_grad(3)

    data = assemble_tier12_graph_data(
        x_tensor=x_tensor,
        edge_index=ctx["edge_index"],
        edge_attr=ctx["edge_attr"],
        y_labels=y_labels,
        mask_inlet=ctx["mask_inlet"],
        mask_outlet=ctx["mask_outlet"],
        mask_wall=ctx["mask_wall"],
        is_anchor=False,
        d_bar=float(ctx["d_bar"]),
        u_ref=float(ctx["u_ref"]),
        u_prior=priors["u_prior"],
        mu_prior=priors["mu_prior"],
        V=ctx["V"],
        W=ctx["W"],
        M_inv=ctx["M_inv"],
        G_x=gx,
        G_y=gy,
    )

    assert data.V.shape == ctx["V"].shape
    assert data.W.shape == ctx["W"].shape
    assert data.M_inv.shape == ctx["M_inv"].shape
    assert data.G_x.is_sparse and data.G_y.is_sparse
    assert data.G_x.shape == (3, 3) and data.G_y.shape == (3, 3)
    assert not hasattr(data, "Laplacian")


def test_tier3_non_anchor_transient_graph_matches_process_file_shape(tmp_path):
    """Tier 3 non-anchor: transient ``y``, time axis, biochem BCs, sparse operators (same as pipeline)."""
    ctx = _base_context(num_nodes=3)
    bio_cfg = BiochemConfig(tier="tier3")
    num_times = bio_cfg.num_time_steps
    tvec = torch.linspace(0.0, bio_cfg.t_final, num_times, dtype=torch.float32)
    y_series = torch.zeros((num_times, 3, 16), dtype=torch.float32)
    gx = _diag_sparse_grad(3)
    gy = _diag_sparse_grad(3)
    lap = _diag_sparse_grad(3)
    u_prior = torch.tensor([1.0, 0.2, 0.0], dtype=torch.float32)
    v_prior = torch.tensor([0.0, 0.1, 0.0], dtype=torch.float32)
    mu_prior = torch.ones(3, dtype=torch.float32)
    uv_inlet_bc = torch.cat([u_prior.view(-1, 1), v_prior.view(-1, 1)], dim=1)
    x_tensor = torch.zeros((3, 15), dtype=torch.float32)
    bio_inlet_bc = default_tier3_bio_inlet_bc(3)

    data = assemble_tier3_transient_graph_data(
        x_tensor=x_tensor,
        y_tensor_series=y_series,
        eval_times_tensor=tvec,
        edge_index=ctx["edge_index"],
        edge_attr=ctx["edge_attr"],
        mask_inlet=ctx["mask_inlet"],
        mask_outlet=ctx["mask_outlet"],
        mask_wall=ctx["mask_wall"],
        d_bar=float(ctx["d_bar"]),
        u_ref=float(ctx["u_ref"]),
        re_target=100.0,
        G_x=gx,
        G_y=gy,
        Laplacian=lap,
        V=ctx["V"],
        W=ctx["W"],
        M_inv=ctx["M_inv"],
        uv_inlet_bc=uv_inlet_bc,
        mu_prior=mu_prior,
        bio_inlet_bc=bio_inlet_bc,
        outlet_normal=ctx["outlet_normal"],
    )

    assert hasattr(data, "t")
    assert data.y.dim() == 3
    assert data.y.shape == (num_times, 3, 16)
    assert data.t.numel() == num_times
    assert data.y.shape[0] == data.t.shape[0]
    assert data.u_inlet_bc.shape[1] == 2
    assert data.bio_inlet_bc.shape == (3, 9)
    assert data.G_x.is_sparse and data.G_y.is_sparse and data.Laplacian.is_sparse
    assert int(data.is_anchor.sum().item()) == 0


def test_tier3_anchor_steady_graph_matches_steady_label_layout(tmp_path):
    """Tier 3 anchor (COMSOL-labeled): steady ``[N,5]`` kinematics + scalar anchor flag."""
    ctx = _base_context(num_nodes=3)
    gx = _diag_sparse_grad(3)
    gy = _diag_sparse_grad(3)
    lap = _diag_sparse_grad(3)
    u_prior = torch.zeros(3, dtype=torch.float32)
    mu_prior = torch.ones(3, dtype=torch.float32)
    x_tensor = torch.zeros((3, 15), dtype=torch.float32)
    y_labels = torch.zeros((3, 5), dtype=torch.float32)

    data = assemble_tier3_steady_graph_data(
        x_tensor=x_tensor,
        y_labels=y_labels,
        edge_index=ctx["edge_index"],
        edge_attr=ctx["edge_attr"],
        mask_inlet=ctx["mask_inlet"],
        mask_outlet=ctx["mask_outlet"],
        mask_wall=ctx["mask_wall"],
        is_anchor=True,
        d_bar=float(ctx["d_bar"]),
        u_ref=float(ctx["u_ref"]),
        u_prior=u_prior,
        mu_prior=mu_prior,
        outlet_normal=ctx["outlet_normal"],
        V=ctx["V"],
        W=ctx["W"],
        M_inv=ctx["M_inv"],
        G_x=gx,
        G_y=gy,
        Laplacian=lap,
    )

    assert data.y.shape == (3, 5)
    assert data.is_anchor.shape == (1,)
    assert data.is_anchor.item() is True
    assert data.u_inlet_bc.shape[1] == 1
    assert data.G_x.is_sparse and data.Laplacian.is_sparse
