"""Boundary volume-flux diagnostics (inlet FD vs prediction)."""

import math

import pytest
import torch
from src.config import PhysicsConfig
from torch_geometric.data import Data

from src.utils.boundary_flux import (
    compute_inlet_outlet_flux_debug,
    fd_inlet_flux_ref_from_re_nd,
    flux_debug_from_graph_data,
)


def _channel_mesh():
    """Unit square channel: inlet x=0, outlet x=1, walls y=0,1."""
    xs = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ys = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    pos = torch.stack([xs, ys], dim=1)
    rows, cols = [], []
    # Vertical segments along each column (x = 0 and x = 1).
    for i in range(2):
        for j in range(2):
            a = i * 3 + j
            b = i * 3 + (j + 1)
            rows.extend([a, b])
            cols.extend([b, a])
    # Horizontal segments between inlet and outlet columns.
    for j in range(3):
        a = j
        b = 3 + j
        rows.extend([a, b])
        cols.extend([b, a])
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    mask_inlet = xs < 0.01
    mask_outlet = xs > 0.99
    mask_wall = (ys < 0.01) | (ys > 0.99)
    return pos, edge_index, mask_inlet, mask_outlet, mask_wall


def test_uniform_inlet_matches_uav_times_width():
    pos, edge_index, mask_inlet, mask_outlet, _ = _channel_mesh()
    u_ref = 1.0
    vel = torch.zeros(pos.shape[0], 2)
    vel[:, 0] = u_ref
    dbg = compute_inlet_outlet_flux_debug(
        velocity=vel,
        pos=pos,
        edge_index=edge_index,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        u_inlet_bc=vel.clone(),
        u_ref_nd=u_ref,
    )
    assert abs(dbg["Q_pred_inlet_nd"] - dbg["Q_ref_bc_nd"]) < 1e-5
    assert abs(dbg["Q_ref_bc_nd"] - u_ref * 1.0) < 0.05
    assert abs(dbg["Q_ref_re_nd"] - u_ref * 1.0) < 0.05
    assert dbg["Q_flow_ratio"] == pytest.approx(1.0, rel=1e-3)
    assert dbg["flow_trivial_score"] == pytest.approx(0.0, abs=1e-3)


def test_collapsed_flow_has_high_trivial_score():
    pos, edge_index, mask_inlet, mask_outlet, _ = _channel_mesh()
    u_ref = 1.0
    bc = torch.zeros(pos.shape[0], 2)
    bc[mask_inlet, 0] = u_ref
    pred = torch.zeros(pos.shape[0], 2)
    dbg = compute_inlet_outlet_flux_debug(
        velocity=pred,
        pos=pos,
        edge_index=edge_index,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        u_inlet_bc=bc,
        u_ref_nd=u_ref,
    )
    assert dbg["Q_pred_inlet_nd"] < 1e-6
    assert dbg["Q_flow_ratio"] < 0.01
    assert dbg["flow_trivial_score"] > 0.99


def test_flux_debug_from_graph_data():
    pos, edge_index, mask_inlet, mask_outlet, mask_wall = _channel_mesh()
    n = pos.shape[0]
    x = torch.zeros(n, 15)
    x[:, :2] = pos
    u_ref = 0.42
    vel = torch.zeros(n, 2)
    vel[mask_inlet, 0] = u_ref
    vel[:, 0] = u_ref
    data = Data(
        x=x,
        edge_index=edge_index,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        mask_wall=mask_wall,
        u_inlet_bc=vel.clone(),
        u_ref=torch.tensor([u_ref]),
    )
    dbg = flux_debug_from_graph_data(data, vel)
    assert dbg["Q_flow_ratio"] == pytest.approx(1.0, rel=1e-2)


def test_fd_ref_poiseuille_consistent_with_uav():
    q_uav = fd_inlet_flux_ref_from_re_nd(u_ref_nd=2.0, width_nd=0.5)
    assert abs(q_uav - 1.0) < 1e-9


def test_primary_ref_prefers_re_over_bc_when_both_finite():
    pos, edge_index, mask_inlet, mask_outlet, _ = _channel_mesh()
    u_ref = 1.0
    vel = torch.zeros(pos.shape[0], 2)
    vel[:, 0] = u_ref
    dbg = compute_inlet_outlet_flux_debug(
        velocity=vel,
        pos=pos,
        edge_index=edge_index,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        u_inlet_bc=vel.clone(),
        u_ref_nd=u_ref,
    )
    assert abs(dbg["Q_ref_re_nd"] - dbg["Q_ref_bc_nd"]) < 0.05
    assert dbg["Q_inlet_rel_err"] == pytest.approx(0.0, abs=1e-3)


def test_flux_debug_resolves_u_ref_from_phys_cfg():
    pos, edge_index, mask_inlet, mask_outlet, mask_wall = _channel_mesh()
    n = pos.shape[0]
    cfg = PhysicsConfig(phase="biochem")
    d_bar = 0.02
    u_ref = cfg.get_u_ref(d_bar)
    vel = torch.zeros(n, 2)
    vel[mask_inlet, 0] = u_ref
    data = Data(
        x=torch.zeros(n, 15),
        edge_index=edge_index,
        mask_inlet=mask_inlet,
        mask_outlet=mask_outlet,
        mask_wall=mask_wall,
        d_bar=torch.tensor([d_bar]),
        u_inlet_bc=vel.clone(),
    )
    data.x[:, :2] = pos
    dbg = flux_debug_from_graph_data(data, vel, phys_cfg=cfg)
    assert math.isfinite(dbg["Q_ref_re_nd"])
    assert abs(dbg["Q_ref_re_nd"] - u_ref * 1.0) < 0.05
