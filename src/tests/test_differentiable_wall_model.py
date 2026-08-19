"""Unit tests for the differentiable wall model (Level 1.1)."""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from src.config import BiochemConfig, PhysicsConfig
from src.differentiable_wall_model.differentiable_ode import DifferentiableWallModel
from src.differentiable_wall_model.losses import CombinedWallClotLoss, SoftDiceF1Loss
from src.differentiable_wall_model.parameters import GlobalPhysicsParameters


def _create_mock_vessel_graph(num_nodes: int = 40, num_wall: int = 15) -> Data:
    """Create a minimal synthetic graph object with necessary attributes."""
    torch.manual_seed(42)
    # Positions along 2D channel
    x = torch.linspace(0, 2, num_nodes)
    y = torch.sin(x * 3.1415)
    pos = torch.stack((x, y), dim=1)

    # Wall mask
    mask_wall = torch.zeros(num_nodes, dtype=torch.bool)
    mask_wall[:num_wall] = True

    # Simple line graph edges
    edges_src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    edges_dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    # Velocities at t=0
    u0_pred = torch.ones(num_nodes) * 0.1
    v0_pred = torch.zeros(num_nodes)

    # Signed distance function (sdf_nd)
    sdf = torch.abs(y)
    # Node features: column 0,1 pos, column 2 sdf_nd
    node_feat = torch.zeros(num_nodes, 15)
    node_feat[:, 0] = x
    node_feat[:, 1] = y
    node_feat[:, 2] = sdf

    # Time steps
    t = torch.linspace(0, 1000, 10)

    # Synthetic y label with 5 timesteps
    y_full = torch.zeros(10, num_nodes, 4)
    y_full[0, :, 0] = u0_pred
    y_full[0, :, 1] = v0_pred
    # Species log1p channels (RP=2, AP=3)
    y_full[0, :, 2] = 0.0
    y_full[0, :, 3] = 0.0

    data = Data(
        x=node_feat,
        edge_index=edge_index,
        pos=pos,
        mask_wall=mask_wall,
        u0_pred=u0_pred,
        v0_pred=v0_pred,
        t=t,
        y=y_full,
    )
    data.u_ref = torch.tensor([0.1])
    data.d_bar = torch.tensor([0.015])
    data.y_channel_names = "u,v,RP_log1p_nd,AP_log1p_nd"
    return data


def test_parameter_provider_constraints():
    params_mod = GlobalPhysicsParameters()
    eff = params_mod.get_effective_parameters()
    assert eff["da_scale"] > 0
    assert eff["wake"] > 0
    assert eff["lss"] > 0
    assert eff["sgt_cgs"] < 0
    assert eff["tau_low"] > 0
    assert eff["relax"] > 0


def test_differentiable_wall_model_forward():
    bio_cfg = BiochemConfig(phase="biochem")
    model = DifferentiableWallModel(bio_cfg=bio_cfg, default_grow_hops=2)
    data = _create_mock_vessel_graph()

    out = model(data, flow_source="pred")
    prob = out["prob_clot"]
    mat_final = out["mat_final"]

    assert prob.shape == (data.num_nodes,)
    assert (prob >= 0.0).all() and (prob <= 1.0).all()
    assert mat_final.shape == (data.num_nodes,)
    assert not torch.isnan(prob).any()
    assert not torch.isnan(mat_final).any()


def test_differentiable_wall_model_gradient_flow():
    bio_cfg = BiochemConfig(phase="biochem")
    params_mod = GlobalPhysicsParameters()
    model = DifferentiableWallModel(bio_cfg=bio_cfg, parameter_provider=params_mod, default_grow_hops=2)
    data = _create_mock_vessel_graph()

    out = model(data, flow_source="pred")
    prob = out["prob_clot"]

    # Target: first 5 wall nodes have clot
    target = torch.zeros_like(prob)
    target[:5] = 1.0

    loss_fn = CombinedWallClotLoss()
    loss_dict = loss_fn(prob, target, data.mask_wall)
    loss = loss_dict["loss"]
    loss.backward()

    # Verify gradients exist and are finite for trainable parameters
    for name, p in params_mod.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(p.grad).any(), f"Inf gradient for {name}"


def test_loss_functions():
    pred = torch.tensor([0.9, 0.8, 0.1, 0.2])
    gt = torch.tensor([1.0, 1.0, 0.0, 0.0])
    wall = torch.tensor([True, True, True, True])

    dice = SoftDiceF1Loss()
    l = dice(pred, gt, wall)
    assert 0.0 <= float(l.item()) <= 1.0

    comb = CombinedWallClotLoss()
    res = comb(pred, gt, wall)
    assert res["loss"].item() > 0.0
