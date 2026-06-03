"""Anchor kine prior / Carreau alignment tests."""

from __future__ import annotations

import torch

from src.config import NodeFeat, PhysicsConfig
from src.data_gen.lib.node_feature_assembly import (
    apply_gt_flow_priors_to_kine_x,
    build_kinematics_node_x_tensor,
    resolve_anchor_kine_phys_cfg,
)
from src.utils.kinematics_paths import BIOCHEM_ANCHOR_KINE_RHEOLOGY, kinematics_anchor_graph_dir


def test_resolve_anchor_kine_phys_is_carreau():
    phys = resolve_anchor_kine_phys_cfg()
    assert phys.viscosity_model == "carreau"
    assert phys.re_target == 450.0


def test_build_kinematics_node_x_carreau_rheo_flag():
    n = 8
    pos = torch.randn(n, 2)
    sdf = torch.rand(n, 1).clamp(min=0.01)
    wn = torch.randn(n, 2)
    wn = wn / wn.norm(dim=1, keepdim=True).clamp(min=1e-6)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[0] = True
    mask[1] = True
    mask_wall = ~mask
    edge = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    from scipy.spatial import cKDTree

    phys = PhysicsConfig(phase="kinematics", rheology="carreau")
    x, _, mu = build_kinematics_node_x_tensor(
        pos_nd=pos,
        sdf_nd=sdf,
        wall_normal=wn,
        mask_inlet=mask,
        mask_outlet=torch.tensor([False, True] + [False] * (n - 2)),
        mask_wall=mask_wall,
        d_bar_si=0.015,
        u_ref=0.1,
        phys_cfg=phys,
        wall_tree=cKDTree(pos.numpy()),
        edge_index=edge,
    )
    assert x.shape[1] == NodeFeat.WIDTH_D2.stop
    assert float(x[:, 10:11].mean()) == 1.0
    assert float(mu.mean()) > 1.0


def test_apply_gt_flow_priors_overwrites_uv_mu():
    n = 6
    x = torch.zeros(n, NodeFeat.WIDTH_D2.stop)
    u = torch.linspace(0.1, 1.0, n)
    v = torch.linspace(0.0, 0.2, n)
    mu = torch.ones(n) * 2.5
    mask_wall = torch.zeros(n, dtype=torch.bool)
    mask_wall[-1] = True
    edge = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    wn = torch.zeros(n, 2)
    wn[:, 0] = 1.0
    V = torch.zeros(edge.shape[1], 5)
    W = torch.ones(edge.shape[1])
    M = torch.eye(5).unsqueeze(0).expand(n, 5, 5).clone()
    out = apply_gt_flow_priors_to_kine_x(
        x,
        u_nd=u,
        v_nd=v,
        mu_nd=mu,
        mask_wall=mask_wall,
        wall_normal=wn,
        edge_index=edge,
        M_inv=M,
        V=V,
        W=W,
    )
    assert torch.allclose(out[:, NodeFeat.UV_PRIOR][:-1, 0], u[:-1])
    assert torch.allclose(out[:, NodeFeat.MU_PRIOR].reshape(-1), mu)


def test_kinematics_anchor_dir_uses_carreau():
    p = kinematics_anchor_graph_dir(rheology=BIOCHEM_ANCHOR_KINE_RHEOLOGY)
    assert p.as_posix().endswith("graphs_kinematics_anchors/carreau")
