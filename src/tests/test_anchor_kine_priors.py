"""Anchor kine prior / Carreau alignment tests."""

from __future__ import annotations

import torch

from src.config import NodeFeat, PhysicsConfig
from src.data_gen.lib.node_feature_assembly import (

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





def test_kinematics_anchor_dir_uses_carreau():
    p = kinematics_anchor_graph_dir(rheology=BIOCHEM_ANCHOR_KINE_RHEOLOGY)
    assert p.as_posix().endswith("graphs_kinematics_anchors/carreau")
