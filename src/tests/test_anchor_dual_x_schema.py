"""Anchor graphs must expose aligned kinematics + biochem node features."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import BiochemNodeFeat, NodeFeat, PhysicsConfig
from src.data_gen.lib.node_feature_assembly import (
    build_biochem_bc_x_tensor,
    build_kinematics_node_x_tensor,
)
from src.utils.channel_schema import (
    BIO_X_SCHEMA,
    KINE_X_SCHEMA,
    assert_anchor_dual_x_aligned,
    attach_patient_anchor_graph_metadata,
    biochem_encoder_x,
)


def _minimal_anchor_graph(*, n: int = 12):
    pos = torch.linspace(0, 1, n).unsqueeze(1).repeat(1, 2)
    sdf = torch.linspace(0.05, 0.4, n).unsqueeze(1)
    wall_n = torch.tensor([[0.0, 1.0]], dtype=torch.float32).repeat(n, 1)
    mask_in = torch.zeros(n, dtype=torch.bool)
    mask_in[0] = True
    mask_out = torch.zeros(n, dtype=torch.bool)
    mask_out[-1] = True
    mask_wall = ~(mask_in | mask_out)
    edges = torch.tensor([[i, i + 1] for i in range(n - 1)], dtype=torch.long).t()
    edge_index = torch.cat([edges, edges.flip(0)], dim=1)
    row, col = edge_index
    delta = pos[row] - pos[col]
    edge_attr = torch.cat([delta, delta.norm(dim=1, keepdim=True)], dim=1)

    nodes_si = pos.numpy()
    wall_pts = nodes_si[mask_wall.numpy()]
    tree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(wall_pts)

    phys = PhysicsConfig(phase="biochem")
    d_bar = 0.01
    u_ref = phys.get_u_ref(d_bar)

    u_bc = torch.zeros(n, 1)
    v_bc = torch.zeros(n, 1)
    u_bc[mask_in] = 1.0
    x_bio = build_biochem_bc_x_tensor(
        pos_nd=pos,
        sdf_nd=sdf,
        wall_normal=wall_n,
        mask_inlet=mask_in,
        mask_outlet=mask_out,
        mask_wall=mask_wall,
        u_bc=u_bc,
        v_bc=v_bc,
        p_bc=torch.zeros(n, 1),
        mu_bc_nd=torch.ones(n),
    )
    x_kine, _, _ = build_kinematics_node_x_tensor(
        pos_nd=pos,
        sdf_nd=sdf,
        wall_normal=wall_n,
        mask_inlet=mask_in,
        mask_outlet=mask_out,
        mask_wall=mask_wall,
        d_bar_si=d_bar,
        u_ref=u_ref,
        phys_cfg=phys,
        wall_tree=tree,
        edge_index=edge_index,
    )
    from torch_geometric.data import Data

    data = Data(
        x=x_kine,
        x_biochem=x_bio,
        y=torch.zeros(2, n, 16),
        edge_index=edge_index,
        edge_attr=edge_attr,
        mask_inlet=mask_in,
        mask_outlet=mask_out,
        mask_wall=mask_wall,
    )
    return attach_patient_anchor_graph_metadata(data, mask_wall=mask_wall)


def test_dual_x_metadata_and_alignment():
    data = _minimal_anchor_graph()
    assert data.x_schema == KINE_X_SCHEMA
    assert data.x_biochem_schema == BIO_X_SCHEMA
    assert int(data.x.shape[1]) == 18
    assert int(data.x_biochem.shape[1]) == 15
    assert_anchor_dual_x_aligned(data)
    assert biochem_encoder_x(data).shape == data.x_biochem.shape


def test_biochem_encoder_x_rejects_kine_only_graph():
    from torch_geometric.data import Data

    data = Data(x=torch.randn(8, 18))
    data.x_schema = KINE_X_SCHEMA
    with pytest.raises(ValueError, match="no x_biochem"):
        biochem_encoder_x(data)


@pytest.mark.skipif(
    not (__import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "processed" / "graphs_biochem_anchors" / "patient001.pt").exists(),
    reason="Re-extract anchor graphs to pick up dual-x layout.",
)
def test_on_disk_patient001_dual_x_schema():
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "data" / "processed" / "graphs_biochem_anchors" / "patient001.pt"
    data = torch.load(p, map_location="cpu", weights_only=False)
    if not hasattr(data, "x_biochem"):
        pytest.skip("patient001.pt predates dual-x extractor; re-run PatientDataExtractor.")
    assert data.x_schema == KINE_X_SCHEMA
    assert_anchor_dual_x_aligned(data)
    # Kinematics wall normals must not equal inlet mask columns (historical bug).
    inlet_mask_col = biochem_encoder_x(data)[:, 5]
    wrong_kine_norm = data.x[:, 4]
    assert float((wrong_kine_norm - inlet_mask_col).abs().max()) > 0.5 or float(inlet_mask_col.max()) < 0.5
