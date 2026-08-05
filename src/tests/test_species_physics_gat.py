"""Tests for physics-biased species GAT trunk (Stage-A faithful)."""

from __future__ import annotations

import pytest
import torch

from src.architecture.pushforward_config import PushforwardConfig, use_pushforward_config
from src.architecture.runtime_config import BiochemRuntimeConfig, use_biochem_runtime
from src.biochem_gnn.mat_growth_simple import mat_growth_leg_spec
from src.core_physics.species_deploy_rollout import band_uv_for_model
from src.core_physics.species_physics_gat import (
    GEOM_EDGE_DIM,
    PHYSICS_EDGE_DIM,
    SpeciesPhysicsGATConv,
    build_physics_edge_bundle,
    build_physics_edge_priors,
    estimate_wall_normals,
    geometric_edge_attr,
)
from src.core_physics.species_snapshot_gnn import SpeciesSnapshotGNN


def _toy_band(n: int = 6):
    # Line of nodes: 0,1 wall; 2..n-1 fluid. Bidirectional edges.
    pos = torch.tensor([[float(i), 0.0] for i in range(n)], dtype=torch.float32)
    wall = torch.zeros(n, dtype=torch.bool)
    wall[:2] = True
    rows, cols = [], []
    for i in range(n - 1):
        rows.extend([i, i + 1])
        cols.extend([i + 1, i])
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    normals = torch.zeros((n, 2), dtype=torch.float32)
    normals[:, 0] = 1.0
    sdf = torch.tensor([0.0, 0.0] + [float(i - 1) * 0.1 for i in range(2, n)], dtype=torch.float32)
    return pos, wall, edge_index, normals, sdf


def test_estimate_wall_normals_points_into_fluid():
    pos, wall, edge_index, _, _ = _toy_band()
    nrm = estimate_wall_normals(pos, wall, edge_index)
    assert nrm.shape == (pos.size(0), 2)
    assert float(nrm[3, 0]) > 0.5


def test_build_physics_edge_bundle_mesh_normals_sdf():
    pos, wall, edge_index, normals, sdf = _toy_band()
    bundle = build_physics_edge_bundle(
        edge_index,
        pos=pos,
        wall_normals=normals,
        sdf=sdf,
        wall_mask=wall,
    )
    assert bundle.edge_attr.shape == (edge_index.size(1), GEOM_EDGE_DIM)
    assert bundle.mod_rheo.shape == (edge_index.size(1), 1)
    assert float(bundle.mod_rheo.abs().sum() + bundle.mod_adv.abs().sum()) > 0.0
    geom = geometric_edge_attr(edge_index, pos)
    assert torch.allclose(bundle.edge_attr, geom, atol=1e-6)


def test_ensure_band_mesh_priors_uses_nodefeat_wall_normal():
    """data.x is KINE layout: WALL_NORMAL is NodeFeat slice(4,6), not BiochemNodeFeat(3,5)."""
    from src.config import NodeFeat
    from src.core_physics.species_pushforward_gnn import ensure_band_mesh_priors

    n = 4
    x = torch.zeros(n, int(NodeFeat.WIDTH_D2.stop))
    # Deliberately different values so wrong slice cannot accidentally match.
    x[:, 3] = 9.0  # SHEAR_POT (BiochemNodeFeat would steal this as n_x)
    x[:, NodeFeat.WALL_NORMAL] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    x[:, NodeFeat.SDF] = torch.arange(n, dtype=torch.float32).unsqueeze(1) * 0.1

    class _Data:
        num_nodes = n

        def __init__(self):
            self.x = x

    node_idx = torch.arange(n)
    static = {
        "node_idx": node_idx,
        "wall_normals_band": None,
        "sdf_band": None,
        "edge_attr_band": geometric_edge_attr(
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long), x[:, :2]
        ),
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "pos_band": x[:, :2],
    }
    out = ensure_band_mesh_priors(static, _Data())
    assert torch.allclose(out["wall_normals_band"], x[node_idx, NodeFeat.WALL_NORMAL])
    assert float(out["wall_normals_band"][0, 0]) == 1.0
    assert float(out["wall_normals_band"][0, 1]) == 0.0
    # Wrong BiochemNodeFeat slice would have put 9.0 in n_x.
    assert abs(float(out["wall_normals_band"][0, 0]) - 9.0) > 1.0


def test_build_physics_edge_priors_ignores_velocity():
    pos, wall, edge_index, _, _ = _toy_band()
    ea = build_physics_edge_priors(
        edge_index, pos, velocity=torch.ones_like(pos), wall_mask=wall
    )
    assert ea.shape[-1] == PHYSICS_EDGE_DIM


def test_species_physics_gat_conv_stage_a_message():
    conv = SpeciesPhysicsGATConv(4, 8, priors_multiply_before_add=True, prior_scale=0.05)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    pos = torch.randn(5, 2)
    nrm = torch.randn(5, 2)
    nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sdf = torch.rand(5)
    bundle = build_physics_edge_bundle(edge_index, pos=pos, wall_normals=nrm, sdf=sdf)
    out = conv(x, edge_index, bundle=bundle)
    assert out.shape == (5, 8)


def test_physics_gat_edge_proj_identity_survives_snapshot_xavier():
    """Blanket Xavier must not leave edge_proj≈0 (wipes content under multiply-before-add)."""
    cfg = PushforwardConfig(arch="physics_gat", physics_gat_prior_scale=0.05)
    with use_pushforward_config(cfg):
        gnn = SpeciesSnapshotGNN(in_dim=8, hidden=16, out_dim=2, arch="physics_gat")
    for name in ("conv1", "conv2", "conv3"):
        conv = getattr(gnn, name)
        assert float(conv.edge_proj.weight.abs().max()) == 0.0
        assert torch.allclose(conv.edge_proj.bias, torch.ones_like(conv.edge_proj.bias))
        assert abs(float(conv.prior_scale) - 0.05) < 1e-8


def test_snapshot_gnn_physics_gat_forward_with_typed_config():
    cfg = PushforwardConfig(arch="physics_gat", dual_head=True, physics_gat_prior_scale=0.05)
    with use_pushforward_config(cfg):
        gnn = SpeciesSnapshotGNN(in_dim=8, hidden=16, out_dim=2, arch="physics_gat")
        pos, wall, edge_index, normals, sdf = _toy_band(n=8)
        ea = geometric_edge_attr(edge_index, pos)
        gnn.set_band_geometry(
            pos, edge_index, wall, wall_normals=normals, sdf=sdf, edge_attr=ea
        )
        x = torch.randn(8, 8)
        h = gnn.forward_hidden(x, edge_index)
        assert h.shape == (8, 16)
        y = gnn(x, edge_index)
        assert y.shape == (8, 2)


def test_wg_physgat_01_leg_kwargs():
    leg = mat_growth_leg_spec("WG_physgat_01")
    feat = mat_growth_leg_spec("WG_featfix_03")
    ctrl = mat_growth_leg_spec("WG_physgat_ctrl")
    assert leg.config_kwargs.get("arch") == "physics_gat"
    assert float(leg.config_kwargs.get("physics_gat_prior_scale", 0.0)) == pytest.approx(0.05)
    assert leg.no_init is True
    assert ctrl.config_kwargs.get("arch") == "sage"
    assert ctrl.no_init is True
    for k in ("geom_feats", "geom_feats_rich", "flux_stag_feat", "flow_feats_drop_xy", "flow_feats_source"):
        assert leg.config_kwargs.get(k) == feat.config_kwargs.get(k)
        assert ctrl.config_kwargs.get(k) == feat.config_kwargs.get(k)
    assert leg.runtime_kwargs.get("corrector_coupling") is True


def test_pushforward_config_accepts_physics_gat():
    cfg = PushforwardConfig(arch="physics_gat", physics_gat_prior_scale=0.05)
    cfg.validate()
    assert cfg.to_meta_dict()["arch"] == "physics_gat"
    assert float(cfg.to_meta_dict()["physics_gat_prior_scale"]) == pytest.approx(0.05)


def test_band_uv_for_model_uses_u0_pred_not_gt_when_coupled():
    """Deploy-faithful helper must not silently return COMSOL y UV."""

    class _Data:
        num_nodes = 4

        def __init__(self):
            self.y = torch.zeros(3, 4, 16)
            self.y[:, :, 0] = 9.0
            self.y[:, :, 1] = -9.0
            self.u0_pred = torch.full((4,), 0.25)
            self.v0_pred = torch.full((4,), -0.1)

    data = _Data()
    node_idx = torch.arange(4)
    rt = BiochemRuntimeConfig().with_overrides(
        train_vel_source="coupled",
        rollout_vel_source="coupled",
        corrector_coupling=False,
    )
    with use_biochem_runtime(rt):
        uv = band_uv_for_model(data, 0, torch.device("cpu"), node_idx, for_training=True)
    assert uv.shape == (4, 2)
    assert float(uv[:, 0].mean()) == pytest.approx(0.25)
    assert float(uv[:, 1].mean()) == pytest.approx(-0.1)
    assert abs(float(uv[:, 0].mean()) - 9.0) > 1.0
