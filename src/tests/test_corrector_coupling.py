"""Sanity + shape/index safety for the corrector coupling loop intercept."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.config import NodeFeat, PhysicsConfig
from src.core_physics.coupled_shear_gnn import LocalKinematicCorrector
from src.inference.corrector_coupling import (
    clot_burden_significant,
    clot_nodes_from_delta_mu,
    couple_flow_with_corrector,
    get_coupled_flow,
    inject_mu_prior,
    kine_resolve_enabled,
    reset_coupled_flow_registry,
    set_coupled_flow,
    tile_clot_nodes,
    write_coupled_flow_into_y,
)


def _two_component_graph(device: torch.device) -> Data:
    """Component A = triangle {0,1,2}; component B = edge {3,4} (disconnected from A)."""
    x = torch.zeros(5, 16, dtype=torch.float32, device=device)
    x[:, 0] = torch.tensor([0.0, 1.0, 0.5, 5.0, 6.0], device=device)  # px
    x[:, 1] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device=device)  # py
    x[:, 2] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device=device)  # SDF (col 2)
    ei = torch.tensor(
        [[0, 1, 1, 2, 2, 0, 3, 4], [1, 0, 2, 1, 0, 2, 4, 3]], dtype=torch.long, device=device
    )
    data = Data(x=x, edge_index=ei)
    data.num_nodes = 5
    return data


def _const_diversion_corrector(du: float, dv: float, device: torch.device) -> LocalKinematicCorrector:
    """A corrector whose readout emits a constant ``[du, dv]`` for every node."""
    m = LocalKinematicCorrector().to(device).eval()
    with torch.no_grad():
        m.readout[-1].weight.zero_()
        m.readout[-1].bias.copy_(torch.tensor([du, dv], device=device))
    return m


def test_clot_nodes_threshold():
    delta = torch.tensor([0.0, 5e-4, 2e-3, 1e-2])
    nodes = clot_nodes_from_delta_mu(delta, min_delta_mu_si=1e-3)
    assert nodes.tolist() == [2, 3]


def test_no_clot_is_identity():
    dev = torch.device("cpu")
    data = _two_component_graph(dev)
    phys = PhysicsConfig(phase="kinematics")
    u0 = torch.linspace(0.1, 0.5, 5)
    v0 = torch.linspace(-0.2, 0.2, 5)
    corrector = _const_diversion_corrector(0.5, -0.3, dev)
    delta_mu = torch.zeros(5)  # no clot anywhere
    u, v = couple_flow_with_corrector(
        data, u0, v0, delta_mu, corrector=corrector, phys_cfg=phys, device=dev,
        num_hops=1, min_delta_mu_si=1e-3,
    )
    assert torch.allclose(u, u0)
    assert torch.allclose(v, v0)


def test_patch_only_on_subgraph():
    dev = torch.device("cpu")
    data = _two_component_graph(dev)
    phys = PhysicsConfig(phase="kinematics")
    u0 = torch.zeros(5)
    v0 = torch.zeros(5)
    corrector = _const_diversion_corrector(0.5, -0.3, dev)
    delta_mu = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    delta_mu[0] = 2e-3  # clot seeded on node 0 (component A)
    u, v = couple_flow_with_corrector(
        data, u0, v0, delta_mu, corrector=corrector, phys_cfg=phys, device=dev,
        num_hops=1, min_delta_mu_si=1e-3,
    )
    # 1-hop subgraph around node 0 in component A -> {0,1,2}; component B {3,4} untouched.
    assert torch.allclose(u[:3], torch.full((3,), 0.5))
    assert torch.allclose(v[:3], torch.full((3,), -0.3))
    assert torch.allclose(u[3:], torch.zeros(2))
    assert torch.allclose(v[3:], torch.zeros(2))
    assert u.shape == u0.shape and v.shape == v0.shape


def test_tile_clot_nodes_splits_by_radius():
    dev = torch.device("cpu")
    data = _two_component_graph(dev)
    pos = data.x[:, 0:2]
    clot = torch.tensor([0, 3], device=dev)  # node 0 at (0,0), node 3 at (5,0)
    two = tile_clot_nodes(pos, clot, radius_nd=0.5, max_per_cluster=64)
    assert len(two) == 2
    assert sorted(torch.cat(two).tolist()) == [0, 3]  # disjoint cover
    one = tile_clot_nodes(pos, clot, radius_nd=100.0, max_per_cluster=64)
    assert len(one) == 1 and sorted(one[0].tolist()) == [0, 3]
    capped = tile_clot_nodes(pos, clot, radius_nd=100.0, max_per_cluster=1)
    assert len(capped) == 2  # node cap forces a split even within the radius
    assert tile_clot_nodes(pos, torch.empty(0, dtype=torch.long, device=dev)) == []


def test_local_clusters_patch_each_clot(monkeypatch):
    """Two far-apart clots each get their own local patch (in-distribution application)."""
    monkeypatch.setenv("BIOCHEM_CORRECTOR_LOCAL_CLUSTERS", "1")
    monkeypatch.setenv("BIOCHEM_CORRECTOR_CLUSTER_RADIUS_ND", "0.5")
    dev = torch.device("cpu")
    data = _two_component_graph(dev)
    phys = PhysicsConfig(phase="kinematics")
    u0 = torch.zeros(5)
    v0 = torch.zeros(5)
    corrector = _const_diversion_corrector(0.5, -0.3, dev)
    delta_mu = torch.zeros(5)
    delta_mu[0] = 2e-3  # clot in component A
    delta_mu[3] = 2e-3  # clot in component B (far away -> separate cluster)
    u, v = couple_flow_with_corrector(
        data, u0, v0, delta_mu, corrector=corrector, phys_cfg=phys, device=dev,
        num_hops=1, min_delta_mu_si=1e-3,
    )
    # Each cluster patches its own 1-hop subgraph; non-overlapping -> averaged == single hit.
    assert torch.allclose(u, torch.full((5,), 0.5))
    assert torch.allclose(v, torch.full((5,), -0.3))


def test_registry_roundtrip():
    dev = torch.device("cpu")
    data = _two_component_graph(dev)
    reset_coupled_flow_registry()
    assert get_coupled_flow(data, dev) is None
    u = torch.arange(5, dtype=torch.float32)
    v = -torch.arange(5, dtype=torch.float32)
    set_coupled_flow(data, u, v)
    got = get_coupled_flow(data, dev)
    assert got is not None
    assert torch.allclose(got[0], u) and torch.allclose(got[1], v)
    reset_coupled_flow_registry()
    assert get_coupled_flow(data, dev) is None


def test_clot_burden_significant_thresholds(monkeypatch):
    monkeypatch.setenv("BIOCHEM_KINE_RESOLVE_ON_CLOT", "1")
    monkeypatch.setenv("BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES", "40")
    monkeypatch.setenv("BIOCHEM_KINE_RESOLVE_MIN_BAND_FRAC", "0.0")
    assert kine_resolve_enabled() is True
    assert clot_burden_significant(39, 1000) is False  # below node count, frac disabled
    assert clot_burden_significant(40, 1000) is True    # hits node-count trigger
    # fraction trigger
    monkeypatch.setenv("BIOCHEM_KINE_RESOLVE_MIN_CLOT_NODES", "100000")
    monkeypatch.setenv("BIOCHEM_KINE_RESOLVE_MIN_BAND_FRAC", "0.05")
    assert clot_burden_significant(60, 1000) is True    # 6% >= 5%
    assert clot_burden_significant(40, 1000) is False   # 4% < 5%
    # disabled master switch -> never significant
    monkeypatch.setenv("BIOCHEM_KINE_RESOLVE_ON_CLOT", "0")
    assert clot_burden_significant(100000, 1000) is False


def test_inject_mu_prior_sets_channel():
    dev = torch.device("cpu")
    data = _two_component_graph(dev)
    phys = PhysicsConfig(phase="kinematics")
    mu = torch.full((5,), 0.7)
    data_k = inject_mu_prior(data, mu, phys)
    # original graph untouched; clone carries mu in MU_PRIOR (ND)
    assert torch.allclose(data.x[:, NodeFeat.MU_PRIOR].reshape(-1), torch.zeros(5))
    expected_nd = phys.viscosity_si_to_nd(mu)
    assert torch.allclose(data_k.x[:, NodeFeat.MU_PRIOR].reshape(-1), expected_nd)


def test_write_coupled_flow_into_y():
    dev = torch.device("cpu")
    data = _two_component_graph(dev)
    data.y = torch.zeros(3, 5, 16, dtype=torch.float32)  # (T, N, C)
    u = torch.full((5,), 0.7)
    v = torch.full((5,), -0.4)
    write_coupled_flow_into_y(data, u, v, time_index=1)
    assert torch.allclose(data.y[1, :, 0], u)
    assert torch.allclose(data.y[1, :, 1], v)
    assert torch.allclose(data.y[0, :, 0], torch.zeros(5))  # other times untouched
    write_coupled_flow_into_y(data, u, v, time_index=None)
    assert torch.allclose(data.y[:, :, 0], u.unsqueeze(0).expand(3, 5))
    assert torch.allclose(data.y[:, :, 1], v.unsqueeze(0).expand(3, 5))
