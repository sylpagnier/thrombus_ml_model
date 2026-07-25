"""Regression tests for CorrectorCoupledFlow base-flow cache vs remesh."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.inference.corrector_coupling import (
    CorrectorCoupledFlow,
    couple_flow_with_corrector,
)


class _IdentityCorrector(torch.nn.Module):
    def forward(self, x_sub, _edge_index):
        # Zero diversion so couple only exercises indexing / shapes.
        return torch.zeros(x_sub.shape[0], 2, device=x_sub.device, dtype=x_sub.dtype)


def _toy_graph(n: int, *, u_scale: float) -> Data:
    # Simple line mesh with wall-ish features enough for couple path.
    pos = torch.zeros(n, 2, dtype=torch.float32)
    pos[:, 0] = torch.linspace(0.0, 1.0, n)
    # Minimal x layout: at least 2 coords; MU_PRIOR unused when u0_pred is set.
    x = torch.zeros(n, 8, dtype=torch.float32)
    x[:, 0:2] = pos
    row = torch.arange(0, n - 1, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    data = Data(x=x, edge_index=edge_index, num_nodes=n)
    data.u0_pred = torch.full((n,), float(u_scale), dtype=torch.float32)
    data.v0_pred = torch.zeros(n, dtype=torch.float32)
    return data


def test_base_flow_cache_invalidates_on_remesh_node_count():
    """Customer parametric remesh changes N; cached base must not leak across meshes."""
    provider = CorrectorCoupledFlow(device=torch.device("cpu"))
    g0 = _toy_graph(40, u_scale=1.0)
    u0, v0 = provider.base_flow(g0)
    assert int(u0.numel()) == 40
    assert float(u0[0].item()) == 1.0

    g1 = _toy_graph(25, u_scale=2.0)
    u1, v1 = provider.base_flow(g1)
    assert int(u1.numel()) == 25
    assert float(u1[0].item()) == 2.0
    # Same object returned on second call for same graph.
    u1b, _ = provider.base_flow(g1)
    assert u1b.data_ptr() == u1.data_ptr()


def test_couple_rejects_mismatched_base_flow_size():
    g = _toy_graph(20, u_scale=1.0)
    u_wrong = torch.ones(30)
    v_wrong = torch.zeros(30)
    delta = torch.zeros(20)
    delta[5:10] = 1.0
    try:
        couple_flow_with_corrector(
            g,
            u_wrong,
            v_wrong,
            delta,
            corrector=_IdentityCorrector(),
            phys_cfg=__import__("src.config", fromlist=["PhysicsConfig"]).PhysicsConfig(
                phase="kinematics"
            ),
            device=torch.device("cpu"),
            num_hops=1,
            min_delta_mu_si=0.1,
        )
        raised = False
    except ValueError as exc:
        raised = True
        assert "does not match data.num_nodes" in str(exc)
    assert raised


def test_invalidate_base_cache_forces_refresh():
    provider = CorrectorCoupledFlow(device=torch.device("cpu"))
    g = _toy_graph(12, u_scale=3.0)
    u_a, _ = provider.base_flow(g)
    provider.invalidate_base_cache()
    g.u0_pred = torch.full((12,), 9.0)
    g.v0_pred = torch.zeros(12)
    u_b, _ = provider.base_flow(g)
    assert float(u_a[0].item()) == 3.0
    assert float(u_b[0].item()) == 9.0
