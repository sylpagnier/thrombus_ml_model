"""COMSOL gel-scaled Carreau (spf.mu) physics oracle."""

import os

import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_simple import (
    build_clot_phi_step,
    carreau_mu_si_from_uv,
    comsol_carreau_mu_si_from_uv,
    physics_mu_eff_si,
)


@pytest.fixture
def anchor_graph():
    paths = list(__import__("pathlib").Path("data/processed/graphs_biochem_anchors").glob("patient*.pt"))
    if not paths:
        pytest.skip("no biochem anchor graphs")
    return torch.load(str(paths[0]), map_location="cpu", weights_only=False)


def test_comsol_carreau_bulk_closer_than_fixed_carreau(anchor_graph):
    """Gel-scaled Carreau with max(g,wls,poi,kin) gamma should beat fixed-mu Carreau vs GT bulk."""
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = anchor_graph
    device = torch.device("cpu")
    t = min(35, int(data.y.shape[0]) - 1)
    y = data.y[t]
    u, v = y[:, 0], y[:, 1]
    mu_gt = phys.viscosity_nd_to_si(y[:, 3])
    mu_c = carreau_mu_si_from_uv(data, u, v, phys)
    m_one = torch.ones(int(data.num_nodes), dtype=torch.float32)
    mu_comsol = comsol_carreau_mu_si_from_uv(
        data, u, v, m_one, phys, device=device, gamma_mode="max"
    )
    bulk = mu_gt < 0.012
    if not bulk.any():
        pytest.skip("no bulk nodes")
    err_fixed = (mu_gt[bulk] - mu_c[bulk]).abs().median()
    err_comsol = (mu_gt[bulk] - mu_comsol[bulk]).abs().median()
    assert err_comsol < err_fixed
    ratio = (mu_gt[bulk] / mu_comsol[bulk].clamp(min=1e-8)).median()
    assert 0.85 <= float(ratio) <= 1.15


def test_kinematic_gamma_bulk_matches_gt(anchor_graph):
    """``|u|/width`` ND shear gives COMSOL bulk ``spf.mu`` scale at M=1."""
    phys = PhysicsConfig(phase="biochem")
    data = anchor_graph
    device = torch.device("cpu")
    t = 0
    y = data.y[t]
    u, v = y[:, 0], y[:, 1]
    mu_gt = phys.viscosity_nd_to_si(y[:, 3])
    bulk = mu_gt < 0.012
    if not bulk.any():
        pytest.skip("no bulk nodes")
    m_one = torch.ones(int(data.num_nodes), dtype=torch.float32)
    mu = comsol_carreau_mu_si_from_uv(data, u, v, m_one, phys, device=device, gamma_mode="max")
    ratio = (mu_gt[bulk] / mu[bulk].clamp(min=1e-8)).median()
    assert 0.85 <= float(ratio) <= 1.15


def test_comsol_sr_sidecar_reproduces_gt_mu():
    """Oracle: Carreau(spf.sr) must match GT spf.mu when sidecar is present."""
    import os
    from pathlib import Path

    sidecar = Path("data/processed/cfd_results_biochem_diag/patient007_sr.pt")
    graph_path = Path("data/processed/graphs_biochem_anchors/patient007.pt")
    if not sidecar.is_file() or not graph_path.is_file():
        pytest.skip("patient007 sr sidecar / graph not available")
    os.environ["CLOT_PHI_PHYSICS_MU_BASE"] = "comsol_carreau"
    os.environ["CLOT_PHI_PHYSICS_GAMMA_MODE"] = "comsol_sr"
    os.environ["CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR"] = "patient007"
    try:
        phys = PhysicsConfig(phase="biochem")
        data = torch.load(str(graph_path), map_location="cpu", weights_only=False)
        device = torch.device("cpu")
        t = 0
        y = data.y[t]
        mu_gt = phys.viscosity_nd_to_si(y[:, 3])
        m_one = torch.ones(int(data.num_nodes), dtype=torch.float32)
        mu = comsol_carreau_mu_si_from_uv(
            data, y[:, 0], y[:, 1], m_one, phys, device=device, gamma_mode="comsol_sr", time_index=t
        )
        bulk = mu_gt < 0.012
        ratio = (mu_gt[bulk] / mu[bulk].clamp(min=1e-8)).median()
        assert 0.98 <= float(ratio) <= 1.02
    finally:
        os.environ.pop("CLOT_PHI_PHYSICS_MU_BASE", None)
        os.environ.pop("CLOT_PHI_PHYSICS_GAMMA_MODE", None)
        os.environ.pop("CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR", None)


def test_physics_mu_comsol_carreau_mode(anchor_graph):
    os.environ["CLOT_PHI_PHYSICS_MU_BASE"] = "comsol_carreau"
    os.environ["CLOT_PHI_PHYSICS_GAMMA_MODE"] = "max"
    os.environ["CLOT_PHI_PHYSICS_MU_RATIO_MAX"] = "4"
    try:
        bio = BiochemConfig(phase="biochem")
        phys = PhysicsConfig(phase="biochem")
        device = torch.device("cpu")
        step = build_clot_phi_step(anchor_graph, 0, phys, bio, device)
        mu = physics_mu_eff_si(
            step.mu_c_si,
            step.species_log_gt,
            bio,
            device=device,
            data=anchor_graph,
            u_nd=step.u_flow_nd,
            v_nd=step.v_flow_nd,
            phys_cfg=phys,
            time_index=0,
        )
        assert bool((mu > 0).all())
        assert float(mu.median()) < float(phys.mu_0)
    finally:
        os.environ.pop("CLOT_PHI_PHYSICS_MU_BASE", None)
        os.environ.pop("CLOT_PHI_PHYSICS_GAMMA_MODE", None)
        os.environ.pop("CLOT_PHI_PHYSICS_MU_RATIO_MAX", None)
