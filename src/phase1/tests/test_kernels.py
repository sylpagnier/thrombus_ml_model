import pytest
import torch
import os
from torch_geometric.data import Data
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig


# ==========================================
# CONFIG & FIXTURES
# ==========================================

@pytest.fixture(params=["tier1", "tier2"], scope="module")
def phys_cfg(request):
    """
    Parameterized fixture: Runs every dependent test twice.
    First pass: Tier 1 -> Sets viscosity_model to "newtonian" via post_init
    Second pass: Tier 2 -> Sets viscosity_model to "carreau" via post_init
    """
    return PhysicsConfig(
        re_target=150.0,
        tier=request.param
    )


@pytest.fixture(scope="module")
def shared_test_graph():
    """
    Generates a structured grid [0, 4] x [-0.5, 0.5].
    Structured grids are ideal for verifying mathematical exactness.
    """
    x = torch.linspace(0, 4, 40)
    y = torch.linspace(-0.5, 0.5, 20)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    dist = torch.cdist(nodes, nodes)
    edge_index = (dist < 0.15).nonzero().t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    data = Data(x=nodes, edge_index=edge_index, num_nodes=nodes.shape[0])

    # Dynamic Normalization Layer mocks
    data.u_ref = 1.0
    data.d_bar = 1.0
    data.mask_wall = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.mask_inlet = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.mask_outlet = torch.zeros(data.num_nodes, dtype=torch.bool)

    return data


# ==========================================
# MATHEMATICAL VERIFICATION TESTS
# ==========================================

def test_polynomial_exactness(shared_test_graph, phys_cfg):
    """
    CRITICAL TEST: The WLS kernel uses a quadratic basis [dx, dy, dx^2, dxy, dy^2].
    It MUST be able to perfectly reconstruct the derivatives of a 2nd order polynomial.
    """
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    x, y = data.x[:, 0], data.x[:, 1]
    u = 2 * x ** 2 + 3 * y ** 2 - x * y

    c = kernels._compute_derivatives(u.unsqueeze(1), props)

    u_x_pred = c[:, 0, 0]
    u_xx_pred = c[:, 2, 0]
    u_xy_pred = c[:, 3, 0]

    u_x_true = 4 * x - y
    u_xx_true = torch.full_like(x, 4.0)
    u_xy_true = torch.full_like(x, -1.0)

    interior_mask = (x > 0.2) & (x < 3.8) & (y > -0.4) & (y < 0.4)

    mse_x = torch.nn.functional.mse_loss(u_x_pred[interior_mask], u_x_true[interior_mask])
    mse_xx = torch.nn.functional.mse_loss(u_xx_pred[interior_mask], u_xx_true[interior_mask])
    mse_xy = torch.nn.functional.mse_loss(u_xy_pred[interior_mask], u_xy_true[interior_mask])

    assert mse_x < 1e-4, "Failed to reconstruct 1st derivative of polynomial"
    assert mse_xx < 1e-3, "Failed to reconstruct 2nd derivative of polynomial"
    assert mse_xy < 1e-3, "Failed to reconstruct mixed derivative"


def test_carreau_viscosity_logic(phys_cfg):
    """
    Tests that the Carreau-Yasuda logic correctly computes effective viscosity.
    Skips the heavy assertions if testing the Newtonian tier.
    """
    kernels = PhysicsKernels(phys_cfg)

    # Mock strain rates: [u_x, u_y, v_x, v_y]
    zero_shear = torch.zeros((1, 4))
    high_shear = torch.tensor([[1000.0, 0.0, 0.0, 1000.0]])
    du_ij = torch.cat([zero_shear, high_shear], dim=0)

    mu_eff = kernels._compute_carreau_viscosity(du_ij, u_ref=1.0, d_bar=1.0)

    if phys_cfg.viscosity_model == "carreau":
        mu_0_nd = phys_cfg.mu_0 / phys_cfg.mu_inf
        mu_inf_nd = phys_cfg.mu_inf / phys_cfg.mu_inf  # Should be 1.0

        assert torch.isclose(mu_eff[0], torch.tensor(mu_0_nd), atol=1e-5), "Zero shear viscosity incorrect"
        assert mu_eff[1] < mu_eff[0], "Viscosity did not shear-thin"
        assert mu_eff[1] >= mu_inf_nd, "Viscosity dropped below mu_inf limit"


def test_rheology_loss_exactness(shared_test_graph, phys_cfg):
    """
    Tier 1: Verifies rheology loss strictly returns 0.0.
    Tier 2: Ensures the rheology_loss returns ~0 when predicted viscosity perfectly matches analytical.
    """
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    x, y = data.x[:, 0], data.x[:, 1]
    u = y ** 2
    v = torch.zeros_like(x)
    p = torch.zeros_like(x)

    if phys_cfg.viscosity_model == "newtonian":
        pred = torch.stack([u, v, p], dim=1)
        loss = kernels.rheology_loss(pred, data, props)
        assert loss.item() == 0.0, "Tier 1 Rheology loss must be exactly 0.0"
    else:
        c_u = kernels._compute_derivatives(u.unsqueeze(1), props)
        c_v = kernels._compute_derivatives(v.unsqueeze(1), props)
        du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)

        mu_target = kernels._compute_carreau_viscosity(du_ij, data.u_ref, data.d_bar)
        pred = torch.stack([u, v, p, mu_target], dim=1)

        loss = kernels.rheology_loss(pred, data, props)
        assert loss.item() < 1e-6, "Tier 2 Rheology loss should be nearly zero for perfectly predicted viscosity"


def test_navier_stokes_residual_format(shared_test_graph, phys_cfg):
    """
    Ensures the residual function handles both 3-channel (Tier 1)
    and 4-channel (Tier 2) predictions without shape mismatches.
    """
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)

    num_channels = 4 if phys_cfg.viscosity_model == "carreau" else 3
    pred = torch.randn((data.num_nodes, num_channels))
    props = kernels._get_geometric_props(data)

    loss = kernels.navier_stokes_residual(pred, data, props)

    assert loss.dim() == 0, "Loss must be a scalar"
    assert not torch.isnan(loss), "Loss became NaN"
    assert loss > 0, "Residual of random noise should be positive"


# ==========================================
# INTEGRATION TESTS (REAL DATA)
# ==========================================

def test_laplacian_on_file(phys_cfg):
    cfg = VesselConfig()
    graph_dir = cfg.graph_output_dir

    if not os.path.exists(graph_dir):
        pytest.skip("Graph directory missing.")
    existing_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    if not existing_files:
        pytest.skip("No .pt files found.")

    data = torch.load(os.path.join(graph_dir, existing_files[0]), weights_only=False)
    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    y = data.x[:, 1]
    u = y ** 2
    c = kernels._compute_derivatives(u.unsqueeze(1), props)
    u_yy = c[:, 4, 0]

    if data.x.shape[1] > 2:
        sdf = data.x[:, 2]
        interior_mask = sdf > 0.08
    else:
        interior_mask = (data.x[:, 0] > data.x[:, 0].min() + 0.1)

    if interior_mask.sum() == 0:
        pytest.skip("No interior nodes found for Laplacian test")

    mse_lap = torch.nn.functional.mse_loss(
        u_yy[interior_mask],
        torch.full_like(u_yy[interior_mask], 2.0)
    )

    assert mse_lap < 0.5, f"Laplacian error too high on real mesh: {mse_lap.item()}"


def test_full_physics_pipeline_validity(phys_cfg):
    """
    Validates the Navier-Stokes residual on Ground Truth data,
    dynamically constructing either the 3-channel or 4-channel tensor.
    """
    v_cfg = VesselConfig()
    graph_dir = v_cfg.graph_output_dir

    if not os.path.exists(graph_dir):
        pytest.skip("Graph directory missing.")
    existing_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    if not existing_files:
        pytest.skip("No .pt files found.")

    data = torch.load(os.path.join(graph_dir, existing_files[0]), weights_only=False)

    if not hasattr(data, 'u_ref'): data.u_ref = 1.0
    if not hasattr(data, 'd_bar'): data.d_bar = 1.0

    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    if not hasattr(data, 'y') or data.y is None:
        pytest.skip("File does not contain ground truth 'y'.")

    u_gt, v_gt, p_gt = data.y[:, 0], data.y[:, 1], data.y[:, 2]

    if phys_cfg.viscosity_model == "carreau":
        c_u = kernels._compute_derivatives(u_gt.unsqueeze(1), props)
        c_v = kernels._compute_derivatives(v_gt.unsqueeze(1), props)
        du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)
        mu_gt = kernels._compute_carreau_viscosity(du_ij, data.u_ref, data.d_bar)

        comsol_pred = torch.stack([u_gt, v_gt, p_gt, mu_gt], dim=1)
    else:
        comsol_pred = torch.stack([u_gt, v_gt, p_gt], dim=1)

    loss_comsol = kernels.navier_stokes_residual(comsol_pred, data, props).item()

    noise_pred = torch.randn_like(comsol_pred)
    loss_noise = kernels.navier_stokes_residual(noise_pred, data, props).item()

    assert loss_comsol < loss_noise * 0.1, "Physics loss cannot distinguish Ground Truth from Noise!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])