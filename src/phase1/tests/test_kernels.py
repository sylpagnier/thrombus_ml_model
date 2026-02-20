import pytest
import torch
import os
from torch_geometric.data import Data
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig


# ==========================================
# ACTUAL CONFIG & FIXTURES
# ==========================================

@pytest.fixture(params=["tier1", "tier2"], scope="module")
def tier(request):
    """Parameterizes the tier to test both Newtonian and Carreau sequentially."""
    return request.param


@pytest.fixture(scope="module")
def phys_cfg(tier):
    """Loads the actual PhysicsConfig. __post_init__ handles the logic."""
    return PhysicsConfig(tier=tier)


@pytest.fixture(scope="module")
def vessel_cfg(tier):
    """Loads the actual VesselConfig to provide correct file paths."""
    return VesselConfig(tier=tier)


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

    # Dynamic Normalization Layer mocks (Must be tensors for batch indexing)
    data.u_ref = torch.tensor([1.0])
    data.d_bar = torch.tensor([1.0])
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    # Boundary Condition mocks
    data.mask_wall = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.mask_inlet = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.mask_outlet = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.u_inlet_bc = torch.ones((data.num_nodes, 2)) * 0.5

    return data


# ==========================================
# MATHEMATICAL VERIFICATION TESTS
# ==========================================

def test_polynomial_exactness(shared_test_graph, phys_cfg):
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    x, y = data.x[:, 0], data.x[:, 1]
    u = 2 * x ** 2 + 3 * y ** 2 - x * y

    c = kernels._compute_derivatives(u.unsqueeze(1), props)

    u_x_pred, u_xx_pred, u_xy_pred = c[:, 0, 0], c[:, 2, 0], c[:, 3, 0]
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
    kernels = PhysicsKernels(phys_cfg)

    # Mock strain rates: [u_x, u_y, v_x, v_y]
    zero_shear = torch.zeros((1, 4))
    high_shear = torch.tensor([[1000.0, 0.0, 0.0, 1000.0]])
    du_ij = torch.cat([zero_shear, high_shear], dim=0)

    mock_data = Data()
    mock_data.u_ref = torch.tensor([1.0])
    mock_data.d_bar = torch.tensor([1.0])
    mock_data.batch = torch.zeros(du_ij.size(0), dtype=torch.long)

    if phys_cfg.viscosity_model == "carreau":
        mu_eff = kernels._compute_carreau_viscosity(du_ij, mock_data)
        # Checking shear-thinning behavior
        assert mu_eff[1] < mu_eff[0], "Viscosity did not shear-thin under high strain"


def test_navier_stokes_residual_format(shared_test_graph, phys_cfg):
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)

    num_channels = 4 if phys_cfg.viscosity_model == "carreau" else 3
    pred = torch.randn((data.num_nodes, num_channels))
    props = kernels._get_geometric_props(data)

    # Ensure no crashes on format
    loss = kernels.navier_stokes_residual(pred, data, props)
    assert loss is not None and not torch.isnan(loss)


def test_rheology_loss_exactness(shared_test_graph, phys_cfg):
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

        mu_target = kernels._compute_carreau_viscosity(du_ij, data)
        pred = torch.stack([u, v, p, mu_target], dim=1)

        loss = kernels.rheology_loss(pred, data, props)
        assert loss.item() < 1e-6, "Tier 2 Rheology loss should be nearly zero for perfectly predicted viscosity"


# ==========================================
# BOUNDARY CONDITION TESTS
# ==========================================

def test_boundary_condition_loss(shared_test_graph, phys_cfg):
    data = shared_test_graph.clone()
    kernels = PhysicsKernels(phys_cfg)

    # Activate walls on the first 10 nodes
    data.mask_wall[:10] = True
    num_channels = 4 if phys_cfg.viscosity_model == "carreau" else 3

    # Test violation
    pred_bad = torch.ones((data.num_nodes, num_channels))
    loss_bad = kernels.boundary_condition_loss(pred_bad, data)
    assert loss_bad.item() > 0.0, "Boundary loss failed to penalize non-zero wall velocity"

    # Test compliance
    pred_good = torch.zeros((data.num_nodes, num_channels))
    loss_good = kernels.boundary_condition_loss(pred_good, data)
    assert loss_good.item() == 0.0, "Perfect boundary compliance should yield 0 loss"


def test_inlet_outlet_loss(shared_test_graph, phys_cfg):
    data = shared_test_graph.clone()
    kernels = PhysicsKernels(phys_cfg)

    data.mask_inlet[:5] = True
    data.mask_outlet[-5:] = True
    num_channels = 4 if phys_cfg.viscosity_model == "carreau" else 3

    # Perfect match: u matches target 0.5, v is 0, p is 0 at outlet
    pred_good = torch.zeros((data.num_nodes, num_channels))
    pred_good[:, 0] = 0.5
    loss_good = kernels.inlet_outlet_loss(pred_good, data)
    assert loss_good.item() == 0.0, "Perfect inlet/outlet compliance should yield 0 loss"

    # Bad match
    pred_bad = torch.zeros((data.num_nodes, num_channels))
    pred_bad[:, 0] = 1.0  # Wrong u
    pred_bad[-5:, 2] = 5.0  # Wrong p at outlet
    loss_bad = kernels.inlet_outlet_loss(pred_bad, data)
    assert loss_bad.item() > 0.0, "Inlet/Outlet loss failed to penalize deviations"


# ==========================================
# INTEGRATION TESTS (REAL DATA)
# ==========================================

def test_full_physics_pipeline_validity(phys_cfg, vessel_cfg):
    """Requires both configs: VesselConfig for paths, PhysicsConfig for kernels."""
    graph_dir = vessel_cfg.graph_output_dir

    if not os.path.exists(graph_dir):
        pytest.skip("Graph directory missing.")
    existing_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    if not existing_files:
        pytest.skip("No .pt files found.")

    data = torch.load(os.path.join(graph_dir, existing_files[0]), weights_only=False)

    if not hasattr(data, 'u_ref'): data.u_ref = torch.tensor([1.0])
    if not hasattr(data, 'd_bar'): data.d_bar = torch.tensor([1.0])
    if not hasattr(data, 'batch'): data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    if not hasattr(data, 'y') or data.y is None:
        pytest.skip("File does not contain ground truth 'y'.")

    u_gt, v_gt, p_gt = data.y[:, 0], data.y[:, 1], data.y[:, 2]

    if phys_cfg.viscosity_model == "carreau":
        c_u = kernels._compute_derivatives(u_gt.unsqueeze(1), props)
        c_v = kernels._compute_derivatives(v_gt.unsqueeze(1), props)
        du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)
        mu_gt = kernels._compute_carreau_viscosity(du_ij, data)

        comsol_pred = torch.stack([u_gt, v_gt, p_gt, mu_gt], dim=1)
    else:
        comsol_pred = torch.stack([u_gt, v_gt, p_gt], dim=1)

    loss_comsol = kernels.navier_stokes_residual(comsol_pred, data, props).item()
    noise_pred = torch.randn_like(comsol_pred)
    loss_noise = kernels.navier_stokes_residual(noise_pred, data, props).item()

    assert loss_comsol < loss_noise * 0.1, "Physics loss cannot distinguish Ground Truth from Noise!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])