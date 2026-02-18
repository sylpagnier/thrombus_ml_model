import pytest
import torch
import os
import math
from torch_geometric.data import Data
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig


# ==========================================
# CONFIG & FIXTURES
# ==========================================

@pytest.fixture(scope="module")
def phys_cfg():
    return PhysicsConfig(re_target=150.0, viscosity_model="carreau")


@pytest.fixture(scope="module")
def shared_test_graph():
    """
    Generates a structured grid [0, 4] x [-0.5, 0.5].
    Structured grids are ideal for verifying mathematical exactness.
    """
    # Create a grid
    x = torch.linspace(0, 4, 40)
    y = torch.linspace(-0.5, 0.5, 20)  # Increased density slightly
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # Create edges (radius connectivity)
    dist = torch.cdist(nodes, nodes)
    edge_index = (dist < 0.15).nonzero().t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    data = Data(x=nodes, edge_index=edge_index, num_nodes=nodes.shape[0])

    # Inject attributes required by the new kernel
    data.u_ref = 1.0
    data.d_bar = 1.0
    data.mask_wall = torch.zeros(data.num_nodes, dtype=torch.bool)  # Dummy mask
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

    Test function: u = 2*x^2 + 3*y^2 - x*y
    Expected Derivs:
      u_x  = 4x - y
      u_y  = 6y - x
      u_xx = 4
      u_yy = 6
      u_xy = -1
    """
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    x, y = data.x[:, 0], data.x[:, 1]

    # Define exact polynomial field
    u = 2 * x ** 2 + 3 * y ** 2 - x * y

    # Compute derivatives using the kernel
    # unsqueeze to make it [N, 1]
    c = kernels._compute_derivatives(u.unsqueeze(1), props)

    # Extract components (New Kernel returns [N, 5, 1])
    # Indices: 0:dx, 1:dy, 2:dxx, 3:dxy, 4:dyy (Based on code reading)
    u_x_pred = c[:, 0, 0]
    u_y_pred = c[:, 1, 0]
    u_xx_pred = c[:, 2, 0]
    u_xy_pred = c[:, 3, 0]
    u_yy_pred = c[:, 4, 0]

    # Define exact solutions
    u_x_true = 4 * x - y
    u_y_true = 6 * y - x
    u_xx_true = torch.full_like(x, 4.0)
    u_yy_true = torch.full_like(x, 6.0)
    u_xy_true = torch.full_like(x, -1.0)

    # Filter for interior nodes only (Boundary nodes have high WLS error due to one-sided kernels)
    interior_mask = (x > 0.2) & (x < 3.8) & (y > -0.4) & (y < 0.4)

    # Check Errors
    mse_x = torch.nn.functional.mse_loss(u_x_pred[interior_mask], u_x_true[interior_mask])
    mse_xx = torch.nn.functional.mse_loss(u_xx_pred[interior_mask], u_xx_true[interior_mask])
    mse_xy = torch.nn.functional.mse_loss(u_xy_pred[interior_mask], u_xy_true[interior_mask])

    print(f"\nPolynomial Exactness Results:")
    print(f"  Gradient MSE: {mse_x.item():.6f}")
    print(f"  Laplacian (u_xx) MSE: {mse_xx.item():.6f}")
    print(f"  Mixed (u_xy) MSE: {mse_xy.item():.6f}")

    # Thresholds should be extremely low for structured grids
    assert mse_x < 1e-4, "Failed to reconstruct 1st derivative of polynomial"
    assert mse_xx < 1e-3, "Failed to reconstruct 2nd derivative of polynomial"
    assert mse_xy < 1e-3, "Failed to reconstruct mixed derivative"


def test_carreau_viscosity_logic(phys_cfg):
    """
    Tests that the Carreau-Yasuda logic correctly computes effective viscosity
    based on strain rate.
    """
    kernels = PhysicsKernels(phys_cfg)

    # Mock strain rates
    # Case 1: Zero shear -> Should be mu_0
    # We construct inputs [u_x, u_y, v_x, v_y]
    zero_shear = torch.zeros((1, 4))

    # Case 2: High shear -> Should approach mu_inf
    high_shear = torch.tensor([[1000.0, 0.0, 0.0, 1000.0]])

    du_ij = torch.cat([zero_shear, high_shear], dim=0)

    # Compute
    mu_eff = kernels._compute_carreau_viscosity(du_ij, u_ref=1.0, d_bar=1.0)

    mu_0_nd = phys_cfg.mu_0 / phys_cfg.mu_inf
    mu_inf_nd = phys_cfg.mu_inf / phys_cfg.mu_inf  # = 1.0

    print(f"\nViscosity Check:")
    print(f"  Zero Shear Val: {mu_eff[0].item():.4f} (Expected {mu_0_nd:.4f})")
    print(f"  High Shear Val: {mu_eff[1].item():.4f} (Expected approx {mu_inf_nd:.4f})")

    assert torch.isclose(mu_eff[0], torch.tensor(mu_0_nd), atol=1e-5), "Zero shear viscosity incorrect"
    assert mu_eff[1] < mu_eff[0], "Viscosity did not shear-thin"


def test_navier_stokes_residual_format(shared_test_graph, phys_cfg):
    """
    Ensures the residual function runs and returns a valid scalar
    without shape mismatches.
    """
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)

    # Create random prediction field [u, v, p]
    pred = torch.randn((data.num_nodes, 3))

    # Ensure props are generated
    props = kernels._get_geometric_props(data)

    # Compute Residual
    loss = kernels.navier_stokes_residual(pred, data, props)

    assert loss.dim() == 0, "Loss must be a scalar"
    assert not torch.isnan(loss), "Loss became NaN"
    assert loss > 0, "Residual of random noise should be positive"


# ==========================================
# INTEGRATION TESTS (REAL DATA)
# ==========================================

def test_laplacian_on_file(phys_cfg):
    """
    Loads a real processed .pt file (if available) and tests
    direct Laplacian extraction on a parabolic field.
    """
    cfg = VesselConfig()
    graph_dir = cfg.graph_output_dir

    if not os.path.exists(graph_dir):
        pytest.skip("Graph directory missing.")

    existing_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    if not existing_files:
        pytest.skip("No .pt files found.")

    data_path = os.path.join(graph_dir, existing_files[0])
    data = torch.load(data_path, weights_only=False)

    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    # Test Field: u = y^2
    # Laplacian should be 2.0 everywhere
    y = data.x[:, 1]
    u = y ** 2

    c = kernels._compute_derivatives(u.unsqueeze(1), props)

    # u_yy is index 4
    u_yy = c[:, 4, 0]

    # Filter interior (using Signed Distance Function if available, else heuristic)
    if data.x.shape[1] > 2:
        sdf = data.x[:, 2]
        interior_mask = sdf > 0.08
    else:
        # Fallback for simple grids
        interior_mask = (data.x[:, 0] > data.x[:, 0].min() + 0.1)

    if interior_mask.sum() == 0:
        pytest.skip("No interior nodes found for Laplacian test")

    mse_lap = torch.nn.functional.mse_loss(
        u_yy[interior_mask],
        torch.full_like(u_yy[interior_mask], 2.0)
    )

    print(f"\nReal Mesh Laplacian MSE: {mse_lap.item():.4f}")

    # Unstructured grids are noisier, so tolerance is higher (0.2 - 0.5 is typical for coarse graphs)
    assert mse_lap < 0.5, f"Laplacian error too high on real mesh: {mse_lap.item()}"


def test_full_physics_pipeline_validity(phys_cfg):
    """
    Validates that the Navier-Stokes residual on 'Ground Truth' (COMSOL) data
    is significantly lower than the residual on Random Noise.
    """
    v_cfg = VesselConfig()
    graph_dir = v_cfg.graph_output_dir

    if not os.path.exists(graph_dir):
        pytest.skip("Graph directory missing.")
    existing_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    if not existing_files:
        pytest.skip("No .pt files found.")

    data = torch.load(os.path.join(graph_dir, existing_files[0]), weights_only=False)
    kernels = PhysicsKernels(phys_cfg)

    # 1. Calculate Residual on COMSOL Data (Ground Truth)
    # data.y contains [u, v, p] from COMSOL
    if not hasattr(data, 'y') or data.y is None:
        pytest.skip("File does not contain ground truth 'y'.")

    comsol_pred = torch.stack([data.y[:, 0], data.y[:, 1], data.y[:, 2]], dim=1)

    # Ensure props are precomputed to save time
    props = kernels._get_geometric_props(data)

    loss_comsol = kernels.navier_stokes_residual(comsol_pred, data, props).item()

    # 2. Calculate Residual on Random Noise
    # This proves the loss function actually penalizes bad physics
    noise_pred = torch.randn_like(comsol_pred)
    loss_noise = kernels.navier_stokes_residual(noise_pred, data, props).item()

    print(f"\nPhysics Residual Check:")
    print(f"  COMSOL Data Loss: {loss_comsol:.4f}")
    print(f"  Random Noise Loss: {loss_noise:.4f}")

    # The COMSOL loss won't be 0 due to discretization error,
    # but it should be orders of magnitude smaller than noise.
    assert loss_comsol < loss_noise * 0.1, "Physics loss cannot distinguish Ground Truth from Noise!"

    # Optional: Hard threshold check (adjust based on normalization)
    # assert loss_comsol < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])