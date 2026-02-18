import pytest
import torch
import os
from torch_geometric.data import Data
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig


# ==========================================
# FIXTURES
# ==========================================
@pytest.fixture(scope="module")
def shared_test_graph():
    """Generates a structured grid for baseline math checks."""
    x = torch.linspace(0, 4, 40)
    y = torch.linspace(-0.5, 0.5, 15)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    dist = torch.cdist(nodes, nodes)
    edge_index = (dist < 0.20).nonzero().t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    return Data(x=nodes, edge_index=edge_index, num_nodes=nodes.shape[0])


# ==========================================
# UNIT TESTS (STRUCTURED)
# ==========================================
def test_first_order_gradient(shared_test_graph):
    """Verifies that the first derivative is computed correctly."""
    data = shared_test_graph
    kernels = PhysicsKernels(reynolds=1.0)
    props = kernels._get_geometric_props(data)

    y = data.x[:, 1:2]
    u = y ** 2

    grad_u = kernels._compute_gradients(u, props)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]

    assert torch.allclose(du_dx, torch.zeros_like(du_dx), atol=1e-3), "du/dx is non-zero!"
    mse_dy = torch.nn.functional.mse_loss(du_dy, 2.0 * y)
    assert mse_dy < 0.01, f"First derivative du/dy failed! MSE: {mse_dy.item():.4f}"


def test_second_order_laplacian_structured(shared_test_graph):
    """Tests Div(Grad(u)) by isolating interior nodes from boundary nodes."""
    data = shared_test_graph
    kernels = PhysicsKernels(reynolds=1.0)
    props = kernels._get_geometric_props(data)

    u = data.x[:, 1:2] ** 2

    grad_u = kernels._compute_gradients(u, props)
    grad_du_dy = kernels._compute_gradients(grad_u[:, 1:2], props)
    d2u_dy2 = grad_du_dy[:, 1:2]

    x, y = data.x[:, 0], data.x[:, 1]
    interior_mask = (x > 0.4) & (x < 3.6) & (y > -0.3) & (y < 0.3)
    boundary_mask = ~interior_mask

    lap_interior = d2u_dy2[interior_mask]
    lap_boundary = d2u_dy2[boundary_mask]

    mse_interior = torch.nn.functional.mse_loss(lap_interior, torch.full_like(lap_interior, 2.0))
    mse_boundary = torch.nn.functional.mse_loss(lap_boundary, torch.full_like(lap_boundary, 2.0))

    print(f"\nStructured Interior d^2u/dy^2 MSE: {mse_interior.item():.4f}")
    print(f"Structured Boundary d^2u/dy^2 MSE: {mse_boundary.item():.4f}")

    assert mse_interior < 0.05, f"Interior MSE too high: {mse_interior.item()}"
    assert mse_boundary < 0.05, f"Boundary MSE too high: {mse_boundary.item()}"


# ==========================================
# INTEGRATION TESTS (REAL DATA)
# ==========================================
def test_kernel_on_existing_data():
    """Tests Laplacian accuracy on a real vessel mesh."""
    cfg = VesselConfig()
    graph_dir = cfg.graph_output_dir

    if not os.path.exists(graph_dir):
        pytest.skip("Graph directory missing.")

    existing_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    if not existing_files:
        pytest.skip("No .pt files found.")

    data_path = graph_dir / existing_files[0]
    data = torch.load(data_path, weights_only=False)

    kernels = PhysicsKernels(reynolds=1.0)
    props = kernels._get_geometric_props(data)

    y_coord = data.x[:, 1:2]
    u = y_coord ** 2

    grad_u = kernels._compute_gradients(u, props)
    grad_du_dy = kernels._compute_gradients(grad_u[:, 1:2], props)
    d2u_dy2 = grad_du_dy[:, 1:2]

    sdf = data.x[:, 2]
    interior_mask = sdf > 0.08

    if interior_mask.sum() == 0:
        pytest.fail("No interior nodes found.")

    mse_interior = torch.nn.functional.mse_loss(
        d2u_dy2[interior_mask],
        torch.full_like(d2u_dy2[interior_mask], 2.0)
    )

    print(f"\nTested Laplacian on file: {existing_files[0]}")
    print(f"Unstructured Interior d^2u/dy^2 MSE: {mse_interior.item():.4f}")

    assert mse_interior < 0.20, f"Laplacian accuracy failed: {mse_interior.item()}"


def test_navier_stokes_integration_real_data():
    """
    Validates the full Navier-Stokes Residual using PRE-GENERATED COMSOL DATA.
    Instead of synthetic parabolas, we check if the Kernel agrees with COMSOL.
    """
    # 1. Load Configs
    v_cfg = VesselConfig()
    p_cfg = PhysicsConfig()

    graph_dir = v_cfg.graph_output_dir
    if not os.path.exists(graph_dir):
        pytest.skip("Graph directory missing.")

    existing_files = [f for f in os.listdir(graph_dir) if f.endswith('.pt')]
    if not existing_files:
        pytest.skip("No .pt files found. Run data pipeline first.")

    # 2. Load a real graph (e.g., vessel_0.pt)
    # We try to find a 'straight' vessel if possible as they are numerically cleaner
    # but any file will work.
    filename = existing_files[0]
    data_path = graph_dir / filename
    data = torch.load(data_path, weights_only=False)

    # 3. Setup Physics Kernel with CORRECT Reynolds Number
    # The data in .pt is normalized. We must match the Re used in generation.
    re_target = p_cfg.re_target  # e.g. 150.0
    kernels = PhysicsKernels(reynolds=re_target)

    # 4. Extract COMSOL Ground Truth
    # data.y is [u, v, p] (normalized)
    # We treat this as our "prediction" to see if it satisfies the equations
    u_comsol = data.y[:, 0]
    v_comsol = data.y[:, 1]
    p_comsol = data.y[:, 2]

    pred = torch.stack([u_comsol, v_comsol, p_comsol], dim=1)

    # 5. Compute Residual
    # Note: data.mask_wall or similar might need to be set if the kernel uses it strict
    # But usually residuals are just computed over all nodes.
    try:
        loss = kernels.navier_stokes_residual(pred, data)
        print(f"\n[{filename}] Navier-Stokes Residual (Re={re_target}): {loss.item():.4f}")
    except RuntimeError as e:
        pytest.fail(f"Navier-Stokes assembly crashed: {e}")

    # 6. Assertion
    # The residual won't be 0.0 because:
    #   a) COMSOL uses FEM, we use Finite Difference (discretization error)
    #   b) Interpolation noise from mesh_to_graph
    # However, it should be significantly lower than random noise (which was ~140,000)

    # We normalize by number of nodes to get a sense of "per-node error"
    mse_loss = loss.item() / data.num_nodes
    print(f"Mean Residual per Node: {mse_loss:.6f}")

    assert not torch.isnan(loss), "Loss returned NaN"

    # Threshold: Random noise is ~200.0 per node. Physics should be < 5.0 per node.
    # (Adjust this threshold based on what you see in the first run)
    assert mse_loss < 5.0, f"Residual is too high ({mse_loss:.4f}). Physics might be mismatched."


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])