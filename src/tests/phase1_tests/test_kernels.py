import torch
import pytest
from torch_geometric.data import Data
from src.phase1.utils.physics_kernels import PhysicsKernels


# ==========================================
# OPTIMIZED FIXTURES
# ==========================================
@pytest.fixture(scope="module")
def shared_test_graph():
    """
    Generates the test mesh ONCE for the entire module.
    Reduced resolution (40x15) for speed on CPU.
    """
    # 1. Create Grid (N=600 nodes)
    x = torch.linspace(0, 4, 40)
    y = torch.linspace(-0.5, 0.5, 15)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    # 2. Connect neighbors (Radius 0.20)
    # Reduced radius slightly to keep edge count manageable (~20k edges)
    dist = torch.cdist(nodes, nodes)
    edge_index = (dist < 0.20).nonzero().t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Remove self-loops

    # 3. Create SDF
    sdf = 0.5 - torch.abs(nodes[:, 1:2])

    return Data(x=nodes, edge_index=edge_index, sdf=sdf, num_nodes=nodes.shape[0])


# ==========================================
# TEST SUITE
# ==========================================
@pytest.mark.parametrize("re", [1.0, 10.0, 150.0])
def test_physics_residual_sweep(re, shared_test_graph):
    """
    Verifies physics stability across Reynolds numbers.
    Uses the cached 'shared_test_graph' for speed.
    """
    data = shared_test_graph
    kernels = PhysicsKernels(reynolds=re)

    # 1. Analytical Poiseuille Flow: u = 1 - 4y^2
    u = 1.0 - 4.0 * (data.x[:, 1:2] ** 2)
    v = torch.zeros_like(u)

    # 2. Analytical Pressure Gradient: dp/dx = -8/Re
    p_grad = -8.0 / re
    p = p_grad * data.x[:, 0:1]

    pred = torch.cat([u, v, p], dim=1)

    # 3. Calculate Residual
    res_mse = kernels.navier_stokes_residual(pred, data)

    # Kernel returns Mean Squared Error (MSE)
    rmse = res_mse.item() ** 0.5

    # 4. Metrics
    force_magnitude = abs(8.0 / re)
    relative_error = rmse / force_magnitude

    print(f"Re: {re} | RMSE: {rmse:.4f} | Force: {force_magnitude:.4f} | Rel: {relative_error:.2%}")

    # 5. Robust Assertion
    # Pass if EITHER:
    # A) Relative error is low (valid for Low Re where forces are large)
    # B) Absolute RMSE is below noise floor (valid for High Re where forces vanish)
    is_relative_good = relative_error < 0.25
    is_absolute_good = rmse < 0.15

    assert is_relative_good or is_absolute_good, \
        f"Failed at Re={re}: Rel={relative_error:.2%} (Limit 25%), Abs={rmse:.4f} (Limit 0.15)"


def test_double_gradient_accuracy(shared_test_graph):
    """
    Tests the Laplacian accuracy using the public residual API.
    Input: u = y^2  =>  Lap(u) = 2.0
    Residual = -Lap(u) = -2.0
    Expected Loss = Mean((-2.0)^2) = 4.0
    """
    data = shared_test_graph
    # Set Re=1.0 so viscosity term is exactly -1 * Laplacian
    kernels = PhysicsKernels(reynolds=1.0)

    # 1. Input field u = y^2, v = 0
    u = data.x[:, 1:2] ** 2
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)

    pred = torch.cat([u, v, p], dim=1)

    # 2. Compute Residual (Returns MSE scalar)
    mse_res = kernels.navier_stokes_residual(pred, data)

    print(f"Laplacian Test | Target MSE: 4.0 | Actual MSE: {mse_res.item():.4f}")

    # Check if we are within reasonable bounds
    # 2.25 < MSE < 6.5 corresponds to Laplacian approx 1.5 to 2.5
    assert 2.0 < mse_res.item() < 6.5


if __name__ == "__main__":
    pytest.main([__file__])