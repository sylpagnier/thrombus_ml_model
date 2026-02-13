import pytest
import torch
from torch_geometric.data import Data
from pathlib import Path
import shutil

from src.phase1.utils.physics_kernels import PhysicsKernels
from src.phase1.data_gen.vessel_generator import VesselGenerator
from src.phase1.data_gen.mesh_to_graph import MeshToGraphComplete


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
    # Remove self-loops
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    return Data(x=nodes, edge_index=edge_index, num_nodes=nodes.shape[0])


# ==========================================
# UNIT TESTS (STRUCTURED)
# ==========================================
def test_first_order_gradient(shared_test_graph):
    """
    Verifies that the first derivative is computed correctly.
    Input: u = y^2  =>  du/dy = 2y, du/dx = 0
    """
    data = shared_test_graph
    kernels = PhysicsKernels(reynolds=1.0)
    props = kernels._get_geometric_props(data)

    y = data.x[:, 1:2]
    u = y ** 2

    grad_u = kernels._compute_gradients(u, props)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]

    # dx should be 0, dy should be 2y
    assert torch.allclose(du_dx, torch.zeros_like(du_dx), atol=1e-3), "du/dx is non-zero!"

    mse_dy = torch.nn.functional.mse_loss(du_dy, 2.0 * y)
    assert mse_dy < 0.01, f"First derivative du/dy failed! MSE: {mse_dy.item():.4f}"


def test_second_order_laplacian_structured(shared_test_graph):
    """
    Tests Div(Grad(u)) by isolating interior nodes from boundary nodes.
    Input: u = y^2 => d^2u/dy^2 = 2.0, d^2u/dx^2 = 0.0
    """
    data = shared_test_graph
    kernels = PhysicsKernels(reynolds=1.0)
    props = kernels._get_geometric_props(data)

    u = data.x[:, 1:2] ** 2

    # Compute 1st then 2nd derivative.
    # Note: Even though we have a direct 2nd-order solver in PhysicsKernels now,
    # this test specifically checks the legacy chained behavior. Because the direct
    # solver is strictly 2nd-order, applying it twice on a quadratic works flawlessly!
    grad_u = kernels._compute_gradients(u, props)
    grad_du_dy = kernels._compute_gradients(grad_u[:, 1:2], props)
    d2u_dy2 = grad_du_dy[:, 1:2]

    # Isolate interior vs boundary nodes
    x, y = data.x[:, 0], data.x[:, 1]
    interior_mask = (x > 0.4) & (x < 3.6) & (y > -0.3) & (y < 0.3)
    boundary_mask = ~interior_mask

    lap_interior = d2u_dy2[interior_mask]
    lap_boundary = d2u_dy2[boundary_mask]

    mse_interior = torch.nn.functional.mse_loss(lap_interior, torch.full_like(lap_interior, 2.0))
    mse_boundary = torch.nn.functional.mse_loss(lap_boundary, torch.full_like(lap_boundary, 2.0))

    print(f"\nStructured Interior d^2u/dy^2 MSE: {mse_interior.item():.4f}")
    print(f"Structured Boundary d^2u/dy^2 MSE: {mse_boundary.item():.4f}")

    # The interior should be extremely accurate
    assert mse_interior < 0.05, f"Core WLS math is failing on interior nodes: MSE {mse_interior.item()}"

    # WE FIXED THE KERNEL! The boundary error should now be virtually zero as well.
    assert mse_boundary < 0.05, f"Boundary error is still too high! MSE: {mse_boundary.item()}"


# ==========================================
# INTEGRATION TEST (UNSTRUCTURED)
# ==========================================
def test_kernel_on_real_gmsh_topology(tmp_path):
    """
    Tests the Laplacian on a real unstructured mesh, utilizing the SDF to mask out the boundary.
    """
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()

    gen = VesselGenerator(output_dir=raw_dir)
    gen.generate(idx=0, level=1, show_viz=False, ax=None)

    converter = MeshToGraphComplete(raw_dir=raw_dir, label_dir=raw_dir, proc_dir=proc_dir)
    converter.process_file("vessel_0.msh")

    data = torch.load(proc_dir / "vessel_0.pt", weights_only=False)
    kernels = PhysicsKernels(reynolds=1.0)
    props = kernels._get_geometric_props(data)

    # Input: u = y^2
    y_coord = data.x[:, 1:2]
    u = y_coord ** 2

    # Compute 2nd derivative (d^2u/dy^2)
    grad_u = kernels._compute_gradients(u, props)
    grad_du_dy = kernels._compute_gradients(grad_u[:, 1:2], props)
    d2u_dy2 = grad_du_dy[:, 1:2]

    # Use the Signed Distance Field (Index 2) to dynamically find interior nodes
    sdf = data.x[:, 2]
    interior_mask = sdf > 0.08

    interior_lap = d2u_dy2[interior_mask]
    mse_interior = torch.nn.functional.mse_loss(interior_lap, torch.full_like(interior_lap, 2.0))

    print(f"\nUnstructured Interior d^2u/dy^2 MSE: {mse_interior.item():.4f}")

    # Unstructured grids will have slightly higher natural error than structured grids,
    # but with a 2nd-order WLS it should comfortably beat 0.20
    assert mse_interior < 0.20, f"Unstructured interior gradients failing: {mse_interior.item()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])