import torch
import numpy as np
from torch_geometric.data import Data
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import PhysicsConfig, VesselConfig


def create_physical_test_graph():
    """
    Constructs a synthetic 2D grid representing a 1cm x 2mm vessel segment.
    Uses realistic spatial scaling (meters) to match PhysicsConfig.
    """
    # 1. Geometry Setup (SI Units)
    x = torch.linspace(0, 0.01, 20)  # 1cm length
    y = torch.linspace(-0.001, 0.001, 10)  # 2mm width
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    num_nodes = nodes.size(0)

    # 2. Connectivity (Radius-based for WLS stability)
    dist = torch.cdist(nodes, nodes)
    edge_index = (dist < 0.0012).nonzero(as_tuple=False).t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    # 3. Create Data Object
    data = Data(x=torch.zeros(num_nodes, 11), edge_index=edge_index)
    data.num_nodes = num_nodes

    # 4. Physical Boundary Masks
    data.mask_wall = (nodes[:, 1].abs() > 0.00095).float()
    data.mask_inlet = (nodes[:, 0] < 0.0001).float()
    data.mask_outlet = (nodes[:, 0] > 0.0099).float()

    # 5. Features: Wall Normals (nx, ny at indices 4, 5)
    data.x = torch.zeros(num_nodes, 11)
    data.x[:, 4] = 0.0
    data.x[:, 5] = torch.where(nodes[:, 1] > 0, -1.0, 1.0)  # Normals pointing inward

    # 6. Simulation Parameters (Real Config-driven)
    phys_cfg = PhysicsConfig(tier="tier2")
    data.u_ref = torch.tensor([phys_cfg.get_u_ref(0.0015)])  # Based on 1.5mm D
    data.d_bar = torch.tensor([0.0015])

    # 7. WLS Operators (WLS Basis: [dx, dy, 0.5*dx^2, dx*dy, 0.5*dy^2])
    row, col = edge_index
    dr = nodes[col] - nodes[row]
    dx, dy = dr[:, 0], dr[:, 1]
    data.V = torch.stack([dx, dy, 0.5 * dx ** 2, dx * dy, 0.5 * dy ** 2], dim=1)
    data.W = torch.ones(edge_index.size(1))

    # Mock M_inv as Identity for testing derivative logic flow
    # In production, this must be (V^T W V)^-1
    data.M_inv = torch.eye(5).unsqueeze(0).repeat(num_nodes, 1, 1)

    return data, nodes, phys_cfg


def test_wls_derivative_dimensions():
    """Verifies WLS derivatives handle batch/node dimensions correctly."""
    data, nodes, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    # Linear field test: u = x + 2y
    u = (nodes[:, 0] + 2 * nodes[:, 1]).unsqueeze(1)
    props = kernels._get_geometric_props(data)
    derivs = kernels._compute_derivatives(u, props)

    assert derivs.shape == (nodes.size(0), 5, 1)  # [N, 5 basis coefficients, 1 channel]
    assert not torch.isnan(derivs).any()


def test_carreau_rheology_real_bounds():
    """Tests if the non-Newtonian viscosity stays within Carreau-Yasuda limits."""
    data, _, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    # Simulate Gradient Scenarios: [du_dx, du_dy, dv_dx, dv_dy]
    # 1. Zero shear (should be near mu_0_nd)
    du_zero = torch.zeros((data.num_nodes, 4))
    mu_zero = kernels._compute_carreau_viscosity(du_zero, data)

    # 2. Extreme shear (should approach mu_inf_nd)
    du_high = torch.tensor([[0.0, 1000.0, 0.0, 0.0]]).repeat(data.num_nodes, 1)
    mu_high = kernels._compute_carreau_viscosity(du_high, data)

    print(f"\nViscosity Bounds ND: [{kernels.mu_inf_nd:.4f}, {kernels.mu_0_nd:.4f}]")
    assert torch.all(mu_zero <= kernels.mu_0_nd + 1e-5)
    assert torch.all(mu_high >= kernels.mu_inf_nd - 1e-5)
    assert torch.all(mu_high < mu_zero)  # Confirm shear-thinning behavior


def test_mass_conservation_logic():
    """Validates the continuity loss accurately detects divergence."""
    _, _, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    # Solenoidal (incompressible) field: du/dx + dv/dy = 0
    du_ij_ok = torch.tensor([[0.5, 0.0, 0.0, -0.5]])
    loss_ok = kernels.continuity_loss(du_ij_ok)

    # Source/Sink field: du/dx + dv/dy = 2.0
    du_ij_bad = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    loss_bad = kernels.continuity_loss(du_ij_bad)

    assert loss_ok < 1e-7
    assert loss_bad > 1.0


def test_momentum_residual_execution():
    """Ensures Navier-Stokes residual computes without crashing under real config."""
    data, nodes, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    # Mock Poiseuille-like prediction: [u, v, p, mu_eff]
    y_norm = nodes[:, 1] / 0.001
    u = (1.0 - y_norm ** 2).unsqueeze(1)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    mu = torch.ones_like(u) * kernels.mu_inf_nd

    pred = torch.cat([u, v, p, mu], dim=1)

    # Trigger NS calculation
    loss_mom = kernels.navier_stokes_residual(pred, data)

    assert not torch.isnan(loss_mom)
    assert loss_mom >= 0


def test_wall_shear_stress_consistency():
    """Checks that WSS calculation produces positive magnitudes at the walls."""
    data, nodes, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    # Velocity u = y (Linear shear), v = 0, p = 0, mu = 1, WSS_pred = 1
    u = nodes[:, 1].unsqueeze(1)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    mu = torch.ones_like(u)
    wss_pred = torch.ones_like(nodes[:, 0])

    pred = torch.cat([u, v, p, mu, wss_pred.unsqueeze(1)], dim=1)

    loss_wss = kernels.wall_shear_stress_loss(pred, data)

    assert not torch.isnan(loss_wss)
    print(f"WSS Loss: {loss_wss.item():.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])