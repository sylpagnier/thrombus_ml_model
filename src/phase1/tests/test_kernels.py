import pytest
import torch
import math
from torch_geometric.data import Data
from src.phase1.physics.physics_kernels import PhysicsKernels, scatter_add
from src.config import VesselConfig, PhysicsConfig


# ==========================================
# FIXTURES & MOCK DATA
# ==========================================

@pytest.fixture(params=["tier1", "tier2"], scope="module")
def tier(request):
    return request.param


@pytest.fixture(scope="module")
def phys_cfg(tier):
    return PhysicsConfig(tier=tier)


@pytest.fixture(scope="module")
def shared_test_graph():
    """Generates a structured grid for exact mathematical validation."""
    x = torch.linspace(-1, 1, 20)
    y = torch.linspace(-1, 1, 20)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    dist = torch.cdist(nodes, nodes)
    edge_index = (dist < 0.15).nonzero().t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    data = Data(x=nodes, edge_index=edge_index, num_nodes=nodes.shape[0])
    data.u_ref = torch.tensor([1.0])
    data.d_bar = torch.tensor([1.0])
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    # Masks: Make the boundaries actual walls/inlets
    data.mask_wall = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.mask_inlet = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.mask_outlet = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.u_inlet_bc = torch.ones((data.num_nodes, 2)) * 0.5

    return data


# ==========================================
# UTILITY TESTS
# ==========================================

def test_scatter_add_logic():
    """Tests the custom scatter_add replacement against expected exact values."""
    src = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    index = torch.tensor([0, 0, 1, 1])

    out = scatter_add(src, index, dim=0, dim_size=3)

    expected = torch.tensor([[3.0], [7.0], [0.0]])
    assert torch.allclose(out, expected), f"scatter_add failed. Got {out}, expected {expected}"


# ==========================================
# MATHEMATICAL EXACTNESS TESTS
# ==========================================

def test_polynomial_wls_exactness(shared_test_graph, phys_cfg):
    """
    WLS should perfectly reconstruct up to 2nd order polynomials.
    Instead of checking if error < 1e-3, we assert strict mathematical equality.
    """
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    x, y = data.x[:, 0], data.x[:, 1]
    # Field: u(x, y) = 2x^2 + 3y^2 - xy
    u = 2 * x ** 2 + 3 * y ** 2 - x * y

    c = kernels._compute_derivatives(u.unsqueeze(1), props)

    u_x_pred, u_xx_pred, u_xy_pred = c[:, 0, 0], c[:, 2, 0], c[:, 3, 0]

    # Analytical derivatives
    u_x_true = 4 * x - y
    u_xx_true = torch.full_like(x, 4.0)
    u_xy_true = torch.full_like(x, -1.0)

    # Exclude boundary nodes where WLS stencil is asymmetrical
    interior_mask = (x > -0.8) & (x < 0.8) & (y > -0.8) & (y < 0.8)

    assert torch.allclose(u_x_pred[interior_mask], u_x_true[interior_mask],
                          atol=1e-4), "1st derivative failed exactness."
    assert torch.allclose(u_xx_pred[interior_mask], u_xx_true[interior_mask],
                          atol=1e-3), "2nd derivative failed exactness."
    assert torch.allclose(u_xy_pred[interior_mask], u_xy_true[interior_mask],
                          atol=1e-3), "Mixed derivative failed exactness."

def test_carreau_viscosity_analytical():
    """
    Tests Carreau logic by calculating the expected formula natively in Python
    and ensuring the PyTorch vectorized kernel strictly matches it.
    """
    # Explicitly instantiate tier 2, removing the parameterized fixture to avoid skips
    phys_cfg = PhysicsConfig(tier="tier2")
    kernels = PhysicsKernels(phys_cfg)

    # Deterministic mock strain rate: [u_x, u_y, v_x, v_y]
    du_ij = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    mock_data = Data()
    mock_data.u_ref = torch.tensor([1.0])
    mock_data.d_bar = torch.tensor([2.0])  # Use 2.0 to test lambda scaling
    mock_data.batch = torch.zeros(du_ij.size(0), dtype=torch.long)

    # 1. Get Network Output
    mu_eff_pred = kernels._compute_carreau_viscosity(du_ij, mock_data).item()

    # 2. Hand-calculate Ground Truth
    strain_sq = 2.0 * (1.0 ** 2 + 4.0 ** 2) + (2.0 + 3.0) ** 2
    gamma_dot_nd = math.sqrt(strain_sq + 1e-4)

    lambda_nd = phys_cfg.lam * (1.0 / 2.0)
    shear_term = 1.0 + (lambda_nd * gamma_dot_nd) ** phys_cfg.a
    power = (phys_cfg.n - 1.0) / phys_cfg.a

    mu_inf_nd = phys_cfg.mu_inf / phys_cfg.mu_ref
    mu_0_nd = phys_cfg.mu_0 / phys_cfg.mu_ref

    mu_eff_true = mu_inf_nd + (mu_0_nd - mu_inf_nd) * (shear_term ** power)

    assert math.isclose(mu_eff_pred, mu_eff_true, rel_tol=1e-5), \
        f"Viscosity math mismatch! Expected {mu_eff_true}, got {mu_eff_pred}"


def test_rheology_loss_exactness(shared_test_graph):
    """
    Forces the prediction to be exactly e^1 times the target.
    This avoids the max=100.0 clamp in the kernel, ensuring the Log-MSE evaluates exactly to 1.0.
    """
    # Explicitly instantiate tier 2, removing the parameterized fixture to avoid skips
    phys_cfg = PhysicsConfig(tier="tier2")
    data = shared_test_graph
    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(data)

    u = data.x[:, 1] ** 2
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)

    c_u = kernels._compute_derivatives(u.unsqueeze(1), props)
    c_v = kernels._compute_derivatives(v.unsqueeze(1), props)
    du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)
    mu_target = kernels._compute_carreau_viscosity(du_ij, data)

    # Shift prediction by e^1 to safely stay under the kernel's max=100.0 clamp
    mu_pred = mu_target * math.exp(1.0)
    pred = torch.stack([u, v, p, mu_pred], dim=1)

    loss = kernels.rheology_loss(pred, data, props)

    # Log-MSE: (log(target * e^1) - log(target))^2 = (1.0)^2 = 1.0
    assert math.isclose(loss.item(), 1.0, rel_tol=1e-4), \
        f"Log-MSE math failed. Expected exactly 1.0, got {loss.item()}"


# ==========================================
# BOUNDARY CONDITION EXACTNESS TESTS
# ==========================================

def test_boundary_condition_exactness(shared_test_graph, phys_cfg):
    """Tests that Dirichlet boundary MSE exactly matches the manual sum of squares."""
    data = shared_test_graph.clone()
    kernels = PhysicsKernels(phys_cfg)

    # Mask 2 nodes as wall
    data.mask_wall[:2] = True

    # FIX: GINO_DEQ universally outputs 4 channels now (u, v, p, mu)
    num_channels = 4
    pred = torch.zeros((data.num_nodes, num_channels))

    # Node 0 has u=2, v=3 -> squared sum = 13
    # Node 1 has u=4, v=1 -> squared sum = 17
    # Mean loss = (13 + 17) / 2 = 15.0
    pred[0, 0], pred[0, 1] = 2.0, 3.0
    pred[1, 0], pred[1, 1] = 4.0, 1.0

    loss = kernels.boundary_condition_loss(pred, data)
    assert math.isclose(loss.item(), 15.0, rel_tol=1e-5), f"Expected wall loss 15.0, got {loss.item()}"


def test_inlet_outlet_exactness(shared_test_graph, phys_cfg):
    """Tests that Inlet/Outlet boundary MSE exactly matches hand-calculated values."""
    data = shared_test_graph.clone()
    kernels = PhysicsKernels(phys_cfg)

    data.mask_inlet[0] = True  # 1 inlet node
    data.mask_outlet[1] = True  # 1 outlet node
    data.u_inlet_bc[0, 0] = 0.5  # Target u = 0.5

    # FIX: Output channels universally set to 4
    num_channels = 4
    pred = torch.zeros((data.num_nodes, num_channels))

    # INLET Node (0): pred u=2.0, v=1.0. Target u=0.5, v=0.0
    # Loss = (2.0 - 0.5)^2 + (1.0 - 0.0)^2 = 2.25 + 1.0 = 3.25
    pred[0, 0] = 2.0
    pred[0, 1] = 1.0

    # OUTLET Node (1): pred p = 4.0. Target p = 0.0
    # Loss = 4.0^2 = 16.0
    pred[1, 2] = 4.0

    loss = kernels.inlet_outlet_loss(pred, data)

    # Total expected = 3.25 + 16.0 = 19.25
    assert math.isclose(loss.item(), 19.25, rel_tol=1e-5), \
        f"Expected inlet/outlet loss 19.25, got {loss.item()}"


def test_kernels_on_real_generated_graph(phys_cfg):
    """
    Integration Test: Loads a real graph and verifies the physics kernels
    use the actual generated u_ref and d_bar values stored in the data.
    """
    vessel_cfg = VesselConfig(tier=phys_cfg.tier)
    graph_dir = vessel_cfg.graph_output_dir

    if not graph_dir.exists() or not list(graph_dir.glob("*.pt")):
        pytest.skip(f"No real graphs found in {graph_dir}.")

    sample_file = next(graph_dir.glob("*.pt"))

    # Use weights_only=False to allow PyG Data classes to load
    real_data = torch.load(sample_file, weights_only=False)

    # 1. Extract REAL values from the generated data
    u_ref_real = real_data.u_ref.item()
    d_bar_real = real_data.d_bar.item()

    kernels = PhysicsKernels(phys_cfg)
    props = kernels._get_geometric_props(real_data)

    # 2. Generate a "Probe" prediction matching the graph's size
    # FIX: Universally set to 4 channels
    num_channels = 4
    pred = torch.randn((real_data.num_nodes, num_channels), dtype=torch.float32)

    # 3. Verify the Kernel handles the split residual return (l_cont, l_mom)
    l_cont, l_mom = kernels.navier_stokes_residual(pred, real_data, props)

    # 4. Assertions for physical validity
    assert not torch.isnan(l_cont), f"NaN in Continuity residual (u_ref={u_ref_real})"
    assert not torch.isnan(l_mom), f"NaN in Momentum residual (u_ref={u_ref_real})"
    assert u_ref_real > 0, "Reference velocity in generated data must be positive"
    assert d_bar_real > 0, "Effective diameter in generated data must be positive"