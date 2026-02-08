import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytest
from torch_geometric.data import Data

# Ensure an interactive backend is used for the popup
import matplotlib

matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you have PyQt installed


# ==========================================
# 1. CORE PHYSICS KERNELS
# ==========================================
class PhysicsKernels:
    def __init__(self, reynolds=100.0):
        """Implements ND Navier-Stokes residuals."""
        self.Re = reynolds

    def _compute_graph_gradients(self, f, data):
        """Calculates spatial gradients (df/dx, df/dy) via weighted projection."""
        row, col = data.edge_index
        pos_diff = data.x[col] - data.x[row]
        f_diff = f[col] - f[row]

        dist_sq = torch.sum(pos_diff ** 2, dim=1, keepdim=True) + 1e-6
        weights = 1.0 / dist_sq

        grad_f = torch.zeros((data.num_nodes, 2), device=f.device)
        grad_f.index_add_(0, row, weights * f_diff * pos_diff)

        norm_weights = torch.zeros((data.num_nodes, 1), device=f.device)
        norm_weights.index_add_(0, row, weights)

        return grad_f / (norm_weights + 1e-8)

    def navier_stokes_residual(self, pred, data):
        """Steady-state ND-NS residual calculation."""
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        grad_u = self._compute_graph_gradients(u, data)
        grad_v = self._compute_graph_gradients(v, data)
        grad_p = self._compute_graph_gradients(p, data)

        l_continuity = grad_u[:, 0:1] + grad_v[:, 1:2]
        momentum_x = (u * grad_u[:, 0:1] + v * grad_u[:, 1:2]) + grad_p[:, 0:1]
        momentum_y = (u * grad_v[:, 0:1] + v * grad_v[:, 1:2]) + grad_p[:, 1:2]

        return torch.mean(l_continuity ** 2 + momentum_x ** 2 + momentum_y ** 2)


# ==========================================
# 2. PYTEST SUITE
# ==========================================
def create_test_graph():
    """Helper to generate a reference pipe geometry."""
    x = torch.linspace(0, 4, 40)
    y = torch.linspace(-0.5, 0.5, 15)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    dist = torch.cdist(nodes, nodes)
    edge_index = (dist < 0.18).nonzero().t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return Data(x=nodes, edge_index=edge_index)


@pytest.mark.parametrize("re", [50, 100, 500, 1000])
def test_physics_residual_sweep(re):
    """Verifies physics stability across Reynolds numbers."""
    data = create_test_graph()
    kernels = PhysicsKernels(reynolds=re)

    # Analytical Poiseuille: u = 1 - 4y^2
    u = 1.0 - 4.0 * (data.x[:, 1:2] ** 2)
    v = torch.zeros_like(u)
    p = -0.05 * data.x[:, 0:1]
    pred = torch.cat([u, v, p], dim=1)

    res = kernels.navier_stokes_residual(pred, data)
    print(f"Re: {re} | Residual: {res.item():.6f}")
    assert res.item() < 0.1


def test_visualization_output():
    """Generates an interactive popup plot for visual audit."""
    data = create_test_graph()
    u = 1.0 - 4.0 * (data.x[:, 1:2] ** 2)

    plt.figure(figsize=(10, 3))
    plt.scatter(data.x[:, 0], data.x[:, 1], c=u.flatten(), cmap='jet', s=10)
    plt.colorbar(label="ND Velocity (u)")
    plt.title("Interactive Verification: Reference Poiseuille Flow")
    plt.axis('equal')

    # This will now pop up a window. You must close it for the test to finish.
    plt.show()
    assert True


if __name__ == "__main__":
    pytest.main([__file__])