import torch
import torch.nn.functional as F


def scatter_add(src, index, dim=0, dim_size=None):
    """
    A standalone replacement for torch_scatter.scatter_add.
    Avoids 'ModuleNotFoundError' on Windows/Python 3.13.
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    # 1. Create the output tensor of zeros
    out_size = list(src.size())
    out_size[dim] = dim_size
    out = torch.zeros(out_size, dtype=src.dtype, device=src.device)

    # 2. Handle broadcasting: ensure index matches src shape
    if index.dim() != src.dim():
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        index = index.view(view_shape).expand_as(src)

    # 3. Add values
    return out.scatter_add_(dim, index, src)


class PhysicsKernels:
    def __init__(self, reynolds=150.0):
        self.Re = reynolds

    def _compute_graph_gradients(self, f, data):
        """
        Calculates gradients (df/dx, df/dy) using Weighted Least-Squares.
        """
        row, col = data.edge_index
        num_nodes = data.num_nodes

        # 1. Edge Distances (dx, dy)
        pos_diff = data.x[col] - data.x[row]
        dx, dy = pos_diff[:, 0], pos_diff[:, 1]
        dist_sq = dx ** 2 + dy ** 2 + 1e-8

        # Weight function: Inverse distance weighting
        w = 1.0 / (dist_sq + 1e-8)

        # 2. Build Least-Squares Matrix Components
        # w, dx, dy are all 1D [E]. Multiplication works as expected.
        m_xx = scatter_add(w * dx * dx, row, dim=0, dim_size=num_nodes)
        m_xy = scatter_add(w * dx * dy, row, dim=0, dim_size=num_nodes)
        m_yy = scatter_add(w * dy * dy, row, dim=0, dim_size=num_nodes)

        det = m_xx * m_yy - m_xy ** 2 + 1e-5

        inv_xx = m_yy / det
        inv_xy = -m_xy / det
        inv_yy = m_xx / det

        # 3. Solve for Gradients
        f_diff = f[col] - f[row]  # Shape [E, 1]

        # --- FIX START ---
        # Explicitly reshape w to [E, 1] to prevent accidental broadcasting to [E, E]
        w_vec = w.unsqueeze(1)
        dx_vec = dx.unsqueeze(1)
        dy_vec = dy.unsqueeze(1)

        b_x = scatter_add(w_vec * f_diff * dx_vec, row, dim=0, dim_size=num_nodes)
        b_y = scatter_add(w_vec * f_diff * dy_vec, row, dim=0, dim_size=num_nodes)
        # --- FIX END ---

        grad_x = inv_xx.unsqueeze(1) * b_x + inv_xy.unsqueeze(1) * b_y
        grad_y = inv_xy.unsqueeze(1) * b_x + inv_yy.unsqueeze(1) * b_y

        return torch.cat([grad_x, grad_y], dim=1)

    def _compute_laplacian(self, f, data):
        """Approximates \Delta f ~ \sum w_ij (f_j - f_i)"""
        row, col = data.edge_index
        pos_diff = data.x[col] - data.x[row]
        dist_sq = pos_diff.pow(2).sum(dim=1, keepdim=True) + 1e-8
        w = 1.0 / dist_sq
        f_diff = f[col] - f[row]

        # w is [E, 1], f_diff is [E, 1]. Safe.
        lap = scatter_add(w * f_diff, row, dim=0, dim_size=data.num_nodes)
        return 4.0 * lap

    def navier_stokes_residual(self, pred, data):
        """Computes the residual of the steady-state Navier-Stokes equations."""
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        grad_u = self._compute_graph_gradients(u, data)
        grad_v = self._compute_graph_gradients(v, data)
        grad_p = self._compute_graph_gradients(p, data)

        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        grad_u_x = self._compute_graph_gradients(u_x, data)
        grad_u_y = self._compute_graph_gradients(u_y, data)
        grad_v_x = self._compute_graph_gradients(v_x, data)
        grad_v_y = self._compute_graph_gradients(v_y, data)

        lap_u = grad_u_x[:, 0:1] + grad_u_y[:, 1:2]
        lap_v = grad_v_x[:, 0:1] + grad_v_y[:, 1:2]

        l_cont = u_x + v_y
        mom_x = (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * lap_u
        mom_y = (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * lap_v

        return torch.mean(l_cont ** 2 + mom_x ** 2 + mom_y ** 2)

    def boundary_condition_loss(self, pred, data):
        """Penalizes velocity at walls (No-Slip)."""
        u, v = pred[:, 0:1], pred[:, 1:2]
        mask = torch.sigmoid(10.0 * (0.05 - data.sdf))
        return torch.mean(mask * (u ** 2 + v ** 2))

    def inlet_outlet_loss(self, pred, data):
        """Forces parabolic flow at Inlet and Zero Pressure at Outlet."""
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        x, y = data.x[:, 0], data.x[:, 1]

        inlet_mask = (x < 0.05)
        if inlet_mask.sum() > 0:
            u_target = 1.0 - 4.0 * (y[inlet_mask] ** 2)
            u_target = torch.clamp(u_target, min=0.0)
            l_inlet_u = F.mse_loss(u[inlet_mask].squeeze(), u_target)
            l_inlet_v = F.mse_loss(v[inlet_mask].squeeze(), torch.zeros_like(u_target))
        else:
            l_inlet_u, l_inlet_v = 0.0, 0.0

        max_x = x.max()
        outlet_mask = (x > (max_x - 0.05))
        if outlet_mask.sum() > 0:
            l_outlet_p = torch.mean(p[outlet_mask] ** 2)
        else:
            l_outlet_p = 0.0

        return l_inlet_u + l_inlet_v + l_outlet_p