import torch
import torch.nn.functional as F


def scatter_add(src, index, dim=0, dim_size=None):
    """Standalone replacement for torch_scatter.scatter_add."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    out_size = list(src.size())
    out_size[dim] = dim_size
    out = torch.zeros(out_size, dtype=src.dtype, device=src.device)
    if index.dim() != src.dim():
        view_shape = [1] * src.dim()
        view_shape[dim] = -1
        index = index.view(view_shape).expand_as(src)
    return out.scatter_add_(dim, index, src)


class PhysicsKernels:
    def __init__(self, reynolds=150.0):
        self.Re = reynolds

    def _compute_graph_gradients(self, f, data):
        """Calculates gradients (df/dx, df/dy) using Weighted Least-Squares."""
        row, col = data.edge_index
        num_nodes = data.num_nodes

        # 1. Edge Distances
        pos_diff = data.x[col] - data.x[row]
        dx, dy = pos_diff[:, 0], pos_diff[:, 1]
        dist_sq = dx ** 2 + dy ** 2 + 1e-8
        w = 1.0 / (dist_sq + 1e-8)

        # 2. Least-Squares Matrix
        m_xx = scatter_add(w * dx * dx, row, dim=0, dim_size=num_nodes)
        m_xy = scatter_add(w * dx * dy, row, dim=0, dim_size=num_nodes)
        m_yy = scatter_add(w * dy * dy, row, dim=0, dim_size=num_nodes)
        det = m_xx * m_yy - m_xy ** 2 + 1e-5

        inv_xx = m_yy / det
        inv_xy = -m_xy / det
        inv_yy = m_xx / det

        # 3. Gradients
        f_diff = f[col] - f[row]
        w_vec = w.unsqueeze(1)
        dx_vec, dy_vec = dx.unsqueeze(1), dy.unsqueeze(1)

        b_x = scatter_add(w_vec * f_diff * dx_vec, row, dim=0, dim_size=num_nodes)
        b_y = scatter_add(w_vec * f_diff * dy_vec, row, dim=0, dim_size=num_nodes)

        grad_x = inv_xx.unsqueeze(1) * b_x + inv_xy.unsqueeze(1) * b_y
        grad_y = inv_xy.unsqueeze(1) * b_x + inv_yy.unsqueeze(1) * b_y

        return torch.cat([grad_x, grad_y], dim=1)

    def navier_stokes_residual(self, pred, data):
        """Residual of steady-state Navier-Stokes."""
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        grad_u = self._compute_graph_gradients(u, data)
        grad_v = self._compute_graph_gradients(v, data)
        grad_p = self._compute_graph_gradients(p, data)

        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        # Second derivatives (Laplacian approximation)
        grad_u_x = self._compute_graph_gradients(u_x, data)
        grad_u_y = self._compute_graph_gradients(u_y, data)
        grad_v_x = self._compute_graph_gradients(v_x, data)
        grad_v_y = self._compute_graph_gradients(v_y, data)

        lap_u = grad_u_x[:, 0:1] + grad_u_y[:, 1:2]
        lap_v = grad_v_x[:, 0:1] + grad_v_y[:, 1:2]

        # Equations
        l_cont = u_x + v_y
        mom_x = (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * lap_u
        mom_y = (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * lap_v

        return torch.mean(l_cont ** 2 + mom_x ** 2 + mom_y ** 2)

    def boundary_condition_loss(self, pred, data):
        """Penalizes velocity at walls using explicit Wall Masks."""
        u, v = pred[:, 0:1], pred[:, 1:2]
        # Use explicit mask_wall (converted to float for multiplication)
        mask = data.mask_wall.float().unsqueeze(1)
        return torch.mean(mask * (u ** 2 + v ** 2))

    def inlet_outlet_loss(self, pred, data):
        """Forces Parabolic Inlet and Zero Pressure Outlet."""
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        # 1. Inlet: Parabolic profile u = 1 - 4*y^2 (ND coords)
        if data.mask_inlet.any():
            y_nd = data.x[data.mask_inlet, 1]
            u_target = 1.0 - 4.0 * (y_nd ** 2)  # Now mathematically correct
            y = data.x[data.mask_inlet, 1]

            l_inlet_u = F.mse_loss(u[data.mask_inlet], u_target)
            l_inlet_v = F.mse_loss(v[data.mask_inlet], torch.zeros_like(u_target))
        else:
            l_inlet_u, l_inlet_v = 0.0, 0.0

        # 2. Outlet: Zero Pressure
        if data.mask_outlet.any():
            l_outlet_p = torch.mean(p[data.mask_outlet] ** 2)
        else:
            l_outlet_p = 0.0

        return l_inlet_u + l_inlet_v + l_outlet_p