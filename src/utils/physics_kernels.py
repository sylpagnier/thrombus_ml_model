import torch
import torch.nn.functional as F


class PhysicsKernels:
    def __init__(self, reynolds=150.0):
        """
        Implements differentiable ND Navier-Stokes residuals + Boundary Constraints.
        Includes Least-Squares Gradients and Inlet/Outlet forcing.

        """
        self.Re = reynolds

    def _manual_scatter_add(self, src, index, dim_size):
        """Native PyTorch scatter_add replacement for compatibility."""
        out_shape = list(src.shape)
        out_shape[0] = dim_size
        out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
        if src.dim() > 1 and index.dim() == 1:
            index = index.unsqueeze(-1).expand_as(src)
        return out.scatter_add_(0, index, src)

    def _compute_graph_gradients(self, f, data):
        """Calculates gradients (df/dx, df/dy) using Weighted Least-Squares."""
        row, col = data.edge_index
        num_nodes = data.num_nodes

        # 1. Edge Distances (dx, dy)
        pos_diff = data.x[col] - data.x[row]
        dx, dy = pos_diff[:, 0], pos_diff[:, 1]
        dist_sq = dx ** 2 + dy ** 2 + 1e-8
        w = 1.0 / dist_sq

        # 2. Build Least-Squares Matrix Components
        m_xx = self._manual_scatter_add(w * dx * dx, row, dim_size=num_nodes)
        m_xy = self._manual_scatter_add(w * dx * dy, row, dim_size=num_nodes)
        m_yy = self._manual_scatter_add(w * dy * dy, row, dim_size=num_nodes)

        det = m_xx * m_yy - m_xy ** 2 + 1e-10
        inv_xx, inv_xy, inv_yy = m_yy / det, -m_xy / det, m_xx / det

        # 3. Solve for Gradients
        f_diff = f[col] - f[row]
        b_x = self._manual_scatter_add(w * f_diff * dx.unsqueeze(1), row, dim_size=num_nodes)
        b_y = self._manual_scatter_add(w * f_diff * dy.unsqueeze(1), row, dim_size=num_nodes)

        grad_x = inv_xx.unsqueeze(1) * b_x + inv_xy.unsqueeze(1) * b_y
        grad_y = inv_xy.unsqueeze(1) * b_x + inv_yy.unsqueeze(1) * b_y

        return torch.cat([grad_x, grad_y], dim=1)

    def navier_stokes_residual(self, pred, data):
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        grad_u = self._compute_graph_gradients(u, data)
        grad_v = self._compute_graph_gradients(v, data)
        grad_p = self._compute_graph_gradients(p, data)

        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        # Second Derivatives (Laplacian)
        grad_u_x = self._compute_graph_gradients(u_x, data)
        grad_u_y = self._compute_graph_gradients(u_y, data)
        grad_v_x = self._compute_graph_gradients(v_x, data)
        grad_v_y = self._compute_graph_gradients(v_y, data)

        lap_u = grad_u_x[:, 0:1] + grad_u_y[:, 1:2]
        lap_v = grad_v_x[:, 0:1] + grad_v_y[:, 1:2]

        # Physics Residuals
        l_cont = u_x + v_y
        mom_x = (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * lap_u
        mom_y = (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * lap_v

        return torch.mean(l_cont ** 2 + mom_x ** 2 + mom_y ** 2)

    def boundary_condition_loss(self, pred, data, decay_rate=20.0):
        """Penalizes velocity at walls (No-Slip)."""
        u, v = pred[:, 0:1], pred[:, 1:2]
        mask = torch.exp(-decay_rate * torch.abs(data.sdf))
        return torch.mean(mask * (u ** 2 + v ** 2))

    def inlet_outlet_loss(self, pred, data):
        """
        The 'Pump': Forces parabolic flow at Inlet (x ~ 0) and Zero Pressure at Outlet.
        """
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        x, y = data.x[:, 0], data.x[:, 1]

        # 1. Inlet Condition (x < 0.1): Parabolic Profile u = 1 - 4y^2
        inlet_mask = (x < 0.1)
        if inlet_mask.sum() > 0:
            # Target parabolic profile
            u_target = 1.0 - 4.0 * (y[inlet_mask] ** 2)
            u_target = torch.clamp(u_target, min=0.0)  # Safety

            l_inlet_u = F.mse_loss(u[inlet_mask].squeeze(), u_target)
            l_inlet_v = F.mse_loss(v[inlet_mask].squeeze(), torch.zeros_like(u_target))
        else:
            l_inlet_u, l_inlet_v = 0.0, 0.0

        # 2. Outlet Condition (x > max_x - 0.1): Pressure Pinning (p=0)
        # This prevents pressure drift and establishes the gradient
        max_x = x.max()
        outlet_mask = (x > (max_x - 0.1))
        if outlet_mask.sum() > 0:
            l_outlet_p = torch.mean(p[outlet_mask] ** 2)
        else:
            l_outlet_p = 0.0

        return l_inlet_u + l_inlet_v + l_outlet_p