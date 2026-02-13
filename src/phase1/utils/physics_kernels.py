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

    def _get_geometric_props(self, data):
        """
        Computes and caches geometric properties (distances, least-squares matrices).
        Crucial for WLS consistency on unstructured meshes.
        """
        row, col = data.edge_index
        num_nodes = data.num_nodes

        # 1. Edge Distances
        pos_diff = data.x[col, :2] - data.x[row, :2]
        dx, dy = pos_diff[:, 0], pos_diff[:, 1]
        dist_sq = dx ** 2 + dy ** 2 + 1e-8

        # Weighting: Inverse distance squared (Standard WLS)
        w = 1.0 / (dist_sq + 1e-8)

        # 2. Least-Squares Matrix Components
        # We solve for gradients [gx, gy] minimizing: sum w * (df - gx*dx - gy*dy)^2
        m_xx = scatter_add(w * dx * dx, row, dim=0, dim_size=num_nodes)
        m_xy = scatter_add(w * dx * dy, row, dim=0, dim_size=num_nodes)
        m_yy = scatter_add(w * dy * dy, row, dim=0, dim_size=num_nodes)

        # Determinant for analytic 2x2 inversion
        det = m_xx * m_yy - m_xy ** 2 + 1e-8

        # Inverse components (Cramer's Rule)
        inv_xx = m_yy / det
        inv_xy = -m_xy / det
        inv_yy = m_xx / det

        return {
            'row': row, 'col': col, 'num_nodes': num_nodes,
            'dx': dx, 'dy': dy, 'w': w,
            'inv_xx': inv_xx, 'inv_xy': inv_xy, 'inv_yy': inv_yy
        }

    def _compute_gradients(self, f, props):
        """Calculates First Derivatives (df/dx, df/dy) using Weighted Least Squares."""
        row, col = props['row'], props['col']

        # Difference in function values
        f_diff = f[col] - f[row]

        # Weighted difference vectors
        w_f = props['w'].unsqueeze(1) * f_diff

        b_x = scatter_add(w_f * props['dx'].unsqueeze(1), row, dim=0, dim_size=props['num_nodes'])
        b_y = scatter_add(w_f * props['dy'].unsqueeze(1), row, dim=0, dim_size=props['num_nodes'])

        # Multiply by inverse geometric matrix
        grad_x = props['inv_xx'].unsqueeze(1) * b_x + props['inv_xy'].unsqueeze(1) * b_y
        grad_y = props['inv_xy'].unsqueeze(1) * b_x + props['inv_yy'].unsqueeze(1) * b_y

        return torch.cat([grad_x, grad_y], dim=1)

    def navier_stokes_residual(self, pred, data):
        """Residual of steady-state Navier-Stokes using Consistent Div-Grad Laplacian."""
        # 1. Precompute Geometry
        # In training, you might cache this if topology is static, but dynamic batching requires recompute
        props = self._get_geometric_props(data)

        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        # 2. First Derivatives (Velocity & Pressure)
        grad_u = self._compute_gradients(u, props)  # [N, 2] -> (u_x, u_y)
        grad_v = self._compute_gradients(v, props)  # [N, 2] -> (v_x, v_y)
        grad_p = self._compute_gradients(p, props)

        u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
        v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        # 3. Laplacian via Divergence of Gradient (Consistent WLS)
        # This re-uses the WLS coefficients, ensuring spectral consistency with the gradient
        grad_u_x = self._compute_gradients(u_x, props)  # [N, 2] -> (u_xx, u_xy)
        grad_u_y = self._compute_gradients(u_y, props)  # [N, 2] -> (u_yx, u_yy)
        lap_u = grad_u_x[:, 0:1] + grad_u_y[:, 1:2]

        grad_v_x = self._compute_gradients(v_x, props)
        grad_v_y = self._compute_gradients(v_y, props)
        lap_v = grad_v_x[:, 0:1] + grad_v_y[:, 1:2]

        # 4. Navier-Stokes Equations
        # Continuity: div(u) = 0
        l_cont = u_x + v_y

        # Momentum X: (u.grad)u + grad_p - (1/Re)lap_u
        mom_x = (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * lap_u

        # Momentum Y: (u.grad)v + grad_p - (1/Re)lap_v
        mom_y = (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * lap_v

        return torch.mean(l_cont ** 2 + mom_x ** 2 + mom_y ** 2)

    def boundary_condition_loss(self, pred, data):
        """Penalizes velocity at walls."""
        u, v = pred[:, 0:1], pred[:, 1:2]
        if data.mask_wall.any():
            u_wall = u[data.mask_wall]
            v_wall = v[data.mask_wall]
            return torch.mean(u_wall ** 2 + v_wall ** 2)
        return torch.tensor(0.0, device=pred.device)

    def inlet_outlet_loss(self, pred, data):
        """Forces Parabolic Inlet and Zero Pressure Outlet."""
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        # 1. Inlet: Parabolic profile
        if data.mask_inlet.any():
            y_nd = data.x[data.mask_inlet, 1]
            # Dynamic centering for robustness against mesh shifts
            y_centered = y_nd - y_nd.mean()

            # Profile: 1.5 * (1 - (2y)^2) assuming channel width ~1.0 in ND space
            u_target = 1.5 * (1.0 - 4.0 * (y_centered ** 2))
            u_target = torch.clamp(u_target, min=0.0)

            l_inlet_u = F.mse_loss(u[data.mask_inlet].squeeze(), u_target)
            l_inlet_v = F.mse_loss(v[data.mask_inlet].squeeze(), torch.zeros_like(u_target))
        else:
            l_inlet_u, l_inlet_v = 0.0, 0.0

        # 2. Outlet: Zero Pressure
        if data.mask_outlet.any():
            l_outlet_p = torch.mean(p[data.mask_outlet] ** 2)
        else:
            l_outlet_p = 0.0

        return l_inlet_u + l_inlet_v + l_outlet_p