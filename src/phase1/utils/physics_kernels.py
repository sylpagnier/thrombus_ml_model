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
        Computes geometric properties for direct 2nd-order Weighted Least Squares (WLS).
        Extracts [u_x, u_y, u_xx, u_xy, u_yy] simultaneously, completely eliminating
        boundary truncation errors associated with chained 1st-order gradients.
        """
        row, col = data.edge_index
        num_nodes = data.num_nodes

        # 1. Edge Distances
        pos_diff = data.x[col, :2] - data.x[row, :2]
        dx, dy = pos_diff[:, 0], pos_diff[:, 1]

        # Regularizer to prevent division by zero in weights
        dist_sq = dx ** 2 + dy ** 2 + 1e-8

        # 2nd Order Polynomial Basis: [dx, dy, 0.5*dx^2, dx*dy, 0.5*dy^2]
        dx2 = 0.5 * dx ** 2
        dxy = dx * dy
        dy2 = 0.5 * dy ** 2

        V = torch.stack([dx, dy, dx2, dxy, dy2], dim=1)  # [E, 5]

        # Weights: inverse distance squared
        W = 1.0 / dist_sq  # [E]

        # Compute M_e = W * V^T * V for each edge
        V_unsqueezed = V.unsqueeze(2)  # [E, 5, 1]
        V_T_unsqueezed = V.unsqueeze(1)  # [E, 1, 5]
        M_e = W.view(-1, 1, 1) * torch.bmm(V_unsqueezed, V_T_unsqueezed)  # [E, 5, 5]

        # Scatter sum to get M for each node
        M_e_flat = M_e.view(-1, 25)
        M_flat = scatter_add(M_e_flat, row, dim=0, dim_size=num_nodes)
        M = M_flat.view(num_nodes, 5, 5)

        # Compute pseudo-inverse directly. This handles low-degree boundary nodes safely.
        M_inv = torch.linalg.pinv(M)  # [N, 5, 5]

        return {
            'row': row,
            'col': col,
            'num_nodes': num_nodes,
            'V': V,
            'W': W,
            'M_inv': M_inv
        }

    def _compute_derivatives(self, u, props):
        """
        Computes 1st and 2nd derivatives using the 2nd-order WLS operator.
        Returns tensor of shape [N, 5, C].
        """
        row, col = props['row'], props['col']
        num_nodes = props['num_nodes']
        V, W, M_inv = props['V'], props['W'], props['M_inv']

        if u.dim() == 1:
            u = u.unsqueeze(1)

        C = u.shape[1]
        du = u[col] - u[row]  # [E, C]

        W_unsqueezed = W.view(-1, 1, 1)  # [E, 1, 1]
        V_unsqueezed = V.unsqueeze(2)  # [E, 5, 1]
        du_unsqueezed = du.unsqueeze(1)  # [E, 1, C]

        b_e = W_unsqueezed * torch.bmm(V_unsqueezed, du_unsqueezed)  # [E, 5, C]

        b_e_flat = b_e.view(-1, 5 * C)
        b_flat = scatter_add(b_e_flat, row, dim=0, dim_size=num_nodes)
        b = b_flat.view(num_nodes, 5, C)  # [N, 5, C]

        # [N, 5, 5] x [N, 5, C] -> [N, 5, C]
        c = torch.bmm(M_inv, b)

        return c

    def _compute_gradients(self, u, props):
        """Legacy wrapper to maintain compatibility with existing tests."""
        c = self._compute_derivatives(u, props)
        C = c.shape[2]
        if C == 1:
            return c[:, 0:2, 0]
        else:
            return c[:, 0:2, :].reshape(-1, 2 * C)

    def navier_stokes_residual(self, pred, data, props=None):
        if props is None:
            props = self._get_geometric_props(data)

        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        c_u = self._compute_derivatives(u, props)  # [N, 5, 1]
        c_v = self._compute_derivatives(v, props)  # [N, 5, 1]
        c_p = self._compute_derivatives(p, props)  # [N, 5, 1]

        u_x, u_y = c_u[:, 0, 0], c_u[:, 1, 0]
        v_x, v_y = c_v[:, 0, 0], c_v[:, 1, 0]
        p_x, p_y = c_p[:, 0, 0], c_p[:, 1, 0]

        # Use explicitly calculated Laplacian instead of chained gradients!
        u_xx, u_yy = c_u[:, 2, 0], c_u[:, 4, 0]
        v_xx, v_yy = c_v[:, 2, 0], c_v[:, 4, 0]

        lap_u = u_xx + u_yy
        lap_v = v_xx + v_yy

        l_cont = u_x + v_y
        mom_x = (u * u_x + v * u_y) + p_x - (1.0 / self.Re) * lap_u
        mom_y = (u * v_x + v * v_y) + p_y - (1.0 / self.Re) * lap_v

        # Standard PINN procedure: evaluate PDE strictly on interior nodes
        mask_wall_1d = data.mask_wall.view(-1).bool()
        interior_mask = ~mask_wall_1d

        if interior_mask.any():
            res = torch.mean(l_cont[interior_mask] ** 2 + mom_x[interior_mask] ** 2 + mom_y[interior_mask] ** 2)
        else:
            res = torch.tensor(0.0, device=pred.device)

        return res

    def boundary_condition_loss(self, pred, data):
        u, v = pred[:, 0:1], pred[:, 1:2]
        mask_wall_1d = data.mask_wall.view(-1).bool()
        if mask_wall_1d.any():
            u_wall = u[mask_wall_1d]
            v_wall = v[mask_wall_1d]
            return torch.mean(u_wall ** 2 + v_wall ** 2)
        return torch.tensor(0.0, device=pred.device)

    def inlet_outlet_loss(self, pred, data):
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        loss_inlet = torch.tensor(0.0, device=pred.device)
        mask_inlet_1d = data.mask_inlet.view(-1).bool()
        if mask_inlet_1d.any():
            y_nd = data.x[mask_inlet_1d, 1]
            y_centered = y_nd - y_nd.mean()

            u_target = 1.5 * (1.0 - 4.0 * (y_centered ** 2))
            u_target = torch.clamp(u_target, min=0.0)

            u_in = u[mask_inlet_1d].squeeze()
            v_in = v[mask_inlet_1d].squeeze()

            loss_inlet = torch.mean((u_in - u_target) ** 2 + v_in ** 2)

        loss_outlet = torch.tensor(0.0, device=pred.device)
        mask_outlet_1d = data.mask_outlet.view(-1).bool()
        if mask_outlet_1d.any():
            p_out = p[mask_outlet_1d]
            loss_outlet = torch.mean(p_out ** 2)

        return loss_inlet + loss_outlet