import torch


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
    def __init__(self, phys_cfg):
        self.cfg = phys_cfg
        # Normalize non-Newtonian parameters by reference values
        self.mu_inf_nd = self.cfg.mu_inf / self.cfg.mu_inf
        self.mu_0_nd = self.cfg.mu_0 / self.cfg.mu_inf
        # Note: lambda must be scaled by (u_ref / d_bar) to be non-dimensional

    def _compute_carreau_viscosity(self, du_ij, u_ref, d_bar):
        """
        Calculates local effective viscosity based on Carreau-Yasuda model.
        du_ij: tensor containing [u_x, u_y, v_x, v_y]
        """
        # 1. Compute Shear Rate (gamma_dot)
        # gamma_dot = sqrt(2 * D : D) where D is strain rate tensor
        u_x, u_y, v_x, v_y = du_ij[:, 0], du_ij[:, 1], du_ij[:, 2], du_ij[:, 3]

        # Second invariant of strain rate tensor
        gamma_dot = torch.sqrt(2 * u_x ** 2 + 2 * v_y ** 2 + (u_y + v_x) ** 2 + 1e-8)

        # Non-dimensionalize lambda
        lam_nd = self.cfg.lam * (u_ref / d_bar)

        # Carreau-Yasuda Equation
        pow_term = (1 + (lam_nd * gamma_dot) ** self.cfg.a) ** ((self.cfg.n - 1) / self.cfg.a)
        mu_eff = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * pow_term
        return mu_eff

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
        if props is None: props = self._get_geometric_props(data)
        u, v, p = pred[:, 0], pred[:, 1], pred[:, 2]
        u_ref, d_bar = data.u_ref, data.d_bar

        # Compute first and second derivatives
        c_u = self._compute_derivatives(u.unsqueeze(1), props)
        c_v = self._compute_derivatives(v.unsqueeze(1), props)
        c_p = self._compute_derivatives(p.unsqueeze(1), props)

        u_x, u_y, u_xx, u_yy = c_u[:, 0, 0], c_u[:, 1, 0], c_u[:, 2, 0], c_u[:, 4, 0]
        v_x, v_y, v_xx, v_yy = c_v[:, 0, 0], c_v[:, 1, 0], c_v[:, 2, 0], c_v[:, 4, 0]
        p_x, p_y = c_p[:, 0, 0], c_p[:, 1, 0]

        # --- Dynamic Viscosity Logic (Tier 2 Update) ---
        if self.cfg.viscosity_model == "carreau":
            # Extract the predicted mu directly from the 4th channel
            mu_eff = pred[:, 3]

            # Compute spatial gradients of the predicted mu
            c_mu = self._compute_derivatives(mu_eff.unsqueeze(1), props)
            mu_x, mu_y = c_mu[:, 0, 0], c_mu[:, 1, 0]

            # Re relative to mu_inf
            Re = (self.cfg.rho * u_ref * d_bar) / self.cfg.mu_inf
        else:
            # Newtonian path: Viscosity is constant, gradients are 0
            mu_eff = torch.ones_like(u)
            mu_x = torch.zeros_like(u)
            mu_y = torch.zeros_like(u)

            # Re relative to mu_newtonian
            Re = (self.cfg.rho * u_ref * d_bar) / self.cfg.mu_newtonian

        # Momentum equations (this unified form handles both cases)
        visc_x = (1.0 / Re) * (mu_eff * (u_xx + u_yy) + 2 * mu_x * u_x + mu_y * (u_y + v_x))
        visc_y = (1.0 / Re) * (mu_eff * (v_xx + v_yy) + 2 * mu_y * v_y + mu_x * (u_y + v_x))

        l_cont = u_x + v_y
        mom_x = (u * u_x + v * u_y) + p_x - visc_x
        mom_y = (u * v_x + v * v_y) + p_y - visc_y

        # Standard PINN procedure: evaluate PDE strictly on interior nodes
        mask_wall_1d = data.mask_wall.view(-1).bool()
        interior_mask = ~mask_wall_1d

        if interior_mask.any():
            # Ensure EVERY term has [interior_mask]
            res = torch.mean(
                l_cont[interior_mask] ** 2 +
                mom_x[interior_mask] ** 2 +
                mom_y[interior_mask] ** 2
            )
        else:
            res = torch.tensor(0.0, device=pred.device)

        return res

    def rheology_loss(self, pred, data, props=None):
        """
        Penalizes the network if its predicted viscosity (pred[:, 3])
        deviates from the analytical Carreau-Yasuda model given its
        predicted velocity gradients.
        """
        if self.cfg.viscosity_model != "carreau":
            return torch.tensor(0.0, device=pred.device)

        if props is None:
            props = self._get_geometric_props(data)

        # Extract predictions
        u, v, mu_pred = pred[:, 0], pred[:, 1], pred[:, 3]
        u_ref, d_bar = data.u_ref, data.d_bar

        # We only need 1st order derivatives for the strain rate
        c_u = self._compute_derivatives(u.unsqueeze(1), props)
        c_v = self._compute_derivatives(v.unsqueeze(1), props)

        u_x, u_y = c_u[:, 0, 0], c_u[:, 1, 0]
        v_x, v_y = c_v[:, 0, 0], c_v[:, 1, 0]

        du_ij = torch.stack([u_x, u_y, v_x, v_y], dim=1)

        # Compute the theoretical target viscosity based on current predicted flow
        mu_target = self._compute_carreau_viscosity(du_ij, u_ref, d_bar)

        # Mean Squared Error between the network's mu prediction and the analytical mu
        return torch.mean((mu_pred - mu_target) ** 2)

    def boundary_condition_loss(self, pred, data):
        # Squeeze to [N] to be safe
        u, v = pred[:, 0], pred[:, 1]
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
            # Use robust pre-calculated BC ---
            # data.u_inlet_bc is shape [N, 2], we need u component (idx 0)
            u_target = data.u_inlet_bc[mask_inlet_1d, 0]

            u_in = u[mask_inlet_1d].squeeze()
            v_in = v[mask_inlet_1d].squeeze()

            loss_inlet = torch.mean((u_in - u_target) ** 2 + v_in ** 2)

        loss_outlet = torch.tensor(0.0, device=pred.device)
        mask_outlet_1d = data.mask_outlet.view(-1).bool()
        if mask_outlet_1d.any():
            p_out = p[mask_outlet_1d]
            loss_outlet = torch.mean(p_out ** 2)

        return loss_inlet + loss_outlet