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

        # Normalize the viscosity extremes by the target reference
        self.mu_inf_nd = self.cfg.mu_inf / self.cfg.mu_ref
        self.mu_0_nd = self.cfg.mu_0 / self.cfg.mu_ref

    def _compute_carreau_viscosity(self, du_ij, data):
        """
        Calculates local effective non-dimensional viscosity based on the Carreau-Yasuda model.
        Implements batch-aware variable broadcasting and pseudo-Huber smooth regularization.
        """
        # du_ij contains the spatial gradients of velocity: [du/dx, du/dy, dv/dx, dv/dy]
        du_dx, du_dy = du_ij[:, 0], du_ij[:, 1]
        dv_dx, dv_dy = du_ij[:, 2], du_ij[:, 3]

        # --- NON-DIMENSIONAL SHEAR RATE ---
        # Calculate the 2nd invariant of the strain rate tensor: 2*D:D
        strain_sq = 2.0 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2
        eps_sq = 1e-4  # You can tune this (e.g., 1e-4 to 1e-6) depending on precision
        gamma_dot_nd = torch.sqrt(strain_sq + eps_sq)

        # --- BATCH-AWARE BROADCASTING ---
        if hasattr(data, 'batch') and data.batch is not None:
            u_ref_b = data.u_ref[data.batch].squeeze()
            d_bar_b = data.d_bar[data.batch].squeeze()
        else:
            u_ref_b = data.u_ref.squeeze()
            d_bar_b = data.d_bar.squeeze()

        # Scale the relaxation time (lambda) into the non-dimensional domain dynamically
        lambda_nd = self.cfg.lam * (u_ref_b / d_bar_b)

        # --- CARREAU-YASUDA EVALUATION ---
        a = self.cfg.a
        n = self.cfg.n

        # mu = mu_inf + (mu_0 - mu_inf) * [1 + (lambda * gamma_dot)^a]^((n-1)/a)
        shear_term = 1.0 + (lambda_nd * gamma_dot_nd) ** a
        power = (n - 1.0) / a

        mu_nd = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * (shear_term ** power)

        return mu_nd


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

    def _compute_derivatives(self, u, data_or_props):
        """
        Computes 1st and 2nd derivatives using the precomputed 2nd-order WLS operator.
        Safely handles either a PyG Data object or a props dictionary.
        """
        if isinstance(data_or_props, dict):
            row, col = data_or_props['row'], data_or_props['col']
            num_nodes = data_or_props['num_nodes']
            V, W, M_inv = data_or_props['V'], data_or_props['W'], data_or_props['M_inv']
        else:
            row, col = data_or_props.edge_index
            num_nodes = data_or_props.num_nodes
            V, W, M_inv = data_or_props.V, data_or_props.W, data_or_props.M_inv

        if u.dim() == 1:
            u = u.unsqueeze(-1)

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
            if hasattr(data, 'M_inv'):
                props = data
            else:
                props = self._get_geometric_props(data)
        u, v, p = pred[:, 0], pred[:, 1], pred[:, 2]

        batch_idx = getattr(data, 'batch', None)
        if batch_idx is not None:
            u_ref = data.u_ref[batch_idx]
            d_bar = data.d_bar[batch_idx]
        else:
            # Handle un-batched single graphs and raw Python floats safely
            u_ref = data.u_ref.squeeze() if isinstance(data.u_ref, torch.Tensor) else data.u_ref
            d_bar = data.d_bar.squeeze() if isinstance(data.d_bar, torch.Tensor) else data.d_bar
        # --------------------------------------

        # Compute first and second derivatives
        c_u = self._compute_derivatives(u.unsqueeze(1), props)
        c_v = self._compute_derivatives(v.unsqueeze(1), props)
        c_p = self._compute_derivatives(p.unsqueeze(1), props)

        # EXTRACT CROSS-DERIVATIVES (Index 3 maps to the 'dx*dy' basis term)
        u_x, u_y, u_xx, u_xy, u_yy = c_u[:, 0, 0], c_u[:, 1, 0], c_u[:, 2, 0], c_u[:, 3, 0], c_u[:, 4, 0]
        v_x, v_y, v_xx, v_xy, v_yy = c_v[:, 0, 0], c_v[:, 1, 0], c_v[:, 2, 0], c_v[:, 3, 0], c_v[:, 4, 0]
        p_x, p_y = c_p[:, 0, 0], c_p[:, 1, 0]

        # --- Non-Newtonian logic ---
        if self.cfg.viscosity_model == "carreau":
            # Extract the predicted mu directly from the 4th channel
            mu_eff = pred[:, 3]

            # Detach mu_eff before calculating spatial gradients
            # This stops the NS loss from penalizing sharp peaks in viscosity.
            c_mu = self._compute_derivatives(mu_eff.detach().unsqueeze(1), props)
            mu_x, mu_y = c_mu[:, 0, 0], c_mu[:, 1, 0]

            # Re relative to mu_0
            Re = self.cfg.get_re(u_ref, d_bar)
        else:
            # Newtonian path: Viscosity is constant, gradients are 0
            mu_eff = torch.ones_like(u)
            mu_x = torch.zeros_like(u)
            mu_y = torch.zeros_like(u)

            # Re relative to default mu_ref (Newtonian)
            Re = self.cfg.get_re(u_ref, d_bar)

        # Momentum equations (Using the full divergence of the strain rate tensor)
        visc_x = (1.0 / Re) * (mu_eff * (u_xx + v_xy) + 2 * mu_x * u_x + mu_y * (u_y + v_x))
        visc_y = (1.0 / Re) * (mu_eff * (v_yy + u_xy) + 2 * mu_y * v_y + mu_x * (u_y + v_x))

        l_cont = u_x + v_y
        mom_x = (u * u_x + v * u_y) + p_x - visc_x
        mom_y = (u * v_x + v * v_y) + p_y - visc_y

        # Standard PINN procedure: evaluate PDE strictly on interior nodes
        mask_wall_1d = data.mask_wall.view(-1).bool()
        mask_inlet_1d = data.mask_inlet.view(-1).bool()
        mask_outlet_1d = data.mask_outlet.view(-1).bool()

        interior_mask = ~(mask_wall_1d | mask_inlet_1d | mask_outlet_1d)

        if interior_mask.any():
            loss_cont = torch.mean(l_cont[interior_mask] ** 2)
            loss_mom = torch.mean(mom_x[interior_mask] ** 2 + mom_y[interior_mask] ** 2)
        else:
            loss_cont = torch.tensor(0.0, device=pred.device)
            loss_mom = torch.tensor(0.0, device=pred.device)

        return loss_cont, loss_mom

    def rheology_loss(self, pred, data, props=None):
        """
        Penalizes the network if its predicted viscosity (pred[:, 3])
        deviates from the analytical Carreau-Yasuda model given its
        predicted velocity gradients.

        Best Practice Updates:
        1. Detaches velocity fields to prevent gradient collapse.
        2. Detaches the target viscosity to ensure strict 1-way supervision.
        3. Removes unused variable extractions for cleaner execution.
        """
        if self.cfg.viscosity_model != "carreau":
            return torch.tensor(0.0, device=pred.device)

        if props is None:
            props = self._get_geometric_props(data)

        # 1. Extract and DETACH velocities
        # We detach u and v so the optimizer cannot 'cheat' by smoothing the velocity
        # field to artificially lower the shear rate and minimize this loss.
        u = pred[:, 0].detach()
        v = pred[:, 1].detach()
        mu_pred = pred[:, 3]

        # 2. Compute 1st order derivatives using the detached velocities
        c_u = self._compute_derivatives(u.unsqueeze(1), props)
        c_v = self._compute_derivatives(v.unsqueeze(1), props)

        u_x, u_y = c_u[:, 0, 0], c_u[:, 1, 0]
        v_x, v_y = c_v[:, 0, 0], c_v[:, 1, 0]

        du_ij = torch.stack([u_x, u_y, v_x, v_y], dim=1)

        # 3. Compute the theoretical target viscosity
        # _compute_carreau_viscosity safely handles batching internally
        mu_target = self._compute_carreau_viscosity(du_ij, data)

        # Detach the target to be absolutely certain no gradients
        # flow backward through the analytical calculation framework.
        mu_target = mu_target.detach()

        # 4. Safe Log-MSE Loss Calculation
        # Clamp predictions to prevent negative viscosity and NaNs in log
        mu_pred_safe = torch.clamp(mu_pred, min=1e-6, max=100.0)

        # Log-space comparison handles the order-of-magnitude differences
        # typical in shear-thinning fluids without exploding the gradients.
        loss = torch.mean((torch.log(mu_pred_safe) - torch.log(mu_target)) ** 2)

        return loss

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