import torch
from typing import Optional


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

        # ND Carreau bounds use the same scale as label channel STATE_CHANNEL_MU_EFF_ND
        _mu_nd_scale = self.cfg.mu_viscosity_nd_scale
        self.mu_inf_nd = self.cfg.mu_inf / _mu_nd_scale
        self.mu_0_nd = self.cfg.mu_0 / _mu_nd_scale

    def _get_geometric_props(self, data):
        """
        Extracts precomputed geometric properties for direct 2nd-order Weighted Least Squares (WLS).
        """
        return {
            'row': data.edge_index[0],
            'col': data.edge_index[1],
            'num_nodes': data.num_nodes,
            'V': data.V,
            'W': data.W,
            'M_inv': data.M_inv
        }

    def _compute_derivatives(self, u, data_or_props):
        """
        Computes 1st and 2nd derivatives using the precomputed 2nd-order WLS operator.
        Safely handles either a PyG Data object or a props dictionary.

        Contract for ``u`` (no batch/time axis):
            - ``[num_nodes]`` per-node scalar field, or
            - ``[num_nodes, n_channels]`` stacked nodal channels.

        Do not pass ``[1, N, C]`` or other ranks; edge indices address nodes along dim 0.
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

        if u.dim() != 2:
            raise ValueError(
                "_compute_derivatives expects u shaped [num_nodes] or [num_nodes, n_channels]; "
                f"got shape {tuple(u.shape)}."
            )
        if u.shape[0] != num_nodes:
            raise ValueError(
                "_compute_derivatives: length of u along dim 0 must equal num_nodes "
                f"({u.shape[0]} != {num_nodes})."
            )

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

    def navier_stokes_residual(self, pred, data, props=None, re_ref: Optional[float] = None):
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

        # Extract primary 1st and 2nd derivatives, plus cross derivatives (xy)
        u_x, u_y, u_xx, u_yy = c_u[:, 0, 0], c_u[:, 1, 0], c_u[:, 2, 0], c_u[:, 4, 0]
        v_x, v_y, v_xx, v_yy = c_v[:, 0, 0], c_v[:, 1, 0], c_v[:, 2, 0], c_v[:, 4, 0]
        u_xy = c_u[:, 3, 0]
        v_xy = c_v[:, 3, 0]
        p_x, p_y = c_p[:, 0, 0], c_p[:, 1, 0]

        # Re uses per-graph ``u_ref`` / ``d_bar`` (see ``PhysicsConfig.get_re``), not ``re_target`` directly.
        # Callers (e.g. Tier 3 training) may override with ``re_ref`` from ``data.re_actual``.
        Re = self.cfg.get_re(u_ref, d_bar)
        if re_ref is not None:
            ref_t = torch.as_tensor(re_ref, device=Re.device, dtype=Re.dtype)
            Re = ref_t.expand_as(Re)

        # --- Tier-Dependent Physics Formulation ---
        if self.cfg.viscosity_model == "carreau":
            # Extract predicted mu
            mu_eff = pred[:, 3]

            mu_for_grad = mu_eff.detach() if self.cfg.detach_mu_for_ns_gradient else mu_eff
            # Compute viscosity gradients in log-space to reduce ringing across sharp clot interfaces.
            log_mu = torch.log(mu_for_grad + 1e-8)
            c_log_mu = self._compute_derivatives(log_mu.unsqueeze(1), props)
            log_mu_x, log_mu_y = c_log_mu[:, 0, 0], c_log_mu[:, 1, 0]
            mu_x = mu_for_grad * log_mu_x
            mu_y = mu_for_grad * log_mu_y
            # Bound viscosity-gradient spikes from strong-form WLS around sharp clot interfaces.
            max_grad = 5.0 * self.cfg.mu_viscosity_nd_scale
            mu_x = torch.clamp(mu_x, min=-max_grad, max=max_grad)
            mu_y = torch.clamp(mu_y, min=-max_grad, max=max_grad)

            # Full divergence of the stress tensor (NO strict incompressibility assumption)
            visc_x = (1.0 / Re) * (2 * mu_x * u_x + mu_y * (u_y + v_x) + mu_eff * (2 * u_xx + u_yy + v_xy))
            visc_y = (1.0 / Re) * (2 * mu_y * v_y + mu_x * (u_y + v_x) + mu_eff * (2 * v_yy + v_xx + u_xy))

        else:
            # Full Newtonian formulation
            visc_x = (1.0 / Re) * (2 * u_xx + u_yy + v_xy)
            visc_y = (1.0 / Re) * (2 * v_yy + v_xx + u_xy)

        # Calculate Convective Momentum
        mom_x = (u * u_x + v * u_y) + p_x - visc_x
        mom_y = (u * v_x + v * v_y) + p_y - visc_y

        # Standard PINN procedure: evaluate PDE strictly on interior nodes
        mask_wall_1d = data.mask_wall.view(-1).bool()
        mask_inlet_1d = data.mask_inlet.view(-1).bool()
        mask_outlet_1d = data.mask_outlet.view(-1).bool()

        interior_mask = ~(mask_wall_1d | mask_inlet_1d | mask_outlet_1d)

        if interior_mask.any():
            loss_mom = torch.mean(mom_x[interior_mask] ** 2 + mom_y[interior_mask] ** 2)
        else:
            loss_mom = pred.sum() * 0.0

        # Return ONLY momentum. Continuity is handled by the dedicated continuity_loss method.
        return loss_mom

    def continuity_loss(self, du_ij):
        """
        Strictly penalizes mass creation/destruction.
        du_ij: [du_dx, du_dy, dv_dx, dv_dy]
        """
        du_dx = du_ij[:, 0]
        dv_dy = du_ij[:, 3]

        # Divergence of velocity field
        div_u = du_dx + dv_dy

        # L2 norm of the divergence
        return torch.mean(div_u ** 2)

    def _compute_carreau_viscosity(self, du_ij, data, carreau_n: Optional[float] = None):
        """
        Calculates local effective non-dimensional viscosity based on the Carreau-Yasuda model.
        Implements batch-aware variable broadcasting and pseudo-Huber smooth regularization.
        """
        du_dx, du_dy = du_ij[:, 0], du_ij[:, 1]
        dv_dx, dv_dy = du_ij[:, 2], du_ij[:, 3]

        # --- NON-DIMENSIONAL SHEAR RATE ---
        strain_sq = 2.0 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2

        # BEST PRACTICE: eps_nd must be scaled to the non-dimensional latent space.
        # Since u_nd and x_nd are ~ O(1), shear is O(1).
        # A value of 1e-6 (eps_nd = 1e-3) provides smooth derivatives near 0 without destroying rheology.
        eps_sq_nd = 1e-6
        gamma_dot_nd = torch.sqrt(strain_sq + eps_sq_nd)

        # --- BATCH-AWARE BROADCASTING ---
        if hasattr(data, 'batch') and data.batch is not None:
            u_ref_b = data.u_ref[data.batch].squeeze()
            d_bar_b = data.d_bar[data.batch].squeeze()
        else:
            u_ref_b = data.u_ref.squeeze()
            d_bar_b = data.d_bar.squeeze()

        # Scale the relaxation time (lambda) into the non-dimensional domain dynamically
        lambda_nd = self.cfg.lam * (u_ref_b / d_bar_b)

        a = self.cfg.a
        n = carreau_n if carreau_n is not None else self.cfg.n

        # Evaluate the Carreau-Yasuda equation
        shear_term = 1.0 + (lambda_nd * gamma_dot_nd) ** a
        power = (n - 1.0) / a

        mu_nd = self.mu_inf_nd + (self.mu_0_nd - self.mu_inf_nd) * (shear_term ** power)

        return mu_nd

    def rheology_loss(self, pred, data, props=None, carreau_n: Optional[float] = None):
        """
        Detached Rheology Supervisor:
        Forces the network's surrogate viscosity output to perfectly match the
        analytical Carreau-Yasuda target, while strictly preventing the optimizer
        from 'cheating' by artificially smoothing the underlying velocity fields.
        """
        if self.cfg.viscosity_model != "carreau":
            return pred.sum() * 0.0

        if props is None:
            props = self._get_geometric_props(data)

        # 1. Isolate the current velocity fields and strictly DETACH them from the graph.
        # This acts as a frozen 1-way mirror for the supervisor.
        u_detached = pred[:, 0].detach()
        v_detached = pred[:, 1].detach()
        mu_pred = pred[:, 3]  # This is what we want to train

        # 2. Recompute the gradients using the detached velocity fields
        c_u = self._compute_derivatives(u_detached.unsqueeze(1), props)
        c_v = self._compute_derivatives(v_detached.unsqueeze(1), props)

        u_x, u_y = c_u[:, 0, 0], c_u[:, 1, 0]
        v_x, v_y = c_v[:, 0, 0], c_v[:, 1, 0]
        du_ij = torch.stack([u_x, u_y, v_x, v_y], dim=1)

        # 3. Compute the analytical target and explicitly detach it as the ground truth
        mu_target = self._compute_carreau_viscosity(du_ij, data, carreau_n=carreau_n).detach()

        dynamic_max = self.mu_0_nd * 1.2
        mu_pred_safe = torch.clamp(mu_pred, min=1e-6, max=dynamic_max)

        # Use Log-MSE or Log-Absolute to normalize gradient scale
        pointwise_loss = torch.abs(torch.log(mu_pred_safe) - torch.log(mu_target))

        # 4. Physical Attention Weighting
        # Reconstruct the shear rate (from detached fields) to weight the loss physically
        strain_sq = 2.0 * (u_x ** 2 + v_y ** 2) + (u_y + v_x) ** 2
        gamma_dot_nd = torch.sqrt(strain_sq + 1e-6)

        # Emphasize high-shear regions (platelet activation risk) and non-Newtonian zones
        shear_multiplier = 1.0 + 2.0 * gamma_dot_nd

        loss = torch.mean(shear_multiplier * pointwise_loss)

        return loss

    def boundary_condition_loss(self, pred, data):
        u, v = pred[:, 0], pred[:, 1]
        loss_wall = torch.tensor(0.0, device=pred.device)
        mask_wall_1d = data.mask_wall.view(-1).bool()

        if mask_wall_1d.any():
            # 1. Kinematics (No-slip velocity)
            u_wall = u[mask_wall_1d]
            v_wall = v[mask_wall_1d]
            loss_wall = torch.mean(u_wall ** 2 + v_wall ** 2)

            # --- Rheology (Minimum Viscosity at Max Shear) ---
            if self.cfg.viscosity_model == "carreau" and hasattr(data, 'mu_wall_bc'):
                mu_pred = pred[:, 3]
                mu_wall_pred = mu_pred[mask_wall_1d].squeeze()

                # Ensure the BC is on the correct GPU device
                mu_target_bc = data.mu_wall_bc[mask_wall_1d].squeeze().to(pred.device)

                # Weight matches the inlet penalty
                loss_wall += 2.0 * torch.nn.functional.smooth_l1_loss(mu_wall_pred, mu_target_bc)

        return loss_wall

    def wall_shear_stress_loss(self, pred, data, props=None):
        """
        Enforces that the model's predicted WSS strictly matches the physical WSS
        derived from its own velocity gradients at the wall boundaries.
        """
        if props is None:
            props = self._get_geometric_props(data)

        u, v = pred[:, 0:1], pred[:, 1:2]
        mu = pred[:, 3:4]
        wss_pred = pred[:, 4]

        mask_wall = data.mask_wall.view(-1).bool()
        if not mask_wall.any():
            return pred.sum() * 0.0

        c_u = self._compute_derivatives(u, props)
        c_v = self._compute_derivatives(v, props)

        dudx, dudy = c_u[:, 0, 0], c_u[:, 1, 0]
        dvdx, dvdy = c_v[:, 0, 0], c_v[:, 1, 0]

        # --- 1. Compute 2D Viscous Stress Tensor (tau) at the walls ---
        mu_wall = mu[mask_wall, 0]
        tau_xx = 2.0 * mu_wall * dudx[mask_wall]
        tau_yy = 2.0 * mu_wall * dvdy[mask_wall]
        tau_xy = mu_wall * (dudy[mask_wall] + dvdx[mask_wall])

        # --- 2. Extract Wall Normals (Columns 4 and 5 in x_tensor) ---
        nx = data.x[mask_wall, 4]
        ny = data.x[mask_wall, 5]

        # --- 3. Project to find Traction Vector (T = tau * n) ---
        tx = tau_xx * nx + tau_xy * ny
        ty = tau_xy * nx + tau_yy * ny

        # --- 4. Extract Tangential Component (WSS) ---
        t_n = tx * nx + ty * ny
        wss_x_phys = tx - t_n * nx
        wss_y_phys = ty - t_n * ny
        wss_mag_phys = torch.sqrt(wss_x_phys ** 2 + wss_y_phys ** 2 + 1e-8)

        # Target WSS is what the model explicitly predicted in channel 4
        wss_pred_wall = wss_pred[mask_wall]

        return torch.nn.functional.mse_loss(wss_pred_wall, wss_mag_phys)


    def inlet_outlet_loss(self, pred, data):
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        device = pred.device # Get active device

        loss_inlet = torch.tensor(0.0, device=device)
        mask_inlet_1d = data.mask_inlet.view(-1).bool()

        if mask_inlet_1d.any():
            u_target = data.u_inlet_bc[mask_inlet_1d, 0].to(device)

            u_in = u[mask_inlet_1d].squeeze()
            v_in = v[mask_inlet_1d].squeeze()

            loss_inlet = torch.mean((u_in - u_target) ** 2 + v_in ** 2)

            if self.cfg.viscosity_model == "carreau" and hasattr(data, 'mu_inlet_bc'):
                mu_pred = pred[:, 3]
                mu_in = mu_pred[mask_inlet_1d].squeeze()
                mu_target_bc = data.mu_inlet_bc[mask_inlet_1d].squeeze().to(device)

                loss_inlet += 2.0 * torch.nn.functional.smooth_l1_loss(mu_in, mu_target_bc)

        loss_outlet = torch.tensor(0.0, device=pred.device)
        mask_outlet_1d = data.mask_outlet.view(-1).bool()
        if mask_outlet_1d.any():
            p_out = p[mask_outlet_1d]
            loss_outlet = torch.mean(p_out ** 2)

        return loss_inlet + loss_outlet