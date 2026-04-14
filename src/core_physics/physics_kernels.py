import os
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
        self.ns_derivative_mode = os.environ.get("PHYSICS_NS_DERIVATIVE_MODE", "wls").strip().lower()
        if self.ns_derivative_mode not in ("wls", "autograd"):
            self.ns_derivative_mode = "wls"
        self.advect_detach = os.environ.get("TIER1_ADVECT_DETACH", "0").strip().lower() in ("1", "true", "yes", "on")
        self.momentum_loss_mode = os.environ.get("TIER1_MOMENTUM_LOSS_MODE", "huber").strip().lower()
        if self.momentum_loss_mode not in ("huber", "mse"):
            self.momentum_loss_mode = "huber"
        self.momentum_huber_delta = float(os.environ.get("TIER1_MOMENTUM_HUBER_DELTA", "0.01"))
        self.pressure_bc_mode = os.environ.get("TIER1_PRESSURE_BC_MODE", "mean").strip().lower()
        if self.pressure_bc_mode not in ("mean", "pointwise", "mean_var"):
            self.pressure_bc_mode = "mean"
        self._autograd_mode_warned = False

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

        # Backward compatibility: older graphs may store M_inv with shape [N, 1, 5, 5].
        if M_inv.dim() == 4 and M_inv.shape[1] == 1:
            M_inv = M_inv.squeeze(1)

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

    def fluid_interior_mask(self, data) -> torch.Tensor:
        """Boolean mask of nodes where steady NS momentum and ∇·u penalties are evaluated.

        Excludes **wall**, **inlet**, and **outlet** nodes. Boundary conditions are enforced
        via separate losses; WLS derivatives are least reliable on tagged boundaries, so bulk
        collocation matches common PINN / physics-informed practice and aligns momentum with
        continuity in training.
        """
        mask_wall_1d = data.mask_wall.view(-1).bool()
        mask_inlet_1d = data.mask_inlet.view(-1).bool()
        mask_outlet_1d = data.mask_outlet.view(-1).bool()
        return ~(mask_wall_1d | mask_inlet_1d | mask_outlet_1d)

    @staticmethod
    def _safe_grad(outputs, inputs, retain_graph=True, create_graph=True):
        g = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
        if g is None:
            return torch.zeros_like(inputs)
        return g

    def _compute_autograd_derivatives(self, field: torch.Tensor, coords_xy: torch.Tensor):
        """Compute [x, y, xx, xy, yy] derivatives using torch.autograd."""
        d1 = self._safe_grad(field, coords_xy)
        dfdx = d1[:, 0]
        dfdy = d1[:, 1]
        d2x = self._safe_grad(dfdx, coords_xy)
        d2y = self._safe_grad(dfdy, coords_xy)
        dfdxx = d2x[:, 0]
        dfdxy = d2x[:, 1]
        dfdyy = d2y[:, 1]
        return dfdx, dfdy, dfdxx, dfdxy, dfdyy

    def navier_stokes_residual(
        self,
        pred,
        data,
        props=None,
        re_ref: Optional[float] = None,
        re_scale: Optional[float] = None,
    ):
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

        if self.ns_derivative_mode == "autograd" and not self._autograd_mode_warned:
            print("⚠️ ns_derivative_mode='autograd' requested, but PDE derivatives in message-passing GNNs are non-local. Falling back to WLS derivatives.")
            self._autograd_mode_warned = True

        # Compute first and second derivatives from the WLS operator.
        c_u = self._compute_derivatives(u.unsqueeze(1), props)
        c_v = self._compute_derivatives(v.unsqueeze(1), props)
        c_p = self._compute_derivatives(p.unsqueeze(1), props)

        # Extract primary 1st and 2nd derivatives, plus cross derivatives (xy).
        # NOTE: WLS basis is [dx, dy, 0.5*dx^2, dx*dy, 0.5*dy^2] in mesh_wls.py,
        # so c[:,2] and c[:,4] already correspond to true u_xx / u_yy (no extra factor of 2 needed).
        u_x, u_y, u_xx, u_yy = c_u[:, 0, 0], c_u[:, 1, 0], c_u[:, 2, 0], c_u[:, 4, 0]
        v_x, v_y, v_xx, v_yy = c_v[:, 0, 0], c_v[:, 1, 0], c_v[:, 2, 0], c_v[:, 4, 0]
        u_xy = c_u[:, 3, 0]
        v_xy = c_v[:, 3, 0]
        p_x, p_y = c_p[:, 0, 0], c_p[:, 1, 0]

        # Re uses per-graph ``u_ref`` / ``d_bar`` (see ``PhysicsConfig.get_re``), not ``re_target`` directly.
        # Callers (e.g. Tier 3 training) may override with ``re_ref`` from ``data.re_actual``.
        Re = self.cfg.get_re(u_ref, d_bar)
        if re_scale is not None:
            scale_t = torch.as_tensor(re_scale, device=Re.device, dtype=Re.dtype)
            Re = Re * scale_t
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
            # Newtonian stream-function formulation: robust Laplacian form.
            visc_x = (1.0 / Re) * (u_xx + u_yy)
            visc_y = (1.0 / Re) * (v_xx + v_yy)

        # Calculate Convective Momentum
        if self.advect_detach:
            mom_x = (u.detach() * u_x + v.detach() * u_y) + p_x - visc_x
            mom_y = (u.detach() * v_x + v.detach() * v_y) + p_y - visc_y
        else:
            mom_x = (u * u_x + v * u_y) + p_x - visc_x
            mom_y = (u * v_x + v * v_y) + p_y - visc_y

        interior_mask = self.fluid_interior_mask(data)

        if interior_mask.any():
            mom_sq = mom_x[interior_mask] ** 2 + mom_y[interior_mask] ** 2
            if self.momentum_loss_mode == "mse":
                loss_mom = torch.mean(mom_sq)
            else:
                delta = max(float(self.momentum_huber_delta), 1e-8)
                loss_mom = torch.mean(delta ** 2 * (torch.sqrt(1.0 + mom_sq / (delta ** 2)) - 1.0))
        else:
            loss_mom = pred.sum() * 0.0

        # Return ONLY momentum. Continuity is handled by the dedicated continuity_loss method.
        return loss_mom

    def continuity_loss(
        self,
        du_ij: torch.Tensor,
        data=None,
        interior_mask: Optional[torch.Tensor] = None,
        disabled: bool = False,
    ):
        """
        Mean squared divergence (∇·u)².

        When ``data`` is provided (or ``interior_mask`` is passed), averages only over
        :meth:`fluid_interior_mask` so continuity matches the momentum residual domain.
        If neither is given, averages over all rows of ``du_ij`` (synthetic / unit tests).
        """
        if disabled:
            return du_ij.sum() * 0.0

        du_dx = du_ij[:, 0]
        dv_dy = du_ij[:, 3]

        div_u = du_dx + dv_dy

        if interior_mask is not None:
            m = interior_mask.view(-1).bool()
        elif data is not None:
            m = self.fluid_interior_mask(data)
        else:
            return torch.mean(div_u**2)

        if m.any():
            return torch.mean(div_u[m] ** 2)
        return div_u.sum() * 0.0

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
        # Soft wall no-slip penalty is retained even when a hard architectural constraint exists,
        # so direct-velocity and stream-function branches can be compared safely.
        mask_wall = data.mask_wall.view(-1).bool()
        if not mask_wall.any():
            return pred.sum() * 0.0
        uv_wall = pred[mask_wall, 0:2]
        return torch.mean(torch.sum(uv_wall * uv_wall, dim=1))

    def wall_shear_stress_loss(self, pred, data, props=None):
        """
        Enforces that the model's predicted WSS strictly matches the physical WSS
        derived from its own velocity gradients at the wall boundaries.
        """
        if props is None:
            props = self._get_geometric_props(data)

        wss_pred = pred[:, 4]

        mask_wall = data.mask_wall.view(-1).bool()
        if not mask_wall.any():
            return pred.sum() * 0.0

        if (not hasattr(data, "y")) or (data.y is None) or (data.y.shape[1] <= 4):
            return pred.sum() * 0.0
        # Supervise against precomputed COMSOL wall WSS target stored in label channel 4.
        wss_mag_phys = data.y[:, 4]
        wss_pred_wall = wss_pred[mask_wall]
        wss_true_wall = wss_mag_phys[mask_wall]
        loss_data = torch.nn.functional.smooth_l1_loss(wss_pred_wall, wss_true_wall, beta=0.01)

        # Couple WSS head to predicted near-wall kinematics through analytical shear-rate consistency.
        u = pred[:, 0:1]
        v = pred[:, 1:2]
        c_u = self._compute_derivatives(u, props)
        c_v = self._compute_derivatives(v, props)
        u_x, u_y = c_u[:, 0, 0], c_u[:, 1, 0]
        v_x, v_y = c_v[:, 0, 0], c_v[:, 1, 0]
        strain_sq = 2.0 * (u_x ** 2 + v_y ** 2) + (u_y + v_x) ** 2
        gamma_dot = torch.sqrt(strain_sq + 1e-8)
        if self.cfg.viscosity_model == "carreau" and pred.shape[1] > 3:
            mu_wall = pred[:, 3][mask_wall]
        else:
            mu_wall = torch.full_like(wss_pred_wall, float(self.cfg.mu_viscosity_nd_scale))
        wss_phys_wall = mu_wall * gamma_dot[mask_wall]
        loss_phys = torch.nn.functional.smooth_l1_loss(wss_pred_wall, wss_phys_wall, beta=0.01)

        return loss_data + (0.5 * loss_phys)


    def inlet_outlet_loss(self, pred, data):
        """Soft inlet/outlet alignment with COMSOL-style BCs.

        **Outlet:** COMSOL uses fixed **pressure = 0** (gauge) on the outlet; labels use the
        same ``p`` scaling as :meth:`PhysicsConfig.get_p_ref`, so we penalize ``p²`` on outlet
        nodes to match that BC.

        **Inlet:** velocity (and Carreau ``μ`` when present) vs stored BC targets on inlet nodes.
        """
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        device = pred.device # Get active device

        loss_inlet = torch.tensor(0.0, device=device)
        mask_inlet_1d = data.mask_inlet.view(-1).bool()

        if mask_inlet_1d.any():
            u_target = data.u_inlet_bc[mask_inlet_1d, 0].to(device)
            has_v_target = (
                torch.is_tensor(data.u_inlet_bc)
                and data.u_inlet_bc.dim() > 1
                and data.u_inlet_bc.shape[1] > 1
            )
            v_target = data.u_inlet_bc[mask_inlet_1d, 1].to(device) if has_v_target else None

            u_in = u[mask_inlet_1d].squeeze(-1)
            v_in = v[mask_inlet_1d].squeeze(-1)

            if v_target is not None:
                loss_inlet = torch.mean((u_in - u_target) ** 2 + (v_in - v_target) ** 2)
            else:
                loss_inlet = torch.mean((u_in - u_target) ** 2 + v_in ** 2)

            if self.cfg.viscosity_model == "carreau" and hasattr(data, 'mu_inlet_bc'):
                mu_pred = pred[:, 3]
                mu_in = mu_pred[mask_inlet_1d].squeeze(-1)
                mu_target_bc = data.mu_inlet_bc[mask_inlet_1d].squeeze(-1).to(device)

                loss_inlet += 2.0 * torch.nn.functional.smooth_l1_loss(mu_in, mu_target_bc)

        loss_outlet = torch.tensor(0.0, device=pred.device)
        mask_outlet_1d = data.mask_outlet.view(-1).bool()
        if mask_outlet_1d.any():
            p_out = p[mask_outlet_1d]
            if self.pressure_bc_mode == "pointwise":
                loss_outlet = torch.mean(p_out ** 2)
            elif self.pressure_bc_mode == "mean_var":
                loss_outlet = (torch.mean(p_out) ** 2) + (0.1 * torch.var(p_out, unbiased=False))
            else:
                loss_outlet = torch.mean(p_out) ** 2

        return loss_inlet + loss_outlet