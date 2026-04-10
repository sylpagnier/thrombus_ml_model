import torch
import torch.nn as nn
import torch.nn.functional as F


class BiochemPhysicsKernels:
    """
    Tier 3 Biochemical Physics Kernels for Eulerian Thrombosis Modeling.
    Translates COMSOL Phase 2 multiphysics equations into differentiable DEQ operations.
    Includes full Fibrin/Fibrinogen coagulation cascade and Dual Viscosity Pseudo-Huber Regularization.
    """

    def __init__(self, biochem_cfg, core_physics_kernels):
        self.cfg = biochem_cfg
        self.core = core_physics_kernels

        # --- USE CENTRALIZED SCALES ---
        self.species_scales = self.cfg.get_species_scales()
        self.D_scale = self.cfg.d_scale
        self.C_scale = self.cfg.bulk_scale

        self.kinetics = self.BiochemKinetics(self.cfg, self.C_scale)
        self.adr_norm_scales = self.cfg.get_adr_norm_scales()

        self.D_coeff = {
            'RP': self.cfg.D_RP * self.D_scale,
            'AP': self.cfg.D_AP * self.D_scale,
            'APR': self.cfg.D_APR * self.D_scale,
            'APS': self.cfg.D_APS * self.D_scale,
            'PT': self.cfg.D_PT * self.D_scale,
            'T': self.cfg.D_T * self.D_scale,
            'AT': self.cfg.D_AT * self.D_scale,
            'FG': self.cfg.D_FG * self.D_scale,
            'FI': self.cfg.D_FI * self.D_scale
        }

    class BiochemKinetics:
        """
        Differentiable implementation of the COMSOL biochemical kinetic rate equations.
        Replaces rigid step functions with temperature-scaled sigmoids to ensure smooth gradients.
        """

        def __init__(self, cfg, C_scale):
            self.cfg = cfg
            self.C_scale = C_scale

            self.T_omega = self.cfg.soft_step_T_omega
            self.T_shear = self.cfg.soft_step_T_shear
            self.T_grad = self.cfg.soft_step_T_grad
            self.T_low_shear = self.cfg.soft_step_T_low_shear
            self.T_scale = self.cfg.soft_step_T_scale

            # --- COMSOL Biochemical Parameters (Mapped & Scaled) ---
            self.APScrit = self.cfg.APScrit * self.C_scale
            self.APRcrit = self.cfg.APRcrit * self.C_scale
            self.Tcrit = self.cfg.Tcrit * self.C_scale
            self.t_act = self.cfg.t_act
            self.shear_crit = self.cfg.shear_crit

            # Fibrin reaction parameters
            self.kfi = self.cfg.kfi
            self.kmfi = self.cfg.kmfi * self.C_scale

            # Thrombin inhibition (Gamma) concentration constants in working concentration space
            self.c_H = self.cfg.c_H * self.C_scale
            self.K_at = self.cfg.K_at * self.C_scale
            self.K_T = self.cfg.K_T * self.C_scale

        def _soft_step(self, x, threshold, temperature, reverse=False):
            """Smooth approximation of Heaviside step function using STE."""
            sign = -1.0 if reverse else 1.0
            scaled_temp = temperature * self.T_scale
            # Implement Straight-Through Estimator to bypass vanishing gradients
            return SoftStepSTE.apply(x, threshold, scaled_temp, sign)

        def compute_omega(self, APR, APS, T):
            """Analytic 1: Chemical activation function."""
            return (APS / self.APScrit) + (APR / self.APRcrit) + (T / self.Tcrit)

        def compute_k_pa(self, omega, shear_rate):
            """
            Analytic 7: Total Platelet Activation Rate (kpa_chem + kpa_mech)
            Uses soft-logic for differentiable conditionals.
            """
            # kpa_chem (Analytic 2)
            chem_active = self._soft_step(omega, 1.0, self.T_omega)
            cap_mask = self._soft_step(omega, 500.0, self.T_omega, reverse=True)
            kpa_chem = cap_mask * (omega / self.t_act) * chem_active + (1.0 - cap_mask) * 500.0

            # kpa_mech (Analytic 6)
            mech_active = self._soft_step(shear_rate, self.shear_crit, self.T_shear)
            kpa_mech = mech_active * (shear_rate / self.shear_crit)

            return kpa_chem + kpa_mech

        # Update signature to include FI concentration
        def compute_fibrin_kinetics(self, T, FG, FI):
            eps = 1e-8
            reaction_rate = (self.kfi * T * FG) / (self.kmfi + FG + eps)

            c_max = self.cfg.fi_reaction_saturation_si * self.C_scale + eps
            raw_sat = 1.0 - FI / c_max
            # Smoothly approximate clamp(raw_sat, 0, 1) to avoid derivative kinks near saturation limits.
            k_steep = 10.0
            saturation_term = 0.5 * (torch.tanh(k_steep * (raw_sat - 0.5)) + 1.0)

            R_FI = reaction_rate * saturation_term
            R_FG = -reaction_rate * saturation_term  # Conservation of species mass
            return R_FG, R_FI

        def compute_gamma(self, T, AT):
            """
            Analytic 3: Thrombin inhibition by Antithrombin/Heparin complex.
            Gamma = (k_1t * c_H * AT) / (K_at * K_T + T * K_at + AT * T)
            """
            numerator = self.cfg.k_1t * self.c_H * AT
            denominator = (self.K_at * self.K_T) + (T * self.K_at) + (AT * T) + 1e-8
            return numerator / denominator

        def compute_species_reactions(self, species_dict, shear_rate):
            """Computes net reaction source/sink terms for all bulk species using config params."""
            RP = species_dict['RP']
            AP = species_dict['AP']
            APR = species_dict['APR']
            APS = species_dict['APS']
            PT = species_dict['PT']
            T = species_dict['T']
            AT = species_dict['AT']
            FG = species_dict['FG']
            FI = species_dict['FI']

            omega = self.compute_omega(APR, APS, T)
            k_pa = self.compute_k_pa(omega, shear_rate)

            # Agonist Release (APR, APS)
            lambda_apr = self.cfg.lambda_adp
            s_t_aps = self.cfg.s_t

            R_RP = -k_pa * RP
            R_AP = k_pa * RP
            R_APR = lambda_apr * R_AP
            R_APS = (s_t_aps * AP) - (self.cfg.k_i * APS)

            # Thrombin & Antithrombin
            phi_at = self.cfg.phi_at * self.cfg.beta
            phi_rt = self.cfg.phi_rt * self.cfg.beta

            # PT and platelet terms are in C_scale working concentration; divide once to keep rate dimensions stable.
            enzymatic_term = PT * (phi_at * AP + phi_rt * RP) / self.C_scale
            R_PT = -enzymatic_term

            Gamma_inhibit = self.compute_gamma(T, AT)
            R_T = enzymatic_term - (Gamma_inhibit * T)
            R_AT = - (Gamma_inhibit * T)

            R_FG, R_FI = self.compute_fibrin_kinetics(T, FG, FI)
            return {
                'RP': R_RP, 'AP': R_AP, 'APR': R_APR, 'APS': R_APS,
                'PT': R_PT, 'T': R_T, 'AT': R_AT, 'FG': R_FG, 'FI': R_FI
            }

    def compute_dual_viscosity_penalty(self, M_wall, FI_field, spatial_props, data, delta=1e-3):
        """
        Computes the Pseudo-Huber regularization loss for the spatial gradients
        of the dual viscosity field. Stabilizes PINN training.
        """
        max_ratio = self.cfg.mu_ratio_max

        # mu1 maxes out at (max_ratio - 1.0) so the base fluid remains 1.0x
        mu1_mat = self.kinetics._soft_step(
            M_wall, self.cfg.viscosity_mat_crit, self.cfg.viscosity_penalty_soft_temp_mat
        ) * (max_ratio - 1.0) + 1.0
        mu2_fi = self.kinetics._soft_step(
            FI_field, self.cfg.viscosity_fi_crit, self.cfg.viscosity_penalty_soft_temp_fi
        ) * max_ratio

        mu_total = mu1_mat + mu2_fi
        if mu_total.dim() > 1:
            mu_total = mu_total.view(-1)

        # Sparse Matrix Gradient Calculation
        mu_col = mu_total.unsqueeze(1)
        dmu_dx = torch.sparse.mm(data.G_x, mu_col).squeeze(1)
        dmu_dy = torch.sparse.mm(data.G_y, mu_col).squeeze(1)

        grad_mu_sq = dmu_dx ** 2 + dmu_dy ** 2

        pseudo_huber_loss = torch.mean((delta ** 2) * (torch.sqrt(1 + grad_mu_sq / (delta ** 2)) - 1))
        return pseudo_huber_loss

    def biochem_adr_residual(self, species_preds, velocity_field, spatial_props, data, d_pred_dt=None):
        """Computes Advection-Diffusion-Reaction (L_ADR) residuals with Transient Time Derivatives."""
        u, v = velocity_field[..., 0], velocity_field[..., 1]

        # Pass `data` down into the shear computation
        shear_rate = self._compute_shear_rate(u, v, spatial_props, data)

        d_RBC_m = self.cfg.d_RBC
        D_s = 0.18 * (d_RBC_m ** 2) * shear_rate

        fast_keys = ['RP', 'AP', 'APR', 'APS', 'T']
        slow_keys = ['AT', 'FG', 'FI']

        z = species_preds.sum() * 0.0
        adr_losses_fast = z
        adr_losses_slow = z

        u_ref = spatial_props['u_ref'].to(species_preds.device)
        d_bar = spatial_props['d_bar'].to(species_preds.device)

        u_raw = u * u_ref
        v_raw = v * u_ref

        keys = ['RP', 'AP', 'APR', 'APS', 'PT', 'T', 'AT', 'FG', 'FI']
        scales = self.species_scales.to(species_preds.device)
        adr_norm = self.adr_norm_scales.to(species_preds.device)

        species_preds_safe = torch.clamp(species_preds, min=-10.0, max=8.0)
        nd_species_preds = torch.expm1(species_preds_safe)
        linear_species_preds = nd_species_preds * scales[:9]

        species_dict = {keys[i]: linear_species_preds[..., i] for i in range(len(keys))}
        reaction_terms = self.kinetics.compute_species_reactions(species_dict, shear_rate)
        keller_species = ['RP', 'AP', 'PT', 'T', 'AT']

        for key in fast_keys + slow_keys:
            scale_idx = keys.index(key)
            scale_c = scales[scale_idx]

            # Fast chemistry is treated as quasi-steady over coarse macro timesteps.
            # Only slow species receive explicit transient dC/dt supervision.
            if d_pred_dt is not None and key in slow_keys:
                exp_pred = torch.exp(species_preds_safe[:, scale_idx])
                dC_dt = scale_c * exp_pred * d_pred_dt[:, scale_idx]
            else:
                dC_dt = species_preds_safe[:, scale_idx] * 0.0

            C = species_dict[key]
            base_D = self.D_coeff[key]
            D = base_D + D_s if key in keller_species else base_D
            R = reaction_terms[key]

            # Sparse Matrix Gradient Calculation
            C_col = C.unsqueeze(1)
            dC_dx = torch.sparse.mm(data.G_x, C_col).squeeze(1) / d_bar
            dC_dy = torch.sparse.mm(data.G_y, C_col).squeeze(1) / d_bar
            advection = u_raw * dC_dx + v_raw * dC_dy

            # Full diffusion operator: div(D grad(C)) = D laplacian(C) + grad(D) dot grad(C)
            laplacian_C = torch.sparse.mm(data.Laplacian, C_col).squeeze(1) / (d_bar ** 2)
            if key in keller_species:
                D_col = D.unsqueeze(1)
                dD_dx = torch.sparse.mm(data.G_x, D_col).squeeze(1) / d_bar
                dD_dy = torch.sparse.mm(data.G_y, D_col).squeeze(1) / d_bar
                diffusion = (D * laplacian_C) + (dD_dx * dC_dx + dD_dy * dC_dy)
            else:
                diffusion = D * laplacian_C

            if key == 'FI':
                # Polymerized fibrin is a solid matrix; it does not advect or diffuse.
                advection = torch.zeros_like(advection)
                diffusion = torch.zeros_like(diffusion)

            # Include dC_dt in the residual
            residual = dC_dt + advection - diffusion - R

            # Use fixed physically-meaningful normalization for ADR residuals.
            norm_scale = adr_norm[scale_idx]
            # Reaction-aware scaling stabilizes coarse-dt supervision for stiff kinetics.
            local_stiffness = torch.abs(R) + norm_scale
            residual_nd = residual / (local_stiffness + 1e-8)
            loss_c = F.huber_loss(residual_nd, torch.zeros_like(residual_nd), delta=1.0)

            if key in fast_keys:
                adr_losses_fast += loss_c
            else:
                adr_losses_slow += loss_c

        return adr_losses_fast, adr_losses_slow

    def biochem_inlet_outlet_residual(self, biochem_preds, spatial_props, data):
        """
        Enforces Danckwerts (Dirichlet-equivalent) concentration at the inlet
        and Outflow (Zero normal gradient) conditions at the outlet.
        """
        mask_inlet = data.mask_inlet.view(-1).bool()
        mask_outlet = data.mask_outlet.view(-1).bool()

        z = biochem_preds.sum() * 0.0
        loss_inlet = z
        loss_outlet = z

        # 1. Inlet: Force predictions to match the transformed baseline concentrations
        if mask_inlet.any() and hasattr(data, 'bio_inlet_bc'):
            preds_inlet = biochem_preds[mask_inlet, 0:9]
            targs_inlet = data.bio_inlet_bc[mask_inlet, 0:9]
            loss_inlet = F.mse_loss(preds_inlet, targs_inlet)

        # 2. Outlet: zero normal gradient of physical concentration (same linear C as ADR / wall flux).
        # Use ``data.outlet_normal`` when present (Gmsh outlet lines in mesh_to_graph). Slots x[:,3:5] are
        # wall-distance / wall-segment features, not the outlet face normal — wrong n gave huge erroneous
        # dC/dn on synthetic vessels; raw MSE could overflow float32 to inf.
        if mask_outlet.any():
            if hasattr(data, "outlet_normal") and data.outlet_normal is not None:
                on = data.outlet_normal.to(device=biochem_preds.device, dtype=biochem_preds.dtype)
                nx = on[mask_outlet, 0]
                ny = on[mask_outlet, 1]
                mag = torch.sqrt(nx * nx + ny * ny + 1e-12)
                weak = mag < 1e-5
                if weak.any():
                    nx_fb = data.x[mask_outlet, 3]
                    ny_fb = data.x[mask_outlet, 4]
                    nx = torch.where(weak, nx_fb, nx)
                    ny = torch.where(weak, ny_fb, ny)
            else:
                nx = data.x[mask_outlet, 3]
                ny = data.x[mask_outlet, 4]

            nmag = torch.sqrt(nx * nx + ny * ny + 1e-12)
            nx = nx / nmag
            ny = ny / nmag

            scales = self.species_scales.to(biochem_preds.device)
            biochem_safe = torch.clamp(biochem_preds, min=-10.0, max=8.0)
            linear_bulk = torch.expm1(biochem_safe) * scales[:9]

            d_bar = spatial_props['d_bar'].to(device=biochem_preds.device, dtype=biochem_preds.dtype)

            flux_scale = self.adr_norm_scales[:9].to(biochem_preds.device).mean().clamp(min=1e-12)
            loss_outlet_acc = torch.zeros((), device=biochem_preds.device, dtype=biochem_preds.dtype)
            for i in range(9):
                C_col = linear_bulk[:, i].unsqueeze(1)
                dC_dx = torch.sparse.mm(data.G_x, C_col).squeeze(1) / d_bar
                dC_dy = torch.sparse.mm(data.G_y, C_col).squeeze(1) / d_bar
                dC_dn = dC_dx[mask_outlet] * nx + dC_dy[mask_outlet] * ny
                dC_dn = torch.nan_to_num(dC_dn, nan=0.0, posinf=0.0, neginf=0.0)
                d_nd = dC_dn / flux_scale
                loss_outlet_acc = loss_outlet_acc + F.huber_loss(d_nd, torch.zeros_like(d_nd), delta=1.0)

            loss_outlet = loss_outlet_acc / 9.0

        return loss_inlet, loss_outlet

    def biochem_wall_residual(self, biochem_preds, wall_preds, velocity_field, spatial_props, data, dM_pred_dt=None):
        """Enforces Surface Platelet Adhesion with Transient Surface ODEs."""
        mask_wall = data.mask_wall.view(-1).bool()
        if not mask_wall.any():
            z = biochem_preds.sum() * 0.0
            return z, z

        scales = self.species_scales.to(biochem_preds.device)
        biochem_preds_safe = torch.clamp(biochem_preds, min=-10.0, max=8.0)
        nd_biochem_preds = torch.expm1(biochem_preds_safe)
        linear_biochem_preds = nd_biochem_preds * scales[:9]

        RP_wall = linear_biochem_preds[mask_wall, 0]
        AP_wall = linear_biochem_preds[mask_wall, 1]
        APR_wall = linear_biochem_preds[mask_wall, 2]
        APS_wall = linear_biochem_preds[mask_wall, 3]
        PT_wall = linear_biochem_preds[mask_wall, 4]
        T_wall = linear_biochem_preds[mask_wall, 5]

        wall_preds_safe = torch.clamp(wall_preds, min=-10.0, max=8.0)
        nd_wall_preds = torch.expm1(wall_preds_safe)

        # USE CENTRALIZED SCALE
        Minf_scaled = self.cfg.Minf * self.cfg.surface_scale

        M = nd_wall_preds[mask_wall, 0] * Minf_scaled
        Mas = nd_wall_preds[mask_wall, 1] * Minf_scaled
        Mat = nd_wall_preds[mask_wall, 2] * Minf_scaled

        if dM_pred_dt is not None:
            exp_wall = torch.exp(wall_preds_safe[mask_wall])
            dM_dt_phys = exp_wall[:, 0] * dM_pred_dt[mask_wall, 0] * Minf_scaled
            dMas_dt_phys = exp_wall[:, 1] * dM_pred_dt[mask_wall, 1] * Minf_scaled
            dMat_dt_phys = exp_wall[:, 2] * dM_pred_dt[mask_wall, 2] * Minf_scaled
        else:
            z = M * 0.0
            dM_dt_phys = z
            dMas_dt_phys = z
            dMat_dt_phys = z

        M_tot = M + Mas + Mat
        Minf = self.cfg.Minf * self.cfg.surface_scale

        # 4. Compute Local Surface Activation & Spatial Gradients
        global_shear = self._compute_shear_rate(velocity_field[..., 0], velocity_field[..., 1], spatial_props, data)
        shear_wall = global_shear[mask_wall]

        d_bar = spatial_props['d_bar'].to(biochem_preds.device)

        # Sparse Matrix Gradient Calculation for Shear Rate
        dshear_dx = torch.sparse.mm(data.G_x, global_shear.unsqueeze(1)).squeeze(1)
        dshear_dx_wall = (dshear_dx / d_bar)[mask_wall]
        dshear_abs = torch.abs(dshear_dx_wall) + 1e-6

        availability = torch.clamp(1.0 - (M_tot / Minf), min=1e-8, max=1.0)

        # COMSOL Pathological Conditionals (Soft-Logic)
        is_separation = self.kinetics._soft_step(dshear_dx_wall, self.cfg.sgt, self.kinetics.T_grad, reverse=True)
        is_low_shear = self.kinetics._soft_step(shear_wall, self.cfg.lss, self.kinetics.T_low_shear, reverse=True)

        omega_wall = self.kinetics.compute_omega(APR_wall, APS_wall, T_wall)
        k_pa_wall = self.kinetics.compute_k_pa(omega_wall, shear_wall)

        # SI-only convention: adhesion rates and characteristic lengths are stored in SI in BiochemConfig.
        k_rs = self.cfg.k_rs
        k_as = self.cfg.k_as
        k_aa = self.cfg.k_aa
        L_char = self.cfg.L_char

        # 5. Adhesion Rates (Matching COMSOL Inward Flux Rules)
        pathological_RP_adhesion = is_separation * (
                    L_char / self.cfg.gamma_m) * dshear_abs * availability * k_rs * RP_wall
        low_shear_RP_adhesion = is_low_shear * availability * k_rs * RP_wall

        pathological_AP_adhesion = is_separation * (
                    L_char / self.cfg.gamma_m) * dshear_abs * availability * k_as * AP_wall
        low_shear_AP_adhesion = is_low_shear * availability * k_as * AP_wall

        pathological_Mas_adhesion = is_separation * (L_char / self.cfg.gamma_m) * dshear_abs * (
                    Mas / Minf) * k_aa * AP_wall
        low_shear_Mas_adhesion = is_low_shear * (Mas / Minf) * k_aa * AP_wall

        # --- TRANSIENT SURFACE RESIDUALS (ODEs) ---
        R_M = (pathological_RP_adhesion + low_shear_RP_adhesion) - (k_pa_wall * M)
        R_Mas = (pathological_AP_adhesion + low_shear_AP_adhesion) + (
                    pathological_Mas_adhesion + low_shear_Mas_adhesion)
        R_Mat = k_pa_wall * M

        u_ref_wall = spatial_props['u_ref'].to(biochem_preds.device)[mask_wall]
        d_bar_wall = spatial_props['d_bar'].to(biochem_preds.device)[mask_wall]
        t_ref = d_bar_wall / u_ref_wall

        res_M = ((dM_dt_phys - R_M) / Minf) * t_ref
        res_Mas = ((dMas_dt_phys - R_Mas) / Minf) * t_ref
        res_Mat = ((dMat_dt_phys - R_Mat) / Minf) * t_ref

        loss_surface = (
                               F.huber_loss(res_M, torch.zeros_like(res_M), delta=1.0) +
                               F.huber_loss(res_Mas, torch.zeros_like(res_Mas), delta=1.0) +
                               F.huber_loss(res_Mat, torch.zeros_like(res_Mat), delta=1.0)
                       ) / 3.0

        # --- 6. NEUMANN BOUNDARY FLUX COUPLING (Bulk-to-Wall) ---
        J_in_RP = - (pathological_RP_adhesion + low_shear_RP_adhesion)
        J_in_AP = - (
                    pathological_AP_adhesion + low_shear_AP_adhesion + pathological_Mas_adhesion + low_shear_Mas_adhesion)
        J_in_APR = self.cfg.lambda_adp * (pathological_RP_adhesion + low_shear_RP_adhesion)
        J_in_APS = self.cfg.s_t * Mat
        J_in_PT = - self.cfg.beta * self.cfg.phi_at * Mat * PT_wall
        J_in_T = self.cfg.beta * self.cfg.phi_at * Mat * PT_wall

        flux_targets = {0: J_in_RP, 1: J_in_AP, 2: J_in_APR, 3: J_in_APS, 4: J_in_PT, 5: J_in_T}

        nx = data.x[mask_wall, 3]
        ny = data.x[mask_wall, 4]

        d_RBC_m_wall = self.cfg.d_RBC
        D_s_wall = 0.18 * (d_RBC_m_wall ** 2) * shear_wall
        keller_indices = [0, 1, 4, 5, 6]
        keys = ['RP', 'AP', 'APR', 'APS', 'PT', 'T', 'AT', 'FG', 'FI']

        loss_flux = biochem_preds.sum() * 0.0

        for idx, J_in in flux_targets.items():
            C_field = linear_biochem_preds[:, idx].unsqueeze(1)

            # Sparse Matrix Gradient Calculation
            dC_dx = torch.sparse.mm(data.G_x, C_field).squeeze(1) / d_bar
            dC_dy = torch.sparse.mm(data.G_y, C_field).squeeze(1) / d_bar

            # Dot with outward normal vector (D * grad(C) dot n)
            dC_dn_wall = dC_dx[mask_wall] * nx + dC_dy[mask_wall] * ny

            # Diffusion Coefficient at Wall
            base_D = self.D_coeff[keys[idx]]
            D_eff = base_D + D_s_wall if idx in keller_indices else base_D

            predicted_flux = -D_eff * dC_dn_wall
            flux_residual = predicted_flux - J_in

            # Use convective flux for stable normalization
            char_flux = scales[idx] * u_ref_wall
            flux_residual_nd = flux_residual / (char_flux + 1e-8)

            loss_flux += F.huber_loss(flux_residual_nd, torch.zeros_like(flux_residual_nd), delta=1.0)

        return loss_surface, loss_flux

    def _compute_shear_rate(self, u, v, spatial_props, data):
        # Sparse Matrix Gradient Calculation
        du_dx = torch.sparse.mm(data.G_x, u.unsqueeze(1)).squeeze(1)
        du_dy = torch.sparse.mm(data.G_y, u.unsqueeze(1)).squeeze(1)
        dv_dx = torch.sparse.mm(data.G_x, v.unsqueeze(1)).squeeze(1)
        dv_dy = torch.sparse.mm(data.G_y, v.unsqueeze(1)).squeeze(1)

        # This is non-dimensional
        gamma_dot_nd = torch.sqrt(2 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2 + 1e-8)

        # Redimensionalize to physical 1/s
        u_ref = spatial_props['u_ref'].to(u.device)
        d_bar = spatial_props['d_bar'].to(u.device)

        return gamma_dot_nd * (u_ref / d_bar)


class SoftStepSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, temperature, sign):
        thresh_t = torch.tensor(threshold, dtype=x.dtype, device=x.device)
        temp_t = torch.tensor(temperature, dtype=x.dtype, device=x.device)
        sign_t = torch.tensor(sign, dtype=x.dtype, device=x.device)

        ctx.save_for_backward(x, thresh_t, temp_t, sign_t)
        return (sign_t * (x - thresh_t) > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Unpack the saved tensors
        x, threshold, temperature, sign = ctx.saved_tensors

        # Backward pass: Smooth sigmoid derivative to keep gradients flowing
        sig = torch.sigmoid(sign * (x - threshold) / temperature)
        grad_x = grad_output * sign * (1.0 / temperature) * sig * (1.0 - sig)

        return grad_x, None, None, None