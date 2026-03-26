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

        # --- Species Re-dimensionalization Scales ---
        self.species_scales = torch.tensor([
            self.cfg.c_RP0, self.cfg.c_RP0, self.cfg.APRcrit, self.cfg.APScrit,
            self.cfg.c_pT0, self.cfg.c_pT0, self.cfg.cAT0, self.cfg.c_Fg0, self.cfg.c_Fg0,
            self.cfg.Minf, self.cfg.Minf, self.cfg.Minf
        ])

        self.D_scale = 1e4
        self.C_scale = 1e3

        self.kinetics = self.BiochemKinetics(self.cfg, self.C_scale)

        # Diffusion coefficients mapped directly from COMSOL config with explicit PINN scaling
        self.D_coeff = {
            'RP': self.cfg.D_RP * self.D_scale,
            'AP': self.cfg.D_AP * self.D_scale,
            'APR': self.cfg.D_APR * self.D_scale,
            'APS': self.cfg.D_APS * self.D_scale,
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

            # --- Soft-Logic Temperature Parameters (ML Hyperparameters) ---
            self.T_omega = 0.05  # Chemical activation threshold
            self.T_shear = 500.0  # Mechanical shear threshold
            self.T_grad = 50.0  # Spatial shear gradient threshold
            self.T_low_shear = 5.0  # Low shear stagnation threshold
            self.T_scale = 1.0  # Dynamic temperature scalar

            # --- COMSOL Biochemical Parameters (Mapped & Scaled) ---
            self.APScrit = self.cfg.APScrit * self.C_scale
            self.APRcrit = self.cfg.APRcrit * self.C_scale
            self.Tcrit = self.cfg.Tcrit * self.C_scale
            self.t_act = self.cfg.t_act
            self.shear_crit = self.cfg.shear_crit

            # Fibrin reaction parameters
            self.kfi = self.cfg.kfi
            self.kmfi = self.cfg.kmfi * self.C_scale

        def _soft_step(self, x, threshold, temperature, reverse=False):
            """Smooth approximation of Heaviside step function."""
            sign = -1.0 if reverse else 1.0
            # Apply T_scale to the temperature
            return torch.sigmoid(sign * (x - threshold) / (temperature * self.T_scale))

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

        def compute_fibrin_kinetics(self, T, FG):
            """
            Computes source/sink terms for Fibrinogen (FG) and Fibrin (FI)
            Formula: kfi * T * FG / (kmfi + FG)
            """
            eps = 1e-8
            reaction_rate = (self.kfi * T * FG) / (self.kmfi + FG + eps)

            R_FG = -reaction_rate
            R_FI = reaction_rate

            return R_FG, R_FI

        def compute_gamma(self, T, AT):
            """
            Analytic 3: Thrombin inhibition by Antithrombin/Heparin complex.
            Gamma = (k_1t * c_H * AT) / (K_at * K_T + T * K_at + AT * T)
            """
            numerator = self.cfg.k_1t * self.cfg.c_H * AT
            denominator = (self.cfg.K_at * self.cfg.K_T) + (T * self.cfg.K_at) + (AT * T) + 1e-8
            return numerator / denominator

        def compute_species_reactions(self, species_dict, shear_rate):
            """Computes net reaction source/sink terms for all 8 species using config params."""
            RP = species_dict['RP']
            AP = species_dict['AP']
            APR = species_dict['APR']
            APS = species_dict['APS']
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

            Gamma_inhibit = self.compute_gamma(T, AT)
            R_T = (phi_at * AP + phi_rt * RP) - (Gamma_inhibit * T)
            R_AT = - (Gamma_inhibit * T)

            R_FG, R_FI = self.compute_fibrin_kinetics(T, FG)

            return {
                'RP': R_RP, 'AP': R_AP, 'APR': R_APR, 'APS': R_APS,
                'T': R_T, 'AT': R_AT, 'FG': R_FG, 'FI': R_FI
            }

    def compute_dual_viscosity_penalty(self, M_wall, FI_field, spatial_props, delta=1e-3):
        """
        Computes the Pseudo-Huber regularization loss for the spatial gradients
        of the dual viscosity field. Stabilizes PINN training.
        """
        # FIX: Use the dynamically assigned config limits, not hardcoded 7000s
        max_ratio = self.cfg.mu_ratio_max

        # mu1 maxes out at (max_ratio - 1.0) so the base fluid remains 1.0x
        mu1_mat = self.kinetics._soft_step(M_wall, 2e7, 7e6) * (max_ratio - 1.0) + 1.0
        mu2_fi = self.kinetics._soft_step(FI_field, 0.6, 0.01) * max_ratio

        mu_total = mu1_mat + mu2_fi

        dmu_dx = self.core._compute_derivatives(mu_total.unsqueeze(1), spatial_props)
        grad_mu_sq = torch.sum(dmu_dx ** 2, dim=-1)

        pseudo_huber_loss = torch.mean((delta ** 2) * (torch.sqrt(1 + grad_mu_sq / (delta ** 2)) - 1))
        return pseudo_huber_loss

    def biochem_adr_residual(self, species_preds, velocity_field, spatial_props):
        """Computes Advection-Diffusion-Reaction (L_ADR) residuals."""
        u, v = velocity_field[..., 0], velocity_field[..., 1]
        shear_rate = self._compute_shear_rate(u, v, spatial_props)

        # --- Shear-Enhanced Diffusion (Keller Effect) ---
        # Ds = 0.18 * dRBC^2 * shear_rate
        D_s = 0.18 * (self.cfg.d_RBC ** 2) * shear_rate

        # Split species based on kinetic timescale stiffness
        fast_keys = ['RP', 'AP', 'APR', 'APS', 'T']
        slow_keys = ['AT', 'FG', 'FI']

        adr_losses_fast = 0.0
        adr_losses_slow = 0.0

        u_ref = spatial_props['u_ref'].to(species_preds.device)
        d_bar = spatial_props['d_bar'].to(species_preds.device)

        u_raw = u * u_ref
        v_raw = v * u_ref

        keys = ['RP', 'AP', 'APR', 'APS', 'PT', 'T', 'AT', 'FG', 'FI']
        # --- Reverse log1p and re-dimensionalize ---
        # Move scales to the correct device automatically
        scales = self.species_scales.to(species_preds.device)

        # Expand clamp max to 80.0 to allow massive physical spikes without float32 overflow
        species_preds_safe = torch.clamp(species_preds, min=-10.0, max=20.0)
        # expm1 reverses log1p: (e^y - 1)
        nd_species_preds = torch.expm1(species_preds_safe)
        linear_species_preds = nd_species_preds * scales[:9]

        # Build the dictionary using the linear values
        species_dict = {keys[i]: linear_species_preds[..., i] for i in range(len(keys))}

        reaction_terms = self.kinetics.compute_species_reactions(species_dict, shear_rate)
        keller_species = ['RP', 'AP', 'PT', 'T', 'AT']

        for key in fast_keys + slow_keys:
            C = species_dict[key]

            # Add Keller diffusion dynamically
            base_D = self.D_coeff[key]
            D = base_D + D_s if key in keller_species else base_D

            R = reaction_terms[key]
            grad_C = self.core._compute_derivatives(C.unsqueeze(1), spatial_props)

            # Re-dimensionalize Spatial Derivatives (1st Order)
            dC_dx = grad_C[:, 0, 0] / d_bar
            dC_dy = grad_C[:, 1, 0] / d_bar
            advection = u_raw * dC_dx + v_raw * dC_dy

            # Re-dimensionalize Spatial Derivatives (2nd Order)
            dC_dxx = grad_C[:, 2, 0] / (d_bar ** 2)
            dC_dyy = grad_C[:, 4, 0] / (d_bar ** 2)
            diffusion = D * (dC_dxx + dC_dyy)

            residual = advection - diffusion - R
            scale_idx = keys.index(key)
            scale_c = scales[scale_idx]

            residual_nd = residual / (scale_c + 1e-8)
            loss_c = torch.mean(residual_nd ** 2)
            if key in fast_keys:
                adr_losses_fast += loss_c
            else:
                adr_losses_slow += loss_c

        return adr_losses_fast, adr_losses_slow

    def biochem_wall_residual(self, biochem_preds, wall_preds, velocity_field, spatial_props, mask_wall):
        """
        Enforces Surface Platelet Adhesion and Activation Kinetics at the boundary.
        """
        if not mask_wall.any():
            return torch.tensor(0.0, device=biochem_preds.device)

        # --- Reverse log1p and re-dimensionalize ---
        scales = self.species_scales.to(biochem_preds.device)

        # FIX: Expand clamp max to 80.0
        biochem_preds_safe = torch.clamp(biochem_preds, min=-10.0, max=20.0)

        nd_biochem_preds = torch.expm1(biochem_preds_safe)
        linear_biochem_preds = nd_biochem_preds * scales[:9]  # Only bulk species passed here

        # 1. Extract Bulk Species at the Wall (using the linear values)
        RP_wall = linear_biochem_preds[mask_wall, 0]
        AP_wall = linear_biochem_preds[mask_wall, 1]

        # We need local agonists to compute surface activation (k_pa)
        APR_wall = linear_biochem_preds[mask_wall, 2]
        APS_wall = linear_biochem_preds[mask_wall, 3]
        T_wall = linear_biochem_preds[mask_wall, 5]

        # 2. Extract Surface Species
        M = wall_preds[mask_wall, 0] * self.cfg.Minf
        Mas = wall_preds[mask_wall, 1] * self.cfg.Minf
        Mat = wall_preds[mask_wall, 2] * self.cfg.Minf

        # 3. Compute Available Binding Sites (Saturation constraint)
        M_tot = M + Mas + Mat
        Minf = self.cfg.Minf

        # 4. Compute Local Surface Activation & Spatial Gradients
        global_shear = self._compute_shear_rate(velocity_field[..., 0], velocity_field[..., 1], spatial_props)
        shear_wall = global_shear[mask_wall]

        # Calculate spatial derivative of the shear rate (dshear/dx)
        c_shear = self.core._compute_derivatives(global_shear.unsqueeze(1), spatial_props)
        d_bar = spatial_props['d_bar'].to(biochem_preds.device)
        dshear_dx_global = c_shear[:, 0, 0] / d_bar
        dshear_dx_wall = dshear_dx_global[mask_wall]

        dshear_abs = torch.abs(dshear_dx_wall) + 1e-6
        availability = torch.clamp(1.0 - (M_tot / Minf), min=1e-8, max=1.0)

        # COMSOL Pathological Conditionals (Soft-Logic)
        # Condition 1: High negative spatial shear gradient (Separation zone / Stagnation point)
        # if d(spf.sr, x) < sgt
        is_separation = self.kinetics._soft_step(dshear_dx_wall, self.cfg.sgt, self.kinetics.T_grad, reverse=True)

        # Condition 2: Absolute low shear (Recirculation zone)
        # if spf.sr < lss
        is_low_shear = self.kinetics._soft_step(shear_wall, self.cfg.lss, self.kinetics.T_low_shear, reverse=True)

        omega_wall = self.kinetics.compute_omega(APR_wall, APS_wall, T_wall)
        k_pa_wall = self.kinetics.compute_k_pa(omega_wall, shear_wall)

        # 5. Steady-State Residual Equations (Matching COMSOL Inward Flux Rules)

        # Pathological adhesion rates using the safe dshear_abs variable
        # (L/gamma_m) * abs(d(spf.sr, x)) * Sat(M) * k_rs * RP
        pathological_RP_adhesion = is_separation * (
                    self.cfg.L_char / self.cfg.gamma_m) * dshear_abs * availability * self.cfg.k_rs * RP_wall
        low_shear_RP_adhesion = is_low_shear * availability * self.cfg.k_rs * RP_wall

        pathological_AP_adhesion = is_separation * (
                    self.cfg.L_char / self.cfg.gamma_m) * dshear_abs * availability * self.cfg.k_as * AP_wall
        low_shear_AP_adhesion = is_low_shear * availability * self.cfg.k_as * AP_wall

        # AP sticking to previously deposited Mas
        pathological_Mas_adhesion = is_separation * (self.cfg.L_char / self.cfg.gamma_m) * dshear_abs * (
                Mas / Minf) * self.cfg.k_aa * AP_wall
        low_shear_Mas_adhesion = is_low_shear * (Mas / Minf) * self.cfg.k_aa * AP_wall

        # Residual M: Deposition minus activation
        R_M = (pathological_RP_adhesion + low_shear_RP_adhesion) - (k_pa_wall * M)

        # Residual Mas: Direct deposition of AP
        R_Mas = (pathological_AP_adhesion + low_shear_AP_adhesion) + (
                    pathological_Mas_adhesion + low_shear_Mas_adhesion)

        # Residual Mat: Activation of resting wall platelets
        R_Mat = k_pa_wall * M

        # --- Non-Dimensionalize the Wall Residuals ---
        R_M_nd = R_M / Minf
        R_Mas_nd = R_Mas / Minf
        R_Mat_nd = R_Mat / Minf

        # Calculate MSE loss using the normalized residuals
        loss_wall = torch.mean(R_M_nd ** 2 + R_Mas_nd ** 2 + R_Mat_nd ** 2)

        return loss_wall

    def _compute_shear_rate(self, u, v, spatial_props):
        """Helper to extract scalar shear rate magnitude from velocity fields."""
        c_u = self.core._compute_derivatives(u.unsqueeze(1), spatial_props)
        c_v = self.core._compute_derivatives(v.unsqueeze(1), spatial_props)

        du_dx, du_dy = c_u[:, 0, 0], c_u[:, 1, 0]
        dv_dx, dv_dy = c_v[:, 0, 0], c_v[:, 1, 0]

        gamma_dot = torch.sqrt(2 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2 + 1e-8)
        return gamma_dot