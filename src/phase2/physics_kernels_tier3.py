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
        self.core = core_physics_kernels  # Reference to baseline PhysicsKernels for _compute_derivatives

        # --- PINN Normalization Scales ---
        # To avoid vanishing gradients, inputs are scaled up from standard SI units.
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
            return torch.sigmoid(sign * (x - threshold) / temperature)

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

            # 1. Platelet Activation (RP -> AP)
            # RP is consumed, AP is produced at rate k_pa
            R_RP = -k_pa * RP
            R_AP = k_pa * RP

            # 2. Agonist Release (APR, APS)
            scale_release = 1e9
            lambda_apr = self.cfg.lambda_adp * scale_release
            s_t_aps = self.cfg.s_t * scale_release

            # DIMENSIONAL FIX: ADP is a burst release (depends on activation RATE R_AP)
            R_APR = lambda_apr * R_AP

            # DIMENSIONAL FIX: TxA2 is continuously synthesized (depends on activated CONCENTRATION AP)
            R_APS = (s_t_aps * AP) - (self.cfg.k_i * APS)

            # 3. Thrombin & Antithrombin
            scale_phi = 1e-3
            phi_at = self.cfg.phi_at * scale_phi
            phi_rt = self.cfg.phi_rt * scale_phi

            R_T = (phi_at * AP + phi_rt * RP) - (self.cfg.k_1t * AT * T)
            R_AT = - (self.cfg.k_1t * AT * T)

            # 4. Fibrin Cascade
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
        mu1_mat = self.kinetics._soft_step(M_wall, 2e7, 7e6) * 79.0 + 1.0
        mu2_fi = self.kinetics._soft_step(FI_field, 0.6, 0.01) * 80.0
        mu_total = mu1_mat + mu2_fi

        dmu_dx = self.core._compute_derivatives(mu_total.unsqueeze(1), spatial_props)
        grad_mu_sq = torch.sum(dmu_dx ** 2, dim=-1)

        pseudo_huber_loss = torch.mean((delta ** 2) * (torch.sqrt(1 + grad_mu_sq / (delta ** 2)) - 1))
        return pseudo_huber_loss

    def biochem_adr_residual(self, species_preds, velocity_field, spatial_props):
        """Computes Advection-Diffusion-Reaction (L_ADR) residuals."""
        u, v = velocity_field[..., 0], velocity_field[..., 1]
        shear_rate = self._compute_shear_rate(u, v, spatial_props)

        # FIX: Explicitly include 'PT' at index 4 to prevent downstream array shifting!
        keys = ['RP', 'AP', 'APR', 'APS', 'PT', 'T', 'AT', 'FG', 'FI']
        species_dict = {keys[i]: species_preds[..., i] for i in range(len(keys))}

        reaction_terms = self.kinetics.compute_species_reactions(species_dict, shear_rate)

        adr_losses = {}
        total_adr_loss = 0.0

        # Only compute ADR for the 8 active species (ignoring the passive PT)
        active_keys = ['RP', 'AP', 'APR', 'APS', 'T', 'AT', 'FG', 'FI']

        for key in active_keys:
            C = species_dict[key]
            D = self.D_coeff[key]
            R = reaction_terms[key]

            grad_C = self.core._compute_derivatives(C.unsqueeze(1), spatial_props)

            # Extract 1st derivatives (Index 0 = dx, Index 1 = dy)
            dC_dx, dC_dy = grad_C[:, 0, 0], grad_C[:, 1, 0]
            advection = u * dC_dx + v * dC_dy

            # Extract 2nd derivatives directly (Index 2 = dxx, Index 4 = dyy)
            dC_dxx, dC_dyy = grad_C[:, 2, 0], grad_C[:, 4, 0]
            diffusion = D * (dC_dxx + dC_dyy)

            residual = advection - diffusion - R
            loss_c = torch.mean(residual ** 2)
            adr_losses[key] = loss_c
            total_adr_loss += loss_c

        return total_adr_loss

    def biochem_wall_residual(self, biochem_preds, wall_preds, velocity_field, spatial_props, mask_wall):
        """
        Enforces Surface Platelet Adhesion and Activation Kinetics at the boundary.
        Calculates the steady-state residual for M, Mas, and Mat.
        """
        if not mask_wall.any():
            return torch.tensor(0.0, device=biochem_preds.device)

        # 1. Extract Bulk Species at the Wall
        RP_wall = biochem_preds[mask_wall, 0]
        AP_wall = biochem_preds[mask_wall, 1]

        # We need local agonists to compute surface activation (k_pa)
        APR_wall = biochem_preds[mask_wall, 2]
        APS_wall = biochem_preds[mask_wall, 3]
        T_wall = biochem_preds[mask_wall, 5]  # T is index 5 (after PT)

        # 2. Extract Surface Species
        M = wall_preds[mask_wall, 0]
        Mas = wall_preds[mask_wall, 1]
        Mat = wall_preds[mask_wall, 2]

        # 3. Compute Available Binding Sites (Saturation constraint)
        M_tot = M + Mas + Mat
        Minf = self.cfg.Minf

        # Soft-clamp to prevent negative availability during early training
        availability = torch.clamp(1.0 - (M_tot / Minf), min=0.0, max=1.0)

        # 4. Compute Local Surface Activation Rate (k_pa)
        u_wall = velocity_field[mask_wall, 0]
        v_wall = velocity_field[mask_wall, 1]
        # Shear rate requires the full field to compute gradients, so we compute globally and mask
        global_shear = self._compute_shear_rate(velocity_field[..., 0], velocity_field[..., 1], spatial_props)
        shear_wall = global_shear[mask_wall]

        omega_wall = self.kinetics.compute_omega(APR_wall, APS_wall, T_wall)
        k_pa_wall = self.kinetics.compute_k_pa(omega_wall, shear_wall)

        # 5. Steady-State Residual Equations (Target = 0)
        # Residual M: Deposition of RP minus activation of M into Mat
        R_M = (self.cfg.k_rs * RP_wall * availability) - (k_pa_wall * M)

        # Residual Mas: Direct deposition of AP
        R_Mas = self.cfg.k_as * AP_wall * availability

        # Residual Mat: Activation of previously resting wall platelets
        R_Mat = k_pa_wall * M

        # Calculate MSE loss for the residuals
        loss_wall = torch.mean(R_M ** 2 + R_Mas ** 2 + R_Mat ** 2)

        return loss_wall

    def _compute_shear_rate(self, u, v, spatial_props):
        """Helper to extract scalar shear rate magnitude from velocity fields."""
        c_u = self.core._compute_derivatives(u.unsqueeze(1), spatial_props)
        c_v = self.core._compute_derivatives(v.unsqueeze(1), spatial_props)

        du_dx, du_dy = c_u[:, 0, 0], c_u[:, 1, 0]
        dv_dx, dv_dy = c_v[:, 0, 0], c_v[:, 1, 0]

        gamma_dot = torch.sqrt(2 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2 + 1e-8)
        return gamma_dot