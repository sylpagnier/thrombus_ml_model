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
        self.kinetics = self.BiochemKinetics(self.cfg)

        # Diffusion coefficients mapping (scaled dynamically in DEQ if needed)
        # Sourced from COMSOL Parameters
        self.D_coeff = {
            'RP': 1.58e-9,
            'AP': 1.58e-9,
            'APR': 2.57e-6,
            'APS': 2.14e-6,
            'T': 4.16e-7,
            'AT': 3.49e-7,
            'FG': 3.10e-7,
            'FI': 2.47e-7
        }

    class BiochemKinetics:
        """
        Differentiable implementation of the COMSOL biochemical kinetic rate equations.
        Replaces rigid step functions with temperature-scaled sigmoids to ensure smooth gradients.
        """

        def __init__(self, cfg):
            self.cfg = cfg

            # --- Soft-Logic Temperature Parameters ---
            self.T_omega = 0.05  # Chemical activation threshold
            self.T_shear = 500.0  # Mechanical shear threshold
            self.T_grad = 50.0  # Spatial shear gradient threshold
            self.T_low_shear = 5.0  # Low shear stagnation threshold

            # --- COMSOL Biochemical Parameters ---
            self.APScrit = 0.6
            self.APRcrit = 2.0
            self.Tcrit = 0.0005
            self.t_act = 1.0
            self.shear_crit = 10000.0

            # Fibrin reaction parameters
            self.kfi = 59.0  # Reaction rate fibrinogen [1/s]
            self.kmfi = 3.16  # Rate constant fibrin reaction [uM]

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
            # if(Omega<500, (Omega/t_act)*Act_step(Omega), 500)
            chem_active = self._soft_step(omega, 1.0, self.T_omega)
            cap_mask = self._soft_step(omega, 500.0, self.T_omega, reverse=True)
            kpa_chem = cap_mask * (omega / self.t_act) * chem_active + (1.0 - cap_mask) * 500.0

            # kpa_mech (Analytic 6)
            # if(spf.sr>shear_crit, spf.sr/shear_crit, 0)
            mech_active = self._soft_step(shear_rate, self.shear_crit, self.T_shear)
            kpa_mech = mech_active * (shear_rate / self.shear_crit)

            return kpa_chem + kpa_mech

        def compute_fibrin_kinetics(self, T, FG):
            """
            Computes source/sink terms for Fibrinogen (FG) and Fibrin (FI)
            Formula: kfi * T * FG / (kmfi + FG)
            """
            # Add epsilon to denominator for numerical stability in PINN
            eps = 1e-8
            reaction_rate = (self.kfi * T * FG) / (self.kmfi + FG + eps)

            # Fibrinogen is consumed (Sink), Fibrin is created (Source)
            R_FG = -reaction_rate
            R_FI = reaction_rate

            return R_FG, R_FI

        def compute_species_reactions(self, species_dict, shear_rate):
            """Computes net reaction source/sink terms for all 8 species."""
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

            # Platelet Activation (RP -> AP)
            R_RP = -k_pa * RP
            R_AP = k_pa * RP

            # Agonist Release (APR, APS) - simplified proportional release
            lambda_apr = 2.4e-8
            s_t_aps = 9.5e-12
            R_APR = lambda_apr * R_AP
            R_APS = s_t_aps * R_AP - 0.0161 * APS  # Includes inactivation k_i

            # Thrombin & Antithrombin
            # Note: Gamma analytic and specific phi_at/phi_rt constants applied
            phi_at = 3.69e-9
            phi_rt = 6.5e-10
            k_1t = 13.33
            R_T = (phi_at * AP + phi_rt * RP) - (k_1t * AT * T)  # simplified interaction
            R_AT = - (k_1t * AT * T)

            # Fibrin Cascade
            R_FG, R_FI = self.compute_fibrin_kinetics(T, FG)

            return {
                'RP': R_RP, 'AP': R_AP, 'APR': R_APR, 'APS': R_APS,
                'T': R_T, 'AT': R_AT, 'FG': R_FG, 'FI': R_FI
            }

    def compute_dual_viscosity_penalty(self, M_wall, FI_field, spatial_props, delta=1e-3):
        """
        Computes the Pseudo-Huber regularization loss for the spatial gradients
        of the dual viscosity field (mu_b * (mu1(Mat) + mu2(FI))).
        Helps stabilize PINN training against the 0->80 step function cliffs.
        """
        # Soft-step models of mu1 and mu2 derived from COMSOL step parameters
        mu1_mat = self.kinetics._soft_step(M_wall, 2e7, 7e6) * 79.0 + 1.0
        mu2_fi = self.kinetics._soft_step(FI_field, 0.6, 0.01) * 80.0

        # Combine viscosities
        mu_total = mu1_mat + mu2_fi

        # Compute spatial gradient of the total viscosity field
        dmu_dx = self.core._compute_derivatives(mu_total.unsqueeze(1), spatial_props)
        grad_mu_sq = torch.sum(dmu_dx ** 2, dim=-1)

        # Pseudo-Huber Loss Formulation: L = delta^2 * (sqrt(1 + (grad/delta)^2) - 1)
        # Prevents extreme gradient explosion where the step functions jump
        pseudo_huber_loss = torch.mean((delta ** 2) * (torch.sqrt(1 + grad_mu_sq / (delta ** 2)) - 1))

        return pseudo_huber_loss

    def compute_adr_loss(self, species_preds, velocity_field, spatial_props):
        """
        Computes the Advection-Diffusion-Reaction (L_ADR) residuals for ALL 8 species.
        Species vector index mapping:
        0: RP, 1: AP, 2: APR, 3: APS, 4: T, 5: AT, 6: FG, 7: FI
        """
        u, v = velocity_field[..., 0], velocity_field[..., 1]
        shear_rate = self._compute_shear_rate(u, v, spatial_props)

        keys = ['RP', 'AP', 'APR', 'APS', 'T', 'AT', 'FG', 'FI']
        species_dict = {keys[i]: species_preds[..., i] for i in range(8)}

        # 1. Compute Reaction Terms (Source/Sinks)
        reaction_terms = self.kinetics.compute_species_reactions(species_dict, shear_rate)

        adr_losses = {}
        total_adr_loss = 0.0

        # 2. Compute Advection and Diffusion for each species
        for i, key in enumerate(keys):
            C = species_dict[key]
            D = self.D_coeff[key]
            R = reaction_terms[key]

            # First derivatives for Advection
            grad_C = self.core._compute_derivatives(C.unsqueeze(1), spatial_props)
            dC_dx, dC_dy = grad_C[..., 0], grad_C[..., 1]
            advection = u * dC_dx + v * dC_dy

            # Second derivatives for Diffusion
            grad_C_x = self.core._compute_derivatives(dC_dx.unsqueeze(1), spatial_props)[..., 0]
            grad_C_y = self.core._compute_derivatives(dC_dy.unsqueeze(1), spatial_props)[..., 1]
            diffusion = D * (grad_C_x + grad_C_y)

            # ADR Residual: Advection - Diffusion - Reaction = 0 (assuming steady-state)
            # If time-dependent, add dC_dt term here.
            residual = advection - diffusion - R

            # Mean Squared Error of the residual
            loss_c = torch.mean(residual ** 2)
            adr_losses[key] = loss_c
            total_adr_loss += loss_c

        return total_adr_loss, adr_losses

    def _compute_shear_rate(self, u, v, spatial_props):
        """Helper to extract scalar shear rate magnitude from velocity fields."""
        c_u = self.core._compute_derivatives(u.unsqueeze(1), spatial_props)
        c_v = self.core._compute_derivatives(v.unsqueeze(1), spatial_props)

        du_dx, du_dy = c_u[..., 0], c_u[..., 1]
        dv_dx, dv_dy = c_v[..., 0], c_v[..., 1]

        gamma_dot = torch.sqrt(2 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2 + 1e-8)
        return gamma_dot