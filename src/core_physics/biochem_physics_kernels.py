import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BULK_SPECIES_ORDER, SPECIES_GROUPS, BulkSpecies, BiochemNodeFeat, WallSpecies
from src.utils.rheology import compute_shear_rate
from src.utils.nondim import time_ratio_global_to_convective
from src.utils.tensor_utils import as_tensor_like


class BiochemPhysicsKernels:
    """
    Biochem Biochemical Physics Kernels for Eulerian Thrombosis Modeling.
    Translates COMSOL Phase 2 multiphysics equations into differentiable DEQ operations.
    Includes full Fibrin/Fibrinogen coagulation cascade and Dual Viscosity Pseudo-Huber Regularization.
    """

    def __init__(self, biochem_cfg, core_physics_kernels):
        self.cfg = biochem_cfg
        self.core = core_physics_kernels
        self._biochem_huber_delta = max(float(self.cfg.biochem_huber_delta), 1e-8)
        self._availability_negative_slope = max(float(self.cfg.availability_negative_slope), 0.0)

        # --- USE CENTRALIZED SCALES ---
        self.species_scales = self.cfg.get_species_scales(device="cpu")
        self.D_scale = self.cfg.d_scale
        self.C_scale = self.cfg.bulk_scale

        self.kinetics = self.BiochemKinetics(self.cfg, self.C_scale)
        self.adr_norm_scales = self.cfg.get_adr_norm_scales(device="cpu")
        self._species_scales_cache = {}
        self._adr_norm_scales_cache = {}

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

    def _get_species_scales(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device.type, device.index, str(dtype))
        if key not in self._species_scales_cache:
            self._species_scales_cache[key] = self.species_scales.to(device=device, dtype=dtype)
        return self._species_scales_cache[key]

    def _get_adr_norm_scales(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device.type, device.index, str(dtype))
        if key not in self._adr_norm_scales_cache:
            self._adr_norm_scales_cache[key] = self.adr_norm_scales.to(device=device, dtype=dtype)
        return self._adr_norm_scales_cache[key]

    def set_biochem_huber_delta(self, delta: float) -> None:
        """Update shared residual Huber delta during curriculum annealing."""
        self._biochem_huber_delta = max(float(delta), 1e-8)

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
            # COMSOL reac1 (Reactions) does not apply a saturation limit for FI.
            # Use the unbounded Michaelis-Menten-like form directly to match COMSOL.
            eps = 1e-8
            reaction_rate = (self.kfi * T * FG) / (self.kmfi + FG + eps)
            R_FI = reaction_rate
            R_FG = -reaction_rate  # Conservation of species mass
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
        # FI threshold is defined in SI concentration units (COMSOL). Convert the
        # model state from log1p(nd) to SI before applying the smooth step.
        fi_scale = float(self._get_species_scales(FI_field.device, FI_field.dtype)[8].item())
        FI_si = torch.expm1(torch.clamp(FI_field, min=-10.0, max=8.0)) * fi_scale
        mu2_fi = self.kinetics._soft_step(
            FI_si, self.cfg.viscosity_fi_crit, self.cfg.viscosity_penalty_soft_temp_fi
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
        a_RBC_m = self.cfg.d_RBC / 2.0
        # COMSOL Keller diffusion uses a global constant maximum shear rate (tau_max).
        # Using local shear here would introduce spatially varying diffusion not present in COMSOL labels.
        tau_max = 2000.0
        D_s = 0.18 * (a_RBC_m ** 2) * tau_max

        fast_species = SPECIES_GROUPS["fast"]
        slow_species = SPECIES_GROUPS["slow"]

        # Independent accumulators: keep both tied to ``species_preds`` for autograd, but
        # ensure ``adr_losses_fast`` and ``adr_losses_slow`` are *separate* tensors. A
        # shared reference combined with the in-place ``+=`` below would silently make
        # both losses equal to the sum across all species (each accumulator mutates the
        # same storage), which neutralises the fast/slow Kendall weighting downstream.
        adr_losses_fast = species_preds.sum() * 0.0
        adr_losses_slow = species_preds.sum() * 0.0

        u_ref = spatial_props['u_ref'].to(device=species_preds.device, dtype=species_preds.dtype)
        d_bar = spatial_props['d_bar'].to(device=species_preds.device, dtype=species_preds.dtype)
        u_ref_safe = torch.clamp(u_ref, min=1e-8)
        d_bar_safe = torch.clamp(d_bar, min=1e-8)

        keys = BULK_SPECIES_ORDER
        bulk_n = len(keys)
        scales = self._get_species_scales(species_preds.device, species_preds.dtype)
        species_preds_safe = torch.clamp(species_preds, min=-10.0, max=8.0)
        nd_species_preds = torch.expm1(species_preds_safe)
        linear_species_preds = nd_species_preds * scales[:bulk_n]

        species_dict = {sp.name: linear_species_preds[..., sp.value] for sp in keys}
        shear_rate = self._compute_shear_rate(u, v, spatial_props, data)
        reaction_terms = self.kinetics.compute_species_reactions(species_dict, shear_rate)
        keller_species = SPECIES_GROUPS["keller"]

        for sp in fast_species + slow_species:
            key = sp.name
            scale_idx = sp.value
            scale_c = scales[scale_idx]

            # Global reference time for chain rule scaling
            t_ref_global = self.cfg.t_final

            # Fast chemistry is treated as quasi-steady over coarse macro timesteps.
            # Only slow species receive explicit transient dC/dt_nd supervision.
            if d_pred_dt is not None and sp in slow_species:
                exp_pred = torch.exp(species_preds_safe[:, scale_idx])
                # d_pred_dt is now d(logC)/dt_nd
                dC_dt_nd = exp_pred * d_pred_dt[:, scale_idx]
            else:
                dC_dt_nd = species_preds_safe[:, scale_idx] * 0.0

            C_nd = nd_species_preds[:, scale_idx]
            C = species_dict[key]
            base_D = self.D_coeff[key]
            D = base_D + D_s if sp in keller_species else base_D
            R = reaction_terms[key]

            # Sparse matrix gradients/Laplacian in normalized coordinates (x* = x / d_bar).
            C_col_nd = C_nd.unsqueeze(1)
            dC_dx_nd = torch.sparse.mm(data.G_x, C_col_nd).squeeze(1)
            dC_dy_nd = torch.sparse.mm(data.G_y, C_col_nd).squeeze(1)
            advection_nd = u * dC_dx_nd + v * dC_dy_nd

            # Diffusion in dimensionless form: div((1/Pe) grad(C_nd)).
            # Even with constant base D, 1/Pe is spatially varying because u_ref and d_bar vary.
            laplacian_C_nd = torch.sparse.mm(data.Laplacian, C_col_nd).squeeze(1)
            inv_pe = as_tensor_like(D, like=u_ref_safe) / (u_ref_safe * d_bar_safe)
            inv_pe_col = inv_pe.unsqueeze(1)
            dinvpe_dx = torch.sparse.mm(data.G_x, inv_pe_col).squeeze(1)
            dinvpe_dy = torch.sparse.mm(data.G_y, inv_pe_col).squeeze(1)
            diffusion_nd = (inv_pe * laplacian_C_nd) + (dinvpe_dx * dC_dx_nd + dinvpe_dy * dC_dy_nd)

            # Local convective-time reaction scale for all species.
            reaction_nd = (d_bar_safe / u_ref_safe) * (R / torch.clamp(scale_c, min=1e-12))

            if sp in SPECIES_GROUPS["solid"]:
                # Polymerized fibrin is a solid matrix; it does not advect or diffuse.
                advection_nd = torch.zeros_like(advection_nd)
                diffusion_nd = torch.zeros_like(diffusion_nd)

            # Scale convective ND terms into global ND time using chain rule.
            time_ratio = time_ratio_global_to_convective(
                t_ref_global=t_ref_global,
                d_bar=d_bar_safe,
                u_ref=u_ref_safe,
            )

            # Fully non-dimensional ADR residual scaled to Convective Time (O(1)).
            # Dividing the temporal term by ``time_ratio`` (instead of multiplying the
            # spatial terms by it) collapses advection / diffusion / reaction onto the
            # same magnitude scale as the chemistry, so the optimizer can "see" the
            # reaction equations alongside transport instead of being dominated by
            # O(1e6) convective gradients.
            residual_nd = (dC_dt_nd / time_ratio) + advection_nd - diffusion_nd - reaction_nd
            loss_c = torch.mean(residual_nd ** 2)

            if sp in fast_species:
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
            bulk_n = len(BULK_SPECIES_ORDER)
            preds_inlet = biochem_preds[mask_inlet, :bulk_n]
            targs_inlet = data.bio_inlet_bc[mask_inlet, :bulk_n]
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
                    normals_fb = data.x[mask_outlet, BiochemNodeFeat.WALL_NORMAL]
                    nx_fb = normals_fb[:, 0]
                    ny_fb = normals_fb[:, 1]
                    nx = torch.where(weak, nx_fb, nx)
                    ny = torch.where(weak, ny_fb, ny)
            else:
                normals_fb = data.x[mask_outlet, BiochemNodeFeat.WALL_NORMAL]
                nx = normals_fb[:, 0]
                ny = normals_fb[:, 1]

            nmag = torch.sqrt(nx * nx + ny * ny + 1e-12)
            nx = nx / nmag
            ny = ny / nmag

            scales = self._get_species_scales(biochem_preds.device, biochem_preds.dtype)
            biochem_safe = torch.clamp(biochem_preds, min=-10.0, max=8.0)
            bulk_n = len(BULK_SPECIES_ORDER)
            linear_bulk = torch.expm1(biochem_safe) * scales[:bulk_n]

            d_bar = spatial_props['d_bar'].to(device=biochem_preds.device, dtype=biochem_preds.dtype)
            d_bar_safe = torch.clamp(d_bar, min=1e-8)

            flux_scale = self._get_adr_norm_scales(
                biochem_preds.device, biochem_preds.dtype
            )[:bulk_n].mean().clamp(min=1e-12)
            loss_outlet_acc = torch.zeros((), device=biochem_preds.device, dtype=biochem_preds.dtype)
            mobile_species = [sp for sp in BULK_SPECIES_ORDER if sp not in SPECIES_GROUPS["solid"]]
            n_mobile = len(mobile_species)
            for sp in mobile_species:
                C_col = linear_bulk[:, sp.value].unsqueeze(1)
                dC_dx = torch.sparse.mm(data.G_x, C_col).squeeze(1) / d_bar_safe
                dC_dy = torch.sparse.mm(data.G_y, C_col).squeeze(1) / d_bar_safe
                dC_dn = dC_dx[mask_outlet] * nx + dC_dy[mask_outlet] * ny
                dC_dn = torch.nan_to_num(dC_dn, nan=0.0, posinf=0.0, neginf=0.0)
                d_nd = dC_dn / flux_scale
                loss_outlet_acc = loss_outlet_acc + F.huber_loss(
                    d_nd, torch.zeros_like(d_nd), delta=self._biochem_huber_delta
                )

            loss_outlet = loss_outlet_acc / float(n_mobile)

        return loss_inlet, loss_outlet

    def biochem_wall_residual(self, biochem_preds, wall_preds, velocity_field, spatial_props, data, dM_pred_dt=None):
        """Enforces Surface Platelet Adhesion with Transient Surface ODEs."""
        mask_wall = data.mask_wall.view(-1).bool()
        if not mask_wall.any():
            z = biochem_preds.sum() * 0.0
            return z, z

        scales = self._get_species_scales(biochem_preds.device, biochem_preds.dtype)
        biochem_preds_safe = torch.clamp(biochem_preds, min=-10.0, max=8.0)
        nd_biochem_preds = torch.expm1(biochem_preds_safe)
        bulk_n = len(BULK_SPECIES_ORDER)
        linear_biochem_preds = nd_biochem_preds * scales[:bulk_n]

        RP_wall = linear_biochem_preds[mask_wall, BulkSpecies.RP.value]
        AP_wall = linear_biochem_preds[mask_wall, BulkSpecies.AP.value]
        APR_wall = linear_biochem_preds[mask_wall, BulkSpecies.APR.value]
        APS_wall = linear_biochem_preds[mask_wall, BulkSpecies.APS.value]
        PT_wall = linear_biochem_preds[mask_wall, BulkSpecies.PT.value]
        T_wall = linear_biochem_preds[mask_wall, BulkSpecies.T.value]

        wall_preds_safe = torch.clamp(wall_preds, min=-10.0, max=8.0)
        nd_wall_preds = torch.expm1(wall_preds_safe)

        # USE CENTRALIZED SCALE
        Minf_scaled = self.cfg.Minf * self.cfg.surface_scale

        M = nd_wall_preds[mask_wall, WallSpecies.M.value] * Minf_scaled
        Mas = nd_wall_preds[mask_wall, WallSpecies.Mas.value] * Minf_scaled
        Mat = nd_wall_preds[mask_wall, WallSpecies.Mat.value] * Minf_scaled

        t_ref_global = self.cfg.t_final

        if dM_pred_dt is not None:
            exp_wall = torch.exp(wall_preds_safe[mask_wall])
            # These derivatives are now dM/dt_nd
            dM_dt_nd = exp_wall[:, WallSpecies.M.value] * dM_pred_dt[mask_wall, WallSpecies.M.value] * Minf_scaled
            dMas_dt_nd = exp_wall[:, WallSpecies.Mas.value] * dM_pred_dt[mask_wall, WallSpecies.Mas.value] * Minf_scaled
            dMat_dt_nd = exp_wall[:, WallSpecies.Mat.value] * dM_pred_dt[mask_wall, WallSpecies.Mat.value] * Minf_scaled
        else:
            z = M * 0.0
            dM_dt_nd = z
            dMas_dt_nd = z
            dMat_dt_nd = z

        M_tot = M + Mas + Mat
        Minf = self.cfg.Minf * self.cfg.surface_scale

        # 4. Compute Local Surface Activation & Spatial Gradients
        u = velocity_field[..., 0]
        v = velocity_field[..., 1]

        global_shear = self._compute_shear_rate(u, v, spatial_props, data)
        shear_wall = global_shear[mask_wall]

        d_bar = spatial_props['d_bar'].to(biochem_preds.device)
        d_bar_safe = torch.clamp(d_bar, min=1e-8)

        # Local convective time at the wall (used by both the surface ODE residuals
        # below and the Neumann flux normalization further down).
        d_bar_wall = d_bar_safe[mask_wall]
        u_ref_wall = spatial_props['u_ref'].to(biochem_preds.device)[mask_wall]
        u_ref_wall_safe = torch.clamp(u_ref_wall, min=1e-8)
        conv_time = d_bar_wall / u_ref_wall_safe

        # Streamwise Directional Derivative (d/ds)
        dshear_dx = torch.sparse.mm(data.G_x, global_shear.unsqueeze(1)).squeeze(1)
        dshear_dy = torch.sparse.mm(data.G_y, global_shear.unsqueeze(1)).squeeze(1)

        vel_mag = torch.sqrt(u ** 2 + v ** 2) + 1e-8
        u_dir = u / vel_mag
        v_dir = v / vel_mag

        dshear_ds = (u_dir * dshear_dx) + (v_dir * dshear_dy)
        dshear_ds_wall = (dshear_ds / d_bar_safe)[mask_wall]
        dshear_abs = torch.abs(dshear_ds_wall) + 1e-6

        raw_availability = 1.0 - (M_tot / Minf)
        if self._availability_negative_slope > 0.0:
            availability = torch.where(
                raw_availability >= 0.0,
                raw_availability,
                raw_availability * self._availability_negative_slope,
            )
            availability = torch.clamp(availability, max=1.0)
        else:
            availability = torch.clamp(raw_availability, min=1e-8, max=1.0)

        # COMSOL Pathological Conditionals (Soft-Logic)
        is_separation = self.kinetics._soft_step(dshear_ds_wall, self.cfg.sgt, self.kinetics.T_grad, reverse=True)
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
        # COMSOL srf1 SurfaceReactionsFlux copy-paste bug:
        # J0_M, J0_Mas, and J0_Mat were set to the same string, so deposition is
        # dumped uniformly into all three species and the -k_pa*M / +k_pa*M
        # activation transfer coupling is effectively ignored.
        R_deposition = pathological_RP_adhesion + low_shear_RP_adhesion
        R_M = R_deposition
        R_Mas = R_deposition
        R_Mat = R_deposition

        # --- TRANSIENT SURFACE RESIDUALS (ODEs) ---
        # Apply chain rule and rescale to local Convective Time (O(1)) so the surface
        # ODE landscape matches the bulk ADR loss landscape.
        # Equivalent to (dM/dt_global - R_M) but expressed via dM/dt_nd:
        #   res = conv_time * (dM/dt_nd / t_ref_global - R_M) / Minf
        res_M = conv_time * ((dM_dt_nd / t_ref_global) - R_M) / Minf
        res_Mas = conv_time * ((dMas_dt_nd / t_ref_global) - R_Mas) / Minf
        res_Mat = conv_time * ((dMat_dt_nd / t_ref_global) - R_Mat) / Minf

        loss_surface = (
                               F.huber_loss(res_M, torch.zeros_like(res_M), delta=self._biochem_huber_delta) +
                               F.huber_loss(res_Mas, torch.zeros_like(res_Mas), delta=self._biochem_huber_delta) +
                               F.huber_loss(res_Mat, torch.zeros_like(res_Mat), delta=self._biochem_huber_delta)
                       ) / 3.0

        # --- 6. NEUMANN BOUNDARY FLUX COUPLING (Bulk-to-Wall) ---
        J_in_RP = - (pathological_RP_adhesion + low_shear_RP_adhesion)
        J_in_AP = - (
                    pathological_AP_adhesion + low_shear_AP_adhesion + pathological_Mas_adhesion + low_shear_Mas_adhesion)
        J_in_APR = self.cfg.lambda_adp * (pathological_RP_adhesion + low_shear_RP_adhesion)
        J_in_APS = self.cfg.s_t * Mat
        J_in_PT = - self.cfg.beta * self.cfg.phi_at * Mat * PT_wall
        J_in_T = self.cfg.beta * self.cfg.phi_at * Mat * PT_wall

        flux_targets = {
            BulkSpecies.RP: J_in_RP,
            BulkSpecies.AP: J_in_AP,
            BulkSpecies.APR: J_in_APR,
            BulkSpecies.APS: J_in_APS,
            BulkSpecies.PT: J_in_PT,
            BulkSpecies.T: J_in_T,
        }

        wall_normals = data.x[mask_wall, BiochemNodeFeat.WALL_NORMAL]
        nx = wall_normals[:, 0]
        ny = wall_normals[:, 1]
        # ``u_ref_wall`` is hoisted above near ``conv_time`` so it can be shared by
        # the surface ODE residuals and this Neumann flux normalization.

        a_RBC_m_wall = self.cfg.d_RBC / 2.0
        tau_max = 2000.0
        D_s_wall = 0.18 * (a_RBC_m_wall ** 2) * tau_max
        keller_species = SPECIES_GROUPS["keller"]

        loss_flux = biochem_preds.sum() * 0.0

        for sp, J_in in flux_targets.items():
            idx = sp.value
            C_field = linear_biochem_preds[:, idx].unsqueeze(1)

            # Sparse Matrix Gradient Calculation
            dC_dx = torch.sparse.mm(data.G_x, C_field).squeeze(1) / d_bar_safe
            dC_dy = torch.sparse.mm(data.G_y, C_field).squeeze(1) / d_bar_safe

            # Dot with outward normal vector (D * grad(C) dot n)
            dC_dn_wall = dC_dx[mask_wall] * nx + dC_dy[mask_wall] * ny

            # Diffusion Coefficient at Wall
            base_D = self.D_coeff[sp.name]
            D_eff = base_D + D_s_wall if sp in keller_species else base_D

            predicted_flux = -as_tensor_like(D_eff, like=dC_dn_wall) * dC_dn_wall
            # J_in is signed inward (negative for sinks), while predicted_flux is outward.
            # Enforce outward + inward = 0 for consistent wall coupling.
            flux_residual = predicted_flux + J_in

            # Use convective flux for stable normalization
            char_flux = scales[idx] * u_ref_wall
            flux_residual_nd = flux_residual / (char_flux + 1e-8)

            loss_flux += F.huber_loss(
                flux_residual_nd, torch.zeros_like(flux_residual_nd), delta=self._biochem_huber_delta
            )

        return loss_surface, loss_flux

    def _compute_shear_rate(self, u, v, spatial_props, data):
        # Sparse Matrix Gradient Calculation
        du_dx = torch.sparse.mm(data.G_x, u.unsqueeze(1)).squeeze(1)
        du_dy = torch.sparse.mm(data.G_y, u.unsqueeze(1)).squeeze(1)
        dv_dx = torch.sparse.mm(data.G_x, v.unsqueeze(1)).squeeze(1)
        dv_dy = torch.sparse.mm(data.G_y, v.unsqueeze(1)).squeeze(1)

        # This is non-dimensional.
        gamma_dot_nd = compute_shear_rate(du_dx, du_dy, dv_dx, dv_dy, eps=1e-6)

        # Redimensionalize to physical 1/s
        u_ref = spatial_props['u_ref'].to(u.device)
        d_bar = spatial_props['d_bar'].to(u.device)
        d_bar_safe = torch.clamp(d_bar, min=1e-8)

        return gamma_dot_nd * (u_ref / d_bar_safe)


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