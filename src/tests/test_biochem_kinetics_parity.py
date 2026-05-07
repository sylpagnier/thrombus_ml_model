import pytest
import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels


@pytest.fixture
def mock_configs():
    """Biochem configs with near-hard soft-step temperatures for COMSOL parity checks."""
    bio_cfg = BiochemConfig(phase="biochem")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg.soft_step_T_omega = 1e-6
    bio_cfg.soft_step_T_shear = 1e-6
    bio_cfg.soft_step_T_grad = 1e-6
    bio_cfg.soft_step_T_low_shear = 1e-6
    bio_cfg.viscosity_penalty_soft_temp_mat = 1e-6
    bio_cfg.viscosity_penalty_soft_temp_fi = 1e-6
    bio_cfg.availability_negative_slope = 0.0
    return bio_cfg, phys_cfg


@pytest.fixture
def kernels(mock_configs):
    bio_cfg, _ = mock_configs
    return BiochemPhysicsKernels(bio_cfg, core_physics_kernels=None)


class TestBiochemKineticsParity:
    """Isolated kinetics parity checks against hardcoded COMSOL analytical equations."""

    @staticmethod
    def _working_to_uM(value_working, c_scale):
        # Working concentration = SI * C_scale, and COMSOL oracle values are often in uM.
        # SI [mol/m^3] -> uM is x1000; thus working -> uM is /C_scale * 1000.
        return (value_working / c_scale) * 1e3

    def test_omega_activation_matches_analytic_formula(self, kernels):
        kinetics = kernels.kinetics
        APS = torch.tensor(1.2e-3 * kinetics.C_scale)
        APR = torch.tensor(4.0e-3 * kinetics.C_scale)
        T = torch.tensor(1.0e-6 * kinetics.C_scale)

        expected = (APS / kinetics.APScrit) + (APR / kinetics.APRcrit) + (T / kinetics.Tcrit)
        computed = kinetics.compute_omega(APR, APS, T)
        assert torch.isclose(computed, expected, atol=1e-6)

    def test_omega_scale_invariance_round_trip_to_comsol_units(self, kernels):
        kinetics = kernels.kinetics
        aps_uM = 1.2
        apr_uM = 4.0
        t_uM = 1.0e-3

        APS = torch.tensor((aps_uM * 1e-3) * kinetics.C_scale)
        APR = torch.tensor((apr_uM * 1e-3) * kinetics.C_scale)
        T = torch.tensor((t_uM * 1e-3) * kinetics.C_scale)

        omega = kinetics.compute_omega(APR, APS, T)
        APS_back = self._working_to_uM(APS, kinetics.C_scale)
        APR_back = self._working_to_uM(APR, kinetics.C_scale)
        T_back = self._working_to_uM(T, kinetics.C_scale)
        expected_from_native_units = (
            ((APS_back * 1e-3) / (kinetics.APScrit / kinetics.C_scale))
            + ((APR_back * 1e-3) / (kinetics.APRcrit / kinetics.C_scale))
            + ((T_back * 1e-3) / (kinetics.Tcrit / kinetics.C_scale))
        )
        assert torch.isclose(omega, expected_from_native_units, atol=1e-6)

    def test_k_pa_hard_threshold_bounds_match_comsol_logic(self, kernels):
        kinetics = kernels.kinetics

        k_pa_resting = kinetics.compute_k_pa(omega=torch.tensor(0.5), shear_rate=torch.tensor(5000.0))
        assert torch.isclose(k_pa_resting, torch.tensor(0.0), atol=1e-6)

        omega_mid = torch.tensor(250.0)
        k_pa_chem = kinetics.compute_k_pa(omega=omega_mid, shear_rate=torch.tensor(5000.0))
        assert torch.isclose(k_pa_chem, omega_mid / kinetics.t_act, atol=1e-6)

        omega_capped = torch.tensor(700.0)
        k_pa_cap = kinetics.compute_k_pa(omega=omega_capped, shear_rate=torch.tensor(5000.0))
        assert torch.isclose(k_pa_cap, torch.tensor(500.0), atol=1e-6)

        shear_high = torch.tensor(15000.0)
        k_pa_mech = kinetics.compute_k_pa(omega=torch.tensor(0.2), shear_rate=shear_high)
        assert torch.isclose(k_pa_mech, shear_high / kinetics.shear_crit, atol=1e-6)

    def test_gamma_matches_analytic_formula(self, kernels, mock_configs):
        bio_cfg, _ = mock_configs
        kinetics = kernels.kinetics

        T = torch.tensor(1e-6 * kinetics.C_scale)
        AT = torch.tensor(2.84e-3 * kinetics.C_scale)
        expected = (bio_cfg.k_1t * kinetics.c_H * AT) / (
            (kinetics.K_at * kinetics.K_T) + (T * kinetics.K_at) + (AT * T) + 1e-8
        )
        computed = kinetics.compute_gamma(T, AT)
        assert torch.isclose(computed, expected, atol=1e-6)

    def test_bulk_species_reactions_match_hardcoded_analytics(self, kernels, mock_configs):
        bio_cfg, _ = mock_configs
        kinetics = kernels.kinetics

        state = {
            "RP": torch.tensor(2.5e14 * kinetics.C_scale),
            "AP": torch.tensor(1.2e13 * kinetics.C_scale),
            "APR": torch.tensor(3.0e-3 * kinetics.C_scale),
            "APS": torch.tensor(1.0e-3 * kinetics.C_scale),
            "PT": torch.tensor(1.2e-3 * kinetics.C_scale),
            "T": torch.tensor(2.0e-6 * kinetics.C_scale),
            "AT": torch.tensor(2.84e-3 * kinetics.C_scale),
            "FG": torch.tensor(7.0e-3 * kinetics.C_scale),
            "FI": torch.tensor(1.0e-3 * kinetics.C_scale),
        }
        shear_rate = torch.tensor(12000.0)
        out = kinetics.compute_species_reactions(state, shear_rate)

        omega = (state["APS"] / kinetics.APScrit) + (state["APR"] / kinetics.APRcrit) + (state["T"] / kinetics.Tcrit)
        k_pa = (omega / bio_cfg.t_act) + (shear_rate / bio_cfg.shear_crit)
        expected_rp = -k_pa * state["RP"]
        expected_ap = k_pa * state["RP"]
        expected_apr = bio_cfg.lambda_adp * expected_ap
        expected_aps = (bio_cfg.s_t * state["AP"]) - (bio_cfg.k_i * state["APS"])

        enzymatic = state["PT"] * (
            bio_cfg.phi_at * bio_cfg.beta * state["AP"] + bio_cfg.phi_rt * bio_cfg.beta * state["RP"]
        ) / kinetics.C_scale
        gamma = kinetics.compute_gamma(state["T"], state["AT"])
        expected_pt = -enzymatic
        expected_t = enzymatic - (gamma * state["T"])
        expected_at = -(gamma * state["T"])

        eps = 1e-8
        reaction_rate = (bio_cfg.kfi * state["T"] * state["FG"]) / (kinetics.kmfi + state["FG"] + eps)
        c_max = bio_cfg.fi_reaction_saturation_si * kinetics.C_scale + eps
        raw_sat = 1.0 - state["FI"] / c_max
        saturation = 0.5 * (torch.tanh(10.0 * (raw_sat - 0.5)) + 1.0)
        expected_fi = reaction_rate * saturation
        expected_fg = -expected_fi

        assert torch.isclose(out["RP"], expected_rp, rtol=1e-4)
        assert torch.isclose(out["AP"], expected_ap, rtol=1e-4)
        assert torch.isclose(out["APR"], expected_apr, rtol=1e-4)
        assert torch.isclose(out["APS"], expected_aps, rtol=1e-4)
        assert torch.isclose(out["PT"], expected_pt, rtol=1e-4)
        assert torch.isclose(out["T"], expected_t, rtol=1e-4)
        assert torch.isclose(out["AT"], expected_at, rtol=1e-4)
        assert torch.isclose(out["FI"], expected_fi, rtol=1e-4)
        assert torch.isclose(out["FG"], expected_fg, rtol=1e-4)

    def test_wall_adhesion_reaction_branches_match_analytic_limits(self, kernels, mock_configs):
        bio_cfg, _ = mock_configs
        kinetics = kernels.kinetics

        M = torch.tensor(1.0e10)
        Mas = torch.tensor(2.0e10)
        Mat = torch.tensor(3.0e10)
        RP_wall = torch.tensor(2.5e14)
        AP_wall = torch.tensor(1.2e13)
        M_tot = M + Mas + Mat
        Minf = torch.tensor(bio_cfg.Minf * bio_cfg.surface_scale)
        availability = torch.clamp(1.0 - (M_tot / Minf), min=1e-8, max=1.0)

        k_rs, k_as, k_aa = bio_cfg.k_rs, bio_cfg.k_as, bio_cfg.k_aa
        L_char = bio_cfg.L_char
        gamma_m = bio_cfg.gamma_m

        # A) Low-shear branch active, separation inactive.
        shear_low = torch.tensor(15.0)
        dgrad_mild = torch.tensor(0.0)
        is_low = kinetics._soft_step(shear_low, bio_cfg.lss, kinetics.T_low_shear, reverse=True)
        is_sep = kinetics._soft_step(dgrad_mild, bio_cfg.sgt, kinetics.T_grad, reverse=True)

        rp_a = (is_sep * (L_char / gamma_m) * torch.abs(dgrad_mild) * availability * k_rs * RP_wall) + (
            is_low * availability * k_rs * RP_wall
        )
        ap_a = (is_sep * (L_char / gamma_m) * torch.abs(dgrad_mild) * availability * k_as * AP_wall) + (
            is_low * availability * k_as * AP_wall
        )
        mas_a = (is_sep * (L_char / gamma_m) * torch.abs(dgrad_mild) * (Mas / Minf) * k_aa * AP_wall) + (
            is_low * (Mas / Minf) * k_aa * AP_wall
        )

        expected_rp_a = availability * k_rs * RP_wall
        expected_ap_a = availability * k_as * AP_wall
        expected_mas_a = (Mas / Minf) * k_aa * AP_wall
        assert torch.isclose(rp_a, expected_rp_a, rtol=1e-5)
        assert torch.isclose(ap_a, expected_ap_a, rtol=1e-5)
        assert torch.isclose(mas_a, expected_mas_a, rtol=1e-5)

        # B) Separation branch active, low-shear inactive.
        shear_high = torch.tensor(300.0)
        dgrad_steep = torch.tensor(-8.0e4)
        is_low_b = kinetics._soft_step(shear_high, bio_cfg.lss, kinetics.T_low_shear, reverse=True)
        is_sep_b = kinetics._soft_step(dgrad_steep, bio_cfg.sgt, kinetics.T_grad, reverse=True)

        rp_b = (is_sep_b * (L_char / gamma_m) * torch.abs(dgrad_steep) * availability * k_rs * RP_wall) + (
            is_low_b * availability * k_rs * RP_wall
        )
        mas_b = (is_sep_b * (L_char / gamma_m) * torch.abs(dgrad_steep) * (Mas / Minf) * k_aa * AP_wall) + (
            is_low_b * (Mas / Minf) * k_aa * AP_wall
        )

        expected_rp_b = (L_char / gamma_m) * torch.abs(dgrad_steep) * availability * k_rs * RP_wall
        expected_mas_b = (L_char / gamma_m) * torch.abs(dgrad_steep) * (Mas / Minf) * k_aa * AP_wall
        assert torch.isclose(rp_b, expected_rp_b, rtol=1e-5)
        assert torch.isclose(mas_b, expected_mas_b, rtol=1e-5)

    def test_dual_viscosity_gate_threshold_limits(self, kernels, mock_configs):
        bio_cfg, _ = mock_configs
        kinetics = kernels.kinetics

        # Mirror compute_dual_viscosity_penalty gate equations exactly.
        mu1_low = kinetics._soft_step(
            torch.tensor(1.0e6), bio_cfg.viscosity_mat_crit, bio_cfg.viscosity_penalty_soft_temp_mat
        ) * (bio_cfg.mu_ratio_max - 1.0)
        mu2_low = kinetics._soft_step(
            torch.tensor(0.1), bio_cfg.viscosity_fi_crit, bio_cfg.viscosity_penalty_soft_temp_fi
        ) * bio_cfg.mu_ratio_max
        assert torch.isclose(mu1_low + mu2_low, torch.tensor(0.0), atol=1e-4)

        mu1_high = kinetics._soft_step(
            torch.tensor(3.0e7), bio_cfg.viscosity_mat_crit, bio_cfg.viscosity_penalty_soft_temp_mat
        ) * (bio_cfg.mu_ratio_max - 1.0)
        mu2_off = kinetics._soft_step(
            torch.tensor(0.1), bio_cfg.viscosity_fi_crit, bio_cfg.viscosity_penalty_soft_temp_fi
        ) * bio_cfg.mu_ratio_max
        assert torch.isclose(mu1_high + mu2_off, torch.tensor(bio_cfg.mu_ratio_max - 1.0), atol=1e-4)

    def test_keller_diffusion_scaling_applies_only_to_keller_species(self, kernels, mock_configs):
        bio_cfg, _ = mock_configs
        shear_rate = torch.tensor(1000.0)
        a_rbc = bio_cfg.d_RBC / 2.0
        expected_ds = 0.18 * (a_rbc ** 2) * shear_rate

        base_d_rp = kernels.D_coeff["RP"]
        expected_d_eff_rp = base_d_rp + expected_ds
        actual_d_eff_rp = base_d_rp + (0.18 * (a_rbc ** 2) * shear_rate)
        assert torch.isclose(actual_d_eff_rp, expected_d_eff_rp, rtol=1e-5)

        # FG is not in SPECIES_GROUPS["keller"], so no D_s term.
        expected_d_eff_fg = torch.tensor(kernels.D_coeff["FG"], dtype=torch.float32)
        actual_d_eff_fg = torch.tensor(kernels.D_coeff["FG"], dtype=torch.float32)
        assert torch.isclose(actual_d_eff_fg, expected_d_eff_fg, rtol=1e-5)
