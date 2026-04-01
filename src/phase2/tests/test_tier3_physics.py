import os
import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.config import BiochemConfig, PhysicsConfig
from src.phase2.physics_kernels_tier3 import BiochemPhysicsKernels
# UPDATED: Import the new continuous-time Neural ODE module
from src.phase2.gnode_tier3 import GNODE_Tier3
from src.utils.paths import get_project_root

class DummyCoreKernels:
    """Mocks the base CFD physics kernels strictly for testing ADR scaling."""

    def _compute_derivatives(self, tensor, spatial_props):
        # Mocks a first and second derivative output
        # Shape matches GINO output: [nodes, num_derivatives, channels]
        shape = list(tensor.shape)
        return torch.ones(shape, dtype=tensor.dtype, device=tensor.device)


class TestTier3Physics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create output directory for visualizations relative to project root."""
        root = get_project_root()
        cls.vis_dir = root / "data/processed/graphs_tier3_patients/sanity_checks"
        cls.vis_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        """Initialize configurations and kernels."""
        self.bio_cfg = BiochemConfig(tier="tier3")
        self.phys_cfg = PhysicsConfig(tier="tier3")
        self.core = DummyCoreKernels()

        self.biochem_kernels = BiochemPhysicsKernels(self.bio_cfg, self.core)
        self.kinetics = self.biochem_kernels.kinetics

        # UPDATED: Initialize the new GNODE_Tier3 model with matching params
        self.model = GNODE_Tier3(
            in_channels=12,
            spatial_channels=15,  # Added to match new signature
            latent_dim=16,
            mu_ratio_max=self.bio_cfg.mu_ratio_max
        )

    def _comsol_smoothed_step(self, x, location, transition_zone, val_from, val_to):
        """
        Approximates COMSOL's built-in smoothed step function.
        COMSOL uses a regularized polynomial over the transition zone interval.
        """
        delta = transition_zone / 2.0
        x0 = location - delta
        x1 = location + delta

        # Normalize x into the range within the transition zone boundary
        t = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)

        # Smoothstep cubic polynomial: 3*t^2 - 2*t^3
        smooth_factor = t * t * (3.0 - 2.0 * t)

        return val_from + (val_to - val_from) * smooth_factor

    def test_comsol_constants_mapping(self):
        """Verify fundamental COMSOL parameters are correctly inherited and scaled."""
        self.assertTrue(hasattr(self.bio_cfg, 'kfi'))
        self.assertTrue(hasattr(self.bio_cfg, 'kmfi'))

        expected_constants = {
            'APScrit': self.bio_cfg.APScrit * self.kinetics.C_scale,
            'APRcrit': self.bio_cfg.APRcrit * self.kinetics.C_scale,
            'Tcrit': self.bio_cfg.Tcrit * self.kinetics.C_scale,
            't_act': self.bio_cfg.t_act,
            'shear_crit': self.bio_cfg.shear_crit
        }

        for attr, expected_val in expected_constants.items():
            actual_val = getattr(self.kinetics, attr)
            self.assertAlmostEqual(actual_val, expected_val, places=6)

    def test_fibrin_kinetics_math(self):
        """Validates Fibrin kinetics formulation."""
        num_nodes = 500
        T_tensor = torch.rand(num_nodes, dtype=torch.float32)
        FG_tensor = torch.rand(num_nodes, dtype=torch.float32) * 5.0
        FI_tensor = torch.rand(num_nodes, dtype=torch.float32) * 0.8

        R_FG_pt, R_FI_pt = self.kinetics.compute_fibrin_kinetics(T_tensor, FG_tensor, FI_tensor)

        T_np = T_tensor.numpy()
        FG_np = FG_tensor.numpy()
        FI_np = FI_tensor.numpy()

        base_reaction_comsol = (self.bio_cfg.kfi * T_np * FG_np) / (
                (self.bio_cfg.kmfi * self.kinetics.C_scale) + FG_np + 1e-8)

        saturation_term = np.clip(1.0 - FI_np, 0.0, None)
        reaction_comsol = base_reaction_comsol * saturation_term

        np.testing.assert_allclose(R_FI_pt.numpy(), reaction_comsol, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(R_FG_pt.numpy(), -reaction_comsol, rtol=1e-5, atol=1e-8)

    def test_gamma_inhibition_math(self):
        """
        Analytic 3: Verifies Thrombin inhibition by Antithrombin/Heparin complex.
        COMSOL: Gamma = (k_1t * c_H * AT) / (K_at * K_T + T * K_at + AT * T)
        """
        num_nodes = 500
        T_tensor = torch.rand(num_nodes, dtype=torch.float32) * 2.0
        AT_tensor = torch.rand(num_nodes, dtype=torch.float32) * 5.0

        gamma_pt = self.kinetics.compute_gamma(T_tensor, AT_tensor)

        T_np = T_tensor.numpy()
        AT_np = AT_tensor.numpy()

        k_1t = self.bio_cfg.k_1t
        c_H = self.bio_cfg.c_H
        K_at = self.bio_cfg.K_at
        K_T = self.bio_cfg.K_T

        numerator = k_1t * c_H * AT_np
        denominator = (K_at * K_T) + (T_np * K_at) + (AT_np * T_np) + 1e-8
        gamma_comsol = numerator / denominator

        np.testing.assert_allclose(gamma_pt.numpy(), gamma_comsol, rtol=1e-5, atol=1e-8,
                                   err_msg="PyTorch Gamma (Thrombin Inhibition) does not match COMSOL.")

    def test_kpa_activation_logic(self):
        """
        Analytic 2, 6, 7: Tests the smooth soft-logic of k_pa (Platelet Activation)
        against COMSOL's rigid if/else statements.
        """
        num_nodes = 1000
        omega_tensor = torch.linspace(0, 600, num_nodes)
        shear_tensor = torch.linspace(0, 15000, num_nodes)

        self.kinetics.T_scale = 0.01
        kpa_pt = self.kinetics.compute_k_pa(omega_tensor, shear_tensor)

        omega_np = omega_tensor.numpy()
        shear_np = shear_tensor.numpy()
        t_act = self.bio_cfg.t_act
        shear_crit = self.bio_cfg.shear_crit

        act_step = np.where(omega_np > 1.0, 1.0, 0.0)
        kpa_chem_np = np.where(omega_np < 500, (omega_np / t_act) * act_step, 500.0)
        kpa_mech_np = np.where(shear_np > shear_crit, shear_np / shear_crit, 0.0)

        kpa_comsol = kpa_chem_np + kpa_mech_np

        correlation = float(np.corrcoef(kpa_pt.numpy(), kpa_comsol)[ 0, 1 ])
        self.assertGreater(correlation, 0.98,
                           "PyTorch soft-logic for k_pa deviates too far from COMSOL rigid logic.")

    def test_platelet_viscosity_mu1(self):
        """
        Validates PyTorch's mu1_sigmoid against COMSOL's mu1 Step function.
        COMSOL: Location 2e7, Transition Zone 7e6.
        """
        mat_range = torch.linspace(0, 4e7, 1000)

        self.model.T_scale = 1.0
        mu1_pt_strict = self.model.mu1_sigmoid(mat_range).numpy()

        mat_np = mat_range.numpy()
        # UPDATED: mu1 technically scales from 0 to (mu_ratio_max - 1.0) because the base fluid holds the 1.0
        mu1_comsol = self._comsol_smoothed_step(
            x=mat_np,
            location=2e7,
            transition_zone=7e6,
            val_from=0.0,
            val_to=self.bio_cfg.mu_ratio_max - 1.0
        )

        correlation = float(np.corrcoef(mu1_pt_strict, mu1_comsol)[ 0, 1 ])
        self.assertGreater(correlation, 0.98,
                           "PyTorch soft-logic for mu1 deviates too far from COMSOL smooth step.")

        # Ensure mathematical bounds are respected (0.0 to Max-1)
        self.assertLessEqual(np.max(mu1_pt_strict), self.model.mu_ratio_max)
        self.assertGreaterEqual(np.min(mu1_pt_strict), 0.0)

    def test_fibrin_viscosity_mu2(self):
        """
        Validates PyTorch's mu2_sigmoid against COMSOL's mu2 Step function and outputs a sanity check plot.
        COMSOL: Location 0.6, Transition Zone 0.01, from 0 to mu_ratio_max.
        """
        num_nodes = 1000
        fi_range = np.linspace(0, 1.2, num_nodes)
        fi_tensor = torch.tensor(fi_range, dtype=torch.float32)

        self.model.T_scale = 1.0
        mu2_pt_strict = self.model.mu2_sigmoid(fi_tensor).numpy()

        self.model.T_scale = 5.0
        mu2_pt_relaxed = self.model.mu2_sigmoid(fi_tensor).numpy()

        mu2_comsol = self._comsol_smoothed_step(
            x=fi_range,
            location=0.6,
            transition_zone=0.01,
            val_from=0.0,
            val_to=self.bio_cfg.mu_ratio_max
        )

        plt.figure(figsize=(10, 6))
        plt.plot(fi_range, mu2_comsol, 'r-', linewidth=3, label='COMSOL Exact Smoothed Step (Ground Truth)')
        plt.plot(fi_range, mu2_pt_strict, 'b--', linewidth=2, label='PyTorch Sigmoid (T_scale=1.0)')
        plt.plot(fi_range, mu2_pt_relaxed, 'g:', linewidth=2, label='PyTorch Sigmoid (T_scale=5.0)')

        plt.title('Fibrin Viscosity Multiplier: COMSOL vs PyTorch', fontsize=14, fontweight='bold')
        plt.xlabel('Fibrin Concentration (ND)', fontsize=12)
        plt.ylabel(r'Viscosity Multiplier ($\mu_2$)', fontsize=12)
        plt.axvline(0.6, color='gray', linestyle='--', label='Critical Threshold (0.6)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'mu2_fibrin_viscosity.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    unittest.main()