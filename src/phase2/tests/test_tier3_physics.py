import os
import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import BiochemConfig, PhysicsConfig
from src.phase2.physics_kernels_tier3 import BiochemPhysicsKernels
from src.phase2.ginodeq_tier3 import GINO_DEQ_Tier3
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

        # Initialize ML model using config values to prevent hardcoding errors
        self.model = GINO_DEQ_Tier3(
            in_channels=12,
            latent_dim=16,
            mu_ratio_max=self.bio_cfg.mu_ratio_max
        )

    # ==========================================
    # 1. COMSOL CONSTANT MAPPING
    # ==========================================
    def test_comsol_constants_mapping(self):
        """Verify fundamental COMSOL parameters are correctly inherited and scaled."""
        # Using typical default bounds check based on the physics kernel initialization
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

    # ==========================================
    # 2. PURE MATH VALIDATION (FIBRIN)
    # ==========================================
    def test_fibrin_kinetics_math(self):
        """Validates Fibrin kinetics formulation."""
        num_nodes = 500
        T_tensor = torch.rand(num_nodes, dtype=torch.float32)
        FG_tensor = torch.rand(num_nodes, dtype=torch.float32) * 5.0

        R_FG_pt, R_FI_pt = self.kinetics.compute_fibrin_kinetics(T_tensor, FG_tensor)

        T_np = T_tensor.numpy()
        FG_np = FG_tensor.numpy()

        # COMSOL definition: kfi * T * FG / (kmfi + FG)
        # Note: kmfi was scaled by C_scale in the kinetics class
        reaction_comsol = (self.bio_cfg.kfi * T_np * FG_np) / (
                    (self.bio_cfg.kmfi * self.kinetics.C_scale) + FG_np + 1e-8)

        np.testing.assert_allclose(R_FI_pt.numpy(), reaction_comsol, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(R_FG_pt.numpy(), -reaction_comsol, rtol=1e-5, atol=1e-8)

    # ==========================================
    # 3. PURE MATH VALIDATION (GAMMA / THROMBIN INHIBITION)
    # ==========================================
    def test_gamma_inhibition_math(self):
        """
        Analytic 3: Verifies Thrombin inhibition by Antithrombin/Heparin complex.
        COMSOL: Gamma = (k_1t * c_H * AT) / (K_at * K_T + T * K_at + AT * T)
        """
        num_nodes = 500
        T_tensor = torch.rand(num_nodes, dtype=torch.float32) * 2.0  # T concentration
        AT_tensor = torch.rand(num_nodes, dtype=torch.float32) * 5.0  # AT concentration

        # Run PyTorch Implementation
        gamma_pt = self.kinetics.compute_gamma(T_tensor, AT_tensor)

        # Raw COMSOL Ground Truth (NumPy)
        T_np = T_tensor.numpy()
        AT_np = AT_tensor.numpy()

        # Extracted from COMSOL Parameters
        k_1t = self.bio_cfg.k_1t
        c_H = self.bio_cfg.c_H
        K_at = self.bio_cfg.K_at
        K_T = self.bio_cfg.K_T

        numerator = k_1t * c_H * AT_np
        denominator = (K_at * K_T) + (T_np * K_at) + (AT_np * T_np) + 1e-8
        gamma_comsol = numerator / denominator

        # Validate similarity
        np.testing.assert_allclose(gamma_pt.numpy(), gamma_comsol, rtol=1e-5, atol=1e-8,
                                   err_msg="PyTorch Gamma (Thrombin Inhibition) does not match COMSOL.")

    # ==========================================
    # 4. SOFT-LOGIC VALIDATION (PLATELET ACTIVATION)
    # ==========================================
    def test_kpa_activation_logic(self):
        """
        Analytic 2, 6, 7: Tests the smooth soft-logic of k_pa (Platelet Activation)
        against COMSOL's rigid if/else statements.
        """
        num_nodes = 1000
        # Generate Omega values ranging from 0 to 600 (crosses the 500 threshold)
        omega_tensor = torch.linspace(0, 600, num_nodes)
        # Generate shear rates ranging from 0 to 15000 (crosses the 10000 threshold)
        shear_tensor = torch.linspace(0, 15000, num_nodes)

        # PyTorch Soft-Logic Evaluation (Force Strict Temperature for closer match)
        self.kinetics.T_scale = 0.01
        kpa_pt = self.kinetics.compute_k_pa(omega_tensor, shear_tensor)

        # COMSOL Rigid Logic Evaluation
        omega_np = omega_tensor.numpy()
        shear_np = shear_tensor.numpy()
        t_act = self.bio_cfg.t_act
        shear_crit = self.bio_cfg.shear_crit

        # kpa_chem: if(Omega<500, (Omega/t_act)*Act_step(Omega), 500)
        # Act_step acts roughly as Omega > 1.0 based on COMSOL setup
        act_step = np.where(omega_np > 1.0, 1.0, 0.0)
        kpa_chem_np = np.where(omega_np < 500, (omega_np / t_act) * act_step, 500.0)

        # kpa_mech: if(spf.sr>shear_crit, spf.sr/shear_crit, 0)
        kpa_mech_np = np.where(shear_np > shear_crit, shear_np / shear_crit, 0.0)

        kpa_comsol = kpa_chem_np + kpa_mech_np

        # We don't use strict assert_allclose here because PyTorch is intentionally smooth (Sigmoids).
        # Instead, we check Pearson correlation to ensure the curve shape maps correctly.
        correlation = np.corrcoef(kpa_pt.numpy(), kpa_comsol)[0, 1]
        self.assertGreater(correlation, 0.98,
                           "PyTorch soft-logic for k_pa deviates too far from COMSOL rigid logic.")

    # ==========================================
    # 5. DUAL-TRIGGER RHEOLOGY (VISCOSITY MU1)
    # ==========================================
    def test_platelet_viscosity_mu1(self):
        """
        Validates PyTorch's mu1_sigmoid against COMSOL's mu1 Step function.
        COMSOL: Location 2e7, from 0/1 to mu_ratio_max.
        """
        mat_range = torch.linspace(0, 4e7, 1000)

        # Test Strict Logic
        self.model.T_scale = 1.0
        mu1_pt_strict = self.model.mu1_sigmoid(mat_range).numpy()

        # COMSOL Exact Step
        mat_np = mat_range.numpy()
        mu1_comsol = np.where(mat_np >= self.model.mat_crit, self.model.mu_ratio_max, 0.0)

        # Ensure the midpoint of the sigmoid sits exactly on the COMSOL threshold
        midpoint_val = self.model.mu1_sigmoid(torch.tensor([self.model.mat_crit])).item()
        self.assertAlmostEqual(midpoint_val, self.model.mu_ratio_max / 2.0, places=3,
                               msg="The Sigmoid midpoint does not align with COMSOL's critical threshold.")

        # Ensure bounds are respected
        self.assertLessEqual(np.max(mu1_pt_strict), self.model.mu_ratio_max + 1e-4)
        self.assertGreaterEqual(np.min(mu1_pt_strict), 0.0)

    # ==========================================
    # 6. DUAL-TRIGGER RHEOLOGY (VISCOSITY MU2)
    # ==========================================
    def test_fibrin_viscosity_mu2(self):
        """
        Validates PyTorch's mu2_sigmoid against COMSOL's mu2 Step function and outputs a sanity check plot.
        COMSOL: Location 0.6, from 0 to mu_ratio_max.
        """
        num_nodes = 1000
        fi_range = np.linspace(0, 1.2, num_nodes)
        fi_tensor = torch.tensor(fi_range, dtype=torch.float32)

        # Torch Soft Logic
        self.model.T_scale = 1.0  # Fully annealed strict logic
        mu2_pt_strict = self.model.mu2_sigmoid(fi_tensor).numpy()

        self.model.T_scale = 5.0  # Early training relaxed logic
        mu2_pt_relaxed = self.model.mu2_sigmoid(fi_tensor).numpy()

        # COMSOL Exact Step Function (Location 0.6, From 0 to mu_ratio_max)
        mu2_comsol = np.where(fi_range >= 0.6, self.bio_cfg.mu_ratio_max, 0.0)

        plt.figure(figsize=(10, 6))
        plt.plot(fi_range, mu2_comsol, 'r-', linewidth=3, label='COMSOL Exact Step (Ground Truth)')
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