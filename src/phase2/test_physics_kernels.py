import unittest
import torch
from src.config import BiochemConfig
from src.phase2.physics_kernels_tier3 import BiochemPhysicsKernels


class DummyCoreKernels:
    """
    Mocks the base CFD physics kernels strictly for testing the
    Biochemical ADR (Advection-Diffusion-Reaction) equations.
    """

    def _compute_derivatives(self, tensor, spatial_props):
        # Returns zero-gradients matching expected shape [..., 2] for (x,y)
        shape = list(tensor.shape) + [2]
        return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)


class TestBiochemPhysicsKernels(unittest.TestCase):

    def setUp(self):
        """Initialize using the real configuration to ensure accurate data flow."""
        # Use BiochemConfig instead of VesselConfig to access kinetic constants
        self.cfg = BiochemConfig(tier="tier3")
        self.core = DummyCoreKernels()

        self.biochem_kernels = BiochemPhysicsKernels(self.cfg, self.core)
        self.kinetics = self.biochem_kernels.kinetics

    def test_comsol_constants_mapping(self):
        """
        Verify that the fundamental COMSOL parameters from config.py
        are correctly inherited and accessible by the kinetics class.
        """
        # Testing against the real config values injected into the kernels
        self.assertEqual(self.cfg.kfi, 59.0)
        self.assertEqual(self.cfg.kmfi, 3.16)

        # Checking hardcoded critical thresholds inside the kinetics logic
        expected_constants = {
            'APScrit': 0.6,
            'APRcrit': 2.0,
            'Tcrit': 5.0e-4,  # Updated to match config.py (5.0e-4 instead of 0.0005 makes no difference mathematically, but good for consistency)
            't_act': 1.0,
            'shear_crit': 10000.0
        }

        for attr, expected_val in expected_constants.items():
            actual_val = getattr(self.kinetics, attr)
            self.assertAlmostEqual(
                actual_val, expected_val,
                msg=f"COMSOL constant {attr} mismatch: Expected {expected_val}, got {actual_val}"
            )

    def test_diffusion_coefficients(self):
        """Verify diffusion coefficients map directly to the COMSOL Parameters."""
        expected_D = {
            'RP': 1.58e-9, 'AP': 1.58e-9,
            'APR': 2.57e-6, 'APS': 2.14e-6,
            'T': 4.16e-7, 'AT': 3.49e-7,
            'FG': 3.10e-7, 'FI': 2.47e-7
        }

        for species, val in expected_D.items():
            self.assertEqual(
                self.biochem_kernels.D_coeff[species], val,
                f"Diffusion coefficient mismatch for {species}"
            )

    def test_compute_omega_analytic1(self):
        """
        Test COMSOL Analytic 1 (Omega).
        Formula: (APS/APScrit) + (APR/APRcrit) + (T/Tcrit)
        """
        APR = torch.tensor([2.0])  # 2.0 / 2.0 = 1.0
        APS = torch.tensor([0.3])  # 0.3 / 0.6 = 0.5
        T = torch.tensor([0.00025])  # 0.00025 / 0.0005 = 0.5

        # Expected Omega = 1.0 + 0.5 + 0.5 = 2.0
        omega = self.kinetics.compute_omega(APR, APS, T)
        self.assertAlmostEqual(omega.item(), 2.0, places=4)

    def test_fibrin_kinetics_rate(self):
        """
        Test Fibrin/Fibrinogen source/sink terms against the COMSOL formula
        using the actual constants from config.py.
        Formula: reaction_rate = (kfi * T * FG) / (kmfi + FG)
        """
        T = torch.tensor([1.0])
        FG = torch.tensor([7.0])

        # Calculate expected based on live config state, not hardcoded floats
        expected_rate = (self.cfg.kfi * 1.0 * 7.0) / (self.cfg.kmfi + 7.0)

        R_FG, R_FI = self.kinetics.compute_fibrin_kinetics(T, FG)

        # FG is consumed (-), FI is produced (+)
        self.assertAlmostEqual(R_FG.item(), -expected_rate, places=4)
        self.assertAlmostEqual(R_FI.item(), expected_rate, places=4)


if __name__ == '__main__':
    unittest.main()