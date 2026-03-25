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
        shape = list(tensor.shape) + [2]
        return torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)


class TestBiochemPhysicsKernels(unittest.TestCase):

    def setUp(self):
        """Initialize using the real configuration to ensure accurate data flow."""
        self.cfg = BiochemConfig(tier="tier3")
        self.core = DummyCoreKernels()

        self.biochem_kernels = BiochemPhysicsKernels(self.cfg, self.core)
        self.kinetics = self.biochem_kernels.kinetics

        self.D_scale = self.biochem_kernels.D_scale
        self.C_scale = self.kinetics.C_scale

    def test_comsol_constants_mapping(self):
        """Verify fundamental COMSOL parameters are correctly inherited and scaled."""
        self.assertEqual(self.cfg.kfi, 59.0)
        self.assertEqual(self.cfg.kmfi, 3.16e-3)

        expected_constants = {
            'APScrit': self.cfg.APScrit * self.C_scale,
            'APRcrit': self.cfg.APRcrit * self.C_scale,
            'Tcrit': self.cfg.Tcrit * self.C_scale,
            't_act': self.cfg.t_act,
            'shear_crit': self.cfg.shear_crit
        }

        for attr, expected_val in expected_constants.items():
            actual_val = getattr(self.kinetics, attr)
            self.assertAlmostEqual(actual_val, expected_val, places=6)

    def test_diffusion_coefficients(self):
        """Verify diffusion coefficients map directly to COMSOL Parameters and apply D_scale."""
        expected_D = {
            'RP': self.cfg.D_RP * self.D_scale,
            'AP': self.cfg.D_AP * self.D_scale,
            'APR': self.cfg.D_APR * self.D_scale,
            'APS': self.cfg.D_APS * self.D_scale,
            'T': self.cfg.D_T * self.D_scale,
            'AT': self.cfg.D_AT * self.D_scale,
            'FG': self.cfg.D_FG * self.D_scale,
            'FI': self.cfg.D_FI * self.D_scale
        }

        for species, val in expected_D.items():
            self.assertEqual(self.biochem_kernels.D_coeff[species], val)

    def test_compute_omega_analytic1(self):
        """Test COMSOL Analytic 1 (Omega)."""
        APR = torch.tensor([2.0])
        APS = torch.tensor([0.3])
        T = torch.tensor([0.00025])

        # Based on default test config expected scaling
        omega = self.kinetics.compute_omega(APR, APS, T)
        self.assertAlmostEqual(omega.item(), 2.0, places=4)

    def test_fibrin_kinetics_rate(self):
        """Test Fibrin/Fibrinogen source/sink terms against the COMSOL formula."""
        T = torch.tensor([1.0])
        FG = torch.tensor([7.0])

        scaled_kmfi = self.cfg.kmfi * self.C_scale
        expected_rate = (self.cfg.kfi * 1.0 * 7.0) / (scaled_kmfi + 7.0)

        R_FG, R_FI = self.kinetics.compute_fibrin_kinetics(T, FG)

        self.assertAlmostEqual(R_FG.item(), -expected_rate, places=4)
        self.assertAlmostEqual(R_FI.item(), expected_rate, places=4)

    def test_agonist_release_dimensionality(self):
        """
        Verify that ADP and TxA2 are computed using correct dimensions.
        ADP should scale by activation rate (R_AP).
        TxA2 should scale by active concentration (AP).
        """
        species_dict = {
            'RP': torch.tensor([100.0]),
            'AP': torch.tensor([50.0]),  # Non-zero active platelets for continuous synthesis
            'APR': torch.tensor([0.0]),
            'APS': torch.tensor([0.0]),
            'T': torch.tensor([0.0]),
            'AT': torch.tensor([0.0]),
            'FG': torch.tensor([0.0]),
            'FI': torch.tensor([0.0])
        }

        # Inject an arbitrary shear rate to trigger activation
        shear_rate = torch.tensor([100.0])

        reactions = self.kinetics.compute_species_reactions(species_dict, shear_rate)

        # Retrieve the activation rate calculated inside the step
        R_AP_actual = reactions['AP'].item()

        # Calculate expected based on fixed formulas
        scale_release = 1e9
        lambda_apr = self.cfg.lambda_adp * scale_release
        s_t_aps = self.cfg.s_t * scale_release

        expected_R_APR = lambda_apr * R_AP_actual
        expected_R_APS = (s_t_aps * 50.0) - (self.cfg.k_i * 0.0)

        self.assertAlmostEqual(reactions['APR'].item(), expected_R_APR, places=4)
        self.assertAlmostEqual(reactions['APS'].item(), expected_R_APS, places=4)


if __name__ == '__main__':
    unittest.main()