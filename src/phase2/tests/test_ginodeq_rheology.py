import unittest
import torch
from src.phase2.ginodeq_tier3 import GINO_DEQ_Tier3


class TestDualTriggerRheology(unittest.TestCase):

    def setUp(self):
        """Initialize the Tier 3 GINO DEQ model with default parameters."""
        # Using a small latent_dim/channels just to instantiate the class quickly for unit testing
        self.model = GINO_DEQ_Tier3(in_channels=12, latent_dim=16)

    def test_rheology_thresholds(self):
        """
        Verify that the model initializes with the correct COMSOL critical
        thresholds for Platelet (Mat) and Fibrin (FI) driven viscosity.
        """
        # COMSOL Step 3 (mu1): Location 2E7
        self.assertEqual(self.model.mat_crit, 2e7)
        # COMSOL Step 5 (mu2): Location 0.6
        self.assertEqual(self.model.fi_crit, 0.6)
        # Updated Physical Max
        self.assertEqual(self.model.mu_ratio_max, 7000.0)

    def test_mu1_sigmoid_platelet_logic(self):
        """
        Test the soft step-function for Platelet-driven viscosity (mu1).
        It should be ~0 well below the 2e7 threshold, and ~7000 well above it.
        """
        # Tensor simulating surface platelet concentration (Mat)
        mat_low = torch.tensor([1e6], dtype=torch.float32)  # Well below 2e7
        mat_thresh = torch.tensor([2e7], dtype=torch.float32)  # Exactly at threshold
        mat_high = torch.tensor([4e7], dtype=torch.float32)  # Well above 2e7

        mu1_low = self.model.mu1_sigmoid(mat_low).item()
        mu1_thresh = self.model.mu1_sigmoid(mat_thresh).item()
        mu1_high = self.model.mu1_sigmoid(mat_high).item()

        # Soft sigmoid at exactly threshold should equal max / 2 (7000 / 2 = 3500)
        self.assertAlmostEqual(mu1_thresh, 3500.0, places=2)

        # Assert limits bounded by mu_ratio_max
        self.assertTrue(mu1_low < 1.0, "Viscosity multiplier should be near 0 at low Mat")
        self.assertTrue(mu1_high > 6990.0, "Viscosity multiplier should be near 7000 at high Mat")

    def test_mu2_sigmoid_fibrin_logic(self):
        """
        Test the soft step-function for Fibrin-driven viscosity (mu2).
        It should be ~0 well below the 0.6 threshold, and ~7000 well above it.
        """
        # Tensor simulating bulk Fibrin concentration (FI)
        fi_low = torch.tensor([0.1], dtype=torch.float32)  # Well below 0.6
        fi_thresh = torch.tensor([0.6], dtype=torch.float32)  # Exactly at threshold
        fi_high = torch.tensor([1.2], dtype=torch.float32)  # Well above 0.6

        mu2_low = self.model.mu2_sigmoid(fi_low).item()
        mu2_thresh = self.model.mu2_sigmoid(fi_thresh).item()
        mu2_high = self.model.mu2_sigmoid(fi_high).item()

        # Soft sigmoid at exactly threshold should equal max / 2 (7000 / 2 = 3500)
        self.assertAlmostEqual(mu2_thresh, 3500.0, places=2)

        # Assert limits
        self.assertTrue(mu2_low < 1.0, "Viscosity multiplier should be near 0 at low Fibrin")
        self.assertTrue(mu2_high > 6990.0, "Viscosity multiplier should be near 7000 at high Fibrin")


if __name__ == '__main__':
    unittest.main()