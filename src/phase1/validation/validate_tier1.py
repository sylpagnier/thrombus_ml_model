import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from src.config import PhysicsConfig

# --- Path Setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.physics_kernels import PhysicsKernels


class Tier1Validator:
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        phys_cfg = PhysicsConfig()
        self.kernels = PhysicsKernels(phys_cfg)

        print(f"⚡ Loading Model: {model_path}")
        self.model = GINO_DEQ(in_channels=11, out_channels=3, latent_dim=64, max_iters=15)
        # Load weights safely for CPU/GPU compatibility
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _compute_wss_proxy(self, pred, data, props):
        """
        Calculates accurate WSS magnitude: tau = mu_eff * gamma_dot.
        Correctly handles Non-Newtonian viscosity variations.
        """
        u, v = pred[:, 0], pred[:, 1]

        # 1. Compute full gradient tensor
        # We need all 4 components to calculate the true strain rate invariant
        c_u = self.kernels._compute_derivatives(u.unsqueeze(1), props)
        c_v = self.kernels._compute_derivatives(v.unsqueeze(1), props)

        u_x, u_y = c_u[:, 0, 0], c_u[:, 1, 0]
        v_x, v_y = c_v[:, 0, 0], c_v[:, 1, 0]

        # 2. Compute Generalized Shear Rate (gamma_dot)
        # Using the second invariant of the rate of strain tensor (same as PhysicsKernels)
        # gamma_dot = sqrt(2 * D : D)
        gamma_dot = torch.sqrt(2 * u_x ** 2 + 2 * v_y ** 2 + (u_y + v_x) ** 2 + 1e-8)

        # 3. Compute Effective Viscosity (mu_eff)
        # We differentiate between Newtonian and Carreau modes here
        if self.kernels.cfg.viscosity_model == "carreau":
            # Extract scalar reference params from the data batch
            u_ref = data.u_ref.item() if hasattr(data, 'u_ref') else 1.0
            d_bar = data.d_bar.item() if hasattr(data, 'd_bar') else 1.0

            # Stack gradients for the kernel function [u_x, u_y, v_x, v_y]
            du_ij = torch.stack([u_x, u_y, v_x, v_y], dim=1)

            # Reuse the EXACT viscosity logic from training
            mu_eff = self.kernels._compute_carreau_viscosity(du_ij, u_ref, d_bar)
        else:
            # Newtonian: ND viscosity is constant (1.0 relative to mu_ref)
            mu_eff = torch.ones_like(gamma_dot)

        # 4. Compute Wall Shear Stress (tau = mu * gamma_dot)
        # This captures the non-linear stress drop in plug-flow regions
        wss = mu_eff * gamma_dot

        # Filter for Wall Nodes only
        mask = data.mask_wall.bool()
        if mask.sum() == 0:
            return torch.tensor([]), mask

        return wss[mask], mask

    def validate_dataset(self, data_dir, level_name="Unknown"):
        path = project_root / data_dir
        files = list(path.glob("*.pt"))
        if not files:
            print(f"⚠️ No files found in {path}")
            return None

        # Load dataset
        dataset = [torch.load(f, weights_only=False) for f in files]
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Initialize lists for metrics
        metrics = {
            "rel_l2_u": [],
            "div_residual": [],
            "wall_slip": [],
            "wss_corr": []
        }

        print(f"\n🔍 Validating {level_name} (N={len(dataset)})...")

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                data = data.to(self.device)
                pred = self.model(data)

                # --- 1. Physics Metrics (Calculated for EVERY sample) ---
                props = self.kernels._get_geometric_props(data)

                # Mass Conservation (Div U)
                grads_u = self.kernels._compute_gradients(pred[:, 0:1], props)
                grads_v = self.kernels._compute_gradients(pred[:, 1:2], props)
                div = torch.abs(grads_u[:, 0] + grads_v[:, 1]).mean()
                metrics["div_residual"].append(div.item())

                # Wall Slip
                if data.mask_wall.any():
                    slip = torch.norm(pred[data.mask_wall, :2], dim=1).mean()
                    metrics["wall_slip"].append(slip.item())
                else:
                    metrics["wall_slip"].append(0.0)

                # --- 2. Supervised Metrics (Handle Missing Labels) ---
                # Check if 'y' exists and is valid (not empty/all-zeros)
                has_labels = (hasattr(data, 'y') and
                              data.y is not None and
                              data.y.shape[0] == data.x.shape[0] and
                              data.y.abs().sum() > 1e-6)

                if has_labels:
                    target = data.y
                    # Relative L2 Error
                    diff = torch.norm(pred[:, :2] - target[:, :2], dim=1)
                    denom = torch.norm(target[:, :2], dim=1) + 1e-6
                    rel_l2 = diff.sum() / denom.sum()
                    metrics["rel_l2_u"].append(rel_l2.item())

                    # WSS Correlation
                    pred_wss, _ = self._compute_wss_proxy(pred, data, props)
                    gt_wss, _ = self._compute_wss_proxy(target, data, props)

                    if len(pred_wss) > 5:
                        vx = pred_wss - pred_wss.mean()
                        vy = gt_wss - gt_wss.mean()
                        # Pearson Correlation
                        corr = torch.sum(vx * vy) / (
                                    torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
                        metrics["wss_corr"].append(corr.item())
                    else:
                        metrics["wss_corr"].append(np.nan)
                else:
                    # FILLER: Append NaNs so lists stay same length
                    metrics["rel_l2_u"].append(np.nan)
                    metrics["wss_corr"].append(np.nan)

                # --- 3. Save Visualization (First 5 samples) ---
                if i < 5:
                    self._plot_comparison(data, pred, data.y if has_labels else None, f"{level_name}_sample_{i}")

        # Summary
        df = pd.DataFrame(metrics)
        print(f"📊 {level_name} Results (Excluding Failed Solves):")
        # .describe() automatically ignores NaNs
        print(df.describe().loc[['mean', 'std', 'count']])

        # Return mean of valid entries
        return df.mean()

    def _plot_comparison(self, data, pred, target, title):
        """Saves a comparison plot: Pred vs GT vs Error."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        pos = data.x[:, :2].cpu().numpy()
        u_pred = pred[:, 0].cpu().numpy()

        # 1. Prediction
        sc1 = axes[0].tripcolor(pos[:, 0], pos[:, 1], u_pred, cmap='jet')
        axes[0].set_title(f"Predicted Velocity (u)\n{title}")
        plt.colorbar(sc1, ax=axes[0])
        axes[0].set_aspect('equal')
        axes[0].axis('off')

        # 2. Ground Truth & Error
        if target is not None:
            u_gt = target[:, 0].cpu().numpy()
            error = np.abs(u_pred - u_gt)

            sc2 = axes[1].tripcolor(pos[:, 0], pos[:, 1], u_gt, cmap='jet')
            axes[1].set_title("Ground Truth (COMSOL)")
            plt.colorbar(sc2, ax=axes[1])

            sc3 = axes[2].tripcolor(pos[:, 0], pos[:, 1], error, cmap='inferno')
            axes[2].set_title("Absolute Error |Pred - GT|")
            plt.colorbar(sc3, ax=axes[2])
        else:
            axes[1].text(0.5, 0.5, "COMSOL Solver Failed\n(NaN Output)", ha='center', va='center')
            axes[2].text(0.5, 0.5, "N/A", ha='center', va='center')

        for ax in axes[1:]:
            ax.set_aspect('equal')
            ax.axis('off')

        # Save
        save_dir = project_root / "reports/validation_tier1"
        save_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize title for filename
        safe_title = title.replace("/", "_").replace(" ", "_")
        plt.savefig(save_dir / f"{safe_title}.png", dpi=150, bbox_inches='tight')
        plt.close()