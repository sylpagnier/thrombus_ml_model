import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# --- Path Setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

from src.phase1.physics.ginodeq import GINO_DEQ
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.config import PhysicsConfig


class ModelValidator:
    def __init__(self, model_path, tier="tier1", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tier = tier

        # Load the correct config based on the tier
        phys_cfg = PhysicsConfig(tier=self.tier)
        self.kernels = PhysicsKernels(phys_cfg)

        print(f"⚡ Loading {self.tier.capitalize()} Model: {model_path}")

        # CRITICAL FIX: in_channels=13 to account for the new Generalized Poiseuille Prior (uv_prior)
        self.model = GINO_DEQ(in_channels=13, out_channels=4, latent_dim=64, max_iters=15)

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _compute_wss_proxy(self, u_x, u_y, v_x, v_y, mu_eff, data):
        """
        Calculates Wall Shear Stress (WSS): tau = mu_eff * gamma_dot.
        """
        # Compute Generalized Shear Rate (gamma_dot)
        gamma_dot = torch.sqrt(2 * u_x ** 2 + 2 * v_y ** 2 + (u_y + v_x) ** 2 + 1e-8)

        # Compute Wall Shear Stress
        wss = mu_eff * gamma_dot

        # Filter for Wall Nodes only
        mask = data.mask_wall.bool()
        if mask.sum() == 0:
            return torch.tensor([]), mask

        return wss[mask], mask

    def validate_dataset(self, data_dir, level_name="Unknown"):
        # CRITICAL FIX: Handle absolute paths correctly when passed from run_benchmark.py
        data_path = Path(data_dir)
        path = data_path if data_path.is_absolute() else project_root / data_dir

        files = list(path.glob("*.pt"))
        if not files:
            print(f"⚠️ No files found in {path}")
            return None

        dataset = [torch.load(f, weights_only=False) for f in files]
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        metrics = {
            "rel_l2_u": [],
            "rel_l2_mu": [],
            "div_residual": [],
            "wall_slip": [],
            "wss_corr": []
        }

        print(f"\n🔍 Validating {self.tier.capitalize()} - {level_name} (N={len(dataset)})...")

        with torch.no_grad():
            for i, data in enumerate(tqdm(loader)):
                data = data.to(self.device)
                pred = self.model(data)

                # --- 1. Physics Metrics ---
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

                # --- 2. Supervised Metrics ---
                has_labels = (hasattr(data, 'y') and data.y is not None and
                              data.y.shape[0] == data.x.shape[0] and data.y.abs().sum() > 1e-6)

                if has_labels:
                    target = data.y

                    # Velocity L2 Error
                    diff_u = torch.norm(pred[:, :2] - target[:, :2], dim=1)
                    denom_u = torch.norm(target[:, :2], dim=1) + 1e-6
                    metrics["rel_l2_u"].append((diff_u.sum() / denom_u.sum()).item())

                    # Viscosity L2 Error (Important for Tier 2)
                    diff_mu = torch.abs(pred[:, 3] - target[:, 3])
                    denom_mu = torch.abs(target[:, 3]) + 1e-6
                    metrics["rel_l2_mu"].append((diff_mu.sum() / denom_mu.sum()).item())

                    # WSS Correlation
                    # Extract target gradients for GT WSS
                    c_u_gt = self.kernels._compute_derivatives(target[:, 0].unsqueeze(1), props)
                    c_v_gt = self.kernels._compute_derivatives(target[:, 1].unsqueeze(1), props)

                    pred_wss, _ = self._compute_wss_proxy(grads_u[:, 0], grads_u[:, 1], grads_v[:, 0], grads_v[:, 1],
                                                          pred[:, 3], data)
                    gt_wss, _ = self._compute_wss_proxy(c_u_gt[:, 0, 0], c_u_gt[:, 1, 0], c_v_gt[:, 0, 0],
                                                        c_v_gt[:, 1, 0], target[:, 3], data)

                    if len(pred_wss) > 5:
                        vx = pred_wss - pred_wss.mean()
                        vy = gt_wss - gt_wss.mean()
                        corr = torch.sum(vx * vy) / (
                                torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
                        metrics["wss_corr"].append(corr.item())
                    else:
                        metrics["wss_corr"].append(np.nan)
                else:
                    metrics["rel_l2_u"].append(np.nan)
                    metrics["rel_l2_mu"].append(np.nan)
                    metrics["wss_corr"].append(np.nan)

                # --- 3. Save Visualization ---
                if i < 5:
                    self._plot_comparison(data, pred, data.y if has_labels else None, f"{level_name}_sample_{i}")

        df = pd.DataFrame(metrics)
        print(f"📊 {level_name} Results:")
        print(df.describe().loc[['mean', 'std', 'count']])

        return df.mean()

    def _plot_comparison(self, data, pred, target, title):
        # Dynamically size plot based on tier (show viscosity for Tier 2)
        rows = 2 if self.tier == "tier2" else 1
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        if rows == 1: axes = np.array([axes])

        pos = data.x[:, :2].cpu().numpy()

        # ROW 1: VELOCITY
        u_pred = pred[:, 0].cpu().numpy()
        sc1 = axes[0, 0].scatter(pos[:, 0], pos[:, 1], c=u_pred, cmap='jet', s=5, edgecolor='none')
        axes[0, 0].set_title(f"Predicted Velocity (u)\n{title}")
        plt.colorbar(sc1, ax=axes[0, 0])

        if target is not None:
            u_gt = target[:, 0].cpu().numpy()
            error_u = np.abs(u_pred - u_gt)
            sc2 = axes[0, 1].tripcolor(pos[:, 0], pos[:, 1], u_gt, cmap='jet')
            axes[0, 1].set_title("Ground Truth (u)")
            plt.colorbar(sc2, ax=axes[0, 1])
            sc3 = axes[0, 2].tripcolor(pos[:, 0], pos[:, 1], error_u, cmap='inferno')
            axes[0, 2].set_title("Error |Pred - GT|")
            plt.colorbar(sc3, ax=axes[0, 2])
        else:
            axes[0, 1].text(0.5, 0.5, "GT Missing", ha='center', va='center')
            axes[0, 2].text(0.5, 0.5, "N/A", ha='center', va='center')

        # ROW 2: VISCOSITY (Tier 2 only)
        if self.tier == "tier2":
            mu_pred = pred[:, 3].cpu().numpy()
            sc4 = axes[1, 0].tripcolor(pos[:, 0], pos[:, 1], mu_pred, cmap='viridis')
            axes[1, 0].set_title("Predicted Viscosity (mu)")
            plt.colorbar(sc4, ax=axes[1, 0])

            if target is not None:
                mu_gt = target[:, 3].cpu().numpy()
                error_mu = np.abs(mu_pred - mu_gt)
                sc5 = axes[1, 1].tripcolor(pos[:, 0], pos[:, 1], mu_gt, cmap='viridis')
                axes[1, 1].set_title("Ground Truth (mu)")
                plt.colorbar(sc5, ax=axes[1, 1])
                sc6 = axes[1, 2].tripcolor(pos[:, 0], pos[:, 1], error_mu, cmap='inferno')
                axes[1, 2].set_title("Viscosity Error")
                plt.colorbar(sc6, ax=axes[1, 2])
            else:
                axes[1, 1].text(0.5, 0.5, "GT Missing", ha='center', va='center')
                axes[1, 2].text(0.5, 0.5, "N/A", ha='center', va='center')

        for ax in axes.flatten():
            ax.set_aspect('equal')
            ax.axis('off')

        save_dir = project_root / f"reports/validation_{self.tier}"
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_title = title.replace("/", "_").replace(" ", "_")
        plt.savefig(save_dir / f"{safe_title}.png", dpi=150, bbox_inches='tight')
        plt.close()