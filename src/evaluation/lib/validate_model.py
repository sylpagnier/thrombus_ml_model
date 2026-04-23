import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from src.utils.paths import get_project_root, reports_evaluation_dir
from src.architecture.ginodeq import GINO_DEQ
from src.core_physics.physics_kernels import PhysicsKernels
from src.config import PhysicsConfig, PredChannels

project_root = get_project_root()


class ModelValidator:
    def __init__(self, model_path, tier="tier1", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tier = tier

        phys_cfg = PhysicsConfig(tier=self.tier)
        self.kernels = PhysicsKernels(phys_cfg)

        print(f"⚡ Loading {self.tier.capitalize()} Model: {model_path}")

        mu_inf_nd = self.kernels.mu_inf_nd
        mu_0_nd = self.kernels.mu_0_nd

        # Mirror training-time model recipes so checkpoint loading is shape-compatible.
        if self.tier == "tier1":
            self.model = GINO_DEQ(
                in_channels=15,
                out_channels=5,
                latent_dim=256,
                max_iters=25,
                num_fourier_freqs=16,
                phys_cfg=phys_cfg,
                activation_fn="silu",
                fourier_base=1.5,
                use_hard_bcs=True,
                num_global_tokens=16,
                use_siren_decoder=True,
                use_width_priors=True,
                mu_inf_nd=mu_inf_nd,
                mu_0_nd=mu_0_nd,
            )
        else:
            self.model = GINO_DEQ(
                in_channels=15,
                out_channels=5,
                latent_dim=64,
                max_iters=15,
                phys_cfg=phys_cfg,
                mu_inf_nd=mu_inf_nd,
                mu_0_nd=mu_0_nd,
            )

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _predict_with_physics_correction(self, data, correction_steps=25, lr=1e-3):
        """
        Inference-Time Physics Correction (ITPC) loop.
        """
        # 1. Base Prediction (Smart Guess)
        with torch.no_grad():
            base_pred = self.model(data, solver="anderson")

        # 2. Optimization Setup
        pred_opt = base_pred.detach().clone()
        pred_opt.requires_grad_(True)
        optimizer = optim.Adam([pred_opt], lr=lr)

        props = self.kernels._get_geometric_props(data)

        for step in range(correction_steps):
            optimizer.zero_grad()

            # --- Physics Residuals ---
            l_mom = self.kernels.navier_stokes_residual(pred_opt, data, props=props)

            c_u = self.kernels._compute_derivatives(pred_opt[:, 0:1], props)
            c_v = self.kernels._compute_derivatives(pred_opt[:, 1:2], props)
            du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)
            l_cont = self.kernels.continuity_loss(du_ij, data=data)

            l_bc = self.kernels.boundary_condition_loss(pred_opt, data)
            l_io = self.kernels.inlet_outlet_loss(pred_opt, data)

            loss = l_mom + (10.0 * l_cont) + (50.0 * l_bc) + (10.0 * l_io)

            loss.backward()
            optimizer.step()

            # Hard No-Slip Enforcement
            with torch.no_grad():
                mask_wall = data.mask_wall.view(-1).bool()
                if mask_wall.any():
                    pred_opt[mask_wall, 0:2] = 0.0

        return base_pred.detach(), pred_opt.detach()

    def validate_dataset(self, data_dir, level_name="Unknown", save_comparison_images=True):
        data_path = Path(data_dir)
        path = data_path if data_path.is_absolute() else project_root / data_dir

        files = list(path.glob("*.pt"))
        if not files:
            print(f"⚠️ No files found in {path}")
            return None

        dataset = [torch.load(f, weights_only=False) for f in files]
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        use_itpc = (self.tier == "tier2")
        metrics = {
            "base_rel_l2_u": [],
            "base_div_res": [],
            "base_wall_slip": [],
            "base_wss_corr": [],
        }
        if use_itpc:
            metrics.update(
                {
                    "itpc_rel_l2_u": [],
                    "itpc_div_res": [],
                    "itpc_wall_slip": [],
                    "itpc_wss_corr": [],
                }
            )

        print(f"\n🔍 A/B Testing {self.tier.capitalize()} - {level_name} (N={len(dataset)})...")

        for i, data in enumerate(tqdm(loader)):
            data = data.to(self.device)

            if use_itpc:
                pred_base, pred_itpc = self._predict_with_physics_correction(data, correction_steps=20)
            else:
                with torch.no_grad():
                    pred_base = self.model(data, solver="anderson")
                pred_itpc = None
            props = self.kernels._get_geometric_props(data)
            has_labels = (hasattr(data, 'y') and data.y is not None and data.y.abs().sum() > 1e-6)

            def evaluate_prediction(pred, prefix):
                # 1. Physics Metrics
                grads_u = self.kernels._compute_gradients(pred[:, PredChannels.U:PredChannels.U + 1], props)
                grads_v = self.kernels._compute_gradients(pred[:, PredChannels.V:PredChannels.V + 1], props)
                div = torch.abs(grads_u[:, 0] + grads_v[:, 1]).mean()
                metrics[f"{prefix}_div_res"].append(div.item())

                mask_wall = data.mask_wall.view(-1).bool()

                if mask_wall.any():
                    slip = torch.norm(pred[mask_wall, PredChannels.UV], dim=1).mean()
                    metrics[f"{prefix}_wall_slip"].append(slip.item())
                else:
                    metrics[f"{prefix}_wall_slip"].append(0.0)

                # 2. Supervised Metrics
                if has_labels:
                    target = data.y
                    diff_u = torch.norm(pred[:, :2] - target[:, :2], dim=1)
                    denom_u = torch.norm(target[:, :2], dim=1) + 1e-6
                    metrics[f"{prefix}_rel_l2_u"].append((diff_u.sum() / denom_u.sum()).item())

                    # DIRECT WSS CORRELATION
                    if mask_wall.any():
                        pred_wss = pred[mask_wall, 4]
                        gt_wss = target[mask_wall, 4]

                        if len(pred_wss) > 5:
                            vx = pred_wss - pred_wss.mean()
                            vy = gt_wss - gt_wss.mean()
                            corr = torch.sum(vx * vy) / (
                                        torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
                            metrics[f"{prefix}_wss_corr"].append(corr.item())
                        else:
                            metrics[f"{prefix}_wss_corr"].append(np.nan)
                    else:
                        metrics[f"{prefix}_wss_corr"].append(np.nan)
                else:
                    metrics[f"{prefix}_rel_l2_u"].append(np.nan)
                    metrics[f"{prefix}_wss_corr"].append(np.nan)

            evaluate_prediction(pred_base, "base")
            if use_itpc and pred_itpc is not None:
                evaluate_prediction(pred_itpc, "itpc")

            # Optionally save comparison visualizations for the first few samples.
            if save_comparison_images and use_itpc and i < 3:
                self._plot_comparison(data, pred_base, pred_itpc, data.y if has_labels else None,
                                      f"{level_name}_sample_{i}_ITPC")

        df = pd.DataFrame(metrics)
        print(f"\n📊 A/B Test Results for {level_name}:")

        print(df.describe().loc[['mean']].T)
        if use_itpc:
            mean_base = df["base_rel_l2_u"].mean()
            mean_itpc = df["itpc_rel_l2_u"].mean()
            improvement = ((mean_base - mean_itpc) / mean_base) * 100
            print(f"\n💡 ITPC reduced L2 Velocity Error by {improvement:.2f}%")

        return df.mean()

    def _plot_comparison(self, data, pred_base, pred_itpc, target, title):
        rows = 3 if self.tier == "tier2" else 2
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        if rows == 1: axes = np.array([axes])

        pos = data.x[:, :2].cpu().numpy()
        mask_wall = data.mask_wall.view(-1).cpu().bool().numpy()
        wall_pos = pos[mask_wall]

        def plot_row(row_idx, gt_val, base_val, itpc_val, name, cmap, is_wall=False):
            vmax = max(gt_val.max() if target is not None else 0, base_val.max(), itpc_val.max())
            vmin = min(gt_val.min() if target is not None else 0, base_val.min(), itpc_val.min())

            if is_wall:
                if target is not None:
                    sc0 = axes[row_idx, 0].scatter(wall_pos[:, 0], wall_pos[:, 1], c=gt_val, cmap=cmap, s=20, vmin=vmin,
                                                   vmax=vmax)
                    axes[row_idx, 0].set_title(f"GT {name}")
                    plt.colorbar(sc0, ax=axes[row_idx, 0])
                else:
                    axes[row_idx, 0].text(0.5, 0.5, "GT Missing", ha='center', va='center')

                sc1 = axes[row_idx, 1].scatter(wall_pos[:, 0], wall_pos[:, 1], c=base_val, cmap=cmap, s=20, vmin=vmin,
                                               vmax=vmax)
                axes[row_idx, 1].set_title(f"Base Pred {name}")
                plt.colorbar(sc1, ax=axes[row_idx, 1])

                sc2 = axes[row_idx, 2].scatter(wall_pos[:, 0], wall_pos[:, 1], c=itpc_val, cmap=cmap, s=20, vmin=vmin,
                                               vmax=vmax)
                axes[row_idx, 2].set_title(f"ITPC {name}")
                plt.colorbar(sc2, ax=axes[row_idx, 2])
            else:
                if target is not None:
                    sc0 = axes[row_idx, 0].tripcolor(pos[:, 0], pos[:, 1], gt_val, cmap=cmap)
                    axes[row_idx, 0].set_title(f"GT {name}")
                    plt.colorbar(sc0, ax=axes[row_idx, 0])
                else:
                    axes[row_idx, 0].text(0.5, 0.5, "GT Missing", ha='center', va='center')

                sc1 = axes[row_idx, 1].tripcolor(pos[:, 0], pos[:, 1], base_val, cmap=cmap)
                axes[row_idx, 1].set_title(f"Base Pred {name}")
                plt.colorbar(sc1, ax=axes[row_idx, 1])

                sc2 = axes[row_idx, 2].tripcolor(pos[:, 0], pos[:, 1], itpc_val, cmap=cmap)
                axes[row_idx, 2].set_title(f"ITPC {name}")
                plt.colorbar(sc2, ax=axes[row_idx, 2])

            for col in range(3):
                axes[row_idx, col].set_aspect('equal')
                axes[row_idx, col].axis('off')

        # Row 0: Velocity Magnitude
        u_base = np.linalg.norm(pred_base[:, :2].cpu().numpy(), axis=1)
        u_itpc = np.linalg.norm(pred_itpc[:, :2].cpu().numpy(), axis=1)
        u_gt = np.linalg.norm(target[:, :2].cpu().numpy(), axis=1) if target is not None else None
        plot_row(0, u_gt, u_base, u_itpc, "Velocity Mag", 'jet')

        # Row 1 (Tier 2): Viscosity
        current_row = 1
        if self.tier == "tier2":
            mu_base = pred_base[:, 3].cpu().numpy()
            mu_itpc = pred_itpc[:, 3].cpu().numpy()
            mu_gt = target[:, 3].cpu().numpy() if target is not None else None
            plot_row(current_row, mu_gt, mu_base, mu_itpc, "Viscosity", 'viridis')
            current_row += 1

        # Row 2 (or 1): Direct WSS (Scatter on walls)
        wss_base = pred_base[mask_wall, 4].cpu().numpy()
        wss_itpc = pred_itpc[mask_wall, 4].cpu().numpy()
        wss_gt = target[mask_wall, 4].cpu().numpy() if target is not None else None
        plot_row(current_row, wss_gt, wss_base, wss_itpc, "WSS", 'plasma', is_wall=True)

        save_dir = reports_evaluation_dir("validation", self.tier)
        safe_title = title.replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(save_dir / f"{safe_title}.png", dpi=150, bbox_inches='tight')
        plt.close()