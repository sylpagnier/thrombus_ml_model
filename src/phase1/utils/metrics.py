import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from torch_geometric.data import Batch
import torch.nn as nn
from typing import Optional, Sequence, Union, List

Number = Union[int, float]


class DynamicLossWeighter(nn.Module):
    """
    Dynamically weights multiple loss components using homoscedastic task uncertainty.
    Clamps log_var per task so effective weights exp(-log_var) stay in a sane range:
    min_log_var lower-bounds log_var (caps maximum precision), max_log_var upper-bounds
    log_var (floors minimum precision). Reference: Kendall et al., 2018.
    """
    def __init__(
        self,
        num_losses: int = 2,
        min_log_var: Union[Number, Sequence[Number]] = -8.0,
        max_log_var: Optional[Union[Number, Sequence[Number]]] = None,
    ):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

        def _bound_vec(
            value: Optional[Union[Number, Sequence[Number]]],
            fill: float,
        ) -> torch.Tensor:
            if value is None:
                return torch.full((num_losses,), fill, dtype=torch.float32)
            if isinstance(value, (int, float)):
                return torch.full((num_losses,), float(value), dtype=torch.float32)
            t = torch.tensor(list(value), dtype=torch.float32)
            if t.numel() != num_losses:
                raise ValueError(
                    f"Expected length {num_losses} for per-task bounds, got {t.numel()}"
                )
            return t

        self.register_buffer("per_task_min_log_var", _bound_vec(min_log_var, -8.0))
        self.register_buffer("per_task_max_log_var", _bound_vec(max_log_var, float("inf")))

        self.min_log_var = (
            float(min_log_var)
            if isinstance(min_log_var, (int, float))
            else float(self.per_task_min_log_var[0].item())
        )

    def clamped_log_vars(self) -> torch.Tensor:
        return torch.clamp(
            self.log_vars,
            min=self.per_task_min_log_var,
            max=self.per_task_max_log_var,
        )

    def forward(
        self,
        losses: Sequence,
        scales: Optional[Sequence[float]] = None,
        task_active: Optional[Union[Sequence[bool], torch.Tensor, List[bool]]] = None,
    ):
        if scales is None:
            scales = [1.0] * len(losses)
        total_loss = 0
        min_lv = self.per_task_min_log_var
        max_lv = self.per_task_max_log_var
        for i, loss in enumerate(losses):
            if task_active is not None:
                act = task_active[i]
                if hasattr(act, "item"):
                    act = bool(act.item())
                if not act:
                    continue
            else:
                li = loss.item() if torch.is_tensor(loss) else float(loss)
                if li <= 0.0:
                    continue

            safe_log_var = torch.clamp(self.log_vars[i], min=min_lv[i], max=max_lv[i])
            precision = torch.exp(-safe_log_var)
            task_loss = precision * loss + safe_log_var
            total_loss += scales[i] * task_loss
        return total_loss

def validate_and_plot(model, val_data, epoch, device, tier="tier1"):
    model.eval()
    with torch.no_grad():
        data_on_device = Batch.from_data_list([val_data]).to(device)
        pred = model(data_on_device, solver="anderson", anderson_beta=0.8)
        coords = data_on_device.x[:, :2].cpu().numpy()

    plt.figure(figsize=(10, 4))

    # --- Setup Tier-Specific Plotting Rules ---
    if tier == "tier1":
        val_pred = pred[:, 0].cpu().numpy()  # u-velocity
        cmap, label = 'jet', r"Predicted ND-Velocity (u)"
        title = f"Tier 1 Validation - Epoch {epoch}"
        use_log_norm = False
    else:
        val_pred = pred[:, 3].cpu().numpy()  # viscosity
        cmap, label = 'viridis', r"Predicted ND-Viscosity ($\mu$)"
        title = f"Tier 2 Validation (Carreau) - Epoch {epoch}"
        # Viscosity MUST be plotted in log-scale to visualize the boundary layer
        use_log_norm = True

        # --- Plotting ---
    if use_log_norm:
        # Prevent log(0) issues and set bounds matching your mu_0 and mu_inf
        val_pred_safe = np.clip(val_pred, a_min=1e-4, a_max=None)
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=val_pred_safe, cmap=cmap, s=5, norm=LogNorm())
    else:
        sc = plt.scatter(coords[:, 0], coords[:, 1], c=val_pred, cmap=cmap, s=5)

    plt.colorbar(sc, label=label)
    plt.title(title)
    plt.axis('equal')

    save_dir = Path(f"reports/figures/{tier}")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"val_epoch_{epoch}.png")
    plt.close()


def quantify_performance(model, val_loader, kernels, device, tier="tier1"):
    model.eval()

    # Initialize metric trackers dynamically
    metrics = {"rel_l2": [], "continuity": [], "wall_slip": [], "shear_mse": []}
    if tier == "tier2":
        metrics.update({"rheology": [], "mu_mae": [], "mu_log_mse": []})

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            pred = model(data, solver="anderson", anderson_beta=0.8)
            props = kernels._get_geometric_props(data)

            # Resolve node mask safely for batched or unbatched data
            if hasattr(data, 'is_anchor'):
                node_mask = data.is_anchor[data.batch] if hasattr(data,
                                                                  'batch') and data.batch is not None else data.is_anchor

                if node_mask.any():
                    # 1. Kinematic Error (u, v, p) only
                    diff_norm = torch.norm(pred[node_mask, :3] - data.y[node_mask, :3], p=2)
                    target_norm = torch.norm(data.y[node_mask, :3], p=2)
                    metrics["rel_l2"].append((diff_norm / (target_norm + 1e-8)).item())

                    # 2. Explicit Shear Error
                    u_t, v_t = data.y[:, 0:1], data.y[:, 1:2]
                    c_u_t, c_v_t = kernels._compute_derivatives(u_t, props), kernels._compute_derivatives(v_t, props)
                    g_dot_t = torch.sqrt(2 * c_u_t[:, 0, 0] ** 2 + 2 * c_v_t[:, 1, 0] ** 2 + (
                            c_u_t[:, 1, 0] + c_v_t[:, 0, 0]) ** 2 + 1e-8)

                    u_p, v_p = pred[:, 0:1], pred[:, 1:2]
                    c_u_p, c_v_p = kernels._compute_derivatives(u_p, props), kernels._compute_derivatives(v_p, props)
                    g_dot_p = torch.sqrt(2 * c_u_p[:, 0, 0] ** 2 + 2 * c_v_p[:, 1, 0] ** 2 + (
                            c_u_p[:, 1, 0] + c_v_p[:, 0, 0]) ** 2 + 1e-8)

                    metrics["shear_mse"].append(F.mse_loss(g_dot_p[node_mask], g_dot_t[node_mask]).item())

                    # --- NEW: Viscosity Tracking (Tier 2) ---
                    if tier == "tier2" and data.y.shape[1] >= 4:
                        mu_p = pred[node_mask, 3]
                        mu_t = data.y[node_mask, 3]

                        # Raw Mean Absolute Error (biased toward vessel center)
                        metrics["mu_mae"].append(F.l1_loss(mu_p, mu_t).item())

                        # Log-Space MSE (unbiased, captures boundary layer accuracy)
                        mu_p_safe = torch.clamp(mu_p, min=1e-6)
                        mu_t_safe = torch.clamp(mu_t, min=1e-6)
                        metrics["mu_log_mse"].append(F.mse_loss(torch.log(mu_p_safe), torch.log(mu_t_safe)).item())

            # 3. Continuity (Divergence)
            u, v = pred[:, 0:1], pred[:, 1:2]
            grad_u, grad_v = kernels._compute_gradients(u, props), kernels._compute_gradients(v, props)
            div_u = grad_u[:, 0:1] + grad_v[:, 1:2]
            metrics["continuity"].append(torch.abs(div_u).mean().item())

            # 4. Wall Slip
            if data.mask_wall.any():
                wall_vel = torch.norm(pred[data.mask_wall, :2], p=2, dim=1)
                metrics["wall_slip"].append(wall_vel.mean().item())

            # 5. Physics Rheology Residual
            if tier == "tier2":
                metrics["rheology"].append(kernels.rheology_loss(pred, data, props).item())

    # Safely compute mean over batches
    return {k: np.mean(v) if len(v) > 0 else float('nan') for k, v in metrics.items()}