import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Batch


def validate_and_plot(model, val_data, epoch, device, tier="tier1"):
    model.eval()
    with torch.no_grad():
        data_on_device = Batch.from_data_list([val_data]).to(device)
        pred = model(data_on_device, solver="anderson", anderson_beta=0.8)
        coords = data_on_device.x[:, :2].cpu().numpy()

    plt.figure(figsize=(10, 4))

    if tier == "tier1":
        # Plot velocity for Tier 1
        val_pred = pred[:, 0].cpu().numpy()
        cmap, label, title = 'jet', r"Predicted ND-Velocity (u)", f"Tier 1 Validation - Epoch {epoch}"
    else:
        # Plot viscosity for Tier 2
        val_pred = pred[:, 3].cpu().numpy()
        cmap, label, title = 'viridis', r"Predicted ND-Viscosity ($\mu$)", f"Tier 2 Validation (Carreau) - Epoch {epoch}"

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
    metrics = {"rel_l2": [], "continuity": [], "wall_slip": [], "shear_mse": []}
    if tier == "tier2":
        metrics["rheology"] = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            pred = model(data, solver="anderson", anderson_beta=0.8)
            props = kernels._get_geometric_props(data)

            # 1. Rel L2 & Explicit Shear Error (Calculated for both tiers now!)
            if hasattr(data, 'is_anchor'):
                node_mask = data.is_anchor[data.batch]
                if node_mask.any():
                    diff_norm = torch.norm(pred[node_mask, :3] - data.y[node_mask, :3], p=2)
                    target_norm = torch.norm(data.y[node_mask, :3], p=2)
                    metrics["rel_l2"].append((diff_norm / (target_norm + 1e-8)).item())

                    # Shear Rate tracking
                    u_t, v_t = data.y[:, 0:1], data.y[:, 1:2]
                    c_u_t, c_v_t = kernels._compute_derivatives(u_t, props), kernels._compute_derivatives(v_t, props)
                    g_dot_t = torch.sqrt(2 * c_u_t[:, 0, 0] ** 2 + 2 * c_v_t[:, 1, 0] ** 2 + (
                                c_u_t[:, 1, 0] + c_v_t[:, 0, 0]) ** 2 + 1e-8)

                    u_p, v_p = pred[:, 0:1], pred[:, 1:2]
                    c_u_p, c_v_p = kernels._compute_derivatives(u_p, props), kernels._compute_derivatives(v_p, props)
                    g_dot_p = torch.sqrt(2 * c_u_p[:, 0, 0] ** 2 + 2 * c_v_p[:, 1, 0] ** 2 + (
                                c_u_p[:, 1, 0] + c_v_p[:, 0, 0]) ** 2 + 1e-8)

                    metrics["shear_mse"].append(F.mse_loss(g_dot_p[node_mask], g_dot_t[node_mask]).item())

            # 2. Continuity
            u, v = pred[:, 0:1], pred[:, 1:2]
            grad_u, grad_v = kernels._compute_gradients(u, props), kernels._compute_gradients(v, props)
            div_u = grad_u[:, 0:1] + grad_v[:, 1:2]
            metrics["continuity"].append(torch.abs(div_u).mean().item())

            # 3. Wall Slip
            if data.mask_wall.any():
                wall_vel = torch.norm(pred[data.mask_wall, :2], p=2, dim=1)
                metrics["wall_slip"].append(wall_vel.mean().item())

            # 4. Rheology (Tier 2 only)
            if tier == "tier2":
                metrics["rheology"].append(kernels.rheology_loss(pred, data, props).item())

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}