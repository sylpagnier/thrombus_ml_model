import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from src.utils.paths import reports_dir
from src.config import PredChannels
from src.utils.rheology import compute_shear_rate
from torch_geometric.data import Batch
import torch.nn as nn
from typing import Optional, Sequence, Union, List, Dict, Any

Number = Union[int, float]


def _list_mean(vals: List[float]) -> float:
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def _list_dispersion(vals: List[float]) -> tuple:
    """Return (std, 90th percentile) for a list of per-batch scalars."""
    if len(vals) == 0:
        return float("nan"), float("nan")
    a = np.asarray(vals, dtype=np.float64)
    return float(np.std(a)), float(np.percentile(a, 90))


def _masked_rel_l2(pred_uvp: torch.Tensor, true_uvp: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    m = mask.view(-1).bool()
    if not m.any():
        return None
    p = pred_uvp[m, :3]
    y = true_uvp[m, :3]
    den = torch.norm(y, p=2)
    if den <= 0:
        return None
    return torch.norm(p - y, p=2) / (den + 1e-8)


def _sdf_grad_proxy(data) -> Optional[torch.Tensor]:
    if not hasattr(data, "edge_index") or data.edge_index is None:
        return None
    edge_index = data.edge_index
    if edge_index.numel() == 0:
        return None
    sdf = data.x[:, 2].abs()
    row, col = edge_index
    diff = (sdf[row] - sdf[col]).abs()
    n = int(data.num_nodes)
    sum_d = torch.zeros(n, dtype=diff.dtype, device=diff.device)
    cnt = torch.zeros(n, dtype=diff.dtype, device=diff.device)
    sum_d.index_add_(0, row, diff)
    cnt.index_add_(0, row, torch.ones_like(diff))
    return sum_d / (cnt + 1e-8)


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
        if len(losses) == 0:
            return self.log_vars.sum() * 0.0

        # Keep accumulator tensor-typed from the start so edge-case batches
        # never return a Python scalar that breaks autograd expectations.
        first = losses[0]
        if torch.is_tensor(first):
            total_loss = first.sum() * 0.0
        else:
            total_loss = self.log_vars.sum() * 0.0
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
        coords = data_on_device.x[:, :2].detach().cpu().numpy()

    plt.figure(figsize=(10, 4))

    # --- Setup Tier-Specific Plotting Rules ---
    if tier == "tier1":
        val_pred = pred[:, 0].detach().cpu().numpy()  # u-velocity
        cmap, label = 'jet', r"Predicted ND-Velocity (u)"
        title = f"Tier 1 Validation - Epoch {epoch}"
        use_log_norm = False
    else:
        val_pred = pred[:, PredChannels.MU_EFF_ND].detach().cpu().numpy()  # viscosity
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

    save_dir = reports_dir() / "figures" / tier
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"val_epoch_{epoch}.png")
    plt.close()


def quantify_performance(model, val_loader, kernels, device, tier="tier1") -> Dict[str, Any]:
    """Aggregate validation metrics over ``val_loader``.

    **Rel L2** (and component breakdowns, shear, μ errors) are computed only on batches
    that contain at least one anchor node with CFD labels; physics-only graphs have no
    ground-truth ``y`` and are skipped for those terms. **Continuity** (mean ``|∇·u|``) is
    averaged on the same **fluid interior** mask as training (excludes wall, inlet,
    outlet). Rheology uses the full batch; wall slip uses wall nodes when present.
    """
    model.eval()

    metrics: Dict[str, List[float]] = {
        "rel_l2": [],
        "rel_l2_u": [],
        "rel_l2_v": [],
        "rel_l2_p": [],
        "rel_l2_near_wall": [],
        "rel_l2_high_sdf_grad": [],
        "continuity": [],
        "wall_slip": [],
        "shear_mse": [],
    }
    if tier == "tier2":
        metrics.update({"rheology": [], "mu_mae": [], "mu_log_mse": []})

    val_total_batches = 0
    val_anchor_batches = 0

    with torch.no_grad():
        for data in val_loader:
            val_total_batches += 1
            data = data.to(device)
            pred = model(data, solver="anderson", anderson_beta=0.8)
            props = kernels._get_geometric_props(data)

            # Resolve node mask safely for batched or unbatched data
            if hasattr(data, "is_anchor"):
                node_mask = (
                    data.is_anchor[data.batch]
                    if hasattr(data, "batch") and data.batch is not None
                    else data.is_anchor
                )

                if node_mask.any():
                    val_anchor_batches += 1
                    y_a = data.y[node_mask, :3]
                    p_a = pred[node_mask, :3]
                    diff_norm = torch.norm(p_a - y_a, p=2)
                    target_norm = torch.norm(y_a, p=2)
                    metrics["rel_l2"].append((diff_norm / (target_norm + 1e-8)).item())
                    for j, key in enumerate(("rel_l2_u", "rel_l2_v", "rel_l2_p")):
                        num = torch.norm(p_a[:, j] - y_a[:, j], p=2)
                        den = torch.norm(y_a[:, j], p=2) + 1e-8
                        metrics[key].append((num / den).item())

                    # Area-specific relative errors on anchor nodes:
                    # near-wall region (low |SDF|) and high |∇SDF| proxy region.
                    sdf_abs = data.x[:, 2].abs()
                    sdf_anchor = sdf_abs[node_mask]
                    if sdf_anchor.numel() > 0:
                        sdf_q25 = torch.quantile(sdf_anchor, torch.tensor(0.25, device=sdf_anchor.device))
                        near_wall_mask = node_mask & (sdf_abs <= sdf_q25)
                        rel_near = _masked_rel_l2(pred, data.y, near_wall_mask)
                        if rel_near is not None:
                            metrics["rel_l2_near_wall"].append(float(rel_near.item()))

                    grad_proxy = _sdf_grad_proxy(data)
                    if grad_proxy is not None:
                        gp_anchor = grad_proxy[node_mask]
                        if gp_anchor.numel() > 0:
                            gp_q75 = torch.quantile(gp_anchor, torch.tensor(0.75, device=gp_anchor.device))
                            high_grad_mask = node_mask & (grad_proxy >= gp_q75)
                            rel_high = _masked_rel_l2(pred, data.y, high_grad_mask)
                            if rel_high is not None:
                                metrics["rel_l2_high_sdf_grad"].append(float(rel_high.item()))

                    # Explicit shear-rate MSE (anchors only; needs labeled fields)
                    u_t = data.y[:, PredChannels.U:PredChannels.U + 1]
                    v_t = data.y[:, PredChannels.V:PredChannels.V + 1]
                    c_u_t, c_v_t = kernels._compute_derivatives(u_t, props), kernels._compute_derivatives(v_t, props)
                    g_dot_t = compute_shear_rate(
                        c_u_t[:, 0, 0], c_u_t[:, 1, 0], c_v_t[:, 0, 0], c_v_t[:, 1, 0], eps=1e-6
                    )

                    u_p = pred[:, PredChannels.U:PredChannels.U + 1]
                    v_p = pred[:, PredChannels.V:PredChannels.V + 1]
                    c_u_p, c_v_p = kernels._compute_derivatives(u_p, props), kernels._compute_derivatives(v_p, props)
                    g_dot_p = compute_shear_rate(
                        c_u_p[:, 0, 0], c_u_p[:, 1, 0], c_v_p[:, 0, 0], c_v_p[:, 1, 0], eps=1e-6
                    )

                    metrics["shear_mse"].append(F.mse_loss(g_dot_p[node_mask], g_dot_t[node_mask]).item())

                    if tier == "tier2" and data.y.shape[1] >= 4:
                        mu_p = pred[node_mask, PredChannels.MU_EFF_ND]
                        mu_t = data.y[node_mask, PredChannels.MU_EFF_ND]
                        metrics["mu_mae"].append(F.l1_loss(mu_p, mu_t).item())
                        mu_p_safe = torch.clamp(mu_p, min=1e-6)
                        mu_t_safe = torch.clamp(mu_t, min=1e-6)
                        metrics["mu_log_mse"].append(
                            F.mse_loss(torch.log(mu_p_safe), torch.log(mu_t_safe)).item()
                        )

            u = pred[:, PredChannels.U:PredChannels.U + 1]
            v = pred[:, PredChannels.V:PredChannels.V + 1]
            grad_u, grad_v = kernels._compute_gradients(u, props), kernels._compute_gradients(v, props)
            div_u = grad_u[:, 0:1] + grad_v[:, 1:2]
            interior = kernels.fluid_interior_mask(data)
            if interior.any():
                metrics["continuity"].append(torch.abs(div_u.view(-1)[interior]).mean().item())
            else:
                metrics["continuity"].append(float("nan"))

            if data.mask_wall.any():
                wall_vel = torch.norm(pred[data.mask_wall, PredChannels.UV], p=2, dim=1)
                metrics["wall_slip"].append(wall_vel.mean().item())

            if tier == "tier2":
                metrics["rheology"].append(kernels.rheology_loss(pred, data, props).item())

    out: Dict[str, Any] = {k: _list_mean(v) for k, v in metrics.items()}

    rl2_std, rl2_p90 = _list_dispersion(metrics["rel_l2"])
    out["rel_l2_std"] = rl2_std
    out["rel_l2_p90"] = rl2_p90

    nw_std, nw_p90 = _list_dispersion(metrics["rel_l2_near_wall"])
    out["rel_l2_near_wall_std"] = nw_std
    out["rel_l2_near_wall_p90"] = nw_p90

    hg_std, hg_p90 = _list_dispersion(metrics["rel_l2_high_sdf_grad"])
    out["rel_l2_high_sdf_grad_std"] = hg_std
    out["rel_l2_high_sdf_grad_p90"] = hg_p90

    c_std, c_p90 = _list_dispersion(metrics["continuity"])
    out["continuity_std"] = c_std
    out["continuity_p90"] = c_p90

    ws_std, ws_p90 = _list_dispersion(metrics["wall_slip"])
    out["wall_slip_std"] = ws_std
    out["wall_slip_p90"] = ws_p90

    sh_std, sh_p90 = _list_dispersion(metrics["shear_mse"])
    out["shear_mse_std"] = sh_std
    out["shear_mse_p90"] = sh_p90

    out["val_total_batches"] = float(val_total_batches)
    out["val_anchor_batches"] = float(val_anchor_batches)

    if tier == "tier2" and metrics["rheology"]:
        r_std, r_p90 = _list_dispersion(metrics["rheology"])
        out["rheology_std"] = r_std
        out["rheology_p90"] = r_p90

    return out