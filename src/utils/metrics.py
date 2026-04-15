import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from src.utils.paths import reports_dir
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
        val_pred = pred[:, 3].detach().cpu().numpy()  # viscosity
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
        "max_err_u": [],
        "max_err_v": [],
        "max_err_p": [],
        "dp_error": [],
        "rel_l2_core": [],
        "rel_l2_boundary": [],
        "flow_mismatch_rel": [],
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
                    
                    # Clone so we don't accidentally modify the underlying data tensors in-place
                    y_a = data.y[node_mask, :3].clone()
                    p_a = pred[node_mask, :3].clone()
                    
                    # --- THE GAUGE PRESSURE POST-PROCESSING FIX ---
                    # Incompressible N-S only solves for \nabla p. Absolute pressure has no physical meaning.
                    # We mean-shift both fields to 0 to remove the arbitrary constant 'C' before evaluating L2.
                    p_a[:, 2] = p_a[:, 2] - p_a[:, 2].mean()
                    y_a[:, 2] = y_a[:, 2] - y_a[:, 2].mean()
                    # ----------------------------------------------

                    diff_norm = torch.norm(p_a - y_a, p=2)
                    target_norm = torch.norm(y_a, p=2)
                    metrics["rel_l2"].append((diff_norm / (target_norm + 1e-8)).item())
                    
                    for j, key in enumerate(("rel_l2_u", "rel_l2_v", "rel_l2_p")):
                        num = torch.norm(p_a[:, j] - y_a[:, j], p=2)
                        den = torch.norm(y_a[:, j], p=2) + 1e-8
                        metrics[key].append((num / den).item())

                    # Localized worst-case errors (L-infinity): useful for sharp stenosis corners/jets.
                    metrics["max_err_u"].append(torch.max(torch.abs(p_a[:, 0] - y_a[:, 0])).item())
                    metrics["max_err_v"].append(torch.max(torch.abs(p_a[:, 1] - y_a[:, 1])).item())
                    metrics["max_err_p"].append(torch.max(torch.abs(p_a[:, 2] - y_a[:, 2])).item())

                    # Clinical pressure-drop fidelity (gauge-invariant). Uses inlet/outlet masks if available.
                    inlet_mask = getattr(data, "mask_inlet", None)
                    outlet_mask = getattr(data, "mask_outlet", None)
                    if inlet_mask is not None and outlet_mask is not None and inlet_mask.any() and outlet_mask.any():
                        p_pred = pred[:, 2]
                        p_true = data.y[:, 2]
                        dp_pred = p_pred[inlet_mask].mean() - p_pred[outlet_mask].mean()
                        dp_true = p_true[inlet_mask].mean() - p_true[outlet_mask].mean()
                        dp_error = torch.abs(dp_pred - dp_true) / (torch.abs(dp_true) + 1e-8)
                        metrics["dp_error"].append(dp_error.item())

                    # Regional error split (core vs near-wall boundary) from SDF magnitude.
                    # x[:,2] is |SDF|-like channel in this graph format.
                    if data.x.shape[1] > 2:
                        sdf_abs = torch.abs(data.x[node_mask, 2])
                        sdf_max = torch.max(sdf_abs) if sdf_abs.numel() > 0 else torch.tensor(0.0, device=sdf_abs.device)
                        if float(sdf_max.item()) > 0.0:
                            core_mask = sdf_abs > 0.25 * sdf_max
                            boundary_mask = ~core_mask
                            if core_mask.any():
                                core_num = torch.norm(p_a[core_mask] - y_a[core_mask], p=2)
                                core_den = torch.norm(y_a[core_mask], p=2) + 1e-8
                                metrics["rel_l2_core"].append((core_num / core_den).item())
                            if boundary_mask.any():
                                bnd_num = torch.norm(p_a[boundary_mask] - y_a[boundary_mask], p=2)
                                bnd_den = torch.norm(y_a[boundary_mask], p=2) + 1e-8
                                metrics["rel_l2_boundary"].append((bnd_num / bnd_den).item())

                    # Explicit shear-rate MSE (anchors only; needs labeled fields)
                    u_t, v_t = data.y[:, 0:1], data.y[:, 1:2]
                    c_u_t, c_v_t = kernels._compute_derivatives(u_t, props), kernels._compute_derivatives(v_t, props)
                    g_dot_t = torch.sqrt(
                        2 * c_u_t[:, 0, 0] ** 2
                        + 2 * c_v_t[:, 1, 0] ** 2
                        + (c_u_t[:, 1, 0] + c_v_t[:, 0, 0]) ** 2
                        + 1e-8
                    )

                    u_p, v_p = pred[:, 0:1], pred[:, 1:2]
                    c_u_p, c_v_p = kernels._compute_derivatives(u_p, props), kernels._compute_derivatives(v_p, props)
                    g_dot_p = torch.sqrt(
                        2 * c_u_p[:, 0, 0] ** 2
                        + 2 * c_v_p[:, 1, 0] ** 2
                        + (c_u_p[:, 1, 0] + c_v_p[:, 0, 0]) ** 2
                        + 1e-8
                    )

                    metrics["shear_mse"].append(F.mse_loss(g_dot_p[node_mask], g_dot_t[node_mask]).item())

                    if tier == "tier2" and data.y.shape[1] >= 4:
                        mu_p = pred[node_mask, 3]
                        mu_t = data.y[node_mask, 3]
                        metrics["mu_mae"].append(F.l1_loss(mu_p, mu_t).item())
                        mu_p_safe = torch.clamp(mu_p, min=1e-6)
                        mu_t_safe = torch.clamp(mu_t, min=1e-6)
                        metrics["mu_log_mse"].append(
                            F.mse_loss(torch.log(mu_p_safe), torch.log(mu_t_safe)).item()
                        )

            u, v = pred[:, 0:1], pred[:, 1:2]
            grad_u, grad_v = kernels._compute_gradients(u, props), kernels._compute_gradients(v, props)
            div_u = grad_u[:, 0:1] + grad_v[:, 1:2]
            interior = kernels.fluid_interior_mask(data)
            if interior.any():
                metrics["continuity"].append(torch.abs(div_u.view(-1)[interior]).mean().item())
            else:
                metrics["continuity"].append(float("nan"))

            if data.mask_wall.any():
                wall_vel = torch.norm(pred[data.mask_wall, :2], p=2, dim=1)
                metrics["wall_slip"].append(wall_vel.mean().item())

            # Global inlet/outlet flow proxy (mean normal velocity is unavailable, so use streamwise u).
            inlet_mask = getattr(data, "mask_inlet", None)
            outlet_mask = getattr(data, "mask_outlet", None)
            if inlet_mask is not None and outlet_mask is not None and inlet_mask.any() and outlet_mask.any():
                q_in = pred[inlet_mask, 0].mean()
                q_out = pred[outlet_mask, 0].mean()
                flow_mismatch_rel = torch.abs(q_in - q_out) / (torch.abs(q_in) + 1e-8)
                metrics["flow_mismatch_rel"].append(flow_mismatch_rel.item())

            if tier == "tier2":
                metrics["rheology"].append(kernels.rheology_loss(pred, data, props).item())

    out: Dict[str, Any] = {k: _list_mean(v) for k, v in metrics.items()}

    rl2_std, rl2_p90 = _list_dispersion(metrics["rel_l2"])
    out["rel_l2_std"] = rl2_std
    out["rel_l2_p90"] = rl2_p90

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

    # Optional pathology stratification: logged when metadata exists on Data objects.
    out["rel_l2_healthy"] = float("nan")
    out["rel_l2_aneurysm"] = float("nan")
    out["rel_l2_stenosis"] = float("nan")

    patho_vals: Dict[str, List[float]] = {"healthy": [], "aneurysm": [], "stenosis": []}
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            if not hasattr(data, "is_anchor"):
                continue
            node_mask = (
                data.is_anchor[data.batch]
                if hasattr(data, "batch") and data.batch is not None
                else data.is_anchor
            )
            if not node_mask.any():
                continue
            label = None
            for key in ("pathology_type", "geometry_type", "vessel_type"):
                if hasattr(data, key):
                    raw = getattr(data, key)
                    if isinstance(raw, str):
                        label = raw.lower()
                    elif hasattr(raw, "item"):
                        label = str(raw.item()).lower()
                    else:
                        label = str(raw).lower()
                    break
            if label is None:
                continue
            if "sten" in label:
                bucket = "stenosis"
            elif "aneu" in label:
                bucket = "aneurysm"
            elif "healthy" in label or "normal" in label or "straight" in label:
                bucket = "healthy"
            else:
                continue
            pred = model(data, solver="anderson", anderson_beta=0.8)
            y_a = data.y[node_mask, :3].clone()
            p_a = pred[node_mask, :3].clone()
            p_a[:, 2] = p_a[:, 2] - p_a[:, 2].mean()
            y_a[:, 2] = y_a[:, 2] - y_a[:, 2].mean()
            diff_norm = torch.norm(p_a - y_a, p=2)
            target_norm = torch.norm(y_a, p=2) + 1e-8
            patho_vals[bucket].append((diff_norm / target_norm).item())

    out["rel_l2_healthy"] = _list_mean(patho_vals["healthy"])
    out["rel_l2_aneurysm"] = _list_mean(patho_vals["aneurysm"])
    out["rel_l2_stenosis"] = _list_mean(patho_vals["stenosis"])

    if tier == "tier2" and metrics["rheology"]:
        r_std, r_p90 = _list_dispersion(metrics["rheology"])
        out["rheology_std"] = r_std
        out["rheology_p90"] = r_p90

    return out