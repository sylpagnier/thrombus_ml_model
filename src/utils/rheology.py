from typing import Optional

import torch


def compute_shear_rate(u_x, u_y, v_x, v_y, eps: float = 1e-8):
    """Compute 2D shear-rate invariant sqrt(2(ux^2+vy^2)+(uy+vx)^2+eps)."""
    strain_sq = 2.0 * (u_x ** 2 + v_y ** 2) + (u_y + v_x) ** 2
    return torch.sqrt(strain_sq + float(eps))


def carreau_yasuda_viscosity(
    gamma_dot_nd: torch.Tensor,
    mu_inf_nd: torch.Tensor,
    mu_0_nd: torch.Tensor,
    lambda_nd: torch.Tensor,
    n: float,
    a: float,
):
    """Evaluate non-dimensional Carreau-Yasuda viscosity."""
    shear_term = 1.0 + (lambda_nd * gamma_dot_nd) ** float(a)
    power = (float(n) - 1.0) / float(a)
    return mu_inf_nd + (mu_0_nd - mu_inf_nd) * (shear_term ** power)


def dual_viscosity_multiplier(
    mat: torch.Tensor,
    fi: torch.Tensor,
    mu_ratio_max: float,
    mu1_sigmoid_fn,
    mu2_sigmoid_fn,
):
    """Shared multiplier for clot-enhanced viscosity terms."""
    return 1.0 + mu1_sigmoid_fn(mat) + mu2_sigmoid_fn(fi)


def clot_trigger_sigmoid(
    field_si: torch.Tensor,
    crit: float,
    temp: float,
) -> torch.Tensor:
    """Differentiable [0, 1] indicator: sigmoid((field - crit) / temp)."""
    safe_temp = max(float(temp), 1e-8)
    norm = (field_si - float(crit)) / safe_temp
    safe = torch.clamp(norm, min=-50.0, max=50.0)
    return torch.sigmoid(safe)


def phi_clot_from_mat_fi(
    mat_si: torch.Tensor,
    fi_si: torch.Tensor,
    *,
    mat_crit: float,
    fi_crit: float,
    temp_mat: float,
    temp_fi: float,
    combine: str = "max",
) -> torch.Tensor:
    """Unified clot indicator in [0, 1] from adhered platelets (Mat) and fibrin (FI).

    ``combine``:
        ``max`` — either trigger activates clot (default).
        ``product`` — both must be active.
    """
    phi_mat = clot_trigger_sigmoid(mat_si, mat_crit, temp_mat)
    phi_fi = clot_trigger_sigmoid(fi_si, fi_crit, temp_fi)
    if combine == "product":
        return phi_mat * phi_fi
    if combine != "max":
        raise ValueError(f"phi_clot combine must be 'max' or 'product', got {combine!r}")
    return torch.maximum(phi_mat, phi_fi)


def multiplicative_clot_mu_eff_nd(
    mu_carreau_nd: torch.Tensor,
    phi_clot: torch.Tensor,
    mu_max_ratio: float,
) -> torch.Tensor:
    """mu_eff = mu_carreau * (1 + (mu_max_ratio - 1) * phi_clot), phi_clot in [0, 1]."""
    ratio = max(float(mu_max_ratio), 1.0)
    phi = phi_clot.reshape(-1, 1).to(dtype=mu_carreau_nd.dtype).clamp(0.0, 1.0)
    mu_c = mu_carreau_nd.reshape(-1, 1)
    return mu_c * (1.0 + (ratio - 1.0) * phi)
