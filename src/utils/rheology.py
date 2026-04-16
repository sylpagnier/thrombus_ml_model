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
